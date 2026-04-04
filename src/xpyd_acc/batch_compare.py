"""Batch dataset comparison: run prompts on two endpoints, find divergences."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class DatasetSample:
    """A single sample from a dataset."""

    id: str
    prompt: str
    expected: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """Result of comparing one sample across two endpoints."""

    sample_id: str
    prompt: str
    baseline_output: str
    target_output: str
    exact_match: bool
    first_divergence_index: int | None  # token index of first divergence
    baseline_logprob_at_divergence: float | None
    target_logprob_at_divergence: float | None
    logprob_gap: float | None  # top1 - top2 gap at divergence point
    classification: str  # "match", "likely_bug", "likely_uncertainty", "unknown"
    context_length: int  # number of prompt tokens (approximated)

    def is_divergent(self) -> bool:
        """Whether this sample diverged."""
        return not self.exact_match


@dataclass
class BatchReport:
    """Statistical report from a batch comparison run."""

    total_samples: int
    divergent_samples: int
    match_samples: int
    divergence_rate: float
    results: list[SampleResult]
    # Stats on divergent samples only
    divergence_index_mean: float | None = None
    divergence_index_median: float | None = None
    logprob_gap_mean: float | None = None
    logprob_gap_median: float | None = None
    likely_bugs: int = 0
    likely_uncertainty: int = 0
    unknown_classification: int = 0
    # Divergence by context length buckets
    divergence_by_context_length: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_markdown(self, *, max_divergent_samples: int = 10) -> str:
        """Serialize the report to a Markdown string."""
        return _to_markdown(self, max_divergent_samples=max_divergent_samples)

    def to_json(self) -> str:
        """Serialize the report to a JSON string."""
        data: dict[str, Any] = {
            "total_samples": self.total_samples,
            "divergent_samples": self.divergent_samples,
            "match_samples": self.match_samples,
            "divergence_rate": self.divergence_rate,
            "divergence_index_mean": self.divergence_index_mean,
            "divergence_index_median": self.divergence_index_median,
            "logprob_gap_mean": self.logprob_gap_mean,
            "logprob_gap_median": self.logprob_gap_median,
            "likely_bugs": self.likely_bugs,
            "likely_uncertainty": self.likely_uncertainty,
            "unknown_classification": self.unknown_classification,
            "divergence_by_context_length": self.divergence_by_context_length,
            "results": [
                {
                    "sample_id": r.sample_id,
                    "prompt": r.prompt,
                    "baseline_output": r.baseline_output,
                    "target_output": r.target_output,
                    "exact_match": r.exact_match,
                    "first_divergence_index": r.first_divergence_index,
                    "baseline_logprob_at_divergence": r.baseline_logprob_at_divergence,
                    "target_logprob_at_divergence": r.target_logprob_at_divergence,
                    "logprob_gap": r.logprob_gap,
                    "classification": r.classification,
                    "context_length": r.context_length,
                }
                for r in self.results
            ],
        }
        return json.dumps(data, indent=2)


def load_dataset(path: str | Path) -> list[DatasetSample]:
    """Load dataset from JSONL file.

    Each line should be a JSON object with at least a "prompt" field.
    Optional fields: "id", "expected", and any other metadata.
    """
    path = Path(path)
    samples: list[DatasetSample] = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                msg = f"Line {i + 1}: missing 'prompt' field"
                raise ValueError(msg)
            sample_id = str(obj.get("id", i))
            prompt = obj["prompt"]
            expected = obj.get("expected")
            metadata = {k: v for k, v in obj.items() if k not in ("id", "prompt", "expected")}
            samples.append(DatasetSample(
                id=sample_id,
                prompt=prompt,
                expected=expected,
                metadata=metadata,
            ))
    return samples


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer."""
    return text.split()


def _find_first_divergence(baseline_tokens: list[str], target_tokens: list[str]) -> int | None:
    """Find index of first differing token. None if identical."""
    for i, (b, t) in enumerate(zip(baseline_tokens, target_tokens)):
        if b != t:
            return i
    if len(baseline_tokens) != len(target_tokens):
        return min(len(baseline_tokens), len(target_tokens))
    return None


def classify_divergence(
    logprob_gap: float | None,
    *,
    threshold: float = 0.1,
) -> str:
    """Classify divergence as likely bug or uncertainty.

    If the gap between top-1 and top-2 logprob at the divergence point is large,
    it's likely a real bug. If small, it's likely normal model uncertainty.
    """
    if logprob_gap is None:
        return "unknown"
    if logprob_gap >= threshold:
        return "likely_bug"
    return "likely_uncertainty"


def _context_length_bucket(length: int) -> str:
    """Bucket context length for grouping."""
    if length <= 50:
        return "0-50"
    if length <= 200:
        return "51-200"
    if length <= 500:
        return "201-500"
    if length <= 1000:
        return "501-1000"
    return "1000+"


async def _collect_output(
    url: str,
    prompt: str,
    *,
    model: str = "default",
    max_tokens: int = 64,
    api_key: str = "no-key",
    retries: int = 3,
    retry_delay: float = 1.0,
) -> tuple[str, list[dict[str, Any]]]:
    """Send prompt to an OpenAI-compatible endpoint, return (text, logprobs_list).

    Returns a tuple of (generated_text, logprobs_per_token).
    Each logprob entry has: {"token": str, "logprob": float, "top_logprobs": [...]}.
    """
    import httpx

    from xpyd_acc.retry import retry_async

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 5,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    async def _do_request() -> tuple[str, list[dict[str, Any]]]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{url}/v1/chat/completions", json=payload, headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        logprobs_content = choice.get("logprobs", {}).get("content", [])
        return text, logprobs_content

    return await retry_async(_do_request, retries=retries, base_delay=retry_delay)


async def run_batch(
    samples: list[DatasetSample],
    baseline_url: str,
    target_url: str,
    *,
    model: str = "default",
    max_tokens: int = 64,
    api_key: str = "no-key",
    logprob_gap_threshold: float = 0.1,
    concurrency: int = 5,
    retries: int = 3,
    retry_delay: float = 1.0,
    on_progress: Callable[[int, int], None] | None = None,
    match_config: Any | None = None,
) -> BatchReport:
    """Run all samples against both endpoints and produce a report.

    Args:
        on_progress: Optional callback called after each sample completes.
            Receives (completed_count, total_count).
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[SampleResult] = []
    completed = 0

    async def process_sample(sample: DatasetSample) -> SampleResult:
        async with semaphore:
            baseline_text, baseline_lp = await _collect_output(
                baseline_url, sample.prompt, model=model,
                max_tokens=max_tokens, api_key=api_key,
                retries=retries, retry_delay=retry_delay,
            )
            target_text, target_lp = await _collect_output(
                target_url, sample.prompt, model=model,
                max_tokens=max_tokens, api_key=api_key,
                retries=retries, retry_delay=retry_delay,
            )

        b_tokens = _tokenize(baseline_text)
        t_tokens = _tokenize(target_text)

        from xpyd_acc.output_compare import normalized_match

        exact = normalized_match(baseline_text, target_text, match_config)
        div_idx = _find_first_divergence(b_tokens, t_tokens)

        b_lp_at_div: float | None = None
        t_lp_at_div: float | None = None
        gap: float | None = None

        if div_idx is not None and div_idx < len(target_lp):
            lp_entry = target_lp[div_idx]
            t_lp_at_div = lp_entry.get("logprob")
            top_lps = lp_entry.get("top_logprobs", [])
            if len(top_lps) >= 2:
                gap = abs(top_lps[0].get("logprob", 0) - top_lps[1].get("logprob", 0))
        if div_idx is not None and div_idx < len(baseline_lp):
            b_lp_at_div = baseline_lp[div_idx].get("logprob")

        classification = "match" if exact else classify_divergence(
            gap, threshold=logprob_gap_threshold,
        )
        ctx_len = len(_tokenize(sample.prompt))

        return SampleResult(
            sample_id=sample.id,
            prompt=sample.prompt,
            baseline_output=baseline_text,
            target_output=target_text,
            exact_match=exact,
            first_divergence_index=div_idx,
            baseline_logprob_at_divergence=b_lp_at_div,
            target_logprob_at_divergence=t_lp_at_div,
            logprob_gap=gap,
            classification=classification,
            context_length=ctx_len,
        )

    total = len(samples)

    async def _tracked_sample(sample: DatasetSample) -> SampleResult:
        nonlocal completed
        result = await process_sample(sample)
        completed += 1
        if on_progress is not None:
            on_progress(completed, total)
        return result

    tasks = [_tracked_sample(s) for s in samples]
    results = await asyncio.gather(*tasks)
    results = list(results)

    return compute_report(results, logprob_gap_threshold=logprob_gap_threshold)


def compute_report(
    results: list[SampleResult],
    *,
    logprob_gap_threshold: float = 0.1,
) -> BatchReport:
    """Compute statistical report from sample results."""
    total = len(results)
    divergent = [r for r in results if r.is_divergent()]
    div_count = len(divergent)
    match_count = total - div_count
    rate = div_count / total if total > 0 else 0.0

    div_indices = [
        r.first_divergence_index for r in divergent
        if r.first_divergence_index is not None
    ]
    gaps = [r.logprob_gap for r in divergent if r.logprob_gap is not None]

    report = BatchReport(
        total_samples=total,
        divergent_samples=div_count,
        match_samples=match_count,
        divergence_rate=rate,
        results=results,
    )

    if div_indices:
        report.divergence_index_mean = statistics.mean(div_indices)
        report.divergence_index_median = statistics.median(div_indices)
    if gaps:
        report.logprob_gap_mean = statistics.mean(gaps)
        report.logprob_gap_median = statistics.median(gaps)

    report.likely_bugs = sum(1 for r in divergent if r.classification == "likely_bug")
    report.likely_uncertainty = sum(
        1 for r in divergent if r.classification == "likely_uncertainty"
    )
    report.unknown_classification = sum(1 for r in divergent if r.classification == "unknown")

    # Context length buckets
    buckets: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = _context_length_bucket(r.context_length)
        if bucket not in buckets:
            buckets[bucket] = {"total": 0, "divergent": 0}
        buckets[bucket]["total"] += 1
        if r.is_divergent():
            buckets[bucket]["divergent"] += 1
    report.divergence_by_context_length = buckets

    return report


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max_length, appending '...' if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def export_csv(
    report: BatchReport,
    path: str | Path | None = None,
    *,
    prompt_max_length: int = 200,
) -> str:
    """Export results as CSV. Returns CSV string. If path given, also writes to file.

    Args:
        report: The batch report to export.
        path: Optional file path to write CSV to.
        prompt_max_length: Max length for prompt column (default 200). Set 0 for no truncation.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "sample_id", "prompt", "baseline_output", "target_output",
        "match", "divergence_index", "logprob_gap",
    ])
    for r in report.results:
        prompt = r.prompt if prompt_max_length <= 0 else _truncate(r.prompt, prompt_max_length)
        writer.writerow([
            r.sample_id,
            prompt,
            r.baseline_output,
            r.target_output,
            r.exact_match,
            r.first_divergence_index if r.first_divergence_index is not None else "",
            f"{r.logprob_gap:.6f}" if r.logprob_gap is not None else "",
        ])
    csv_str = output.getvalue()
    if path is not None:
        Path(path).write_text(csv_str)
    return csv_str


def format_report(report: BatchReport) -> str:
    """Format batch report as human-readable text."""
    lines = [
        "=== Batch Comparison Report ===",
        f"Total samples: {report.total_samples}",
        f"Matches: {report.match_samples}",
        f"Divergent: {report.divergent_samples} ({report.divergence_rate:.1%})",
        "",
    ]

    if report.divergent_samples > 0:
        lines.append("--- Classification ---")
        lines.append(f"  Likely bugs:        {report.likely_bugs}")
        lines.append(f"  Likely uncertainty:  {report.likely_uncertainty}")
        lines.append(f"  Unknown:            {report.unknown_classification}")
        lines.append("")

        if report.divergence_index_mean is not None:
            lines.append("--- Divergence Point ---")
            lines.append(f"  Mean token index:   {report.divergence_index_mean:.1f}")
            lines.append(f"  Median token index: {report.divergence_index_median:.1f}")
            lines.append("")

        if report.logprob_gap_mean is not None:
            lines.append("--- Logprob Gap at Divergence ---")
            lines.append(f"  Mean:   {report.logprob_gap_mean:.6f}")
            lines.append(f"  Median: {report.logprob_gap_median:.6f}")
            lines.append("")

        if report.divergence_by_context_length:
            lines.append("--- Divergence by Context Length ---")
            for bucket in sorted(report.divergence_by_context_length.keys()):
                stats = report.divergence_by_context_length[bucket]
                rate = stats["divergent"] / stats["total"] if stats["total"] > 0 else 0
                lines.append(
                    f"  {bucket:>10s}: {stats['divergent']}/{stats['total']} ({rate:.1%})"
                )

    return "\n".join(lines)


def _to_markdown(report: BatchReport, *, max_divergent_samples: int = 10) -> str:
    """Convert a BatchReport to a Markdown string.

    Args:
        report: The batch report to convert.
        max_divergent_samples: Max number of divergent samples to include in detail section.
    """
    lines: list[str] = []
    lines.append("# Batch Comparison Report")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total samples | {report.total_samples} |")
    lines.append(f"| Matches | {report.match_samples} |")
    lines.append(f"| Divergent | {report.divergent_samples} |")
    lines.append(f"| Divergence rate | {report.divergence_rate:.1%} |")
    lines.append("")

    if report.divergent_samples > 0:
        # Classification
        lines.append("## Classification")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        lines.append(f"| Likely bugs | {report.likely_bugs} |")
        lines.append(f"| Likely uncertainty | {report.likely_uncertainty} |")
        lines.append(f"| Unknown | {report.unknown_classification} |")
        lines.append("")

        # Divergence stats
        if report.divergence_index_mean is not None:
            lines.append("## Divergence Point Statistics")
            lines.append("")
            lines.append("| Stat | Value |")
            lines.append("|------|-------|")
            lines.append(f"| Mean token index | {report.divergence_index_mean:.1f} |")
            lines.append(f"| Median token index | {report.divergence_index_median:.1f} |")
            if report.logprob_gap_mean is not None:
                lines.append(f"| Mean logprob gap | {report.logprob_gap_mean:.6f} |")
                lines.append(f"| Median logprob gap | {report.logprob_gap_median:.6f} |")
            lines.append("")

        # Context length analysis
        if report.divergence_by_context_length:
            lines.append("## Divergence by Context Length")
            lines.append("")
            lines.append("| Bucket | Divergent | Total | Rate |")
            lines.append("|--------|-----------|-------|------|")
            for bucket in sorted(report.divergence_by_context_length.keys()):
                stats = report.divergence_by_context_length[bucket]
                rate = stats["divergent"] / stats["total"] if stats["total"] > 0 else 0
                lines.append(
                    f"| {bucket} | {stats['divergent']} | {stats['total']} | {rate:.1%} |"
                )
            lines.append("")

        # Top divergent samples
        divergent = [r for r in report.results if r.is_divergent()]
        shown = divergent[:max_divergent_samples]
        if shown:
            lines.append(f"## Top Divergent Samples (showing {len(shown)}/{len(divergent)})")
            lines.append("")
            for r in shown:
                lines.append(f"### Sample {r.sample_id}")
                lines.append("")
                lines.append(f"- **Classification:** {r.classification}")
                lines.append(
                    f"- **First divergence at token:** {r.first_divergence_index}"
                )
                if r.logprob_gap is not None:
                    lines.append(f"- **Logprob gap:** {r.logprob_gap:.6f}")
                prompt_preview = _truncate(r.prompt, 200)
                lines.append(f"- **Prompt:** `{prompt_preview}`")
                baseline_preview = _truncate(r.baseline_output, 200)
                target_preview = _truncate(r.target_output, 200)
                lines.append(f"- **Baseline:** `{baseline_preview}`")
                lines.append(f"- **Target:** `{target_preview}`")
                lines.append("")

    return "\n".join(lines)


def export_markdown(
    report: BatchReport,
    path: str | Path | None = None,
    *,
    max_divergent_samples: int = 10,
) -> str:
    """Export report as Markdown. Returns Markdown string. If path given, also writes to file.

    Args:
        report: The batch report to export.
        path: Optional file path to write Markdown to.
        max_divergent_samples: Max divergent samples to show in detail.
    """
    md = _to_markdown(report, max_divergent_samples=max_divergent_samples)
    if path is not None:
        Path(path).write_text(md)
    return md
