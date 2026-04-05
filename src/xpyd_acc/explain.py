"""Deep-dive analysis of a single sample from a batch report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExplainResult:
    """Detailed analysis of a single sample."""

    sample_id: str
    prompt: str
    baseline_output: str
    target_output: str
    exact_match: bool
    classification: str
    context_length: int
    first_divergence_index: int | None
    baseline_logprob_at_divergence: float | None
    target_logprob_at_divergence: float | None
    logprob_gap: float | None
    baseline_tokens: list[str]
    target_tokens: list[str]
    divergence_context: DivergenceContext | None
    classification_reasoning: str
    suggested_next_steps: list[str]

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)


@dataclass
class DivergenceContext:
    """Tokens around the divergence point."""

    before: list[TokenPair]
    at: TokenPair
    after: list[TokenPair]


@dataclass
class TokenPair:
    """A pair of tokens at the same position."""

    index: int
    baseline: str
    target: str
    match: bool


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer matching batch_compare._tokenize."""
    return text.split()


def _build_divergence_context(
    baseline_tokens: list[str],
    target_tokens: list[str],
    divergence_index: int,
    context_size: int = 5,
) -> DivergenceContext:
    """Build context window around divergence point."""
    max_len = max(len(baseline_tokens), len(target_tokens))

    def _get(tokens: list[str], idx: int) -> str:
        return tokens[idx] if idx < len(tokens) else "<END>"

    before: list[TokenPair] = []
    start = max(0, divergence_index - context_size)
    for i in range(start, divergence_index):
        b = _get(baseline_tokens, i)
        t = _get(target_tokens, i)
        before.append(TokenPair(index=i, baseline=b, target=t, match=(b == t)))

    b_at = _get(baseline_tokens, divergence_index)
    t_at = _get(target_tokens, divergence_index)
    at = TokenPair(
        index=divergence_index, baseline=b_at, target=t_at, match=(b_at == t_at)
    )

    after: list[TokenPair] = []
    end = min(max_len, divergence_index + context_size + 1)
    for i in range(divergence_index + 1, end):
        b = _get(baseline_tokens, i)
        t = _get(target_tokens, i)
        after.append(TokenPair(index=i, baseline=b, target=t, match=(b == t)))

    return DivergenceContext(before=before, at=at, after=after)


def _classify_reasoning(classification: str, logprob_gap: float | None) -> str:
    """Explain why a sample received its classification."""
    if classification == "match":
        return "Outputs are identical — no divergence detected."
    if classification == "likely_bug":
        gap_str = f"{logprob_gap:.4f}" if logprob_gap is not None else "N/A"
        return (
            f"Classification: likely_bug. The logprob gap at divergence ({gap_str}) "
            "is large, meaning the baseline model was highly confident in a different "
            "token than what the target produced. This suggests a real precision or "
            "transfer issue rather than sampling randomness."
        )
    if classification == "likely_uncertainty":
        gap_str = f"{logprob_gap:.4f}" if logprob_gap is not None else "N/A"
        return (
            f"Classification: likely_uncertainty. The logprob gap at divergence ({gap_str}) "
            "is small, meaning the top two token choices had similar probabilities. "
            "This divergence is likely due to normal sampling non-determinism."
        )
    return (
        "Classification: unknown. Insufficient logprob data to determine "
        "whether this is a bug or normal variation."
    )


def _suggest_next_steps(
    classification: str,
    divergence_index: int | None,
    context_length: int,
) -> list[str]:
    """Suggest next debugging steps based on the analysis."""
    if classification == "match":
        return ["No action needed — sample matches."]
    steps: list[str] = []
    if classification == "likely_bug":
        steps.append("Investigate KV cache transfer — run `xpyd-acc diagnose` with cache dumps.")
        if divergence_index is not None and divergence_index == 0:
            steps.append("Divergence at token 0 — check prefill stage output.")
        elif divergence_index is not None and divergence_index > 0:
            steps.append(
                f"Divergence at token {divergence_index} — likely decode-stage issue. "
                "Compare KV cache at that position."
            )
        steps.append(
            "Try `xpyd-acc bisect` to find the minimum context "
            "length that triggers divergence."
        )
    elif classification == "likely_uncertainty":
        steps.append(
            "Re-run with temperature=0 and a fixed seed to "
            "confirm this is sampling noise."
        )
        steps.append(
            "Use `xpyd-acc aggregate` across multiple runs to "
            "check if this sample is flaky."
        )
    else:
        steps.append("Collect logprobs with `compare-logprobs --top-k 5` for more detail.")
        steps.append("Re-run with `--verbose` to capture request/response details.")
    return steps


def explain_sample(report_data: dict[str, Any], sample_id: str) -> ExplainResult:
    """Produce a deep-dive explanation for a single sample in a batch report.

    Parameters
    ----------
    report_data:
        Parsed JSON of a BatchReport (as produced by ``BatchReport.to_json()``).
    sample_id:
        The ``sample_id`` to look up.

    Raises
    ------
    KeyError
        If *sample_id* is not found in the report.
    """
    results = report_data.get("results", [])
    sample: dict[str, Any] | None = None
    for r in results:
        if r.get("sample_id") == sample_id:
            sample = r
            break

    if sample is None:
        available = [r.get("sample_id", "?") for r in results]
        raise KeyError(
            f"Sample '{sample_id}' not found in report. "
            f"Available IDs: {available}"
        )

    baseline_output: str = sample.get("baseline_output", "")
    target_output: str = sample.get("target_output", "")
    baseline_tokens = _tokenize(baseline_output)
    target_tokens = _tokenize(target_output)
    divergence_index: int | None = sample.get("first_divergence_index")
    classification: str = sample.get("classification", "unknown")
    logprob_gap: float | None = sample.get("logprob_gap")
    context_length: int = sample.get("context_length", 0)

    context: DivergenceContext | None = None
    if divergence_index is not None:
        context = _build_divergence_context(
            baseline_tokens, target_tokens, divergence_index
        )

    reasoning = _classify_reasoning(classification, logprob_gap)
    next_steps = _suggest_next_steps(classification, divergence_index, context_length)

    return ExplainResult(
        sample_id=sample.get("sample_id", ""),
        prompt=sample.get("prompt", ""),
        baseline_output=baseline_output,
        target_output=target_output,
        exact_match=sample.get("exact_match", True),
        classification=classification,
        context_length=context_length,
        first_divergence_index=divergence_index,
        baseline_logprob_at_divergence=sample.get("baseline_logprob_at_divergence"),
        target_logprob_at_divergence=sample.get("target_logprob_at_divergence"),
        logprob_gap=logprob_gap,
        baseline_tokens=baseline_tokens,
        target_tokens=target_tokens,
        divergence_context=context,
        classification_reasoning=reasoning,
        suggested_next_steps=next_steps,
    )


def load_and_explain(report_path: str | Path, sample_id: str) -> ExplainResult:
    """Load a report JSON file and explain a sample."""
    path = Path(report_path)
    data = json.loads(path.read_text())
    return explain_sample(data, sample_id)


def format_explain(result: ExplainResult) -> str:
    """Format an ExplainResult for terminal display."""
    lines: list[str] = []
    lines.append(f"═══ Sample: {result.sample_id} ═══")
    lines.append("")
    lines.append(f"Classification: {result.classification}")
    lines.append(f"Context length:  {result.context_length} tokens")
    lines.append(f"Exact match:     {result.exact_match}")
    lines.append("")

    # Prompt (truncated)
    prompt_display = result.prompt[:200] + ("..." if len(result.prompt) > 200 else "")
    lines.append(f"Prompt: {prompt_display}")
    lines.append("")

    if result.exact_match:
        lines.append("✅ Outputs are identical.")
        lines.append("")
    else:
        lines.append(f"First divergence at token index: {result.first_divergence_index}")
        if result.logprob_gap is not None:
            lines.append(f"Logprob gap at divergence:       {result.logprob_gap:.6f}")
        if result.baseline_logprob_at_divergence is not None:
            val = result.baseline_logprob_at_divergence
            lines.append(f"Baseline logprob at divergence:  {val:.6f}")
        if result.target_logprob_at_divergence is not None:
            val = result.target_logprob_at_divergence
            lines.append(f"Target logprob at divergence:    {val:.6f}")
        lines.append("")

        if result.divergence_context is not None:
            ctx = result.divergence_context
            lines.append("─── Token Context ───")
            for tp in ctx.before:
                marker = "  " if tp.match else "~ "
                lines.append(f"  [{tp.index:4d}] {marker}{tp.baseline!r}")
            tp = ctx.at
            lines.append(f"  [{tp.index:4d}] ▶ baseline: {tp.baseline!r}")
            lines.append(f"         target:   {tp.target!r}")
            for tp in ctx.after:
                marker = "  " if tp.match else "~ "
                b_str = tp.baseline
                t_str = tp.target
                if tp.match:
                    lines.append(f"  [{tp.index:4d}] {marker}{b_str!r}")
                else:
                    lines.append(f"  [{tp.index:4d}] {marker}B:{b_str!r}  T:{t_str!r}")
            lines.append("")

    lines.append("─── Classification Reasoning ───")
    lines.append(result.classification_reasoning)
    lines.append("")

    lines.append("─── Suggested Next Steps ───")
    for i, step in enumerate(result.suggested_next_steps, 1):
        lines.append(f"  {i}. {step}")

    return "\n".join(lines)
