"""Side-by-side token diff viewer for divergent samples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .batch_compare import BatchReport, SampleResult, load_report


@dataclass
class TokenDiffLine:
    """A single line in the side-by-side token diff."""

    index: int
    baseline_token: str | None
    target_token: str | None
    status: str  # "match", "mismatch", "logprob_warning", "baseline_only", "target_only"
    baseline_logprob: float | None = None
    target_logprob: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TokenDiff:
    """Complete token diff for a sample."""

    sample_id: str
    divergence_index: int | None
    lines: list[TokenDiffLine] = field(default_factory=list)
    baseline_length: int = 0
    target_length: int = 0

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "divergence_index": self.divergence_index,
            "baseline_length": self.baseline_length,
            "target_length": self.target_length,
            "lines": [line.to_dict() for line in self.lines],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _tokenize_simple(text: str) -> list[str]:
    """Simple whitespace-based tokenization for display purposes."""
    if not text:
        return []
    # Split on whitespace but keep tokens meaningful
    tokens = []
    current = ""
    for ch in text:
        if ch in (" ", "\n", "\t"):
            if current:
                tokens.append(current)
            tokens.append(ch)
            current = ""
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens


def build_token_diff(
    result: SampleResult,
    context: int = 10,
) -> TokenDiff:
    """Build a token diff from a sample result.

    Args:
        result: The sample result to diff.
        context: Number of tokens before/after divergence to show.

    Returns:
        TokenDiff with aligned lines.
    """
    baseline_tokens = _tokenize_simple(result.baseline_output)
    target_tokens = _tokenize_simple(result.target_output)

    diff = TokenDiff(
        sample_id=result.sample_id,
        divergence_index=result.first_divergence_index,
        baseline_length=len(baseline_tokens),
        target_length=len(target_tokens),
    )

    max_len = max(len(baseline_tokens), len(target_tokens))
    if max_len == 0:
        return diff

    # Determine range to show based on context window
    div_idx = result.first_divergence_index
    if div_idx is not None:
        start = max(0, div_idx - context)
        end = min(max_len, div_idx + context + 1)
    else:
        # No divergence — show all (up to 2*context+1)
        start = 0
        end = min(max_len, 2 * context + 1)

    for i in range(start, end):
        b_tok = baseline_tokens[i] if i < len(baseline_tokens) else None
        t_tok = target_tokens[i] if i < len(target_tokens) else None

        if b_tok is None:
            status = "target_only"
        elif t_tok is None:
            status = "baseline_only"
        elif b_tok == t_tok:
            # Check if near divergence with low logprob
            if (
                div_idx is not None
                and abs(i - div_idx) <= 2
                and result.logprob_gap is not None
                and result.logprob_gap < 0.1
            ):
                status = "logprob_warning"
            else:
                status = "match"
        else:
            status = "mismatch"

        b_logprob = None
        t_logprob = None
        if status == "mismatch" and div_idx is not None and i == div_idx:
            b_logprob = result.baseline_logprob_at_divergence
            t_logprob = result.target_logprob_at_divergence

        diff.lines.append(
            TokenDiffLine(
                index=i,
                baseline_token=b_tok,
                target_token=t_tok,
                status=status,
                baseline_logprob=b_logprob,
                target_logprob=t_logprob,
            )
        )

    return diff


def format_token_diff(diff: TokenDiff, plain: bool = False) -> str:
    """Format a token diff for terminal display.

    Args:
        diff: The token diff to format.
        plain: If True, use plain text (no ANSI colors).

    Returns:
        Formatted string.
    """
    if not diff.lines:
        return f"Sample {diff.sample_id}: no tokens to display"

    # Color codes
    if plain:
        green = red = yellow = cyan = reset = bold = ""
    else:
        green = "\033[32m"
        red = "\033[31m"
        yellow = "\033[33m"
        cyan = "\033[36m"
        bold = "\033[1m"
        reset = "\033[0m"

    color_map = {
        "match": green,
        "mismatch": red,
        "logprob_warning": yellow,
        "baseline_only": cyan,
        "target_only": cyan,
    }

    lines = []
    lines.append(
        f"{bold}Token Diff: {diff.sample_id}{reset}"
        f" (baseline={diff.baseline_length} tokens, target={diff.target_length} tokens)"
    )
    if diff.divergence_index is not None:
        lines.append(f"First divergence at token index {diff.divergence_index}")
    lines.append("")

    # Header
    idx_w = 5
    tok_w = 30
    header = f"{'Idx':>{idx_w}}  {'Baseline':<{tok_w}}  {'Target':<{tok_w}}  Status"
    lines.append(header)
    lines.append("-" * len(header))

    for dl in diff.lines:
        color = color_map.get(dl.status, "")
        b_str = repr(dl.baseline_token) if dl.baseline_token is not None else "---"
        t_str = repr(dl.target_token) if dl.target_token is not None else "---"

        # Add logprob annotation for mismatched tokens
        logprob_info = ""
        if dl.baseline_logprob is not None or dl.target_logprob is not None:
            b_lp = f"{dl.baseline_logprob:.4f}" if dl.baseline_logprob is not None else "?"
            t_lp = f"{dl.target_logprob:.4f}" if dl.target_logprob is not None else "?"
            logprob_info = f" [logprob b={b_lp} t={t_lp}]"

        marker = ""
        if diff.divergence_index is not None and diff.divergence_index == dl.index:
            marker = " ◀ DIVERGE"

        line = (
            f"{color}{dl.index:>{idx_w}}  {b_str:<{tok_w}}  {t_str:<{tok_w}}"
            f"  {dl.status}{logprob_info}{marker}{reset}"
        )
        lines.append(line)

    return "\n".join(lines)


def build_from_report(
    report: BatchReport,
    sample_id: str,
    context: int = 10,
) -> TokenDiff:
    """Build token diff for a specific sample in a report."""
    for result in report.results:
        if result.sample_id == sample_id:
            return build_token_diff(result, context=context)
    raise ValueError(f"Sample '{sample_id}' not found in report")


def build_all_divergent(
    report: BatchReport,
    context: int = 10,
) -> list[TokenDiff]:
    """Build token diffs for all divergent samples."""
    return [
        build_token_diff(r, context=context)
        for r in report.results
        if r.is_divergent()
    ]


def diff_from_file(
    report_path: str,
    sample_id: str | None = None,
    all_divergent: bool = False,
    context: int = 10,
) -> list[TokenDiff]:
    """Load report and build token diffs.

    Args:
        report_path: Path to batch report JSON.
        sample_id: Specific sample to diff.
        all_divergent: If True, diff all divergent samples.
        context: Context window size.

    Returns:
        List of TokenDiff objects.
    """
    report = load_report(report_path)
    if all_divergent:
        return build_all_divergent(report, context=context)
    if sample_id:
        return [build_from_report(report, sample_id, context=context)]
    raise ValueError("Must specify --sample or --all-divergent")
