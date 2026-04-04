"""Output comparison utilities: text diff, edit distance, token-level diff, semantic similarity."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenDiff:
    """A single diff entry between two token sequences."""

    tag: str  # "equal", "replace", "insert", "delete"
    baseline_tokens: list[str]
    target_tokens: list[str]
    baseline_range: tuple[int, int]  # (start, end) indices
    target_range: tuple[int, int]


@dataclass
class OutputComparisonReport:
    """Full comparison report between two outputs."""

    baseline_text: str
    target_text: str
    exact_match: bool
    edit_distance: int
    normalized_edit_distance: float  # 0.0 = identical, 1.0 = completely different
    token_diffs: list[TokenDiff] = field(default_factory=list)
    semantic_similarity: float | None = None  # None if not computed
    baseline_token_count: int = 0
    target_token_count: int = 0

    def summary(self) -> str:
        """One-line summary."""
        if self.exact_match:
            return "✅ EXACT MATCH"
        return (
            f"❌ MISMATCH — edit_distance={self.edit_distance}, "
            f"normalized={self.normalized_edit_distance:.4f}, "
            f"tokens: {self.baseline_token_count} vs {self.target_token_count}"
        )


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,        # insert
                prev_row[j + 1] + 1,    # delete
                prev_row[j] + cost,     # replace
            ))
        prev_row = curr_row
    return prev_row[-1]


def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace tokenizer. Splits on whitespace, keeps punctuation attached."""
    return text.split()


def compute_token_diffs(baseline_tokens: list[str], target_tokens: list[str]) -> list[TokenDiff]:
    """Compute token-level diffs using SequenceMatcher."""
    import difflib

    matcher = difflib.SequenceMatcher(None, baseline_tokens, target_tokens)
    diffs: list[TokenDiff] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        diffs.append(TokenDiff(
            tag=tag,
            baseline_tokens=baseline_tokens[i1:i2],
            target_tokens=target_tokens[j1:j2],
            baseline_range=(i1, i2),
            target_range=(j1, j2),
        ))
    return diffs


def reassemble_streaming_chunks(chunks: list[str]) -> str:
    """Reassemble streaming output chunks into a single string."""
    return "".join(chunks)


class OutputComparator:
    """Compare two text outputs with multiple methods."""

    def compare(
        self,
        baseline: str,
        target: str,
        *,
        baseline_embeddings: list[float] | None = None,
        target_embeddings: list[float] | None = None,
    ) -> OutputComparisonReport:
        """Run full comparison between baseline and target text."""
        exact = baseline == target
        edit_dist = levenshtein_distance(baseline, target)
        max_len = max(len(baseline), len(target))
        norm_dist = edit_dist / max_len if max_len > 0 else 0.0

        b_tokens = tokenize_simple(baseline)
        t_tokens = tokenize_simple(target)
        token_diffs = compute_token_diffs(b_tokens, t_tokens)

        sem_sim: float | None = None
        if baseline_embeddings is not None and target_embeddings is not None:
            sem_sim = _cosine_similarity(baseline_embeddings, target_embeddings)

        return OutputComparisonReport(
            baseline_text=baseline,
            target_text=target,
            exact_match=exact,
            edit_distance=edit_dist,
            normalized_edit_distance=norm_dist,
            token_diffs=token_diffs,
            semantic_similarity=sem_sim,
            baseline_token_count=len(b_tokens),
            target_token_count=len(t_tokens),
        )

    def compare_streaming(
        self,
        baseline_chunks: list[str],
        target_chunks: list[str],
        *,
        baseline_embeddings: list[float] | None = None,
        target_embeddings: list[float] | None = None,
    ) -> OutputComparisonReport:
        """Compare streaming outputs by first reassembling chunks."""
        baseline = reassemble_streaming_chunks(baseline_chunks)
        target = reassemble_streaming_chunks(target_chunks)
        return self.compare(
            baseline, target,
            baseline_embeddings=baseline_embeddings,
            target_embeddings=target_embeddings,
        )

    @staticmethod
    def format_report(report: OutputComparisonReport) -> str:
        """Format comparison report as human-readable text."""
        lines = [
            "=== Output Comparison Report ===",
            f"Exact match: {'Yes' if report.exact_match else 'No'}",
            f"Edit distance: {report.edit_distance} "
            f"(normalized: {report.normalized_edit_distance:.4f})",
            f"Baseline tokens: {report.baseline_token_count}",
            f"Target tokens:   {report.target_token_count}",
        ]

        if report.semantic_similarity is not None:
            lines.append(f"Semantic similarity: {report.semantic_similarity:.6f}")

        if not report.exact_match:
            lines.append("")
            lines.append("--- Token Diffs ---")
            for d in report.token_diffs:
                if d.tag == "equal":
                    continue
                if d.tag == "replace":
                    lines.append(
                        f"  [{d.baseline_range[0]}:{d.baseline_range[1]}] "
                        f"REPLACE {d.baseline_tokens!r} → {d.target_tokens!r}"
                    )
                elif d.tag == "delete":
                    lines.append(
                        f"  [{d.baseline_range[0]}:{d.baseline_range[1]}] "
                        f"DELETE {d.baseline_tokens!r}"
                    )
                elif d.tag == "insert":
                    lines.append(
                        f"  [{d.target_range[0]}:{d.target_range[1]}] "
                        f"INSERT {d.target_tokens!r}"
                    )

        return "\n".join(lines)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math

    if len(a) != len(b):
        msg = f"Embedding dimension mismatch: {len(a)} vs {len(b)}"
        raise ValueError(msg)

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
