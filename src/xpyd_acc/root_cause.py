"""Divergence root cause heuristics for batch reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .batch_compare import BatchReport, SampleResult, load_report


@dataclass
class Evidence:
    """A single piece of evidence supporting a root cause classification."""

    rule: str
    description: str
    sample_count: int
    sample_ids: list[str] = field(default_factory=list)


@dataclass
class RootCauseAnalysis:
    """Result of root cause analysis on a batch report."""

    classification: str  # prefill, kv_transfer, decode, truncation, mixed, inconclusive
    confidence: float  # 0.0 - 1.0
    evidence: list[Evidence] = field(default_factory=list)
    suggested_steps: list[str] = field(default_factory=list)
    total_divergent: int = 0
    total_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "classification": self.classification,
            "confidence": self.confidence,
            "evidence": [
                {
                    "rule": e.rule,
                    "description": e.description,
                    "sample_count": e.sample_count,
                    "sample_ids": e.sample_ids,
                }
                for e in self.evidence
            ],
            "suggested_steps": self.suggested_steps,
            "total_divergent": self.total_divergent,
            "total_samples": self.total_samples,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> RootCauseAnalysis:
        return cls(
            classification=d["classification"],
            confidence=d["confidence"],
            evidence=[
                Evidence(
                    rule=e["rule"],
                    description=e["description"],
                    sample_count=e["sample_count"],
                    sample_ids=e.get("sample_ids", []),
                )
                for e in d.get("evidence", [])
            ],
            suggested_steps=d.get("suggested_steps", []),
            total_divergent=d.get("total_divergent", 0),
            total_samples=d.get("total_samples", 0),
        )


# Thresholds for heuristic rules
EARLY_DIVERGENCE_INDEX = 5
HIGH_LOGPROB_GAP = 0.5
LOW_LOGPROB_GAP = 0.1


def _classify_sample(result: SampleResult) -> str | None:
    """Classify a single divergent sample by probable root cause."""
    if result.exact_match:
        return None

    idx = result.first_divergence_index
    gap = result.logprob_gap
    is_truncated = (
        result.baseline_finish_reason == "length"
        or result.target_finish_reason == "length"
    )

    if is_truncated and not result.exact_match:
        return "truncation"

    if idx is not None and idx < EARLY_DIVERGENCE_INDEX:
        if gap is not None and gap >= HIGH_LOGPROB_GAP:
            return "prefill"
        if gap is None:
            return "prefill"

    if idx is not None and idx >= EARLY_DIVERGENCE_INDEX:
        if gap is not None and gap < LOW_LOGPROB_GAP:
            return "decode"

    if idx is not None and idx >= EARLY_DIVERGENCE_INDEX:
        if gap is not None and gap >= HIGH_LOGPROB_GAP:
            return "kv_transfer"

    return "unknown"


def analyze_root_cause(report: BatchReport) -> RootCauseAnalysis:
    """Analyze a batch report and suggest probable divergence root cause."""
    divergent = [r for r in report.results if not r.exact_match]
    total_divergent = len(divergent)
    total_samples = len(report.results)

    if total_divergent == 0:
        return RootCauseAnalysis(
            classification="no_divergence",
            confidence=1.0,
            suggested_steps=["No divergent samples found. All outputs match."],
            total_divergent=0,
            total_samples=total_samples,
        )

    # Classify each divergent sample
    buckets: dict[str, list[str]] = {
        "prefill": [],
        "kv_transfer": [],
        "decode": [],
        "truncation": [],
        "unknown": [],
    }

    for sample in divergent:
        cause = _classify_sample(sample) or "unknown"
        buckets[cause].append(sample.sample_id)

    # Build evidence
    evidence: list[Evidence] = []

    if buckets["prefill"]:
        evidence.append(
            Evidence(
                rule="early_divergence_high_gap",
                description=(
                    f"Early divergence (index < {EARLY_DIVERGENCE_INDEX}) with high "
                    f"logprob gap (>= {HIGH_LOGPROB_GAP}): likely prefill stage issue"
                ),
                sample_count=len(buckets["prefill"]),
                sample_ids=buckets["prefill"][:10],
            )
        )

    if buckets["kv_transfer"]:
        evidence.append(
            Evidence(
                rule="mid_divergence_high_gap",
                description=(
                    f"Mid/late divergence (index >= {EARLY_DIVERGENCE_INDEX}) with high "
                    f"logprob gap (>= {HIGH_LOGPROB_GAP}): likely KV cache transfer issue"
                ),
                sample_count=len(buckets["kv_transfer"]),
                sample_ids=buckets["kv_transfer"][:10],
            )
        )

    if buckets["decode"]:
        evidence.append(
            Evidence(
                rule="late_divergence_low_gap",
                description=(
                    f"Mid/late divergence (index >= {EARLY_DIVERGENCE_INDEX}) with low "
                    f"logprob gap (< {LOW_LOGPROB_GAP}): likely decode accumulation issue"
                ),
                sample_count=len(buckets["decode"]),
                sample_ids=buckets["decode"][:10],
            )
        )

    if buckets["truncation"]:
        evidence.append(
            Evidence(
                rule="truncation_correlated",
                description=(
                    "Divergence correlated with output truncation (finish_reason='length'): "
                    "likely max_tokens or stop sequence mismatch"
                ),
                sample_count=len(buckets["truncation"]),
                sample_ids=buckets["truncation"][:10],
            )
        )

    if buckets["unknown"]:
        evidence.append(
            Evidence(
                rule="unclassified",
                description="Samples that don't match any specific heuristic pattern",
                sample_count=len(buckets["unknown"]),
                sample_ids=buckets["unknown"][:10],
            )
        )

    # Determine dominant cause
    cause_counts = {
        k: len(v) for k, v in buckets.items() if k != "unknown" and len(v) > 0
    }

    if not cause_counts:
        classification = "inconclusive"
        confidence = 0.0
    elif len(cause_counts) == 1:
        classification = next(iter(cause_counts))
        classified = sum(cause_counts.values())
        confidence = classified / total_divergent
    else:
        # Multiple causes
        dominant = max(cause_counts, key=cause_counts.get)  # type: ignore[arg-type]
        dominant_frac = cause_counts[dominant] / total_divergent
        if dominant_frac >= 0.6:
            classification = dominant
            confidence = dominant_frac
        else:
            classification = "mixed"
            confidence = max(cause_counts.values()) / total_divergent

    # Build suggested next steps
    suggested_steps = _build_suggestions(classification, buckets)

    return RootCauseAnalysis(
        classification=classification,
        confidence=round(confidence, 3),
        evidence=evidence,
        suggested_steps=suggested_steps,
        total_divergent=total_divergent,
        total_samples=total_samples,
    )


def _build_suggestions(
    classification: str, buckets: dict[str, list[str]]
) -> list[str]:
    """Build suggested debugging steps based on classification."""
    steps: list[str] = []

    if classification == "prefill":
        steps.extend(
            [
                "Check prefill stage output: run with prefill-only mode and compare first tokens",
                "Verify prompt tokenization is identical on both endpoints",
                "Check if model weights/config differ between prefill and decode nodes",
            ]
        )
    elif classification == "kv_transfer":
        steps.extend(
            [
                "Dump and compare KV caches: use `xpyd-acc kv-compare` with cache dumps",
                "Check KV cache serialization format and precision (FP16 vs FP32 vs BF16)",
                "Verify network transfer integrity between prefill and decode stages",
            ]
        )
    elif classification == "decode":
        steps.extend(
            [
                "Run `xpyd-acc reproducibility` to check if decode output is non-deterministic",
                "Compare with temperature=0 and fixed seed to isolate randomness",
                "Check decode stage attention computation precision settings",
            ]
        )
    elif classification == "truncation":
        steps.extend(
            [
                "Ensure max_tokens is set identically on both endpoints",
                "Check stop sequence configuration matches between endpoints",
                "Rerun with higher max_tokens to see if divergence disappears",
            ]
        )
    elif classification == "mixed":
        steps.extend(
            [
                "Multiple root causes detected — investigate each category separately",
                "Use `xpyd-acc filter` to isolate samples by classification",
                "Run `xpyd-acc explain` on representative samples from each category",
            ]
        )
    else:
        steps.extend(
            [
                "Insufficient evidence for automatic classification",
                "Use `xpyd-acc explain --sample <id>` on individual divergent samples",
                "Collect more samples or enable logprobs for better analysis",
            ]
        )

    return steps


def format_root_cause(analysis: RootCauseAnalysis) -> str:
    """Format root cause analysis for terminal display."""
    lines: list[str] = []
    lines.append("╔══════════════════════════════════════════╗")
    lines.append("║     Root Cause Analysis                  ║")
    lines.append("╚══════════════════════════════════════════╝")
    lines.append("")
    lines.append(
        f"Samples: {analysis.total_divergent} divergent / "
        f"{analysis.total_samples} total"
    )
    lines.append(
        f"Classification: {analysis.classification.upper().replace('_', ' ')}"
    )
    lines.append(f"Confidence: {analysis.confidence:.1%}")
    lines.append("")

    if analysis.evidence:
        lines.append("Evidence:")
        for ev in analysis.evidence:
            lines.append(f"  [{ev.rule}] ({ev.sample_count} samples)")
            lines.append(f"    {ev.description}")
        lines.append("")

    if analysis.suggested_steps:
        lines.append("Suggested Next Steps:")
        for i, step in enumerate(analysis.suggested_steps, 1):
            lines.append(f"  {i}. {step}")

    return "\n".join(lines)


def analyze_from_file(path: str | Path) -> RootCauseAnalysis:
    """Load a batch report from file and analyze root cause."""
    report = load_report(Path(path))
    return analyze_root_cause(report)
