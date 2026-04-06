"""Hardware Precision Baseline Library.

Maintains reference precision profiles for common GPU hardware configurations
and classifies observed numerical differences as expected hardware variance
or likely software bugs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PrecisionRange:
    """Expected numerical range for a specific metric."""

    metric: str  # "max_abs_diff", "mean_abs_diff", "cosine_sim"
    expected_min: float
    expected_max: float
    description: str = ""

    def contains(self, value: float) -> bool:
        """Check if value falls within the expected range."""
        return self.expected_min <= value <= self.expected_max

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "expected_min": self.expected_min,
            "expected_max": self.expected_max,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PrecisionRange:
        return cls(
            metric=data["metric"],
            expected_min=data["expected_min"],
            expected_max=data["expected_max"],
            description=data.get("description", ""),
        )


@dataclass
class HardwareProfile:
    """A hardware configuration with expected precision characteristics."""

    name: str
    gpu_arch: str  # e.g. "A100", "H100", "H200", "Gaudi2", "Gaudi3"
    precision_mode: str  # e.g. "FP16", "BF16", "FP8", "INT8-KV"
    attention_impl: str  # e.g. "FlashAttention-v2", "PagedAttention", "xFormers"
    tp_degree: int  # tensor parallelism degree
    ranges: list[PrecisionRange] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "gpu_arch": self.gpu_arch,
            "precision_mode": self.precision_mode,
            "attention_impl": self.attention_impl,
            "tp_degree": self.tp_degree,
            "ranges": [r.to_dict() for r in self.ranges],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HardwareProfile:
        return cls(
            name=data["name"],
            gpu_arch=data["gpu_arch"],
            precision_mode=data["precision_mode"],
            attention_impl=data["attention_impl"],
            tp_degree=data["tp_degree"],
            ranges=[PrecisionRange.from_dict(r) for r in data.get("ranges", [])],
            metadata=data.get("metadata", {}),
        )

    def get_range(self, metric: str) -> PrecisionRange | None:
        """Look up the expected range for a given metric."""
        for r in self.ranges:
            if r.metric == metric:
                return r
        return None


@dataclass
class DifferenceVerdict:
    """Classification result for an observed numerical difference."""

    metric: str
    observed_value: float
    expected_range: PrecisionRange | None
    classification: str  # "expected", "suspicious", "likely_bug", "unknown"
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "observed_value": self.observed_value,
            "expected_range": self.expected_range.to_dict() if self.expected_range else None,
            "classification": self.classification,
            "reasoning": self.reasoning,
        }


@dataclass
class ClassificationReport:
    """Full classification report for a set of observed differences."""

    profile_name: str
    verdicts: list[DifferenceVerdict]

    @property
    def overall_classification(self) -> str:
        """Return the worst classification across all verdicts."""
        priority = {"likely_bug": 3, "suspicious": 2, "unknown": 1, "expected": 0}
        if not self.verdicts:
            return "unknown"
        worst = max(self.verdicts, key=lambda v: priority.get(v.classification, 0))
        return worst.classification

    def to_dict(self) -> dict:
        return {
            "profile_name": self.profile_name,
            "overall_classification": self.overall_classification,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def classify_difference(
    profile: HardwareProfile,
    observations: dict[str, float],
) -> ClassificationReport:
    """Classify observed numerical differences against a hardware profile.

    Args:
        profile: hardware profile with expected ranges
        observations: dict mapping metric names to observed values
            e.g. {"max_abs_diff": 0.001, "mean_abs_diff": 0.0001, "cosine_sim": 0.9999}

    Returns:
        ClassificationReport with per-metric verdicts
    """
    verdicts: list[DifferenceVerdict] = []

    for metric, value in observations.items():
        expected = profile.get_range(metric)

        if expected is None:
            verdicts.append(DifferenceVerdict(
                metric=metric,
                observed_value=value,
                expected_range=None,
                classification="unknown",
                reasoning=f"No expected range for '{metric}' in profile '{profile.name}'",
            ))
            continue

        if expected.contains(value):
            verdicts.append(DifferenceVerdict(
                metric=metric,
                observed_value=value,
                expected_range=expected,
                classification="expected",
                reasoning=(
                    f"{metric}={value} is within expected range "
                    f"[{expected.expected_min}, {expected.expected_max}] "
                    f"for {profile.name}"
                ),
            ))
        else:
            # Determine severity: how far outside the range?
            if metric == "cosine_sim":
                # For cosine similarity, below min is bad
                if value < expected.expected_min:
                    gap = expected.expected_min - value
                    severity = "likely_bug" if gap > 0.01 else "suspicious"
                else:
                    severity = "expected"  # above max for cosine is fine
            else:
                # For diff metrics, above max is bad
                if value > expected.expected_max:
                    max_val = expected.expected_max
                    ratio = value / max_val if max_val > 0 else float("inf")
                    severity = "likely_bug" if ratio > 2.0 else "suspicious"
                else:
                    severity = "expected"  # below min for diff is fine (less error)

            verdicts.append(DifferenceVerdict(
                metric=metric,
                observed_value=value,
                expected_range=expected,
                classification=severity,
                reasoning=(
                    f"{metric}={value} is outside expected range "
                    f"[{expected.expected_min}, {expected.expected_max}] "
                    f"for {profile.name}"
                ),
            ))

    return ClassificationReport(profile_name=profile.name, verdicts=verdicts)


# --- Built-in Profiles ---

_BUILTIN_PROFILES: list[HardwareProfile] = [
    HardwareProfile(
        name="a100-bf16-tp1",
        gpu_arch="A100",
        precision_mode="BF16",
        attention_impl="FlashAttention-v2",
        tp_degree=1,
        ranges=[
            PrecisionRange("max_abs_diff", 0.0, 0.002, "BF16 rounding on A100"),
            PrecisionRange("mean_abs_diff", 0.0, 0.0005, "Average BF16 error"),
            PrecisionRange("cosine_sim", 0.9995, 1.0, "Expected cosine similarity"),
        ],
    ),
    HardwareProfile(
        name="a100-fp16-tp1",
        gpu_arch="A100",
        precision_mode="FP16",
        attention_impl="FlashAttention-v2",
        tp_degree=1,
        ranges=[
            PrecisionRange("max_abs_diff", 0.0, 0.001, "FP16 rounding on A100"),
            PrecisionRange("mean_abs_diff", 0.0, 0.0002, "Average FP16 error"),
            PrecisionRange("cosine_sim", 0.9998, 1.0, "Expected cosine similarity"),
        ],
    ),
    HardwareProfile(
        name="h100-bf16-tp4",
        gpu_arch="H100",
        precision_mode="BF16",
        attention_impl="FlashAttention-v3",
        tp_degree=4,
        ranges=[
            PrecisionRange("max_abs_diff", 0.0, 0.005, "BF16 + TP=4 accumulation on H100"),
            PrecisionRange("mean_abs_diff", 0.0, 0.001, "Average BF16 TP=4 error"),
            PrecisionRange("cosine_sim", 0.999, 1.0, "Expected cosine similarity"),
        ],
    ),
    HardwareProfile(
        name="h100-fp8-tp4",
        gpu_arch="H100",
        precision_mode="FP8",
        attention_impl="FlashAttention-v3",
        tp_degree=4,
        ranges=[
            PrecisionRange("max_abs_diff", 0.0, 0.02, "FP8 quantization noise on H100"),
            PrecisionRange("mean_abs_diff", 0.0, 0.005, "Average FP8 error"),
            PrecisionRange("cosine_sim", 0.995, 1.0, "Expected cosine similarity with FP8"),
        ],
    ),
    HardwareProfile(
        name="h200-bf16-tp8",
        gpu_arch="H200",
        precision_mode="BF16",
        attention_impl="FlashAttention-v3",
        tp_degree=8,
        ranges=[
            PrecisionRange("max_abs_diff", 0.0, 0.008, "BF16 + TP=8 on H200"),
            PrecisionRange("mean_abs_diff", 0.0, 0.002, "Average BF16 TP=8 error"),
            PrecisionRange("cosine_sim", 0.998, 1.0, "Expected cosine similarity"),
        ],
    ),
    HardwareProfile(
        name="gaudi2-bf16-tp1",
        gpu_arch="Gaudi2",
        precision_mode="BF16",
        attention_impl="PagedAttention",
        tp_degree=1,
        ranges=[
            PrecisionRange("max_abs_diff", 0.0, 0.003, "BF16 on Gaudi2"),
            PrecisionRange("mean_abs_diff", 0.0, 0.0008, "Average BF16 error on Gaudi2"),
            PrecisionRange("cosine_sim", 0.9993, 1.0, "Expected cosine similarity"),
        ],
    ),
    HardwareProfile(
        name="gaudi3-fp8-tp4",
        gpu_arch="Gaudi3",
        precision_mode="FP8",
        attention_impl="PagedAttention",
        tp_degree=4,
        ranges=[
            PrecisionRange("max_abs_diff", 0.0, 0.025, "FP8 on Gaudi3 TP=4"),
            PrecisionRange("mean_abs_diff", 0.0, 0.006, "Average FP8 error on Gaudi3"),
            PrecisionRange("cosine_sim", 0.994, 1.0, "Expected cosine similarity"),
        ],
    ),
    HardwareProfile(
        name="a100-int8kv-tp2",
        gpu_arch="A100",
        precision_mode="INT8-KV",
        attention_impl="PagedAttention",
        tp_degree=2,
        ranges=[
            PrecisionRange("max_abs_diff", 0.0, 0.015, "INT8 KV cache quantization on A100"),
            PrecisionRange("mean_abs_diff", 0.0, 0.004, "Average INT8-KV error"),
            PrecisionRange("cosine_sim", 0.996, 1.0, "Expected cosine similarity with INT8-KV"),
        ],
    ),
]


class BaselineDB:
    """Database of hardware precision profiles."""

    def __init__(self) -> None:
        self._profiles: dict[str, HardwareProfile] = {}
        # Load built-in profiles
        for p in _BUILTIN_PROFILES:
            self._profiles[p.name] = p

    def list_profiles(self) -> list[str]:
        """Return sorted list of available profile names."""
        return sorted(self._profiles.keys())

    def get_profile(self, name: str) -> HardwareProfile | None:
        """Look up a profile by name."""
        return self._profiles.get(name)

    def add_profile(self, profile: HardwareProfile) -> None:
        """Add or replace a profile."""
        self._profiles[profile.name] = profile

    def remove_profile(self, name: str) -> bool:
        """Remove a profile. Returns True if it existed."""
        if name in self._profiles:
            del self._profiles[name]
            return True
        return False

    def export_json(self, path: str | Path) -> None:
        """Export all profiles to a JSON file."""
        data = {
            "version": 1,
            "profiles": [p.to_dict() for p in self._profiles.values()],
        }
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    def import_json(self, path: str | Path) -> int:
        """Import profiles from a JSON file. Returns count of imported profiles."""
        data = json.loads(Path(path).read_text())
        profiles = data.get("profiles", [])
        count = 0
        for p_data in profiles:
            profile = HardwareProfile.from_dict(p_data)
            self._profiles[profile.name] = profile
            count += 1
        return count

    def find_profiles(
        self,
        gpu_arch: str | None = None,
        precision_mode: str | None = None,
        tp_degree: int | None = None,
    ) -> list[HardwareProfile]:
        """Find profiles matching given criteria."""
        results = []
        for p in self._profiles.values():
            if gpu_arch and p.gpu_arch.lower() != gpu_arch.lower():
                continue
            if precision_mode and p.precision_mode.lower() != precision_mode.lower():
                continue
            if tp_degree is not None and p.tp_degree != tp_degree:
                continue
            results.append(p)
        return sorted(results, key=lambda p: p.name)


def format_profile(profile: HardwareProfile) -> str:
    """Format a single profile for terminal display."""
    lines: list[str] = []
    lines.append(f"Profile: {profile.name}")
    lines.append(f"  GPU: {profile.gpu_arch}")
    lines.append(f"  Precision: {profile.precision_mode}")
    lines.append(f"  Attention: {profile.attention_impl}")
    lines.append(f"  TP Degree: {profile.tp_degree}")

    if profile.ranges:
        lines.append("  Expected Ranges:")
        for r in profile.ranges:
            desc = f" ({r.description})" if r.description else ""
            lines.append(f"    {r.metric}: [{r.expected_min}, {r.expected_max}]{desc}")

    if profile.metadata:
        lines.append(f"  Metadata: {profile.metadata}")

    return "\n".join(lines)


def format_profile_list(db: BaselineDB) -> str:
    """Format the profile list for terminal display."""
    names = db.list_profiles()
    if not names:
        return "No profiles available."

    lines: list[str] = []
    hdr = f"{'Name':<25} {'GPU':<10} {'Precision':<10} {'Attention':<20} {'TP':>3}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for name in names:
        p = db.get_profile(name)
        if p:
            lines.append(
                f"{p.name:<25} {p.gpu_arch:<10} {p.precision_mode:<10} "
                f"{p.attention_impl:<20} {p.tp_degree:>3}"
            )

    return "\n".join(lines)


def format_classification(report: ClassificationReport) -> str:
    """Format a classification report for terminal display."""
    lines: list[str] = []
    icon_map = {
        "expected": "✅",
        "suspicious": "⚠️",
        "likely_bug": "❌",
        "unknown": "❓",
    }
    overall_icon = icon_map.get(report.overall_classification, "❓")
    lines.append(f"Hardware Profile: {report.profile_name}")
    lines.append(f"Overall: {overall_icon} {report.overall_classification}")
    lines.append("")

    for v in report.verdicts:
        icon = icon_map.get(v.classification, "❓")
        lines.append(f"  {icon} {v.metric} = {v.observed_value}")
        lines.append(f"     {v.reasoning}")

    return "\n".join(lines)
