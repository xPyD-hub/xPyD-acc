"""KV cache loading, comparison, and divergence reporting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LayerMetrics:
    """Numerical comparison metrics for a single KV cache layer."""

    layer_name: str
    max_abs_diff: float
    mean_abs_diff: float
    cosine_similarity: float
    divergent: bool


@dataclass
class KVCacheReport:
    """Full comparison report across all layers."""

    baseline_path: str
    target_path: str
    layers: list[LayerMetrics] = field(default_factory=list)
    divergent_layers: list[str] = field(default_factory=list)
    match: bool = True

    def to_json(self) -> str:
        """Serialize report to JSON string."""
        return json.dumps(asdict(self), indent=2)


class KVCacheLoader:
    """Load KV cache dumps from numpy npz files."""

    @staticmethod
    def load(path: str | Path) -> dict[str, np.ndarray]:
        """Load an npz file and return a dict of layer_name -> array."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"KV cache file not found: {path}")
        data = np.load(str(path))
        return dict(data)


class KVCacheComparator:
    """Compare two KV cache dumps layer by layer."""

    def __init__(
        self,
        max_abs_threshold: float = 1e-3,
        cosine_threshold: float = 0.999,
    ) -> None:
        self.max_abs_threshold = max_abs_threshold
        self.cosine_threshold = cosine_threshold

    def compare(
        self,
        baseline: dict[str, np.ndarray],
        target: dict[str, np.ndarray],
        baseline_path: str = "",
        target_path: str = "",
    ) -> KVCacheReport:
        """Compare two KV cache dicts and produce a report."""
        all_keys = sorted(set(baseline.keys()) | set(target.keys()))
        report = KVCacheReport(baseline_path=baseline_path, target_path=target_path)

        for key in all_keys:
            if key not in baseline:
                metrics = LayerMetrics(
                    layer_name=key,
                    max_abs_diff=float("inf"),
                    mean_abs_diff=float("inf"),
                    cosine_similarity=0.0,
                    divergent=True,
                )
            elif key not in target:
                metrics = LayerMetrics(
                    layer_name=key,
                    max_abs_diff=float("inf"),
                    mean_abs_diff=float("inf"),
                    cosine_similarity=0.0,
                    divergent=True,
                )
            else:
                metrics = self._compare_arrays(key, baseline[key], target[key])

            report.layers.append(metrics)
            if metrics.divergent:
                report.divergent_layers.append(key)
                report.match = False

        return report

    def _compare_arrays(self, name: str, a: np.ndarray, b: np.ndarray) -> LayerMetrics:
        """Compare two arrays and compute metrics."""
        if a.shape != b.shape:
            return LayerMetrics(
                layer_name=name,
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                cosine_similarity=0.0,
                divergent=True,
            )

        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))

        # Cosine similarity on flattened vectors
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)

        if norm_a == 0.0 and norm_b == 0.0:
            cosine_sim = 1.0
        elif norm_a == 0.0 or norm_b == 0.0:
            cosine_sim = 0.0
        else:
            cosine_sim = float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

        divergent = max_abs > self.max_abs_threshold or cosine_sim < self.cosine_threshold

        return LayerMetrics(
            layer_name=name,
            max_abs_diff=max_abs,
            mean_abs_diff=mean_abs,
            cosine_similarity=cosine_sim,
            divergent=divergent,
        )

    @staticmethod
    def format_report(report: KVCacheReport) -> str:
        """Format report as human-readable text."""
        lines = [
            "=== KV Cache Comparison Report ===",
            f"Baseline: {report.baseline_path}",
            f"Target:   {report.target_path}",
            f"Layers:   {len(report.layers)}",
            "",
        ]

        for m in report.layers:
            status = "❌" if m.divergent else "✅"
            lines.append(
                f"  {status} {m.layer_name}: "
                f"max_abs={m.max_abs_diff:.6e}, "
                f"mean_abs={m.mean_abs_diff:.6e}, "
                f"cosine={m.cosine_similarity:.6f}"
            )

        lines.append("")
        if report.match:
            lines.append("✅ MATCH — all layers within tolerance")
        else:
            lines.append(
                f"❌ DIVERGENCE — {len(report.divergent_layers)} layer(s): "
                + ", ".join(report.divergent_layers)
            )

        return "\n".join(lines)
