"""Framework-level inference hooks for capturing intermediate states.

Provides a hook protocol for injecting into inference engines (vLLM, SGLang)
to capture hidden states, attention weights, KV cache, and logits at each
stage of the inference pipeline. Enables root-cause analysis of PD divergence
by comparing intermediate representations between aggregated and PD modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np


class HookPoint(str, Enum):
    """Points in the inference pipeline where hooks can fire."""

    PREFILL = "prefill"
    KV_TRANSFER = "kv_transfer"
    DECODE_STEP = "decode_step"


@dataclass
class HookCapture:
    """Data captured at a single hook point."""

    hook_point: HookPoint
    layer: int
    step: int | None = None  # decode step index (None for prefill/transfer)
    hidden_states: np.ndarray | None = None
    attention_weights: np.ndarray | None = None
    logits: np.ndarray | None = None
    kv_cache: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (arrays become shape + dtype info, not values)."""
        d: dict[str, Any] = {
            "hook_point": self.hook_point.value,
            "layer": self.layer,
            "step": self.step,
            "metadata": self.metadata,
        }
        for name in ("hidden_states", "attention_weights", "logits", "kv_cache"):
            arr = getattr(self, name)
            if arr is not None:
                d[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            else:
                d[name] = None
        return d


@dataclass
class StageComparison:
    """Comparison result for a single stage between baseline and target."""

    hook_point: HookPoint
    layer: int
    step: int | None = None
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    cosine_similarity: float = 1.0
    field_name: str = "hidden_states"  # which field was compared
    diverged: bool = False
    threshold: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook_point": self.hook_point.value,
            "layer": self.layer,
            "step": self.step,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "cosine_similarity": self.cosine_similarity,
            "field_name": self.field_name,
            "diverged": self.diverged,
            "threshold": self.threshold,
        }


@dataclass
class TraceResult:
    """Full trace result with per-stage comparisons."""

    prompt: str
    baseline_url: str
    target_url: str
    hooks: list[HookPoint]
    comparisons: list[StageComparison] = field(default_factory=list)
    baseline_captures: list[HookCapture] = field(default_factory=list)
    target_captures: list[HookCapture] = field(default_factory=list)
    first_divergence: StageComparison | None = None
    overall_diverged: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "prompt": self.prompt,
            "baseline_url": self.baseline_url,
            "target_url": self.target_url,
            "hooks": [h.value for h in self.hooks],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "baseline_captures": [c.to_dict() for c in self.baseline_captures],
            "target_captures": [c.to_dict() for c in self.target_captures],
            "first_divergence": (
                self.first_divergence.to_dict()
                if self.first_divergence
                else None
            ),
            "overall_diverged": self.overall_diverged,
            "errors": self.errors,
        }
        return d


@runtime_checkable
class InferenceHook(Protocol):
    """Protocol for inference hooks that capture intermediate states."""

    def on_prefill(self, layer: int) -> HookCapture | None:
        """Called after prefill for each layer. Return captured data or None."""
        ...

    def on_kv_transfer(self, layer: int) -> HookCapture | None:
        """Called after KV transfer for each layer. Return captured data or None."""
        ...

    def on_decode_step(self, layer: int, step: int) -> HookCapture | None:
        """Called at each decode step for each layer. Return captured data or None."""
        ...


class MockInferenceHook:
    """Mock hook for testing — generates synthetic intermediate states."""

    def __init__(
        self,
        num_layers: int = 4,
        hidden_dim: int = 64,
        seq_len: int = 8,
        noise_scale: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)
        # Pre-generate base states for consistency
        self._base_hidden = {
            layer: self._rng.standard_normal((seq_len, hidden_dim)).astype(np.float32)
            for layer in range(num_layers)
        }
        self._base_kv = {
            layer: self._rng.standard_normal((2, seq_len, hidden_dim)).astype(
                np.float32
            )
            for layer in range(num_layers)
        }

    def _add_noise(self, arr: np.ndarray) -> np.ndarray:
        if self.noise_scale == 0.0:
            return arr.copy()
        noise = self._rng.standard_normal(arr.shape).astype(arr.dtype) * self.noise_scale
        return arr + noise

    def on_prefill(self, layer: int) -> HookCapture | None:
        if layer >= self.num_layers:
            return None
        hs = self._add_noise(self._base_hidden[layer])
        kv = self._add_noise(self._base_kv[layer])
        return HookCapture(
            hook_point=HookPoint.PREFILL,
            layer=layer,
            hidden_states=hs,
            kv_cache=kv,
            metadata={"num_layers": self.num_layers},
        )

    def on_kv_transfer(self, layer: int) -> HookCapture | None:
        if layer >= self.num_layers:
            return None
        kv = self._add_noise(self._base_kv[layer])
        return HookCapture(
            hook_point=HookPoint.KV_TRANSFER,
            layer=layer,
            kv_cache=kv,
            metadata={"num_layers": self.num_layers},
        )

    def on_decode_step(self, layer: int, step: int) -> HookCapture | None:
        if layer >= self.num_layers:
            return None
        hs = self._add_noise(self._base_hidden[layer])
        logits = self._add_noise(
            self._rng.standard_normal((1, self.hidden_dim)).astype(np.float32)
        )
        return HookCapture(
            hook_point=HookPoint.DECODE_STEP,
            layer=layer,
            step=step,
            hidden_states=hs,
            logits=logits,
            metadata={"num_layers": self.num_layers},
        )


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two arrays (flattened)."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = float(np.dot(a_flat, b_flat))
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compare_captures(
    baseline: HookCapture,
    target: HookCapture,
    threshold: float = 1e-5,
) -> list[StageComparison]:
    """Compare two hook captures field by field."""
    results: list[StageComparison] = []
    for field_name in ("hidden_states", "attention_weights", "logits", "kv_cache"):
        b_arr = getattr(baseline, field_name)
        t_arr = getattr(target, field_name)
        if b_arr is None or t_arr is None:
            continue
        if b_arr.shape != t_arr.shape:
            results.append(
                StageComparison(
                    hook_point=baseline.hook_point,
                    layer=baseline.layer,
                    step=baseline.step,
                    field_name=field_name,
                    max_abs_diff=float("inf"),
                    mean_abs_diff=float("inf"),
                    cosine_similarity=0.0,
                    diverged=True,
                    threshold=threshold,
                )
            )
            continue
        diff = np.abs(b_arr.astype(np.float64) - t_arr.astype(np.float64))
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        cos_sim = _cosine_sim(b_arr, t_arr)
        diverged = max_diff > threshold
        results.append(
            StageComparison(
                hook_point=baseline.hook_point,
                layer=baseline.layer,
                step=baseline.step,
                field_name=field_name,
                max_abs_diff=max_diff,
                mean_abs_diff=mean_diff,
                cosine_similarity=cos_sim,
                diverged=diverged,
                threshold=threshold,
            )
        )
    return results


def run_trace(
    baseline_hook: InferenceHook,
    target_hook: InferenceHook,
    prompt: str,
    baseline_url: str = "",
    target_url: str = "",
    hooks: list[HookPoint] | None = None,
    num_layers: int = 4,
    decode_steps: int = 1,
    threshold: float = 1e-5,
) -> TraceResult:
    """Run a full trace comparing baseline and target hooks.

    This is the synchronous version — real async version would call endpoints.
    """
    if hooks is None:
        hooks = [HookPoint.PREFILL, HookPoint.KV_TRANSFER, HookPoint.DECODE_STEP]

    result = TraceResult(
        prompt=prompt,
        baseline_url=baseline_url,
        target_url=target_url,
        hooks=hooks,
    )

    for hook_point in hooks:
        for layer in range(num_layers):
            b_capture: HookCapture | None = None
            t_capture: HookCapture | None = None

            if hook_point == HookPoint.PREFILL:
                b_capture = baseline_hook.on_prefill(layer)
                t_capture = target_hook.on_prefill(layer)
            elif hook_point == HookPoint.KV_TRANSFER:
                b_capture = baseline_hook.on_kv_transfer(layer)
                t_capture = target_hook.on_kv_transfer(layer)
            elif hook_point == HookPoint.DECODE_STEP:
                for step in range(decode_steps):
                    b_cap = baseline_hook.on_decode_step(layer, step)
                    t_cap = target_hook.on_decode_step(layer, step)
                    if b_cap:
                        result.baseline_captures.append(b_cap)
                    if t_cap:
                        result.target_captures.append(t_cap)
                    if b_cap and t_cap:
                        comps = compare_captures(b_cap, t_cap, threshold)
                        result.comparisons.extend(comps)
                continue  # decode handled in inner loop

            if b_capture:
                result.baseline_captures.append(b_capture)
            if t_capture:
                result.target_captures.append(t_capture)
            if b_capture and t_capture:
                comps = compare_captures(b_capture, t_capture, threshold)
                result.comparisons.extend(comps)

    # Find first divergence
    for comp in result.comparisons:
        if comp.diverged:
            result.first_divergence = comp
            result.overall_diverged = True
            break

    return result


def format_trace(result: TraceResult) -> str:
    """Format a trace result for terminal output."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Inference Trace Report")
    lines.append("=" * 60)
    lines.append(f"Prompt: {result.prompt[:80]}{'...' if len(result.prompt) > 80 else ''}")
    if result.baseline_url:
        lines.append(f"Baseline: {result.baseline_url}")
    if result.target_url:
        lines.append(f"Target:   {result.target_url}")
    lines.append(f"Hooks:    {', '.join(h.value for h in result.hooks)}")
    lines.append(f"Stages compared: {len(result.comparisons)}")
    lines.append("")

    if result.errors:
        lines.append("Errors:")
        for e in result.errors:
            lines.append(f"  ❌ {e}")
        lines.append("")

    verdict = "❌ DIVERGED" if result.overall_diverged else "✅ MATCH"
    lines.append(f"Overall: {verdict}")
    lines.append("")

    if result.first_divergence:
        fd = result.first_divergence
        lines.append("First divergence:")
        lines.append(f"  Stage:    {fd.hook_point.value}")
        lines.append(f"  Layer:    {fd.layer}")
        if fd.step is not None:
            lines.append(f"  Step:     {fd.step}")
        lines.append(f"  Field:    {fd.field_name}")
        lines.append(f"  Max diff: {fd.max_abs_diff:.6e}")
        lines.append(f"  Cos sim:  {fd.cosine_similarity:.6f}")
        lines.append("")

    # Per-stage table
    if result.comparisons:
        lines.append("Stage Details:")
        lines.append(
            f"  {'Stage':<14} {'Layer':>5} {'Step':>5} {'Field':<18} "
            f"{'MaxDiff':>12} {'MeanDiff':>12} {'CosSim':>8} {'Status':>8}"
        )
        lines.append("  " + "-" * 88)
        for c in result.comparisons:
            step_str = str(c.step) if c.step is not None else "-"
            status = "❌ FAIL" if c.diverged else "✅ OK"
            lines.append(
                f"  {c.hook_point.value:<14} {c.layer:>5} {step_str:>5} "
                f"{c.field_name:<18} {c.max_abs_diff:>12.6e} "
                f"{c.mean_abs_diff:>12.6e} {c.cosine_similarity:>8.4f} {status:>8}"
            )

    lines.append("=" * 60)
    return "\n".join(lines)
