"""Automatic KV cache export from vLLM — capture KV tensors at configurable points."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class CapturePoint(str, Enum):
    """Points in the inference pipeline where KV cache can be captured."""

    AFTER_PREFILL = "after_prefill"
    AFTER_TRANSFER = "after_transfer"
    DURING_DECODE = "during_decode"


@dataclass
class CaptureConfig:
    """Configuration for a KV cache capture session."""

    url: str
    prompt: str
    output_path: str
    layers: list[int] | None = None  # None = all layers
    capture_points: list[CapturePoint] = field(
        default_factory=lambda: [CapturePoint.AFTER_PREFILL]
    )
    tp_size: int = 1  # tensor parallel size for shard reconstruction
    max_tokens: int = 1  # generate at least 1 token to trigger decode

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        errors: list[str] = []
        if not self.url:
            errors.append("url is required")
        if not self.prompt:
            errors.append("prompt is required")
        if not self.output_path:
            errors.append("output_path is required")
        if self.layers is not None:
            for layer in self.layers:
                if layer < 0:
                    errors.append(f"invalid layer index: {layer}")
        if self.tp_size < 1:
            errors.append(f"tp_size must be >= 1, got {self.tp_size}")
        if not self.capture_points:
            errors.append("at least one capture_point is required")
        return errors


@dataclass
class LayerCapture:
    """Captured KV cache data for a single layer at a single capture point."""

    layer_index: int
    capture_point: CapturePoint
    key: np.ndarray  # shape: (num_heads, seq_len, head_dim)
    value: np.ndarray


@dataclass
class CaptureResult:
    """Result of a KV cache capture session."""

    config: CaptureConfig
    layers: list[LayerCapture] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and len(self.layers) > 0

    def to_dict(self) -> dict:
        """Serialize metadata (not tensors) to dict."""
        return {
            "success": self.success,
            "num_layers": len(self.layers),
            "capture_points": list({lc.capture_point.value for lc in self.layers}),
            "layer_indices": sorted({lc.layer_index for lc in self.layers}),
            "metadata": self.metadata,
            "errors": self.errors,
        }


def reconstruct_tp_shards(shards: list[np.ndarray], axis: int = 0) -> np.ndarray:
    """Reconstruct a full tensor from TP shards by concatenating along head axis.

    Args:
        shards: list of shard arrays, one per TP rank
        axis: concatenation axis (default 0 = num_heads dimension)

    Returns:
        Reconstructed full tensor.

    Raises:
        ValueError: if shards list is empty or shapes are incompatible.
    """
    if not shards:
        raise ValueError("shards list is empty")
    if len(shards) == 1:
        return shards[0]
    # Validate compatible shapes on non-concat axes
    ref_shape = list(shards[0].shape)
    for i, shard in enumerate(shards[1:], 1):
        s = list(shard.shape)
        if len(s) != len(ref_shape):
            raise ValueError(
                f"shard {i} has {len(s)} dims, expected {len(ref_shape)}"
            )
        for d in range(len(ref_shape)):
            if d != axis and s[d] != ref_shape[d]:
                raise ValueError(
                    f"shard {i} shape mismatch on axis {d}: "
                    f"{s[d]} vs {ref_shape[d]}"
                )
    return np.concatenate(shards, axis=axis)


def save_capture(result: CaptureResult, output_path: str | Path) -> Path:
    """Save captured KV cache to .npz file with metadata.

    File contains:
      - layer_{i}_{point}_key: key tensor for layer i at capture point
      - layer_{i}_{point}_value: value tensor for layer i at capture point
      - metadata: JSON string with capture config and result info

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {}
    for lc in result.layers:
        prefix = f"layer_{lc.layer_index}_{lc.capture_point.value}"
        arrays[f"{prefix}_key"] = lc.key
        arrays[f"{prefix}_value"] = lc.value

    arrays["metadata"] = np.array(json.dumps(result.to_dict()))
    np.savez(str(output_path), **arrays)

    # np.savez adds .npz if not present
    actual_path = output_path if output_path.suffix == ".npz" else Path(str(output_path) + ".npz")
    return actual_path


def load_capture(path: str | Path) -> dict[str, np.ndarray]:
    """Load a previously saved capture .npz file.

    Returns:
        Dict of array name -> ndarray (including metadata as string array).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Capture file not found: {path}")
    data = np.load(str(path), allow_pickle=False)
    return dict(data)


def filter_layers(
    layers: list[LayerCapture], selected: list[int] | None
) -> list[LayerCapture]:
    """Filter captures to only include selected layer indices."""
    if selected is None:
        return layers
    selected_set = set(selected)
    return [lc for lc in layers if lc.layer_index in selected_set]


def capture_kv_mock(config: CaptureConfig) -> CaptureResult:
    """Mock capture for testing — generates random KV cache data.

    This simulates what the real vLLM hook would produce.
    Real implementation requires vLLM monkey-patching (see docs).
    """
    errors = config.validate()
    if errors:
        return CaptureResult(config=config, errors=errors)

    num_layers = 32  # typical model
    num_heads = 32
    seq_len = len(config.prompt.split())  # rough approximation
    head_dim = 128

    target_layers = config.layers if config.layers is not None else list(range(num_layers))

    captures: list[LayerCapture] = []
    for point in config.capture_points:
        for layer_idx in target_layers:
            if layer_idx >= num_layers:
                continue
            k = np.random.randn(num_heads, seq_len, head_dim).astype(np.float16)
            v = np.random.randn(num_heads, seq_len, head_dim).astype(np.float16)
            captures.append(LayerCapture(
                layer_index=layer_idx,
                capture_point=point,
                key=k,
                value=v,
            ))

    return CaptureResult(
        config=config,
        layers=captures,
        metadata={
            "model_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "mode": "mock",
        },
    )
