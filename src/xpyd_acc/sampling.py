"""Sampling parameter utilities for reproducible LLM comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SamplingParams:
    """Sampling parameters for LLM requests."""

    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return non-None params as a dict suitable for API payloads."""
        d: dict[str, Any] = {}
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.top_p is not None:
            d["top_p"] = self.top_p
        if self.seed is not None:
            d["seed"] = self.seed
        return d

    @classmethod
    def from_args(cls, args: Any) -> SamplingParams:
        """Create from an argparse Namespace (attributes may be absent)."""
        return cls(
            temperature=getattr(args, "temperature", None),
            top_p=getattr(args, "top_p", None),
            seed=getattr(args, "seed", None),
        )
