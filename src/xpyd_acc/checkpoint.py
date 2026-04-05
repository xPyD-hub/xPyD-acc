"""Checkpoint support for resumable batch comparison runs.

Saves completed sample results to a checkpoint file during batch comparison,
allowing interrupted runs to be resumed without re-processing completed samples.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpyd_acc.log import get_logger

logger = get_logger("checkpoint")


@dataclass
class Checkpoint:
    """In-progress batch comparison checkpoint."""

    # Checkpoint metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # IDs of completed samples
    completed_ids: set[str] = field(default_factory=set)
    # Serialised SampleResult dicts keyed by sample_id
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Run parameters for validation on resume
    baseline_url: str = ""
    target_url: str = ""
    model: str = ""
    total_samples: int = 0

    def add_result(self, sample_id: str, result_dict: dict[str, Any]) -> None:
        """Record a completed sample result."""
        self.completed_ids.add(sample_id)
        self.results[sample_id] = result_dict
        self.updated_at = time.time()

    @property
    def completed_count(self) -> int:
        return len(self.completed_ids)

    def is_completed(self, sample_id: str) -> bool:
        return sample_id in self.completed_ids

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_ids": sorted(self.completed_ids),
            "results": self.results,
            "baseline_url": self.baseline_url,
            "target_url": self.target_url,
            "model": self.model,
            "total_samples": self.total_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Deserialise from a dict."""
        return cls(
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            completed_ids=set(data.get("completed_ids", [])),
            results=data.get("results", {}),
            baseline_url=data.get("baseline_url", ""),
            target_url=data.get("target_url", ""),
            model=data.get("model", ""),
            total_samples=data.get("total_samples", 0),
        )


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> None:
    """Write checkpoint to disk atomically (write-then-rename)."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(checkpoint.to_dict(), indent=2))
    tmp.rename(path)
    logger.debug(
        "Checkpoint saved: %d/%d samples",
        checkpoint.completed_count, checkpoint.total_samples,
    )


def load_checkpoint(path: str | Path) -> Checkpoint | None:
    """Load checkpoint from disk. Returns None if file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        cp = Checkpoint.from_dict(data)
        logger.info(
            "Loaded checkpoint: %d/%d samples completed",
            cp.completed_count, cp.total_samples,
        )
        return cp
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Corrupt checkpoint file %s: %s", path, exc)
        return None


def validate_checkpoint(
    checkpoint: Checkpoint,
    baseline_url: str,
    target_url: str,
    model: str,
    total_samples: int,
) -> bool:
    """Check that a loaded checkpoint matches current run parameters.

    Returns True if compatible, False if the checkpoint should be discarded.
    """
    if checkpoint.baseline_url != baseline_url:
        logger.warning(
            "Checkpoint baseline_url mismatch: %s vs %s",
            checkpoint.baseline_url, baseline_url,
        )
        return False
    if checkpoint.target_url != target_url:
        logger.warning(
            "Checkpoint target_url mismatch: %s vs %s",
            checkpoint.target_url, target_url,
        )
        return False
    if checkpoint.model != model:
        logger.warning(
            "Checkpoint model mismatch: %s vs %s",
            checkpoint.model, model,
        )
        return False
    if checkpoint.total_samples != total_samples:
        logger.warning(
            "Checkpoint total_samples mismatch: %d vs %d",
            checkpoint.total_samples, total_samples,
        )
        return False
    return True


def result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a SampleResult to a JSON-serialisable dict."""
    return {
        "sample_id": result.sample_id,
        "prompt": result.prompt,
        "baseline_output": result.baseline_output,
        "target_output": result.target_output,
        "exact_match": result.exact_match,
        "first_divergence_index": result.first_divergence_index,
        "baseline_logprob_at_divergence": result.baseline_logprob_at_divergence,
        "target_logprob_at_divergence": result.target_logprob_at_divergence,
        "logprob_gap": result.logprob_gap,
        "classification": result.classification,
        "context_length": result.context_length,
        "request_ids": result.request_ids,
        "baseline_finish_reason": result.baseline_finish_reason,
        "target_finish_reason": result.target_finish_reason,
    }


def dict_to_result(data: dict[str, Any]) -> Any:
    """Convert a dict back to a SampleResult."""
    from xpyd_acc.batch_compare import SampleResult

    return SampleResult(
        sample_id=data["sample_id"],
        prompt=data["prompt"],
        baseline_output=data["baseline_output"],
        target_output=data["target_output"],
        exact_match=data["exact_match"],
        first_divergence_index=data.get("first_divergence_index"),
        baseline_logprob_at_divergence=data.get("baseline_logprob_at_divergence"),
        target_logprob_at_divergence=data.get("target_logprob_at_divergence"),
        logprob_gap=data.get("logprob_gap"),
        classification=data.get("classification", "unknown"),
        context_length=data.get("context_length", 0),
        request_ids=data.get("request_ids", {}),
        baseline_finish_reason=data.get("baseline_finish_reason"),
        target_finish_reason=data.get("target_finish_reason"),
    )
