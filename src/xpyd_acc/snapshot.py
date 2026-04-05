"""Snapshot baseline capture & replay for batch comparisons."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from xpyd_acc.batch_compare import DatasetSample, _collect_output
from xpyd_acc.log import get_logger

logger = get_logger("snapshot")


@dataclass
class SnapshotSample:
    """Captured baseline output for a single sample."""

    sample_id: str
    prompt: str
    output: str
    logprobs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Snapshot:
    """Captured baseline outputs for an entire dataset."""

    captured_at: str
    endpoint_url: str
    model: str
    samples: list[SnapshotSample] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize snapshot to JSON."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> Snapshot:
        """Deserialize snapshot from JSON string."""
        data = json.loads(text)
        samples = [SnapshotSample(**s) for s in data.get("samples", [])]
        return cls(
            captured_at=data["captured_at"],
            endpoint_url=data["endpoint_url"],
            model=data["model"],
            samples=samples,
        )


async def capture_snapshot(
    samples: list[DatasetSample],
    baseline_url: str,
    *,
    model: str = "default",
    max_tokens: int = 64,
    api_key: str = "no-key",
    concurrency: int = 5,
    retries: int = 3,
    retry_delay: float = 1.0,
    sampling_params: Any | None = None,
    timeout: float = 120.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> Snapshot:
    """Capture baseline outputs for all samples.

    Sends each sample prompt to the baseline endpoint and stores the output
    and logprobs for later replay.
    """
    logger.info(
        "Capturing snapshot: %d samples from %s", len(samples), baseline_url,
    )
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(samples)

    async def _capture_one(sample: DatasetSample) -> SnapshotSample:
        nonlocal completed
        async with semaphore:
            text, logprobs, _rid, _usage = await _collect_output(
                baseline_url, sample.prompt, model=model,
                max_tokens=max_tokens, api_key=api_key,
                retries=retries, retry_delay=retry_delay,
                sampling_params=sampling_params, timeout=timeout,
            )
        completed += 1
        if on_progress is not None:
            on_progress(completed, total)
        return SnapshotSample(
            sample_id=sample.id,
            prompt=sample.prompt,
            output=text,
            logprobs=logprobs,
        )

    tasks = [_capture_one(s) for s in samples]
    snap_samples = await asyncio.gather(*tasks)

    return Snapshot(
        captured_at=datetime.now(timezone.utc).isoformat(),
        endpoint_url=baseline_url,
        model=model,
        samples=list(snap_samples),
    )


def save_snapshot(snapshot: Snapshot, path: str | Path) -> None:
    """Write snapshot to a JSON file."""
    path = Path(path)
    path.write_text(snapshot.to_json())
    logger.info("Snapshot saved to %s (%d samples)", path, len(snapshot.samples))


def load_snapshot(path: str | Path) -> Snapshot:
    """Load snapshot from a JSON file."""
    path = Path(path)
    text = path.read_text()
    snapshot = Snapshot.from_json(text)
    logger.info(
        "Snapshot loaded from %s (%d samples, captured %s)",
        path, len(snapshot.samples), snapshot.captured_at,
    )
    return snapshot


def validate_snapshot_dataset(
    snapshot: Snapshot, samples: list[DatasetSample],
) -> None:
    """Validate that snapshot sample IDs match the dataset.

    Raises ValueError if there is a mismatch.
    """
    snap_ids = {s.sample_id for s in snapshot.samples}
    dataset_ids = {s.id for s in samples}

    missing = dataset_ids - snap_ids
    extra = snap_ids - dataset_ids

    if missing or extra:
        parts = []
        if missing:
            examples = sorted(missing)[:5]
            parts.append(f"missing from snapshot: {examples}")
        if extra:
            examples = sorted(extra)[:5]
            parts.append(f"extra in snapshot: {examples}")
        msg = f"Snapshot/dataset mismatch: {'; '.join(parts)}"
        raise ValueError(msg)
