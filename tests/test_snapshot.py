"""Tests for snapshot baseline capture & replay."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from xpyd_acc.batch_compare import DatasetSample
from xpyd_acc.cost import TokenUsage
from xpyd_acc.snapshot import (
    Snapshot,
    SnapshotSample,
    capture_snapshot,
    load_snapshot,
    save_snapshot,
    validate_snapshot_dataset,
)


class TestSnapshotSerialization:
    """Test Snapshot JSON round-trip."""

    def test_to_json_and_from_json(self):
        snap = Snapshot(
            captured_at="2026-04-04T20:00:00+00:00",
            endpoint_url="http://baseline:8000",
            model="test-model",
            samples=[
                SnapshotSample(
                    sample_id="0",
                    prompt="What is 2+2?",
                    output="4",
                    logprobs=[{"token": "4", "logprob": -0.01}],
                ),
                SnapshotSample(
                    sample_id="1",
                    prompt="Hello",
                    output="Hi there!",
                    logprobs=[],
                ),
            ],
        )
        json_str = snap.to_json()
        loaded = Snapshot.from_json(json_str)
        assert loaded.captured_at == snap.captured_at
        assert loaded.endpoint_url == snap.endpoint_url
        assert loaded.model == snap.model
        assert len(loaded.samples) == 2
        assert loaded.samples[0].sample_id == "0"
        assert loaded.samples[0].output == "4"
        assert loaded.samples[1].prompt == "Hello"

    def test_empty_snapshot(self):
        snap = Snapshot(
            captured_at="2026-01-01T00:00:00+00:00",
            endpoint_url="http://x:8000",
            model="m",
        )
        loaded = Snapshot.from_json(snap.to_json())
        assert len(loaded.samples) == 0


class TestSaveLoad:
    """Test file I/O for snapshots."""

    def test_save_and_load(self, tmp_path: Path):
        snap = Snapshot(
            captured_at="2026-04-04T20:00:00+00:00",
            endpoint_url="http://baseline:8000",
            model="test-model",
            samples=[
                SnapshotSample(
                    sample_id="s1",
                    prompt="prompt1",
                    output="output1",
                    logprobs=[{"token": "out", "logprob": -0.5}],
                ),
            ],
        )
        path = tmp_path / "snap.json"
        save_snapshot(snap, path)
        assert path.exists()

        loaded = load_snapshot(path)
        assert loaded.model == "test-model"
        assert len(loaded.samples) == 1
        assert loaded.samples[0].output == "output1"


class TestValidateSnapshotDataset:
    """Test snapshot/dataset validation."""

    def test_matching(self):
        snap = Snapshot(
            captured_at="t", endpoint_url="u", model="m",
            samples=[
                SnapshotSample(sample_id="0", prompt="p0", output="o0"),
                SnapshotSample(sample_id="1", prompt="p1", output="o1"),
            ],
        )
        dataset = [
            DatasetSample(id="0", prompt="p0"),
            DatasetSample(id="1", prompt="p1"),
        ]
        # Should not raise
        validate_snapshot_dataset(snap, dataset)

    def test_missing_from_snapshot(self):
        snap = Snapshot(
            captured_at="t", endpoint_url="u", model="m",
            samples=[SnapshotSample(sample_id="0", prompt="p0", output="o0")],
        )
        dataset = [
            DatasetSample(id="0", prompt="p0"),
            DatasetSample(id="1", prompt="p1"),
        ]
        with pytest.raises(ValueError, match="missing from snapshot"):
            validate_snapshot_dataset(snap, dataset)

    def test_extra_in_snapshot(self):
        snap = Snapshot(
            captured_at="t", endpoint_url="u", model="m",
            samples=[
                SnapshotSample(sample_id="0", prompt="p0", output="o0"),
                SnapshotSample(sample_id="99", prompt="px", output="ox"),
            ],
        )
        dataset = [DatasetSample(id="0", prompt="p0")]
        with pytest.raises(ValueError, match="extra in snapshot"):
            validate_snapshot_dataset(snap, dataset)


class TestCaptureSnapshot:
    """Test capture_snapshot with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_capture(self):
        samples = [
            DatasetSample(id="a", prompt="hello"),
            DatasetSample(id="b", prompt="world"),
        ]

        call_count = 0

        async def mock_collect(url, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return (f"reply-{prompt}", [{"token": "r", "logprob": -0.1}], "", TokenUsage())

        with patch("xpyd_acc.snapshot._collect_output", side_effect=mock_collect):
            snap = await capture_snapshot(
                samples, "http://baseline:8000", model="m",
            )

        assert call_count == 2
        assert len(snap.samples) == 2
        assert snap.endpoint_url == "http://baseline:8000"
        assert snap.model == "m"
        assert snap.samples[0].output == "reply-hello"
        assert snap.samples[1].output == "reply-world"
        assert snap.captured_at  # non-empty

    @pytest.mark.asyncio
    async def test_capture_with_progress(self):
        samples = [DatasetSample(id="x", prompt="test")]
        progress_calls = []

        async def mock_collect(url, prompt, **kwargs):
            return ("out", [], "", TokenUsage())

        with patch("xpyd_acc.snapshot._collect_output", side_effect=mock_collect):
            await capture_snapshot(
                samples, "http://b:8000",
                on_progress=lambda done, total: progress_calls.append((done, total)),
            )

        assert progress_calls == [(1, 1)]
