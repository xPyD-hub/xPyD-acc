"""Tests for checkpoint integration into run_batch()."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from xpyd_acc.batch_compare import DatasetSample, SampleResult, run_batch
from xpyd_acc.checkpoint import (
    Checkpoint,
    result_to_dict,
    save_checkpoint,
)
from xpyd_acc.cost import TokenUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _samples(n: int = 3) -> list[DatasetSample]:
    return [DatasetSample(id=f"s{i}", prompt=f"prompt {i}") for i in range(n)]


def _fake_collect_output(text: str = "hello"):
    """Return an AsyncMock that simulates _collect_output."""
    async def _mock(*args, **kwargs):
        return text, [], "", TokenUsage(0, 0), "stop"  # text, logprobs, rid, usage, finish
    return _mock


def _make_sample_result(sid: str, prompt: str = "p", match: bool = True) -> SampleResult:
    return SampleResult(
        sample_id=sid,
        prompt=prompt,
        baseline_output="hello",
        target_output="hello" if match else "world",
        exact_match=match,
        first_divergence_index=None if match else 0,
        baseline_logprob_at_divergence=None,
        target_logprob_at_divergence=None,
        logprob_gap=None,
        classification="match" if match else "likely_bug",
        context_length=1,
        request_ids={},
        baseline_finish_reason="stop",
        target_finish_reason="stop",
    )


def _save_checkpoint_with_results(
    path: Path,
    samples: list[DatasetSample],
    completed_ids: list[str],
    baseline_url: str = "http://base",
    target_url: str = "http://target",
    model: str = "default",
):
    """Save a checkpoint with pre-populated results."""
    cp = Checkpoint(
        baseline_url=baseline_url,
        target_url=target_url,
        model=model,
        total_samples=len(samples),
    )
    for sid in completed_ids:
        sr = _make_sample_result(sid, prompt=f"prompt for {sid}")
        cp.add_result(sid, result_to_dict(sr))
    save_checkpoint(cp, path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckpointIntegration:
    """Test checkpoint wiring into run_batch()."""

    @patch("xpyd_acc.batch_compare._collect_output")
    def test_no_checkpoint_works_normally(self, mock_collect):
        """Without checkpoint_path, run_batch works as before."""
        mock_collect.side_effect = _fake_collect_output()
        samples = _samples(2)
        report = asyncio.run(run_batch(
            samples, "http://base", "http://target",
            checkpoint_path=None,
        ))
        assert report.total_samples == 2

    @patch("xpyd_acc.batch_compare._collect_output")
    def test_checkpoint_created_and_cleaned_up(self, mock_collect, tmp_path):
        """Checkpoint file is created during run and removed on success."""
        mock_collect.side_effect = _fake_collect_output()
        cp_path = tmp_path / "cp.json"
        samples = _samples(2)
        report = asyncio.run(run_batch(
            samples, "http://base", "http://target",
            checkpoint_path=str(cp_path),
        ))
        assert report.total_samples == 2
        # Checkpoint should be cleaned up after successful run
        assert not cp_path.exists()

    @patch("xpyd_acc.batch_compare._collect_output")
    def test_resume_skips_completed_samples(self, mock_collect, tmp_path):
        """Resumed samples are not re-sent to API."""
        mock_collect.side_effect = _fake_collect_output()
        cp_path = tmp_path / "cp.json"
        samples = _samples(3)

        # Pre-save checkpoint with s0 completed
        _save_checkpoint_with_results(
            cp_path, samples, ["s0"],
            baseline_url="http://base",
            target_url="http://target",
        )

        report = asyncio.run(run_batch(
            samples, "http://base", "http://target",
            checkpoint_path=str(cp_path),
        ))
        assert report.total_samples == 3
        # _collect_output should be called 2*2=4 times (2 pending samples, baseline+target each)
        assert mock_collect.call_count == 4

    @patch("xpyd_acc.batch_compare._collect_output")
    def test_mismatched_checkpoint_discarded(self, mock_collect, tmp_path):
        """Checkpoint with different params is discarded."""
        mock_collect.side_effect = _fake_collect_output()
        cp_path = tmp_path / "cp.json"
        samples = _samples(2)

        # Save checkpoint with different model
        _save_checkpoint_with_results(
            cp_path, samples, ["s0"],
            baseline_url="http://base",
            target_url="http://target",
            model="different-model",
        )

        asyncio.run(run_batch(
            samples, "http://base", "http://target",
            model="default",
            checkpoint_path=str(cp_path),
        ))
        # All samples should be processed (checkpoint discarded)
        assert mock_collect.call_count == 4  # 2 samples * 2 endpoints

    @patch("xpyd_acc.batch_compare._collect_output")
    def test_checkpoint_clear_deletes_existing(self, mock_collect, tmp_path):
        """--checkpoint-clear deletes existing checkpoint before starting."""
        mock_collect.side_effect = _fake_collect_output()
        cp_path = tmp_path / "cp.json"
        samples = _samples(2)

        # Save a checkpoint
        _save_checkpoint_with_results(
            cp_path, samples, ["s0"],
            baseline_url="http://base",
            target_url="http://target",
        )
        assert cp_path.exists()

        asyncio.run(run_batch(
            samples, "http://base", "http://target",
            checkpoint_path=str(cp_path),
            checkpoint_clear=True,
        ))
        # All samples processed (checkpoint was cleared)
        assert mock_collect.call_count == 4

    @patch("xpyd_acc.batch_compare._collect_output")
    def test_results_in_original_order(self, mock_collect, tmp_path):
        """Results are returned in original sample order even with resume."""
        call_count = 0

        async def _mock(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return "output", [], "", TokenUsage(0, 0), "stop"

        mock_collect.side_effect = _mock
        cp_path = tmp_path / "cp.json"
        samples = _samples(3)

        # Pre-complete s1 (middle sample)
        _save_checkpoint_with_results(
            cp_path, samples, ["s1"],
            baseline_url="http://base",
            target_url="http://target",
        )

        report = asyncio.run(run_batch(
            samples, "http://base", "http://target",
            checkpoint_path=str(cp_path),
        ))
        result_ids = [r.sample_id for r in report.results]
        assert result_ids == ["s0", "s1", "s2"]

    @patch("xpyd_acc.batch_compare._collect_output")
    def test_all_resumed_no_api_calls(self, mock_collect, tmp_path):
        """If all samples are in checkpoint, no API calls are made."""
        mock_collect.side_effect = _fake_collect_output()
        cp_path = tmp_path / "cp.json"
        samples = _samples(2)

        _save_checkpoint_with_results(
            cp_path, samples, ["s0", "s1"],
            baseline_url="http://base",
            target_url="http://target",
        )

        report = asyncio.run(run_batch(
            samples, "http://base", "http://target",
            checkpoint_path=str(cp_path),
        ))
        assert report.total_samples == 2
        assert mock_collect.call_count == 0

    @patch("xpyd_acc.batch_compare._collect_output")
    def test_nonexistent_checkpoint_starts_fresh(self, mock_collect, tmp_path):
        """Non-existent checkpoint file starts a fresh run."""
        mock_collect.side_effect = _fake_collect_output()
        cp_path = tmp_path / "nonexistent.json"
        samples = _samples(2)

        report = asyncio.run(run_batch(
            samples, "http://base", "http://target",
            checkpoint_path=str(cp_path),
        ))
        assert report.total_samples == 2
        assert mock_collect.call_count == 4

    @patch("xpyd_acc.batch_compare._collect_output")
    def test_progress_callback_includes_resumed(self, mock_collect, tmp_path):
        """Progress callback counts resumed samples in total."""
        mock_collect.side_effect = _fake_collect_output()
        cp_path = tmp_path / "cp.json"
        samples = _samples(3)

        _save_checkpoint_with_results(
            cp_path, samples, ["s0"],
            baseline_url="http://base",
            target_url="http://target",
        )

        progress_calls = []

        def on_progress(done, total):
            progress_calls.append((done, total))

        asyncio.run(run_batch(
            samples, "http://base", "http://target",
            checkpoint_path=str(cp_path),
            on_progress=on_progress,
        ))
        # Should have 2 progress calls (for 2 pending samples)
        assert len(progress_calls) == 2
        # Total should always be 3
        assert all(t == 3 for _, t in progress_calls)

    def test_cli_checkpoint_flags_in_help(self):
        """CLI parser accepts --checkpoint and --checkpoint-clear flags."""
        from xpyd_acc.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["batch-compare", "--help"])
        assert exc_info.value.code == 0

    @patch("xpyd_acc.batch_compare._collect_output")
    def test_checkpoint_clear_with_no_existing_file(self, mock_collect, tmp_path):
        """--checkpoint-clear with no existing file doesn't error."""
        mock_collect.side_effect = _fake_collect_output()
        cp_path = tmp_path / "nope.json"
        samples = _samples(1)

        report = asyncio.run(run_batch(
            samples, "http://base", "http://target",
            checkpoint_path=str(cp_path),
            checkpoint_clear=True,
        ))
        assert report.total_samples == 1

    @patch("xpyd_acc.batch_compare._collect_output")
    def test_corrupt_checkpoint_starts_fresh(self, mock_collect, tmp_path):
        """Corrupt checkpoint file is ignored and fresh run starts."""
        mock_collect.side_effect = _fake_collect_output()
        cp_path = tmp_path / "bad.json"
        cp_path.write_text("not valid json {{{")
        samples = _samples(2)

        report = asyncio.run(run_batch(
            samples, "http://base", "http://target",
            checkpoint_path=str(cp_path),
        ))
        assert report.total_samples == 2
        assert mock_collect.call_count == 4
