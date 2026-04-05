"""Tests for concurrency sweep module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xpyd_acc.concurrency_sweep import (
    SweepLevelResult,
    SweepResult,
    format_sweep,
    run_sweep,
)

# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


def test_sweep_level_result_fields():
    lv = SweepLevelResult(
        concurrency=4, total_samples=100,
        divergent_samples=5, divergence_rate=0.05,
        elapsed_seconds=1.2,
    )
    assert lv.concurrency == 4
    assert lv.divergence_rate == 0.05
    assert lv.elapsed_seconds == 1.2


def test_sweep_result_defaults():
    sr = SweepResult()
    assert sr.levels == []
    assert sr.any_divergence is False
    assert sr.model is None


def test_sweep_result_to_dict():
    sr = SweepResult(
        levels=[SweepLevelResult(1, 10, 0, 0.0), SweepLevelResult(2, 10, 3, 0.3)],
        baseline_url="http://a",
        target_url="http://b",
        model="m1",
        any_divergence=True,
    )
    d = sr.to_dict()
    assert len(d["levels"]) == 2
    assert d["any_divergence"] is True
    assert d["model"] == "m1"


def test_sweep_result_to_json():
    sr = SweepResult(levels=[SweepLevelResult(1, 5, 0, 0.0)])
    j = sr.to_json()
    parsed = json.loads(j)
    assert parsed["levels"][0]["concurrency"] == 1


def test_sweep_result_from_dict():
    d = {
        "levels": [
            {
                "concurrency": 1, "total_samples": 10,
                "divergent_samples": 0, "divergence_rate": 0.0,
                "elapsed_seconds": None,
            },
        ],
        "baseline_url": "http://a",
        "target_url": "http://b",
        "model": "m",
        "any_divergence": False,
    }
    sr = SweepResult.from_dict(d)
    assert len(sr.levels) == 1
    assert sr.levels[0].concurrency == 1
    assert sr.model == "m"


def test_sweep_result_round_trip():
    sr = SweepResult(
        levels=[SweepLevelResult(2, 20, 1, 0.05, 3.5)],
        dataset_path="data.jsonl",
        any_divergence=True,
    )
    restored = SweepResult.from_dict(json.loads(sr.to_json()))
    assert restored.levels[0].concurrency == 2
    assert restored.any_divergence is True
    assert restored.dataset_path == "data.jsonl"


# ---------------------------------------------------------------------------
# Format tests
# ---------------------------------------------------------------------------


def test_format_sweep_no_divergence():
    sr = SweepResult(levels=[SweepLevelResult(1, 10, 0, 0.0)])
    text = format_sweep(sr)
    assert "ALL MATCH" in text
    assert "1" in text


def test_format_sweep_with_divergence():
    sr = SweepResult(
        levels=[
            SweepLevelResult(1, 10, 0, 0.0, 1.0),
            SweepLevelResult(4, 10, 3, 0.3, 2.5),
        ],
        any_divergence=True,
    )
    text = format_sweep(sr)
    assert "DIVERGENCE DETECTED" in text
    assert "30.00%" in text
    assert "2.5s" in text


def test_format_sweep_no_elapsed():
    sr = SweepResult(levels=[SweepLevelResult(1, 5, 0, 0.0, elapsed_seconds=None)])
    text = format_sweep(sr)
    assert "s)" not in text


# ---------------------------------------------------------------------------
# run_sweep tests (mocked)
# ---------------------------------------------------------------------------


def _make_mock_report(divergent: int = 0, total: int = 10):
    """Create a mock BatchReport-like object."""
    m = MagicMock()
    m.total_samples = total
    m.divergent_samples = divergent
    m.divergence_rate = divergent / total if total else 0.0
    return m


@pytest.mark.asyncio
async def test_run_sweep_all_match():
    with patch("xpyd_acc.batch_compare.run_batch", new_callable=AsyncMock) as mock_rb:
        mock_rb.return_value = _make_mock_report(0, 10)
        result = await run_sweep(
            baseline_url="http://a",
            target_url="http://b",
            dataset_path="d.jsonl",
            levels=[1, 2],
        )
    assert len(result.levels) == 2
    assert result.any_divergence is False
    assert mock_rb.call_count == 2


@pytest.mark.asyncio
async def test_run_sweep_some_divergence():
    reports = [_make_mock_report(0, 10), _make_mock_report(2, 10)]
    with patch("xpyd_acc.batch_compare.run_batch", new_callable=AsyncMock, side_effect=reports):
        result = await run_sweep(
            baseline_url="http://a",
            target_url="http://b",
            dataset_path="d.jsonl",
            levels=[1, 4],
        )
    assert result.any_divergence is True
    assert result.levels[1].divergent_samples == 2


@pytest.mark.asyncio
async def test_run_sweep_levels_sorted():
    with patch("xpyd_acc.batch_compare.run_batch", new_callable=AsyncMock) as mock_rb:
        mock_rb.return_value = _make_mock_report(0, 5)
        result = await run_sweep(
            baseline_url="http://a",
            target_url="http://b",
            dataset_path="d.jsonl",
            levels=[8, 2, 1],
        )
    assert [lv.concurrency for lv in result.levels] == [1, 2, 8]


@pytest.mark.asyncio
async def test_run_sweep_single_level():
    with patch("xpyd_acc.batch_compare.run_batch", new_callable=AsyncMock) as mock_rb:
        mock_rb.return_value = _make_mock_report(1, 5)
        result = await run_sweep(
            baseline_url="http://a",
            target_url="http://b",
            dataset_path="d.jsonl",
            levels=[4],
        )
    assert len(result.levels) == 1
    assert result.any_divergence is True


@pytest.mark.asyncio
async def test_run_sweep_callback():
    called: list[SweepLevelResult] = []
    with patch("xpyd_acc.batch_compare.run_batch", new_callable=AsyncMock) as mock_rb:
        mock_rb.return_value = _make_mock_report(0, 10)
        await run_sweep(
            baseline_url="http://a",
            target_url="http://b",
            dataset_path="d.jsonl",
            levels=[1, 2],
            on_level_complete=called.append,
        )
    assert len(called) == 2
    assert called[0].concurrency == 1


@pytest.mark.asyncio
async def test_run_sweep_passes_concurrency():
    with patch("xpyd_acc.batch_compare.run_batch", new_callable=AsyncMock) as mock_rb:
        mock_rb.return_value = _make_mock_report(0, 5)
        await run_sweep(
            baseline_url="http://a",
            target_url="http://b",
            dataset_path="d.jsonl",
            levels=[1, 4],
            model="test-model",
            max_tokens=100,
        )
    # Check concurrency was passed correctly
    calls = mock_rb.call_args_list
    assert calls[0].kwargs["concurrency"] == 1
    assert calls[1].kwargs["concurrency"] == 4
    assert calls[0].kwargs["model"] == "test-model"


@pytest.mark.asyncio
async def test_run_sweep_elapsed_recorded():
    with patch("xpyd_acc.batch_compare.run_batch", new_callable=AsyncMock) as mock_rb:
        mock_rb.return_value = _make_mock_report(0, 5)
        result = await run_sweep(
            baseline_url="http://a",
            target_url="http://b",
            dataset_path="d.jsonl",
            levels=[1],
        )
    assert result.levels[0].elapsed_seconds is not None
    assert result.levels[0].elapsed_seconds >= 0


@pytest.mark.asyncio
async def test_run_sweep_json_export(tmp_path: Path):
    with patch("xpyd_acc.batch_compare.run_batch", new_callable=AsyncMock) as mock_rb:
        mock_rb.return_value = _make_mock_report(1, 10)
        result = await run_sweep(
            baseline_url="http://a",
            target_url="http://b",
            dataset_path="d.jsonl",
            levels=[2],
        )
    out = tmp_path / "sweep.json"
    out.write_text(result.to_json())
    loaded = json.loads(out.read_text())
    assert loaded["levels"][0]["concurrency"] == 2
    assert loaded["any_divergence"] is True


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def test_cli_concurrency_sweep_help(capsys):
    """Verify the concurrency-sweep subcommand is registered."""
    from xpyd_acc.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["concurrency-sweep", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "concurrency-sweep" in captured.out or "levels" in captured.out
