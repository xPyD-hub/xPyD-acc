"""Tests for prompt deduplication in batch comparison."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from xpyd_acc.batch_compare import DatasetSample, run_batch
from xpyd_acc.cost import TokenUsage


def test_deduplicate_reduces_api_calls() -> None:
    """With deduplication, duplicate prompts should only trigger one API call each."""
    samples = [
        DatasetSample(id="1", prompt="What is 2+2?"),
        DatasetSample(id="2", prompt="What is 2+2?"),  # duplicate
        DatasetSample(id="3", prompt="What is 3+3?"),
        DatasetSample(id="4", prompt="What is 2+2?"),  # duplicate
    ]
    mock_output = ("hello world", [], "", TokenUsage())

    with patch(
        "xpyd_acc.batch_compare._collect_output",
        new_callable=AsyncMock,
    ) as mock_co:
        mock_co.return_value = mock_output
        report = asyncio.run(run_batch(
            samples,
            "http://baseline",
            "http://target",
            deduplicate=True,
            enable_request_ids=False,
        ))

    # 2 unique prompts × 2 endpoints = 4 calls (not 4 × 2 = 8)
    assert mock_co.call_count == 4
    assert report.total_samples == 4
    assert len(report.results) == 4


def test_no_deduplicate_sends_all_calls() -> None:
    """Without deduplication, every sample sends its own API calls."""
    samples = [
        DatasetSample(id="1", prompt="What is 2+2?"),
        DatasetSample(id="2", prompt="What is 2+2?"),
        DatasetSample(id="3", prompt="What is 3+3?"),
    ]
    mock_output = ("hello world", [], "", TokenUsage())

    with patch(
        "xpyd_acc.batch_compare._collect_output",
        new_callable=AsyncMock,
    ) as mock_co:
        mock_co.return_value = mock_output
        report = asyncio.run(run_batch(
            samples,
            "http://baseline",
            "http://target",
            deduplicate=False,
            enable_request_ids=False,
        ))

    # 3 samples × 2 endpoints = 6
    assert mock_co.call_count == 6
    assert report.total_samples == 3


def test_deduplicate_no_duplicates() -> None:
    """Dedup with all unique prompts works same as no-dedup."""
    samples = [
        DatasetSample(id="1", prompt="What is 2+2?"),
        DatasetSample(id="2", prompt="What is 3+3?"),
        DatasetSample(id="3", prompt="What is 4+4?"),
    ]
    mock_output = ("hello world", [], "", TokenUsage())

    with patch(
        "xpyd_acc.batch_compare._collect_output",
        new_callable=AsyncMock,
    ) as mock_co:
        mock_co.return_value = mock_output
        report = asyncio.run(run_batch(
            samples,
            "http://baseline",
            "http://target",
            deduplicate=True,
            enable_request_ids=False,
        ))

    # 3 unique × 2 endpoints = 6
    assert mock_co.call_count == 6
    assert report.total_samples == 3


def test_deduplicate_preserves_sample_ids() -> None:
    """Each result has correct sample_id even with dedup."""
    samples = [
        DatasetSample(id="a", prompt="prompt1"),
        DatasetSample(id="b", prompt="prompt1"),
        DatasetSample(id="c", prompt="prompt2"),
    ]
    mock_output = ("hello world", [], "", TokenUsage())

    with patch(
        "xpyd_acc.batch_compare._collect_output",
        new_callable=AsyncMock,
    ) as mock_co:
        mock_co.return_value = mock_output
        report = asyncio.run(run_batch(
            samples,
            "http://baseline",
            "http://target",
            deduplicate=True,
            enable_request_ids=False,
        ))

    result_ids = [r.sample_id for r in report.results]
    assert result_ids == ["a", "b", "c"]


def test_deduplicate_progress_callback() -> None:
    """Progress callback fires once per unique prompt, not per sample."""
    samples = [
        DatasetSample(id="1", prompt="p1"),
        DatasetSample(id="2", prompt="p1"),
        DatasetSample(id="3", prompt="p2"),
    ]
    mock_output = ("hello", [], "", TokenUsage())
    progress_calls: list[tuple[int, int]] = []

    def on_progress(done: int, total: int) -> None:
        progress_calls.append((done, total))

    with patch(
        "xpyd_acc.batch_compare._collect_output",
        new_callable=AsyncMock,
    ) as mock_co:
        mock_co.return_value = mock_output
        asyncio.run(run_batch(
            samples,
            "http://baseline",
            "http://target",
            deduplicate=True,
            enable_request_ids=False,
            on_progress=on_progress,
        ))

    # 2 unique prompts → 2 progress calls
    assert len(progress_calls) == 2
    assert progress_calls[-1] == (2, 2)


def test_deduplicate_config_default_false() -> None:
    """BatchConfig.deduplicate defaults to False."""
    from xpyd_acc.config import BatchConfig

    cfg = BatchConfig()
    assert cfg.deduplicate is False


def test_deduplicate_config_from_toml() -> None:
    """BatchConfig picks up deduplicate from TOML."""
    from xpyd_acc.config import BatchConfig, _parse_section

    raw = {"deduplicate": True, "concurrency": 10}
    cfg = _parse_section(raw, BatchConfig)
    assert cfg.deduplicate is True
    assert cfg.concurrency == 10
