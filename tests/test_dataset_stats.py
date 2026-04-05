"""Tests for dataset_stats module."""

from __future__ import annotations

import json
from pathlib import Path

from xpyd_acc.batch_compare import DatasetSample
from xpyd_acc.dataset_stats import (
    DatasetStatsReport,
    LengthStats,
    _compute_length_stats,
    compute_stats,
    estimate_tokens,
    print_stats,
)


def _make_sample(prompt: str, sid: str = "") -> DatasetSample:
    return DatasetSample(id=sid or "s1", prompt=prompt)


class TestEstimateTokens:
    def test_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_single_word(self) -> None:
        assert estimate_tokens("hello") == 2  # ceil(1/0.75)

    def test_multiple_words(self) -> None:
        result = estimate_tokens("the quick brown fox jumps")
        assert result == 7  # ceil(5/0.75)


class TestComputeLengthStats:
    def test_empty(self) -> None:
        stats = _compute_length_stats([])
        assert stats == LengthStats()

    def test_single(self) -> None:
        stats = _compute_length_stats([42])
        assert stats.min == 42
        assert stats.max == 42
        assert stats.mean == 42.0
        assert stats.median == 42.0

    def test_distribution(self) -> None:
        lengths = list(range(1, 101))  # 1..100
        stats = _compute_length_stats(lengths)
        assert stats.min == 1
        assert stats.max == 100
        assert stats.mean == 50.5
        assert stats.p95 == 95.0


class TestComputeStats:
    def test_empty(self) -> None:
        report = compute_stats([])
        assert report.sample_count == 0

    def test_basic(self) -> None:
        samples = [
            _make_sample("short"),
            _make_sample("a longer prompt here"),
            _make_sample("short"),  # duplicate
        ]
        report = compute_stats(samples)
        assert report.sample_count == 3
        assert report.unique_prompts == 2
        assert report.duplicate_count == 1
        assert len(report.duplicates) == 1
        assert report.duplicates[0]["count"] == 2
        assert report.char_stats.min == 5
        assert report.char_stats.max == 20

    def test_no_duplicates(self) -> None:
        samples = [_make_sample("a"), _make_sample("b"), _make_sample("c")]
        report = compute_stats(samples)
        assert report.duplicate_count == 0
        assert report.duplicates == []

    def test_with_template(self) -> None:
        from xpyd_acc.templates import PromptTemplate

        template = PromptTemplate(
            name="test", template="Question: {question}\nAnswer:"
        )
        samples = [
            DatasetSample(id="s1", prompt="ignored", metadata={"question": "Why?"}),
            DatasetSample(id="s2", prompt="ignored", metadata={"question": "How?"}),
        ]
        report = compute_stats(samples, template)
        assert report.sample_count == 2
        # Template renders "Question: Why?\nAnswer:" which is > len("ignored")
        assert report.char_stats.min > 10


class TestJsonExport:
    def test_to_json(self, tmp_path: Path) -> None:
        samples = [_make_sample("hello world"), _make_sample("foo bar")]
        report = compute_stats(samples)
        out = tmp_path / "stats.json"
        report.to_json(out)
        data = json.loads(out.read_text())
        assert data["sample_count"] == 2
        assert "char_stats" in data
        assert "token_stats" in data
        assert data["unique_prompts"] == 2

    def test_roundtrip_dict(self) -> None:
        report = DatasetStatsReport(
            sample_count=5,
            char_stats=LengthStats(min=1, max=100, mean=50.0, median=45.0, p95=90.0),
            token_stats=LengthStats(min=1, max=50, mean=25.0, median=22.0, p95=45.0),
            duplicate_count=2,
            unique_prompts=3,
            duplicates=[{"prompt": "hi", "count": 3}],
        )
        d = report.to_dict()
        assert d["sample_count"] == 5
        assert d["char_stats"]["min"] == 1


class TestPrintStats:
    def test_print_no_crash(self) -> None:
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, force_terminal=False)
        samples = [_make_sample("hello"), _make_sample("hello"), _make_sample("world")]
        report = compute_stats(samples)
        print_stats(report, console)
        output = buf.getvalue()
        assert "3 samples" in output
        assert "Duplicate" in output


class TestCLI:
    def test_dataset_stats_jsonl(self, tmp_path: Path) -> None:
        ds = tmp_path / "data.jsonl"
        ds.write_text(
            '{"prompt": "hello"}\n{"prompt": "world"}\n{"prompt": "hello"}\n'
        )
        out = tmp_path / "out.json"
        from xpyd_acc.cli import main

        main(["dataset-stats", str(ds), "--json", str(out)])
        data = json.loads(out.read_text())
        assert data["sample_count"] == 3
        assert data["duplicate_count"] == 1

    def test_dataset_stats_csv(self, tmp_path: Path) -> None:
        ds = tmp_path / "data.csv"
        ds.write_text("prompt\nhello\nworld\n")
        from xpyd_acc.cli import main

        main(["dataset-stats", str(ds)])  # just no crash
