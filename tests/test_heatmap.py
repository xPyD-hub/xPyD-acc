"""Tests for heatmap module — divergence heatmap by token position."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from xpyd_acc.heatmap import (
    HeatmapBucket,
    HeatmapReport,
    compute_heatmap,
    format_heatmap,
)


@dataclass
class _FakeResult:
    match: bool = True
    first_divergence_index: int | None = None
    logprob_gap: float | None = None


class TestHeatmapBucket:
    def test_label(self):
        b = HeatmapBucket(0, 10, 3, 3, 0.05)
        assert b.label == "0-9"

    def test_to_dict(self):
        b = HeatmapBucket(10, 20, 5, 5, 0.123456789)
        d = b.to_dict()
        assert d["range_start"] == 10
        assert d["range_end"] == 20
        assert d["divergence_count"] == 5
        assert d["avg_logprob_gap"] == 0.123457


class TestComputeHeatmap:
    def test_no_divergent(self):
        results = [_FakeResult(match=True) for _ in range(5)]
        report = compute_heatmap(results)
        assert report.total_divergent == 0
        assert report.buckets == []
        assert report.max_divergence_index is None

    def test_single_divergent(self):
        results = [_FakeResult(match=False, first_divergence_index=5, logprob_gap=0.1)]
        report = compute_heatmap(results, num_buckets=5)
        assert report.total_divergent == 1
        assert report.max_divergence_index == 5
        assert len(report.buckets) >= 1
        total = sum(b.divergence_count for b in report.buckets)
        assert total == 1

    def test_multiple_buckets(self):
        results = [
            _FakeResult(match=False, first_divergence_index=0, logprob_gap=0.5),
            _FakeResult(match=False, first_divergence_index=1, logprob_gap=0.3),
            _FakeResult(match=False, first_divergence_index=50, logprob_gap=0.1),
            _FakeResult(match=False, first_divergence_index=99, logprob_gap=0.2),
            _FakeResult(match=True),
        ]
        report = compute_heatmap(results, num_buckets=10)
        assert report.total_divergent == 4
        assert report.max_divergence_index == 99
        total = sum(b.divergence_count for b in report.buckets)
        assert total == 4

    def test_buckets_non_overlapping(self):
        results = [
            _FakeResult(match=False, first_divergence_index=i, logprob_gap=0.1)
            for i in range(20)
        ]
        report = compute_heatmap(results, num_buckets=4)
        # Check no gaps between buckets
        for i in range(1, len(report.buckets)):
            assert report.buckets[i].range_start == report.buckets[i - 1].range_end

    def test_avg_logprob_gap(self):
        results = [
            _FakeResult(match=False, first_divergence_index=0, logprob_gap=0.2),
            _FakeResult(match=False, first_divergence_index=1, logprob_gap=0.4),
        ]
        report = compute_heatmap(results, num_buckets=1)
        assert len(report.buckets) == 1
        assert report.buckets[0].avg_logprob_gap == pytest.approx(0.3)

    def test_none_logprob_gap(self):
        results = [
            _FakeResult(match=False, first_divergence_index=0, logprob_gap=None),
        ]
        report = compute_heatmap(results, num_buckets=1)
        assert report.buckets[0].avg_logprob_gap is None

    def test_divergent_without_index_skipped(self):
        results = [
            _FakeResult(match=False, first_divergence_index=None),
            _FakeResult(match=False, first_divergence_index=5, logprob_gap=0.1),
        ]
        report = compute_heatmap(results, num_buckets=5)
        assert report.total_divergent == 1

    def test_num_buckets_zero_defaults_to_one(self):
        results = [_FakeResult(match=False, first_divergence_index=10, logprob_gap=0.1)]
        report = compute_heatmap(results, num_buckets=0)
        assert len(report.buckets) >= 1


class TestHeatmapReport:
    def test_to_dict_roundtrip(self):
        report = HeatmapReport(
            buckets=[HeatmapBucket(0, 10, 3, 3, 0.05)],
            total_divergent=3,
            max_divergence_index=9,
            num_buckets=1,
        )
        d = report.to_dict()
        assert d["total_divergent"] == 3
        assert len(d["buckets"]) == 1

    def test_to_json_file(self):
        report = HeatmapReport(
            buckets=[HeatmapBucket(0, 5, 2, 2, 0.1)],
            total_divergent=2,
            max_divergence_index=4,
            num_buckets=1,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        report.to_json(path)
        data = json.loads(Path(path).read_text())
        assert data["total_divergent"] == 2
        assert len(data["buckets"]) == 1
        Path(path).unlink()


class TestFormatHeatmap:
    def test_empty_report(self):
        report = HeatmapReport([], 0, None, 10)
        out = format_heatmap(report)
        assert "No divergent" in out

    def test_non_empty_report(self):
        report = HeatmapReport(
            buckets=[
                HeatmapBucket(0, 10, 5, 5, 0.3),
                HeatmapBucket(10, 20, 2, 2, 0.1),
            ],
            total_divergent=7,
            max_divergence_index=15,
            num_buckets=2,
        )
        out = format_heatmap(report)
        assert "7 divergent" in out
        assert "0-9" in out
        assert "10-19" in out
        assert "█" in out
