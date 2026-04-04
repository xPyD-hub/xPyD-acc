"""Tests for result history & trend tracking (M37)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_acc.history import HistoryEntry, HistoryStore


def _make_report(total: int = 100, divergent: int = 10, dataset: str = "gsm8k") -> dict:
    return {
        "total_samples": total,
        "divergent_samples": divergent,
        "dataset": dataset,
    }


@pytest.fixture()
def history_dir(tmp_path: Path) -> Path:
    return tmp_path / "history"


@pytest.fixture()
def store(history_dir: Path) -> HistoryStore:
    return HistoryStore(history_dir=history_dir)


@pytest.fixture()
def report_file(tmp_path: Path) -> Path:
    p = tmp_path / "report.json"
    p.write_text(json.dumps(_make_report()))
    return p


class TestHistoryEntry:
    def test_roundtrip(self) -> None:
        entry = HistoryEntry(
            entry_id="abc123",
            timestamp="2026-01-01T00:00:00+00:00",
            tag="test",
            report_path="/tmp/report.json",
            divergence_rate=0.1,
            sample_count=100,
            dataset="gsm8k",
            divergent_count=10,
        )
        d = entry.to_dict()
        restored = HistoryEntry.from_dict(d)
        assert restored == entry


class TestHistoryStoreSave:
    def test_save_from_file(self, store: HistoryStore, report_file: Path) -> None:
        entry = store.save(report_path=str(report_file), tag="v1")
        assert entry.divergence_rate == pytest.approx(0.1)
        assert entry.sample_count == 100
        assert entry.tag == "v1"
        assert entry.divergent_count == 10
        # File written
        files = list(store.history_dir.glob("*.json"))
        assert len(files) == 1

    def test_save_from_data(self, store: HistoryStore) -> None:
        data = _make_report(50, 5)
        entry = store.save(report_path="fake.json", tag="inline", report_data=data)
        assert entry.divergence_rate == pytest.approx(0.1)
        assert entry.sample_count == 50

    def test_save_zero_samples(self, store: HistoryStore) -> None:
        data = _make_report(0, 0)
        entry = store.save(report_path="empty.json", report_data=data)
        assert entry.divergence_rate == 0.0


class TestHistoryStoreList:
    def test_list_empty(self, store: HistoryStore) -> None:
        assert store.list_entries() == []

    def test_list_sorted(self, store: HistoryStore) -> None:
        store.save("a.json", tag="first", report_data=_make_report(10, 1))
        store.save("b.json", tag="second", report_data=_make_report(10, 2))
        entries = store.list_entries()
        assert len(entries) == 2
        assert entries[0].tag == "first"
        assert entries[1].tag == "second"


class TestHistoryStoreTrend:
    def test_trend_deltas(self, store: HistoryStore) -> None:
        store.save("a.json", tag="run1", report_data=_make_report(100, 10))
        store.save("b.json", tag="run2", report_data=_make_report(100, 5))
        store.save("c.json", tag="run3", report_data=_make_report(100, 15))
        trend = store.trend()
        assert len(trend) == 3
        assert trend[0]["delta"] == 0.0  # first entry, no previous
        assert trend[1]["delta"] == pytest.approx(-0.05)  # 0.05 - 0.10
        assert trend[2]["delta"] == pytest.approx(0.10)  # 0.15 - 0.05

    def test_trend_last_n(self, store: HistoryStore) -> None:
        for i in range(5):
            store.save(f"{i}.json", tag=f"r{i}", report_data=_make_report(100, i * 5))
        trend = store.trend(last_n=2)
        assert len(trend) == 2

    def test_trend_empty(self, store: HistoryStore) -> None:
        assert store.trend() == []


class TestHasRegression:
    def test_regression_detected(self, store: HistoryStore) -> None:
        store.save("a.json", report_data=_make_report(100, 5))
        store.save("b.json", report_data=_make_report(100, 10))
        assert store.has_regression() is True

    def test_no_regression(self, store: HistoryStore) -> None:
        store.save("a.json", report_data=_make_report(100, 10))
        store.save("b.json", report_data=_make_report(100, 5))
        assert store.has_regression() is False

    def test_single_entry_no_regression(self, store: HistoryStore) -> None:
        store.save("a.json", report_data=_make_report(100, 10))
        assert store.has_regression() is False
