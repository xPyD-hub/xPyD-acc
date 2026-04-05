"""Tests for history purge command (M51)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from xpyd_acc.history import HistoryEntry, HistoryStore


def _make_entry(
    store: HistoryStore,
    entry_id: str,
    days_ago: int,
    tag: str = "",
    rate: float = 0.1,
) -> HistoryEntry:
    """Create a history entry with a specific age."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    entry = HistoryEntry(
        entry_id=entry_id,
        timestamp=ts,
        tag=tag,
        report_path="/tmp/report.json",
        divergence_rate=rate,
        sample_count=100,
        dataset="test",
        divergent_count=int(rate * 100),
    )
    store.history_dir.mkdir(parents=True, exist_ok=True)
    (store.history_dir / f"{entry_id}.json").write_text(
        json.dumps(entry.to_dict(), indent=2) + "\n"
    )
    return entry


def test_purge_older_than(tmp_path: Path) -> None:
    store = HistoryStore(history_dir=tmp_path / "hist")
    _make_entry(store, "old1", days_ago=60)
    _make_entry(store, "old2", days_ago=45)
    _make_entry(store, "recent", days_ago=5)

    removed = store.purge(older_than_days=30)
    assert len(removed) == 2
    assert {e.entry_id for e in removed} == {"old1", "old2"}
    # Files should be gone
    assert not (tmp_path / "hist" / "old1.json").exists()
    assert not (tmp_path / "hist" / "old2.json").exists()
    assert (tmp_path / "hist" / "recent.json").exists()


def test_purge_keep_last(tmp_path: Path) -> None:
    store = HistoryStore(history_dir=tmp_path / "hist")
    _make_entry(store, "a", days_ago=90)
    _make_entry(store, "b", days_ago=60)
    _make_entry(store, "c", days_ago=30)
    _make_entry(store, "d", days_ago=5)

    # All are older than 1 day, but keep last 2
    removed = store.purge(older_than_days=1, keep_last=2)
    assert len(removed) == 2
    assert {e.entry_id for e in removed} == {"a", "b"}
    remaining = store.list_entries()
    assert len(remaining) == 2


def test_purge_dry_run(tmp_path: Path) -> None:
    store = HistoryStore(history_dir=tmp_path / "hist")
    _make_entry(store, "old", days_ago=60)
    _make_entry(store, "new", days_ago=5)

    removed = store.purge(older_than_days=30, dry_run=True)
    assert len(removed) == 1
    assert removed[0].entry_id == "old"
    # File should still exist
    assert (tmp_path / "hist" / "old.json").exists()


def test_purge_empty_store(tmp_path: Path) -> None:
    store = HistoryStore(history_dir=tmp_path / "hist")
    removed = store.purge(older_than_days=30)
    assert removed == []


def test_purge_nothing_to_remove(tmp_path: Path) -> None:
    store = HistoryStore(history_dir=tmp_path / "hist")
    _make_entry(store, "recent", days_ago=1)
    removed = store.purge(older_than_days=30)
    assert removed == []


def test_purge_keep_last_protects_all(tmp_path: Path) -> None:
    """When keep_last >= total entries, nothing is removed."""
    store = HistoryStore(history_dir=tmp_path / "hist")
    _make_entry(store, "a", days_ago=90)
    _make_entry(store, "b", days_ago=60)

    removed = store.purge(older_than_days=1, keep_last=5)
    assert removed == []


def test_purge_cli_integration(tmp_path: Path) -> None:
    """Test CLI purge subcommand via subprocess."""
    import subprocess

    store = HistoryStore(history_dir=tmp_path / "hist")
    _make_entry(store, "old1", days_ago=60)
    _make_entry(store, "recent", days_ago=5)

    result = subprocess.run(
        [
            "xpyd-acc", "history", "purge",
            "--older-than", "30",
            "--history-dir", str(tmp_path / "hist"),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "1" in combined  # 1 entry removed
    assert not (tmp_path / "hist" / "old1.json").exists()
    assert (tmp_path / "hist" / "recent.json").exists()


def test_purge_cli_dry_run(tmp_path: Path) -> None:
    """Dry run via CLI doesn't delete files."""
    import subprocess

    store = HistoryStore(history_dir=tmp_path / "hist")
    _make_entry(store, "old1", days_ago=60)

    result = subprocess.run(
        [
            "xpyd-acc", "history", "purge",
            "--older-than", "30", "--dry-run",
            "--history-dir", str(tmp_path / "hist"),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "Would remove" in combined
    assert (tmp_path / "hist" / "old1.json").exists()
