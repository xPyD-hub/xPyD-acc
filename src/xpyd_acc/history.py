"""Result history & trend tracking for batch comparison reports."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class HistoryEntry:
    """A single saved batch report summary."""

    entry_id: str
    timestamp: str
    tag: str
    report_path: str
    divergence_rate: float
    sample_count: int
    dataset: str
    divergent_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryEntry:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _default_history_dir() -> Path:
    return Path.home() / ".xpyd-acc" / "history"


class HistoryStore:
    """Manages saved batch report history entries."""

    def __init__(self, history_dir: Path | None = None) -> None:
        self.history_dir = history_dir or _default_history_dir()

    def save(
        self,
        report_path: str,
        tag: str = "",
        report_data: dict[str, Any] | None = None,
    ) -> HistoryEntry:
        """Save a batch report to history.

        If report_data is not provided, reads from report_path.
        """
        if report_data is None:
            with open(report_path) as f:
                report_data = json.load(f)

        total = report_data.get("total_samples", 0)
        divergent = report_data.get("divergent_samples", 0)
        rate = divergent / total if total > 0 else 0.0
        dataset = report_data.get("dataset", "unknown")

        entry = HistoryEntry(
            entry_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            tag=tag,
            report_path=str(report_path),
            divergence_rate=round(rate, 6),
            sample_count=total,
            dataset=dataset,
            divergent_count=divergent,
        )

        self.history_dir.mkdir(parents=True, exist_ok=True)
        entry_file = self.history_dir / f"{entry.entry_id}.json"
        entry_file.write_text(json.dumps(entry.to_dict(), indent=2) + "\n")
        return entry

    def list_entries(self) -> list[HistoryEntry]:
        """List all history entries, sorted by timestamp ascending."""
        if not self.history_dir.exists():
            return []
        entries: list[HistoryEntry] = []
        for p in sorted(self.history_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                entries.append(HistoryEntry.from_dict(data))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        entries.sort(key=lambda e: e.timestamp)
        return entries

    def trend(self, last_n: int | None = None) -> list[dict[str, Any]]:
        """Compute divergence rate trend with deltas.

        Returns list of dicts with: timestamp, tag, divergence_rate, delta.
        """
        entries = self.list_entries()
        if last_n is not None and last_n > 0:
            entries = entries[-last_n:]

        result: list[dict[str, Any]] = []
        prev_rate: float | None = None
        for entry in entries:
            delta = (
                round(entry.divergence_rate - prev_rate, 6)
                if prev_rate is not None
                else 0.0
            )
            result.append({
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp,
                "tag": entry.tag,
                "divergence_rate": entry.divergence_rate,
                "delta": delta,
                "sample_count": entry.sample_count,
            })
            prev_rate = entry.divergence_rate
        return result

    def purge(
        self,
        older_than_days: int | None = None,
        keep_last: int = 0,
        dry_run: bool = False,
    ) -> list[HistoryEntry]:
        """Remove old history entries.

        Args:
            older_than_days: Remove entries older than this many days.
            keep_last: Always keep at least this many most recent entries.
            dry_run: If True, return entries that would be removed without deleting.

        Returns:
            List of entries that were (or would be) removed.
        """
        entries = self.list_entries()
        if not entries:
            return []

        # Determine which entries to protect (keep_last most recent)
        protected: set[str] = set()
        if keep_last > 0:
            for entry in entries[-keep_last:]:
                protected.add(entry.entry_id)

        to_remove: list[HistoryEntry] = []
        now = datetime.now(timezone.utc)

        for entry in entries:
            if entry.entry_id in protected:
                continue
            if older_than_days is not None:
                entry_time = datetime.fromisoformat(entry.timestamp)
                age_days = (now - entry_time).total_seconds() / 86400
                if age_days > older_than_days:
                    to_remove.append(entry)

        if not dry_run:
            for entry in to_remove:
                entry_file = self.history_dir / f"{entry.entry_id}.json"
                if entry_file.exists():
                    entry_file.unlink()

        return to_remove

    def has_regression(self, last_n: int | None = None) -> bool:
        """Check if the most recent entry shows increased divergence vs previous."""
        trend_data = self.trend(last_n=last_n)
        if len(trend_data) < 2:
            return False
        return trend_data[-1]["delta"] > 0
