"""Sample annotation for batch reports — sidecar-based notes and labels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SampleAnnotation:
    """Annotation for a single sample."""

    sample_id: str
    note: str | None = None
    labels: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return True if annotation has no content."""
        return not self.note and not self.labels


@dataclass
class AnnotationStore:
    """Manages annotations stored in a sidecar JSON file."""

    annotations: dict[str, SampleAnnotation] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def sidecar_path(report_path: str | Path) -> Path:
        """Return the sidecar file path for a given report."""
        p = Path(report_path)
        return p.parent / f"{p.name}.annotations.json"

    @classmethod
    def load(cls, report_path: str | Path) -> "AnnotationStore":
        """Load annotations from the sidecar file.  Returns empty store if missing."""
        path = cls.sidecar_path(report_path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        store = cls()
        for sid, entry in data.get("annotations", {}).items():
            store.annotations[sid] = SampleAnnotation(
                sample_id=sid,
                note=entry.get("note"),
                labels=entry.get("labels", []),
            )
        return store

    def save(self, report_path: str | Path) -> Path:
        """Persist annotations to the sidecar file.  Returns the path written."""
        path = self.sidecar_path(report_path)
        # Remove empty annotations before saving
        clean = {
            sid: asdict(ann)
            for sid, ann in self.annotations.items()
            if not ann.is_empty()
        }
        data: dict[str, Any] = {"annotations": clean}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def set_note(self, sample_id: str, note: str) -> None:
        """Set or update the note for a sample."""
        ann = self.annotations.setdefault(sample_id, SampleAnnotation(sample_id=sample_id))
        ann.note = note

    def add_label(self, sample_id: str, label: str) -> None:
        """Add a label to a sample (no duplicates)."""
        ann = self.annotations.setdefault(sample_id, SampleAnnotation(sample_id=sample_id))
        if label not in ann.labels:
            ann.labels.append(label)

    def clear(self, sample_id: str) -> bool:
        """Remove all annotations for a sample.  Returns True if anything was removed."""
        return self.annotations.pop(sample_id, None) is not None

    def get(self, sample_id: str) -> SampleAnnotation | None:
        """Return annotation for a sample, or None."""
        return self.annotations.get(sample_id)

    def list_annotated_ids(self) -> list[str]:
        """Return sorted list of sample IDs that have annotations."""
        return sorted(sid for sid, ann in self.annotations.items() if not ann.is_empty())

    def samples_with_label(self, label: str) -> list[str]:
        """Return sorted list of sample IDs that have a specific label."""
        return sorted(
            sid for sid, ann in self.annotations.items() if label in ann.labels
        )


# ------------------------------------------------------------------
# Helpers for report integration
# ------------------------------------------------------------------


def annotations_for_markdown(
    store: AnnotationStore,
    sample_ids: list[str],
) -> dict[str, str]:
    """Return a dict mapping sample_id -> human-readable annotation string."""
    result: dict[str, str] = {}
    for sid in sample_ids:
        ann = store.get(sid)
        if ann is None or ann.is_empty():
            continue
        parts: list[str] = []
        if ann.labels:
            parts.append(f"[{', '.join(ann.labels)}]")
        if ann.note:
            parts.append(ann.note)
        result[sid] = " ".join(parts)
    return result
