"""Tests for M54: Sample Annotation for Batch Reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_acc.annotate import AnnotationStore, SampleAnnotation, annotations_for_markdown


@pytest.fixture()
def report_file(tmp_path: Path) -> Path:
    """Create a minimal batch report JSON."""
    report = {
        "total_samples": 3,
        "divergent_samples": 1,
        "match_samples": 2,
        "divergence_rate": 0.333,
        "results": [
            {"sample_id": "s1", "exact_match": True},
            {"sample_id": "s2", "exact_match": False},
            {"sample_id": "s3", "exact_match": True},
        ],
    }
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report))
    return p


# ── SampleAnnotation ──


def test_annotation_is_empty():
    ann = SampleAnnotation(sample_id="s1")
    assert ann.is_empty()
    ann.note = "hello"
    assert not ann.is_empty()


def test_annotation_label_not_empty():
    ann = SampleAnnotation(sample_id="s1", labels=["known_issue"])
    assert not ann.is_empty()


# ── AnnotationStore basic CRUD ──


def test_set_note():
    store = AnnotationStore()
    store.set_note("s1", "this is a false positive")
    ann = store.get("s1")
    assert ann is not None
    assert ann.note == "this is a false positive"


def test_add_label_no_duplicates():
    store = AnnotationStore()
    store.add_label("s1", "known_issue")
    store.add_label("s1", "known_issue")
    ann = store.get("s1")
    assert ann is not None
    assert ann.labels == ["known_issue"]


def test_clear():
    store = AnnotationStore()
    store.set_note("s1", "note")
    assert store.clear("s1")
    assert store.get("s1") is None
    assert not store.clear("s1")  # second clear returns False


def test_list_annotated_ids():
    store = AnnotationStore()
    store.set_note("s3", "n3")
    store.set_note("s1", "n1")
    assert store.list_annotated_ids() == ["s1", "s3"]


def test_samples_with_label():
    store = AnnotationStore()
    store.add_label("s1", "known_issue")
    store.add_label("s2", "false_positive")
    store.add_label("s3", "known_issue")
    assert store.samples_with_label("known_issue") == ["s1", "s3"]


# ── Persistence (save / load) ──


def test_save_and_load(report_file: Path):
    store = AnnotationStore()
    store.set_note("s1", "looks good")
    store.add_label("s2", "known_issue")
    store.save(report_file)

    sidecar = AnnotationStore.sidecar_path(report_file)
    assert sidecar.exists()

    loaded = AnnotationStore.load(report_file)
    assert loaded.get("s1") is not None
    assert loaded.get("s1").note == "looks good"
    assert loaded.get("s2").labels == ["known_issue"]


def test_load_missing_sidecar(report_file: Path):
    store = AnnotationStore.load(report_file)
    assert store.list_annotated_ids() == []


def test_empty_annotations_not_saved(report_file: Path):
    store = AnnotationStore()
    store.set_note("s1", "note")
    store.clear("s1")
    store.save(report_file)
    data = json.loads(AnnotationStore.sidecar_path(report_file).read_text())
    assert data["annotations"] == {}


# ── annotations_for_markdown helper ──


def test_annotations_for_markdown():
    store = AnnotationStore()
    store.set_note("s1", "false positive")
    store.add_label("s1", "known_issue")
    store.add_label("s2", "fixed")

    result = annotations_for_markdown(store, ["s1", "s2", "s3"])
    assert "known_issue" in result["s1"]
    assert "false positive" in result["s1"]
    assert "fixed" in result["s2"]
    assert "s3" not in result


def test_annotations_for_markdown_empty():
    store = AnnotationStore()
    assert annotations_for_markdown(store, ["s1"]) == {}


# ── CLI integration (annotate subcommand) ──


def test_cli_annotate_note(report_file: Path):
    from xpyd_acc.cli import main

    main(["annotate", "--report", str(report_file), "--sample", "s1", "--note", "test note"])
    store = AnnotationStore.load(report_file)
    assert store.get("s1").note == "test note"


def test_cli_annotate_label(report_file: Path):
    from xpyd_acc.cli import main

    main(["annotate", "--report", str(report_file), "--sample", "s2", "--label", "known_issue"])
    store = AnnotationStore.load(report_file)
    assert "known_issue" in store.get("s2").labels


def test_cli_annotate_list(report_file: Path, capsys):
    from xpyd_acc.cli import main

    store = AnnotationStore()
    store.set_note("s1", "my note")
    store.save(report_file)

    main(["annotate", "--report", str(report_file), "--list"])
    out = capsys.readouterr().out
    assert "s1" in out
    assert "my note" in out


def test_cli_annotate_clear(report_file: Path):
    from xpyd_acc.cli import main

    # Add then clear
    main(["annotate", "--report", str(report_file), "--sample", "s1", "--note", "temp"])
    main(["annotate", "--report", str(report_file), "--sample", "s1", "--clear"])
    store = AnnotationStore.load(report_file)
    assert store.get("s1") is None


def test_cli_annotate_missing_report(tmp_path: Path):
    from xpyd_acc.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["annotate", "--report", str(tmp_path / "nope.json"), "--list"])
    assert exc.value.code == 1


def test_cli_annotate_no_sample_no_list(report_file: Path):
    from xpyd_acc.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["annotate", "--report", str(report_file)])
    assert exc.value.code == 1


def test_cli_annotate_sample_no_action(report_file: Path):
    from xpyd_acc.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["annotate", "--report", str(report_file), "--sample", "s1"])
    assert exc.value.code == 1
