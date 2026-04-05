"""Tests for report diff functionality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_acc.diff import DiffResult, SampleTransition, diff_reports, format_diff_report


def _write_report(path: Path, samples: list[dict]) -> None:
    """Helper to write a batch report JSON."""
    path.write_text(json.dumps({"results": samples}))


def _sample(sid: str, prompt: str, output: str, match: bool) -> dict:
    return {
        "sample_id": sid,
        "prompt": prompt,
        "target_output": output,
        "exact_match": match,
    }


class TestDiffReports:
    """Tests for diff_reports()."""

    def test_identical_reports(self, tmp_path: Path) -> None:
        samples = [_sample("s1", "hello", "world", True)]
        _write_report(tmp_path / "old.json", samples)
        _write_report(tmp_path / "new.json", samples)

        result = diff_reports(tmp_path / "old.json", tmp_path / "new.json")
        assert result.regressions == 0
        assert result.fixes == 0
        assert result.unchanged_match == 1
        assert result.output_changes == 0
        assert result.new_samples == 0
        assert result.removed_samples == 0

    def test_regression(self, tmp_path: Path) -> None:
        _write_report(tmp_path / "old.json", [_sample("s1", "hi", "ok", True)])
        _write_report(tmp_path / "new.json", [_sample("s1", "hi", "bad", False)])

        result = diff_reports(tmp_path / "old.json", tmp_path / "new.json")
        assert result.regressions == 1
        assert result.fixes == 0
        assert result.output_changes == 1
        t = result.transitions[0]
        assert t.status == "regression"
        assert t.output_diff is not None

    def test_fix(self, tmp_path: Path) -> None:
        _write_report(tmp_path / "old.json", [_sample("s1", "hi", "bad", False)])
        _write_report(tmp_path / "new.json", [_sample("s1", "hi", "ok", True)])

        result = diff_reports(tmp_path / "old.json", tmp_path / "new.json")
        assert result.regressions == 0
        assert result.fixes == 1

    def test_unchanged_diverge(self, tmp_path: Path) -> None:
        _write_report(tmp_path / "old.json", [_sample("s1", "hi", "bad1", False)])
        _write_report(tmp_path / "new.json", [_sample("s1", "hi", "bad2", False)])

        result = diff_reports(tmp_path / "old.json", tmp_path / "new.json")
        assert result.unchanged_diverge == 1
        assert result.output_changes == 1

    def test_new_samples(self, tmp_path: Path) -> None:
        _write_report(tmp_path / "old.json", [_sample("s1", "hi", "ok", True)])
        _write_report(tmp_path / "new.json", [
            _sample("s1", "hi", "ok", True),
            _sample("s2", "hey", "yo", True),
        ])

        result = diff_reports(tmp_path / "old.json", tmp_path / "new.json")
        assert result.new_samples == 1
        assert result.common == 1
        new = [t for t in result.transitions if t.status == "new"]
        assert len(new) == 1
        assert new[0].sample_id == "s2"

    def test_removed_samples(self, tmp_path: Path) -> None:
        _write_report(tmp_path / "old.json", [
            _sample("s1", "hi", "ok", True),
            _sample("s2", "hey", "yo", True),
        ])
        _write_report(tmp_path / "new.json", [_sample("s1", "hi", "ok", True)])

        result = diff_reports(tmp_path / "old.json", tmp_path / "new.json")
        assert result.removed_samples == 1
        removed = [t for t in result.transitions if t.status == "removed"]
        assert len(removed) == 1
        assert removed[0].sample_id == "s2"

    def test_empty_reports(self, tmp_path: Path) -> None:
        _write_report(tmp_path / "old.json", [])
        _write_report(tmp_path / "new.json", [])

        result = diff_reports(tmp_path / "old.json", tmp_path / "new.json")
        assert result.total_old == 0
        assert result.total_new == 0
        assert result.common == 0

    def test_file_not_found(self, tmp_path: Path) -> None:
        _write_report(tmp_path / "old.json", [])
        with pytest.raises(FileNotFoundError):
            diff_reports(tmp_path / "old.json", tmp_path / "missing.json")

    def test_mixed_transitions(self, tmp_path: Path) -> None:
        _write_report(tmp_path / "old.json", [
            _sample("s1", "a", "ok", True),
            _sample("s2", "b", "bad", False),
            _sample("s3", "c", "ok", True),
        ])
        _write_report(tmp_path / "new.json", [
            _sample("s1", "a", "wrong", False),  # regression
            _sample("s2", "b", "ok", True),       # fix
            _sample("s3", "c", "ok", True),       # unchanged match
            _sample("s4", "d", "new", True),       # new
        ])

        result = diff_reports(tmp_path / "old.json", tmp_path / "new.json")
        assert result.regressions == 1
        assert result.fixes == 1
        assert result.unchanged_match == 1
        assert result.new_samples == 1

    def test_json_export(self, tmp_path: Path) -> None:
        _write_report(tmp_path / "old.json", [_sample("s1", "hi", "ok", True)])
        _write_report(tmp_path / "new.json", [_sample("s1", "hi", "bad", False)])

        result = diff_reports(tmp_path / "old.json", tmp_path / "new.json")
        data = json.loads(result.to_json())
        assert data["regressions"] == 1
        assert len(data["transitions"]) == 1

    def test_raw_list_format(self, tmp_path: Path) -> None:
        """Support raw list format (not wrapped in {results: []})."""
        (tmp_path / "old.json").write_text(json.dumps([
            _sample("s1", "hi", "ok", True),
        ]))
        (tmp_path / "new.json").write_text(json.dumps([
            _sample("s1", "hi", "ok", True),
        ]))
        result = diff_reports(tmp_path / "old.json", tmp_path / "new.json")
        assert result.unchanged_match == 1

    def test_output_same_but_match_changed(self, tmp_path: Path) -> None:
        """Output text is the same but match status flipped (edge case)."""
        _write_report(tmp_path / "old.json", [_sample("s1", "hi", "x", True)])
        _write_report(tmp_path / "new.json", [_sample("s1", "hi", "x", False)])

        result = diff_reports(tmp_path / "old.json", tmp_path / "new.json")
        assert result.regressions == 1
        assert result.output_changes == 0  # output didn't change, just match status


class TestFormatDiffReport:
    """Tests for format_diff_report()."""

    def test_basic_format(self) -> None:
        result = DiffResult(
            total_old=3, total_new=3, common=3,
            regressions=1, fixes=1, unchanged_match=1,
            unchanged_diverge=0, new_samples=0, removed_samples=0,
            output_changes=2,
            transitions=[
                SampleTransition(
                    "s1", "regression", True, False, "ok", "bad",
                    "--- old\n+++ new\n", "prompt1",
                ),
                SampleTransition("s2", "fix", False, True, "bad", "ok", None, "prompt2"),
            ],
        )
        text = format_diff_report(result)
        assert "Regressions:" in text
        assert "Fixes:" in text
        assert "❌" in text
        assert "✅" in text

    def test_no_regressions(self) -> None:
        result = DiffResult(
            total_old=1, total_new=1, common=1,
            regressions=0, fixes=0, unchanged_match=1,
            unchanged_diverge=0, new_samples=0, removed_samples=0,
            output_changes=0, transitions=[],
        )
        text = format_diff_report(result)
        assert "Regressions:       0" in text
