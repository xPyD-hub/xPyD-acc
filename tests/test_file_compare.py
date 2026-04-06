"""Tests for file_compare module (M85: Offline File-Based Comparison)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_acc.file_compare import (
    FileOutput,
    _estimate_context_length,
    format_file_compare,
    load_outputs,
    run_file_compare,
)
from xpyd_acc.output_compare import MatchConfig


def _write_jsonl(path: Path, items: list[dict]) -> None:
    """Helper to write JSONL file."""
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


class TestLoadOutputs:
    """Tests for load_outputs()."""

    def test_basic_load(self, tmp_path: Path) -> None:
        p = tmp_path / "outputs.jsonl"
        _write_jsonl(p, [
            {"id": "s1", "output": "hello world"},
            {"id": "s2", "output": "foo bar", "logprobs": [-0.1, -0.3]},
        ])
        result = load_outputs(p)
        assert len(result) == 2
        assert result[0].id == "s1"
        assert result[0].output == "hello world"
        assert result[0].logprobs is None
        assert result[1].logprobs == [-0.1, -0.3]

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_outputs(tmp_path / "missing.jsonl")

    def test_missing_id(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        _write_jsonl(p, [{"output": "hello"}])
        with pytest.raises(ValueError, match="missing required field 'id'"):
            load_outputs(p)

    def test_missing_output(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        _write_jsonl(p, [{"id": "s1"}])
        with pytest.raises(ValueError, match="missing required field 'output'"):
            load_outputs(p)

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_outputs(p)

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("\n\n")
        with pytest.raises(ValueError, match="No samples found"):
            load_outputs(p)

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "outputs.jsonl"
        p.write_text('\n{"id":"s1","output":"hi"}\n\n')
        result = load_outputs(p)
        assert len(result) == 1


class TestRunFileCompare:
    """Tests for run_file_compare()."""

    def test_all_match(self) -> None:
        bl = [FileOutput("s1", "hello"), FileOutput("s2", "world")]
        tg = [FileOutput("s1", "hello"), FileOutput("s2", "world")]
        report = run_file_compare(bl, tg)
        assert report.total_samples == 2
        assert report.divergent_samples == 0
        assert report.divergence_rate == 0.0

    def test_divergence_detected(self) -> None:
        bl = [FileOutput("s1", "hello world")]
        tg = [FileOutput("s1", "hello earth")]
        report = run_file_compare(bl, tg)
        assert report.divergent_samples == 1
        assert report.results[0].first_divergence_index == 1

    def test_id_mismatch_raises(self) -> None:
        bl = [FileOutput("s1", "hello")]
        tg = [FileOutput("s2", "hello")]
        with pytest.raises(ValueError, match="Sample ID mismatch"):
            run_file_compare(bl, tg)

    def test_with_logprobs_likely_bug(self) -> None:
        bl = [FileOutput("s1", "a b c", logprobs=[-0.01, -0.5, -0.01])]
        tg = [FileOutput("s1", "a x c", logprobs=[-0.01, -0.01, -0.01])]
        report = run_file_compare(bl, tg, logprob_gap_threshold=0.1)
        assert report.results[0].classification == "likely_bug"
        assert report.results[0].logprob_gap is not None
        assert report.results[0].logprob_gap == pytest.approx(0.49)

    def test_with_logprobs_likely_uncertainty(self) -> None:
        bl = [FileOutput("s1", "a b c", logprobs=[-0.01, -0.05, -0.01])]
        tg = [FileOutput("s1", "a x c", logprobs=[-0.01, -0.04, -0.01])]
        report = run_file_compare(bl, tg, logprob_gap_threshold=0.1)
        assert report.results[0].classification == "likely_uncertainty"

    def test_no_logprobs_unknown(self) -> None:
        bl = [FileOutput("s1", "a b")]
        tg = [FileOutput("s1", "a x")]
        report = run_file_compare(bl, tg)
        assert report.results[0].classification == "unknown"

    def test_match_config_ignore_case(self) -> None:
        bl = [FileOutput("s1", "Hello World")]
        tg = [FileOutput("s1", "hello world")]
        cfg = MatchConfig(ignore_case=True)
        report = run_file_compare(bl, tg, match_config=cfg)
        assert report.divergent_samples == 0

    def test_prefix_divergence(self) -> None:
        bl = [FileOutput("s1", "a b c")]
        tg = [FileOutput("s1", "a b")]
        report = run_file_compare(bl, tg)
        assert report.results[0].first_divergence_index == 2


class TestFormatFileCompare:
    """Tests for format_file_compare()."""

    def test_format_output(self) -> None:
        bl = [FileOutput("s1", "hello"), FileOutput("s2", "world")]
        tg = [FileOutput("s1", "hello"), FileOutput("s2", "earth")]
        report = run_file_compare(bl, tg)
        text = format_file_compare(report)
        assert "File Comparison Report" in text
        assert "Divergent:         1" in text
        assert "50.0%" in text


class TestEstimateContextLength:
    """Tests for _estimate_context_length()."""

    def test_basic(self) -> None:
        assert _estimate_context_length("hello world foo bar") >= 1

    def test_empty(self) -> None:
        assert _estimate_context_length("") >= 1


class TestCLIIntegration:
    """Tests for compare-files CLI subcommand."""

    def test_cli_basic(self, tmp_path: Path) -> None:
        bl_path = tmp_path / "baseline.jsonl"
        tg_path = tmp_path / "target.jsonl"
        _write_jsonl(bl_path, [{"id": "s1", "output": "hello"}])
        _write_jsonl(tg_path, [{"id": "s1", "output": "hello"}])

        import argparse

        from xpyd_acc.cli.parsers import register_all
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        register_all(sub)
        args = parser.parse_args([
            "compare-files", "--baseline", str(bl_path), "--target", str(tg_path),
        ])
        assert args.command == "compare-files"
        assert args.baseline == str(bl_path)

    def test_cli_json_export(self, tmp_path: Path) -> None:
        bl_path = tmp_path / "baseline.jsonl"
        tg_path = tmp_path / "target.jsonl"
        json_out = tmp_path / "report.json"
        _write_jsonl(bl_path, [{"id": "s1", "output": "hello"}])
        _write_jsonl(tg_path, [{"id": "s1", "output": "hello"}])

        import argparse

        from xpyd_acc.cli.parsers import register_all
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        register_all(sub)
        args = parser.parse_args([
            "compare-files", "--baseline", str(bl_path),
            "--target", str(tg_path), "--json", str(json_out),
        ])
        assert args.json == str(json_out)
