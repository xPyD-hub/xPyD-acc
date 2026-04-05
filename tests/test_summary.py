"""Tests for M48: Compact Summary Command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_acc.summary import SummaryData, extract_summary, load_and_summarize


@pytest.fixture()
def basic_report() -> dict:
    return {
        "dataset": "gsm8k",
        "total_samples": 100,
        "divergent_samples": 12,
        "divergence_rate": 0.12,
        "divergence_index_mean": 5.3,
        "divergence_index_median": 4.0,
        "results": [],
    }


@pytest.fixture()
def basic_report_file(tmp_path: Path, basic_report: dict) -> Path:
    p = tmp_path / "report.json"
    p.write_text(json.dumps(basic_report), encoding="utf-8")
    return p


class TestSummaryData:
    def test_to_oneline(self) -> None:
        s = SummaryData("gsm8k", 100, 12, 0.12, 5.3, 4.0, None)
        line = s.to_oneline()
        assert "gsm8k" in line
        assert "100 samples" in line
        assert "12 divergent (12.0%)" in line
        assert "mean div index: 5.3" in line

    def test_to_oneline_no_divergence(self) -> None:
        s = SummaryData("mmlu", 50, 0, 0.0, None, None, None)
        line = s.to_oneline()
        assert "0 divergent (0.0%)" in line
        assert "mean div index" not in line

    def test_to_json(self) -> None:
        s = SummaryData("gsm8k", 100, 12, 0.12, 5.3, 4.0, None)
        parsed = json.loads(s.to_json())
        assert parsed["dataset"] == "gsm8k"
        assert parsed["total_samples"] == 100
        assert parsed["divergent_samples"] == 12
        assert parsed["divergence_rate"] == 0.12
        assert parsed["divergence_index_mean"] == 5.3
        assert "targets" not in parsed

    def test_to_json_with_targets(self) -> None:
        s = SummaryData("gsm8k", 100, 12, 0.12, None, None, ["http://a", "http://b"])
        parsed = json.loads(s.to_json())
        assert parsed["targets"] == ["http://a", "http://b"]

    def test_to_kv(self) -> None:
        s = SummaryData("gsm8k", 100, 12, 0.12, 5.3, 4.0, None)
        kv = s.to_kv()
        assert "dataset=gsm8k" in kv
        assert "total_samples=100" in kv
        assert "divergent_samples=12" in kv
        assert "divergence_rate=0.1200" in kv
        assert "divergence_index_mean=5.3" in kv

    def test_to_kv_with_targets(self) -> None:
        s = SummaryData("x", 10, 1, 0.1, None, None, ["a", "b"])
        kv = s.to_kv()
        assert "targets=a,b" in kv
        assert "divergence_index_mean" not in kv

    def test_format_oneline(self) -> None:
        s = SummaryData("test", 10, 1, 0.1, None, None, None)
        assert s.format("oneline") == s.to_oneline()

    def test_format_json(self) -> None:
        s = SummaryData("test", 10, 1, 0.1, None, None, None)
        assert s.format("json") == s.to_json()

    def test_format_kv(self) -> None:
        s = SummaryData("test", 10, 1, 0.1, None, None, None)
        assert s.format("kv") == s.to_kv()

    def test_format_unknown_raises(self) -> None:
        s = SummaryData("test", 10, 1, 0.1, None, None, None)
        with pytest.raises(ValueError, match="Unknown format"):
            s.format("xml")


class TestExtractSummary:
    def test_basic(self, basic_report: dict) -> None:
        s = extract_summary(basic_report)
        assert s.dataset == "gsm8k"
        assert s.total_samples == 100
        assert s.divergent_samples == 12
        assert s.divergence_rate == 0.12
        assert s.divergence_index_mean == 5.3
        assert s.targets is None

    def test_multi_target(self) -> None:
        report = {
            "dataset": "mmlu",
            "total_samples": 50,
            "divergent_samples": 5,
            "divergence_rate": 0.1,
            "per_target": {"http://a:8000": {}, "http://b:8000": {}},
        }
        s = extract_summary(report)
        assert s.targets == ["http://a:8000", "http://b:8000"]

    def test_missing_fields_defaults(self) -> None:
        s = extract_summary({})
        assert s.dataset == "unknown"
        assert s.total_samples == 0
        assert s.divergent_samples == 0
        assert s.divergence_rate == 0.0
        assert s.divergence_index_mean is None
        assert s.targets is None


class TestLoadAndSummarize:
    def test_oneline(self, basic_report_file: Path) -> None:
        result = load_and_summarize(basic_report_file, "oneline")
        assert "gsm8k" in result
        assert "100 samples" in result

    def test_json(self, basic_report_file: Path) -> None:
        result = load_and_summarize(basic_report_file, "json")
        parsed = json.loads(result)
        assert parsed["total_samples"] == 100

    def test_kv(self, basic_report_file: Path) -> None:
        result = load_and_summarize(basic_report_file, "kv")
        assert "dataset=gsm8k" in result

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_and_summarize(tmp_path / "nonexistent.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_and_summarize(p)


class TestCLI:
    def test_summary_oneline(self, basic_report_file: Path) -> None:
        import io
        from unittest.mock import patch

        from xpyd_acc.cli import main

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            main(["summary", str(basic_report_file)])
        output = mock_out.getvalue().strip()
        assert "gsm8k" in output
        assert "100 samples" in output

    def test_summary_json_format(self, basic_report_file: Path) -> None:
        import io
        from unittest.mock import patch

        from xpyd_acc.cli import main

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            main(["summary", str(basic_report_file), "--format", "json"])
        parsed = json.loads(mock_out.getvalue().strip())
        assert parsed["dataset"] == "gsm8k"

    def test_summary_kv_format(self, basic_report_file: Path) -> None:
        import io
        from unittest.mock import patch

        from xpyd_acc.cli import main

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            main(["summary", str(basic_report_file), "--format", "kv"])
        output = mock_out.getvalue().strip()
        assert "dataset=gsm8k" in output

    def test_summary_missing_file(self, tmp_path: Path) -> None:
        from xpyd_acc.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["summary", str(tmp_path / "nope.json")])
        assert exc_info.value.code == 1
