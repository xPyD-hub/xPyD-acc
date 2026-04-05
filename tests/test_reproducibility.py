"""Tests for reproducibility score (M62)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from xpyd_acc.reproducibility import (
    ReproducibilityReport,
    ReproducibilityResult,
    _compute_result,
    _edit_distance,
    format_reproducibility,
    run_reproducibility,
)


class TestEditDistance:
    def test_same_strings(self):
        assert _edit_distance("abc", "abc") == 0

    def test_empty_strings(self):
        assert _edit_distance("", "") == 0

    def test_one_empty(self):
        assert _edit_distance("abc", "") == 3
        assert _edit_distance("", "xyz") == 3

    def test_single_char_diff(self):
        assert _edit_distance("abc", "axc") == 1

    def test_insertion(self):
        assert _edit_distance("ac", "abc") == 1

    def test_deletion(self):
        assert _edit_distance("abc", "ac") == 1


class TestComputeResult:
    def test_all_identical(self):
        result = _compute_result("http://a", ["hello", "hello", "hello"])
        assert result.unique_count == 1
        assert result.majority_fraction == 1.0
        assert result.avg_pairwise_distance == 0.0
        assert result.runs == 3

    def test_all_different(self):
        result = _compute_result("http://a", ["a", "b", "c"])
        assert result.unique_count == 3
        assert result.majority_fraction == pytest.approx(1 / 3, abs=0.01)
        assert result.avg_pairwise_distance > 0

    def test_majority(self):
        result = _compute_result("http://a", ["x", "x", "x", "y", "z"])
        assert result.unique_count == 3
        assert result.majority_fraction == 0.6
        assert result.runs == 5

    def test_single_run(self):
        result = _compute_result("http://a", ["hello"])
        assert result.unique_count == 1
        assert result.majority_fraction == 1.0
        assert result.avg_pairwise_distance == 0.0


class TestReproducibilityResult:
    def test_to_dict(self):
        r = ReproducibilityResult(
            url="http://a", runs=3, outputs=["a", "a", "b"],
            unique_count=2, majority_fraction=0.6667,
            avg_pairwise_distance=1.0,
        )
        d = r.to_dict()
        assert d["url"] == "http://a"
        assert d["runs"] == 3
        assert d["unique_count"] == 2


class TestReproducibilityReport:
    def test_single_mode(self):
        r = ReproducibilityResult(
            url="http://a", runs=3, outputs=["a", "a", "a"],
            unique_count=1, majority_fraction=1.0,
            avg_pairwise_distance=0.0,
        )
        report = ReproducibilityReport(single=r)
        d = report.to_dict()
        assert "single" in d
        assert "baseline" not in d
        assert "target" not in d

    def test_dual_mode(self):
        r1 = _compute_result("http://base", ["a", "a"])
        r2 = _compute_result("http://target", ["b", "b"])
        report = ReproducibilityReport(baseline=r1, target=r2)
        d = report.to_dict()
        assert "baseline" in d
        assert "target" in d
        assert "single" not in d

    def test_to_json(self):
        r = _compute_result("http://a", ["x", "x"])
        report = ReproducibilityReport(single=r)
        j = report.to_json()
        data = json.loads(j)
        assert data["single"]["url"] == "http://a"


class TestFormatReproducibility:
    def test_single_format(self):
        r = _compute_result("http://a", ["hello", "hello", "world"])
        report = ReproducibilityReport(single=r)
        output = format_reproducibility(report)
        assert "Reproducibility Report" in output
        assert "http://a" in output
        assert "Unique outputs" in output

    def test_dual_format(self):
        r1 = _compute_result("http://base", ["a", "a"])
        r2 = _compute_result("http://target", ["b", "c"])
        report = ReproducibilityReport(baseline=r1, target=r2)
        output = format_reproducibility(report)
        assert "Baseline" in output
        assert "Target" in output


class TestRunReproducibility:
    @pytest.mark.asyncio
    async def test_single_endpoint(self):
        with patch(
            "xpyd_acc.reproducibility._collect_outputs",
            new_callable=AsyncMock,
            return_value=["hello", "hello", "hello"],
        ):
            report = await run_reproducibility(
                url="http://a", prompt="test", model="m", runs=3,
            )
        assert report.single is not None
        assert report.single.unique_count == 1

    @pytest.mark.asyncio
    async def test_dual_endpoint(self):
        with patch(
            "xpyd_acc.reproducibility._collect_outputs",
            new_callable=AsyncMock,
            side_effect=[
                ["a", "a", "a"],
                ["b", "c", "b"],
            ],
        ):
            report = await run_reproducibility(
                baseline_url="http://base",
                target_url="http://target",
                prompt="test",
                model="m",
                runs=3,
            )
        assert report.baseline is not None
        assert report.target is not None
        assert report.baseline.majority_fraction == 1.0
        assert report.target.majority_fraction == pytest.approx(2 / 3, abs=0.01)

    @pytest.mark.asyncio
    async def test_missing_args(self):
        with pytest.raises(ValueError, match="Either"):
            await run_reproducibility(prompt="test", model="m")

    @pytest.mark.asyncio
    async def test_json_export_roundtrip(self):
        with patch(
            "xpyd_acc.reproducibility._collect_outputs",
            new_callable=AsyncMock,
            return_value=["x", "y", "x"],
        ):
            report = await run_reproducibility(
                url="http://a", prompt="test", model="m", runs=3,
            )
        j = report.to_json()
        data = json.loads(j)
        assert data["single"]["unique_count"] == 2
        assert data["single"]["majority_fraction"] == pytest.approx(2 / 3, abs=0.01)


class TestCLIIntegration:
    def test_reproducibility_help(self, capsys):
        from xpyd_acc.cli import main
        with pytest.raises(SystemExit) as exc:
            main(["reproducibility", "--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "--runs" in captured.out
        assert "--threshold" in captured.out

    def test_reproducibility_missing_prompt(self):
        from xpyd_acc.cli import main
        with pytest.raises(SystemExit) as exc:
            main(["reproducibility", "--url", "http://a"])
        assert exc.value.code != 0
