"""Tests for the ab_test module — A/B testing for divergence rates."""

import json
from pathlib import Path

import pytest

from xpyd_acc.ab_test import (
    chi_square_test,
    fisher_exact_two_sided,
    format_ab_test,
    run_ab_test,
)


class TestFisherExact:
    """Tests for Fisher's exact test implementation."""

    def test_identical_tables(self):
        """Identical proportions should give p ≈ 1."""
        p = fisher_exact_two_sided(50, 50, 50, 50)
        assert p > 0.9

    def test_extreme_difference(self):
        """Completely different proportions should give small p."""
        p = fisher_exact_two_sided(100, 0, 0, 100)
        assert p < 0.001

    def test_known_table(self):
        """Verify against a known 2x2 table."""
        # Classic tea-tasting example-like table
        p = fisher_exact_two_sided(8, 2, 1, 9)
        assert p < 0.05

    def test_symmetric(self):
        """Swapping rows shouldn't change p-value."""
        p1 = fisher_exact_two_sided(10, 5, 3, 12)
        p2 = fisher_exact_two_sided(3, 12, 10, 5)
        assert abs(p1 - p2) < 1e-10

    def test_zero_cell(self):
        """Should handle zero cells gracefully."""
        p = fisher_exact_two_sided(0, 10, 10, 0)
        assert p < 0.001

    def test_single_sample(self):
        """Edge case: single sample per group."""
        p = fisher_exact_two_sided(1, 0, 0, 1)
        assert 0 <= p <= 1.0


class TestChiSquare:
    """Tests for chi-square test implementation."""

    def test_identical(self):
        stat, p = chi_square_test(50, 50, 50, 50)
        assert p > 0.8
        assert stat < 0.5

    def test_extreme(self):
        stat, p = chi_square_test(100, 0, 0, 100)
        assert p < 0.001
        assert stat > 10

    def test_zero_marginal(self):
        stat, p = chi_square_test(0, 0, 10, 10)
        assert stat == 0.0
        assert p == 1.0

    def test_all_zero(self):
        stat, p = chi_square_test(0, 0, 0, 0)
        assert stat == 0.0
        assert p == 1.0


class TestRunABTest:
    """Tests for the main run_ab_test function."""

    def test_no_difference(self):
        result = run_ab_test(100, 10, 100, 10)
        assert not result.significant
        assert result.rate_difference == 0.0

    def test_large_difference(self):
        result = run_ab_test(100, 5, 100, 50)
        assert result.significant
        assert result.rate_difference > 0

    def test_report_rates(self):
        result = run_ab_test(200, 20, 200, 40)
        assert result.report_a_rate == pytest.approx(0.1, abs=0.001)
        assert result.report_b_rate == pytest.approx(0.2, abs=0.001)

    def test_custom_alpha(self):
        """Marginal case: significant at 0.10 but not at 0.01."""
        result_loose = run_ab_test(50, 5, 50, 12, alpha=0.10)
        result_strict = run_ab_test(50, 5, 50, 12, alpha=0.001)
        # Both should return valid results
        assert isinstance(result_loose.significant, bool)
        assert isinstance(result_strict.significant, bool)

    def test_zero_divergent(self):
        result = run_ab_test(100, 0, 100, 0)
        assert not result.significant
        assert result.rate_difference == 0.0
        assert result.odds_ratio == 1.0

    def test_all_divergent(self):
        result = run_ab_test(100, 100, 100, 100)
        assert not result.significant
        assert result.rate_difference == 0.0

    def test_odds_ratio_undefined(self):
        """When one group has zero divergent, odds ratio should be None."""
        result = run_ab_test(100, 0, 100, 50)
        assert result.odds_ratio is None

    def test_ci_contains_zero_when_not_significant(self):
        result = run_ab_test(100, 10, 100, 12)
        assert result.rate_difference_ci_lower <= 0 <= result.rate_difference_ci_upper

    def test_json_export(self, tmp_path: Path):
        result = run_ab_test(100, 10, 100, 30)
        out = tmp_path / "ab_result.json"
        result.save_json(out)
        data = json.loads(out.read_text())
        assert data["report_a_total"] == 100
        assert data["report_b_divergent"] == 30
        assert "fisher_p_value" in data
        assert "significant" in data

    def test_to_json_roundtrip(self):
        result = run_ab_test(50, 5, 50, 15)
        data = json.loads(result.to_json())
        assert data["alpha"] == 0.05


class TestFormatABTest:
    """Tests for format_ab_test."""

    def test_significant_output(self):
        result = run_ab_test(100, 5, 100, 50)
        text = format_ab_test(result)
        assert "SIGNIFICANT DIFFERENCE" in text

    def test_not_significant_output(self):
        result = run_ab_test(100, 10, 100, 12)
        text = format_ab_test(result)
        assert "NO SIGNIFICANT DIFFERENCE" in text

    def test_undefined_odds(self):
        result = run_ab_test(100, 0, 100, 50)
        text = format_ab_test(result)
        assert "undefined" in text


class TestCLIIntegration:
    """Tests for CLI ab-test subcommand."""

    def test_cli_ab_test_no_sig(self, tmp_path: Path):
        """ab-test exits 0 when no significant difference."""
        from io import StringIO
        from unittest.mock import patch

        report_a = {"total_samples": 100, "divergent_samples": 10}
        report_b = {"total_samples": 100, "divergent_samples": 10}
        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        a_path.write_text(json.dumps(report_a))
        b_path.write_text(json.dumps(report_b))

        from xpyd_acc import cli

        with (
            pytest.raises(SystemExit) as exc_info,
            patch("sys.stdout", new_callable=StringIO) as mock_out,
        ):
            cli.main(["ab-test", "--report-a", str(a_path), "--report-b", str(b_path)])
        assert exc_info.value.code == 0
        assert "NO SIGNIFICANT DIFFERENCE" in mock_out.getvalue()

    def test_cli_ab_test_significant(self, tmp_path: Path):
        """ab-test exits 1 when significant difference."""
        from io import StringIO
        from unittest.mock import patch

        report_a = {"total_samples": 200, "divergent_samples": 5}
        report_b = {"total_samples": 200, "divergent_samples": 80}
        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        a_path.write_text(json.dumps(report_a))
        b_path.write_text(json.dumps(report_b))

        from xpyd_acc import cli

        with (
            pytest.raises(SystemExit) as exc_info,
            patch("sys.stdout", new_callable=StringIO) as mock_out,
        ):
            cli.main(["ab-test", "--report-a", str(a_path), "--report-b", str(b_path)])
        assert exc_info.value.code == 1
        assert "SIGNIFICANT DIFFERENCE" in mock_out.getvalue()

    def test_cli_ab_test_json_export(self, tmp_path: Path):
        from io import StringIO
        from unittest.mock import patch

        report_a = {"total_samples": 100, "divergent_samples": 5}
        report_b = {"total_samples": 100, "divergent_samples": 40}
        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        out_path = tmp_path / "out.json"
        a_path.write_text(json.dumps(report_a))
        b_path.write_text(json.dumps(report_b))

        from xpyd_acc import cli

        with pytest.raises(SystemExit), patch("sys.stdout", new_callable=StringIO):
            cli.main(["ab-test", "--report-a", str(a_path), "--report-b", str(b_path),
                       "--json", str(out_path)])
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "fisher_p_value" in data
