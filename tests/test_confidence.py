"""Tests for confidence interval computation and integration."""

from __future__ import annotations

import json

import pytest

from xpyd_acc.confidence import wilson_ci


class TestWilsonCI:
    """Tests for the wilson_ci function."""

    def test_basic_50_percent(self):
        """50% divergence on 100 samples."""
        lower, upper = wilson_ci(50, 100, 0.95)
        assert 0.39 < lower < 0.42
        assert 0.58 < upper < 0.61

    def test_zero_divergence(self):
        """0 out of N divergent."""
        lower, upper = wilson_ci(0, 100, 0.95)
        assert lower == 0.0
        assert 0.0 < upper < 0.05

    def test_all_divergent(self):
        """N out of N divergent."""
        lower, upper = wilson_ci(100, 100, 0.95)
        assert 0.95 < lower <= 1.0
        assert upper >= 0.99

    def test_single_sample_divergent(self):
        """1 out of 1 divergent."""
        lower, upper = wilson_ci(1, 1, 0.95)
        assert 0.0 < lower
        assert upper == 1.0

    def test_single_sample_match(self):
        """0 out of 1 divergent."""
        lower, upper = wilson_ci(0, 1, 0.95)
        assert lower == 0.0
        assert upper < 1.0

    def test_small_sample(self):
        """Small sample: 1 out of 10."""
        lower, upper = wilson_ci(1, 10, 0.95)
        assert lower >= 0.0
        assert upper <= 1.0
        assert lower < 0.1
        assert upper > 0.1

    def test_bounds_always_valid(self):
        """Lower <= rate <= upper for various inputs."""
        for div in range(0, 21):
            lower, upper = wilson_ci(div, 20, 0.95)
            rate = div / 20
            assert lower <= rate + 1e-9  # small epsilon for float
            assert upper >= rate - 1e-9
            assert 0.0 <= lower <= upper <= 1.0

    def test_higher_confidence_wider(self):
        """99% CI should be wider than 90% CI."""
        low90, high90 = wilson_ci(10, 100, 0.90)
        low99, high99 = wilson_ci(10, 100, 0.99)
        assert low99 < low90
        assert high99 > high90

    def test_invalid_total_zero(self):
        with pytest.raises(ValueError, match="total must be positive"):
            wilson_ci(0, 0, 0.95)

    def test_invalid_negative_divergent(self):
        with pytest.raises(ValueError, match="divergent must be in"):
            wilson_ci(-1, 10, 0.95)

    def test_invalid_divergent_exceeds_total(self):
        with pytest.raises(ValueError, match="divergent must be in"):
            wilson_ci(11, 10, 0.95)

    def test_invalid_confidence_zero(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            wilson_ci(5, 10, 0.0)

    def test_invalid_confidence_one(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            wilson_ci(5, 10, 1.0)


class TestApplyConfidence:
    """Tests for apply_confidence integration with BatchReport."""

    def test_apply_confidence_basic(self):
        from xpyd_acc.batch_compare import BatchReport, apply_confidence

        report = BatchReport(
            total_samples=100,
            divergent_samples=10,
            match_samples=90,
            divergence_rate=0.1,
            results=[],
        )
        apply_confidence(report, 0.95)
        assert report.divergence_ci_lower is not None
        assert report.divergence_ci_upper is not None
        assert report.confidence_level == 0.95
        assert report.divergence_ci_lower < 0.1
        assert report.divergence_ci_upper > 0.1

    def test_ci_in_json_export(self):
        from xpyd_acc.batch_compare import BatchReport, apply_confidence

        report = BatchReport(
            total_samples=50,
            divergent_samples=5,
            match_samples=45,
            divergence_rate=0.1,
            results=[],
        )
        apply_confidence(report, 0.95)
        data = json.loads(report.to_json())
        assert "divergence_ci_lower" in data
        assert "divergence_ci_upper" in data
        assert "confidence_level" in data
        assert data["confidence_level"] == 0.95

    def test_ci_none_by_default(self):
        from xpyd_acc.batch_compare import BatchReport

        report = BatchReport(
            total_samples=10,
            divergent_samples=1,
            match_samples=9,
            divergence_rate=0.1,
            results=[],
        )
        assert report.divergence_ci_lower is None
        assert report.divergence_ci_upper is None
        data = json.loads(report.to_json())
        assert data["divergence_ci_lower"] is None

    def test_ci_in_markdown_export(self):
        from xpyd_acc.batch_compare import BatchReport, apply_confidence

        report = BatchReport(
            total_samples=100,
            divergent_samples=10,
            match_samples=90,
            divergence_rate=0.1,
            results=[],
        )
        apply_confidence(report, 0.95)
        md = report.to_markdown()
        assert "95% CI" in md
        assert "[" in md  # bracket notation for interval

    def test_ci_in_format_report(self):
        from xpyd_acc.batch_compare import BatchReport, apply_confidence, format_report

        report = BatchReport(
            total_samples=100,
            divergent_samples=10,
            match_samples=90,
            divergence_rate=0.1,
            results=[],
        )
        apply_confidence(report, 0.95)
        text = format_report(report)
        assert "95% CI" in text

    def test_no_ci_in_format_when_not_applied(self):
        from xpyd_acc.batch_compare import BatchReport, format_report

        report = BatchReport(
            total_samples=100,
            divergent_samples=10,
            match_samples=90,
            divergence_rate=0.1,
            results=[],
        )
        text = format_report(report)
        assert "CI" not in text

    def test_apply_confidence_custom_level(self):
        from xpyd_acc.batch_compare import BatchReport, apply_confidence

        report = BatchReport(
            total_samples=200,
            divergent_samples=20,
            match_samples=180,
            divergence_rate=0.1,
            results=[],
        )
        apply_confidence(report, 0.99)
        assert report.confidence_level == 0.99
        # 99% CI should be wider than default 95%
        low99 = report.divergence_ci_lower
        up99 = report.divergence_ci_upper

        apply_confidence(report, 0.90)
        assert report.divergence_ci_lower > low99
        assert report.divergence_ci_upper < up99
