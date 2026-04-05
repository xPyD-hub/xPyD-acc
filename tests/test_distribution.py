"""Tests for distribution analysis module."""

from __future__ import annotations

import math

import pytest

from xpyd_acc.distribution import (
    DistributionReport,
    PositionMetrics,
    TokenDistribution,
    compare_distributions,
    format_distribution_report,
    js_divergence,
    kl_divergence,
    top_k_overlap,
)


class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_identical_distributions(self) -> None:
        p = {"a": -0.5, "b": -1.0, "c": -2.0}
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-8)

    def test_different_distributions(self) -> None:
        p = {"a": -0.1, "b": -2.5}
        q = {"a": -0.5, "b": -1.2}
        result = kl_divergence(p, q)
        assert result > 0.0

    def test_non_overlapping_tokens(self) -> None:
        p = {"a": -0.5}
        q = {"b": -0.5}
        result = kl_divergence(p, q)
        assert result > 0.0

    def test_single_token(self) -> None:
        p = {"a": -0.5}
        q = {"a": -0.5}
        assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-8)

    def test_empty_distributions(self) -> None:
        assert kl_divergence({}, {}) == pytest.approx(0.0, abs=1e-8)


class TestJSDivergence:
    """Tests for Jensen-Shannon divergence computation."""

    def test_identical_distributions(self) -> None:
        p = {"a": -0.5, "b": -1.0}
        assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self) -> None:
        p = {"a": -0.1, "b": -2.5}
        q = {"a": -0.5, "b": -1.2}
        assert js_divergence(p, q) == pytest.approx(js_divergence(q, p), abs=1e-8)

    def test_bounded(self) -> None:
        p = {"a": -0.1}
        q = {"b": -0.1}
        result = js_divergence(p, q)
        assert 0.0 <= result <= math.log(2) + 1e-6

    def test_different_distributions(self) -> None:
        p = {"a": -0.1, "b": -2.5}
        q = {"a": -0.5, "b": -1.2}
        result = js_divergence(p, q)
        assert result > 0.0


class TestTopKOverlap:
    """Tests for Jaccard overlap of top-K token sets."""

    def test_identical_sets(self) -> None:
        p = {"a": -0.5, "b": -1.0}
        q = {"a": -0.3, "b": -0.8}
        assert top_k_overlap(p, q) == pytest.approx(1.0)

    def test_disjoint_sets(self) -> None:
        p = {"a": -0.5, "b": -1.0}
        q = {"c": -0.3, "d": -0.8}
        assert top_k_overlap(p, q) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        p = {"a": -0.5, "b": -1.0}
        q = {"a": -0.3, "c": -0.8}
        # intersection=1, union=3
        assert top_k_overlap(p, q) == pytest.approx(1 / 3)

    def test_empty_sets(self) -> None:
        assert top_k_overlap({}, {}) == pytest.approx(1.0)


class TestCompareDistributions:
    """Tests for full distribution comparison."""

    def _make_dists(
        self,
        tokens_list: list[dict[str, float]],
    ) -> list[TokenDistribution]:
        return [TokenDistribution(index=i, tokens=t) for i, t in enumerate(tokens_list)]

    def test_identical(self) -> None:
        dists = self._make_dists([{"a": -0.5, "b": -1.0}, {"c": -0.3}])
        report = compare_distributions(dists, dists, kl_threshold=0.1)
        assert report.total_positions == 2
        assert report.flagged_count == 0
        assert report.mean_kl == pytest.approx(0.0, abs=1e-6)

    def test_divergent(self) -> None:
        baseline = self._make_dists([{"a": -0.1}, {"b": -0.1}])
        target = self._make_dists([{"z": -0.1}, {"b": -0.1}])
        report = compare_distributions(baseline, target, kl_threshold=0.01)
        assert report.flagged_count >= 1
        assert report.max_kl > 0.01

    def test_different_lengths(self) -> None:
        baseline = self._make_dists([{"a": -0.5}, {"b": -1.0}, {"c": -0.3}])
        target = self._make_dists([{"a": -0.5}])
        report = compare_distributions(baseline, target)
        assert report.total_positions == 1

    def test_kl_threshold(self) -> None:
        baseline = self._make_dists([{"a": -0.1, "b": -2.3}])
        target = self._make_dists([{"a": -0.5, "b": -1.0}])
        report_low = compare_distributions(baseline, target, kl_threshold=0.001)
        report_high = compare_distributions(baseline, target, kl_threshold=100.0)
        assert report_low.flagged_count >= report_high.flagged_count

    def test_empty_distributions(self) -> None:
        report = compare_distributions([], [])
        assert report.total_positions == 0
        assert report.flagged_count == 0


class TestFormatDistributionReport:
    """Tests for report formatting."""

    def test_no_flagged(self) -> None:
        report = DistributionReport(
            baseline_endpoint="http://a",
            target_endpoint="http://b",
            total_positions=5,
            flagged_count=0,
        )
        text = format_distribution_report(report)
        assert "No positions flagged" in text

    def test_with_flagged(self) -> None:
        report = DistributionReport(
            baseline_endpoint="http://a",
            target_endpoint="http://b",
            positions=[
                PositionMetrics(
                    index=2,
                    kl_divergence=0.5,
                    js_divergence=0.2,
                    top_k_overlap=0.5,
                    baseline_top="hello",
                    target_top="world",
                    flagged=True,
                ),
            ],
            total_positions=5,
            flagged_count=1,
            max_kl=0.5,
        )
        text = format_distribution_report(report)
        assert "Flagged positions" in text
        assert "[2]" in text


class TestTokenLogprobTopK:
    """Tests for top_logprobs field on TokenLogprob."""

    def test_default_empty(self) -> None:
        from xpyd_acc.logprobs import TokenLogprob
        t = TokenLogprob(index=0, token="hi", logprob=-0.5)
        assert t.top_logprobs == {}

    def test_with_top_logprobs(self) -> None:
        from xpyd_acc.logprobs import TokenLogprob
        t = TokenLogprob(index=0, token="hi", logprob=-0.5, top_logprobs={"hi": -0.5, "hey": -1.2})
        assert len(t.top_logprobs) == 2
