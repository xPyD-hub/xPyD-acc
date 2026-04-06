"""Tests for latency regression detection (M84)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_acc.latency_regression import (
    LatencyRegressionResult,
    _cohens_d,
    _percentile,
    _welch_t_test,
    format_latency_regression,
    run_latency_regression,
)

# --- Unit tests for statistics helpers ---


class TestWelchTTest:
    def test_identical_samples(self):
        t, p = _welch_t_test(100.0, 10.0, 30, 100.0, 10.0, 30)
        assert t == 0.0
        assert p >= 0.99  # not significant

    def test_clearly_different(self):
        # Large difference, small variance
        t, p = _welch_t_test(100.0, 5.0, 50, 200.0, 5.0, 50)
        assert p < 0.001

    def test_small_samples(self):
        t, p = _welch_t_test(100.0, 10.0, 1, 200.0, 10.0, 1)
        assert p == 1.0  # n < 2

    def test_zero_variance(self):
        t, p = _welch_t_test(100.0, 0.0, 10, 100.0, 0.0, 10)
        assert p == 1.0


class TestCohensD:
    def test_identical(self):
        d = _cohens_d(100.0, 10.0, 30, 100.0, 10.0, 30)
        assert d == 0.0

    def test_large_effect(self):
        d = _cohens_d(100.0, 10.0, 30, 120.0, 10.0, 30)
        assert d == pytest.approx(2.0, abs=0.01)

    def test_small_samples(self):
        d = _cohens_d(100.0, 10.0, 1, 200.0, 10.0, 1)
        assert d == 0.0

    def test_zero_std(self):
        d = _cohens_d(100.0, 0.0, 10, 100.0, 0.0, 10)
        assert d == 0.0


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single(self):
        assert _percentile([42.0], 50) == 42.0

    def test_median(self):
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0

    def test_p95(self):
        data = list(range(1, 101))
        p95 = _percentile([float(x) for x in data], 95)
        assert p95 == pytest.approx(95.05, abs=0.1)


# --- Integration tests ---


class TestRunLatencyRegression:
    def _write_benchmark(self, path: Path, latencies: list[float]) -> None:
        data = {
            "url": "http://test",
            "model": "test",
            "requests": len(latencies),
            "concurrency": 1,
            "latencies_ms": latencies,
        }
        path.write_text(json.dumps(data))

    def test_no_regression(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        # Similar latencies
        self._write_benchmark(old, [100.0, 102.0, 98.0, 101.0, 99.0] * 10)
        self._write_benchmark(new, [101.0, 103.0, 99.0, 100.0, 98.0] * 10)

        result = run_latency_regression(old, new)
        assert result.verdict == "unchanged"
        assert result.p_value > 0.05

    def test_regression_detected(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        # Clear regression: new is much slower
        self._write_benchmark(old, [100.0, 102.0, 98.0, 101.0, 99.0] * 10)
        self._write_benchmark(new, [200.0, 202.0, 198.0, 201.0, 199.0] * 10)

        result = run_latency_regression(old, new)
        assert result.verdict == "slower"
        assert result.p_value < 0.05
        assert result.mean_diff_ms > 90

    def test_improvement_detected(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        self._write_benchmark(old, [200.0, 202.0, 198.0, 201.0, 199.0] * 10)
        self._write_benchmark(new, [100.0, 102.0, 98.0, 101.0, 99.0] * 10)

        result = run_latency_regression(old, new)
        assert result.verdict == "faster"
        assert result.mean_diff_ms < -90

    def test_custom_alpha(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        self._write_benchmark(old, [100.0, 102.0, 98.0, 101.0, 99.0] * 10)
        self._write_benchmark(new, [200.0, 202.0, 198.0, 201.0, 199.0] * 10)

        result = run_latency_regression(old, new, alpha=0.001)
        assert result.alpha == 0.001
        assert result.verdict == "slower"

    def test_percentiles_computed(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        self._write_benchmark(old, [100.0, 110.0, 120.0, 130.0, 140.0] * 10)
        self._write_benchmark(new, [200.0, 210.0, 220.0, 230.0, 240.0] * 10)

        result = run_latency_regression(old, new)
        assert result.old_p50_ms > 0
        assert result.new_p50_ms > 0
        assert result.old_p95_ms > 0
        assert result.new_p95_ms > 0
        assert result.old_p99_ms > 0
        assert result.new_p99_ms > 0

    def test_empty_latencies_raises(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        old.write_text(json.dumps({"latencies_ms": []}))
        new.write_text(json.dumps({"latencies_ms": [100.0]}))

        with pytest.raises(ValueError, match="non-empty"):
            run_latency_regression(old, new)

    def test_to_dict(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        self._write_benchmark(old, [100.0, 102.0, 98.0])
        self._write_benchmark(new, [200.0, 202.0, 198.0])

        result = run_latency_regression(old, new)
        d = result.to_dict()
        assert "old_mean_ms" in d
        assert "verdict" in d
        assert "cohens_d" in d

    def test_json_export(self, tmp_path):
        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        self._write_benchmark(old, [100.0, 102.0, 98.0] * 10)
        self._write_benchmark(new, [200.0, 202.0, 198.0] * 10)

        result = run_latency_regression(old, new)
        out = tmp_path / "result.json"
        with open(out, "w") as f:
            json.dump(result.to_dict(), f)

        loaded = json.loads(out.read_text())
        assert loaded["verdict"] == "slower"


class TestFormatLatencyRegression:
    def test_format_slower(self, capsys):
        result = LatencyRegressionResult(
            old_mean_ms=100.0, new_mean_ms=200.0, mean_diff_ms=100.0,
            p_value=0.001, cohens_d=1.5,
            old_p50_ms=100.0, new_p50_ms=200.0,
            old_p95_ms=110.0, new_p95_ms=210.0,
            old_p99_ms=120.0, new_p99_ms=220.0,
            verdict="slower", alpha=0.05,
            old_count=50, new_count=50,
        )
        format_latency_regression(result)
        # Just verify it runs without error

    def test_format_faster(self, capsys):
        result = LatencyRegressionResult(
            old_mean_ms=200.0, new_mean_ms=100.0, mean_diff_ms=-100.0,
            p_value=0.001, cohens_d=-1.5,
            old_p50_ms=200.0, new_p50_ms=100.0,
            old_p95_ms=210.0, new_p95_ms=110.0,
            old_p99_ms=220.0, new_p99_ms=120.0,
            verdict="faster", alpha=0.05,
            old_count=50, new_count=50,
        )
        format_latency_regression(result)

    def test_format_unchanged(self, capsys):
        result = LatencyRegressionResult(
            old_mean_ms=100.0, new_mean_ms=101.0, mean_diff_ms=1.0,
            p_value=0.8, cohens_d=0.05,
            old_p50_ms=100.0, new_p50_ms=101.0,
            old_p95_ms=110.0, new_p95_ms=111.0,
            old_p99_ms=120.0, new_p99_ms=121.0,
            verdict="unchanged", alpha=0.05,
            old_count=50, new_count=50,
        )
        format_latency_regression(result)


class TestCLI:
    def _write_benchmark(self, path: Path, latencies: list[float]) -> None:
        data = {
            "url": "http://test",
            "model": "test",
            "requests": len(latencies),
            "concurrency": 1,
            "latencies_ms": latencies,
        }
        path.write_text(json.dumps(data))

    def test_cli_no_regression(self, tmp_path):
        from xpyd_acc.cli import main

        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        # Use identical latencies so no significant difference
        self._write_benchmark(old, [100.0, 110.0, 90.0, 105.0, 95.0] * 10)
        self._write_benchmark(new, [101.0, 109.0, 91.0, 104.0, 96.0] * 10)

        main(["latency-regression", "--old", str(old), "--new", str(new)])

    def test_cli_regression_exit_1(self, tmp_path):
        from xpyd_acc.cli import main

        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        self._write_benchmark(old, [100.0, 102.0, 98.0] * 10)
        self._write_benchmark(new, [300.0, 302.0, 298.0] * 10)

        with pytest.raises(SystemExit) as exc_info:
            main(["latency-regression", "--old", str(old), "--new", str(new)])
        assert exc_info.value.code == 1

    def test_cli_json_export(self, tmp_path):
        from xpyd_acc.cli import main

        old = tmp_path / "old.json"
        new = tmp_path / "new.json"
        out = tmp_path / "result.json"
        self._write_benchmark(old, [100.0, 102.0, 98.0] * 10)
        self._write_benchmark(new, [100.0, 102.0, 98.0] * 10)

        main(["latency-regression", "--old", str(old), "--new", str(new),
              "--json", str(out)])
        loaded = json.loads(out.read_text())
        assert "verdict" in loaded
