"""Tests for output length bias detection module."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from xpyd_acc.length_bias import (
    LengthBiasResult,
    SampleLength,
    _normal_cdf,
    _paired_t_test,
    analyze_length_bias,
    format_length_bias,
    load_report_file,
)

# ---------------------------------------------------------------------------
# _paired_t_test
# ---------------------------------------------------------------------------


def test_paired_t_test_no_samples():
    t, p = _paired_t_test([])
    assert t == 0.0
    assert p == 1.0


def test_paired_t_test_one_sample():
    t, p = _paired_t_test([5])
    assert t == 0.0
    assert p == 1.0


def test_paired_t_test_no_diff():
    t, p = _paired_t_test([0, 0, 0, 0])
    assert t == 0.0
    assert p == 1.0


def test_paired_t_test_all_same_nonzero():
    t, p = _paired_t_test([10, 10, 10, 10])
    assert math.isinf(t)
    assert p == 0.0


def test_paired_t_test_significant():
    # Large consistent positive diffs → significant
    diffs = [100, 110, 90, 105, 95, 102, 98, 108, 92, 103]
    t, p = _paired_t_test(diffs)
    assert t > 0
    assert p < 0.001


def test_paired_t_test_not_significant():
    # Small mixed diffs → not significant
    diffs = [1, -1, 2, -2, 1, -1, 0, 0]
    t, p = _paired_t_test(diffs)
    assert p > 0.1


# ---------------------------------------------------------------------------
# _normal_cdf
# ---------------------------------------------------------------------------


def test_normal_cdf_zero():
    assert abs(_normal_cdf(0) - 0.5) < 1e-9


def test_normal_cdf_large_positive():
    assert _normal_cdf(5.0) > 0.999


def test_normal_cdf_large_negative():
    assert _normal_cdf(-5.0) < 0.001


# ---------------------------------------------------------------------------
# analyze_length_bias
# ---------------------------------------------------------------------------


def _make_report(pairs: list[tuple[str, str]]) -> dict:
    """Create a minimal report from (baseline_output, target_output) pairs."""
    return {
        "results": [
            {
                "sample_id": f"s{i}",
                "baseline_output": bl,
                "target_output": tg,
            }
            for i, (bl, tg) in enumerate(pairs)
        ]
    }


def test_analyze_empty_report():
    result = analyze_length_bias({"results": []})
    assert result.sample_count == 0
    assert result.classification == "no_bias"


def test_analyze_no_bias():
    pairs = [("hello", "world"), ("foo", "bar"), ("abc", "xyz")]
    result = analyze_length_bias(_make_report(pairs))
    assert result.sample_count == 3
    assert result.classification == "no_bias"


def test_analyze_shorter_bias():
    # Target consistently shorter
    pairs = [("a" * 100, "b" * 20)] * 20
    result = analyze_length_bias(_make_report(pairs))
    assert result.classification == "shorter_bias"
    assert result.mean_diff < 0
    assert result.p_value < 0.05


def test_analyze_longer_bias():
    # Target consistently longer
    pairs = [("b" * 20, "a" * 100)] * 20
    result = analyze_length_bias(_make_report(pairs))
    assert result.classification == "longer_bias"
    assert result.mean_diff > 0
    assert result.p_value < 0.05


def test_analyze_equal_lengths():
    pairs = [("abc", "xyz"), ("de", "fg"), ("h", "i")]
    result = analyze_length_bias(_make_report(pairs))
    assert result.mean_diff == 0.0
    assert result.classification == "no_bias"


def test_analyze_empty_outputs():
    pairs = [("", ""), ("", "")]
    result = analyze_length_bias(_make_report(pairs))
    assert result.sample_count == 2
    assert result.mean_diff == 0.0
    for s in result.samples:
        assert s.length_ratio == 0.0


def test_analyze_missing_output_fields():
    report = {"results": [{"sample_id": "s0"}]}
    result = analyze_length_bias(report)
    assert result.sample_count == 1
    assert result.samples[0].baseline_length == 0


def test_analyze_custom_alpha():
    # With very high alpha, even small diffs might be "significant"
    pairs = [("ab", "abc"), ("ab", "abc"), ("ab", "abc")]
    result = analyze_length_bias(_make_report(pairs), alpha=0.99)
    assert result.alpha == 0.99


# ---------------------------------------------------------------------------
# SampleLength
# ---------------------------------------------------------------------------


def test_sample_length_fields():
    s = SampleLength(
        sample_id="s0",
        baseline_length=100,
        target_length=80,
        length_diff=-20,
        length_ratio=0.8,
    )
    assert s.length_diff == -20
    assert s.length_ratio == 0.8


# ---------------------------------------------------------------------------
# LengthBiasResult.to_dict
# ---------------------------------------------------------------------------


def test_result_to_dict():
    result = LengthBiasResult(
        sample_count=2,
        mean_baseline_length=50.0,
        mean_target_length=30.0,
        mean_diff=-20.0,
        median_diff=-20.0,
        stdev_diff=0.0,
        t_statistic=-5.0,
        p_value=0.001,
        classification="shorter_bias",
        alpha=0.05,
        samples=[
            SampleLength("s0", 50, 30, -20, 0.6),
        ],
    )
    d = result.to_dict()
    assert d["classification"] == "shorter_bias"
    assert len(d["samples"]) == 1
    assert d["samples"][0]["length_diff"] == -20


# ---------------------------------------------------------------------------
# format_length_bias
# ---------------------------------------------------------------------------


def test_format_no_bias():
    result = LengthBiasResult(
        sample_count=5, mean_baseline_length=50.0, mean_target_length=50.0,
        mean_diff=0.0, median_diff=0.0, stdev_diff=0.0,
        t_statistic=0.0, p_value=1.0, classification="no_bias", alpha=0.05,
    )
    text = format_length_bias(result)
    assert "No significant" in text


def test_format_shorter_bias():
    result = LengthBiasResult(
        sample_count=10, mean_baseline_length=100.0, mean_target_length=60.0,
        mean_diff=-40.0, median_diff=-40.0, stdev_diff=5.0,
        t_statistic=-8.0, p_value=0.0001, classification="shorter_bias", alpha=0.05,
        samples=[SampleLength(f"s{i}", 100, 60, -40, 0.6) for i in range(10)],
    )
    text = format_length_bias(result)
    assert "SHORTER" in text
    assert "Distribution" in text


def test_format_longer_bias():
    result = LengthBiasResult(
        sample_count=5, mean_baseline_length=50.0, mean_target_length=90.0,
        mean_diff=40.0, median_diff=40.0, stdev_diff=3.0,
        t_statistic=10.0, p_value=0.0001, classification="longer_bias", alpha=0.05,
        samples=[SampleLength(f"s{i}", 50, 90, 40, 1.8) for i in range(5)],
    )
    text = format_length_bias(result)
    assert "LONGER" in text


# ---------------------------------------------------------------------------
# load_report_file
# ---------------------------------------------------------------------------


def test_load_report_file(tmp_path: Path):
    report = {"results": [{"sample_id": "s0", "baseline_output": "hi", "target_output": "hey"}]}
    f = tmp_path / "report.json"
    f.write_text(json.dumps(report))
    loaded = load_report_file(str(f))
    assert loaded["results"][0]["sample_id"] == "s0"


def test_load_report_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_report_file("/nonexistent/path.json")


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_length_bias_help(capsys):
    from xpyd_acc.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["length-bias", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "length-bias" in captured.out or "report" in captured.out


def test_cli_length_bias_basic(tmp_path: Path, capsys):
    from xpyd_acc.cli import main

    # Equal-length outputs → no bias → exit 0
    report = {
        "results": [
            {"sample_id": "s0", "baseline_output": "hello", "target_output": "world"},
            {"sample_id": "s1", "baseline_output": "foo", "target_output": "bar"},
        ]
    }
    f = tmp_path / "report.json"
    f.write_text(json.dumps(report))

    main(["length-bias", "--report", str(f)])
    captured = capsys.readouterr()
    assert "Length Bias" in captured.out


def test_cli_length_bias_json_export(tmp_path: Path, capsys):
    from xpyd_acc.cli import main

    # Equal-length outputs → no bias → exit 0
    report = {
        "results": [
            {"sample_id": f"s{i}", "baseline_output": "abcde", "target_output": "vwxyz"}
            for i in range(10)
        ]
    }
    f = tmp_path / "report.json"
    f.write_text(json.dumps(report))
    out = tmp_path / "result.json"

    main(["length-bias", "--report", str(f), "--json", str(out)])
    result = json.loads(out.read_text())
    assert "classification" in result
    assert "samples" in result


def test_cli_length_bias_exit_code_on_bias(tmp_path: Path):
    from xpyd_acc.cli import main

    report = {
        "results": [
            {"sample_id": f"s{i}", "baseline_output": "a" * 200, "target_output": "b" * 10}
            for i in range(30)
        ]
    }
    f = tmp_path / "report.json"
    f.write_text(json.dumps(report))

    with pytest.raises(SystemExit) as exc:
        main(["length-bias", "--report", str(f)])
    assert exc.value.code == 1
