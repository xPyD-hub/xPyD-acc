"""Tests for root cause heuristics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from xpyd_acc.batch_compare import BatchReport, SampleResult
from xpyd_acc.root_cause import (
    Evidence,
    RootCauseAnalysis,
    _classify_sample,
    analyze_from_file,
    analyze_root_cause,
    format_root_cause,
)


def _make_result(
    sample_id: str = "s1",
    match: bool = False,
    divergence_index: int | None = 2,
    logprob_gap: float | None = 0.8,
    context_length: int = 100,
    baseline_finish_reason: str | None = "stop",
    target_finish_reason: str | None = "stop",
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        prompt="test prompt",
        baseline_output="aaa",
        target_output="bbb",
        exact_match=match,
        first_divergence_index=divergence_index,
        baseline_logprob_at_divergence=None,
        target_logprob_at_divergence=None,
        logprob_gap=logprob_gap,
        classification="likely_bug" if not match else "match",
        context_length=context_length,
    )


def _make_report(results: list[SampleResult]) -> BatchReport:
    divergent = [r for r in results if not r.exact_match]
    return BatchReport(




        total_samples=len(results),
        match_samples=len(results) - len(divergent),
        divergent_samples=len(divergent),
        divergence_rate=len(divergent) / max(len(results), 1),
        results=results,
    )


# --- _classify_sample tests ---


def test_classify_matching_sample():
    r = _make_result(match=True)
    assert _classify_sample(r) is None


def test_classify_prefill_early_high_gap():
    r = _make_result(divergence_index=2, logprob_gap=0.8)
    assert _classify_sample(r) == "prefill"


def test_classify_prefill_early_no_gap():
    r = _make_result(divergence_index=1, logprob_gap=None)
    assert _classify_sample(r) == "prefill"


def test_classify_kv_transfer_mid_high_gap():
    r = _make_result(divergence_index=20, logprob_gap=0.7)
    assert _classify_sample(r) == "kv_transfer"


def test_classify_decode_late_low_gap():
    r = _make_result(divergence_index=50, logprob_gap=0.05)
    assert _classify_sample(r) == "decode"


def test_classify_truncation():
    r = _make_result(divergence_index=10, logprob_gap=0.3)
    r.baseline_finish_reason = "length"
    assert _classify_sample(r) == "truncation"


def test_classify_truncation_target():
    r = _make_result(divergence_index=10, logprob_gap=0.3)
    r.target_finish_reason = "length"
    assert _classify_sample(r) == "truncation"


# --- analyze_root_cause tests ---


def test_no_divergent_samples():
    report = _make_report([_make_result(match=True)])
    analysis = analyze_root_cause(report)
    assert analysis.classification == "no_divergence"
    assert analysis.confidence == 1.0
    assert analysis.total_divergent == 0


def test_all_prefill():
    results = [
        _make_result(sample_id=f"s{i}", divergence_index=2, logprob_gap=0.9)
        for i in range(5)
    ]
    report = _make_report(results)
    analysis = analyze_root_cause(report)
    assert analysis.classification == "prefill"
    assert analysis.confidence == 1.0
    assert analysis.total_divergent == 5


def test_all_decode():
    results = [
        _make_result(sample_id=f"s{i}", divergence_index=30, logprob_gap=0.05)
        for i in range(5)
    ]
    report = _make_report(results)
    analysis = analyze_root_cause(report)
    assert analysis.classification == "decode"


def test_mixed_causes():
    results = [
        _make_result(sample_id="s1", divergence_index=1, logprob_gap=0.9),  # prefill
        _make_result(sample_id="s2", divergence_index=30, logprob_gap=0.05),  # decode
        _make_result(sample_id="s3", divergence_index=20, logprob_gap=0.8),  # kv_transfer
    ]
    report = _make_report(results)
    analysis = analyze_root_cause(report)
    assert analysis.classification == "mixed"


def test_dominant_cause_above_threshold():
    results = [
        _make_result(sample_id=f"s{i}", divergence_index=2, logprob_gap=0.9)
        for i in range(7)
    ] + [
        _make_result(sample_id="s10", divergence_index=30, logprob_gap=0.05),
    ]
    report = _make_report(results)
    analysis = analyze_root_cause(report)
    assert analysis.classification == "prefill"
    assert analysis.confidence > 0.6


# --- Serialization tests ---


def test_to_dict_round_trip():
    analysis = RootCauseAnalysis(
        classification="prefill",
        confidence=0.85,
        evidence=[
            Evidence(
                rule="early_divergence_high_gap",
                description="test",
                sample_count=3,
                sample_ids=["s1", "s2"],
            )
        ],
        suggested_steps=["Do something"],
        total_divergent=3,
        total_samples=10,
    )
    d = analysis.to_dict()
    restored = RootCauseAnalysis.from_dict(d)
    assert restored.classification == "prefill"
    assert restored.confidence == 0.85
    assert len(restored.evidence) == 1
    assert restored.evidence[0].sample_count == 3


def test_to_json():
    analysis = RootCauseAnalysis(
        classification="decode",
        confidence=0.9,
        total_divergent=5,
        total_samples=10,
    )
    j = analysis.to_json()
    parsed = json.loads(j)
    assert parsed["classification"] == "decode"


# --- format_root_cause tests ---


def test_format_contains_classification():
    analysis = RootCauseAnalysis(
        classification="kv_transfer",
        confidence=0.75,
        evidence=[
            Evidence(
                rule="mid_divergence_high_gap",
                description="test evidence",
                sample_count=5,
            )
        ],
        suggested_steps=["Check KV cache"],
        total_divergent=5,
        total_samples=10,
    )
    output = format_root_cause(analysis)
    assert "KV TRANSFER" in output
    assert "75.0%" in output
    assert "test evidence" in output
    assert "Check KV cache" in output


def test_format_no_divergence():
    analysis = RootCauseAnalysis(
        classification="no_divergence",
        confidence=1.0,
        total_divergent=0,
        total_samples=10,
    )
    output = format_root_cause(analysis)
    assert "NO DIVERGENCE" in output


# --- analyze_from_file test ---


def test_analyze_from_file():
    report = _make_report(
        [_make_result(sample_id=f"s{i}", divergence_index=2, logprob_gap=0.9) for i in range(3)]
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write(report.to_json())
        path = f.name

    analysis = analyze_from_file(path)
    assert analysis.classification == "prefill"
    assert analysis.total_divergent == 3
    Path(path).unlink()


# --- CLI integration test ---


def test_cli_root_cause(capsys):
    report = _make_report(
        [_make_result(sample_id=f"s{i}", divergence_index=2, logprob_gap=0.9) for i in range(3)]
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write(report.to_json())
        path = f.name

    from argparse import Namespace

    from xpyd_acc.cli.analysis import handle_root_cause

    args = Namespace(report=path, rc_json=None)
    handle_root_cause(args)

    captured = capsys.readouterr()
    assert "PREFILL" in captured.out

    Path(path).unlink()


def test_cli_root_cause_json_export():
    report = _make_report(
        [_make_result(sample_id="s1", divergence_index=30, logprob_gap=0.05)]
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as rf:
        rf.write(report.to_json())
        report_path = rf.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as jf:
        json_path = jf.name

    from argparse import Namespace

    from xpyd_acc.cli.analysis import handle_root_cause

    args = Namespace(report=report_path, rc_json=json_path)
    handle_root_cause(args)

    exported = json.loads(Path(json_path).read_text())
    assert exported["classification"] == "decode"

    Path(report_path).unlink()
    Path(json_path).unlink()


# --- Suggestions tests ---


def test_suggestions_for_each_classification():
    from xpyd_acc.root_cause import _build_suggestions

    for cls in ("prefill", "kv_transfer", "decode", "truncation", "mixed", "inconclusive"):
        steps = _build_suggestions(cls, {})
        assert len(steps) >= 2, f"No suggestions for {cls}"


def test_inconclusive_when_all_unknown():
    """Samples with mid-range gap that don't match any clear heuristic."""
    results = [
        _make_result(sample_id=f"s{i}", divergence_index=10, logprob_gap=0.3)
        for i in range(5)
    ]
    # Make sure they're not truncated
    for r in results:
        r.baseline_finish_reason = "stop"
        r.target_finish_reason = "stop"
    report = _make_report(results)
    analysis = analyze_root_cause(report)
    assert analysis.classification == "inconclusive"
