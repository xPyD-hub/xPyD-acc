"""Tests for smart_retry module."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from xpyd_acc.batch_compare import BatchReport, SampleResult
from xpyd_acc.smart_retry import (
    SampleRetryResult,
    SmartRetryResult,
    format_smart_retry,
    run_smart_retry,
)


def _make_result(
    sample_id: str,
    prompt: str = "test",
    match: bool = True,
    classification: str = "match",
    divergence_index: int | None = None,
    logprob_gap: float | None = None,
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        prompt=prompt,
        baseline_output="hello",
        target_output="hello" if match else "world",
        exact_match=match,
        first_divergence_index=divergence_index,
        baseline_logprob_at_divergence=None,
        target_logprob_at_divergence=None,
        logprob_gap=logprob_gap,
        classification=classification,
        context_length=10,
    )


def _make_report(results: list[SampleResult]) -> BatchReport:
    divergent = [r for r in results if not r.exact_match]
    total = len(results)
    return BatchReport(
        total_samples=total,
        divergent_samples=len(divergent),
        match_samples=total - len(divergent),
        divergence_rate=len(divergent) / total if total else 0.0,
        results=results,
    )


# --- SampleRetryResult tests ---


def test_sample_retry_result_to_dict():
    r = SampleRetryResult(
        sample_id="s1",
        original_classification="likely_bug",
        retry_match=False,
        retry_classification="deterministic",
    )
    d = r.to_dict()
    assert d["sample_id"] == "s1"
    assert d["retry_classification"] == "deterministic"
    assert d["retry_match"] is False


# --- SmartRetryResult tests ---


def test_smart_retry_result_to_dict():
    r = SmartRetryResult(
        original_divergent=3,
        deterministic_count=2,
        stochastic_count=1,
        deterministic_rate=2 / 3,
        stochastic_rate=1 / 3,
        per_sample=[
            SampleRetryResult("s1", "likely_bug", False, "deterministic"),
            SampleRetryResult("s2", "likely_bug", False, "deterministic"),
            SampleRetryResult("s3", "likely_uncertainty", True, "stochastic"),
        ],
    )
    d = r.to_dict()
    assert d["original_divergent"] == 3
    assert d["deterministic_count"] == 2
    assert d["stochastic_count"] == 1
    assert len(d["per_sample"]) == 3


def test_smart_retry_result_to_json():
    r = SmartRetryResult(
        original_divergent=1,
        deterministic_count=0,
        stochastic_count=1,
        deterministic_rate=0.0,
        stochastic_rate=1.0,
        per_sample=[
            SampleRetryResult("s1", "likely_bug", True, "stochastic"),
        ],
    )
    j = json.loads(r.to_json())
    assert j["stochastic_count"] == 1


# --- run_smart_retry tests ---


def test_run_smart_retry_no_divergent():
    """No divergent samples → empty result."""
    report = _make_report([_make_result("s1", match=True)])
    result = asyncio.run(
        run_smart_retry(report, "http://base", "http://target")
    )
    assert result.original_divergent == 0
    assert result.deterministic_count == 0
    assert result.stochastic_count == 0


@patch("xpyd_acc.smart_retry.run_batch")
def test_run_smart_retry_all_deterministic(mock_run_batch):
    """All divergent samples still diverge under greedy → deterministic."""
    original = _make_report([
        _make_result("s1", match=False, classification="likely_bug"),
        _make_result("s2", match=False, classification="likely_bug"),
    ])

    # Retry also diverges
    retry_report = _make_report([
        _make_result("s1", match=False, classification="likely_bug"),
        _make_result("s2", match=False, classification="likely_bug"),
    ])
    mock_run_batch.return_value = retry_report

    result = asyncio.run(
        run_smart_retry(original, "http://base", "http://target")
    )
    assert result.original_divergent == 2
    assert result.deterministic_count == 2
    assert result.stochastic_count == 0
    assert result.deterministic_rate == 1.0


@patch("xpyd_acc.smart_retry.run_batch")
def test_run_smart_retry_all_stochastic(mock_run_batch):
    """All divergent samples match under greedy → stochastic."""
    original = _make_report([
        _make_result("s1", match=False, classification="likely_uncertainty"),
    ])

    retry_report = _make_report([
        _make_result("s1", match=True, classification="match"),
    ])
    mock_run_batch.return_value = retry_report

    result = asyncio.run(
        run_smart_retry(original, "http://base", "http://target")
    )
    assert result.original_divergent == 1
    assert result.stochastic_count == 1
    assert result.deterministic_count == 0
    assert result.stochastic_rate == 1.0


@patch("xpyd_acc.smart_retry.run_batch")
def test_run_smart_retry_mixed(mock_run_batch):
    """Mix of deterministic and stochastic."""
    original = _make_report([
        _make_result("s1", match=False, classification="likely_bug"),
        _make_result("s2", match=False, classification="likely_uncertainty"),
        _make_result("s3", match=True),
    ])

    retry_report = _make_report([
        _make_result("s1", match=False, classification="likely_bug"),
        _make_result("s2", match=True, classification="match"),
    ])
    mock_run_batch.return_value = retry_report

    result = asyncio.run(
        run_smart_retry(original, "http://base", "http://target")
    )
    assert result.original_divergent == 2
    assert result.deterministic_count == 1
    assert result.stochastic_count == 1
    assert result.deterministic_rate == 0.5
    assert result.stochastic_rate == 0.5
    assert len(result.per_sample) == 2


@patch("xpyd_acc.smart_retry.run_batch")
def test_run_smart_retry_greedy_params(mock_run_batch):
    """Verify greedy sampling params are passed to run_batch."""
    original = _make_report([
        _make_result("s1", match=False, classification="likely_bug"),
    ])
    retry_report = _make_report([
        _make_result("s1", match=False),
    ])
    mock_run_batch.return_value = retry_report

    asyncio.run(
        run_smart_retry(original, "http://base", "http://target")
    )

    call_kwargs = mock_run_batch.call_args[1]
    sp = call_kwargs["sampling_params"]
    assert sp.temperature == 0.0
    assert sp.seed == 42


@patch("xpyd_acc.smart_retry.run_batch")
def test_run_smart_retry_progress_callback(mock_run_batch):
    """Progress callback is forwarded to run_batch."""
    original = _make_report([
        _make_result("s1", match=False, classification="likely_bug"),
    ])
    retry_report = _make_report([
        _make_result("s1", match=False),
    ])
    mock_run_batch.return_value = retry_report

    progress_calls = []
    asyncio.run(
        run_smart_retry(
            original, "http://base", "http://target",
            on_progress=lambda c, t: progress_calls.append((c, t)),
        )
    )
    assert mock_run_batch.call_args[1]["on_progress"] is not None


# --- format_smart_retry tests ---


def test_format_smart_retry_basic():
    result = SmartRetryResult(
        original_divergent=2,
        deterministic_count=1,
        stochastic_count=1,
        deterministic_rate=0.5,
        stochastic_rate=0.5,
        per_sample=[
            SampleRetryResult("s1", "likely_bug", False, "deterministic"),
            SampleRetryResult("s2", "likely_uncertainty", True, "stochastic"),
        ],
    )
    text = format_smart_retry(result)
    assert "Smart Retry Results" in text
    assert "Deterministic (real bugs)" in text
    assert "Stochastic (sampling noise)" in text
    assert "s1" in text
    assert "s2" in text
    assert "deterministic" in text
    assert "stochastic" in text


def test_format_smart_retry_empty():
    result = SmartRetryResult(
        original_divergent=0,
        deterministic_count=0,
        stochastic_count=0,
        deterministic_rate=0.0,
        stochastic_rate=0.0,
    )
    text = format_smart_retry(result)
    assert "Original divergent samples: 0" in text


def test_smart_retry_result_json_roundtrip():
    """JSON serialization round-trip."""
    r = SmartRetryResult(
        original_divergent=2,
        deterministic_count=1,
        stochastic_count=1,
        deterministic_rate=0.5,
        stochastic_rate=0.5,
        per_sample=[
            SampleRetryResult("s1", "likely_bug", False, "deterministic"),
            SampleRetryResult("s2", "likely_uncertainty", True, "stochastic"),
        ],
    )
    j = json.loads(r.to_json())
    assert j["original_divergent"] == 2
    assert j["per_sample"][0]["retry_classification"] == "deterministic"
    assert j["per_sample"][1]["retry_classification"] == "stochastic"
