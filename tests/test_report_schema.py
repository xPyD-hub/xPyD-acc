"""Tests for report schema versioning and load_report()."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_acc.batch_compare import (
    REPORT_SCHEMA_VERSION,
    BatchReport,
    SampleResult,
    load_report,
)


def _make_sample(**overrides: object) -> SampleResult:
    defaults = {
        "sample_id": "s1",
        "prompt": "hello",
        "baseline_output": "world",
        "target_output": "world",
        "exact_match": True,
        "first_divergence_index": None,
        "baseline_logprob_at_divergence": None,
        "target_logprob_at_divergence": None,
        "logprob_gap": None,
        "classification": "match",
        "context_length": 5,
    }
    defaults.update(overrides)
    return SampleResult(**defaults)  # type: ignore[arg-type]


def _make_report(samples: list[SampleResult] | None = None) -> BatchReport:
    samples = samples or [_make_sample()]
    return BatchReport(
        total_samples=len(samples),
        divergent_samples=0,
        match_samples=len(samples),
        divergence_rate=0.0,
        results=samples,
    )


# -- to_json includes schema_version ------------------------------------------

def test_to_json_includes_schema_version() -> None:
    report = _make_report()
    data = json.loads(report.to_json())
    assert data["schema_version"] == REPORT_SCHEMA_VERSION


def test_schema_version_is_positive_int() -> None:
    assert isinstance(REPORT_SCHEMA_VERSION, int)
    assert REPORT_SCHEMA_VERSION >= 1


# -- load_report round-trip ----------------------------------------------------

def test_load_report_round_trip(tmp_path: Path) -> None:
    report = _make_report([_make_sample(), _make_sample(sample_id="s2", exact_match=False,
                                                         classification="likely_bug")])
    report.divergent_samples = 1
    report.match_samples = 1
    report.divergence_rate = 0.5
    report.likely_bugs = 1
    path = tmp_path / "report.json"
    path.write_text(report.to_json())

    loaded = load_report(path)
    assert loaded.total_samples == 2
    assert loaded.divergent_samples == 1
    assert loaded.divergence_rate == 0.5
    assert loaded.likely_bugs == 1
    assert len(loaded.results) == 2
    assert loaded.results[0].sample_id == "s1"
    assert loaded.results[1].classification == "likely_bug"


# -- backward compat: missing schema_version treated as v1 --------------------

def test_load_report_missing_version(tmp_path: Path) -> None:
    """Reports produced before schema versioning have no schema_version field."""
    data = json.loads(_make_report().to_json())
    del data["schema_version"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data))

    loaded = load_report(path)
    assert loaded.total_samples == 1


# -- future version raises -----------------------------------------------------

def test_load_report_future_version(tmp_path: Path) -> None:
    data = json.loads(_make_report().to_json())
    data["schema_version"] = REPORT_SCHEMA_VERSION + 1
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="newer than supported"):
        load_report(path)


# -- missing optional fields default gracefully --------------------------------

def test_load_report_missing_optional_fields(tmp_path: Path) -> None:
    """Minimal JSON with only required fields should load without error."""
    minimal: dict = {
        "total_samples": 1,
        "divergent_samples": 0,
        "match_samples": 1,
        "divergence_rate": 0.0,
        "results": [
            {
                "sample_id": "s1",
                "prompt": "hi",
                "baseline_output": "ho",
                "target_output": "ho",
                "exact_match": True,
            }
        ],
    }
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps(minimal))

    loaded = load_report(path)
    assert loaded.total_samples == 1
    assert loaded.results[0].classification == "unknown"
    assert loaded.results[0].context_length == 0
    assert loaded.results[0].request_ids == {}


# -- confidence interval fields round-trip -------------------------------------

def test_load_report_confidence_fields(tmp_path: Path) -> None:
    report = _make_report()
    report.divergence_ci_lower = 0.01
    report.divergence_ci_upper = 0.15
    report.confidence_level = 0.95
    path = tmp_path / "ci.json"
    path.write_text(report.to_json())

    loaded = load_report(path)
    assert loaded.divergence_ci_lower == 0.01
    assert loaded.divergence_ci_upper == 0.15
    assert loaded.confidence_level == 0.95


# -- request_ids round-trip ----------------------------------------------------

def test_load_report_request_ids(tmp_path: Path) -> None:
    sample = _make_sample()
    sample.request_ids = {"baseline": "abc-123", "target": "def-456"}
    report = _make_report([sample])
    path = tmp_path / "rid.json"
    path.write_text(report.to_json())

    loaded = load_report(path)
    assert loaded.results[0].request_ids == {"baseline": "abc-123", "target": "def-456"}
