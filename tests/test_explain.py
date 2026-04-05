"""Tests for the explain module (M52)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_acc.explain import (
    explain_sample,
    format_explain,
    load_and_explain,
)


def _make_report(*samples: dict) -> dict:
    """Build a minimal batch report dict."""
    return {
        "total_samples": len(samples),
        "divergent_samples": sum(1 for s in samples if not s.get("exact_match", True)),
        "match_samples": sum(1 for s in samples if s.get("exact_match", True)),
        "divergence_rate": 0.0,
        "results": list(samples),
    }


_MATCH_SAMPLE = {
    "sample_id": "s1",
    "prompt": "What is 2+2?",
    "baseline_output": "The answer is 4.",
    "target_output": "The answer is 4.",
    "exact_match": True,
    "first_divergence_index": None,
    "baseline_logprob_at_divergence": None,
    "target_logprob_at_divergence": None,
    "logprob_gap": None,
    "classification": "match",
    "context_length": 5,
    "request_ids": {},
}

_BUG_SAMPLE = {
    "sample_id": "s2",
    "prompt": "Solve x^2 = 9",
    "baseline_output": "x = 3 or x = -3",
    "target_output": "x = 3 or x = 3",
    "exact_match": False,
    "first_divergence_index": 5,
    "baseline_logprob_at_divergence": -0.01,
    "target_logprob_at_divergence": -2.5,
    "logprob_gap": 2.0,
    "classification": "likely_bug",
    "context_length": 10,
    "request_ids": {"baseline": "abc", "target": "def"},
}

_UNCERTAIN_SAMPLE = {
    "sample_id": "s3",
    "prompt": "Tell me a joke",
    "baseline_output": "Why did the chicken cross the road",
    "target_output": "Why did the duck cross the road",
    "exact_match": False,
    "first_divergence_index": 3,
    "baseline_logprob_at_divergence": -1.2,
    "target_logprob_at_divergence": -1.3,
    "logprob_gap": 0.05,
    "classification": "likely_uncertainty",
    "context_length": 8,
    "request_ids": {},
}


class TestExplainSample:
    """Tests for explain_sample()."""

    def test_match_sample(self) -> None:
        report = _make_report(_MATCH_SAMPLE)
        result = explain_sample(report, "s1")
        assert result.exact_match is True
        assert result.classification == "match"
        assert result.divergence_context is None
        assert "identical" in result.classification_reasoning.lower()
        assert len(result.suggested_next_steps) >= 1

    def test_bug_sample(self) -> None:
        report = _make_report(_MATCH_SAMPLE, _BUG_SAMPLE)
        result = explain_sample(report, "s2")
        assert result.exact_match is False
        assert result.classification == "likely_bug"
        assert result.first_divergence_index == 5
        assert result.logprob_gap == 2.0
        assert result.divergence_context is not None
        assert result.divergence_context.at.index == 5
        assert "likely_bug" in result.classification_reasoning
        assert any("KV cache" in s for s in result.suggested_next_steps)

    def test_uncertainty_sample(self) -> None:
        report = _make_report(_UNCERTAIN_SAMPLE)
        result = explain_sample(report, "s3")
        assert result.classification == "likely_uncertainty"
        assert "likely_uncertainty" in result.classification_reasoning
        assert any("temperature=0" in s for s in result.suggested_next_steps)

    def test_missing_sample_raises_key_error(self) -> None:
        report = _make_report(_MATCH_SAMPLE)
        with pytest.raises(KeyError, match="not_exist"):
            explain_sample(report, "not_exist")

    def test_divergence_context_window(self) -> None:
        report = _make_report(_BUG_SAMPLE)
        result = explain_sample(report, "s2")
        ctx = result.divergence_context
        assert ctx is not None
        # Before tokens should be indices 0..4
        assert len(ctx.before) == 5
        assert all(tp.match for tp in ctx.before)
        # Token 5 in both outputs is "=" — the metadata says divergence is at 5
        # (the batch_compare module sets this, explain just uses it)
        assert ctx.at.index == 5
        assert ctx.at.baseline == "="
        assert ctx.at.target == "="

    def test_divergence_at_token_zero(self) -> None:
        sample = {
            **_BUG_SAMPLE,
            "sample_id": "s4",
            "first_divergence_index": 0,
            "baseline_output": "yes it is",
            "target_output": "no it is",
        }
        report = _make_report(sample)
        result = explain_sample(report, "s4")
        assert result.divergence_context is not None
        assert len(result.divergence_context.before) == 0
        assert result.divergence_context.at.index == 0
        assert any("token 0" in s for s in result.suggested_next_steps)

    def test_unknown_classification(self) -> None:
        sample = {
            **_BUG_SAMPLE,
            "sample_id": "s5",
            "classification": "unknown",
            "logprob_gap": None,
        }
        report = _make_report(sample)
        result = explain_sample(report, "s5")
        assert "unknown" in result.classification_reasoning.lower()


class TestLoadAndExplain:
    """Tests for load_and_explain()."""

    def test_load_from_file(self, tmp_path: Path) -> None:
        report = _make_report(_MATCH_SAMPLE, _BUG_SAMPLE)
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report))
        result = load_and_explain(str(report_path), "s2")
        assert result.sample_id == "s2"
        assert result.exact_match is False

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_and_explain("/nonexistent/path.json", "s1")


class TestFormatExplain:
    """Tests for format_explain()."""

    def test_format_match(self) -> None:
        report = _make_report(_MATCH_SAMPLE)
        result = explain_sample(report, "s1")
        output = format_explain(result)
        assert "s1" in output
        assert "identical" in output.lower()

    def test_format_divergent(self) -> None:
        report = _make_report(_BUG_SAMPLE)
        result = explain_sample(report, "s2")
        output = format_explain(result)
        assert "s2" in output
        assert "likely_bug" in output
        assert "Token Context" in output
        assert "▶" in output  # divergence marker


class TestExplainJsonExport:
    """Tests for JSON serialization."""

    def test_to_json_roundtrip(self) -> None:
        report = _make_report(_BUG_SAMPLE)
        result = explain_sample(report, "s2")
        data = json.loads(result.to_json())
        assert data["sample_id"] == "s2"
        assert data["classification"] == "likely_bug"
        assert data["divergence_context"] is not None
        assert data["divergence_context"]["at"]["index"] == 5


class TestExplainCli:
    """Tests for CLI integration."""

    def test_explain_cli(self, tmp_path: Path) -> None:
        from xpyd_acc.cli import main

        report = _make_report(_MATCH_SAMPLE, _BUG_SAMPLE)
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report))
        json_out = tmp_path / "explain.json"

        main(["explain", "--report", str(report_path), "--sample", "s2", "--json", str(json_out)])

        data = json.loads(json_out.read_text())
        assert data["sample_id"] == "s2"

    def test_explain_cli_missing_sample(self, tmp_path: Path) -> None:
        from xpyd_acc.cli import main

        report = _make_report(_MATCH_SAMPLE)
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report))

        with pytest.raises(SystemExit):
            main(["explain", "--report", str(report_path), "--sample", "nope"])

    def test_explain_cli_missing_file(self) -> None:
        from xpyd_acc.cli import main

        with pytest.raises(SystemExit):
            main(["explain", "--report", "/nonexistent.json", "--sample", "s1"])
