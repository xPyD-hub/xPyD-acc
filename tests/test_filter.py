"""Tests for sample filtering (M42)."""

from __future__ import annotations

import json
from pathlib import Path

from xpyd_acc.filter import FilterConfig, filter_samples, load_report, save_report


def _make_report(samples: list[dict]) -> dict:
    total = len(samples)
    divergent = sum(1 for s in samples if not s.get("exact_match", True))
    return {
        "total_samples": total,
        "divergent_samples": divergent,
        "match_samples": total - divergent,
        "divergence_rate": divergent / total if total else 0.0,
        "samples": samples,
    }


def _sample(
    sid: str = "s1",
    exact_match: bool = True,
    classification: str = "match",
    logprob_gap: float | None = None,
    context_length: int = 100,
    prompt: str = "hello",
    baseline_output: str = "world",
    target_output: str = "world",
) -> dict:
    return {
        "sample_id": sid,
        "prompt": prompt,
        "baseline_output": baseline_output,
        "target_output": target_output,
        "exact_match": exact_match,
        "classification": classification,
        "logprob_gap": logprob_gap,
        "context_length": context_length,
    }


class TestFilterSamples:
    def test_no_filters(self) -> None:
        report = _make_report([_sample(), _sample(sid="s2")])
        result = filter_samples(report, FilterConfig())
        assert result["total_samples"] == 2

    def test_divergent_only(self) -> None:
        report = _make_report([
            _sample(sid="s1", exact_match=True),
            _sample(sid="s2", exact_match=False, classification="likely_bug"),
        ])
        result = filter_samples(report, FilterConfig(divergent_only=True))
        assert result["total_samples"] == 1
        assert result["samples"][0]["sample_id"] == "s2"
        assert result["divergent_samples"] == 1
        assert result["divergence_rate"] == 1.0

    def test_matched_only(self) -> None:
        report = _make_report([
            _sample(sid="s1", exact_match=True),
            _sample(sid="s2", exact_match=False),
        ])
        result = filter_samples(report, FilterConfig(matched_only=True))
        assert result["total_samples"] == 1
        assert result["samples"][0]["sample_id"] == "s1"

    def test_classification_filter(self) -> None:
        report = _make_report([
            _sample(sid="s1", classification="likely_bug"),
            _sample(sid="s2", classification="likely_uncertainty"),
            _sample(sid="s3", classification="likely_bug"),
        ])
        result = filter_samples(report, FilterConfig(classification="likely_bug"))
        assert result["total_samples"] == 2
        assert all(s["classification"] == "likely_bug" for s in result["samples"])

    def test_min_logprob_gap(self) -> None:
        report = _make_report([
            _sample(sid="s1", logprob_gap=0.1),
            _sample(sid="s2", logprob_gap=0.5),
            _sample(sid="s3", logprob_gap=None),
        ])
        result = filter_samples(report, FilterConfig(min_logprob_gap=0.3))
        assert result["total_samples"] == 1
        assert result["samples"][0]["sample_id"] == "s2"

    def test_max_logprob_gap(self) -> None:
        report = _make_report([
            _sample(sid="s1", logprob_gap=0.1),
            _sample(sid="s2", logprob_gap=0.5),
        ])
        result = filter_samples(report, FilterConfig(max_logprob_gap=0.3))
        assert result["total_samples"] == 1
        assert result["samples"][0]["sample_id"] == "s1"

    def test_min_context_length(self) -> None:
        report = _make_report([
            _sample(sid="s1", context_length=50),
            _sample(sid="s2", context_length=200),
        ])
        result = filter_samples(report, FilterConfig(min_context_length=100))
        assert result["total_samples"] == 1
        assert result["samples"][0]["sample_id"] == "s2"

    def test_max_context_length(self) -> None:
        report = _make_report([
            _sample(sid="s1", context_length=50),
            _sample(sid="s2", context_length=200),
        ])
        result = filter_samples(report, FilterConfig(max_context_length=100))
        assert result["total_samples"] == 1
        assert result["samples"][0]["sample_id"] == "s1"

    def test_search_in_prompt(self) -> None:
        report = _make_report([
            _sample(sid="s1", prompt="What is Python?"),
            _sample(sid="s2", prompt="Tell me about Rust"),
        ])
        result = filter_samples(report, FilterConfig(search="python"))
        assert result["total_samples"] == 1
        assert result["samples"][0]["sample_id"] == "s1"

    def test_search_in_output(self) -> None:
        report = _make_report([
            _sample(sid="s1", baseline_output="Python is great"),
            _sample(sid="s2", baseline_output="Rust is fast"),
        ])
        result = filter_samples(report, FilterConfig(search="GREAT"))
        assert result["total_samples"] == 1
        assert result["samples"][0]["sample_id"] == "s1"

    def test_search_in_target_output(self) -> None:
        report = _make_report([
            _sample(sid="s1", target_output="error occurred"),
            _sample(sid="s2", target_output="success"),
        ])
        result = filter_samples(report, FilterConfig(search="error"))
        assert result["total_samples"] == 1

    def test_combined_filters(self) -> None:
        report = _make_report([
            _sample(
                sid="s1", exact_match=False, classification="likely_bug",
                logprob_gap=0.8, context_length=200,
            ),
            _sample(
                sid="s2", exact_match=False, classification="likely_bug",
                logprob_gap=0.1, context_length=200,
            ),
            _sample(
                sid="s3", exact_match=False,
                classification="likely_uncertainty",
                logprob_gap=0.8, context_length=200,
            ),
            _sample(
                sid="s4", exact_match=True, classification="match",
                logprob_gap=None, context_length=50,
            ),
        ])
        result = filter_samples(report, FilterConfig(
            divergent_only=True,
            classification="likely_bug",
            min_logprob_gap=0.5,
        ))
        assert result["total_samples"] == 1
        assert result["samples"][0]["sample_id"] == "s1"

    def test_empty_result(self) -> None:
        report = _make_report([_sample()])
        result = filter_samples(report, FilterConfig(divergent_only=True))
        assert result["total_samples"] == 0
        assert result["divergence_rate"] == 0.0

    def test_recalculates_stats(self) -> None:
        report = _make_report([
            _sample(sid="s1", exact_match=True),
            _sample(sid="s2", exact_match=False),
            _sample(sid="s3", exact_match=False),
        ])
        result = filter_samples(report, FilterConfig(divergent_only=True))
        assert result["total_samples"] == 2
        assert result["divergent_samples"] == 2
        assert result["match_samples"] == 0
        assert result["divergence_rate"] == 1.0


class TestLoadSaveReport:
    def test_round_trip(self, tmp_path: Path) -> None:
        report = _make_report([_sample()])
        path = tmp_path / "report.json"
        save_report(report, path)
        loaded = load_report(path)
        assert loaded["total_samples"] == 1
        assert loaded["samples"][0]["sample_id"] == "s1"


class TestCLIIntegration:
    def test_filter_cli(self, tmp_path: Path) -> None:
        """Test filter subcommand via CLI."""
        from xpyd_acc.cli import main

        report = _make_report([
            _sample(sid="s1", exact_match=True),
            _sample(sid="s2", exact_match=False, classification="likely_bug"),
        ])
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        with open(input_path, "w") as f:
            json.dump(report, f)

        main([
            "filter", "--input", str(input_path),
            "--output", str(output_path), "--divergent-only",
        ])

        with open(output_path) as f:
            result = json.load(f)
        assert result["total_samples"] == 1
        assert result["samples"][0]["sample_id"] == "s2"
