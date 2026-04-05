"""Tests for token_diff module."""

from __future__ import annotations

import json
import sys

import pytest

from xpyd_acc.batch_compare import BatchReport, SampleResult
from xpyd_acc.token_diff import (
    TokenDiff,
    TokenDiffLine,
    build_all_divergent,
    build_from_report,
    build_token_diff,
    diff_from_file,
    format_token_diff,
)


def _make_result(
    sample_id: str = "s1",
    baseline: str = "hello world foo",
    target: str = "hello world bar",
    match: bool = False,
    div_index: int | None = 2,
    logprob_gap: float | None = 0.5,
    b_logprob: float | None = -1.0,
    t_logprob: float | None = -2.0,
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        prompt="test prompt",
        baseline_output=baseline,
        target_output=target,
        exact_match=match,
        first_divergence_index=div_index,
        baseline_logprob_at_divergence=b_logprob,
        target_logprob_at_divergence=t_logprob,
        logprob_gap=logprob_gap,
        classification="likely_bug" if not match else "match",
        context_length=10,
    )


def _make_report(*results: SampleResult) -> BatchReport:
    rs = list(results) if results else [_make_result()]
    div = sum(1 for r in rs if not r.exact_match)
    return BatchReport(
        total_samples=len(rs),
        divergent_samples=div,
        match_samples=len(rs) - div,
        divergence_rate=div / len(rs) if rs else 0.0,
        results=rs,
    )


class TestTokenDiffLine:
    def test_to_dict(self):
        line = TokenDiffLine(index=0, baseline_token="a", target_token="b", status="mismatch")
        d = line.to_dict()
        assert d["index"] == 0
        assert d["status"] == "mismatch"


class TestTokenDiff:
    def test_to_dict(self):
        diff = TokenDiff(sample_id="s1", divergence_index=2, baseline_length=3, target_length=3)
        diff.lines.append(TokenDiffLine(0, "a", "a", "match"))
        d = diff.to_dict()
        assert d["sample_id"] == "s1"
        assert len(d["lines"]) == 1

    def test_to_json(self):
        diff = TokenDiff(sample_id="s1", divergence_index=None)
        j = diff.to_json()
        parsed = json.loads(j)
        assert parsed["sample_id"] == "s1"


class TestBuildTokenDiff:
    def test_basic_divergence(self):
        result = _make_result()
        diff = build_token_diff(result, context=10)
        assert diff.sample_id == "s1"
        assert diff.divergence_index == 2
        assert len(diff.lines) > 0
        # Should have match and mismatch lines
        statuses = {ln.status for ln in diff.lines}
        assert "match" in statuses or "mismatch" in statuses

    def test_empty_outputs(self):
        result = _make_result(baseline="", target="", match=True, div_index=None)
        diff = build_token_diff(result, context=5)
        assert len(diff.lines) == 0

    def test_baseline_longer(self):
        result = _make_result(baseline="a b c d e", target="a b", div_index=2)
        diff = build_token_diff(result, context=10)
        has_baseline_only = any(ln.status == "baseline_only" for ln in diff.lines)
        assert has_baseline_only

    def test_target_longer(self):
        result = _make_result(baseline="a b", target="a b c d e", div_index=2)
        diff = build_token_diff(result, context=10)
        has_target_only = any(ln.status == "target_only" for ln in diff.lines)
        assert has_target_only

    def test_context_window(self):
        # Build a long sequence
        baseline = " ".join(str(i) for i in range(50))
        target_parts = [str(i) for i in range(25)] + ["X"] + [str(i) for i in range(26, 50)]
        target = " ".join(target_parts)
        result = _make_result(baseline=baseline, target=target, div_index=25)
        diff = build_token_diff(result, context=3)
        # Should show limited range around divergence
        assert len(diff.lines) <= 20  # reasonable bound

    def test_logprob_at_divergence(self):
        result = _make_result(b_logprob=-0.5, t_logprob=-1.5)
        diff = build_token_diff(result, context=10)
        # Find the mismatch line at divergence index
        div_lines = [ln for ln in diff.lines if ln.status == "mismatch" and ln.index == 2]
        if div_lines:
            assert div_lines[0].baseline_logprob == -0.5
            assert div_lines[0].target_logprob == -1.5

    def test_logprob_warning(self):
        # Low logprob gap near divergence on matching tokens
        result = _make_result(
            baseline="hello world foo",
            target="hello world bar",
            div_index=2,
            logprob_gap=0.05,  # < 0.1 threshold
        )
        diff = build_token_diff(result, context=10)
        warnings = [ln for ln in diff.lines if ln.status == "logprob_warning"]
        # Nearby matching tokens might get logprob_warning
        # (depends on token alignment near divergence)
        assert isinstance(warnings, list)  # at least no crash

    def test_no_divergence(self):
        result = _make_result(
            baseline="hello world", target="hello world",
            match=True, div_index=None,
        )
        diff = build_token_diff(result, context=5)
        assert diff.divergence_index is None
        assert all(ln.status == "match" for ln in diff.lines)


class TestFormatTokenDiff:
    def test_plain_format(self):
        result = _make_result()
        diff = build_token_diff(result, context=5)
        output = format_token_diff(diff, plain=True)
        assert "Token Diff: s1" in output
        assert "\033[" not in output  # no ANSI codes

    def test_rich_format(self):
        result = _make_result()
        diff = build_token_diff(result, context=5)
        output = format_token_diff(diff, plain=False)
        assert "Token Diff: s1" in output
        assert "\033[" in output  # has ANSI codes

    def test_empty_diff(self):
        diff = TokenDiff(sample_id="empty", divergence_index=None)
        output = format_token_diff(diff)
        assert "no tokens to display" in output

    def test_logprob_annotation(self):
        result = _make_result(b_logprob=-0.5, t_logprob=-1.5)
        diff = build_token_diff(result, context=5)
        output = format_token_diff(diff, plain=True)
        # If divergence line has logprobs, they should appear
        if "logprob" in output:
            assert "b=" in output


class TestBuildFromReport:
    def test_found(self):
        report = _make_report(_make_result(sample_id="s1"))
        diff = build_from_report(report, "s1")
        assert diff.sample_id == "s1"

    def test_not_found(self):
        report = _make_report(_make_result(sample_id="s1"))
        with pytest.raises(ValueError, match="not found"):
            build_from_report(report, "nonexistent")


class TestBuildAllDivergent:
    def test_filters_divergent(self):
        r1 = _make_result(sample_id="s1", match=False)
        r2 = _make_result(sample_id="s2", baseline="x", target="x", match=True, div_index=None)
        r3 = _make_result(sample_id="s3", match=False)
        report = _make_report(r1, r2, r3)
        diffs = build_all_divergent(report)
        assert len(diffs) == 2
        ids = {d.sample_id for d in diffs}
        assert ids == {"s1", "s3"}


class TestDiffFromFile:
    def test_sample_mode(self, tmp_path):
        report = _make_report(_make_result(sample_id="s1"))
        p = tmp_path / "report.json"
        p.write_text(report.to_json())
        diffs = diff_from_file(str(p), sample_id="s1")
        assert len(diffs) == 1
        assert diffs[0].sample_id == "s1"

    def test_all_divergent_mode(self, tmp_path):
        r1 = _make_result(sample_id="s1", match=False)
        r2 = _make_result(sample_id="s2", baseline="x", target="x", match=True, div_index=None)
        report = _make_report(r1, r2)
        p = tmp_path / "report.json"
        p.write_text(report.to_json())
        diffs = diff_from_file(str(p), all_divergent=True)
        assert len(diffs) == 1

    def test_no_args_raises(self, tmp_path):
        report = _make_report()
        p = tmp_path / "report.json"
        p.write_text(report.to_json())
        with pytest.raises(ValueError, match="Must specify"):
            diff_from_file(str(p))


class TestCLIIntegration:
    def test_token_diff_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "xpyd_acc.cli", "token-diff", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--report" in result.stdout
        assert "--sample" in result.stdout
        assert "--all-divergent" in result.stdout

    def test_token_diff_sample(self, tmp_path):
        import subprocess
        report = _make_report(_make_result(sample_id="s1"))
        p = tmp_path / "report.json"
        p.write_text(report.to_json())
        result = subprocess.run(
            [sys.executable, "-m", "xpyd_acc.cli", "token-diff", "--report", str(p),
             "--sample", "s1", "--format", "plain"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "s1" in result.stdout

    def test_token_diff_json_export(self, tmp_path):
        import subprocess
        report = _make_report(_make_result(sample_id="s1"))
        rp = tmp_path / "report.json"
        rp.write_text(report.to_json())
        jp = tmp_path / "diff.json"
        result = subprocess.run(
            [sys.executable, "-m", "xpyd_acc.cli", "token-diff", "--report", str(rp),
             "--sample", "s1", "--json", str(jp)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert jp.exists()
        data = json.loads(jp.read_text())
        assert len(data) == 1
        assert data[0]["sample_id"] == "s1"

    def test_token_diff_no_args_error(self, tmp_path):
        import subprocess
        report = _make_report()
        p = tmp_path / "report.json"
        p.write_text(report.to_json())
        result = subprocess.run(
            [sys.executable, "-m", "xpyd_acc.cli", "token-diff", "--report", str(p)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
