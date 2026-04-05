"""Tests for prometheus module — Prometheus text exposition format export."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from xpyd_acc.batch_compare import BatchReport, SampleResult
from xpyd_acc.cost import UsageSummary
from xpyd_acc.prometheus import _escape_label, _metric_line, push_to_gateway, to_prometheus


def _make_report(
    total: int = 10,
    divergent: int = 2,
    usage: UsageSummary | None = None,
    truncated: int = 0,
) -> BatchReport:
    results = []
    for i in range(total):
        match = i >= divergent
        cls = "match" if match else "likely_bug"
        results.append(
            SampleResult(
                sample_id=str(i),
                prompt=f"prompt {i}",
                baseline_output=f"out {i}",
                target_output=f"out {i}" if match else f"diff {i}",
                exact_match=match,
                first_divergence_index=None if match else 5,
                baseline_logprob_at_divergence=None,
                target_logprob_at_divergence=None,
                logprob_gap=None if match else 0.5,
                classification=cls,
                context_length=100,
            )
        )
    return BatchReport(
        total_samples=total,
        divergent_samples=divergent,
        match_samples=total - divergent,
        divergence_rate=divergent / total if total else 0,
        results=results,
        likely_bugs=divergent,
        likely_uncertainty=0,
        unknown_classification=0,
        usage=usage,
        truncated_count=truncated,
    )


class TestEscapeLabel:
    def test_no_escaping(self):
        assert _escape_label("hello") == "hello"

    def test_escape_quotes(self):
        assert _escape_label('say "hi"') == 'say \\"hi\\"'

    def test_escape_backslash(self):
        assert _escape_label("a\\b") == "a\\\\b"

    def test_escape_newline(self):
        assert _escape_label("a\nb") == "a\\nb"


class TestMetricLine:
    def test_no_labels(self):
        assert _metric_line("my_metric", 42) == "my_metric 42"

    def test_with_labels(self):
        result = _metric_line("m", 1.5, {"a": "x", "b": "y"})
        assert result == 'm{a="x",b="y"} 1.5'


class TestToPrometheus:
    def test_basic_metrics(self):
        report = _make_report(total=10, divergent=3)
        text = to_prometheus(report)
        assert "xpyd_acc_divergence_rate 0.3" in text
        assert "xpyd_acc_total_samples 10" in text
        assert "xpyd_acc_divergent_samples 3" in text
        assert "xpyd_acc_truncated_samples 0" in text

    def test_classification_counts(self):
        report = _make_report(total=10, divergent=2)
        text = to_prometheus(report)
        assert 'xpyd_acc_classification_count{classification="likely_bug"} 2' in text
        assert 'xpyd_acc_classification_count{classification="match"} 8' in text
        assert 'xpyd_acc_classification_count{classification="likely_uncertainty"} 0' in text
        assert 'xpyd_acc_classification_count{classification="unknown"} 0' in text

    def test_with_labels(self):
        report = _make_report()
        text = to_prometheus(report, model="gpt-4", dataset="gsm8k")
        assert 'model="gpt-4"' in text
        assert 'dataset="gsm8k"' in text

    def test_no_cost_without_usage(self):
        report = _make_report()
        text = to_prometheus(report)
        assert "xpyd_acc_total_tokens" not in text
        assert "xpyd_acc_total_cost_usd" not in text

    def test_cost_metrics_with_usage(self):
        usage = UsageSummary(
            total_prompt_tokens=500,
            total_completion_tokens=300,
            num_requests=10,
            estimated_cost_usd=0.05,
        )
        report = _make_report(usage=usage)
        text = to_prometheus(report)
        assert "xpyd_acc_total_tokens 800" in text
        assert "xpyd_acc_total_cost_usd 0.05" in text

    def test_truncated_count(self):
        report = _make_report(truncated=3)
        text = to_prometheus(report)
        assert "xpyd_acc_truncated_samples 3" in text

    def test_ends_with_newline(self):
        report = _make_report()
        text = to_prometheus(report)
        assert text.endswith("\n")

    def test_type_and_help_lines(self):
        report = _make_report()
        text = to_prometheus(report)
        assert "# TYPE xpyd_acc_divergence_rate gauge" in text
        assert "# HELP xpyd_acc_divergence_rate" in text


class TestPushToGateway:
    @patch("xpyd_acc.prometheus.urlopen")
    def test_push_calls_urlopen(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        push_to_gateway("some metrics\n", "http://localhost:9091")
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:9091/metrics/job/xpyd_acc"
        assert req.data == b"some metrics\n"

    @patch("xpyd_acc.prometheus.urlopen")
    def test_push_custom_job(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        push_to_gateway("metrics\n", "http://gw:9091/", job="custom")
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://gw:9091/metrics/job/custom"


class TestCLIIntegration:
    def test_prometheus_subcommand_registered(self):
        import argparse

        from xpyd_acc.cli.parsers import register_all
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        register_all(sub)
        args = parser.parse_args(["prometheus", "--report", "test.json"])
        assert args.command == "prometheus"
        assert args.report == "test.json"

    def test_batch_compare_prometheus_flag_registered(self):
        import argparse

        from xpyd_acc.cli.parsers import register_all
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        register_all(sub)
        args = parser.parse_args([
            "batch-compare", "--baseline", "http://a", "--target", "http://b",
            "--dataset", "d.jsonl", "--prometheus", "out.prom",
        ])
        assert args.prometheus == "out.prom"

    @patch("xpyd_acc.batch_compare.load_report")
    def test_run_prometheus_writes_file(self, mock_load, tmp_path):
        mock_load.return_value = _make_report()
        import argparse
        args = argparse.Namespace(
            report="fake.json",
            output=str(tmp_path / "metrics.prom"),
            model="test-model",
            dataset="test-ds",
            push_gateway=None,
            job="xpyd_acc",
        )
        from xpyd_acc.cli.report import _run_prometheus
        _run_prometheus(args)
        content = (tmp_path / "metrics.prom").read_text()
        assert "xpyd_acc_divergence_rate" in content
