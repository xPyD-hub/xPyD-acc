"""Tests for serve module — report dashboard server."""

from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

import pytest

from xpyd_acc.batch_compare import BatchReport, SampleResult
from xpyd_acc.serve import (
    create_handler,
    render_dashboard,
    report_to_api_dict,
    run_server,
)


def _make_result(
    sample_id: str = "s1",
    match: bool = False,
    classification: str = "likely_bug",
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        prompt="test prompt",
        baseline_output="hello world",
        target_output="hello earth" if not match else "hello world",
        exact_match=match,
        first_divergence_index=1 if not match else None,
        baseline_logprob_at_divergence=-1.0 if not match else None,
        target_logprob_at_divergence=-2.0 if not match else None,
        logprob_gap=0.5 if not match else None,
        classification="match" if match else classification,
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


def _write_report(path: Path, report: BatchReport | None = None) -> Path:
    report = report or _make_report()
    path.write_text(report.to_json())
    return path


class TestRenderDashboard:
    def test_returns_html(self):
        html = render_dashboard()
        assert "<!DOCTYPE html>" in html
        assert "xPyD-acc Report Dashboard" in html

    def test_contains_app_div(self):
        html = render_dashboard()
        assert 'id="app"' in html

    def test_contains_fetch_script(self):
        html = render_dashboard()
        assert "/api/report" in html
        assert "fetchReport" in html


class TestReportToApiDict:
    def test_basic_conversion(self):
        report = _make_report()
        d = report_to_api_dict(report)
        assert d["total_samples"] == 1
        assert d["divergent_samples"] == 1
        assert len(d["results"]) == 1

    def test_results_fields(self):
        report = _make_report()
        d = report_to_api_dict(report)
        r = d["results"][0]
        assert r["sample_id"] == "s1"
        assert r["exact_match"] is False
        assert r["classification"] == "likely_bug"


class TestCreateHandler:
    def test_creates_bound_handler(self):
        handler_cls = create_handler("/some/path.json")
        assert handler_cls.report_path == "/some/path.json"

    def test_different_paths_different_classes(self):
        h1 = create_handler("/path1.json")
        h2 = create_handler("/path2.json")
        assert h1.report_path != h2.report_path


class TestServerIntegration:
    """Integration tests that start a real HTTP server."""

    @pytest.fixture()
    def report_file(self, tmp_path: Path) -> Path:
        return _write_report(tmp_path / "report.json")

    @pytest.fixture()
    def server_url(self, report_file: Path):
        """Start server on a free port, yield URL, then shut down."""
        server = run_server(str(report_file), host="127.0.0.1", port=0)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        yield f"127.0.0.1:{port}"
        server.shutdown()
        server.server_close()

    def test_dashboard_route(self, server_url: str):
        conn = HTTPConnection(server_url, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        body = resp.read().decode()
        assert "xPyD-acc Report Dashboard" in body
        conn.close()

    def test_api_report_route(self, server_url: str):
        conn = HTTPConnection(server_url, timeout=5)
        conn.request("GET", "/api/report")
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "report" in data
        assert "mtime" in data
        assert data["report"]["total_samples"] == 1
        conn.close()

    def test_404_unknown_route(self, server_url: str):
        conn = HTTPConnection(server_url, timeout=5)
        conn.request("GET", "/unknown")
        resp = conn.getresponse()
        assert resp.status == 404
        conn.close()

    def test_auto_refresh_on_file_change(self, server_url: str, report_file: Path):
        conn = HTTPConnection(server_url, timeout=5)
        conn.request("GET", "/api/report")
        data1 = json.loads(conn.getresponse().read())
        conn.close()

        # Write updated report
        new_report = _make_report(
            _make_result("s1", match=True),
            _make_result("s2", match=False),
        )
        time.sleep(0.1)  # ensure mtime changes
        report_file.write_text(new_report.to_json())

        conn = HTTPConnection(server_url, timeout=5)
        conn.request("GET", "/api/report")
        data2 = json.loads(conn.getresponse().read())
        conn.close()

        assert data2["report"]["total_samples"] == 2
        assert data2["mtime"] != data1["mtime"]


class TestRunServerValidation:
    def test_invalid_report_path(self, tmp_path: Path):
        with pytest.raises(Exception):
            run_server(str(tmp_path / "nonexistent.json"))

    def test_open_browser_flag(self, tmp_path: Path):
        _write_report(tmp_path / "report.json")
        with patch("xpyd_acc.serve.webbrowser") as mock_wb:
            server = run_server(
                str(tmp_path / "report.json"),
                host="127.0.0.1",
                port=0,
                open_browser=True,
            )
            mock_wb.open.assert_called_once()
            server.server_close()


class TestCLIIntegration:
    def test_serve_registered(self):
        """Verify serve subcommand is registered in CLI."""
        from xpyd_acc.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["serve", "--help"])
        assert exc_info.value.code == 0
