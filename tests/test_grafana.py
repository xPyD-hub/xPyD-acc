"""Tests for Grafana dashboard template generation."""

from __future__ import annotations

import json
from pathlib import Path

from xpyd_acc.grafana import generate_dashboard


def _make_report():
    """Create a minimal BatchReport for testing."""
    from xpyd_acc.batch_compare import BatchReport

    return BatchReport(
        total_samples=100,
        divergent_samples=5,
        match_samples=95,
        divergence_rate=0.05,
        results=[],
    )


class TestGrafanaDashboard:
    def test_default_dashboard_has_panels(self):
        dashboard = generate_dashboard()
        assert len(dashboard.panels) == 6

    def test_default_title(self):
        dashboard = generate_dashboard()
        assert dashboard.title == "xPyD-acc Divergence Dashboard"

    def test_custom_title(self):
        dashboard = generate_dashboard(title="My Dashboard")
        assert dashboard.title == "My Dashboard"

    def test_default_datasource(self):
        dashboard = generate_dashboard()
        assert dashboard.datasource == "Prometheus"

    def test_custom_datasource(self):
        dashboard = generate_dashboard(datasource="MyProm")
        assert dashboard.datasource == "MyProm"
        d = dashboard.to_dict()
        tpl = d["templating"]["list"][0]
        assert tpl["current"]["text"] == "MyProm"

    def test_to_json_valid(self):
        dashboard = generate_dashboard()
        raw = dashboard.to_json()
        parsed = json.loads(raw)
        assert "panels" in parsed
        assert parsed["title"] == "xPyD-acc Divergence Dashboard"

    def test_to_dict_schema_version(self):
        dashboard = generate_dashboard()
        d = dashboard.to_dict()
        assert d["schemaVersion"] == 39

    def test_to_dict_tags(self):
        dashboard = generate_dashboard()
        d = dashboard.to_dict()
        assert "xpyd-acc" in d["tags"]

    def test_to_dict_templating(self):
        dashboard = generate_dashboard()
        d = dashboard.to_dict()
        tpl_list = d["templating"]["list"]
        assert len(tpl_list) == 1
        assert tpl_list[0]["type"] == "datasource"

    def test_panel_types(self):
        dashboard = generate_dashboard()
        types = [p["type"] for p in dashboard.panels]
        assert "gauge" in types
        assert "piechart" in types
        assert "stat" in types
        assert "barchart" in types

    def test_divergence_gauge_thresholds(self):
        dashboard = generate_dashboard()
        gauge = [p for p in dashboard.panels if p["type"] == "gauge"][0]
        steps = gauge["fieldConfig"]["defaults"]["thresholds"]["steps"]
        assert len(steps) == 3
        assert steps[1]["value"] == 0.05
        assert steps[2]["value"] == 0.2

    def test_with_report(self):
        report = _make_report()
        dashboard = generate_dashboard(report, title="Test")
        assert dashboard.title == "Test"
        assert len(dashboard.panels) == 6

    def test_panels_have_ids(self):
        dashboard = generate_dashboard()
        ids = [p["id"] for p in dashboard.panels]
        assert len(ids) == len(set(ids)), "Panel IDs must be unique"

    def test_panels_have_grid_pos(self):
        dashboard = generate_dashboard()
        for panel in dashboard.panels:
            gp = panel["gridPos"]
            assert "h" in gp and "w" in gp and "x" in gp and "y" in gp

    def test_datasource_uid_uses_variable(self):
        dashboard = generate_dashboard()
        for panel in dashboard.panels:
            ds = panel.get("datasource", {})
            if isinstance(ds, dict) and "uid" in ds:
                assert ds["uid"] == "${datasource}"


class TestGrafanaCLI:
    def test_cli_creates_file(self, tmp_path: Path):
        """Test CLI writes dashboard JSON file."""
        from argparse import Namespace

        from xpyd_acc.cli.report import _run_grafana_dashboard

        out = tmp_path / "dashboard.json"
        args = Namespace(
            report=None,
            output=str(out),
            datasource="Prometheus",
            title="CLI Test",
        )
        _run_grafana_dashboard(args)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["title"] == "CLI Test"

    def test_cli_with_report(self, tmp_path: Path):
        """Test CLI works with a report file."""
        from argparse import Namespace

        from xpyd_acc.cli.report import _run_grafana_dashboard

        report = _make_report()
        report_path = tmp_path / "report.json"
        report_path.write_text(report.to_json(), encoding="utf-8")

        out = tmp_path / "dashboard.json"
        args = Namespace(
            report=str(report_path),
            output=str(out),
            datasource="MyDS",
            title="Report Test",
        )
        _run_grafana_dashboard(args)
        data = json.loads(out.read_text())
        assert data["title"] == "Report Test"
        tpl = data["templating"]["list"][0]
        assert tpl["current"]["text"] == "MyDS"

    def test_cli_integration_via_main(self, tmp_path: Path):
        """Test the subcommand is registered and parses args."""
        import argparse as _ap

        from xpyd_acc.cli.parsers import register_all

        parser = _ap.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        register_all(sub)
        out = tmp_path / "dash.json"
        args = parser.parse_args(["grafana-dashboard", "--output", str(out)])
        assert args.command == "grafana-dashboard"
        assert args.output == str(out)
        assert args.datasource == "Prometheus"
