"""Grafana dashboard JSON template generation from batch reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from xpyd_acc.batch_compare import BatchReport
from xpyd_acc.log import get_logger

logger = get_logger("grafana")


@dataclass
class GrafanaDashboard:
    """Represents a Grafana dashboard configuration."""

    title: str
    datasource: str
    panels: list[dict]
    uid: str = ""
    tags: list[str] = field(default_factory=lambda: ["xpyd-acc"])
    schema_version: int = 39

    def to_dict(self) -> dict:
        """Serialize to Grafana dashboard JSON structure."""
        return {
            "uid": self.uid,
            "title": self.title,
            "tags": self.tags,
            "schemaVersion": self.schema_version,
            "editable": True,
            "templating": {
                "list": [
                    {
                        "name": "datasource",
                        "type": "datasource",
                        "query": "prometheus",
                        "current": {
                            "text": self.datasource,
                            "value": self.datasource,
                        },
                    }
                ]
            },
            "panels": self.panels,
            "time": {"from": "now-1h", "to": "now"},
            "refresh": "10s",
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def _make_divergence_rate_gauge(datasource: str, grid_pos: dict) -> dict:
    """Divergence rate gauge panel."""
    return {
        "type": "gauge",
        "title": "Divergence Rate",
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "targets": [
            {
                "expr": "xpyd_acc_divergence_rate",
                "legendFormat": "{{model}}",
                "refId": "A",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "min": 0,
                "max": 1,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 0.05},
                        {"color": "red", "value": 0.2},
                    ],
                },
                "unit": "percentunit",
            }
        },
        "gridPos": grid_pos,
        "id": 1,
    }


def _make_classification_pie(datasource: str, grid_pos: dict) -> dict:
    """Classification breakdown pie chart panel."""
    return {
        "type": "piechart",
        "title": "Classification Breakdown",
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "targets": [
            {
                "expr": "xpyd_acc_classification_count",
                "legendFormat": "{{classification}}",
                "refId": "A",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
            }
        },
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "pieType": "pie",
            "legend": {"displayMode": "list", "placement": "right"},
        },
        "gridPos": grid_pos,
        "id": 2,
    }


def _make_samples_stat(
    datasource: str, grid_pos: dict, *, metric: str, title: str, panel_id: int
) -> dict:
    """Stat panel for a single metric."""
    return {
        "type": "stat",
        "title": title,
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "targets": [
            {"expr": metric, "legendFormat": "", "refId": "A"}
        ],
        "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}}},
        "gridPos": grid_pos,
        "id": panel_id,
    }


def _make_context_length_scatter(datasource: str, grid_pos: dict) -> dict:
    """Context length vs divergence rate scatter/bar panel."""
    return {
        "type": "barchart",
        "title": "Context Length vs Divergence",
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "targets": [
            {
                "expr": "xpyd_acc_divergent_samples",
                "legendFormat": "Divergent",
                "refId": "A",
            },
            {
                "expr": "xpyd_acc_total_samples",
                "legendFormat": "Total",
                "refId": "B",
            },
        ],
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
            }
        },
        "options": {
            "orientation": "horizontal",
            "showValue": "auto",
        },
        "gridPos": grid_pos,
        "id": 5,
    }


def _make_cost_stat(datasource: str, grid_pos: dict) -> dict:
    """Cost stat panel (optional)."""
    return {
        "type": "stat",
        "title": "Total Cost (USD)",
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "targets": [
            {
                "expr": "xpyd_acc_total_cost_usd",
                "legendFormat": "",
                "refId": "A",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "currencyUSD",
                "color": {"mode": "thresholds"},
            }
        },
        "gridPos": grid_pos,
        "id": 6,
    }


def generate_dashboard(
    report: BatchReport | None = None,
    *,
    title: str = "xPyD-acc Divergence Dashboard",
    datasource: str = "Prometheus",
) -> GrafanaDashboard:
    """Generate a Grafana dashboard template.

    Args:
        report: Optional batch report (used for metadata only; panels query Prometheus).
        title: Dashboard title.
        datasource: Default Prometheus datasource name.

    Returns:
        GrafanaDashboard ready to export as JSON.
    """
    panels = [
        _make_divergence_rate_gauge(datasource, {"h": 8, "w": 8, "x": 0, "y": 0}),
        _make_classification_pie(datasource, {"h": 8, "w": 8, "x": 8, "y": 0}),
        _make_samples_stat(
            datasource,
            {"h": 4, "w": 4, "x": 16, "y": 0},
            metric="xpyd_acc_total_samples",
            title="Total Samples",
            panel_id=3,
        ),
        _make_samples_stat(
            datasource,
            {"h": 4, "w": 4, "x": 20, "y": 0},
            metric="xpyd_acc_divergent_samples",
            title="Divergent Samples",
            panel_id=4,
        ),
        _make_context_length_scatter(
            datasource, {"h": 8, "w": 12, "x": 0, "y": 8}
        ),
        _make_cost_stat(datasource, {"h": 4, "w": 4, "x": 16, "y": 4}),
    ]

    return GrafanaDashboard(
        title=title,
        datasource=datasource,
        panels=panels,
    )
