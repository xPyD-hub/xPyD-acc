"""Prometheus text exposition format export for batch reports."""

from __future__ import annotations

from urllib.request import Request, urlopen

from xpyd_acc.batch_compare import BatchReport
from xpyd_acc.log import get_logger

logger = get_logger("prometheus")


def _escape_label(value: str) -> str:
    """Escape a label value for Prometheus text format."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _metric_line(
    name: str,
    value: float | int,
    labels: dict[str, str] | None = None,
) -> str:
    """Format a single metric line."""
    if labels:
        parts = ",".join(
            f'{k}="{_escape_label(v)}"'
            for k, v in sorted(labels.items())
        )
        return f"{name}{{{parts}}} {value}"
    return f"{name} {value}"


def _append_gauge(
    lines: list[str],
    name: str,
    help_text: str,
    value: float | int,
    labels: dict[str, str] | None = None,
) -> None:
    """Append HELP, TYPE, and metric line for a gauge."""
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} gauge")
    lines.append(_metric_line(name, value, labels))


def to_prometheus(
    report: BatchReport,
    *,
    model: str = "",
    dataset: str = "",
) -> str:
    """Convert a BatchReport to Prometheus text exposition format.

    Args:
        report: The batch report to convert.
        model: Model label.
        dataset: Dataset label.

    Returns:
        Prometheus text exposition format string.
    """
    bl: dict[str, str] = {}
    if model:
        bl["model"] = model
    if dataset:
        bl["dataset"] = dataset
    lbl = bl or None

    lines: list[str] = []

    _append_gauge(
        lines, "xpyd_acc_divergence_rate",
        "Fraction of divergent samples",
        round(report.divergence_rate, 6), lbl,
    )
    _append_gauge(
        lines, "xpyd_acc_total_samples",
        "Total number of samples in batch run",
        report.total_samples, lbl,
    )
    _append_gauge(
        lines, "xpyd_acc_divergent_samples",
        "Number of divergent samples",
        report.divergent_samples, lbl,
    )
    _append_gauge(
        lines, "xpyd_acc_truncated_samples",
        "Number of truncated samples",
        report.truncated_count, lbl,
    )

    # Classification counts
    classifications = {
        "likely_bug": report.likely_bugs,
        "likely_uncertainty": report.likely_uncertainty,
        "match": report.match_samples,
        "unknown": report.unknown_classification,
    }
    lines.append(
        "# HELP xpyd_acc_classification_count"
        " Count of samples by classification"
    )
    lines.append("# TYPE xpyd_acc_classification_count gauge")
    for cls_name, count in sorted(classifications.items()):
        labels = {**bl, "classification": cls_name}
        lines.append(
            _metric_line("xpyd_acc_classification_count", count, labels)
        )

    # Cost metrics (optional)
    if report.usage is not None:
        _append_gauge(
            lines, "xpyd_acc_total_tokens",
            "Total tokens consumed",
            report.usage.total_tokens, lbl,
        )
        if report.usage.estimated_cost_usd is not None:
            cost = round(report.usage.estimated_cost_usd, 6)
            _append_gauge(
                lines, "xpyd_acc_total_cost_usd",
                "Estimated total cost in USD",
                cost, lbl,
            )

    # Trailing newline per spec
    lines.append("")
    return "\n".join(lines)


def push_to_gateway(
    metrics: str,
    gateway_url: str,
    *,
    job: str = "xpyd_acc",
) -> None:
    """Push metrics to a Prometheus Pushgateway.

    Args:
        metrics: Prometheus text exposition format string.
        gateway_url: Pushgateway URL (e.g. http://localhost:9091).
        job: Job label for Pushgateway grouping.
    """
    url = f"{gateway_url.rstrip('/')}/metrics/job/{job}"
    req = Request(
        url,
        data=metrics.encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "text/plain; version=0.0.4; charset=utf-8",
        },
    )
    logger.info("Pushing metrics to %s", url)
    with urlopen(req, timeout=30) as resp:
        status = resp.status
    logger.info("Pushgateway responded with status %d", status)
