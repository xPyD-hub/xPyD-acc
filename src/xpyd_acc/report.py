"""HTML report generation with dashboard, per-sample details, and charts."""

from __future__ import annotations

import html
import json
from pathlib import Path

from xpyd_acc.batch_compare import BatchReport, SampleResult


def _escape(text: str) -> str:
    """HTML-escape text."""
    return html.escape(text, quote=True)


def _logprob_heatmap_cells(result: SampleResult) -> str:
    """Generate a simple inline heatmap representation for divergence point."""
    if result.exact_match or result.logprob_gap is None:
        return '<span class="tag match">match</span>'
    gap = result.logprob_gap
    # Map gap to color intensity: larger gap = more red
    intensity = min(gap / 1.0, 1.0)
    r = int(255 * intensity)
    g = int(255 * (1 - intensity * 0.7))
    b = int(100 * (1 - intensity))
    return (
        f'<span class="heatmap-cell" style="background:rgb({r},{g},{b})">'
        f'{gap:.4f}</span>'
    )


def _chart_data_context_length(report: BatchReport) -> str:
    """Generate JSON data for context length vs divergence rate chart."""
    labels = []
    rates = []
    for bucket in sorted(report.divergence_by_context_length.keys()):
        stats = report.divergence_by_context_length[bucket]
        labels.append(bucket)
        rate = stats["divergent"] / stats["total"] if stats["total"] > 0 else 0
        rates.append(round(rate * 100, 1))
    return json.dumps({"labels": labels, "rates": rates})


def _chart_data_divergence_index(report: BatchReport) -> str:
    """Generate JSON data for divergence point distribution histogram."""
    divergent = [
        r for r in report.results
        if r.is_divergent() and r.first_divergence_index is not None
    ]
    if not divergent:
        return json.dumps({"labels": [], "counts": []})
    indices = [r.first_divergence_index for r in divergent]
    # Create buckets of 5 tokens
    if not indices:
        return json.dumps({"labels": [], "counts": []})
    max_idx = max(indices)
    bucket_size = max(1, (max_idx + 1) // 10) if max_idx > 0 else 1
    buckets: dict[str, int] = {}
    for idx in indices:
        bucket_start = (idx // bucket_size) * bucket_size
        label = f"{bucket_start}-{bucket_start + bucket_size - 1}"
        buckets[label] = buckets.get(label, 0) + 1
    sorted_buckets = sorted(buckets.items(), key=lambda x: int(x[0].split("-")[0]))
    return json.dumps({
        "labels": [b[0] for b in sorted_buckets],
        "counts": [b[1] for b in sorted_buckets],
    })


def _sample_rows(report: BatchReport) -> str:
    """Generate HTML table rows for each sample."""
    rows = []
    for r in report.results:
        cls = "match" if r.exact_match else "divergent"
        div_idx = str(r.first_divergence_index) if r.first_divergence_index is not None else "-"
        heatmap = _logprob_heatmap_cells(r)
        baseline_preview = _escape(r.baseline_output[:120])
        detail_id = f"detail-{_escape(r.sample_id)}"
        rows.append(f"""
        <tr class="sample-row {cls}" onclick="toggleDetail('{detail_id}')">
            <td>{_escape(r.sample_id)}</td>
            <td><span class="tag {cls}">{r.classification}</span></td>
            <td>{div_idx}</td>
            <td>{heatmap}</td>
            <td>{r.context_length}</td>
            <td>{baseline_preview}{'…' if len(r.baseline_output) > 120 else ''}</td>
        </tr>
        <tr id="{detail_id}" class="detail-row" style="display:none">
            <td colspan="6">
                <div class="detail-content">
                    <div class="detail-section">
                        <strong>Prompt:</strong>
                        <pre>{_escape(r.prompt[:500])}</pre>
                    </div>
                    <div class="detail-section">
                        <strong>Baseline output:</strong>
                        <pre>{_escape(r.baseline_output)}</pre>
                    </div>
                    <div class="detail-section">
                        <strong>Target output:</strong>
                        <pre>{_escape(r.target_output)}</pre>
                    </div>
                    <div class="detail-section">
                        <strong>First divergence at token:</strong> {div_idx}
                        &nbsp;|&nbsp;
                        <strong>Logprob gap:</strong> \
{f'{r.logprob_gap:.6f}' if r.logprob_gap is not None else 'N/A'}
                    </div>
                </div>
            </td>
        </tr>""")
    return "\n".join(rows)


def generate_html_report(report: BatchReport) -> str:
    """Generate a full HTML report from a BatchReport.

    Includes:
    - Summary dashboard (pass rate per dataset)
    - Per-sample divergence detail (click to expand)
    - Logprob heatmap at divergence points
    - Context length vs divergence rate chart
    """
    ctx_chart = _chart_data_context_length(report)
    div_chart = _chart_data_divergence_index(report)
    sample_rows = _sample_rows(report)

    match_pct = (
        report.match_samples / report.total_samples * 100
        if report.total_samples > 0 else 0
    )

    avg_div = (
        f"{report.divergence_index_mean:.1f}"
        if report.divergence_index_mean is not None else "-"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>xPyD-acc Batch Comparison Report</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-dim: #8b949e; --green: #3fb950;
    --red: #f85149; --yellow: #d29922; --blue: #58a6ff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); padding: 24px; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 8px; }}
  .subtitle {{ color: var(--text-dim); margin-bottom: 24px; }}
  .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 16px; margin-bottom: 32px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 8px; padding: 16px; }}
  .card .label {{ color: var(--text-dim); font-size: 0.85rem; margin-bottom: 4px; }}
  .card .value {{ font-size: 1.8rem; font-weight: 600; }}
  .card .value.green {{ color: var(--green); }}
  .card .value.red {{ color: var(--red); }}
  .card .value.yellow {{ color: var(--yellow); }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 32px; }}
  .chart-box {{ background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; padding: 16px; }}
  .chart-box h3 {{ font-size: 0.95rem; margin-bottom: 12px; color: var(--text-dim); }}
  .bar-chart {{ display: flex; flex-direction: column; gap: 6px; }}
  .bar-row {{ display: flex; align-items: center; gap: 8px; }}
  .bar-label {{ width: 80px; text-align: right; font-size: 0.8rem; color: var(--text-dim); }}
  .bar-track {{ flex: 1; height: 20px; background: var(--border);
               border-radius: 4px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  .bar-fill.blue {{ background: var(--blue); }}
  .bar-fill.red {{ background: var(--red); }}
  .bar-value {{ width: 50px; font-size: 0.8rem; color: var(--text-dim); }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--border);
       color: var(--text-dim); font-size: 0.85rem; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }}
  .sample-row {{ cursor: pointer; }}
  .sample-row:hover {{ background: var(--surface); }}
  .tag {{ padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 500; }}
  .tag.match {{ background: #0d2818; color: var(--green); }}
  .tag.divergent, .tag.likely_bug {{ background: #3d1117; color: var(--red); }}
  .tag.likely_uncertainty {{ background: #2d2000; color: var(--yellow); }}
  .tag.unknown {{ background: #1c2333; color: var(--blue); }}
  .heatmap-cell {{ padding: 2px 8px; border-radius: 4px; color: #fff; font-size: 0.8rem;
                   font-weight: 600; }}
  .detail-row td {{ padding: 0; }}
  .detail-content {{ padding: 16px 24px; background: var(--surface); }}
  .detail-section {{ margin-bottom: 12px; }}
  .detail-section pre {{ background: var(--bg); padding: 8px 12px; border-radius: 4px;
                         margin-top: 4px; white-space: pre-wrap; word-break: break-all;
                         font-size: 0.85rem; max-height: 200px; overflow-y: auto; }}
  @media (max-width: 768px) {{ .charts {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>xPyD-acc Batch Comparison Report</h1>
<p class="subtitle">{report.total_samples} samples analyzed</p>

<div class="dashboard">
  <div class="card"><div class="label">Total Samples</div>
    <div class="value">{report.total_samples}</div></div>
  <div class="card"><div class="label">Match Rate</div>
    <div class="value green">{match_pct:.1f}%</div></div>
  <div class="card"><div class="label">Divergent</div>
    <div class="value red">{report.divergent_samples}</div></div>
  <div class="card"><div class="label">Likely Bugs</div>
    <div class="value red">{report.likely_bugs}</div></div>
  <div class="card"><div class="label">Likely Uncertainty</div>
    <div class="value yellow">{report.likely_uncertainty}</div></div>
  <div class="card"><div class="label">Avg Divergence Token</div>
    <div class="value">{avg_div}</div></div>
</div>

<div class="charts">
  <div class="chart-box">
    <h3>Context Length vs Divergence Rate</h3>
    <div class="bar-chart" id="ctx-chart"></div>
  </div>
  <div class="chart-box">
    <h3>Divergence Point Distribution (Token Index)</h3>
    <div class="bar-chart" id="div-chart"></div>
  </div>
</div>

<div class="card" style="margin-bottom:16px">
  <h3 style="margin-bottom:12px;color:var(--text-dim)">Per-Sample Results (click to expand)</h3>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Status</th><th>Div Index</th>
        <th>Logprob Gap</th><th>Ctx Len</th><th>Baseline Preview</th>
      </tr>
    </thead>
    <tbody>
      {sample_rows}
    </tbody>
  </table>
</div>

<script>
function toggleDetail(id) {{
  var el = document.getElementById(id);
  el.style.display = el.style.display === 'none' ? 'table-row' : 'none';
}}
function renderBarChart(containerId, labels, values, maxVal, color) {{
  var c = document.getElementById(containerId);
  if (!labels.length) {{ c.innerHTML = '<p style="color:var(--text-dim)">No data</p>'; return; }}
  var mx = maxVal || Math.max(...values, 1);
  var html = '';
  for (var i = 0; i < labels.length; i++) {{
    var pct = (values[i] / mx) * 100;
    html += '<div class="bar-row">' +
      '<div class="bar-label">' + labels[i] + '</div>' +
      '<div class="bar-track"><div class="bar-fill ' +
      color + '" style="width:' + pct + '%"></div></div>' +
      '<div class="bar-value">' + values[i] +
      (color === 'red' ? '%' : '') + '</div></div>';
  }}
  c.innerHTML = html;
}}
var ctx = {ctx_chart};
renderBarChart('ctx-chart', ctx.labels, ctx.rates, 100, 'red');
var div = {div_chart};
renderBarChart('div-chart', div.labels, div.counts, 0, 'blue');
</script>
</body>
</html>"""


def write_html_report(report: BatchReport, path: str | Path) -> None:
    """Generate and write HTML report to file."""
    path = Path(path)
    path.write_text(generate_html_report(report))


def format_terminal_report(report: BatchReport) -> str:
    """Rich terminal output for quick checks (delegates to batch_compare.format_report)."""
    from xpyd_acc.batch_compare import format_report
    return format_report(report)
