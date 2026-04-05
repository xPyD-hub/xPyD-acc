"""Report Dashboard Server — local HTTP server for interactive report exploration."""

from __future__ import annotations

import json
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any

from xpyd_acc.batch_compare import BatchReport, load_report

DASHBOARD_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>xPyD-acc Report Dashboard</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f5f5f5; color: #333; line-height: 1.6; }
.header { background: #1a1a2e; color: #fff; padding: 1rem 2rem; }
.header h1 { font-size: 1.4rem; }
.cards { display: flex; gap: 1rem; padding: 1.5rem 2rem; flex-wrap: wrap; }
.card { background: #fff; border-radius: 8px; padding: 1.2rem; min-width: 180px;
        box-shadow: 0 1px 3px rgba(0,0,0,.12); flex: 1; }
.card .label { font-size: .85rem; color: #666; }
.card .value { font-size: 1.8rem; font-weight: 700; margin-top: .3rem; }
.card .value.pass { color: #2e7d32; }
.card .value.fail { color: #c62828; }
.controls { padding: 0 2rem 1rem; display: flex; gap: 1rem; flex-wrap: wrap; }
.controls input, .controls select { padding: .5rem .8rem; border: 1px solid #ccc;
  border-radius: 4px; font-size: .9rem; }
.controls input { flex: 1; min-width: 200px; }
table { width: 100%; border-collapse: collapse; background: #fff; }
th, td { padding: .6rem .8rem; text-align: left; border-bottom: 1px solid #eee;
         font-size: .85rem; }
th { background: #f9f9f9; font-weight: 600; position: sticky; top: 0; }
.table-wrap { margin: 0 2rem 2rem; border-radius: 8px; overflow: auto;
              max-height: 60vh; box-shadow: 0 1px 3px rgba(0,0,0,.12); }
tr.divergent { background: #fff5f5; }
tr:hover { background: #f0f7ff; }
.badge { display: inline-block; padding: .15rem .5rem; border-radius: 4px;
         font-size: .75rem; font-weight: 600; }
.badge.match { background: #e8f5e9; color: #2e7d32; }
.badge.likely_bug { background: #ffebee; color: #c62828; }
.badge.likely_uncertainty { background: #fff3e0; color: #e65100; }
.badge.unknown { background: #e3f2fd; color: #1565c0; }
.detail { display: none; }
.detail.open { display: table-row; }
.detail td { background: #fafafa; padding: 1rem; }
.detail pre { white-space: pre-wrap; word-break: break-word; font-size: .8rem;
              max-height: 200px; overflow-y: auto; background: #f5f5f5; padding: .5rem;
              border-radius: 4px; margin-top: .3rem; }
.refresh-note { padding: .5rem 2rem; font-size: .75rem; color: #999; }
</style>
</head>
<body>
<div class="header"><h1>📊 xPyD-acc Report Dashboard</h1></div>
<div id="app"></div>
<script>
let reportData = null;
let lastMtime = 0;

async function fetchReport() {
  const resp = await fetch('/api/report');
  const data = await resp.json();
  if (data.mtime !== lastMtime) {
    lastMtime = data.mtime;
    reportData = data.report;
    render();
  }
}

function render() {
  if (!reportData) return;
  const r = reportData;
  const app = document.getElementById('app');
  const rate = (r.divergence_rate * 100).toFixed(1);
  const rateClass = r.divergent_samples === 0 ? 'pass' : 'fail';

  let html = `
    <div class="cards">
      <div class="card"><div class="label">Total</div>
        <div class="value">${r.total_samples}</div></div>
      <div class="card"><div class="label">Divergent</div>
        <div class="value fail">${r.divergent_samples}</div></div>
      <div class="card"><div class="label">Match</div>
        <div class="value pass">${r.match_samples}</div></div>
      <div class="card"><div class="label">Rate</div>
        <div class="value ${rateClass}">${rate}%</div></div>
    </div>
    <div class="controls">
      <input type="text" id="search" placeholder="Search prompt or output..." oninput="render()">
      <select id="classFilter" onchange="render()">
        <option value="">All classifications</option>
        <option value="match">match</option>
        <option value="likely_bug">likely_bug</option>
        <option value="likely_uncertainty">likely_uncertainty</option>
        <option value="unknown">unknown</option>
      </select>
    </div>
    <div class="refresh-note">Auto-refreshes every 5 seconds</div>
    <div class="table-wrap"><table>
      <thead><tr><th>#</th><th>ID</th><th>Class</th>
        <th>Div Idx</th><th>Gap</th>
        <th>Prompt</th></tr></thead>
      <tbody id="samples"></tbody>
    </table></div>`;
  app.innerHTML = html;

  const search = (document.getElementById('search')?.value || '').toLowerCase();
  const classFilter = document.getElementById('classFilter')?.value || '';
  const tbody = document.getElementById('samples');
  let rows = '';
  r.results.forEach((s, i) => {
    if (classFilter && s.classification !== classFilter) return;
    const text = (s.prompt + ' ' + s.baseline_output + ' ' + s.target_output).toLowerCase();
    if (search && !text.includes(search)) return;
    const cls = s.exact_match ? '' : 'divergent';
    const prompt = s.prompt.length > 80 ? s.prompt.slice(0, 80) + '…' : s.prompt;
    const gap = s.logprob_gap !== null ? s.logprob_gap.toFixed(3) : '—';
    const divIdx = s.first_divergence_index !== null ? s.first_divergence_index : '—';
    rows += `<tr class="${cls}" onclick="toggleDetail(${i})" style="cursor:pointer">
      <td>${i+1}</td><td>${esc(s.sample_id)}</td>
      <td><span class="badge ${s.classification}">${s.classification}</span></td>
      <td>${divIdx}</td><td>${gap}</td><td>${esc(prompt)}</td></tr>`;
    rows += `<tr class="detail" id="detail-${i}"><td colspan="6">
      <strong>Baseline:</strong><pre>${esc(s.baseline_output)}</pre>
      <strong>Target:</strong><pre>${esc(s.target_output)}</pre>
    </td></tr>`;
  });
  tbody.innerHTML = rows;
}

function toggleDetail(i) {
  document.getElementById('detail-' + i)?.classList.toggle('open');
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

fetchReport();
setInterval(fetchReport, 5000);
</script>
</body>
</html>
"""


def render_dashboard() -> str:
    """Return the dashboard HTML template."""
    return DASHBOARD_TEMPLATE


def report_to_api_dict(report: BatchReport) -> dict[str, Any]:
    """Convert a BatchReport to a JSON-serializable dict for the API."""
    return json.loads(report.to_json())


class ReportHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving dashboard and report API."""

    report_path: str = ""
    _cached_report: dict[str, Any] | None = None
    _cached_mtime: float = 0.0

    def _get_report_data(self) -> dict[str, Any]:
        """Load report, caching by mtime."""
        try:
            mtime = os.path.getmtime(self.report_path)
        except OSError:
            mtime = 0.0

        if mtime != self.__class__._cached_mtime or self.__class__._cached_report is None:
            report = load_report(self.report_path)
            self.__class__._cached_report = report_to_api_dict(report)
            self.__class__._cached_mtime = mtime

        return {"report": self.__class__._cached_report, "mtime": self.__class__._cached_mtime}

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path == "":
            html = render_dashboard()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html.encode())))
            self.end_headers()
            self.wfile.write(html.encode())
        elif self.path == "/api/report":
            try:
                data = self._get_report_data()
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                err = json.dumps({"error": str(exc)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
        else:
            self.send_response(404)
            body = b"Not Found"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Suppress default stderr logging."""
        pass


def create_handler(report_path: str) -> type[ReportHandler]:
    """Create a handler class bound to a specific report path."""
    handler = type("BoundReportHandler", (ReportHandler,), {
        "report_path": report_path,
        "_cached_report": None,
        "_cached_mtime": 0.0,
    })
    return handler


def run_server(
    report_path: str,
    *,
    host: str = "localhost",
    port: int = 8080,
    open_browser: bool = False,
) -> HTTPServer:
    """Start the dashboard server. Returns the server instance."""
    # Validate report exists and is loadable
    load_report(report_path)

    handler_class = create_handler(report_path)
    server = HTTPServer((host, port), handler_class)

    if open_browser:
        webbrowser.open(f"http://{host}:{port}")

    return server
