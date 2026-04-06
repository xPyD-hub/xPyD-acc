# Current Iteration — M81

## Milestone

**M81** — Automatic Threshold Tuning

## What Was Done

Added `xpyd-acc auto-threshold --reports r1.json r2.json ...` command that analyzes
historical batch reports and recommends optimal `--fail-threshold` and `--numeric-tolerance`
values based on percentile-based statistical analysis.

### Implementation

- `auto_threshold.py` module:
  - `ThresholdRecommendation` dataclass with `to_dict()` serialization
  - `analyze_thresholds()` — percentile-based recommendation engine
  - `load_reports()` — fault-tolerant multi-report loader
  - `format_recommendations()` — terminal display
  - `_percentile()` — interpolating percentile computation
- Fail threshold: p95 of observed divergence rates × 1.1 + headroom
- Numeric tolerance: p95 of logprob gaps at divergence points
- Confidence scoring: high (≥500 samples, ≥3 reports), medium, low
- `--percentile <float>` flag to control recommendation aggressiveness
- `--json <path>` export

### Files Changed

- `src/xpyd_acc/auto_threshold.py` (new)
- `src/xpyd_acc/cli/report.py` — added `_run_auto_threshold()` handler
- `src/xpyd_acc/cli/parsers.py` — registered `auto-threshold` subcommand
- `src/xpyd_acc/cli/__init__.py` — wired handler to early dispatch
- `tests/test_auto_threshold.py` (new, 18 tests)
- `docs/iterations/current.md` — updated

### Tests

18 tests covering:
- Percentile computation (basic, p95, empty, single)
- Threshold analysis (empty reports, single report, multiple/high-confidence, no logprob gaps, custom percentile)
- ThresholdRecommendation serialization and JSON round-trip
- Format output with data and without data
- Report loading (valid, missing, mixed)
- CLI integration (basic and custom percentile)

## Iteration History

| # | Date | Task | Result | Reviewer Comments |
|---|------|------|--------|-------------------|
| M74 | 2026-04-05 | Divergence Root Cause Heuristics | ✅ merged | Both approved |
| M75 | 2026-04-06 | Side-by-Side Token Diff Viewer | ✅ merged | Both approved |
| M76 | 2026-04-06 | Report Dashboard Server | ✅ merged | Both approved |
| M77 | 2026-04-06 | Prometheus Metrics Export | ✅ merged | Both approved |
| M78 | 2026-04-06 | Grafana Dashboard Template Export | ✅ merged | Both approved |
| M79 | 2026-04-06 | Parallel Multi-Dataset Batch Run | ✅ merged | Both approved |
| M80 | 2026-04-06 | Multi-Dataset CLI Integration | ✅ merged | Both approved |
| M81 | 2026-04-06 | Automatic Threshold Tuning | ⏳ pending review | — |
