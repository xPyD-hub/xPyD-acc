# Current Iteration — M73

## Milestone

**M73** — Multi-Model Comparison in Single Batch Run

## What Was Done

Added `multi_model.py` module enabling `batch-compare` to accept multiple `--model` flags
and run the same dataset against each model, producing a `MultiModelBatchReport` with:

- Per-model `BatchReport` (reuses existing `run_batch()`)
- Cross-model analysis: systematic divergences (all models), model-specific, all-match
- JSON and Markdown export with per-model breakdowns
- Terminal formatting with per-model pass/fail indicators

### Files Changed

- **`src/xpyd_acc/multi_model.py`** (new): `MultiModelBatchReport`, `CrossModelSummary`,
  `compute_cross_model_summary()`, `format_multi_model_report()`, `run_multi_model()`
- **`src/xpyd_acc/cli.py`**: `--model` changed to `action="append"` for repeatable flag;
  multi-model branch added in `_run_batch_compare()` with JSON/Markdown export and exit code
- **`tests/test_multi_model.py`** (new): 15 tests covering dataclasses, cross-model summary,
  report serialization, formatting, and async `run_multi_model()` with mocked `run_batch`

### Tests

15 tests all passing:
- `CrossModelSummary` serialization
- `compute_cross_model_summary` for all-match, systematic, model-specific, empty, single-model
- `MultiModelBatchReport` JSON round-trip, Markdown generation
- `format_multi_model_report` terminal output
- `run_multi_model` with callback, backward compat, parameter forwarding, mixed results

## Iteration History

| # | Date | Task | Result | Reviewer Comments |
|---|------|------|--------|-------------------|
| 1 | 2026-04-06 | M73: Multi-Model Comparison | ⏳ pending review | — |
| 2 | 2026-04-06 | M73: CLI Modularization — split cli.py into cli/ package | ⏳ pending review | — |

---

## Iteration 3: M74 — Divergence Root Cause Heuristics

**Date:** 2026-04-06
**Issue:** #161
**Branch:** `feat/m74-root-cause-heuristics`

### What was done

Added `xpyd-acc root-cause --report <path>` CLI subcommand that analyzes batch report divergence patterns and suggests probable root cause (prefill, KV transfer, decode, truncation, mixed, or inconclusive).

### Files changed

- **`src/xpyd_acc/root_cause.py`** (new): `analyze_root_cause()`, `RootCauseAnalysis`, `Evidence` dataclasses, `_classify_sample()` heuristic rules, `format_root_cause()` terminal display, `analyze_from_file()` convenience function
- **`src/xpyd_acc/cli/analysis.py`**: Added `handle_root_cause()` CLI handler
- **`src/xpyd_acc/cli/parsers.py`**: Registered `root-cause` subcommand with `--report` and `--json` flags
- **`src/xpyd_acc/cli/__init__.py`**: Wired `root-cause` to handler
- **`tests/test_root_cause.py`** (new): 21 tests covering heuristic classification, analysis, serialization, formatting, file I/O, CLI integration
- **`ROADMAP.md`**: Marked M73 ✅, added M74-M76

### Tests

21 tests all passing. Full suite: 1085 passed.

---

## Iteration: M75 — Side-by-Side Token Diff Viewer

**Date:** 2026-04-06
**Issue:** #163
**Branch:** feat/m75-token-diff

### What was done

- Created `src/xpyd_acc/token_diff.py` module with:
  - `TokenDiffLine` and `TokenDiff` dataclasses with JSON serialization
  - `build_token_diff()` — builds aligned diff with context window
  - `format_token_diff()` — rich terminal output with ANSI colors or plain text
  - `build_from_report()`, `build_all_divergent()`, `diff_from_file()` helpers
  - Color-coded output: green (match), red (mismatch), yellow (logprob warning), cyan (one-sided)
  - Logprob annotations at divergence point
- Added CLI subcommand `xpyd-acc token-diff`:
  - `--report`, `--sample`, `--all-divergent`, `--context`, `--format`, `--json`
- Created `tests/test_token_diff.py` with 25 tests covering:
  - Dataclass serialization, diff building, empty/asymmetric outputs
  - Context window, logprob annotations, logprob warning status
  - Report lookup, divergent filtering, file I/O
  - CLI help, sample mode, JSON export, error handling

### Tests

25 tests all passing.

## Iteration History

| # | Date | Task | Result | Reviewer Comments |
|---|------|------|--------|-------------------|
| M74 | 2026-04-05 | Divergence Root Cause Heuristics | ✅ merged | Both approved |
| M75 | 2026-04-06 | Side-by-Side Token Diff Viewer | ✅ merged | Both approved |
| M76 | 2026-04-06 | Report Dashboard Server | ✅ merged | Both approved |
| M77 | 2026-04-06 | Prometheus Metrics Export | ✅ merged | Both approved |
| M78 | 2026-04-06 | Grafana Dashboard Template Export | ✅ merged | Both approved |

## Iteration M79: Parallel Multi-Dataset Batch Run

### What
Added `multi_dataset.py` module for running multiple evaluation datasets concurrently.

### Implementation
- `MultiDatasetReport` dataclass with per-dataset `BatchReport` and aggregate stats
- `run_multi_dataset()` async function runs datasets in parallel via `asyncio.gather()`
- `format_multi_dataset_report()` for terminal display with per-dataset pass/fail indicators
- JSON and Markdown export with per-dataset breakdowns
- Callback support via `on_dataset_complete` parameter

### Files Changed
- `src/xpyd_acc/multi_dataset.py` (new)
- `tests/test_multi_dataset.py` (new, 18 tests)

### Tests
18 tests covering:
- Dataclass construction, aggregation, edge cases (empty, all match, all diverge)
- JSON serialization and round-trip
- Markdown export with truncation
- Async execution with mocked run_batch
- Callback invocation
- Kwargs forwarding
- Empty dataset map handling

---

## M80: Multi-Dataset CLI Integration

**Date:** 2026-04-06
**Issue:** #173
**Status:** In progress

### Problem
M79 added `multi_dataset.py` with `run_multi_dataset()` but no CLI integration — users cannot invoke multi-dataset runs from the command line.

### Solution
- Made `--dataset` flag repeatable (argparse `action="append"`)
- When >1 dataset given, `_run_batch_compare` delegates to `_run_multi_dataset()`
- Added `_run_multi_dataset()` in `batch.py`: loads datasets, runs concurrent comparison, prints results
- Per-dataset `--fail-threshold` checking
- JSON/Markdown export support

### Files Changed
- `src/xpyd_acc/cli/parsers.py` (--dataset becomes repeatable)
- `src/xpyd_acc/cli/batch.py` (multi-dataset routing + handler)
- `tests/test_multi_dataset_cli.py` (new, 10 tests)
- `ROADMAP.md` (M79 ✅, M80 added)

### Tests
10 tests covering:
- Dataclass aggregation
- JSON/Markdown export
- Terminal formatting
- CLI flag repeatability
- Multi-dataset delegation
- Single-dataset backward compatibility
