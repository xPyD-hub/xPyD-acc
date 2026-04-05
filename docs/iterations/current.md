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
