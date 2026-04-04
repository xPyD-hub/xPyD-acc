# xPyD-acc Roadmap

## M1: Project Skeleton ✅
- Basic project structure, CLI stub, CI

## M2: Logprobs Comparison Tool ✅
- Send same prompt to two endpoints
- Collect logprobs token by token
- Find first divergence point
- Report: token index, expected vs actual, probability diff

## M3: KV Cache Comparison ✅
- Load two KV cache dumps (numpy npz format)
- Compute: max absolute diff, mean absolute diff, cosine similarity per layer
- Flag layers with significant divergence
- Report with per-layer breakdown

## M4: Automated Diagnostic Pipeline ✅
- xpyd-acc diagnose: run all checks in sequence
- Step 1: baseline vs prefill-only (first token match?)
- Step 2: KV cache check (if dumps available)
- Step 3: baseline vs decode output (full sequence match?)
- Rich terminal output with ✅/❌ per step
- JSON report export

## M5: Output Comparison Utilities ✅
- Full text comparison (exact match, edit distance)
- Token-level diff visualization
- Semantic similarity score (optional, if embeddings available)
- Support for comparing streaming vs non-streaming outputs

## M6: Batch Dataset Comparison ✅
- Run full benchmark dataset on both aggregated and PD modes
- Auto-extract divergent samples (aggregated correct, PD wrong)
- Per-sample token-level diff with first divergence point
- Logprob sensitivity analysis: is divergence due to a bug or model uncertainty?
  - Compare top-1 vs top-2 logprob gap at divergence point
  - If gap is tiny (< threshold) → likely normal randomness, not a bug
  - If gap is large → likely a real precision issue
- Statistical report:
  - Total divergent samples / total samples
  - Divergence point distribution (at which token index do things go wrong?)
  - Logprob gap distribution at divergence points
  - Divergence grouped by context length (are longer contexts more prone?)
- Support common evaluation datasets:
  - GSM8K (math reasoning)
  - MMLU (knowledge)
  - MATH-500 (harder math)
  - HumanEval (code generation)
  - MT-Bench (multi-turn conversation)
  - Others as needed — pluggable dataset format
- Dataset runner: send all prompts to both endpoints, collect results, auto-compare
- Export: CSV of all samples with pass/fail/divergence info for manual review

## M7: Integration with xPyD Ecosystem ✅
- Work with xPyD-proxy endpoints directly
- Work with xPyD-sim for controlled testing (tool logic only, not real accuracy)
- Auto-detect endpoint type (aggregated vs disaggregated)

## M8: Reporting & Visualization ✅
- HTML report with:
  - Summary dashboard (pass rate per dataset)
  - Per-sample divergence detail (click to expand)
  - Logprob heatmap at divergence points
  - Context length vs divergence rate chart
- Terminal-friendly rich output for quick checks

## M9: Configuration File Support ✅
- TOML config file (`xpyd-acc.toml`) for repeated runs
- Auto-discovery in current directory
- CLI flags override config values
- Sections: defaults, batch, kv, report
- Type-safe config with dataclasses

## M10: JSON Export & Version Flag ✅
- `xpyd-acc --version` prints version string
- `batch-compare --json <path>` exports full BatchReport as JSON
- `BatchReport.to_json()` method for programmatic serialization

## M11: HTTP Retry with Exponential Backoff
- Reusable async retry decorator for all HTTP requests
- Retry on: connection errors, timeouts, HTTP 429/502/503/504
- Exponential backoff with jitter, Retry-After header support
- CLI flags: `--retries`, `--retry-delay`
- TOML config support in `[defaults]` section

## M12: Progress Bars for Batch Comparison
- Rich progress bars during batch dataset runs
- Per-sample progress tracking with ETA

## M13: Streaming Output Comparison
- Compare SSE streaming responses token-by-token
- Real-time divergence detection during streaming
