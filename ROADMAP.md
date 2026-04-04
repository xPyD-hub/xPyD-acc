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

## M11: HTTP Retry with Exponential Backoff ✅
- Reusable async retry decorator for all HTTP requests
- Retry on: connection errors, timeouts, HTTP 429/502/503/504
- Exponential backoff with jitter, Retry-After header support
- CLI flags: `--retries`, `--retry-delay`
- TOML config support in `[defaults]` section

## M12: Progress Bars for Batch Comparison ✅
- Rich progress bars during batch dataset runs
- Per-sample progress tracking with ETA
- `--no-progress` CLI flag and auto-disable for non-TTY

## M13: Streaming Output Comparison ✅
- Compare SSE streaming responses token-by-token
- Real-time divergence detection during streaming
- CLI subcommand `compare-streaming` with `--baseline`, `--target`, `--prompt`

## M14: Endpoint Health Check ✅
- Pre-flight check: verify both endpoints are reachable before running comparisons
- `xpyd-acc healthcheck <url>` CLI subcommand
- Reports: connectivity, response time, model availability
- Auto-check before `batch-compare` and `compare-streaming` with `--skip-healthcheck` opt-out

## M15: CSV Export for Batch Results ✅
- `batch-compare --csv <path>` exports per-sample results to CSV
- Columns: sample_id, prompt (truncated), baseline_output, target_output, match, divergence_index, logprob_gap
- Useful for spreadsheet analysis and filtering

## M16: Token Timing Analysis ✅
- Measure TTFT (time to first token) for both endpoints
- Inter-token latency statistics (p50, p95, p99)
- Timing comparison report between baseline and target
- Integrated into `compare-streaming` output

## M17: Prompt Template Support ✅
- Load prompt templates from YAML/TOML files
- Variable substitution in prompts (e.g., `{question}`, `{context}`)
- Built-in templates for common eval formats (GSM8K, MMLU, etc.)
- CLI flag `--template <path>` for `batch-compare`
- Template validation and error reporting

## M18: Environment Variable Support ✅
- `XPYD_ACC_API_KEY`, `XPYD_ACC_BASELINE_URL`, `XPYD_ACC_TARGET_URL`, `XPYD_ACC_MODEL`
- Priority chain: CLI flags > env vars > config file > defaults
- Avoids API keys in shell history
- `env.py` module with `get_env_defaults()` and `apply_env_defaults()`

## M19: Markdown Report Export ✅
- `batch-compare --markdown <path>` exports a Markdown report
- `BatchReport.to_markdown()` method for programmatic use
- Summary table, classification breakdown, divergence stats
- Context length analysis table
- Top divergent samples with details

## M20: Dry Run Mode for Batch Comparison ✅
- `batch-compare --dry-run` validates everything without sending API requests
- Checks: dataset loads correctly, template renders, endpoints reachable (via healthcheck)
- Reports: number of samples, estimated tokens, resolved config values
- Exits 0 if all validations pass, non-zero with actionable error messages
- Useful for CI pipelines and pre-flight validation before long batch runs

## M21: Regression Detection ✅
- `xpyd-acc regression --baseline <old.json> --current <new.json>` compares two batch runs
- Detects regressions (previously matched, now diverges), fixes, and persistent issues
- Summary: regression count, fix count, net change, divergence rate comparison
- Exit 0 if no regressions, exit 1 if regressions found (CI-friendly)
- JSON export for regression reports
- Rich terminal output with clear pass/fail indicators

## M22: Tolerance-Based Matching for Batch Comparison ✅
- `--normalize-whitespace` flag: collapse/strip whitespace before comparison
- `--ignore-case` flag: case-insensitive text matching
- `--numeric-tolerance <float>` flag: treat numbers within tolerance as equal
- New `MatchConfig` dataclass in `output_compare.py` for match settings
- `normalized_match()` function applying configured tolerances
- Integrated into `batch-compare` and `compare-streaming` commands
- TOML config section `[matching]` for default tolerance settings
- Tests for all tolerance modes and combinations

## M23: Selective Sample Rerun ✅
- `batch-compare --rerun <report.json>` reruns only divergent samples from a previous run
- Filters to samples where `match == false` in the input report
- Outputs a new report containing only rerun results
- Merge mode: `--rerun-merge` combines rerun results back into the original report
- Useful for quick retesting after fixes without rerunning the full dataset
- Exit code reflects rerun results (0 = all match, 1 = still divergent)

## M24: Sampling Parameter Support ✅
- CLI flags: `--temperature`, `--top-p`, `--seed` for all comparison commands
- Propagated to logprobs collection, batch comparison, streaming, and diagnose
- TOML config support in `[defaults]` section
- `temperature=0` + `seed` enables deterministic/reproducible comparisons
- Environment variables: `XPYD_ACC_TEMPERATURE`, `XPYD_ACC_TOP_P`, `XPYD_ACC_SEED`
- All flags default to None (server decides) to preserve backward compatibility

## M25: Named Profiles (Presets) ✅
- Named profiles in `[profiles.<name>]` TOML sections
- Each profile can override: model, temperature, top_p, seed, max_tokens, retries, retry_delay, matching settings
- CLI flag `--profile <name>` to activate a profile
- Profile values sit between config defaults and CLI flags in priority chain
- `xpyd-acc profiles` subcommand to list available profiles
- Built-in profiles: `greedy` (temperature=0, seed=42), `stochastic` (temperature=0.7)
- Reduces repetitive CLI flags for common testing scenarios

## M26: Multi-Run Aggregation ✅
- `xpyd-acc aggregate --reports r1.json r2.json ...` combines multiple batch run reports
- Classify each sample: persistent divergence (diverges in all runs), flaky (some runs), stable match (all runs)
- Per-sample consistency score: fraction of runs where the sample diverged
- Summary: total persistent, flaky, stable counts and rates
- Export aggregated report as JSON
- Rich terminal output with per-sample status
- Useful for distinguishing real bugs from non-deterministic behavior across multiple runs

## M27: Configurable HTTP Timeout for Batch Comparison ✅
- `_collect_output()` accepts `timeout` parameter (default 120.0s)
- `run_batch()` forwards `timeout` to HTTP requests
- `batch-compare --timeout <seconds>` CLI flag
- `XPYD_ACC_TIMEOUT` environment variable support
- TOML config `[defaults] timeout = 30.0`
- Priority: CLI > env > config > default (120.0s)

## M28: Logging & Verbosity Control ✅
- `--verbose` (`-v`) for INFO, `-vv` for DEBUG level logging
- `--quiet` (`-q`) for ERROR-only output (CI-friendly)
- `-v` and `-q` are mutually exclusive
- `log.py` module with `setup_logging()` and `get_logger()`
- Logging integrated into retry, healthcheck, batch_compare, streaming, config

## M29: Watch Mode — Continuous Divergence Monitoring
- `xpyd-acc watch --baseline <url> --target <url> --prompt <text> --interval <seconds>`
- Repeatedly runs logprobs comparison at configurable interval (default 60s)
- Reports each iteration: pass/fail, first divergence index, latency
- Rich live display with iteration counter and rolling stats
- `--max-iterations <n>` to stop after N checks (default: unlimited)
- `--alert-threshold <n>` exits with code 1 after N consecutive failures
- JSON log file via `--log <path>` for post-hoc analysis
- Ctrl+C gracefully stops and prints summary
- Useful for monitoring PD accuracy during long-running deployments
