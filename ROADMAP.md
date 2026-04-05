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

## M29: Watch Mode — Continuous Divergence Monitoring ✅
- `xpyd-acc watch --baseline <url> --target <url> --prompt <text> --interval <seconds>`
- Repeatedly runs logprobs comparison at configurable interval (default 60s)
- Reports each iteration: pass/fail, first divergence index, latency
- Rich live display with iteration counter and rolling stats
- `--max-iterations <n>` to stop after N checks (default: unlimited)
- `--alert-threshold <n>` exits with code 1 after N consecutive failures
- JSON log file via `--log <path>` for post-hoc analysis
- Ctrl+C gracefully stops and prints summary
- Useful for monitoring PD accuracy during long-running deployments

## M30: Snapshot Baseline Capture & Replay ✅
- `xpyd-acc snapshot capture --baseline <url> --dataset <path> --output <snap.json>` — capture baseline once
- `batch-compare --snapshot <snap.json> --target <url>` — replay without baseline endpoint
- Snapshot stores: timestamp, endpoint URL, model, per-sample outputs + logprobs
- Dataset/snapshot validation (mismatched sample IDs raise clear error)
- Progress bars, template support, all export formats (CSV, JSON, Markdown)
- Eliminates redundant baseline API calls for repeated comparisons

## M31: CI-Friendly Fail Threshold for Batch Comparison ✅
- `--fail-threshold <float>` flag (0.0–1.0) for `batch-compare`
- Exit 0 if divergence rate ≤ threshold, exit 1 if exceeded
- Default: None (backward compatible — exits 1 on any divergence)
- TOML config: `[batch] fail_threshold = 0.05`
- Environment variable: `XPYD_ACC_FAIL_THRESHOLD`
- Priority: CLI > env > config > None
- Clear terminal PASS/FAIL message with threshold comparison
- Also applies to `--rerun` mode

## M32: Multi-Target Comparison ✅
- `batch-compare --target <url1> --target <url2> ...` supports multiple targets
- Compare one baseline against N target endpoints simultaneously
- Per-target results in the batch report (separate divergence stats per target)
- Combined summary: which targets diverge on which samples
- Cross-target agreement matrix (do different targets agree with each other?)
- JSON/CSV/Markdown export includes per-target breakdowns
- Useful for comparing different PD configurations side by side

## M33: Request ID Tracking for API Call Correlation ✅
- All HTTP requests include `X-Request-ID` header with a UUID4 value
- `SampleResult` includes `request_ids` field (baseline + target request IDs)
- Request IDs appear in JSON export output
- Request IDs logged at DEBUG level
- `--no-request-id` flag to disable the feature
- Tests for request ID generation and header injection

## M34: Prompt Deduplication for Batch Comparison ✅
- `--deduplicate` flag on `batch-compare` to send each unique prompt only once per endpoint
- TOML config: `[batch] deduplicate = true`
- Results mapped back to all samples sharing the same prompt
- Reduces API calls when dataset contains duplicate prompts
- Works with single-target batch comparison and rerun mode
- No behavior change when flag is off (default)
- Tests for dedup on/off, with and without duplicates

## M35: Shell Completion Generation ✅
- `xpyd-acc completion bash` outputs Bash completion script
- `xpyd-acc completion zsh` outputs Zsh completion script
- `xpyd-acc completion fish` outputs Fish completion script
- Completions cover all subcommands, flags, and profile names
- Usage: `eval "$(xpyd-acc completion bash)"` or save to a file
- `--output <path>` flag to write directly to a file
- Tests for completion script generation (non-empty output, valid syntax markers)

## M36: Response Caching for Batch Comparison ✅
- Content-addressable cache: hash(endpoint_url + model + prompt + sampling_params) → cached response
- `--cache-dir <path>` flag (default: `.xpyd-acc-cache/`)
- `--no-cache` flag to bypass cache entirely
- TOML config: `[batch] cache_dir = ".xpyd-acc-cache"`
- Cache entries stored as JSON with TTL metadata
- `--cache-ttl <seconds>` flag (default: 3600) — entries older than TTL are re-fetched
- `xpyd-acc cache clear` subcommand to purge cached responses
- `xpyd-acc cache stats` subcommand to show cache size, hit rate, entry count
- Cache hits logged at INFO level, misses at DEBUG
- Dramatically speeds up reruns and iterative debugging sessions
- Tests for cache hit/miss, TTL expiry, clear, stats, and no-cache bypass

## M37: Result History & Trend Tracking ✅
- `xpyd-acc history save --report <path> --tag <label>` stores a batch report in local history DB
- `xpyd-acc history list` shows all saved reports with date, tag, divergence rate
- `xpyd-acc history trend --last <n>` shows divergence rate trend across last N runs
- History stored in `~/.xpyd-acc/history/` as timestamped JSON files with metadata
- `HistoryEntry` dataclass: timestamp, tag, report_path, divergence_rate, sample_count, dataset
- `HistoryStore` class for save/list/query operations
- Trend output: table with date, tag, divergence rate, delta from previous run
- Exit code 1 if trend shows increasing divergence (configurable via `--fail-on-regression`)
- Tests for save, list, trend calculation, regression detection

## M38: Endpoint Latency Benchmarking ✅
- `xpyd-acc benchmark <url>` sends N requests and measures latency
- Reports: min, max, mean, p50, p95, p99 latency statistics
- `--requests <n>` (default 10) and `--concurrency <n>` (default 1) flags
- `--prompt`, `--model`, `--max-tokens`, `--api-key` flags
- JSON export via `--json <path>`
- Rich terminal output with latency distribution table
- Sampling params support (--temperature, --top-p, --seed, --profile)
- Graceful error handling (failed requests counted separately)
- Tests for stats computation, JSON export, error handling

## M39: Configuration Validation & Init Command ✅
- `xpyd-acc init` generates a well-commented starter `xpyd-acc.toml`
- `--output <path>` and `--force` flags for custom path and overwrite
- `xpyd-acc config validate [path]` checks TOML for unknown sections/keys and type mismatches
- Warnings for unknown keys, errors for type mismatches
- Exit 0 if valid, exit 1 if errors found
- Profiles section treated as free-form (no validation)
- 20 tests covering init, overwrite protection, validation pass/fail, CLI integration

## M40: CSV and JSON Array Dataset Format Support ✅
- `load_dataset()` auto-detects format by file extension (`.csv`, `.json`, `.jsonl`)
- CSV support: header row with `prompt` column required, optional `id`, `expected`, metadata
- JSON array support: `[{"prompt": ...}, ...]` format
- JSONL remains default for unknown extensions
- Clear error messages for missing `prompt` column/field, non-array JSON, non-object items
- 11 tests covering all formats, error cases, and fallback behavior

## M41: Webhook Notifications for Divergence Alerts ✅
- `--webhook <url>` flag for `batch-compare`
- POST JSON payload: event, divergence_detected, total_samples, divergent_samples, divergence_rate
- `--webhook-header 'Key: Value'` for custom headers (repeatable)
- `--webhook-always` sends on every run, not just on divergence
- TOML config: `[notifications] webhook_url`, `webhook_headers`, `webhook_always`
- Environment variable: `XPYD_ACC_WEBHOOK_URL`
- Priority chain: CLI > env > TOML
- `notify.py` module with `send_webhook()`, `resolve_webhook_config()`
- Reuses existing retry logic for delivery reliability
- Tests for webhook send, skip, retry failure, header parsing, config resolution

## M42: Sample Filtering for Batch Reports ✅
- `xpyd-acc filter --input <report.json> --output <filtered.json>` filters samples from existing reports
- `--classification <value>` filter by classification (likely_bug, likely_uncertainty, match, unknown)
- `--divergent-only` / `--matched-only` quick filters
- `--min-logprob-gap <float>` filter samples with logprob gap above threshold
- `--max-logprob-gap <float>` filter samples with logprob gap below threshold
- `--min-context-length <int>` / `--max-context-length <int>` filter by context length
- `--search <text>` filter samples where prompt or output contains text (case-insensitive)
- Recalculates report statistics for the filtered subset
- Rich terminal summary of filtered results
- Tests for all filter criteria and combinations

## M43: Report Diff — Side-by-Side Comparison of Two Batch Reports ✅
- `xpyd-acc diff --old <report.json> --new <report.json>` compares two batch reports
- Per-sample status transition: match→diverge (regression), diverge→match (fix), unchanged
- Summary: total regressions, fixes, new samples, removed samples
- Output changes: show text diff of baseline/target outputs for changed samples
- `--json <path>` export the diff as JSON
- Rich terminal output with colored status transitions
- Complements `regression` (which only checks match/diverge) by also showing output text changes

## M44: Request Rate Limiting ✅
- `--rate-limit <float>` flag: max requests per second to each endpoint
- Token bucket algorithm for smooth rate control
- TOML config: `[defaults] rate_limit = 10.0`
- Environment variable: `XPYD_ACC_RATE_LIMIT`
- Integrated into `batch-compare`, `snapshot capture`, and `benchmark`
- `rate_limit.py` module with async `RateLimiter` class
- Logged at INFO level when requests are throttled
- Tests for rate limiter timing, config resolution, CLI integration

## M45: Custom Output Normalizers ✅
- Plugin-style output normalizers loaded from Python modules
- `--normalizer <module:function>` flag for `batch-compare`
- Built-in normalizers: strip_thinking_tags, normalize_json, normalize_numbers
- Normalizers applied before comparison (after existing tolerance matching)
- TOML config: `[matching] normalizers = ["strip_thinking_tags"]`
- Tests for built-in normalizers and custom normalizer loading
