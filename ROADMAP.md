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

## M49: Divergence Pattern Clustering ✅
- `xpyd-acc cluster --input <report.json>` groups divergent samples by divergence pattern
- Feature vector: divergence index, logprob gap, context length, output length ratio
- K-means clustering with automatic K selection via silhouette score
- `--clusters <n>` to override auto-selection
- `--json <path>` exports cluster results
- Per-cluster summary: representative sample, avg metrics, member count
- Rich terminal output with cluster table
- 12 tests covering clustering, edge cases, JSON export, CLI integration

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

## M46: Top-K Logprob Distribution Analysis ✅
- `--top-k <int>` flag (default 5) for `compare-logprobs`
- Collect top-K logprobs from each endpoint per position
- KL divergence, Jensen-Shannon divergence, Jaccard top-K overlap per position
- `TokenDistribution` dataclass, `DistributionReport` with aggregate metrics
- `--kl-threshold <float>` flag to flag positions exceeding threshold
- `distribution.py` module with all metric computations
- 22 tests covering KL/JS computation, overlap, edge cases

## M47: Auto-Bisect Divergence by Context Length ✅
- `xpyd-acc bisect --baseline <url> --target <url> --prompt <text> --model <model>`
- Binary search over prompt prefix lengths to find minimum divergence threshold
- `--min-length` and `--max-length` to bound search range
- `BisectResult` dataclass with threshold_length, steps, always/never_diverges
- JSON export via `--json <path>`
- Rich terminal progress output with per-step pass/fail
- Sampling params support (--temperature, --top-p, --seed, --profile)
- Retry, timeout flags supported
- `bisect.py` module with `run_bisect()` async function
- 10 tests covering binary search, edge cases, JSON export, callbacks

## M48: Compact Summary Command ✅
- `xpyd-acc summary <report.json>` outputs a compact summary of a batch report
- `--format oneline` (default): single-line summary for CI logs and dashboards
- `--format json`: compact JSON on one line (for piping to jq, webhooks)
- `--format kv`: key=value pairs, one per line (for shell scripts)
- Works with single-target and multi-target reports
- `summary.py` module with `SummaryData`, `extract_summary()`, `load_and_summarize()`
- 22 tests covering all formats, edge cases, and CLI integration

## M50: JUnit XML Export for CI Integration ✅
- `batch-compare --junit <path>` exports results as JUnit XML
- Each sample becomes a test case: passing samples → passed, divergent → failed
- Test suite name includes dataset, model, and timestamp
- Failure message contains divergence index, logprob gap, and truncated output diff
- Works with multi-target reports (one test suite per target)
- TOML config: `[batch] junit_path = "results.xml"`
- Compatible with Jenkins, GitLab CI, GitHub Actions test reporters
- `junit.py` module with `BatchReport.to_junit()` method
- Tests for XML generation, multi-target, edge cases, CLI integration

## M51: History Purge Command ✅
- `xpyd-acc history purge --older-than <days>` removes entries older than N days
- `--keep-last <n>` always retains the most recent N entries regardless of age
- `--dry-run` shows what would be removed without deleting
- `HistoryStore.purge()` method for programmatic use
- Rich terminal output with table of purged entries
- 8 tests covering purge logic, keep_last, dry-run, empty store, CLI integration

## M52: Sample Deep-Dive Explain Command ✅
- `xpyd-acc explain --report <path> --sample <id>` for detailed single-sample analysis
- Token-by-token alignment with divergence context window (5 tokens before/after)
- Logprob comparison at divergence point (baseline vs target)
- Classification reasoning (why likely_bug vs likely_uncertainty vs unknown)
- Suggested next debugging steps based on classification and divergence position
- `--json <path>` export for programmatic use
- Error handling for missing sample ID and missing report file
- 15 tests covering all functionality and CLI integration

## M53: Confidence Intervals for Divergence Rate ✅
- `batch-compare --confidence` flag adds 95% confidence interval to divergence rate
- Wilson score interval for binomial proportion (works well with small samples)
- `BatchReport.divergence_ci_lower` and `divergence_ci_upper` fields
- JSON/Markdown/JUnit exports include CI when computed
- `--confidence-level <float>` to change from default 0.95
- `confidence.py` module with `wilson_ci(successes, total, confidence)` function
- `--fail-threshold` integrates with CI: fail only if CI lower bound exceeds threshold
- Useful for deciding if sample size is large enough to trust the divergence rate
- Tests for Wilson CI math, integration with batch report, edge cases (0%, 100%, n=1)

## M54: Sample Annotation for Batch Reports ✅
- `xpyd-acc annotate --report <path> --sample <id> --note <text>` adds free-text note
- `xpyd-acc annotate --report <path> --sample <id> --label <tag>` adds classification label
- `xpyd-acc annotate --report <path> --list` shows all annotations
- `xpyd-acc annotate --report <path> --sample <id> --clear` removes annotations
- Sidecar storage (`<report>.annotations.json`) keeps original report immutable
- Labels: `known_issue`, `false_positive`, `needs_investigation`, `fixed`, or custom strings
- `filter` gains `--annotation-label`, `--annotated`, `--unannotated` flags
- `annotations_for_markdown()` helper for report integration
- 19 tests covering CRUD, persistence, CLI integration, edge cases

## M55: Model Fingerprinting ✅
- `xpyd-acc fingerprint --baseline <url>` sends deterministic probes and prints a 16-char hash
- `xpyd-acc fingerprint --baseline <url> --target <url>` compares two endpoint fingerprints
- 5 default probes with temperature=0, seed=42, max_tokens=16
- SHA-256 hash of concatenated outputs for quick identity check
- Per-probe diff on mismatch showing baseline vs target output
- `--json <path>` export, `--model`, `--api-key`, `--retries`, `--timeout` flags
- Exit 0 on match, exit 1 on mismatch (CI-friendly)
- `fingerprint.py` module: `collect_fingerprint()`, `compare_fingerprints()`, dataclasses
- 17 tests covering hash computation, comparison, mocked collection, edge cases

## M56: Report Schema Version & Backward-Compatible Loading ✅
- `REPORT_SCHEMA_VERSION` constant in `batch_compare.py` (starts at 1)
- `to_json()` includes `schema_version` field in every exported report
- `load_report(path)` function: deserialize JSON back to `BatchReport`
- Backward compatible: reports without `schema_version` treated as version 1
- Future-proof: raises `ValueError` if report version is newer than supported
- Missing optional fields default gracefully (classification, context_length, request_ids)
- Round-trip fidelity: confidence intervals, request IDs, all statistics preserved
- 8 tests covering round-trip, missing version, future version, minimal fields, CI fields

## M57: Endpoint Response Validation (Schema Check) ✅
- `response_validate.py` module with `validate_chat_response()` function
- `ResponseValidationError` exception with descriptive messages
- Validates: top-level structure, choices array, message.content, logprobs (when requested)
- Integrated into `_collect_output()` in `batch_compare.py` and `LogprobsCollector.collect()`
- `--skip-validation` CLI flag for non-standard endpoints
- TOML config: `[defaults] skip_validation = true`
- 12 tests covering valid responses, missing fields, malformed structure, skip flag

## M58: Dataset Statistics Command ✅
- `xpyd-acc dataset-stats <path>` analyzes dataset before batch comparison
- Character-level stats: min, max, mean, median, p95
- Estimated token counts using word/0.75 heuristic
- Duplicate prompt detection and count
- Template rendering support via `--template <path>`
- JSON export via `--json <path>`
- Rich terminal output with stats tables
- Works with JSONL, CSV, JSON array formats
- 15 tests covering stats computation, duplicates, template, JSON export, CLI

## M59: Cost Estimation from API Usage ✅
- `cost.py` module: extract token usage from OpenAI-compatible API responses
- `TokenUsage` dataclass: prompt_tokens, completion_tokens, total_tokens
- `CostConfig`: configurable per-million-token pricing (input/output)
- `UsageSummary`: aggregate usage across multiple requests with cost estimation
- `extract_usage()` parses `usage` field from API response JSON
- `format_usage_summary()` for terminal display
- JSON export via `UsageSummary.to_json()`
- 16 tests covering usage extraction, cost calculation, formatting, serialization

## M60: Cost Tracking Integration into Batch Compare ✅
- `_collect_output()` extracts and returns `TokenUsage` from API responses
- `BatchReport` has `usage: UsageSummary | None` field
- `format_report()` shows usage summary when available
- `to_json()` and `to_markdown()` include usage data
- CLI flags `--input-price` and `--output-price` set pricing (USD per 1M tokens)
- TOML `[cost]` section: `input_price_per_m`, `output_price_per_m`
- `load_report()` round-trips usage data
- `MultiTargetBatchReport` also tracks aggregate usage
- 17 tests covering integration, round-trip, config, CLI flags

## M61: Output Truncation Detection ✅
- `_collect_output()` extracts and returns `finish_reason` from API responses
- `SampleResult` has `baseline_finish_reason` and `target_finish_reason` fields
- `BatchReport` has `truncated_count` field (samples where either finish_reason is "length")
- `compute_report()` counts truncated samples automatically
- `format_report()` shows truncated count with ⚠️ when non-zero
- `to_json()` includes `finish_reason` per result and `truncated_count` in report
- `to_markdown()` shows truncation summary and flags truncated divergent samples
- `load_report()` deserializes truncation fields; backward-compatible with v1 reports
- Report schema version bumped to 2
- CLI `--warn-truncated <float>` flag: exit 2 if truncated ratio exceeds threshold
- TOML config: `[batch] warn_truncated = 0.1`
- 18 tests covering detection, classification, JSON round-trip, backward compat, formatting

## M62: Reproducibility Score — Multi-Run Consistency Measurement ✅
- `xpyd-acc reproducibility --url <endpoint> --prompt <text>` sends prompt N times
- `ReproducibilityResult` dataclass: unique_count, majority_fraction, avg_pairwise_distance
- `--runs <n>` configurable (default 5)
- `--baseline <url> --target <url>` dual-endpoint comparison mode
- `--json <path>` export
- `--threshold <float>` CI-friendly exit code (exit 1 if majority fraction below threshold)
- Sampling params (--temperature, --top-p, --seed, --profile) supported
- Levenshtein edit distance for pairwise output comparison
- Rich terminal output with consistency summary
- `reproducibility.py` module with `run_reproducibility()` async function
- 18 tests covering single/dual endpoint, metrics, edge cases, JSON export, CLI integration

## M63: Environment Variable Support for Max Tokens & Concurrency ✅
- `XPYD_ACC_MAX_TOKENS` environment variable for default max tokens
- `XPYD_ACC_CONCURRENCY` environment variable for default concurrency
- Priority chain: CLI flags > env vars > config file > defaults
- `EnvDefaults` extended with `max_tokens` and `concurrency` fields
- CLI applies env values before hardcoded defaults
- 4 new tests covering env var read and unset behavior

## M64: Checkpoint Resume for Batch Comparison ✅
- `batch-compare --checkpoint <path>` saves progress to a checkpoint file during batch runs
- `--checkpoint-clear` deletes existing checkpoint before starting fresh
- `Checkpoint` dataclass with completed_ids, serialised results, run metadata
- `save_checkpoint()` uses atomic write-then-rename for crash safety
- `load_checkpoint()` with corrupt file handling (returns None)
- `validate_checkpoint()` verifies baseline_url, target_url, model, total_samples match
- `result_to_dict()` / `dict_to_result()` for SampleResult serialisation round-trip
- `checkpoint.py` module with full save/load/validate/serialise API
- 19 tests covering dataclass, save/load, validation, serialisation, CLI integration

## M65: Checkpoint Integration into Batch Compare ✅
- `run_batch()` accepts `checkpoint_path` and `checkpoint_clear` parameters
- On start: loads existing checkpoint, validates against current run parameters
- Mismatched checkpoints are discarded with warning
- Completed samples from checkpoint are skipped (not re-sent to API)
- Checkpoint saved after each sample completion (crash-safe incremental resume)
- `--checkpoint-clear` deletes existing checkpoint before starting fresh
- Checkpoint file deleted on successful completion of all samples
- CLI flags `--checkpoint` and `--checkpoint-clear` wired through to `run_batch()`
- Results maintained in original sample order regardless of resume
- 12 tests covering resume, skip, mismatch discard, clear, cleanup, ordering

## M66: Retry Statistics Reporting ✅
- `retry_async()` returns `RetryResult(value, attempts)` instead of bare value
- `RetryStats` dataclass: total_requests, total_retries, max_retries_single, retried_request_count
- `RetryStats.record(result)` aggregates individual `RetryResult` instances
- `RetryStats.to_dict()` / `from_dict()` for serialization round-trip
- `BatchReport.retry_stats` field (optional, backward compatible)
- `format_report()` shows retry summary when retries occurred
- `to_json()` / `to_markdown()` include retry stats
- `load_report()` round-trips retry stats; old reports without field load fine
- All callers updated: batch_compare, logprobs, notify, reproducibility
- Fixed pre-existing bug: reproducibility.py used retry_async as decorator
- 25 tests: RetryResult, RetryStats aggregation, serialization, report integration, format, round-trip

## M67: Custom HTTP Headers for API Requests ✅
- `--header "Key: Value"` CLI flag (repeatable) on `batch-compare`
- `headers.py` module: `parse_header_arg()`, `parse_header_args()`, `parse_env_headers()`, `resolve_headers()`, `merge_with_defaults()`
- Environment variable: `XPYD_ACC_HEADERS` (comma-separated `Key:Value` pairs)
- TOML config: `[defaults] headers = {"X-Custom" = "value"}`
- Priority chain: CLI > env > config (consistent with other settings)
- Custom headers merged with default Authorization header (custom takes precedence)
- `_collect_output()` accepts `custom_headers` parameter
- `run_batch()` and `run_multi_batch()` forward custom headers
- CLI `batch-compare` resolves and passes headers through full chain
- 18 tests: parsing, env var, priority chain, merging, integration, CLI flag

## M69: Concurrency Scaling Analysis ✅
- `xpyd-acc concurrency-sweep --baseline <url> --target <url> --dataset <path> --levels 1,2,4,8`
- Runs batch comparison at each concurrency level
- Reports divergence rate per concurrency level in a table
- `SweepResult` and `SweepLevelResult` dataclasses with JSON serialization
- `format_sweep()` for rich terminal output
- `--json <path>` export
- `--model`, `--api-key`, `--max-tokens`, sampling params, `--template` supported
- Exit 1 if any level shows divergence (CI-friendly)
- `concurrency_sweep.py` module with `run_sweep()` async function
- Callback support via `on_level_complete` parameter
- 20 tests covering dataclasses, formatting, mocked sweep runs, CLI integration

## M68: Endpoint A/B Testing with Statistical Significance ✅
- `xpyd-acc ab-test --report-a <path> --report-b <path>` compares divergence rates
- Fisher's exact test (pure Python, no scipy) for statistical significance
- Chi-square test with Yates' correction as secondary test
- Effect size via odds ratio and 95% confidence interval for rate difference
- `--alpha <float>` flag (default 0.05) for significance level
- `--json <path>` export of full test results
- Exit code: 0 if no significant difference, 1 if significant (CI-friendly)
- Rich terminal output with clear verdict
- `ab_test.py` module: `ABTestResult`, `run_ab_test()`, `format_ab_test()`
- 26 tests: Fisher exact, chi-square, A/B test logic, JSON export, CLI integration

## M70: Output Entropy Analysis ✅
- `entropy.py` module: per-token Shannon entropy from top-K logprob distributions
- `token_entropy()`, `sequence_entropy()`, `entropy_stats()`, `entropy_at_divergence()`
- `EntropyStats` and `EntropyComparison` dataclasses with `to_dict()` serialization
- CLI: `xpyd-acc entropy --baseline-logprobs <path> [--target-logprobs <path>]`
- `--divergence-index <int>` for focused analysis at divergence point with context window
- `--context-window <int>` configurable (default 5)
- `--json <path>` export
- Rich terminal formatting for stats and comparison
- Handles edge cases: empty, single token, non-normalized logprobs
- 22 tests covering computation, edge cases, formatting, file I/O, CLI integration

## M71: Output Length Bias Detection ✅
- `length_bias.py` module: detect systematic output length differences between baseline and target
- `SampleLength` and `LengthBiasResult` dataclasses with `to_dict()` serialization
- Per-sample: baseline_length, target_length, length_diff, length_ratio
- Paired t-test for statistical significance of length bias
- Classification: shorter_bias, longer_bias, no_bias
- CLI: `xpyd-acc length-bias --report <path>`
- `--alpha <float>` significance level (default 0.05)
- `--json <path>` export
- Exit 1 if significant bias detected (CI-friendly)
- Rich terminal output with summary and distribution breakdown
- 28 tests covering t-test, analysis, formatting, file I/O, CLI integration

## M72: Prompt Sensitivity Analysis ✅
- `xpyd-acc sensitivity --baseline <url> --target <url> --prompt <text>` tests if divergence persists across prompt perturbations
- Generates N prompt perturbations (whitespace variations, prefix/suffix changes)
- Runs logprobs comparison on original + each perturbation
- Classification: `systematic` (all diverge), `sensitive` (some diverge), `robust` (none diverge)
- `--perturbations <n>` configurable (default 5)
- `--json <path>` export
- Exit 0 if robust, exit 1 if systematic (CI-friendly)
- Sampling params support (--temperature, --top-p, --seed, --profile)
- `sensitivity.py` module: `generate_perturbations()`, `run_sensitivity()`, `SensitivityResult`, `format_sensitivity()`
- 22 tests covering perturbation generation, classification, analysis, formatting, JSON export, CLI integration

## M73: Multi-Model Comparison in Single Batch Run ✅
- `batch-compare --model m1 --model m2 --baseline ... --target ... --dataset ...`
- Run same dataset against each model via existing `run_batch()`
- `MultiModelBatchReport` with per-model `BatchReport` and `CrossModelSummary`
- Cross-model analysis: systematic (all models diverge), model-specific, all-match
- JSON and Markdown export with per-model breakdowns
- Terminal formatting with per-model pass/fail indicators
- `multi_model.py` module: `run_multi_model()`, `compute_cross_model_summary()`, `format_multi_model_report()`
- 15 tests covering dataclasses, cross-model summary, serialization, async runs

## M74: Divergence Root Cause Heuristics ✅
- `xpyd-acc root-cause --report <path>` analyzes batch report to suggest probable root cause
- Heuristic rules based on divergence patterns:
  - Early divergence (index < 5) + high logprob gap → likely prefill issue
  - Mid-sequence divergence + context length correlation → likely KV cache transfer issue
  - Late divergence + low logprob gap → likely decode accumulation issue
  - Truncation-correlated divergence → likely max_tokens or stop sequence mismatch
- `RootCauseAnalysis` dataclass: classification, confidence, evidence list, suggested next steps
- Aggregate analysis across all divergent samples for overall diagnosis
- `--json <path>` export
- Rich terminal output with evidence breakdown
- `root_cause.py` module: `analyze_root_cause()`, `RootCauseAnalysis`, `format_root_cause()`
- 15 tests covering heuristic rules, edge cases, formatting, JSON export, CLI integration

## M75: Side-by-Side Token Diff Viewer ✅
- `xpyd-acc token-diff --report <path> --sample <id>` shows rich side-by-side token diff
- Baseline tokens on left, target tokens on right, aligned at divergence point
- Color-coded: green (match), red (mismatch), yellow (logprob warning)
- Context window: configurable tokens before/after divergence (default 10)
- Logprob annotations inline (show probability for each mismatched token)
- `--all-divergent` mode: iterate through all divergent samples in report
- `--format plain` for non-TTY output (CI logs)
- `token_diff.py` module: `build_token_diff()`, `format_token_diff()`, `TokenDiffLine`
- 12 tests covering alignment, coloring, edge cases, plain format, CLI integration

## M76: Report Dashboard Server ✅
- `xpyd-acc serve --report <path>` launches a local HTTP server with interactive dashboard
- Single-page HTML app served from bundled template (no external dependencies)
- Summary cards: divergence rate, sample count, top divergent samples
- Clickable sample list with per-sample divergence detail
- Filter by classification, search by prompt text
- Auto-refresh when report file changes on disk
- `--port <int>` (default 8080), `--host <str>` (default localhost)
- `--open` flag to auto-open browser
- `serve.py` module with `run_server()` using stdlib `http.server`
- 10 tests covering template rendering, route handling, report loading

## M77: Prometheus Metrics Export ✅
- `batch-compare --prometheus <path>` exports results in Prometheus text exposition format
- Metrics: `xpyd_acc_divergence_rate`, `xpyd_acc_total_samples`, `xpyd_acc_divergent_samples`, `xpyd_acc_truncated_samples`
- Per-classification gauge: `xpyd_acc_classification_count{classification="likely_bug|likely_uncertainty|match|unknown"}`
- Optional cost metrics: `xpyd_acc_total_cost_usd`, `xpyd_acc_total_tokens`
- Labels: model, dataset (extracted from report metadata)
- `xpyd-acc prometheus --report <path>` standalone subcommand to convert existing report
- `--push-gateway <url>` flag to push metrics to Prometheus Pushgateway
- `prometheus.py` module: `to_prometheus()`, `push_to_gateway()`
- 10 tests covering metric generation, labels, push mock, CLI integration

## M78: Grafana Dashboard Template Export ✅
- `xpyd-acc grafana-dashboard --report <path> --output <dashboard.json>` generates a Grafana dashboard JSON
- Pre-configured panels: divergence rate gauge, classification pie chart, context length vs divergence scatter
- Template variables for Prometheus datasource name
- `--datasource <name>` flag (default "Prometheus")
- `--title <text>` flag for dashboard title
- Dashboard compatible with Grafana 9+ import
- Pairs naturally with M77 Prometheus export for full observability stack
- `grafana.py` module: `generate_dashboard()`, `GrafanaDashboard` dataclass
- 10 tests covering dashboard generation, panels, template vars, JSON export, CLI integration

## M79: Parallel Multi-Dataset Batch Run ✅
- `batch-compare --dataset d1.jsonl --dataset d2.jsonl ...` runs multiple datasets in one command
- Concurrent dataset execution (datasets run in parallel, samples within each dataset also concurrent)
- Per-dataset report + combined summary report
- `MultiDatasetReport` dataclass with per-dataset `BatchReport` and aggregate stats
- Overall divergence rate across all datasets
- JSON/Markdown/CSV export with per-dataset breakdowns
- `--fail-threshold` applies per-dataset (any dataset exceeding threshold → exit 1)
- Template support: `--template` applies to all datasets, or per-dataset templates via TOML config
- TOML config: `[multi_dataset]` section with dataset list and per-dataset overrides
- Useful for running GSM8K + MMLU + HumanEval in one command

## M80: Multi-Dataset CLI Integration
- `batch-compare --dataset d1.jsonl --dataset d2.jsonl` accepts multiple `--dataset` flags
- When multiple datasets given, uses `run_multi_dataset()` from `multi_dataset.py`
- Per-dataset results printed with pass/fail indicators
- Overall divergence rate across all datasets
- `--fail-threshold` applies per-dataset (any exceeding → exit 1)
- JSON export via `--json` includes per-dataset breakdowns
- Markdown export via `--markdown` includes per-dataset sections
- Template support: `--template` applies to all datasets
- Progress bars per dataset
- Tests for CLI integration with multiple datasets
