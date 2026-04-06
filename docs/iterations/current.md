# Current Iteration — M82

## Milestone

**M82** — Interactive REPL for Exploratory Comparison

## What Was Done

Added `xpyd-acc repl --baseline <url> --target <url> --model <model>` — an interactive
shell for exploratory comparison of two endpoints.

### Implementation

- `repl.py` module:
  - `ReplSession` dataclass: tracks state, params, history
  - `ReplCommand` dataclass: parsed REPL command
  - `ReplEntry` dataclass: single comparison result
  - `run_repl()` async function: main REPL loop with injectable I/O
  - `run_repl_iteration()`: sends prompt to both endpoints concurrently
  - `parse_command()`: parses `:command args` syntax
  - `format_comparison()`, `format_diff()`, `format_history()`: display helpers
  - `send_prompt()`: async HTTP request to OpenAI-compatible endpoint
- REPL commands: `:logprobs`, `:diff`, `:history`, `:set key=value`, `:export <path>`, `:quit`, `:help`
- `ReplSession.set_param()` for live parameter adjustment (temperature, top_p, seed, max_tokens)
- `ReplSession.export_json()` for session history export
- CLI: `xpyd-acc repl` subcommand with `--baseline`, `--target`, `--model`, `--api-key` flags
- CLI wired via `cli/parsers.py` → `cli/report.py` → `cli/__init__.py`

### Tests (32 tests)

- `TestParseCommand`: command parsing, args, empty colon, case insensitivity
- `TestReplSession`: set_param for all types, unknown param, export_json
- `TestFormatFunctions`: comparison match/diverge, diff, history empty/with entries
- `TestReplLoop`: quit, help, set, logprobs toggle, diff no history, history, export,
  EOF exit, unknown command, empty input, set without equals, export without path
- `TestCLIIntegration`: parser registration

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
| M81 | 2026-04-06 | Automatic Threshold Tuning | ✅ merged | Both approved |
| M82 | 2026-04-06 | Interactive REPL for Exploratory Comparison | ✅ merged | Both approved |
| M83 | 2026-04-06 | Divergence Heatmap by Token Position | ✅ merged | Both approved |
| M84 | 2026-04-06 | Endpoint Response Time Regression Detection | ✅ merged | Both approved |
| M85 | 2026-04-06 | Offline Mode — File-Based Comparison | ✅ merged | Both approved |
| M87 | 2026-04-06 | Automatic KV Cache Export from vLLM | ⏳ pending review | — |
