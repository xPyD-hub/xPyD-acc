# Current Iteration — M72

## Milestone

**M72** — Feature-complete diagnostic toolkit with batch analysis, reporting, and CI integration support.

## Main Features

- **Full diagnostic pipeline** (`diagnose`) — automated healthcheck → compare → report flow
- **Output comparison** — text-level and logprob-level comparison between aggregated and PD endpoints
- **KV cache analysis** — direct numerical comparison of KV cache tensors (`.npz`)
- **Batch comparison** — dataset-driven testing with JSONL input
- **HTML reporting** — rich report generation from batch results
- **Streaming comparison** — SSE token-by-token diff
- **Entropy & length-bias analysis** — statistical detection of distribution shifts
- **Prompt sensitivity analysis** — measure divergence stability under prompt perturbations
- **Regression detection** — compare two batch runs to catch accuracy regressions
- **Bisect** — binary search for minimum context length triggering divergence
- **Fingerprinting & reproducibility** — model identity verification and multi-run consistency
- **History & trends** — save, list, and trend divergence rates over time
- **Caching** — response cache to avoid redundant API calls during iteration
- **Configuration** — TOML-based config with validation, profiles, and `init` scaffolding
- **Shell completion** — auto-generated completions for bash/zsh/fish

## Known Limitations

- No native support for multi-model comparison (single model per run)
- KV cache dump (`.npz`) must be obtained externally; xPyD-acc does not trigger dumps itself
- HTML report does not support real-time / streaming updates
- `watch` mode does not persist state across restarts
- No built-in authentication beyond API key pass-through
- Dataset format limited to JSONL (no CSV or Parquet support yet)

## Next Steps

- Add Parquet / CSV dataset support
- Native KV dump triggering via xPyD control plane API
- Interactive HTML report with filtering and drill-down
- CI/CD integration examples (GitHub Actions, GitLab CI)
- Multi-model and multi-version comparison in a single run
- Prometheus / OpenTelemetry metrics export for `watch` mode
