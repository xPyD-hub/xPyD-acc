# xPyD-acc User Guide

Complete guide for using xPyD-acc to diagnose accuracy issues in PD (Prefill/Decode) disaggregated LLM inference.

## Installation

```bash
pip install xpyd-acc
```

For development:

```bash
git clone https://github.com/xPyD-hub/xPyD-acc.git
cd xPyD-acc
pip install -e ".[dev]"
```

Requirements: Python ≥ 3.10.

## Core Subcommands

### Diagnostic Pipeline

| Command | Description |
|---|---|
| `diagnose` | Run the full diagnostic pipeline (healthcheck → compare → report) |
| `healthcheck` | Check endpoint availability and basic API compatibility |
| `compare-output` | Compare text outputs between baseline (aggregated) and target (PD) endpoints |
| `compare-logprobs` | Compare per-token log-probabilities between two endpoints |
| `check-kv` | Check KV cache numerical accuracy between two `.npz` dumps |
| `report` | Generate an HTML report from batch comparison JSON results |

### Batch & Analysis

| Command | Description |
|---|---|
| `batch-compare` | Run comparison across a dataset of prompts |
| `entropy` | Analyze output entropy distribution from logprob data |
| `length-bias` | Detect systematic output length differences in batch reports |
| `sensitivity` | Prompt sensitivity analysis — test how small prompt changes affect divergence |
| `regression` | Detect regressions between two batch runs |
| `diff` | Side-by-side comparison of two batch reports |
| `ab-test` | A/B test divergence rates from two batch reports |
| `aggregate` | Aggregate multiple batch run reports |

### Exploration & Debugging

| Command | Description |
|---|---|
| `compare-streaming` | Compare SSE streaming outputs token-by-token |
| `detect` | Auto-detect xPyD endpoint type (aggregated vs prefill vs decode) |
| `bisect` | Binary search for minimum context length causing divergence |
| `snapshot` | Capture baseline outputs as a reference snapshot |
| `fingerprint` | Model fingerprinting via deterministic probes |
| `reproducibility` | Multi-run consistency measurement |
| `explain` | Deep-dive analysis of a single divergent sample |
| `cluster` | Cluster divergent samples by divergence pattern |
| `filter` | Filter samples from a batch report |
| `annotate` | Add notes and labels to batch report samples |
| `summary` | Compact summary of a batch report |
| `benchmark` | Benchmark endpoint latency |
| `watch` | Continuous divergence monitoring |

### Utilities

| Command | Description |
|---|---|
| `init` | Generate a starter `xpyd-acc.toml` config file |
| `config validate` | Validate a TOML config file |
| `cache clear` | Remove all cached responses |
| `cache stats` | Show cache statistics |
| `history save/list/trend/purge` | Result history & trend tracking |
| `dataset-stats` | Analyze dataset characteristics before batch comparison |
| `profiles` | List available named profiles (e.g., greedy, stochastic) |
| `completion` | Generate shell completion script |

## Step-by-Step Diagnostic Flow

The recommended diagnostic workflow isolates accuracy issues stage by stage.

### Step 1: Healthcheck — Confirm Endpoints Are Reachable

```bash
xpyd-acc healthcheck --url http://aggregated:8000
xpyd-acc healthcheck --url http://pd-endpoint:8001
```

Verifies that both endpoints respond correctly and expose a compatible API. Fix any connectivity or auth issues before proceeding.

### Step 2: Compare Output — Baseline vs PD

```bash
xpyd-acc compare-output \
  --baseline http://aggregated:8000 \
  --target http://pd-endpoint:8001 \
  --prompt "The quick brown fox jumps over the lazy dog" \
  --max-tokens 128
```

Sends the same prompt to both endpoints and compares the generated text. This is the first signal — if outputs match, PD disaggregation is likely accurate for this prompt.

For broader coverage, use `batch-compare` with a dataset:

```bash
xpyd-acc batch-compare \
  --baseline http://aggregated:8000 \
  --target http://pd-endpoint:8001 \
  --dataset prompts.jsonl \
  --output results.json
```

### Step 3: Compare Logprobs — Token-Level Precision

```bash
xpyd-acc compare-logprobs \
  --baseline http://aggregated:8000 \
  --target http://pd-endpoint:8001 \
  --prompt "Hello world" \
  --top-k 10
```

Compares per-token log-probabilities. Even when final text matches, logprob divergence can reveal hidden precision issues that surface under different prompts or longer contexts.

### Step 4: Check KV Cache — Numerical Accuracy

```bash
xpyd-acc check-kv \
  --kv-dump-a baseline_kv.npz \
  --kv-dump-b transfer_kv.npz
```

Directly compares KV cache tensors (requires `.npz` dumps from both modes). This isolates whether the KV transfer step introduces numerical drift.

### Step 5: Report — Generate Diagnostic Report

```bash
xpyd-acc report --input results.json --output report.html
```

Generates a comprehensive HTML report from batch comparison results, including divergence statistics, per-sample details, and visualizations.

## Interpreting Results

### Key Metrics

| Metric | What It Means |
|---|---|
| **Divergence Rate** | Fraction of samples where PD output differs from baseline. 0% = perfect match. >5% warrants investigation. |
| **Token Accuracy** | Fraction of generated tokens that match between baseline and target at each position. Lower accuracy at later positions may indicate KV cache drift. |
| **Entropy** | Shannon entropy of the output distribution. Higher entropy = more uncertainty. A significant entropy gap between modes suggests the model's confidence is affected by disaggregation. |
| **Max Logprob Difference** | Largest absolute difference in log-probability for the top token at any position. Values >0.1 are notable; >1.0 indicates a serious precision issue. |
| **KV MSE (Mean Squared Error)** | Numerical difference between KV cache tensors. Values near 0 are ideal. Increasing MSE across layers points to accumulating precision loss. |
| **Length Bias** | Systematic difference in output length between modes. Positive = PD generates longer outputs; negative = shorter. |

### Quick Decision Guide

- **All metrics green** → PD disaggregation is accurate; safe to deploy.
- **High divergence rate, low KV MSE** → Issue is likely in decode-stage sampling, not KV transfer.
- **High KV MSE, high divergence** → KV transfer is the root cause. Check serialization format, quantization, and memory alignment.
- **Divergence increases with context length** → Use `bisect` to find the critical length. Likely a position encoding or attention mask issue.
- **Entropy gap but text matches** → Latent precision issue. May surface with different prompts or temperatures. Monitor with `watch`.

## Global Options

All subcommands support these flags:

```
-v / --verbose      Increase verbosity (-v for INFO, -vv for DEBUG)
-q / --quiet        Quiet mode (ERROR level only)
--config FILE       Path to TOML config file (auto-discovers xpyd-acc.toml in cwd)
```

Sampling-related subcommands also accept:

```
--profile NAME      Named profile (e.g., greedy, stochastic)
--temperature F     Sampling temperature (0 = greedy)
--top-p F           Nucleus sampling top-p
--seed N            Random seed for reproducibility
```
