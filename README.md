# xPyD-acc

PD disaggregation accuracy diagnostic tool.

When PD disaggregated inference produces wrong output, xPyD-acc pinpoints where the problem is: Prefill, KV transfer, or Decode.

## Key Features

- **Logprobs comparison** — token-by-token divergence detection between endpoints
- **KV cache analysis** — numerical accuracy checks (max abs diff, cosine similarity)
- **Automated diagnostics** — full pipeline: baseline → isolate → compare → report
- **Interactive REPL** — exploratory comparison with live parameter tuning
- **Batch & offline modes** — multi-dataset runs and file-based comparison

## Install

```bash
pip install xpyd-acc
```

## Quick Start

```bash
# Run full diagnostic
xpyd-acc diagnose \
  --baseline-url http://aggregated:8000 \
  --prefill-url http://prefill:8001 \
  --decode-url http://decode:8002 \
  --prompt "The quick brown fox"

# Compare logprobs between two endpoints
xpyd-acc compare-logprobs \
  --endpoint-a http://aggregated:8000 \
  --endpoint-b http://prefill:8001 \
  --prompt "Hello world"
```

## Part of xPyD

| Component | Description |
|-----------|-------------|
| [xPyD-proxy](https://github.com/xPyD-hub/xPyD-proxy) | PD-aware reverse proxy |
| [xPyD-sim](https://github.com/xPyD-hub/xPyD-sim) | PD disaggregation simulator |
| [xPyD-bench](https://github.com/xPyD-hub/xPyD-bench) | Benchmarking tool |
| **xPyD-acc** | Accuracy diagnostics (this repo) |
| [xPyD-plan](https://github.com/xPyD-hub/xPyD-plan) | Project planning |

## Resources

- [User Guide](docs/guide.md)
- [Contributing](CONTRIBUTING.md)

## License

[Apache 2.0](LICENSE)
