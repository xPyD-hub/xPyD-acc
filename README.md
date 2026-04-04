# xPyD-acc

Accuracy diagnostic tool for PD (Prefill/Decode) disaggregated LLM inference.

When PD disaggregation introduces accuracy issues, xPyD-acc helps you pinpoint
exactly where the problem is: Prefill, KV transfer, or Decode.

## The Problem

In PD disaggregated deployment, the inference pipeline is split across nodes.
Any of these stages can introduce accuracy drift:
- **Prefill** — TP parallelism, numerical precision (FP16/BF16), attention implementation
- **KV Transfer** — serialization/deserialization, quantization, memory alignment
- **Decode** — context length handling, position encoding, KV cache interpretation

## Approach

Step-by-step isolation:
1. Establish baseline (aggregated mode output)
2. Isolate and test each stage independently
3. Compare outputs at each boundary
4. Report exactly where divergence starts

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

# Check KV cache numerical accuracy
xpyd-acc check-kv \
  --kv-dump-a baseline_kv.npz \
  --kv-dump-b transfer_kv.npz
```

## License

TBD
