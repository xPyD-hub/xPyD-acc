# xPyD-acc Roadmap

## M1: Project Skeleton ✅
- Basic project structure, CLI stub, CI

## M2: Logprobs Comparison Tool ✅
- Send same prompt to two endpoints
- Collect logprobs token by token
- Find first divergence point
- Report: token index, expected vs actual, probability diff

## M3: KV Cache Comparison ⬜
- Load two KV cache dumps (numpy npz format)
- Compute: max absolute diff, mean absolute diff, cosine similarity per layer
- Flag layers with significant divergence
- Report with per-layer breakdown

## M4: Automated Diagnostic Pipeline ⬜
- xpyd-acc diagnose: run all checks in sequence
- Step 1: baseline vs prefill-only (first token match?)
- Step 2: KV cache check (if dumps available)
- Step 3: baseline vs decode output (full sequence match?)
- Rich terminal output with ✅/❌ per step
- JSON report export

## M5: Output Comparison Utilities ⬜
- Full text comparison (exact match, edit distance)
- Token-level diff visualization
- Semantic similarity score (optional, if embeddings available)
- Support for comparing streaming vs non-streaming outputs

## M6: Integration with xPyD Ecosystem ⬜
- Work with xPyD-proxy endpoints directly
- Work with xPyD-sim for controlled testing
- Auto-detect endpoint type (aggregated vs disaggregated)
