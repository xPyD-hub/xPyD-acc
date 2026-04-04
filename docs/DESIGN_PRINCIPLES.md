# xPyD-acc Design Principles

## Core Positioning
Diagnostic tool for PD disaggregation accuracy issues. Not a fix tool — a pinpointing tool.

## Goal
Help users answer: "My PD disaggregated output is wrong. Is it Prefill, KV transfer, or Decode?"

## Approach
- Step-by-step isolation: test each stage independently
- Compare against aggregated baseline
- Report the first point of divergence with numerical evidence

## Key Capabilities
- Logprobs comparison (token-by-token, find first divergence)
- KV cache numerical comparison (max abs diff, cosine similarity)
- Automated diagnostic pipeline (run all checks in sequence)
- Clear human-readable report

## Rules
- Committer must be hlin99 <tony.lin@intel.com>
- All code, docs, issues, PRs in English
- Commit messages: conventional commits format
- Code in src/xpyd_acc/, tests in tests/
