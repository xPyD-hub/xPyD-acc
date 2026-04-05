#!/usr/bin/env bash
# run_diagnose.sh — One-shot diagnostic: healthcheck → compare-output → compare-logprobs → report
set -euo pipefail

usage() {
  echo "Usage: $0 <aggregated_url> <pd_url> [--prompt TEXT] [--output-dir DIR]"
  echo
  echo "Arguments:"
  echo "  aggregated_url   Aggregated (baseline) endpoint URL"
  echo "  pd_url           PD disaggregated endpoint URL"
  echo
  echo "Options:"
  echo "  --prompt TEXT     Prompt to use (default: built-in test prompt)"
  echo "  --output-dir DIR  Output directory (default: ./diagnose-output)"
  exit 1
}

if [[ $# -lt 2 ]]; then
  usage
fi

AGGREGATED_URL="$1"
PD_URL="$2"
shift 2

PROMPT="The quick brown fox jumps over the lazy dog. Explain the history of this phrase."
OUTPUT_DIR="./diagnose-output"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_OUTPUT="$OUTPUT_DIR/batch_${TIMESTAMP}.json"
REPORT_OUTPUT="$OUTPUT_DIR/report_${TIMESTAMP}.html"

echo "=== xPyD-acc Diagnostic Run ==="
echo "Aggregated: $AGGREGATED_URL"
echo "PD Target:  $PD_URL"
echo "Timestamp:  $TIMESTAMP"
echo

# Step 1: Healthcheck
echo "[Step 1/4] Healthcheck — verifying endpoints..."
xpyd-acc healthcheck --url "$AGGREGATED_URL"
xpyd-acc healthcheck --url "$PD_URL"
echo "✓ Both endpoints healthy."
echo

# Step 2: Compare output
echo "[Step 2/4] Compare output — text-level comparison..."
xpyd-acc compare-output \
  --baseline "$AGGREGATED_URL" \
  --target "$PD_URL" \
  --prompt "$PROMPT" \
  --max-tokens 128
echo

# Step 3: Compare logprobs
echo "[Step 3/4] Compare logprobs — token-level precision..."
xpyd-acc compare-logprobs \
  --baseline "$AGGREGATED_URL" \
  --target "$PD_URL" \
  --prompt "$PROMPT" \
  --top-k 10
echo

# Step 4: Generate report
echo "[Step 4/4] Generating report..."
# Run a quick batch comparison to produce JSON, then generate HTML report
xpyd-acc batch-compare \
  --baseline "$AGGREGATED_URL" \
  --target "$PD_URL" \
  --dataset <(echo "{\"prompt\": \"$PROMPT\"}") \
  --output "$BATCH_OUTPUT" 2>/dev/null || true

if [[ -f "$BATCH_OUTPUT" ]]; then
  xpyd-acc report --input "$BATCH_OUTPUT" --output "$REPORT_OUTPUT"
  echo
  echo "=== Diagnostic Complete ==="
  echo "Batch results: $BATCH_OUTPUT"
  echo "HTML report:   $REPORT_OUTPUT"
else
  echo
  echo "=== Diagnostic Complete (no batch output) ==="
  echo "Individual comparisons above show the results."
  echo "To generate a full HTML report, run batch-compare with a dataset file."
fi
