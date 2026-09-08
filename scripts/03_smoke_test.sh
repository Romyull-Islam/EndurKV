#!/usr/bin/env bash
# Phase 0 — smoke-test the existing llama-cli on the downloaded model.
# Run from the EndurKV root:  bash scripts/03_smoke_test.sh
# Should print a coherent ~32-token continuation of "The capital of France is".

set -e
cd "$(dirname "$0")/.."

CLI=llama.cpp/build/bin/llama-cli
MODEL=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

[ -x "$CLI" ]   || { echo "ERROR: $CLI not found or not executable. Build llama.cpp first."; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: $MODEL missing. Run scripts/02_download_model.sh first."; exit 1; }

echo "=== llama-cli smoke test ==="
echo "  cli:    $CLI"
echo "  model:  $MODEL"
echo "  prompt: 'The capital of France is'"
echo "  tokens: 32, temp=0, seed=42 (deterministic)"
echo

# -no-cnv: disable conversation mode so the prompt is taken verbatim
# --temp 0: greedy (reproducible)
# --seed 42: only matters if we sampled; harmless here
"$CLI" \
    -m "$MODEL" \
    -p "The capital of France is" \
    -n 32 \
    --temp 0 \
    --seed 42 \
    -no-cnv \
    2>&1 | tail -40

echo
echo "=== smoke test complete ==="
