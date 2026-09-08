#!/usr/bin/env bash
# Phase B' — run attention_probe on a single demo prompt to collect entropy +
# per-decode-step per-layer attention scores for proposal slides.
# Run from the EndurKV root:  bash scripts/07_attention_demo.sh

set -e
cd "$(dirname "$0")/.."

ROOT="$(pwd)"
SRC="$ROOT/entropy_probe"
BUILD="$SRC/build"
PROBE="$BUILD/attention_probe"
MODEL=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
N_TOKENS=${N_TOKENS:-32}
PROMPT_TEXT=${PROMPT_TEXT:-"The capital of France is"}
PROMPT_ID=${PROMPT_ID:-attn_demo}

# Build if missing
if [ ! -x "$PROBE" ]; then
    echo "=== building entropy_probe + attention_probe ==="
    cmake -S "$SRC" -B "$BUILD" \
        -DLLAMA_CPP_DIR="$ROOT/llama.cpp" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_PROBE=ON
    cmake --build "$BUILD" -j
fi

[ -x "$PROBE" ] || { echo "ERROR: $PROBE not built."; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: $MODEL missing."; exit 1; }

mkdir -p logs/attention
PROMPT=$(mktemp /tmp/attn_demo.XXXX.txt)
trap "rm -f $PROMPT" EXIT
printf '%s' "$PROMPT_TEXT" > "$PROMPT"

# Pin to a single GPU for reproducibility and to avoid pipeline-parallel sync.
export CUDA_VISIBLE_DEVICES=0

OUT_CSV=logs/attention/${PROMPT_ID}.csv
OUT_BIN=logs/attention/${PROMPT_ID}.attn.bin

echo "=== running attention_probe ==="
echo "  prompt:     '$PROMPT_TEXT'"
echo "  prompt_id:  $PROMPT_ID"
echo "  max_tokens: $N_TOKENS"
echo "  out csv:    $OUT_CSV"
echo "  out attn:   $OUT_BIN"
echo

"$PROBE" \
    --model "$MODEL" \
    --prompt-file "$PROMPT" \
    --prompt-id "$PROMPT_ID" \
    --max-tokens "$N_TOKENS" \
    --seed 42 \
    --output      "$OUT_CSV" \
    --output-attn "$OUT_BIN" 2>&1 | tail -25

echo
echo "=== CSV head ==="
head -n 8 "$OUT_CSV"

echo
echo "=== attention.bin header ==="
python3 - "$OUT_BIN" <<'PY'
import sys, struct
path = sys.argv[1]
with open(path, "rb") as f:
    magic = f.read(4)
    n_steps, n_layers, n_head = struct.unpack("<III", f.read(12))
print(f"  path:     {path}")
print(f"  magic:    {magic!r}  (expect b'ATTN')")
print(f"  n_steps:  {n_steps}")
print(f"  n_layers: {n_layers}")
print(f"  n_head:   {n_head}")
import os
print(f"  size:     {os.path.getsize(path):,} bytes")
PY
