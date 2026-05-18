#!/usr/bin/env bash
# Phase B — smoke-test the entropy_probe on one short prompt.
# Run from the EndurKV root:  bash scripts/05_probe_smoke.sh

set -e
cd "$(dirname "$0")/.."

PROBE=entropy_probe/build/entropy_probe
MODEL=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

[ -x "$PROBE" ] || { echo "ERROR: $PROBE not built. Run scripts/04_build_probe.sh first."; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: $MODEL missing."; exit 1; }

mkdir -p logs
PROMPT_FILE=$(mktemp /tmp/probe_smoke.XXXX.txt)
trap "rm -f $PROMPT_FILE" EXIT
printf 'The capital of France is' > "$PROMPT_FILE"

OUT=logs/smoke.csv
echo "=== running entropy_probe (max_tokens=24) ==="
"$PROBE" \
    --model "$MODEL" \
    --prompt-file "$PROMPT_FILE" \
    --prompt-id smoke \
    --max-tokens 24 \
    --seed 42 \
    --output "$OUT"

echo
echo "=== first 11 lines of $OUT (header + first 10 rows) ==="
head -n 11 "$OUT"

echo
echo "=== last 3 rows of $OUT ==="
tail -n 3 "$OUT"

echo
echo "=== column sanity ==="
python3 - "$OUT" <<'PY'
import csv, sys, math
path = sys.argv[1]
with open(path) as f:
    rd = csv.DictReader(f)
    rows = list(rd)
print(f"rows: {len(rows)}")
print(f"columns: {rd.fieldnames}")
H = [float(r['H_nats']) for r in rows]
top1 = [float(r['top1_prob']) for r in rows]
top5 = [float(r['top5_cumprob']) for r in rows]
print(f"H_nats     min={min(H):.4f}  max={max(H):.4f}  mean={sum(H)/len(H):.4f}")
print(f"top1_prob  min={min(top1):.4f}  max={max(top1):.4f}")
print(f"top5_cumprob  min={min(top5):.4f}  max={max(top5):.4f}")
# Sanity: top1 <= top5 <= 1, H >= 0
assert all(0.0 <= h for h in H), "H_nats has negatives"
assert all(t1 <= t5 + 1e-5 for t1, t5 in zip(top1, top5)), "top1 > top5"
assert all(t5 <= 1.0 + 1e-5 for t5 in top5), "top5 > 1"
print("invariants OK (H>=0, top1<=top5<=1)")
PY
