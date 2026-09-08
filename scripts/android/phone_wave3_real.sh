#!/system/bin/sh
# phone_wave3_real.sh — REAL sustained-stress benchmark, runs ENTIRELY on phone.
#
# This script is pushed to /data/local/tmp/endurkv/ and launched detached via
# phone_nohup. It survives ADB disconnects (writes meta+csv to disk).
#
# What it does:
#   For each policy (vanilla, v1 K=2048, v1 K=512, v1_fa K=512):
#     1. Strict cool-down to skin ≤ 32 °C (wait up to 10 min)
#     2. Start sensor sampler at 5 Hz
#     3. Loop iterations of (prefill 7700-token narrativeqa + decode 256 tokens)
#        for DURATION_S seconds
#     4. Stop sampler
#     5. Write per-iteration row to stress.csv
#
# Each policy is a separate, independent run from a known cold state. This is
# what makes the comparison clean.
#
# Output structure:
#   $OUT_DIR/
#     progress.log
#     vanilla/
#       sensors.csv  (5 Hz thermal trace)
#       stress.csv   (per-iteration: iter, t, prefill_ms, decode_tps, peak_kv_mb, ...)
#       iter*/meta.json, steps.csv
#     v1_K2048/
#       ...same structure...
#     v1_K512/
#       ...
#     v1_fa_K512/
#       ...
#     DONE  (sentinel file written at very end)

set -u

DURATION_S=${DURATION_S:-1500}    # 25 minutes per policy
COOL_TARGET=${COOL_TARGET:-330}    # 33.0 °C battery (deci-celsius)
COOL_MAX_S=${COOL_MAX_S:-900}      # 15 min max cooldown wait
SAMPLE_HZ=${SAMPLE_HZ:-5}

WORKDIR=/data/local/tmp/endurkv
OUT_DIR=${OUT_DIR:-$WORKDIR/logs/wave3_real_$(date +%s)}
mkdir -p "$OUT_DIR"
PROG="$OUT_DIR/progress.log"
echo "[$(date)] wave3_real START — duration=$DURATION_S s, cool_target=$COOL_TARGET (33.0 C)" > "$PROG"
echo "  output: $OUT_DIR" >> "$PROG"

MODEL=$WORKDIR/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
PROMPT=$WORKDIR/prompts_chat/Llama-3.2-1B/narrativeqa_pub_001.txt

cool_phone() {
    local cell="$1"
    local t0
    t0=$(date +%s)
    while true; do
        skin=$(dumpsys battery | grep temperature | awk '{print $2}')
        if [ -n "$skin" ] && [ "$skin" -le "$COOL_TARGET" ] 2>/dev/null; then
            echo "[$(date)]   $cell: cooled to $skin (target $COOL_TARGET)" >> "$PROG"
            return 0
        fi
        elapsed=$(( $(date +%s) - t0 ))
        if [ "$elapsed" -ge "$COOL_MAX_S" ]; then
            echo "[$(date)]   $cell: WARN cool timeout at $skin after ${elapsed}s, proceeding" >> "$PROG"
            return 0
        fi
        sleep 10
    done
}

run_policy() {
    local label="$1" policy="$2" kval="$3"
    local cell_dir="$OUT_DIR/$label"
    mkdir -p "$cell_dir"
    echo "" >> "$PROG"
    echo "[$(date)] === START $label (policy=$policy K=$kval, ${DURATION_S}s) ===" >> "$PROG"

    cool_phone "$label"

    # Start sensor sampler
    sh "$WORKDIR/scripts/sample_sensors.sh" --out "$cell_dir/sensors.csv" --hz $SAMPLE_HZ < /dev/null > /dev/null 2>&1 &
    SAMPLER=$!
    sleep 1

    echo "iter,t_elapsed_s,exit,prefill_ms,decode_tps,peak_kv_cells,peak_rss_kb,evicted" > "$cell_dir/stress.csv"
    T_START=$(date +%s)
    ITER=0

    while true; do
        T_NOW=$(date +%s)
        ELAPSED=$(( T_NOW - T_START ))
        if [ "$ELAPSED" -ge "$DURATION_S" ]; then break; fi
        ITER=$(( ITER + 1 ))
        IDIR=$cell_dir/iter$(printf %04d $ITER)
        mkdir -p "$IDIR"

        LD_LIBRARY_PATH=$WORKDIR/bin_cpu $WORKDIR/bin_cpu/eviction_bench \
            --model "$MODEL" \
            --prompt "$PROMPT" \
            --prompt-id narrativeqa_pub_001 \
            --policy "$policy" --k-nominal "$kval" \
            --max-tokens 256 --ctx-size 12288 --seed 42 \
            --threads 4 --n-gpu-layers 0 --n-batch 512 --ubatch-size 64 \
            --n-sink 4 --greedy \
            --out-csv "$IDIR/steps.csv" \
            --out-meta "$IDIR/meta.json" 2> "$IDIR/stderr.log"
        EXIT=$?

        if [ -f "$IDIR/meta.json" ]; then
            PF=$(grep -oE '"prefill_ms": *[0-9.]+'   "$IDIR/meta.json" | grep -oE '[0-9.]+$')
            DT=$(grep -oE '"decode_tps": *[0-9.]+'   "$IDIR/meta.json" | grep -oE '[0-9.]+$')
            KV=$(grep -oE '"peak_kv_cells": *[0-9]+' "$IDIR/meta.json" | grep -oE '[0-9]+$')
            RS=$(grep -oE '"peak_rss_kb": *[0-9]+'   "$IDIR/meta.json" | grep -oE '[0-9]+$')
            EV=$(grep -oE '"evicted_total_decode": *[0-9]+' "$IDIR/meta.json" | grep -oE '[0-9]+$')
        else
            PF=0; DT=0; KV=0; RS=0; EV=0
        fi
        echo "$ITER,$ELAPSED,$EXIT,$PF,$DT,$KV,$RS,$EV" >> "$cell_dir/stress.csv"

        # Compact per-iter summary line for the progress log (every 5th iter)
        if [ $(( ITER % 5 )) -eq 0 ]; then
            echo "[$(date)]   $label iter=$ITER t=${ELAPSED}s exit=$EXIT decode_tps=$DT" >> "$PROG"
        fi
    done

    # Stop sampler
    kill $SAMPLER 2>/dev/null
    pgrep -f sample_sensors | xargs -r kill 2>/dev/null
    sleep 2

    echo "[$(date)] === DONE $label: iters=$ITER, ran=${ELAPSED}s ===" >> "$PROG"
}

# The four conditions for the real benchmark
run_policy vanilla     vanilla 0
run_policy v1_K2048    v1      2048
run_policy v1_K512     v1      512
run_policy v1_fa_K512  v1_fa   512

echo "[$(date)] === WAVE3 REAL COMPLETE ===" >> "$PROG"
touch "$OUT_DIR/DONE"
