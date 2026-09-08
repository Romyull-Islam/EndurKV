#!/system/bin/sh
# phone_wave5_v1fa_filebacked.sh — re-run Phi-3 narrativeqa v1_FA cell with the
# file-backed state-swap binary, to measure the spillover reduction vs Wave-3's
# 528 MB swap-out baseline.
#
# Single cell only. Same protocol as Wave-3 v2.

set -u

DURATION_S=${DURATION_S:-3600}
COOL_TARGET=${COOL_TARGET:-330}
COOL_MAX_S=${COOL_MAX_S:-900}
SAMPLE_HZ=${SAMPLE_HZ:-5}

WORKDIR=/data/local/tmp/endurkv
OUT_DIR=${OUT_DIR:-$WORKDIR/logs/wave5_v1fa_filebacked_$(date +%s)}
mkdir -p "$OUT_DIR"
PROG="$OUT_DIR/progress.log"
echo "[$(date)] wave5 v1_FA-filebacked START -> $OUT_DIR" > "$PROG"

MODEL=$WORKDIR/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
PROMPT=$WORKDIR/prompts_chat/Phi-3-128k/narrativeqa_pub_001.txt

cool_phone() {
    local cell="$1"; local t0
    t0=$(date +%s)
    while true; do
        skin=$(dumpsys battery | grep temperature | awk '{print $2}')
        if [ -n "$skin" ] && [ "$skin" -le "$COOL_TARGET" ] 2>/dev/null; then
            echo "[$(date)]   $cell: cooled to $skin" >> "$PROG"; return 0
        fi
        elapsed=$(( $(date +%s) - t0 ))
        if [ "$elapsed" -ge "$COOL_MAX_S" ]; then
            echo "[$(date)]   $cell: WARN cool timeout at $skin after ${elapsed}s, proceeding" >> "$PROG"
            return 0
        fi
        sleep 10
    done
}

run_v1fa() {
    local cell_dir="$OUT_DIR/v1_fa_K512_filebacked"
    mkdir -p "$cell_dir"
    echo "" >> "$PROG"
    echo "[$(date)] === START v1_fa_K512_filebacked ===" >> "$PROG"
    cool_phone "v1_fa"
    sh "$WORKDIR/scripts/sample_sensors.sh" --out "$cell_dir/sensors.csv" --hz $SAMPLE_HZ < /dev/null > /dev/null 2>&1 &
    SAMPLER=$!
    sleep 1
    echo "iter,t_elapsed_s,exit,prefill_ms,decode_tps,peak_kv_cells,peak_rss_kb,evicted" > "$cell_dir/stress.csv"
    T_START=$(date +%s); ITER=0
    while true; do
        T_NOW=$(date +%s); ELAPSED=$(( T_NOW - T_START ))
        if [ "$ELAPSED" -ge "$DURATION_S" ]; then break; fi
        ITER=$(( ITER + 1 ))
        IDIR=$cell_dir/iter$(printf %04d $ITER)
        mkdir -p "$IDIR"

        LD_LIBRARY_PATH=$WORKDIR/bin_cpu $WORKDIR/bin_cpu/eviction_bench \
            --model "$MODEL" --prompt "$PROMPT" --prompt-id narrativeqa_pub_001 \
            --policy v1_fa --k-nominal 512 \
            --max-tokens 256 --ignore-eos --ctx-size 12288 --seed 42 \
            --threads 4 --n-gpu-layers 0 --n-batch 512 --ubatch-size 64 \
            --n-sink 4 --greedy \
            --out-csv "$IDIR/steps.csv" --out-meta "$IDIR/meta.json" 2> "$IDIR/stderr.log"
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
        echo "[$(date)]   v1_fa iter=$ITER t=${ELAPSED}s exit=$EXIT decode_tps=$DT peak_kv=$KV" >> "$PROG"
    done
    kill $SAMPLER 2>/dev/null
    pgrep -f sample_sensors | xargs -r kill 2>/dev/null
    sleep 2
    echo "[$(date)] === DONE v1_fa_K512_filebacked: iters=$ITER ===" >> "$PROG"
}

echo "[$(date)] pinning DVFS" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh pin 2>> "$PROG"
run_v1fa
echo "[$(date)] restoring DVFS" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh restore 2>> "$PROG"
echo "[$(date)] === WAVE5 COMPLETE ===" >> "$PROG"
touch "$OUT_DIR/DONE"
