#!/system/bin/sh
# phone_wave7_v1fa2.sh — v1_FA² validation on Wave-4 long-decode protocol.
#
# v1_FA² combines:
#   * FA-off prefill with v1 spread-gate eviction → n_anchored attention-aware positions
#   * State-swap to FA-on decode
#   * TIERED decode-time eviction: preserves anchored positions, drops oldest decode-generated
#   * Recent window protects newly generated tokens
#
# Compared to Wave-6 v1_fa bounded (recency only), v1_fa2 should:
#   - Preserve PPL better (prompt context survives across decode)
#   - Similar throughput
#   - Similar thermal (same K)
#
# Optional: --adaptive-k (sets K based on DDR temp between iters).

set -u

DURATION_S=${DURATION_S:-3600}
COOL_TARGET=${COOL_TARGET:-330}
COOL_MAX_S=${COOL_MAX_S:-900}
SAMPLE_HZ=${SAMPLE_HZ:-5}
ADAPTIVE_K=${ADAPTIVE_K:-0}      # 1 = adjust K between iters based on DDR

WORKDIR=/data/local/tmp/endurkv
OUT_DIR=${OUT_DIR:-$WORKDIR/logs/wave7_v1fa2_$(date +%s)}
mkdir -p "$OUT_DIR"
PROG="$OUT_DIR/progress.log"
echo "[$(date)] wave7 v1_FA² START -> $OUT_DIR adaptive_k=$ADAPTIVE_K" > "$PROG"

MODEL=$WORKDIR/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
PROMPT=$WORKDIR/prompts_chat/Phi-3-128k/longgen_prompt.txt

# Find DDR thermal zone (look for one with type matching ddr)
DDR_ZONE=""
for z in /sys/class/thermal/thermal_zone*; do
    if [ -f "$z/type" ]; then
        t=$(cat "$z/type" 2>/dev/null)
        if echo "$t" | grep -qiE "ddr"; then
            DDR_ZONE="$z"
            break
        fi
    fi
done

ddr_temp_c() {
    if [ -n "$DDR_ZONE" ] && [ -f "$DDR_ZONE/temp" ]; then
        raw=$(cat "$DDR_ZONE/temp" 2>/dev/null)
        echo $(( raw / 1000 ))
    else
        echo 0
    fi
}

pick_k_adaptive() {
    # adaptive K based on DDR temperature
    local ddr=$(ddr_temp_c)
    if [ "$ddr" -ge 62 ]; then
        echo "256"   # hot: aggressive eviction
    elif [ "$ddr" -ge 55 ]; then
        echo "512"   # warm: default
    else
        echo "1024"  # cool: preserve more context
    fi
}

cool_phone() {
    local cell="$1"; local t0
    t0=$(date +%s)
    while true; do
        skin=$(dumpsys battery | grep temperature | awk '{print $2}')
        if [ -n "$skin" ] && [ "$skin" -le "$COOL_TARGET" ] 2>/dev/null; then
            echo "[$(date)]   $cell: cooled to $skin (DDR=$(ddr_temp_c)C)" >> "$PROG"; return 0
        fi
        elapsed=$(( $(date +%s) - t0 ))
        if [ "$elapsed" -ge "$COOL_MAX_S" ]; then
            echo "[$(date)]   $cell: WARN cool timeout at $skin after ${elapsed}s, DDR=$(ddr_temp_c)C, proceeding" >> "$PROG"
            return 0
        fi
        sleep 10
    done
}

run_v1fa2() {
    local cell_dir="$OUT_DIR/v1_fa2"
    mkdir -p "$cell_dir"
    echo "" >> "$PROG"
    echo "[$(date)] === START v1_FA² (Wave-4 protocol) ===" >> "$PROG"
    cool_phone "v1_fa2"
    sh "$WORKDIR/scripts/sample_sensors.sh" --out "$cell_dir/sensors.csv" --hz $SAMPLE_HZ < /dev/null > /dev/null 2>&1 &
    SAMPLER=$!
    sleep 1
    echo "iter,t_elapsed_s,exit,prefill_ms,decode_tps,n_decode_steps,peak_kv_cells,peak_rss_kb,evicted,k_used,ddr_at_start_c" > "$cell_dir/stress.csv"
    T_START=$(date +%s); ITER=0
    while true; do
        T_NOW=$(date +%s); ELAPSED=$(( T_NOW - T_START ))
        if [ "$ELAPSED" -ge "$DURATION_S" ]; then break; fi
        ITER=$(( ITER + 1 ))
        IDIR=$cell_dir/iter$(printf %04d $ITER)
        mkdir -p "$IDIR"

        if [ "$ADAPTIVE_K" = "1" ]; then
            K_NOW=$(pick_k_adaptive)
        else
            K_NOW=512
        fi
        DDR_NOW=$(ddr_temp_c)

        LD_LIBRARY_PATH=$WORKDIR/bin_cpu $WORKDIR/bin_cpu/eviction_bench \
            --model "$MODEL" --prompt "$PROMPT" --prompt-id longgen \
            --policy v1_fa2 --k-nominal "$K_NOW" --recent-budget 256 \
            --max-tokens 2048 --ignore-eos --ctx-size 4096 --seed 42 \
            --threads 4 --n-gpu-layers 0 --n-batch 512 --ubatch-size 64 \
            --n-sink 4 --greedy \
            --out-csv "$IDIR/steps.csv" --out-meta "$IDIR/meta.json" 2> "$IDIR/stderr.log"
        EXIT=$?

        if [ -f "$IDIR/meta.json" ]; then
            PF=$(grep -oE '"prefill_ms": *[0-9.]+'   "$IDIR/meta.json" | grep -oE '[0-9.]+$')
            DT=$(grep -oE '"decode_tps": *[0-9.]+'   "$IDIR/meta.json" | grep -oE '[0-9.]+$')
            NS=$(grep -oE '"n_decode_steps": *[0-9]+' "$IDIR/meta.json" | grep -oE '[0-9]+$')
            KV=$(grep -oE '"peak_kv_cells": *[0-9]+' "$IDIR/meta.json" | grep -oE '[0-9]+$')
            RS=$(grep -oE '"peak_rss_kb": *[0-9]+'   "$IDIR/meta.json" | grep -oE '[0-9]+$')
            EV=$(grep -oE '"evicted_total_decode": *[0-9]+' "$IDIR/meta.json" | grep -oE '[0-9]+$')
        else
            PF=0; DT=0; NS=0; KV=0; RS=0; EV=0
        fi
        echo "$ITER,$ELAPSED,$EXIT,$PF,$DT,$NS,$KV,$RS,$EV,$K_NOW,$DDR_NOW" >> "$cell_dir/stress.csv"
        echo "[$(date)]   v1_fa2 iter=$ITER t=${ELAPSED}s exit=$EXIT tps=$DT kv=$KV evicted=$EV K=$K_NOW DDR=${DDR_NOW}C" >> "$PROG"
    done
    kill $SAMPLER 2>/dev/null
    pgrep -f sample_sensors | xargs -r kill 2>/dev/null
    sleep 2
    echo "[$(date)] === DONE v1_fa2: iters=$ITER ===" >> "$PROG"
}

echo "[$(date)] DDR_ZONE=$DDR_ZONE" >> "$PROG"
echo "[$(date)] pinning DVFS" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh pin 2>> "$PROG"
run_v1fa2
echo "[$(date)] restoring DVFS" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh restore 2>> "$PROG"
echo "[$(date)] === WAVE7 COMPLETE ===" >> "$PROG"
touch "$OUT_DIR/DONE"
