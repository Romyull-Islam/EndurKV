#!/system/bin/sh
# phone_wave8_v1fa2_selective.sh — v1_FA² with SELECTIVE ANCHORING (Wave-7 fixes).
#
# Wave-7 failed because: (a) all prompt tokens anchored → recent context starved → PPL up
# and (b) insufficient cool-down left 1.8 GB free → state-swap forced 1552 MB swap-out.
#
# Wave-8 fixes:
#   1. v1_fa2 binary: --anchor-top-k 32 keeps only top-32 attention-scored prompt
#      positions as anchored, leaving recent_budget = 476 (huge recent window).
#   2. Launcher MemAvailable gate: require >= 4 GB free before iter-1 starts.
#   3. Strict cool-down to 33 C (no proceed-on-timeout).

set -u

DURATION_S=${DURATION_S:-3600}
COOL_TARGET=${COOL_TARGET:-330}
COOL_MAX_S=${COOL_MAX_S:-1500}     # longer ceiling, but soft target
SAMPLE_HZ=${SAMPLE_HZ:-5}
MIN_FREE_GB=${MIN_FREE_GB:-4}       # NEW: require >= 4 GB free RAM before iter-1

WORKDIR=/data/local/tmp/endurkv
OUT_DIR=${OUT_DIR:-$WORKDIR/logs/wave8_v1fa2_sel_$(date +%s)}
mkdir -p "$OUT_DIR"
PROG="$OUT_DIR/progress.log"
echo "[$(date)] wave8 v1_FA² selective anchoring START -> $OUT_DIR" > "$PROG"
echo "[$(date)] config: anchor_top_k=32, recent_budget=476, K=512, MIN_FREE_GB=$MIN_FREE_GB" >> "$PROG"

MODEL=$WORKDIR/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
PROMPT=$WORKDIR/prompts_chat/Phi-3-128k/longgen_prompt.txt
DDR_ZONE=/sys/class/thermal/thermal_zone47

ddr_temp_c() { awk '{printf "%d", $1/1000}' "$DDR_ZONE/temp" 2>/dev/null || echo 0; }
mem_free_gb() { awk '/MemAvailable/{printf "%.2f", $2/1024/1024}' /proc/meminfo; }

cool_phone() {
    local cell="$1"; local t0
    t0=$(date +%s)
    while true; do
        skin=$(dumpsys battery | grep temperature | awk '{print $2}')
        ddr=$(ddr_temp_c)
        if [ -n "$skin" ] && [ "$skin" -le "$COOL_TARGET" ] 2>/dev/null && [ "$ddr" -le 40 ]; then
            echo "[$(date)]   $cell: cooled to skin=$skin DDR=${ddr}C" >> "$PROG"; return 0
        fi
        elapsed=$(( $(date +%s) - t0 ))
        if [ "$elapsed" -ge "$COOL_MAX_S" ]; then
            echo "[$(date)]   $cell: WARN cool timeout at skin=$skin DDR=${ddr}C after ${elapsed}s, proceeding" >> "$PROG"
            return 0
        fi
        sleep 15
    done
}

# Memory gate — wait for at least MIN_FREE_GB of MemAvailable.
wait_for_memory() {
    local cell="$1"; local t0
    t0=$(date +%s)
    local target_kb=$(( MIN_FREE_GB * 1024 * 1024 ))
    while true; do
        local mfr_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
        if [ "$mfr_kb" -ge "$target_kb" ] 2>/dev/null; then
            echo "[$(date)]   $cell: mem-gate OK, free=$(mem_free_gb) GB" >> "$PROG"
            return 0
        fi
        elapsed=$(( $(date +%s) - t0 ))
        if [ "$elapsed" -ge 300 ]; then
            echo "[$(date)]   $cell: WARN mem-gate timeout at free=$(mem_free_gb) GB (target ${MIN_FREE_GB} GB)" >> "$PROG"
            return 0
        fi
        echo "[$(date)]   $cell: mem-gate waiting, free=$(mem_free_gb) GB" >> "$PROG"
        sleep 20
    done
}

run_v1fa2_sel() {
    local cell_dir="$OUT_DIR/v1_fa2_selective"
    mkdir -p "$cell_dir"
    echo "" >> "$PROG"
    echo "[$(date)] === START v1_fa2 selective ===" >> "$PROG"

    cool_phone "v1_fa2_sel"
    wait_for_memory "v1_fa2_sel"

    sh "$WORKDIR/scripts/sample_sensors.sh" --out "$cell_dir/sensors.csv" --hz $SAMPLE_HZ < /dev/null > /dev/null 2>&1 &
    SAMPLER=$!
    sleep 1
    echo "iter,t_elapsed_s,exit,prefill_ms,decode_tps,n_decode_steps,peak_kv_cells,peak_rss_kb,evicted,ppl,ddr_start_c,mem_free_gb_start" > "$cell_dir/stress.csv"
    T_START=$(date +%s); ITER=0
    while true; do
        T_NOW=$(date +%s); ELAPSED=$(( T_NOW - T_START ))
        if [ "$ELAPSED" -ge "$DURATION_S" ]; then break; fi
        ITER=$(( ITER + 1 ))
        IDIR=$cell_dir/iter$(printf %04d $ITER)
        mkdir -p "$IDIR"

        DDR_NOW=$(ddr_temp_c)
        MFR_NOW=$(mem_free_gb)

        LD_LIBRARY_PATH=$WORKDIR/bin_cpu $WORKDIR/bin_cpu/eviction_bench \
            --model "$MODEL" --prompt "$PROMPT" --prompt-id longgen \
            --policy v1_fa2 --k-nominal 512 \
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
            PP=$(grep -oE '"perplexity": *[0-9.]+' "$IDIR/meta.json" | grep -oE '[0-9.]+$')
        else
            PF=0; DT=0; NS=0; KV=0; RS=0; EV=0; PP=0
        fi
        echo "$ITER,$ELAPSED,$EXIT,$PF,$DT,$NS,$KV,$RS,$EV,$PP,$DDR_NOW,$MFR_NOW" >> "$cell_dir/stress.csv"
        echo "[$(date)]   v1_fa2_sel iter=$ITER t=${ELAPSED}s exit=$EXIT tps=$DT kv=$KV ev=$EV ppl=$PP DDR=${DDR_NOW}C free=${MFR_NOW}GB" >> "$PROG"
    done
    kill $SAMPLER 2>/dev/null
    pgrep -f sample_sensors | xargs -r kill 2>/dev/null
    sleep 2
    echo "[$(date)] === DONE v1_fa2_sel: iters=$ITER ===" >> "$PROG"
}

echo "[$(date)] pinning DVFS" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh pin 2>> "$PROG"
run_v1fa2_sel
echo "[$(date)] restoring DVFS" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh restore 2>> "$PROG"
echo "[$(date)] === WAVE8 COMPLETE ===" >> "$PROG"
touch "$OUT_DIR/DONE"
