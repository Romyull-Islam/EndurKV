#!/system/bin/sh
# phone_wave10_ksweep.sh — Wave-10 K-sweep.
#
# Wave-9 ran with K_nominal=512 all 10 iters; the adaptive K ladder was never
# exercised. This wave runs 3 cells at FIXED K values to map the K → (PPL,
# thermal, throughput) Pareto frontier directly.
#
# Cells (in cool→hot order to give each cell a fair start):
#   1. K=1024  (larger cache, more bandwidth, expected hotter + better PPL)
#   2. K=512   (Wave-9 baseline, repeated for control)
#   3. K=384   (smaller, expected cooler + worse PPL)
#   4. K=256   (much smaller, expected coolest + much worse PPL)
#
# Inherits full Wave-9 stack: Q8 K cache + preempt-throttle watchdog + mem-gate.

set -u

DURATION_S=${DURATION_S:-3600}
COOL_TARGET=${COOL_TARGET:-330}
COOL_MAX_S=${COOL_MAX_S:-1800}        # longer cool ceiling — phone may stay warm between cells
SAMPLE_HZ=${SAMPLE_HZ:-5}
MIN_FREE_GB=${MIN_FREE_GB:-4}

WORKDIR=/data/local/tmp/endurkv
OUT_DIR=${OUT_DIR:-$WORKDIR/logs/wave10_ksweep_$(date +%s)}
mkdir -p "$OUT_DIR"
PROG="$OUT_DIR/progress.log"
echo "[$(date)] wave10 K-sweep START -> $OUT_DIR" > "$PROG"

MODEL=$WORKDIR/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
PROMPT=$WORKDIR/prompts_chat/Phi-3-128k/longgen_prompt.txt
DDR_ZONE=/sys/class/thermal/thermal_zone47

WATCHDOG_LOG="$OUT_DIR/watchdog.log"
WATCHDOG_STOP=/sdcard/wave10_watchdog.stop
rm -f "$WATCHDOG_STOP"

ddr_temp_c() { awk '{printf "%d", $1/1000}' "$DDR_ZONE/temp" 2>/dev/null || echo 0; }
mem_free_gb() { awk '/MemAvailable/{printf "%.2f", $2/1024/1024}' /proc/meminfo; }

cool_phone() {
    local cell="$1"; local t0
    t0=$(date +%s)
    while true; do
        skin=$(dumpsys battery | grep temperature | awk '{print $2}')
        ddr=$(ddr_temp_c)
        if [ -n "$skin" ] && [ "$skin" -le "$COOL_TARGET" ] 2>/dev/null && [ "$ddr" -le 40 ]; then
            echo "[$(date)]   $cell: cooled skin=$skin DDR=${ddr}C" >> "$PROG"; return 0
        fi
        elapsed=$(( $(date +%s) - t0 ))
        if [ "$elapsed" -ge "$COOL_MAX_S" ]; then
            echo "[$(date)]   $cell: WARN cool timeout skin=$skin DDR=${ddr}C after ${elapsed}s" >> "$PROG"
            return 0
        fi
        sleep 15
    done
}

wait_for_memory() {
    local cell="$1"; local t0
    t0=$(date +%s)
    local target_kb=$(( MIN_FREE_GB * 1024 * 1024 ))
    while true; do
        local mfr_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
        if [ "$mfr_kb" -ge "$target_kb" ] 2>/dev/null; then
            echo "[$(date)]   $cell: mem-gate OK, free=$(mem_free_gb) GB" >> "$PROG"; return 0
        fi
        elapsed=$(( $(date +%s) - t0 ))
        if [ "$elapsed" -ge 300 ]; then
            echo "[$(date)]   $cell: WARN mem-gate timeout, free=$(mem_free_gb) GB" >> "$PROG"; return 0
        fi
        sleep 20
    done
}

run_kcell() {
    local KVAL="$1"
    local cell_dir="$OUT_DIR/K${KVAL}"
    mkdir -p "$cell_dir"
    # recent_budget = K - anchor_top_k(32) - n_sink(4)
    local REC_BUDGET=$(( KVAL - 32 - 4 ))
    [ "$REC_BUDGET" -lt 32 ] && REC_BUDGET=32

    echo "" >> "$PROG"
    echo "[$(date)] === START K=$KVAL (recent_budget=$REC_BUDGET) ===" >> "$PROG"
    cool_phone "K=$KVAL"
    wait_for_memory "K=$KVAL"

    sh "$WORKDIR/scripts/sample_sensors.sh" --out "$cell_dir/sensors.csv" --hz $SAMPLE_HZ < /dev/null > /dev/null 2>&1 &
    SAMPLER=$!
    sleep 2

    echo "iter,t_elapsed_s,exit,prefill_ms,decode_tps,n_decode_steps,peak_kv_cells,peak_rss_kb,evicted,ppl,k_used,ddr_start_c,mem_free_gb_start" > "$cell_dir/stress.csv"
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
            --policy v1_fa2 --k-nominal "$KVAL" \
            --anchor-top-k 32 --recent-budget "$REC_BUDGET" \
            --cache-type-k q8_0 --cache-type-v f16 \
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
        echo "$ITER,$ELAPSED,$EXIT,$PF,$DT,$NS,$KV,$RS,$EV,$PP,$KVAL,$DDR_NOW,$MFR_NOW" >> "$cell_dir/stress.csv"
        echo "[$(date)]   K=$KVAL iter=$ITER t=${ELAPSED}s exit=$EXIT tps=$DT kv=$KV ev=$EV ppl=$PP DDR=${DDR_NOW}C free=${MFR_NOW}GB" >> "$PROG"
    done
    kill $SAMPLER 2>/dev/null
    pgrep -f sample_sensors | xargs -r kill 2>/dev/null
    sleep 3
    echo "[$(date)] === DONE K=$KVAL: iters=$ITER ===" >> "$PROG"
}

# Start watchdog ONCE for the whole sweep (will keep running during all cells)
echo "[$(date)] starting preempt-throttle watchdog (root) for entire sweep" >> "$PROG"
su -c "sh $WORKDIR/scripts/preempt_throttle_watchdog.sh $WATCHDOG_LOG $WATCHDOG_STOP $DDR_ZONE" < /dev/null > /dev/null 2>&1 &

echo "[$(date)] pinning DVFS" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh pin 2>> "$PROG"

# Run the 3 cells (K=512 baseline already done in Wave-9, skip it to save time)
run_kcell 1024
run_kcell 384
run_kcell 256

echo "[$(date)] restoring DVFS" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh restore 2>> "$PROG"
touch "$WATCHDOG_STOP"
sleep 3
echo "[$(date)] === WAVE10 COMPLETE ===" >> "$PROG"
touch "$OUT_DIR/DONE"
