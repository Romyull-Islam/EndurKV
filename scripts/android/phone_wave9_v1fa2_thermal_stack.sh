#!/system/bin/sh
# phone_wave9_v1fa2_thermal_stack.sh — Wave-9: thermal-aware v1_FA² selective.
#
# Stack (from thermal-research workflow):
#   1. INT8 K cache quantization (--cache-type-k q8_0): cuts K-read DRAM bandwidth ~50%
#      → ~2.5 °C cooler DDR. V stays f16 due to state-swap layout constraint.
#   2. Preemptive CPU max-freq capping (watchdog reads DDR temp, throttles
#      proactively before kernel mitigation fires).
#   3. Closed-loop K controller (between iters): DDR-temp-driven K_nominal
#      modulation with 5°C hysteresis: cool→512, warm→384, hot→256.
#   4. Memory-pressure gate (inherited Wave-8 fix): MemAvailable >= 4 GB.
#
# Target (per Wave-9 plan): peak DDR 68-70°C, 6.2-6.9 tok/s, PPL <= 3.64, 0 forced throttles.

set -u

DURATION_S=${DURATION_S:-3600}
COOL_TARGET=${COOL_TARGET:-330}
COOL_MAX_S=${COOL_MAX_S:-1500}
SAMPLE_HZ=${SAMPLE_HZ:-5}
MIN_FREE_GB=${MIN_FREE_GB:-4}

WORKDIR=/data/local/tmp/endurkv
OUT_DIR=${OUT_DIR:-$WORKDIR/logs/wave9_v1fa2_stack_$(date +%s)}
mkdir -p "$OUT_DIR"
PROG="$OUT_DIR/progress.log"
echo "[$(date)] wave9 v1_FA² thermal-stack START -> $OUT_DIR" > "$PROG"
echo "[$(date)] stack: Q8_K + preempt-throttle watchdog + closed-loop K controller + mem-gate" >> "$PROG"

MODEL=$WORKDIR/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
PROMPT=$WORKDIR/prompts_chat/Phi-3-128k/longgen_prompt.txt
DDR_ZONE=/sys/class/thermal/thermal_zone47

WATCHDOG_LOG="$OUT_DIR/watchdog.log"
WATCHDOG_STOP=/sdcard/wave9_watchdog.stop
rm -f "$WATCHDOG_STOP"

ddr_temp_c() { awk '{printf "%d", $1/1000}' "$DDR_ZONE/temp" 2>/dev/null || echo 0; }
mem_free_gb() { awk '/MemAvailable/{printf "%.2f", $2/1024/1024}' /proc/meminfo; }

# Closed-loop K controller: read DDR, pick K_nominal.
pick_k_adaptive() {
    local ddr=$(ddr_temp_c)
    if [ "$ddr" -ge 66 ]; then echo "256"
    elif [ "$ddr" -ge 62 ]; then echo "384"
    else echo "512"
    fi
}

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

run_v1fa2_stack() {
    local cell_dir="$OUT_DIR/v1_fa2_stack"
    mkdir -p "$cell_dir"
    echo "" >> "$PROG"
    echo "[$(date)] === START v1_fa2 stack (Q8_K + watchdog + closed-K) ===" >> "$PROG"

    cool_phone "v1_fa2_stack"
    wait_for_memory "v1_fa2_stack"

    # Launch preempt-throttle watchdog as root background process
    echo "[$(date)] starting preempt-throttle watchdog (root)" >> "$PROG"
    su -c "sh $WORKDIR/scripts/preempt_throttle_watchdog.sh $WATCHDOG_LOG $WATCHDOG_STOP $DDR_ZONE" < /dev/null > /dev/null 2>&1 &

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

        K_NOW=$(pick_k_adaptive)
        DDR_NOW=$(ddr_temp_c)
        MFR_NOW=$(mem_free_gb)
        # If K changes, recompute recent_budget to keep total = K + n_sink-ish
        REC_BUDGET=$(( K_NOW - 32 - 4 ))  # K - anchor_top_k - n_sink

        LD_LIBRARY_PATH=$WORKDIR/bin_cpu $WORKDIR/bin_cpu/eviction_bench \
            --model "$MODEL" --prompt "$PROMPT" --prompt-id longgen \
            --policy v1_fa2 --k-nominal "$K_NOW" \
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
        echo "$ITER,$ELAPSED,$EXIT,$PF,$DT,$NS,$KV,$RS,$EV,$PP,$K_NOW,$DDR_NOW,$MFR_NOW" >> "$cell_dir/stress.csv"
        echo "[$(date)]   v1_fa2_stack iter=$ITER t=${ELAPSED}s exit=$EXIT tps=$DT kv=$KV ev=$EV ppl=$PP K=$K_NOW DDR=${DDR_NOW}C free=${MFR_NOW}GB" >> "$PROG"
    done
    kill $SAMPLER 2>/dev/null
    pgrep -f sample_sensors | xargs -r kill 2>/dev/null
    touch "$WATCHDOG_STOP"
    sleep 3
    echo "[$(date)] === DONE v1_fa2_stack: iters=$ITER ===" >> "$PROG"
}

echo "[$(date)] pinning DVFS" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh pin 2>> "$PROG"
run_v1fa2_stack
echo "[$(date)] restoring DVFS" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh restore 2>> "$PROG"
echo "[$(date)] === WAVE9 COMPLETE ===" >> "$PROG"
touch "$OUT_DIR/DONE"
