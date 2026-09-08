#!/system/bin/sh
# phone_wave3_phi3.sh — sustained-stress on Phi-3-mini-128k with the corrected
# 5-policy set: vanilla, v1, v1_FA, TOVA-layer (paper-faithful), llama.cpp.
#
# Phi-3 has ~14× larger KV cache per token than Llama-1B → biggest cache-pressure
# scenario → strongest expected eviction thermal benefit. Per the dissertation
# bandwidth-scaling argument.
#
# 60 min per cell (Phi-3 prefill is ~14 min/iter on 4-thread CPU at 1.63 GHz;
# this allows ~3 iters per cell).

set -u

DURATION_S=${DURATION_S:-3600}     # 60 min per policy
COOL_TARGET=${COOL_TARGET:-330}
COOL_MAX_S=${COOL_MAX_S:-900}
SAMPLE_HZ=${SAMPLE_HZ:-5}

WORKDIR=/data/local/tmp/endurkv
OUT_DIR=${OUT_DIR:-$WORKDIR/logs/wave3_phi3_$(date +%s)}
mkdir -p "$OUT_DIR"
PROG="$OUT_DIR/progress.log"
echo "[$(date)] wave3 Phi-3 START — duration=$DURATION_S s/cell" > "$PROG"

MODEL=$WORKDIR/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
# Phi-3 uses its own chat-templated prompts
PROMPT=$WORKDIR/prompts_chat/Phi-3-128k/narrativeqa_pub_001.txt

cool_phone() {
    local cell="$1"
    local t0
    t0=$(date +%s)
    while true; do
        skin=$(dumpsys battery | grep temperature | awk '{print $2}')
        if [ -n "$skin" ] && [ "$skin" -le "$COOL_TARGET" ] 2>/dev/null; then
            echo "[$(date)]   $cell: cooled to $skin" >> "$PROG"
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

run_eviction_policy() {
    local label="$1" policy="$2" kval="$3"
    local cell_dir="$OUT_DIR/$label"
    mkdir -p "$cell_dir"
    echo "" >> "$PROG"
    echo "[$(date)] === START $label (policy=$policy K=$kval, ${DURATION_S}s) ===" >> "$PROG"

    cool_phone "$label"

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
            --policy "$policy" --k-nominal "$kval" \
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
        echo "[$(date)]   $label iter=$ITER t=${ELAPSED}s exit=$EXIT decode_tps=$DT" >> "$PROG"
    done

    kill $SAMPLER 2>/dev/null
    pgrep -f sample_sensors | xargs -r kill 2>/dev/null
    sleep 2
    echo "[$(date)] === DONE $label: iters=$ITER, ran=${ELAPSED}s ===" >> "$PROG"
}

run_llamacpp_policy() {
    local cell_dir="$OUT_DIR/llamacpp_stock"
    mkdir -p "$cell_dir"
    echo "" >> "$PROG"
    echo "[$(date)] === START llamacpp_stock (llama-completion stock) ===" >> "$PROG"

    cool_phone "llamacpp_stock"

    sh "$WORKDIR/scripts/sample_sensors.sh" --out "$cell_dir/sensors.csv" --hz $SAMPLE_HZ < /dev/null > /dev/null 2>&1 &
    SAMPLER=$!
    sleep 1

    echo "iter,t_elapsed_s,exit,prefill_ms,decode_ms,decode_tps,prefill_tps,total_ms" > "$cell_dir/stress.csv"
    T_START=$(date +%s); ITER=0
    while true; do
        T_NOW=$(date +%s); ELAPSED=$(( T_NOW - T_START ))
        if [ "$ELAPSED" -ge "$DURATION_S" ]; then break; fi
        ITER=$(( ITER + 1 ))
        IDIR=$cell_dir/iter$(printf %04d $ITER)
        mkdir -p "$IDIR"

        LD_LIBRARY_PATH=$WORKDIR/bin_cpu $WORKDIR/bin_cpu/llama-completion \
            -m "$MODEL" -f "$PROMPT" \
            -n 256 --ignore-eos -c 12288 -b 512 -ub 64 -t 4 -tb 4 \
            -ngl 0 -fa 1 --temp 0 --seed 42 --no-warmup --no-display-prompt \
            > "$IDIR/gen.txt" 2> "$IDIR/stderr.log"
        EXIT=$?

        PF=$(grep -E "prompt eval time" "$IDIR/stderr.log" | tail -1 | sed -E 's/.*= *([0-9.]+) *ms.*/\1/')
        DM=$(grep -E "eval time " "$IDIR/stderr.log" | grep -v "prompt eval" | tail -1 | sed -E 's/.*= *([0-9.]+) *ms.*/\1/')
        DT=$(grep -E "eval time " "$IDIR/stderr.log" | grep -v "prompt eval" | tail -1 | sed -E 's/.*\( *([0-9.]+) *tokens per second\)/\1/')
        PFTPS=$(grep -E "prompt eval time" "$IDIR/stderr.log" | tail -1 | sed -E 's/.*\( *([0-9.]+) *tokens per second\)/\1/')
        TOT=$(grep -E "total time " "$IDIR/stderr.log" | tail -1 | sed -E 's/.*= *([0-9.]+) *ms.*/\1/')
        : "${PF:=0}" "${DM:=0}" "${DT:=0}" "${PFTPS:=0}" "${TOT:=0}"
        echo "$ITER,$ELAPSED,$EXIT,$PF,$DM,$DT,$PFTPS,$TOT" >> "$cell_dir/stress.csv"
        echo "[$(date)]   llamacpp iter=$ITER t=${ELAPSED}s exit=$EXIT decode_tps=$DT" >> "$PROG"
    done

    kill $SAMPLER 2>/dev/null
    pgrep -f sample_sensors | xargs -r kill 2>/dev/null
    sleep 2
    echo "[$(date)] === DONE llamacpp_stock ===" >> "$PROG"
}

# Pin DVFS at start, restore at end
echo "[$(date)] pinning DVFS" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh pin 2>> "$PROG"

# 5 cells matched: vanilla, v1, v1_FA, TOVA-layer, llama.cpp stock
run_eviction_policy vanilla     vanilla 0
run_eviction_policy v1_K512     v1      512
run_eviction_policy v1_fa_K512  v1_fa   512
run_eviction_policy tova_K512   tova    512
run_llamacpp_policy

echo "[$(date)] restoring DVFS" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh restore 2>> "$PROG"

echo "[$(date)] === WAVE3 PHI-3 COMPLETE ===" >> "$PROG"
touch "$OUT_DIR/DONE"
