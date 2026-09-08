#!/system/bin/sh
# phone_wave3_llamacpp_baseline.sh — stock-llama.cpp baseline cell for Wave-3.
#
# Runs `llama-completion` (the upstream llama.cpp interactive binary) in a loop
# for 25 min, with identical prompt + settings to our eviction_bench cells.
# Adds a true "pure llama.cpp" data point to the master comparison table
# (separate from "vanilla via eviction_bench" which also uses the llama.cpp
# decode path but goes through our binary).
#
# Same as the other cells:
#  * Pin DVFS to 1.63 GHz (big cores) before starting
#  * v4 sample_sensors at 5 Hz with USB rail capture
#  * 25 min loop of {prefill narrativeqa_pub_001 + decode 256 tokens}
#  * Sensor + per-iter timing captured

set -u

DURATION_S=${DURATION_S:-1500}
COOL_TARGET=${COOL_TARGET:-330}
COOL_MAX_S=${COOL_MAX_S:-900}
SAMPLE_HZ=${SAMPLE_HZ:-5}

WORKDIR=/data/local/tmp/endurkv
OUT_DIR=${OUT_DIR:-$WORKDIR/logs/wave3_llamacpp_$(date +%s)}
mkdir -p "$OUT_DIR"
PROG="$OUT_DIR/progress.log"
echo "[$(date)] wave3 llama-completion baseline START -> $OUT_DIR" > "$PROG"

MODEL=$WORKDIR/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
PROMPT=$WORKDIR/prompts_chat/Llama-3.2-1B/narrativeqa_pub_001.txt

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

# Pin DVFS at start
echo "[$(date)] pinning DVFS via pin_dvfs.sh" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh pin 2>> "$PROG"

CELL_DIR=$OUT_DIR/llamacpp_stock
mkdir -p "$CELL_DIR"

cool_phone llamacpp_stock

# Start sampler
sh "$WORKDIR/scripts/sample_sensors.sh" --out "$CELL_DIR/sensors.csv" --hz $SAMPLE_HZ < /dev/null > /dev/null 2>&1 &
SAMPLER=$!
sleep 1

# Per-iter CSV — parse timings from llama-completion's stderr
echo "iter,t_elapsed_s,exit,prefill_ms,decode_ms,decode_tps,prefill_tps,total_ms" > "$CELL_DIR/stress.csv"
T_START=$(date +%s); ITER=0

while true; do
    T_NOW=$(date +%s); ELAPSED=$(( T_NOW - T_START ))
    if [ "$ELAPSED" -ge "$DURATION_S" ]; then break; fi
    ITER=$(( ITER + 1 ))
    IDIR=$CELL_DIR/iter$(printf %04d $ITER)
    mkdir -p "$IDIR"

    # Pure llama-completion invocation. -fa 1 enables Flash Attention (the
    # vanilla-with-FA path). -n 256 to match eviction_bench --max-tokens 256.
    LD_LIBRARY_PATH=$WORKDIR/bin_cpu $WORKDIR/bin_cpu/llama-completion \
        -m "$MODEL" \
        -f "$PROMPT" \
        -n 256 \
        -c 12288 \
        -b 512 \
        -ub 64 \
        -t 4 -tb 4 \
        -ngl 0 \
        -fa 1 \
        --temp 0 \
        --seed 42 \
        --no-warmup \
        --no-display-prompt \
        > "$IDIR/gen.txt" 2> "$IDIR/stderr.log"
    EXIT=$?

    # Parse llama_print_timings lines from stderr
    # Format examples:
    #   llama_print_timings: prompt eval time = N ms / M tokens ( X ms per token, Y tokens per second)
    #   llama_print_timings: eval time        = N ms / M runs   ( X ms per token, Y tokens per second)
    #   llama_print_timings: total time       = N ms / M tokens
    PF=$(grep -E "prompt eval time" "$IDIR/stderr.log" | tail -1 | sed -E 's/.*= *([0-9.]+) *ms.*/\1/')
    DT=$(grep -E "eval time " "$IDIR/stderr.log" | grep -v "prompt eval" | tail -1 | sed -E 's/.*\( *([0-9.]+) *tokens per second\)/\1/')
    # eval time totals
    DM=$(grep -E "eval time " "$IDIR/stderr.log" | grep -v "prompt eval" | tail -1 | sed -E 's/.*= *([0-9.]+) *ms.*/\1/')
    PFTPS=$(grep -E "prompt eval time" "$IDIR/stderr.log" | tail -1 | sed -E 's/.*\( *([0-9.]+) *tokens per second\)/\1/')
    TOT=$(grep -E "total time " "$IDIR/stderr.log" | tail -1 | sed -E 's/.*= *([0-9.]+) *ms.*/\1/')
    : "${PF:=0}" "${DT:=0}" "${DM:=0}" "${PFTPS:=0}" "${TOT:=0}"

    echo "$ITER,$ELAPSED,$EXIT,$PF,$DM,$DT,$PFTPS,$TOT" >> "$CELL_DIR/stress.csv"
    if [ $(( ITER % 5 )) -eq 0 ]; then
        echo "[$(date)]   llamacpp iter=$ITER t=${ELAPSED}s exit=$EXIT decode_tps=$DT prefill_tps=$PFTPS" >> "$PROG"
    fi
done

kill $SAMPLER 2>/dev/null
pgrep -f sample_sensors | xargs -r kill 2>/dev/null
sleep 2
echo "[$(date)] === DONE llamacpp baseline: iters=$ITER ===" >> "$PROG"

# Restore DVFS
echo "[$(date)] restoring DVFS via pin_dvfs.sh" >> "$PROG"
su -c sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh restore 2>> "$PROG"

touch "$OUT_DIR/DONE"
