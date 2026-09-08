#!/system/bin/sh
# phone_run_longbench_sweep.sh — Master on-phone runner.
#
# Loops over (policy × K_nominal × LongBench prompt) and runs eviction_bench
# with thermal/memory/latency capture. Per the deployment plan, deploys our
# v1 first, then vanilla, then TOVA, then PyramidKV.
#
# Run on phone, from /data/local/tmp/endurkv/:
#   sh scripts/phone_run_longbench_sweep.sh --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
#
# Outputs (per run) in logs/sweep_<timestamp>/<policy>/<K>/<prompt_id>/:
#   steps.csv      per-decode-step latency, n_kv cells, RSS
#   meta.json      run summary (prefill ms, decode tok/s, peak RSS, peak KV)
#   gen.txt        generated text
#   sensors.csv    10 Hz thermal+battery sampling
#
# Default: 1 model × 4 policies × 2 K budgets × 25 prompts = 200 runs.
# Estimated time: ~3-6 hours on Snapdragon (varies by ctx length + cooldowns).

set -e

# ---- defaults / args ----
MODEL=""
POLICIES="v1 vanilla tova pyramid"
K_BUDGETS="512 1024"
PROMPT_DIR="prompts/longbench"
CTX_SIZE=12288   # 12K — fits ~10K prompts + 2K decode
MAX_TOKENS=64
SEED=42
N_THREADS=4
COOL_TEMP_C=40  # cool down if shell_front skin temp > this
COOL_SLEEP_S=30
OUT_BASE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --policies)    POLICIES="$2"; shift 2 ;;
        --k-budgets)   K_BUDGETS="$2"; shift 2 ;;
        --prompts)     PROMPT_DIR="$2"; shift 2 ;;
        --ctx-size)    CTX_SIZE="$2"; shift 2 ;;
        --max-tokens)  MAX_TOKENS="$2"; shift 2 ;;
        --threads)     N_THREADS="$2"; shift 2 ;;
        --cool-temp)   COOL_TEMP_C="$2"; shift 2 ;;
        --cool-sleep)  COOL_SLEEP_S="$2"; shift 2 ;;
        --out-base)    OUT_BASE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -z "$MODEL" ] && { echo "--model required" >&2; exit 1; }
[ -z "$OUT_BASE" ] && OUT_BASE="logs/sweep_$(date +%s)"
mkdir -p "$OUT_BASE"

ROOT=/data/local/tmp/endurkv
BIN="$ROOT/bin"
EVICT="$BIN/eviction_bench"
SAMPLER="$ROOT/scripts/sample_sensors.sh"
export LD_LIBRARY_PATH="$BIN"

# Quick cool-down check
cool_phone() {
    if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        # convert millidegrees to degrees
        temp=$(awk '{printf "%d", $1/1000}' /sys/class/thermal/thermal_zone0/temp)
        if [ "$temp" -gt "$COOL_TEMP_C" ]; then
            echo "  [cool] zone0=${temp}C > ${COOL_TEMP_C}C; sleeping ${COOL_SLEEP_S}s"
            sleep "$COOL_SLEEP_S"
        fi
    fi
}

# Get model basename for log structure
MODEL_TAG=$(basename "$MODEL" .gguf)
SWEEP_LOG="$OUT_BASE/sweep_progress.log"
echo "[$(date)] sweep_start model=$MODEL_TAG policies='$POLICIES' K='$K_BUDGETS'" >> "$SWEEP_LOG"

run_count=0
total=$(echo $POLICIES | wc -w)
total=$((total * $(echo $K_BUDGETS | wc -w) * $(ls $PROMPT_DIR/*.txt | wc -l)))
echo "[sweep] total runs: $total"

for POLICY in $POLICIES; do
    for K in $K_BUDGETS; do
        for PROMPT in $PROMPT_DIR/*.txt; do
            PROMPT_ID=$(basename "$PROMPT" .txt)
            RUN_DIR="$OUT_BASE/$MODEL_TAG/$POLICY/K${K}/$PROMPT_ID"
            mkdir -p "$RUN_DIR"
            cool_phone
            run_count=$((run_count + 1))
            echo "[$(date)] [${run_count}/${total}] policy=$POLICY K=$K prompt=$PROMPT_ID" | tee -a "$SWEEP_LOG"

            # Start sensor sampling for this run
            sh "$SAMPLER" --out "$RUN_DIR/sensors.csv" --hz 10 &
            SAMPLER_PID=$!

            # Run eviction_bench
            "$EVICT" \
                --model "$MODEL" \
                --prompt "$PROMPT" \
                --prompt-id "$PROMPT_ID" \
                --policy "$POLICY" \
                --k-nominal "$K" \
                --max-tokens "$MAX_TOKENS" \
                --ctx-size "$CTX_SIZE" \
                --seed "$SEED" \
                --threads "$N_THREADS" \
                --out-csv "$RUN_DIR/steps.csv" \
                --out-meta "$RUN_DIR/meta.json" \
                --out-gen "$RUN_DIR/gen.txt" \
                2> "$RUN_DIR/stderr.log" || echo "  [warn] run failed"

            # Stop sensor sampler
            kill "$SAMPLER_PID" 2>/dev/null || true
            wait "$SAMPLER_PID" 2>/dev/null || true

            # Quick summary
            if [ -f "$RUN_DIR/meta.json" ]; then
                grep -E '"(decode_tps|peak_kv_cells|peak_rss_kb|prefill_ms)"' "$RUN_DIR/meta.json" \
                    | sed 's/^/    /'
            fi
        done
    done
done

echo "[$(date)] sweep_done runs=$run_count" >> "$SWEEP_LOG"
echo ""
echo "[sweep] DONE. Pull results with:"
echo "  adb pull $ROOT/$OUT_BASE ."
