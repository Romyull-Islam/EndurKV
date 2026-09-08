#!/system/bin/sh
# phone_sweep_cooled.sh — Paper-grade sweep with cool-down + replicates.
#
# Standard practice for mobile LLM benchmarks:
#   1. Cool phone to skin ≤ 38°C OR max 5 min wait before each run
#   2. Run eviction_bench with full instrumentation (eviction_bench + sample_sensors)
#   3. Record starting temps + cool-down duration alongside the run meta
#   4. Repeat each (policy, K, prompt) cell N_REPLICATES times for variance
#   5. Order: policies cycle per prompt to spread thermal bias
#
# Usage on phone:
#   sh scripts/phone_sweep_cooled.sh --model models/X.gguf
#
# Outputs (in logs/sweep_cooled_<ts>/):
#   <policy>/K<K>/<prompt_id>/rep<N>/
#     meta.json, steps.csv, gen.txt, sensors.csv, cooldown.json, stderr.log

set -e

MODEL=""
POLICIES="v1 vanilla tova pyramid"
K_BUDGETS="512 1024"
PROMPT_DIR="prompts/longbench"
CTX_SIZE=12288
MAX_TOKENS=128
SEED=42
N_THREADS=4
N_REPLICATES=3
COOL_THRESH_C=38
COOL_MAX_WAIT=300
OUT_BASE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --model)         MODEL="$2"; shift 2 ;;
        --policies)      POLICIES="$2"; shift 2 ;;
        --k-budgets)     K_BUDGETS="$2"; shift 2 ;;
        --prompts)       PROMPT_DIR="$2"; shift 2 ;;
        --ctx-size)      CTX_SIZE="$2"; shift 2 ;;
        --max-tokens)    MAX_TOKENS="$2"; shift 2 ;;
        --threads)       N_THREADS="$2"; shift 2 ;;
        --replicates)    N_REPLICATES="$2"; shift 2 ;;
        --cool-thresh-c) COOL_THRESH_C="$2"; shift 2 ;;
        --cool-max-wait) COOL_MAX_WAIT="$2"; shift 2 ;;
        --out-base)      OUT_BASE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -z "$MODEL" ] && { echo "--model required" >&2; exit 1; }
[ -z "$OUT_BASE" ] && OUT_BASE="logs/sweep_cooled_$(date +%s)"
mkdir -p "$OUT_BASE"

ROOT=/data/local/tmp/endurkv
BIN="$ROOT/bin"
EVICT="$BIN/eviction_bench"
SAMPLER="$ROOT/scripts/sample_sensors.sh"
COOL="$ROOT/scripts/phone_cool_then_run.sh"
export LD_LIBRARY_PATH="$BIN"

MODEL_TAG=$(basename "$MODEL" .gguf)
SWEEP_LOG="$OUT_BASE/sweep_progress.log"
echo "[$(date)] sweep_start model=$MODEL_TAG policies='$POLICIES' K='$K_BUDGETS' replicates=$N_REPLICATES" >> "$SWEEP_LOG"

# Count total cells
n_p=$(echo $POLICIES | wc -w)
n_k=$(echo $K_BUDGETS | wc -w)
n_pr=$(ls $PROMPT_DIR/*.txt | wc -l)
total=$((n_p * n_k * n_pr * N_REPLICATES))
echo "[sweep] total runs: $total (policies=$n_p × K=$n_k × prompts=$n_pr × replicates=$N_REPLICATES)"

run_count=0
# Outer: replicate index (so each rep cycles through all combos before next rep)
# This spreads any thermal bias evenly across policies.
REP=0
while [ "$REP" -lt "$N_REPLICATES" ]; do
    REP=$((REP + 1))
    # Inner: prompt → policy → K
    # Per replicate, cycle policies in different order to reduce sequence bias
    if [ $((REP % 2)) -eq 1 ]; then
        ORDER="$POLICIES"
    else
        # Reverse for even replicates
        ORDER=$(echo $POLICIES | awk '{for(i=NF;i>=1;i--) printf "%s ", $i}')
    fi

    for PROMPT in $PROMPT_DIR/*.txt; do
        PROMPT_ID=$(basename "$PROMPT" .txt)
        for POLICY in $ORDER; do
            for K in $K_BUDGETS; do
                RUN_DIR="$OUT_BASE/$MODEL_TAG/$POLICY/K${K}/$PROMPT_ID/rep${REP}"
                mkdir -p "$RUN_DIR"
                run_count=$((run_count + 1))

                # COOL DOWN first
                echo "[$(date)] [${run_count}/${total}] rep=$REP policy=$POLICY K=$K prompt=$PROMPT_ID — cooling..." | tee -a "$SWEEP_LOG"
                sh "$COOL" --out-dir "$RUN_DIR" --thresh-c "$COOL_THRESH_C" --max-wait "$COOL_MAX_WAIT" -- \
                    true   # cool-down wrapper, then no-op (we run eviction_bench manually below)

                # Start sensors
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
                    2> "$RUN_DIR/stderr.log" \
                    || echo "  [warn] run failed (continuing)"

                kill "$SAMPLER_PID" 2>/dev/null || true
                wait "$SAMPLER_PID" 2>/dev/null || true

                # Quick summary
                if [ -f "$RUN_DIR/meta.json" ]; then
                    grep -E '"(decode_tps|peak_kv_mb|peak_rss_kb|prefill_ms|perplexity|evicted_prefill)"' "$RUN_DIR/meta.json" | sed 's/^/    /'
                fi
            done
        done
    done
done

echo "[$(date)] sweep_done total_runs=$run_count" >> "$SWEEP_LOG"
echo ""
echo "[sweep] DONE. Pull results with:"
echo "  adb pull $ROOT/$OUT_BASE ."
