#!/bin/bash
# phone_full_sweep_cpu.sh — all-CPU re-run after diagnosing the GPU output-degeneration bug.
#
# What changed vs Wave-1:
#   1. ngl=0 (CPU only) — stock llama-completion confirmed CPU output is correct.
#   2. Uses chat-templated prompts (prompts_chat/<model>/<prompt>.txt) instead of raw text.
#   3. --greedy decoding (deterministic, removes RNG noise).
#   4. bin_cpu/ binary stack (no Vulkan registered → no GPU mis-scheduling).
#
# Grid:
#   3 models × 3 policies × 3 prompts × 1 rep = 27 cells (greedy is deterministic).
# Wall-time estimate:
#   Llama-1B  : ~1-2 min/cell × 9 = ~15 min
#   Gemma-2-2B: ~3-6 min/cell × 9 = ~45 min
#   Phi-3-128k: ~6-10 min/cell × 9 = ~90 min
#   Total     : ~2.5 hr

set -e
export PATH=/home/mislam22/tools/platform-tools:$PATH

POLICIES="h2o"
K_BUDGETS="1024"
PROMPT_IDS="qasper_pub_001 hotpotqa_pub_001 multifieldqa_en_pub_001"
N_REPLICATES=1
MAX_TOKENS=128
COOL_THRESH_C=38
COOL_MAX_WAIT=300

# CPU-only stack (no Vulkan registered)
BIN_DIR=bin_cpu

# model_path|model_tag|ngl|n_batch|ubatch|ctx_size
MODELS=(
    "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf|Llama-3.2-1B|0|512|64|10240"
    "models/gemma-2-2b-it-Q4_K_M.gguf|Gemma-2-2B|0|512|64|8192"
    "models/Phi-3-mini-128k-instruct-Q4_K_M.gguf|Phi-3-128k|0|512|64|10240"
)

OUT_BASE_PHONE="/data/local/tmp/endurkv/logs/h2o_sweep_$(date +%s)"
OUT_BASE_HOST="/home/mislam22/EndurKV_workspace/phone-logs/$(basename $OUT_BASE_PHONE)"
mkdir -p "$OUT_BASE_HOST"
PROG_LOG="$OUT_BASE_HOST/progress.log"

total=0
for _ in "${MODELS[@]}"; do for p in $POLICIES; do for k in $K_BUDGETS; do
    for pid in $PROMPT_IDS; do for r in $(seq 1 $N_REPLICATES); do
        total=$((total+1))
done; done; done; done; done

{
echo "[$(date)] cpu_sweep_start"
echo "  models: ${#MODELS[@]} (Llama-3.2-1B, Gemma-2-2B, Phi-3-mini-128k)"
echo "  policies: $POLICIES"
echo "  K: $K_BUDGETS"
echo "  prompts: $PROMPT_IDS  (chat-templated, in prompts_chat/<model>/)"
echo "  decoding: greedy (deterministic, no RNG)"
echo "  total cells: $total"
echo "  backend: CPU-only via $BIN_DIR/"
} | tee -a "$PROG_LOG"

adb shell "mkdir -p $OUT_BASE_PHONE"

run_count=0
for MODEL_ENTRY in "${MODELS[@]}"; do
    IFS='|' read -r MODEL MODEL_TAG NGL NBATCH UB CTX <<< "$MODEL_ENTRY"
    echo "" | tee -a "$PROG_LOG"
    echo "===== $MODEL_TAG  ctx=$CTX =====" | tee -a "$PROG_LOG"

    for r in $(seq 1 $N_REPLICATES); do
    for PROMPT_ID in $PROMPT_IDS; do
        for POLICY in $POLICIES; do
            for K in $K_BUDGETS; do
                run_count=$((run_count+1))
                RUN_DIR_PHONE="$OUT_BASE_PHONE/$MODEL_TAG/$POLICY/K${K}/$PROMPT_ID/rep${r}"
                echo "" | tee -a "$PROG_LOG"
                echo "[$(date)] [${run_count}/${total}] model=$MODEL_TAG policy=$POLICY K=$K prompt=$PROMPT_ID" | tee -a "$PROG_LOG"

                adb shell "
mkdir -p $RUN_DIR_PHONE
cd /data/local/tmp/endurkv

sh scripts/phone_cool_then_run.sh --out-dir $RUN_DIR_PHONE --thresh-c $COOL_THRESH_C --max-wait $COOL_MAX_WAIT -- true

sh scripts/sample_sensors.sh --out $RUN_DIR_PHONE/sensors.csv --hz 10 &
SAMPLER=\$!

LD_LIBRARY_PATH=$BIN_DIR $BIN_DIR/eviction_bench \
  --model $MODEL \
  --prompt prompts_chat/$MODEL_TAG/${PROMPT_ID}.txt \
  --prompt-id $PROMPT_ID \
  --policy $POLICY \
  --k-nominal $K \
  --max-tokens $MAX_TOKENS \
  --ctx-size $CTX \
  --seed 42 \
  --threads 4 \
  --n-gpu-layers $NGL \
  --n-batch $NBATCH \
  --ubatch-size $UB \
  --n-sink 4 \
  --repeat-penalty 1.1 \
  --repeat-last-n 64 \
  --greedy \
  --out-csv  $RUN_DIR_PHONE/steps.csv \
  --out-meta $RUN_DIR_PHONE/meta.json \
  --out-gen  $RUN_DIR_PHONE/gen.txt 2>$RUN_DIR_PHONE/stderr.log
EXIT=\$?
kill \$SAMPLER 2>/dev/null
wait \$SAMPLER 2>/dev/null
echo \"  exit=\$EXIT\"
" 2>&1 | tee -a "$PROG_LOG"

                LOCAL_DIR="$OUT_BASE_HOST/$MODEL_TAG/$POLICY/K${K}/$PROMPT_ID/rep${r}"
                mkdir -p "$LOCAL_DIR"
                adb pull -p "$RUN_DIR_PHONE/" "$LOCAL_DIR/" 2>&1 | tail -1 | tee -a "$PROG_LOG"
                META="$LOCAL_DIR/$(basename $RUN_DIR_PHONE)/meta.json"
                if [ -f "$META" ]; then
                    grep -E '"(decode_tps|peak_kv_mb|peak_rss_kb|perplexity|mean_mass_retained|mean_retention_ratio|mean_eviction_efficiency|prefill_ms|evicted_total)"' \
                        "$META" 2>/dev/null | sed 's/^/    /' | tee -a "$PROG_LOG"
                fi
            done
        done
    done
    done
done

{
echo ""
echo "[$(date)] cpu_sweep_done total=$run_count"
echo "Local logs: $OUT_BASE_HOST"
} | tee -a "$PROG_LOG"
