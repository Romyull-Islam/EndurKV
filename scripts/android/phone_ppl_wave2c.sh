#!/bin/bash
# phone_ppl_wave2c.sh — CORRECTED PPL evaluation.
#
# Wave-2  used 9-token seed + 1K eval → no eviction pressure → all tied
# Wave-2b used 3K seed + 1K eval but text was sliced at byte boundary across
#            different WT2 articles → vanilla PPL inflated, results unusable
#
# Wave-2c fixes both:
#   - short seed (9 tokens, just sets up the cb_eval path)
#   - long CONTINUOUS eval text (~16K tokens from a single WT2 region)
#   - K-sweep over {1024, 1500, 2200} to find the eviction curve
#
# How eviction engages naturally:
#   - First ~K eval tokens scored with vanilla-equivalent cache (no eviction)
#   - As cache grows past K_nominal, eviction triggers per step
#   - Subsequent tokens are scored under eviction pressure
#   - vanilla baseline PPL matches canonical llama-perplexity (~7-8 on Llama-1B)
#   - eviction policies start to diverge as cache hits K threshold
#
# Per-step CSV is written by the patched binary (each step records log_prob,
# n_kv_cells), so we can plot the PPL trajectory and see exactly when each
# policy starts degrading.

set -e
export PATH=/home/mislam22/tools/platform-tools:$PATH

POLICIES="vanilla v1 tova"
K_BUDGETS="512 1024"           # K sweep — CMIYC protocol
PROMPT_IDS="wiki_seed_001"           # short seed
EVAL_TEXT=corpora/wiki_eval_1500.txt # 64 KB continuous WT2
N_REPLICATES=1

MODELS=(
    "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf|Llama-3.2-1B|0|512|64|20480"
    "models/gemma-2-2b-it-Q4_K_M.gguf|Gemma-2-2B|0|512|64|8192"
    "models/Phi-3-mini-128k-instruct-Q4_K_M.gguf|Phi-3-128k|0|512|64|20480"
)
BIN_DIR=${BIN_DIR:-bin_cpu}

OUT_BASE_PHONE="/data/local/tmp/endurkv/logs/ppl2c_$(date +%s)"
OUT_BASE_HOST="/home/mislam22/EndurKV_workspace/phone-logs/$(basename $OUT_BASE_PHONE)"
mkdir -p "$OUT_BASE_HOST"
PROG_LOG="$OUT_BASE_HOST/progress.log"

# Count cells: 3 models × 3 policies × 3 K values × 1 rep = 27
total=0
for _ in "${MODELS[@]}"; do for p in $POLICIES; do for k in $K_BUDGETS; do
    for pid in $PROMPT_IDS; do for r in $(seq 1 $N_REPLICATES); do
        total=$((total+1))
done; done; done; done; done

{
echo "[$(date)] ppl_wave2c_start"
echo "  short seed = wiki_seed_001 (9 tokens)"
echo "  eval text  = $EVAL_TEXT (64 KB continuous WT2)"
echo "  K sweep:    $K_BUDGETS"
echo "  total cells: $total  (=3 models × 3 policies × 3 K × 1 rep)"
} | tee -a "$PROG_LOG"

adb shell "mkdir -p $OUT_BASE_PHONE"

run_count=0
for MODEL_ENTRY in "${MODELS[@]}"; do
    IFS='|' read -r MODEL MODEL_TAG NGL NBATCH UB CTX <<< "$MODEL_ENTRY"
    echo "" | tee -a "$PROG_LOG"
    echo "===== $MODEL_TAG  ctx=$CTX =====" | tee -a "$PROG_LOG"

    for r in $(seq 1 $N_REPLICATES); do
    for PROMPT_ID in $PROMPT_IDS; do
        for K in $K_BUDGETS; do
            for POLICY in $POLICIES; do
                run_count=$((run_count+1))
                RUN_DIR_PHONE="$OUT_BASE_PHONE/$MODEL_TAG/$POLICY/K${K}/$PROMPT_ID/rep${r}"
                echo "[$(date)] [${run_count}/${total}] model=$MODEL_TAG policy=$POLICY K=$K" | tee -a "$PROG_LOG"

                adb shell "
mkdir -p $RUN_DIR_PHONE
cd /data/local/tmp/endurkv
LD_LIBRARY_PATH=$BIN_DIR $BIN_DIR/eviction_bench \
  --model $MODEL \
  --prompt prompts/${PROMPT_ID}.txt \
  --prompt-id $PROMPT_ID \
  --policy $POLICY \
  --k-nominal $K \
  --max-tokens 0 \
  --ctx-size $CTX \
  --seed 42 \
  --threads 4 \
  --n-gpu-layers $NGL \
  --n-batch $NBATCH \
  --ubatch-size $UB \
  --n-sink 4 \
  --eval-mode ppl \
  --eval-text $EVAL_TEXT \
  --out-csv  $RUN_DIR_PHONE/steps.csv \
  --out-meta $RUN_DIR_PHONE/meta.json 2>$RUN_DIR_PHONE/stderr.log
echo \"  exit=\$?\"
" 2>&1 | tee -a "$PROG_LOG"

                LOCAL_DIR="$OUT_BASE_HOST/$MODEL_TAG/$POLICY/K${K}/$PROMPT_ID/rep${r}"
                mkdir -p "$LOCAL_DIR"
                adb pull -p "$RUN_DIR_PHONE/" "$LOCAL_DIR/" 2>&1 | tail -1 | tee -a "$PROG_LOG"
                META="$LOCAL_DIR/$(basename $RUN_DIR_PHONE)/meta.json"
                if [ -f "$META" ]; then
                    grep -E '"(perplexity|mean_mass_retained|mean_retention_ratio)"' \
                        "$META" 2>/dev/null | sed 's/^/    /' | tee -a "$PROG_LOG"
                fi
            done
        done
    done
    done
done

{
echo ""
echo "[$(date)] ppl_wave2c_done total=$run_count"
echo "Local logs: $OUT_BASE_HOST"
} | tee -a "$PROG_LOG"
