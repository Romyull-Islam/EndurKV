#!/bin/bash
# phone_ppl_eval_wave2.sh — proper teacher-forced PPL evaluation.
#
# Methodology (matches `llama-perplexity` / standard NLP):
#   1. Prefill the LongBench prompt under the eviction policy.
#   2. Apply post-prefill eviction (now KV holds K_nominal positions/head).
#   3. Teacher-force a held-out reference text (~1024 tokens of WikiText-2).
#      For each ref token, take the RAW logits (no rep-penalty, no temperature,
#      no sampling) and compute -log P(ref_token | prefix_so_far).
#   4. Mean NLL → PPL = exp(mean_nll).
#
# This gives a number that's *directly* comparable across policies because:
#   - The reference text is identical for every (model, policy, prompt) cell.
#   - No sampling RNG ⇒ no variance from RNG.
#   - Raw logits ⇒ no temperature/rep-penalty distortion.
#   - Vanilla and policy use the same eval loop ⇒ vanilla PPL is well-defined.
#
# Compare policy_PPL / vanilla_PPL per (model, prompt) cell — the right ratio
# to report in the paper.

set -e
export PATH=/home/mislam22/tools/platform-tools:$PATH

POLICIES="vanilla v1 tova"
K_BUDGETS="1024"
# Intrinsic PPL on WikiText-2 — no LongBench prefix.
# We pass a tiny seed-prompt as --prompt (eviction_bench needs *some* prefill
# to set up the cb_eval callback path) and the long WT2 slice as --eval-text.
# The seed-prompt is small enough that it doesn't bias the PPL number meaningfully.
PROMPT_DIR=prompts
PROMPT_IDS="wiki_seed_001"
EVAL_TEXT=corpora/wiki_ref_4k.txt
N_REPLICATES=1                  # PPL is deterministic; no need to replicate

# model_path|model_tag|ngl|n_batch|ubatch|ctx_size
MODELS=(
    "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf|Llama-3.2-1B|0|512|64|10240"
    "models/gemma-2-2b-it-Q4_K_M.gguf|Gemma-2-2B|0|512|64|8192"
    "models/Phi-3-mini-128k-instruct-Q4_K_M.gguf|Phi-3-128k|0|512|64|10240"
)

# Use CPU-only stack — GPU has output-degeneration bug on this Adreno
BIN_DIR=${BIN_DIR:-bin_cpu}

OUT_BASE_PHONE="/data/local/tmp/endurkv/logs/ppl_$(date +%s)"
OUT_BASE_HOST="/home/mislam22/EndurKV_workspace/phone-logs/$(basename $OUT_BASE_PHONE)"
mkdir -p "$OUT_BASE_HOST"
PROG_LOG="$OUT_BASE_HOST/progress.log"

# Count cells
total=0
for _ in "${MODELS[@]}"; do for p in $POLICIES; do for k in $K_BUDGETS; do
    for pid in $PROMPT_IDS; do for r in $(seq 1 $N_REPLICATES); do
        total=$((total+1))
done; done; done; done; done

{
echo "[$(date)] ppl_eval_start"
echo "  models: ${#MODELS[@]}  policies: $POLICIES  K: $K_BUDGETS"
echo "  prompts: $PROMPT_IDS"
echo "  eval-text: $EVAL_TEXT  (raw-logit teacher-forced PPL)"
echo "  total cells: $total"
} | tee -a "$PROG_LOG"

adb shell "mkdir -p $OUT_BASE_PHONE"

run_count=0
for MODEL_ENTRY in "${MODELS[@]}"; do
    IFS='|' read -r MODEL MODEL_TAG NGL NBATCH UB CTX <<< "$MODEL_ENTRY"
    echo "" | tee -a "$PROG_LOG"
    echo "===== $MODEL_TAG  ngl=$NGL =====" | tee -a "$PROG_LOG"

    for r in $(seq 1 $N_REPLICATES); do
    for PROMPT_ID in $PROMPT_IDS; do
        for POLICY in $POLICIES; do
            for K in $K_BUDGETS; do
                run_count=$((run_count+1))
                RUN_DIR_PHONE="$OUT_BASE_PHONE/$MODEL_TAG/$POLICY/K${K}/$PROMPT_ID/rep${r}"
                echo "[$(date)] [${run_count}/${total}] model=$MODEL_TAG policy=$POLICY K=$K prompt=$PROMPT_ID" | tee -a "$PROG_LOG"

                adb shell "
mkdir -p $RUN_DIR_PHONE
cd /data/local/tmp/endurkv

LD_LIBRARY_PATH=$BIN_DIR $BIN_DIR/eviction_bench \
  --model $MODEL \
  --prompt $PROMPT_DIR/${PROMPT_ID}.txt \
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
EXIT=\$?
echo \"  exit=\$EXIT\"
" 2>&1 | tee -a "$PROG_LOG"

                LOCAL_DIR="$OUT_BASE_HOST/$MODEL_TAG/$POLICY/K${K}/$PROMPT_ID/rep${r}"
                mkdir -p "$LOCAL_DIR"
                adb pull -p "$RUN_DIR_PHONE/" "$LOCAL_DIR/" 2>&1 | tail -1 | tee -a "$PROG_LOG"
                META="$LOCAL_DIR/$(basename $RUN_DIR_PHONE)/meta.json"
                if [ -f "$META" ]; then
                    grep -E '"(perplexity|mean_nll|mean_mass_retained|mean_retention_ratio)"' \
                        "$META" 2>/dev/null | sed 's/^/    /' | tee -a "$PROG_LOG"
                fi
            done
        done
    done
    done
done

{
echo ""
echo "[$(date)] ppl_eval_done  total=$run_count"
echo "Local logs: $OUT_BASE_HOST"
} | tee -a "$PROG_LOG"
