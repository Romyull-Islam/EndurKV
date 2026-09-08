#!/bin/bash
# phone_full_sweep_3model_3policy.sh — full GPU sweep across 3 models × 3 policies.
#
# Models (Q4_K_M, on phone):
#   1. Llama-3.2-1B-Instruct          ngl=16  (all layers on GPU)
#   2. Phi-3-mini-4k-instruct (~3.8B) ngl=32
#   3. Llama-3.1-8B-Instruct          ngl=32  (NOT 33 → output layer crashes Adreno)
#
# Policies (priority order):  v1 (ours)  →  vanilla  →  tova
# Prompts:                    5 LongBench tasks
# K budgets:                  1024  (aggressive ≈12%),  2048 (paper-standard ≈25%)
# Replicates:                 3 per cell → mean ± std
# Sampling protocol:          IDENTICAL across all policies
#     repeat-penalty 1.1  temperature 0.8  top-k 40  top-p 0.95  n-sink 4
# Vanilla:                    FA-on (production-realistic baseline)
# Eviction policies:          FA-off (need kq_soft_max)
# Vulkan / Adreno safety:     ubatch=64  (TDR-safe — see README)
#
# Cells: 3 models × 3 policies × 5 prompts × 2 K × 3 reps = 270 runs
# Wall time estimate (GPU): 3 - 6 min/cell on 8B, less on smaller models
#   → roughly 12 - 24 h total
#
# Output:  /data/local/tmp/endurkv/logs/sweep3M_<ts>/<model>/<policy>/K<K>/<prompt>/rep<N>/
#
# Pulled to: /home/mislam22/EndurKV_workspace/phone-logs/sweep3M_<ts>/

set -e
export PATH=/home/mislam22/tools/platform-tools:$PATH

POLICIES="v1 vanilla tova"
# Single K budget (1024) for the first wave — fits 8B CPU runs in ~10h total.
# Add K=2048 in a follow-on phase once Wave-1 results land.
K_BUDGETS="1024"
PROMPT_DIR=prompts/longbench
# 3 prompts (one per task category) — Wave-1 grid is intentionally small.
# Full 5-prompt grid is a Wave-2 expansion.
PROMPT_IDS="qasper_pub_001 hotpotqa_pub_001 multifieldqa_en_pub_001"
N_REPLICATES=2
MAX_TOKENS=128
COOL_THRESH_C=38
COOL_MAX_WAIT=300
UBATCH=${UBATCH:-64}

# GPU stack: bin/  has the Vulkan-linked eviction_bench + libggml-vulkan.so
# CPU stack: bin_cpu/  has the CPU-only eviction_bench (no Vulkan registration)
BIN_DIR=${BIN_DIR:-bin}

# model_path|model_tag|ngl|n_batch|ubatch|ctx_size
# All-GPU sweep across 3 distinct architectures (Llama, Gemma, Phi), all <7B.
# Why not 7B/8B: Adreno 840 TDR watchdog kills attention kernels at long prompts
# for any 7B+ model (Llama-8B, Mistral-7B, Qwen2-7B, DeepSeek-Llama-8B all
# DeviceLost-crashed at LongBench prompt lengths). Phi-3 (3.8B) is the upper
# end of what fits on this GPU.
MODELS=(
    "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf|Llama-3.2-1B|16|512|64|10240"
    "models/gemma-2-2b-it-Q4_K_M.gguf|Gemma-2-2B|26|512|64|8192"
    "models/Phi-3-mini-128k-instruct-Q4_K_M.gguf|Phi-3-128k|32|512|64|10240"
)

OUT_BASE_PHONE="/data/local/tmp/endurkv/logs/sweep3M_$(date +%s)"
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
echo "[$(date)] sweep3M_start"
echo "  models: ${#MODELS[@]} (Llama-3.2-1B, Gemma-2-2B, Phi-3-mini-128k)"
echo "  policies: $POLICIES"
echo "  K: $K_BUDGETS"
echo "  prompts: $PROMPT_IDS"
echo "  replicates: $N_REPLICATES"
echo "  ubatch: $UBATCH    max-tokens: $MAX_TOKENS"
echo "  total runs: $total"
echo "  out (phone): $OUT_BASE_PHONE"
echo "  out (host):  $OUT_BASE_HOST"
} | tee -a "$PROG_LOG"

adb shell "mkdir -p $OUT_BASE_PHONE"

run_count=0
for MODEL_ENTRY in "${MODELS[@]}"; do
    IFS='|' read -r MODEL MODEL_TAG NGL NBATCH UB CTX <<< "$MODEL_ENTRY"
    echo "" | tee -a "$PROG_LOG"
    echo "=========================================================================" | tee -a "$PROG_LOG"
    echo "[$(date)] MODEL=$MODEL_TAG  ngl=$NGL  n_batch=$NBATCH  ubatch=$UB  ctx=$CTX" | tee -a "$PROG_LOG"
    echo "=========================================================================" | tee -a "$PROG_LOG"

    REP=0
    while [ $REP -lt $N_REPLICATES ]; do
        REP=$((REP+1))
        # Alternate policy order per rep to spread thermal bias
        if [ $((REP % 2)) -eq 1 ]; then
            ORDER="$POLICIES"
        else
            ORDER=$(echo $POLICIES | awk '{for(i=NF;i>=1;i--) printf "%s ", $i}')
        fi

        for PROMPT_ID in $PROMPT_IDS; do
            for POLICY in $ORDER; do
                for K in $K_BUDGETS; do
                    run_count=$((run_count+1))
                    RUN_DIR_PHONE="$OUT_BASE_PHONE/$MODEL_TAG/$POLICY/K${K}/$PROMPT_ID/rep${REP}"
                    echo "" | tee -a "$PROG_LOG"
                    echo "[$(date)] [${run_count}/${total}] model=$MODEL_TAG policy=$POLICY K=$K prompt=$PROMPT_ID rep=$REP" | tee -a "$PROG_LOG"

                    adb shell "
mkdir -p $RUN_DIR_PHONE
cd /data/local/tmp/endurkv

# Cool-down with starting-temp capture
sh scripts/phone_cool_then_run.sh --out-dir $RUN_DIR_PHONE --thresh-c $COOL_THRESH_C --max-wait $COOL_MAX_WAIT -- true

# Start sensor sampling for this run
sh scripts/sample_sensors.sh --out $RUN_DIR_PHONE/sensors.csv --hz 10 &
SAMPLER=\$!

# Eviction bench (CPU-uniform stack)
LD_LIBRARY_PATH=$BIN_DIR $BIN_DIR/eviction_bench \
  --model $MODEL \
  --prompt $PROMPT_DIR/${PROMPT_ID}.txt \
  --prompt-id $PROMPT_ID \
  --policy $POLICY \
  --k-nominal $K \
  --max-tokens $MAX_TOKENS \
  --ctx-size $CTX \
  --seed \$((REP * 100 + 42)) \
  --threads 4 \
  --n-gpu-layers $NGL \
  --n-batch $NBATCH \
  --ubatch-size $UB \
  --n-sink 4 \
  --repeat-penalty 1.1 \
  --repeat-last-n 64 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 40 \
  --out-csv  $RUN_DIR_PHONE/steps.csv \
  --out-meta $RUN_DIR_PHONE/meta.json \
  --out-gen  $RUN_DIR_PHONE/gen.txt 2>$RUN_DIR_PHONE/stderr.log
EXIT=\$?
kill \$SAMPLER 2>/dev/null
wait \$SAMPLER 2>/dev/null
echo \"  exit=\$EXIT\"
" 2>&1 | tee -a "$PROG_LOG"

                    # Pull immediately for live aggregation
                    LOCAL_DIR="$OUT_BASE_HOST/$MODEL_TAG/$POLICY/K${K}/$PROMPT_ID/rep${REP}"
                    mkdir -p "$LOCAL_DIR"
                    adb pull -p "$RUN_DIR_PHONE/" "$LOCAL_DIR/" 2>&1 | tail -1 | tee -a "$PROG_LOG"
                    META="$LOCAL_DIR/$(basename $RUN_DIR_PHONE)/meta.json"
                    if [ -f "$META" ]; then
                        grep -E '"(decode_tps|peak_kv_mb|peak_rss_kb|perplexity|mean_mass_retained|mean_eviction_efficiency|prefill_ms)"' \
                            "$META" 2>/dev/null | sed 's/^/    /' | tee -a "$PROG_LOG"
                    fi
                done
            done
        done
    done
done

{
echo ""
echo "[$(date)] sweep3M_done total_runs=$run_count"
echo "Local logs: $OUT_BASE_HOST"
} | tee -a "$PROG_LOG"
