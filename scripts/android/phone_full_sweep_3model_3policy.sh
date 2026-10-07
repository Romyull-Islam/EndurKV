#!/bin/bash
# Phone GPU sweep: MODELS x POLICIES x K x prompts x reps (settings below).
# Sampling settings are the same for every policy. Vanilla runs FA-on, eviction
# policies run FA-off because they need kq_soft_max. ubatch=64 keeps Vulkan on
# Adreno under the TDR limit.
# Phone output: /data/local/tmp/endurkv/logs/sweep3M_<ts>/<model>/<policy>/K<K>/<prompt>/rep<N>/
# Pulled to:    /home/mislam22/EndurKV_workspace/phone-logs/sweep3M_<ts>/

set -e
export PATH=/home/mislam22/tools/platform-tools:$PATH

POLICIES="v1 vanilla tova"
# One K budget keeps the sweep short.
K_BUDGETS="1024"
PROMPT_DIR=prompts/longbench
# 3 prompts, one per task category.
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
# Models stay under 7B: on Adreno 840 the TDR watchdog kills attention kernels
# of 7B+ models at LongBench prompt lengths (DeviceLost).
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
    echo "=" | tee -a "$PROG_LOG"
    echo "[$(date)] MODEL=$MODEL_TAG  ngl=$NGL  n_batch=$NBATCH  ubatch=$UB  ctx=$CTX" | tee -a "$PROG_LOG"
    echo "=" | tee -a "$PROG_LOG"

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
