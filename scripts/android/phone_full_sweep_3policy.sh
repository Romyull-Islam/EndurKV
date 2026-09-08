#!/bin/bash
# phone_full_sweep_3policy.sh — Full sweep for our 3 priority policies.
#
# Configuration (publication-grade):
#   - 3 policies: v1 (ours) → vanilla → tova    (user-specified order)
#   - 5 prompts (1 per LongBench task)
#   - 2 K budgets: 1024 (current "stress") and 2048 (published-standard 25% retention)
#   - 3 replicates per cell — for mean ± std
#   - Standard sampling protocol (top-p 0.95, T=0.8, top-k 40, rep-penalty 1.1)
#     applied IDENTICALLY to all 3 policies (vanilla, v1, tova)
#   - Sink-token protection (n_sink=4) for v1 and tova
#   - Vanilla gets FA-on (production-realistic latency)
#   - Cool-down between runs (skin ≤ 38°C, max 5 min wait)
#
# Total: 3 × 5 × 2 × 3 = 90 runs
# Estimated wall time on Snapdragon 8 Elite Gen 5 (FA-off for eviction policies):
#   per run: ~35 min Llama-8B FA-off,  ~10 min vanilla FA-on
#   total:   ~30-40 hours
#
# Output dir: /data/local/tmp/endurkv/logs/sweep3_<timestamp>/
#   <policy>/K<K>/<prompt_id>/rep<N>/
#     meta.json, steps.csv, gen.txt, sensors.csv, cooldown.json, stderr.log

set -e
export PATH=/home/mislam22/tools/platform-tools:$PATH

MODEL=${MODEL:-models/Llama-3.1-8B-Instruct-Q4_K_M.gguf}
MODEL_TAG=$(basename "$MODEL" .gguf)
POLICIES="v1 vanilla tova"
K_BUDGETS="1024 2048"
N_GPU_LAYERS=${N_GPU_LAYERS:-0}
PROMPT_DIR=prompts/longbench
# Pick 5 prompts (one per task)
PROMPT_IDS="narrativeqa_pub_001 qasper_pub_001 hotpotqa_pub_001 gov_report_pub_001 multifieldqa_en_pub_001"
N_REPLICATES=3
MAX_TOKENS=128
CTX_SIZE=12288
COOL_THRESH_C=38
COOL_MAX_WAIT=300

OUT_BASE_PHONE="/data/local/tmp/endurkv/logs/sweep3_$(date +%s)"
OUT_BASE_HOST="/home/mislam22/EndurKV_workspace/phone-logs/$(basename $OUT_BASE_PHONE)"
mkdir -p "$OUT_BASE_HOST"

# Local progress log
PROG_LOG="$OUT_BASE_HOST/progress.log"
echo "[$(date)] sweep3_start model=$MODEL_TAG" | tee -a "$PROG_LOG"
echo "  policies: $POLICIES" | tee -a "$PROG_LOG"
echo "  K: $K_BUDGETS" | tee -a "$PROG_LOG"
echo "  prompts: $PROMPT_IDS" | tee -a "$PROG_LOG"
echo "  replicates: $N_REPLICATES, max_tokens: $MAX_TOKENS, ctx: $CTX_SIZE" | tee -a "$PROG_LOG"

total=0
for p in $POLICIES; do for k in $K_BUDGETS; do for pid in $PROMPT_IDS; do for r in $(seq 1 $N_REPLICATES); do
    total=$((total+1))
done; done; done; done
echo "  total runs: $total" | tee -a "$PROG_LOG"

adb shell "mkdir -p $OUT_BASE_PHONE"

run_count=0
REP=0
while [ $REP -lt $N_REPLICATES ]; do
    REP=$((REP+1))
    # Cycle policies: even reps use reversed order to spread thermal bias
    if [ $((REP % 2)) -eq 1 ]; then
        ORDER="$POLICIES"
    else
        ORDER=$(echo $POLICIES | awk '{for(i=NF;i>=1;i--) printf "%s ", $i}')
    fi

    for PROMPT_ID in $PROMPT_IDS; do
        for POLICY in $ORDER; do
            for K in $K_BUDGETS; do
                run_count=$((run_count+1))
                RUN_DIR_PHONE="$OUT_BASE_PHONE/$POLICY/K${K}/$PROMPT_ID/rep${REP}"
                echo "" | tee -a "$PROG_LOG"
                echo "[$(date)] [${run_count}/${total}] rep=$REP policy=$POLICY K=$K prompt=$PROMPT_ID" | tee -a "$PROG_LOG"

                adb shell "
mkdir -p $RUN_DIR_PHONE
cd /data/local/tmp/endurkv

# Cool-down with starting-temp capture
sh scripts/phone_cool_then_run.sh --out-dir $RUN_DIR_PHONE --thresh-c $COOL_THRESH_C --max-wait $COOL_MAX_WAIT -- true

# Start sensor sampling for this run
sh scripts/sample_sensors.sh --out $RUN_DIR_PHONE/sensors.csv --hz 10 &
SAMPLER=\$!

# Run eviction_bench with STANDARD sampling protocol (top-p applies to ALL policies)
LD_LIBRARY_PATH=bin bin/eviction_bench \
  --model $MODEL \
  --prompt $PROMPT_DIR/${PROMPT_ID}.txt \
  --prompt-id $PROMPT_ID \
  --policy $POLICY \
  --k-nominal $K \
  --max-tokens $MAX_TOKENS \
  --ctx-size $CTX_SIZE \
  --seed \$((REP * 100 + 42)) \
  --threads 4 \
  --n-gpu-layers $N_GPU_LAYERS \
  --n-sink 4 \
  --repeat-penalty 1.1 \
  --repeat-last-n 64 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 40 \
  --out-csv $RUN_DIR_PHONE/steps.csv \
  --out-meta $RUN_DIR_PHONE/meta.json \
  --out-gen  $RUN_DIR_PHONE/gen.txt 2>$RUN_DIR_PHONE/stderr.log
EXIT=\$?
kill \$SAMPLER 2>/dev/null
wait \$SAMPLER 2>/dev/null
echo \"  exit=\$EXIT\"
" 2>&1 | tee -a "$PROG_LOG"

                # Pull immediately for live aggregation
                LOCAL_DIR="$OUT_BASE_HOST/$POLICY/K${K}/$PROMPT_ID/rep${REP}"
                mkdir -p "$LOCAL_DIR"
                adb pull -p "$RUN_DIR_PHONE/" "$LOCAL_DIR/" 2>&1 | tail -1 | tee -a "$PROG_LOG"
                # Show key metrics
                if [ -f "$LOCAL_DIR/$(basename $RUN_DIR_PHONE)/meta.json" ]; then
                    META="$LOCAL_DIR/$(basename $RUN_DIR_PHONE)/meta.json"
                    grep -E '"(decode_tps|peak_kv_mb|peak_rss_kb|perplexity|mean_mass_retained|mean_eviction_efficiency|prefill_ms)"' "$META" 2>/dev/null | sed 's/^/    /' | tee -a "$PROG_LOG"
                fi
            done
        done
    done
done

echo "" | tee -a "$PROG_LOG"
echo "[$(date)] sweep3_done total_runs=$run_count" | tee -a "$PROG_LOG"
echo "Local logs at: $OUT_BASE_HOST" | tee -a "$PROG_LOG"
