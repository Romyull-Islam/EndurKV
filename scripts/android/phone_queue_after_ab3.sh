#!/bin/bash
# phone_queue_after_ab3.sh — runs AFTER A/B #3 finishes on phone.
# Sequentially executes (single-CPU phone, can't parallelize):
#   1. Vanilla llama-perplexity on WikiText-2 (standard PPL reference)
#   2. A/B #4 at K=2048 (25% retention, fair vs published baselines)
# All previous K=1024 data in /data/local/tmp/endurkv/logs/ab4_* is preserved.

set -e
export PATH=/home/mislam22/tools/platform-tools:$PATH
unset ANDROID_ADB_SERVER_PORT

LOG=/home/mislam22/EndurKV_workspace/EndurKV/figures/phone_queue.log
echo "[queue] starting at $(date)" | tee -a "$LOG"

# Wait until no eviction_bench process is left on the phone
echo "[queue] waiting for A/B #3 to finish ..." | tee -a "$LOG"
while adb shell "ps -A | grep -q eviction_bench" 2>/dev/null; do
    sleep 60
done
echo "[queue] A/B #3 finished at $(date)" | tee -a "$LOG"

MODEL_BASENAME="Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PROMPT_BASENAME="narrativeqa_pub_001.txt"
PROMPT_ID="narrativeqa_pub_001"

# =====================================================================
# STAGE A: llama-perplexity on WikiText-2 (vanilla llama.cpp tool)
# This is the standard PPL benchmark — same formula every published paper uses.
# Single run with vanilla model (no policy applied — llama-perplexity has no
# policy hooks; this gives us the "ground truth model PPL" reference).
# =====================================================================
echo "" | tee -a "$LOG"
echo "##### STAGE A: WikiText-2 PPL via llama-perplexity #####" | tee -a "$LOG"
echo "[$(date)]" | tee -a "$LOG"

STAGE_A_DIR="/data/local/tmp/endurkv/logs/wt2_ppl"
adb shell "mkdir -p $STAGE_A_DIR"

adb shell "
cd /data/local/tmp/endurkv
sh scripts/sample_sensors.sh --out $STAGE_A_DIR/sensors.csv --hz 10 &
SAMPLER=\$!
LD_LIBRARY_PATH=bin bin/llama-perplexity \
  -m models/$MODEL_BASENAME \
  -f corpora/wiki.test.raw \
  -c 4096 \
  -t 4 \
  --no-warmup \
  > $STAGE_A_DIR/ppl_output.txt 2> $STAGE_A_DIR/stderr.log
EXIT=\$?
kill \$SAMPLER 2>/dev/null
wait \$SAMPLER 2>/dev/null
echo \"[wt2_ppl] exit=\$EXIT\"
" | tee -a "$LOG"

echo "[$(date)] STAGE A done. Final PPL line:" | tee -a "$LOG"
adb shell "grep -E 'Final estimate|estimate.*PPL' $STAGE_A_DIR/ppl_output.txt 2>/dev/null" | tee -a "$LOG"

# =====================================================================
# STAGE B: A/B #4 at K=2048 (25% retention, fair vs published baselines)
# Same model + prompt as A/B #3, only K differs. v1 → vanilla → tova → pyramid.
# =====================================================================
echo "" | tee -a "$LOG"
echo "##### STAGE B: A/B #4 at K=2048 (25% retention) #####" | tee -a "$LOG"

adb shell "rm -rf /data/local/tmp/endurkv/logs/ab5_*"

for POLICY in v1 vanilla tova pyramid; do
    echo "" | tee -a "$LOG"
    echo "##### policy=$POLICY  $(date +%H:%M:%S) #####" | tee -a "$LOG"
    adb shell "
mkdir -p /data/local/tmp/endurkv/logs/ab5_${POLICY}
cd /data/local/tmp/endurkv
sh scripts/sample_sensors.sh --out logs/ab5_${POLICY}/sensors.csv --hz 10 &
SAMPLER=\$!
LD_LIBRARY_PATH=bin bin/eviction_bench \
  --model models/$MODEL_BASENAME \
  --prompt prompts/longbench/$PROMPT_BASENAME \
  --prompt-id $PROMPT_ID \
  --policy $POLICY \
  --k-nominal 2048 \
  --max-tokens 64 \
  --ctx-size 12288 \
  --threads 4 \
  --n-sink 4 \
  --repeat-penalty 1.1 \
  --repeat-last-n 64 \
  --out-csv logs/ab5_${POLICY}/steps.csv \
  --out-meta logs/ab5_${POLICY}/meta.json \
  --out-gen  logs/ab5_${POLICY}/gen.txt 2>logs/ab5_${POLICY}/stderr.log
EXIT=\$?
kill \$SAMPLER 2>/dev/null
wait \$SAMPLER 2>/dev/null
echo \"  policy=$POLICY exit=\$EXIT\"
" | tee -a "$LOG"
    echo "--- meta ---" | tee -a "$LOG"
    adb shell "cat /data/local/tmp/endurkv/logs/ab5_${POLICY}/meta.json 2>/dev/null" \
        | grep -E "(policy|prefill|decode_tps|peak_kv_mb|peak_rss|perplexity|evicted|mass_retained|retention_ratio|efficiency)" \
        | tee -a "$LOG"
    echo "--- gen (first 300 chars) ---" | tee -a "$LOG"
    adb shell "head -c 300 /data/local/tmp/endurkv/logs/ab5_${POLICY}/gen.txt 2>/dev/null" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "[queue] ALL DONE at $(date)" | tee -a "$LOG"
