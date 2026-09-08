#!/bin/bash
# phone_v1fa_validate.sh — quick quality sanity check for v1_fa.
#
# Runs v1 and v1_fa with IDENTICAL seed/K/prompt on Llama-3.2-1B; outputs both
# generations side-by-side. Use to verify v1_fa is producing coherent text and
# is in the same quality ballpark as v1.
#
# Each run is short (--max-tokens 32) so the whole thing takes ~10 min.

set -e
export PATH=/home/mislam22/tools/platform-tools:$PATH
TS=$(date +%s)
PHONE_OUT=/data/local/tmp/endurkv/logs/v1fa_validate_${TS}
HOST_OUT=/home/mislam22/EndurKV_workspace/phone-logs/v1fa_validate_${TS}
mkdir -p $HOST_OUT

adb shell "mkdir -p $PHONE_OUT"

run_one() {
    local policy=$1
    DIR=$PHONE_OUT/$policy
    adb shell "mkdir -p $DIR"
    echo "[$(date)] === policy=$policy ==="
    adb shell "
cd /data/local/tmp/endurkv
LD_LIBRARY_PATH=bin_cpu ./bin_cpu/eviction_bench \
    --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --prompt prompts_chat/Llama-3.2-1B/hotpotqa_pub_001.txt \
    --prompt-id hotpotqa_pub_001 \
    --policy $policy --k-nominal 512 \
    --max-tokens 32 --ctx-size 12288 --seed 42 \
    --threads 4 --n-gpu-layers 0 --n-batch 512 --ubatch-size 64 \
    --n-sink 4 --greedy \
    --out-csv $DIR/steps.csv \
    --out-meta $DIR/meta.json \
    --out-gen $DIR/gen.txt 2>$DIR/stderr.log
echo \$policy_exit=\$?
"
}

run_one v1
run_one v1_fa

adb pull -q $PHONE_OUT $HOST_OUT/ 2>&1 | tail -1
echo ""
echo "=== v1 generation ==="
cat $HOST_OUT/$(basename $PHONE_OUT)/v1/gen.txt 2>/dev/null
echo ""
echo ""
echo "=== v1_fa generation ==="
cat $HOST_OUT/$(basename $PHONE_OUT)/v1_fa/gen.txt 2>/dev/null
echo ""
echo ""
echo "=== Quality side-by-side ==="
printf "%-12s %-15s %-15s %-15s %-15s\n" "policy" "decode_tps" "perplexity" "mean_nll" "evicted_prefill"
for p in v1 v1_fa; do
    META=$HOST_OUT/$(basename $PHONE_OUT)/$p/meta.json
    if [ -f $META ]; then
        TPS=$(grep -oE '"decode_tps":[^,]*' $META | cut -d: -f2)
        PPL=$(grep -oE '"perplexity":[^,]*' $META | cut -d: -f2)
        NLL=$(grep -oE '"mean_nll":[^,]*' $META | cut -d: -f2)
        EVP=$(grep -oE '"evicted_prefill":[^,]*' $META | cut -d: -f2)
        printf "%-12s %-15s %-15s %-15s %-15s\n" "$p" "$TPS" "$PPL" "$NLL" "$EVP"
    fi
done
echo ""
echo "saved: $HOST_OUT"
