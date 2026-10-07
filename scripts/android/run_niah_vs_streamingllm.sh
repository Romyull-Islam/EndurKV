#!/bin/bash
# run_niah_vs_streamingllm.sh: needle retrieval, muKV vs StreamingLLM vs vanilla.
# StreamingLLM keeps 4 sinks plus the most recent 2000 tokens and drops the middle by position,
# while muKV selects by attention, so shallow needles should separate them.
# Arms, each at its own published budget: vanilla (full cache), mukv (frozen config, K=1024),
# sfown (StreamingLLM start_size 4 + recent_size 2000, the streaming-llm repo defaults).
# No cool gate and no pinning: only hit/miss and retained cells are used, so timing from
# this run is not comparable to the gated campaigns.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv_n3
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/niah
DEV=/data/local/tmp/endurkv/logs/niahsllm_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/niah_vs_sllm; mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024"
SF="--policy streamingllm --n-sink 4 --k-nominal 2004 --compact-inplace"
VA="--policy vanilla --k-nominal 1024"
adb_safe_shell "mkdir -p $DEV" < /dev/null
for f in $SRC/niah_L*_n0.txt; do adb push "$f" "$DEV/$(basename $f)" < /dev/null >/dev/null 2>&1; done
STIMS=$(cd $SRC && ls niah_L*_n0.txt)

cell(){ # arm stim ctx flags...
  local ARM=$1 STIM=$2 CTX=$3; shift 3
  local id="${ARM}__${STIM%.txt}"; local D=$HOST/$id
  [ -f "$D/meta.json" ] && return
  mkdir -p "$D"
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN timeout 600 ./eviction_bench \
    --prompt $DEV/$STIM --prompt-id $id --eval-mode gen --max-tokens 64 --ignore-eos \
    --ctx-size $CTX --model $M --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 --n-batch 512 --n-ubatch 64 $* \
    --out-meta $DEV/$id.json --out-gen $DEV/$id.gen --out-csv /dev/null \
    > /dev/null 2> $DEV/$id.err" < /dev/null
  adb_safe_pull "$DEV/$id.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$id.gen"  "$D/gen.txt"   >/dev/null 2>&1
}
i=0; n=$(echo "$STIMS" | wc -w)
for STIM in $STIMS; do
  i=$((i+1)); case "$STIM" in *_L8K_*) CTX=8192;; *) CTX=4096;; esac
  echo "[$(date +%H:%M:%S)] ($i/$n) $STIM ctx=$CTX"
  cell vanilla "$STIM" "$CTX" $VA
  cell mukv    "$STIM" "$CTX" $MU
  cell sfown   "$STIM" "$CTX" $SF
done
echo NIAH_VS_SLLM_DONE
