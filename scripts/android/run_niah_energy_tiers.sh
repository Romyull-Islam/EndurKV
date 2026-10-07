#!/bin/bash
# run_niah_energy_tiers.sh: needle retrieval at each energy-aware controller tier.
# Disjoint-slice perplexity cannot separate the tiers, but a needle either survives
# eviction or not. Arms: vanilla and levels 0/1/2, reached through --energy-aware with
# moved SoC thresholds so each level uses the budget the controller itself picks.
# No cool gate: retrieval does not depend on clock state, so timing and energy from
# this campaign are not comparable to the cooled campaigns.
# Needs /data/local/tmp/ukv_n3 (the older ukv build logs the budget but does not apply it).
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv_n3
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/niah
DEV=/data/local/tmp/endurkv/logs/niahtiers_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/niah_tiers; mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace"
adb_safe_shell "mkdir -p $DEV" < /dev/null
for f in $SRC/niah_L*_n0.txt; do adb push "$f" "$DEV/$(basename $f)" < /dev/null >/dev/null 2>&1; done
STIMS=$(cd $SRC && ls niah_L*_n0.txt)

cell(){ # arm  stim  ctx  extra_args...
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
  i=$((i+1))
  case "$STIM" in *_L8K_*) CTX=8192;; *) CTX=4096;; esac
  echo "[$(date +%H:%M:%S)] ($i/$n) $STIM ctx=$CTX"
  cell vanilla "$STIM" "$CTX" --policy vanilla
  cell level0  "$STIM" "$CTX" $MU --energy-aware --ea-soc-hi 50 --ea-soc-lo 20
  cell level1  "$STIM" "$CTX" $MU --energy-aware --ea-soc-hi 99 --ea-soc-lo 20
  cell level2  "$STIM" "$CTX" $MU --energy-aware --ea-soc-hi 99 --ea-soc-lo 99
done
echo NIAH_TIERS_DONE
