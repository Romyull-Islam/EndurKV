#!/bin/bash
# ============================================================================
# run_niah_energy_tiers.sh -- the experiment that decides whether the energy-aware
# controller is a contribution or a mis-tuned constant. (2026-08-14)
#
# THE PROBLEM THIS EXISTS TO SETTLE. The n=3 energy campaign (/tmp/ea_n3) measured what
# each controller tier costs in joules:
#     level 0  k-pct 20  K=1947  1379 cells   364.5 +- 6.9 mJ/tok
#     level 1  k-pct 10  K= 974   688 cells   340.1 +- 7.3 mJ/tok   (-6.7%, separated)
#     level 2  k-pct  5  K= 487   342 cells   345.2 +- 4.6 mJ/tok   (-5.3%, NOT separable from L1)
# Level 1 is simultaneously the cheapest and the fastest arm. As it stands the table
# argues against its own design: a reviewer asks "why does level 0 exist -- why not always
# run level 1?" and there is no answer, because no measured quantity improves with a
# larger budget. An adaptive controller whose most conservative setting is dominated by
# its middle setting is not energy-aware, it is mis-tuned.
#
# WHY PERPLEXITY CANNOT ANSWER IT. Every tier already beats the FULL cache on WikiText
# (23.25 / 23.00 / 22.73 vs vanilla 23.44). That is not a fluke: the eval slice is
# verified disjoint from the prompt, so the prompt is mostly irrelevant context and
# dropping it reduces distraction. A disjoint-slice perplexity is the right test for "did
# eviction damage language modelling" (it did not) and the WRONG test for "did eviction
# throw away information the request needed". It cannot separate these tiers and never
# will, however many times it is run.
#
# WHAT CAN. Long-range retrieval is the axis where cache size actually bites, because the
# needle sits at a known depth and either survives eviction or does not. If retrieval
# degrades as the budget shrinks, the controller has a real trade to make -- keep recall
# while the battery allows, spend it when the battery is low -- and the energy table gains
# the column it is missing. If retrieval is flat at k-pct 5, the honest conclusion is that
# the budget should simply be pinned at the energy optimum and the tiers deleted. BOTH
# outcomes are publishable; the current ambiguity is not.
#
# DESIGN. 14 needle stimuli (4K and 8K contexts x 7 depths) x 4 arms:
#   vanilla   full cache, the retrieval ceiling
#   level 0 / 1 / 2   driven through --energy-aware with the SAME threshold trick the
#                     energy campaign used, so these are the budgets the controller
#                     actually selects rather than budgets we hand it. The controller
#                     still reads the real SoC; only the tier boundaries move.
#
# NO COOL GATE, DELIBERATELY. Retrieval correctness is thermally invariant -- a needle is
# either in the keep-set or it is not, and clock state cannot change that. Cooling 56
# cells would cost ~14 hours and buy nothing. Timing/energy from this campaign is
# therefore NOT comparable to the cooled campaigns and must not be quoted.
#
# Binary: /data/local/tmp/ukv_n3, the build verified to actually APPLY the energy-aware
# budget (the older /data/local/tmp/ukv logs the decision and ignores it -- it retained
# 723 cells when asked for 342).
# ============================================================================
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
