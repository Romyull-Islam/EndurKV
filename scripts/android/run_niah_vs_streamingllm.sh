#!/bin/bash
# ============================================================================
# run_niah_vs_streamingllm.sh -- the one axis where muKV and StreamingLLM are not
# equivalent by physics. (2026-08-15)
#
# WHY THIS IS THE DECIDING EXPERIMENT. The head-to-head at ctx 16384 (pinned, n=3,
# /tmp/sllm_faithful) came back a tie on everything it measured:
#     decode  muKV 29.05 +- 0.13   StreamingLLM 29.07 +- 0.18   tok/s   (0.07% apart)
#     energy  muKV 1439 +- 35      StreamingLLM 1445 +- 33      J       (0.4% apart)
#     PPL     muKV 22.980          StreamingLLM 23.218                  (1.0% apart)
# and a roofline says that is not fixable by tuning: decode is bandwidth-bound, so
# speedup = (W + KV_full)/(W + KV_kept), and on Llama-3.2-1B at 16K the weights are
# 800 MB against a 319 MB full cache (KV/W = 0.40). Once both policies are down to
# 1-2k cells the remaining difference is ~42 MB out of ~865 MB per step -- a 5% ceiling
# on any advantage muKV could have, and we measured 0%.
#
# PERPLEXITY CANNOT SEPARATE THEM EITHER, and that is structural rather than bad luck.
# StreamingLLM keeps a 2000-token RECENT window, and continuation perplexity is dominated
# by recent context -- it is configured almost optimally for exactly what the metric
# rewards. It will keep tying at any budget.
#
# WHAT IS DIFFERENT: WHERE THE KEPT BYTES ARE. StreamingLLM keeps 4 sinks plus the last
# 2000 tokens and discards everything between -- at a 6099-token stimulus that is 67% of
# the document, chosen by POSITION and without looking at it. muKV chooses by attention,
# so its 723 cells can sit anywhere. A needle at shallow depth falls in the gap that
# StreamingLLM deletes by construction and cannot fall in a gap muKV selected against.
# Predicted, from the window arithmetic alone: at the 8K stimuli StreamingLLM should MISS
# depths below roughly 67% (positions before 6099-2000 = 4099) and HIT above it; at the
# 4K stimuli (3120 tokens) the 2000-token window covers depth 36% upward, so it should
# hit nearly everything. If that pattern appears, the mechanism is confirmed by its shape
# and not merely by a score.
#
# ARMS -- each at its OWN published budget, never a matched one:
#   vanilla   full cache, the retrieval ceiling
#   mukv      frozen config + in-place compaction, K=1024 (its deployment budget)
#   sfown     StreamingLLM, FA-on + compacted, start_size 4 + recent_size 2000 = K 2004
#             (mit-han-lab/streaming-llm, examples/run_streaming_llama.py argparse defaults)
#
# NO COOL GATE, DELIBERATELY: a needle is either in the keep-set or it is not, and clock
# state cannot change that. Gating 42 cells would cost ~10 hours and buy nothing. Timing
# from this campaign is therefore NOT comparable to the gated campaigns and must not be
# quoted -- only the hit/miss column and the retained-cell column are valid here.
#
# NOT PINNED, for the same reason: pinning exists to stabilise throughput, which this run
# does not measure.
# ============================================================================
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
