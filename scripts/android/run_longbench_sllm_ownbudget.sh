#!/bin/bash
# ============================================================================
# run_longbench_sllm_ownbudget.sh -- LongBench with StreamingLLM at ITS OWN budget.
# (2026-08-15)
#
# THE DEFECT BEING CORRECTED. /tmp/phone_longbench_full ran every policy at
# `--k-nominal 1024`, StreamingLLM included. Its own documented budget is
# start_size 4 + recent_size 2000 = K 2004 (mit-han-lab/streaming-llm,
# examples/run_streaming_llama.py argparse defaults), so it was evaluated at roughly HALF
# the cache its authors specify. Forcing our K onto a baseline deletes the policy being
# compared -- the same defect that made the phone-GPU speed table understate it by 4.5x.
#
# WHAT IS *NOT* WRONG WITH THE OLD CELLS, and this matters for how much has to be redone.
# The old LongBench ran on CPU (bin_cpu/eviction_bench_v10), where the FA-off path is
# numerically clean -- the Adreno graph-split corruption is a Vulkan defect and does not
# apply. And F1 depends only on WHICH cells are kept: compaction rearranges the survivors
# but does not change the keep-set, so it cannot move an F1 score. The FA-on/compaction
# corrections that mattered so much for throughput are irrelevant here. The budget is the
# whole defect, which is why only StreamingLLM is re-run and every other policy's existing
# cells stay valid and directly comparable.
#
# IDENTICAL PROTOCOL, ONE VARIABLE CHANGED: same binary (v10, so the comparison stays
# within one build), same prompts, same CPU config, same ctx 16384, same greedy decoding,
# same max_gen per task (qasper 128, hotpotqa 32). Only --k-nominal moves, 1024 -> 2004.
#
# NO COOL GATE, and here that is sound rather than a shortcut: F1 is decided by the
# keep-set under greedy decoding, both deterministic. Temperature changes how fast tokens
# are produced, not which tokens. NO TIMING OR ENERGY MAY BE QUOTED FROM THESE CELLS --
# the existing cells were cooled and these are not, so only the F1 column is comparable.
#
# PRE-EXISTING CAVEAT CARRIED FORWARD, not introduced here: gemma-2-2b has
# n_ctx_train = 8192 and the campaign runs ctx 16384. That is wrong for gemma in both the
# old cells and these, and it is kept identical so the comparison is like-for-like. It
# should be fixed for BOTH arms before either is published.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/endurkv/bin_cpu/eviction_bench_v10
OUT=/data/local/tmp/endurkv/logs/lbsllm_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/lb_sllm_own; mkdir -p $HOST
N_SAMPLES=${N_SAMPLES:-5}
adb_safe_shell "mkdir -p $OUT" < /dev/null
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM

for task in qasper hotpotqa; do
  for i in $(seq 0 $((N_SAMPLES - 1))); do
    idx=$(printf "%03d" $i)
    src=/tmp/longbench_adaptive_3x3/phi3_vanilla_${task}/prompt_${idx}.txt
    [ -f "$src" ] && adb push "$src" "$OUT/${task}_${idx}.txt" < /dev/null >/dev/null 2>&1
  done
done
echo "prompts pushed"

get_max_gen(){ case $1 in qasper) echo 128;; hotpotqa) echo 32;; *) echo 64;; esac; }

for task in qasper hotpotqa; do
  MG=$(get_max_gen $task)
  for i in $(seq 0 $((N_SAMPLES - 1))); do
    idx=$(printf "%03d" $i)
    for MNAME in phi3 llama1b gemma2b; do
      case $MNAME in
        phi3)    M=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf ;;
        llama1b) M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf ;;
        gemma2b) M=/data/local/tmp/endurkv/models/gemma-2-2b-it-Q4_K_M.gguf ;;
      esac
      CELL="${MNAME}_sllm2004_${task}_${idx}"
      D=$HOST/$CELL; [ -f "$D/gen.txt" ] && { echo "  [$CELL] cached"; continue; }
      mkdir -p "$D"
      echo "[$(date +%H:%M:%S)] $CELL"
      adb_safe_shell "mkdir -p $OUT/$CELL; LD_LIBRARY_PATH=/data/local/tmp/endurkv/bin_cpu timeout 900 $BIN \
        --prompt $OUT/${task}_${idx}.txt --prompt-id $CELL --eval-mode gen --max-tokens $MG \
        --ignore-eos --ctx-size 16384 --model $M --seed 42 --threads 4 --n-gpu-layers 0 \
        --n-batch 512 --ubatch-size 64 --greedy \
        --policy streamingllm --k-nominal 2004 --n-sink 4 --cache-type-k f16 --cache-type-v f16 \
        --out-meta $OUT/$CELL/meta.json --out-gen $OUT/$CELL/gen.txt --out-csv /dev/null \
        > /dev/null 2> $OUT/$CELL/err" < /dev/null
      adb_safe_pull "$OUT/$CELL/gen.txt"   "$D/gen.txt"   >/dev/null 2>&1
      adb_safe_pull "$OUT/$CELL/meta.json" "$D/meta.json" >/dev/null 2>&1
      [ -f "$D/gen.txt" ] && echo "    ok" || echo "    FAILED"
    done
  done
done
echo LB_SLLM_OWN_DONE
