#!/bin/bash
# run_keydiff_full_ladder.sh: KeyDiff (Park et al., NeurIPS 2025) at its published
# budgets 8192, 6144 and 4096, largest first. 2048 comes from an earlier run.
# A budget only evicts when it is below the prompt length, so hotpotqa is the only
# task where all budgets bind. Each cell records n_prompt_tokens so the scorer can
# mark no-op cells (budget >= prompt).
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
WS=/home/mislam22/EndurKV_workspace
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }

LOG "master table done -- starting LongBench KeyDiff budget ladder"

M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
DEV=/data/local/tmp/endurkv/logs/kdladder_$(date +%Y%m%d_%H%M%S)
adb_safe_shell "mkdir -p $DEV" < /dev/null

# LongBench on the phone CPU, results join /tmp/lb_native (no --ignore-eos).
# hotpotqa first: the only task whose prompts exceed the headline budgets.
for B in 8192 6144 4096; do
  for task in hotpotqa qasper; do
    case $task in qasper) MG=128;; hotpotqa) MG=32;; esac
    for i in $(seq 0 14); do
      idx=$(printf "%03d" $i)
      src=/tmp/longbench_adaptive_3x3/phi3_vanilla_${task}/prompt_${idx}.txt
      [ -f "$src" ] || continue
      CELL="keydiff${B}_${task}_${idx}"; D=/tmp/lb_native/$CELL
      [ -f "$D/gen.txt" ] && { LOG "$CELL cached"; continue; }
      mkdir -p "$D"
      adb push "$src" "$DEV/${task}_${idx}.txt" < /dev/null >/dev/null 2>&1
      LOG "LB $CELL"
      adb_safe_shell "mkdir -p $DEV/$CELL; LD_LIBRARY_PATH=/data/local/tmp/endurkv/bin_cpu_kd timeout 1800 \
        /data/local/tmp/endurkv/bin_cpu_kd/eviction_bench \
        --prompt $DEV/${task}_${idx}.txt --prompt-id $CELL --eval-mode gen --max-tokens $MG \
        --ctx-size 16384 --model $M --seed 42 --threads 4 --n-gpu-layers 0 \
        --n-batch 512 --ubatch-size 64 --greedy --cache-type-k f16 --cache-type-v f16 \
        --policy keydiff --n-sink 0 --k-nominal $B --compact-inplace \
        --out-meta $DEV/$CELL/meta.json --out-gen $DEV/$CELL/gen.txt --out-csv /dev/null \
        > /dev/null 2> $DEV/$CELL/err" < /dev/null
      adb_safe_pull "$DEV/$CELL/gen.txt"   "$D/gen.txt"   >/dev/null 2>&1
      adb_safe_pull "$DEV/$CELL/meta.json" "$D/meta.json" >/dev/null 2>&1
    done
  done
done

# NIAH: L8K at 4096 is the only new cell where the budget binds.
SRC=$WS/EndurKV/benchmarks/niah
for f in $SRC/niah_L8K_*_n0.txt; do adb push "$f" "$DEV/$(basename $f)" < /dev/null >/dev/null 2>&1; done
for STIM in $(cd $SRC && ls niah_L8K_*_n0.txt); do
  id="keydiff4096__${STIM%.txt}"; D=/tmp/niah_vs_sllm/$id
  [ -f "$D/meta.json" ] && continue
  mkdir -p "$D"
  LOG "NIAH $id"
  adb_safe_shell "cd /data/local/tmp/ukv_kd && LD_LIBRARY_PATH=/data/local/tmp/ukv_kd timeout 900 ./eviction_bench \
    --prompt $DEV/$STIM --prompt-id $id --eval-mode gen --max-tokens 64 --ignore-eos \
    --ctx-size 8192 --model $M --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 \
    --policy keydiff --n-sink 0 --k-nominal 4096 --compact-inplace --n-batch 512 --n-ubatch 64 \
    --out-meta $DEV/$id.json --out-gen $DEV/$id.gen --out-csv /dev/null \
    > /dev/null 2> $DEV/$id.err" < /dev/null
  adb_safe_pull "$DEV/$id.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$id.gen"  "$D/gen.txt"   >/dev/null 2>&1
done
LOG "KD_LADDER_DONE"
