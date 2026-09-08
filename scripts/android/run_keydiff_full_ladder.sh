#!/bin/bash
# ============================================================================
# run_keydiff_full_ladder.sh -- KeyDiff at ALL FOUR of its published budgets.
# (2026-08-17)
#
# WHY. The chain already running measures KeyDiff at 2048 -- their SMALLEST
# published budget. Their HEADLINE claims live at the top of the ladder:
# "<=1.5% accuracy drop with a 6K cache budget and <=.04% with 8K"
# (KeyDiff abstract; Park et al., NeurIPS 2025). Reporting only 2K would show
# KeyDiff at its weakest published setting, which is not a fair comparison and
# is exactly the defect we corrected for StreamingLLM. Their published set is
# {2K, 4K, 6K, 8K} (Sec. 4, "We denote the cache budgets of 2048, 4096, 6144
# and 8192 as 2K, 4K, 6K and 8K"), so all four run.
#
# WHICH CELLS ARE REAL. A budget only evicts when N < prompt length; the paper
# handles this in its own compression-ratio table ("We replace the summand with
# 1 whenever N >= L_i, as compression doesn't occur in that setting", App F.3).
# Measured on OUR stimuli (2026-08-17):
#     NIAH L4K (median 3122):  2048 evicts;  4096/6144/8192 are NO-OPS
#     NIAH L8K (median 6099):  2048,4096 evict;  6144/8192 are NO-OPS
#     qasper   (median 4439):  2048 15/15; 4096 9/15; 6144 2/15; 8192 1/15
#     hotpotqa (median 15476): all four budgets evict on 14-15/15
# So hotpotqa is the ONLY benchmark here where the headline budgets bind, and
# it runs first. Every cell records n_prompt_tokens; the scorer marks no-op
# cells (budget >= prompt) explicitly rather than reporting them as KeyDiff
# results -- a no-op cell is the full cache wearing a KeyDiff label.
#
# ORDER. 8192 -> 6144 -> 4096, headline budgets first, so an interruption still
# leaves the numbers their abstract claims. 2048 is NOT re-run (the chain has it).
#
# WHAT THIS BUYS THE PAPER. KeyDiff's own four budgets trace a quality-vs-
# retention curve across its entire published operating range; muKV is a single
# point to be placed against that whole curve, not against one budget we chose.
# That is the strongest form of the comparison and the hardest to attack.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
WS=/home/mislam22/EndurKV_workspace
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }

# ── wait for the running chain; never share the phone ────────────────────────
# 2026-08-22: bounded wait removed. The master table it waited for is complete
# (/tmp/def_cpu_kd_DONE, Aug 21 23:54); the old 15 h cap made this script abort
# mid-outage and it was never restarted.
LOG "master table done -- starting LongBench KeyDiff budget ladder"

M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
DEV=/data/local/tmp/endurkv/logs/kdladder_$(date +%Y%m%d_%H%M%S)
adb_safe_shell "mkdir -p $DEV" < /dev/null

# ── LongBench, phone CPU, joins /tmp/lb_native (no --ignore-eos) ─────────────
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

# ── NIAH: only L8K@4096 is a new binding cell (see header) ───────────────────
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
