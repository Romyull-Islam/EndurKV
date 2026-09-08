#!/bin/bash
# ============================================================================
# run_kd_lb_headline.sh -- KeyDiff LongBench at its HEADLINE budgets. (2026-08-23)
#
# SCOPE, and why this is not the full four-budget grid. KeyDiff reports LongBench
# at 2K/4K/6K/8K (their Sec. 4) but pins its claims to two of them: "<=1.5%
# accuracy drop with a 6K cache budget and <=.04% with 8K" (abstract). We already
# have 2048 on both tasks, so this run adds ONLY 6144 and 8192, and ONLY hotpotqa.
#
# WHY hotpotqa ONLY. A budget is meaningless where it does not bind. Measured on
# our own stimuli: qasper (median prompt 4439 tok) is compressed by 6144 on just
# 2/15 prompts and by 8192 on 1/15 -- running those would mostly measure the full
# cache wearing a KeyDiff label. hotpotqa (median 15476) is compressed by every
# budget on 14-15/15, so it is the only task on which their headline claims can be
# tested at all. 30 cells instead of 90.
#
# NO COOL GATE, deliberately. These are QUALITY cells (token-F1); no timing or
# energy from them is ever quoted, so they do not need the thermal protocol that
# every timed cell uses. Joins /tmp/lb_native, no --ignore-eos, matching exactly
# how the other LongBench cells were produced.
#
# The bench runs DETACHED on the device with its own timeout, so a tunnel drop
# does not kill the cell in flight; the host only polls for the sentinel. Cells
# already scored are skipped, so a restart costs at most the one in progress.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
CB=/data/local/tmp/endurkv/bin_cpu_kd
DEV=/data/local/tmp/endurkv/logs/kdlb_$(date +%Y%m%d_%H%M%S)
adb_safe_shell "mkdir -p $DEV" < /dev/null

for B in 8192 6144; do
  for i in $(seq 0 14); do
    idx=$(printf "%03d" $i)
    src=/tmp/longbench_adaptive_3x3/phi3_vanilla_hotpotqa/prompt_${idx}.txt
    [ -f "$src" ] || continue
    CELL="keydiff${B}_hotpotqa_${idx}"; D=/tmp/lb_native/$CELL
    [ -s "$D/gen.txt" ] && { LOG "$CELL cached"; continue; }
    mkdir -p "$D"
    timeout 180 adb push "$src" "$DEV/hp_${idx}.txt" < /dev/null >/dev/null 2>&1
    LOG "$CELL"
    adb_safe_shell "su -c 'mkdir -p $DEV/$CELL; rm -f $DEV/$CELL/.done; setsid nohup sh -c \"timeout 1800 env LD_LIBRARY_PATH=$CB $CB/eviction_bench --prompt $DEV/hp_${idx}.txt --prompt-id $CELL --eval-mode gen --max-tokens 32 --ctx-size 16384 --model $M --seed 42 --threads 4 --n-gpu-layers 0 --n-batch 512 --ubatch-size 64 --greedy --cache-type-k f16 --cache-type-v f16 --policy keydiff --n-sink 0 --k-nominal $B --compact-inplace --keydiff-decode-block 128 --out-meta $DEV/$CELL/meta.json --out-gen $DEV/$CELL/gen.txt --out-csv /dev/null > /dev/null 2> $DEV/$CELL/err ; echo DONE > $DEV/$CELL/.done\" >/dev/null 2>&1 &'" < /dev/null
    w=0
    while [ $w -lt 1800 ]; do
      adb_safe_shell "[ -f $DEV/$CELL/.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
      sleep 20; w=$((w+20))
    done
    adb_safe_pull "$DEV/$CELL/gen.txt"   "$D/gen.txt"   >/dev/null 2>&1
    adb_safe_pull "$DEV/$CELL/meta.json" "$D/meta.json" >/dev/null 2>&1
  done
done
LOG "KD_LB_HEADLINE_DONE"
touch /tmp/kd_lb_headline_DONE
