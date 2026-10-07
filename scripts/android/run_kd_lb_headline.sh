#!/bin/bash
# KeyDiff on LongBench hotpotqa at budgets 6144 and 8192, the two its abstract
# makes claims about. Only hotpotqa because shorter qasper prompts rarely exceed
# these budgets. Quality cells (token-F1), so no cool gate. Output joins
# /tmp/lb_native. The bench runs detached on the phone with its own timeout and
# the host polls for a sentinel, so a tunnel drop does not kill a cell.
# Cells with a gen.txt are skipped.
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
