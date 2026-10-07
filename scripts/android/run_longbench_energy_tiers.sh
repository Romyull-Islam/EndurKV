#!/bin/bash
# LongBench F1 across the energy-aware controller's three tiers (--k-pct 20, 10, 5).
# Llama-3.2-1B on the phone CPU, qasper + hotpotqa x 15 samples, same prompts and binary
# as /tmp/lb_native so its vanilla and muKV K=1024 cells stay the reference. No
# --ignore-eos. No cool gate, so use these cells for F1 only. Energy per tier comes from
# the cooled n=3 campaign in /tmp/ea_n3.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/endurkv/bin_cpu_cur
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
OUT=/data/local/tmp/endurkv/logs/lbtier_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/lb_tiers; mkdir -p $HOST
N=${N_SAMPLES:-15}
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace"
adb_safe_shell "mkdir -p $OUT" < /dev/null
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
cleanup(){ adb_safe_shell "su -c 'echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM
for task in qasper hotpotqa; do
  for i in $(seq 0 $((N-1))); do
    idx=$(printf "%03d" $i)
    src=/tmp/longbench_adaptive_3x3/phi3_vanilla_${task}/prompt_${idx}.txt
    [ -f "$src" ] && adb push "$src" "$OUT/${task}_${idx}.txt" < /dev/null >/dev/null 2>&1
  done
done
mg(){ case $1 in qasper) echo 128;; hotpotqa) echo 32;; *) echo 64;; esac; }
for task in qasper hotpotqa; do
  for i in $(seq 0 $((N-1))); do
    idx=$(printf "%03d" $i)
    for PCT in 20 10 5; do
      CELL="kpct${PCT}_${task}_${idx}"
      D=$HOST/$CELL; [ -f "$D/gen.txt" ] && { echo "  [$CELL] cached"; continue; }
      mkdir -p "$D"
      echo "[$(date +%H:%M:%S)] $CELL"
      adb_safe_shell "mkdir -p $OUT/$CELL; LD_LIBRARY_PATH=$BIN timeout 1200 $BIN/eviction_bench \
        --prompt $OUT/${task}_${idx}.txt --prompt-id $CELL --eval-mode gen --max-tokens $(mg $task) \
        --ctx-size 16384 --model $M --seed 42 --threads 4 --n-gpu-layers 0 \
        --n-batch 512 --ubatch-size 64 --greedy --cache-type-k f16 --cache-type-v f16 \
        $MU --k-pct $PCT \
        --out-meta $OUT/$CELL/meta.json --out-gen $OUT/$CELL/gen.txt --out-csv /dev/null \
        > /dev/null 2> $OUT/$CELL/err" < /dev/null
      adb_safe_pull "$OUT/$CELL/gen.txt"   "$D/gen.txt"   >/dev/null 2>&1
      adb_safe_pull "$OUT/$CELL/meta.json" "$D/meta.json" >/dev/null 2>&1
      [ -f "$D/gen.txt" ] && echo "    ok" || echo "    FAILED"
    done
  done
done
echo LB_TIERS_DONE
