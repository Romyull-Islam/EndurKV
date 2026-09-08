#!/bin/bash
# ============================================================================
# run_longbench_energy_tiers.sh -- does the energy-aware controller have a real
# quality trade? LongBench F1 across its three tiers. (2026-08-16)
#
# WHY THIS EXISTS. The --energy-aware controller demonstrably closes its loop (n=3,
# rotated order: K moves 1947 -> 974 -> 487 from battery state alone, backend-aware).
# What it lacked was a REASON: on WikiText PPL and single-needle NIAH, k-pct 10 was
# simultaneously the cheapest tier AND tied the retrieval ceiling, so both outer
# tiers were dominated and the ladder collapsed to a constant. "Energy-aware" was a
# mechanism without a benefit.
#
# LONGBENCH REOPENED THE QUESTION. At n=50 on the RTX, quality DOES move with
# retention on some models (muKV at ~13% loses 2.8-3.7 F1 to vanilla on Gemma-2 and
# Bonsai; SnapKV holds quality by retaining 83-94%). Multi-fact QA is the workload
# where a bigger budget can buy something. If F1 at k-pct 20 > 10 > 5 with the
# measured energy 364.5 / 340.1 / 345.2 mJ/token, the controller finally has a real
# frontier: full QA quality above 50% charge, bounded F1 loss below 20% -- and
# energy-awareness becomes a defensible claim instead of a collapsed ladder. If F1
# is flat across tiers, the honest conclusion stays "pin k-pct 10" and the
# controller is reported as a negative result. Either way this is the deciding cell.
#
# DESIGN. Llama-3.2-1B phone CPU, qasper + hotpotqa x 15 samples, the SAME prompts
# and binary as /tmp/lb_native so vanilla and muKV@K=1024 cells there remain the
# reference points. Three arms = the controller's tiers, driven as --k-pct so K
# resolves per prompt exactly as the controller sets it. NO --ignore-eos (that
# defect is fixed; see run_longbench_native_budgets.sh). No cool gate: F1 only, no
# timing or energy may be quoted from these cells -- the energy column comes from
# the cooled n=3 campaign (/tmp/ea_n3), which is the point: F1 from here, joules
# from there, joined by tier.
# ============================================================================
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
