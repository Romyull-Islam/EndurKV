#!/bin/bash
# LongBench (qasper, hotpotqa) on Llama-3.2-1B, CPU, every policy at its own published budget.
# Budgets as in run_64k_native_budgets.sh: muKV K=1024, SnapKV/Ada-KV/TOVA 2048,
# StreamingLLM 4+2000, H2O 20% of each prompt's token count.
# Pass 1 runs vanilla, whose meta.json gives n_prompt_tokens for the H2O budget.
# Accuracy only: no cool gate, so do not quote timing or energy from these cells.
# No --ignore-eos: generating past EOS makes predictions too long and lowers token F1.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/endurkv/bin_cpu_cur
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
OUT=/data/local/tmp/endurkv/logs/lbnat_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/lb_native; mkdir -p $HOST
N=${N_SAMPLES:-15}
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024"
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

run(){ # cell task idx flags...
  local CELL=$1 task=$2 idx=$3; shift 3
  local D=$HOST/$CELL; [ -f "$D/gen.txt" ] && return
  mkdir -p "$D"
  adb_safe_shell "mkdir -p $OUT/$CELL; LD_LIBRARY_PATH=$BIN timeout 1200 $BIN/eviction_bench \
    --prompt $OUT/${task}_${idx}.txt --prompt-id $CELL --eval-mode gen --max-tokens $(mg $task) \
    --ctx-size 16384 --model $M --seed 42 --threads 4 --n-gpu-layers 0 \
    --n-batch 512 --ubatch-size 64 --greedy --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $OUT/$CELL/meta.json --out-gen $OUT/$CELL/gen.txt --out-csv /dev/null \
    > /dev/null 2> $OUT/$CELL/err" < /dev/null
  adb_safe_pull "$OUT/$CELL/gen.txt"   "$D/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "$OUT/$CELL/meta.json" "$D/meta.json" >/dev/null 2>&1
}

echo "PASS 1: vanilla (also yields n_prompt_tokens for H2O's ratio)"
for task in qasper hotpotqa; do
  for i in $(seq 0 $((N-1))); do
    idx=$(printf "%03d" $i)
    echo "[$(date +%H:%M:%S)] vanilla_${task}_${idx}"
    run "vanilla_${task}_${idx}" $task $idx --policy vanilla
  done
done

echo "PASS 2: every other policy at its own budget"
for task in qasper hotpotqa; do
  for i in $(seq 0 $((N-1))); do
    idx=$(printf "%03d" $i)
    NP=$(python3 -c "
import json,re,os
p='$HOST/vanilla_${task}_${idx}/meta.json'
print(json.loads(re.sub(r':\s*-?nan\b',': NaN',open(p).read())).get('n_prompt_tokens',0) if os.path.exists(p) else 0)" 2>/dev/null)
    [ "${NP:-0}" -lt 100 ] && { echo "  skip ${task}_${idx}: no vanilla token count"; continue; }
    H2OK=$(( NP / 5 ))    # 20% of N, H2O's published ratio
    echo "[$(date +%H:%M:%S)] ${task}_${idx}  N=$NP  H2O K=$H2OK"
    run "mukv_${task}_${idx}"         $task $idx $MU
    run "snapkv_${task}_${idx}"       $task $idx --policy snapkv --obs-window 32 --snapkv-kernel 5 --n-sink 0 --k-nominal 2048
    run "adakv_${task}_${idx}"        $task $idx --policy adakv --obs-window 32 --n-sink 0 --k-nominal 2048
    run "tova_${task}_${idx}"         $task $idx --policy tova --k-nominal 2048
    run "streamingllm_${task}_${idx}" $task $idx --policy streamingllm --n-sink 4 --k-nominal 2004
    run "h2o_${task}_${idx}"          $task $idx --policy h2o --obs-window 64 --n-sink 0 --k-nominal $H2OK
  done
done
echo LB_NATIVE_DONE
