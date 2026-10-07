#!/bin/bash
# run_longbench_wide.sh: LongBench on the phone CPU, 4 tasks (hotpotqa, qasper, 2wikimqa,
# triviaqa) at n=25 per task, so results can be pooled across tasks.
# multifieldqa_en is excluded because gold.json has no answers for it.
# Prompts come from benchmarks/longbench/<task>/prompt_NNN.txt, indices align with gold.json.
# Writes into /tmp/lb_native and skips cells that already exist there.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }

exec 9>/tmp/.endurkv_queue.lock; flock 9
BIN=/data/local/tmp/endurkv/bin_cpu_cur
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
OUT=/data/local/tmp/endurkv/logs/lbwide_$(date +%Y%m%d_%H%M%S)
SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/longbench
HOST=/tmp/lb_native; mkdir -p $HOST
N=${N_SAMPLES:-25}
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024"

adb_safe_shell "mkdir -p $OUT" < /dev/null
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
cleanup(){ adb_safe_shell "su -c 'echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM

TASKS="hotpotqa qasper 2wikimqa triviaqa"

# Take the first N prompts under MAXBYTES. Many prompts exceed the 16384-token context.
# At the worst measured 3.70 bytes per token, 58,000 bytes is at most 15,676 tokens,
# which leaves room for the 128-token generation.
MAXBYTES=${MAXBYTES:-58000}
pick_prompts(){   # task -> prints up to N indices whose prompt fits
  local task=$1 c=0
  for f in $(ls "$SRC/$task"/prompt_*.txt 2>/dev/null | sort); do
    [ "$(stat -c%s "$f")" -le "$MAXBYTES" ] || continue
    basename "$f" .txt | sed 's/^prompt_//'
    c=$((c+1)); [ "$c" -ge "$N" ] && break
  done
}
declare -A PICKED
for task in $TASKS; do
  PICKED[$task]="$(pick_prompts $task | tr '\n' ' ')"
  LOG "  $task: selected $(echo ${PICKED[$task]} | wc -w) prompts under ${MAXBYTES}B"
done

LOG "staging prompts for: $TASKS  (n=$N, size-filtered)"
for task in $TASKS; do
  for idx in ${PICKED[$task]}; do
    adb push "$SRC/$task/prompt_${idx}.txt" "$OUT/${task}_${idx}.txt" < /dev/null >/dev/null 2>&1
  done
done

# LongBench's own max-generation lengths
mg(){ case $1 in qasper) echo 128;; hotpotqa|2wikimqa) echo 32;; triviaqa) echo 32;; *) echo 64;; esac; }

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

LOG "=== PASS 1: vanilla (also yields n_prompt_tokens for H2O's ratio) ==="
for task in $TASKS; do
  for idx in ${PICKED[$task]}; do
    [ -f "$HOST/vanilla_${task}_${idx}/gen.txt" ] && continue
    LOG "  vanilla_${task}_${idx}"
    run "vanilla_${task}_${idx}" $task $idx --policy vanilla
  done
done

LOG "=== PASS 2: every other policy at its own published budget ==="
for task in $TASKS; do
  for idx in ${PICKED[$task]}; do
    NP=$(python3 -c "
import json,re,os
p='$HOST/vanilla_${task}_${idx}/meta.json'
print(json.loads(re.sub(r':\s*-?nan\b',': NaN',open(p).read())).get('n_prompt_tokens',0) if os.path.exists(p) else 0)" 2>/dev/null)
    [ "${NP:-0}" -lt 100 ] && { LOG "  skip ${task}_${idx}: no vanilla token count"; continue; }
    # Exact guard from the token count vanilla measured.
    if [ "$NP" -gt $((16384 - $(mg $task) - 64)) ]; then
      LOG "  skip ${task}_${idx}: $NP tokens + $(mg $task) gen does not fit 16384"; continue
    fi
    H2OK=$(( NP / 5 ))    # 20% of N, H2O's published ratio
    LOG "  ${task}_${idx}  N=$NP  H2O K=$H2OK"
    run "mukv_${task}_${idx}"         $task $idx $MU
    run "snapkv_${task}_${idx}"       $task $idx --policy snapkv --obs-window 32 --snapkv-kernel 5 --n-sink 0 --k-nominal 2048
    run "adakv_${task}_${idx}"        $task $idx --policy adakv --obs-window 32 --n-sink 0 --k-nominal 2048
    run "tova_${task}_${idx}"         $task $idx --policy tova --k-nominal 2048
    run "streamingllm_${task}_${idx}" $task $idx --policy streamingllm --n-sink 4 --k-nominal 2004
    run "h2o_${task}_${idx}"          $task $idx --policy h2o --obs-window 64 --n-sink 0 --k-nominal $H2OK
  done
done
LOG "LB_WIDE_DONE"
touch /tmp/lb_wide_DONE
