#!/bin/bash
# Phone LongBench for StreamingLLM as in its paper (4 sinks, rolling window of 2000, positions
# inside the cache, --sllm-window). Reruns exactly the 110 StreamingLLM cells behind Table 3:
# same prompt files on the phone, same answer caps and CPU settings as run_longbench_wide.sh and
# run_longbench_native_budgets.sh. Each cell must report the same prompt length as the old one.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-3C15B8003ZA00000}
export ANDROID_ADB_SERVER_PORT=${ANDROID_ADB_SERVER_PORT:-5162} ADB_CALL_TIMEOUT=${ADB_CALL_TIMEOUT:-1500}
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
BIN=/data/local/tmp/endurkv/bin_cpu_sllm
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
L=/data/local/tmp/endurkv/logs
OLD=/home/mislam22/EndurKV_workspace/tmp_archive/lb_native
HOST=/home/mislam22/EndurKV_workspace/tmp_archive/lb_sllmw; mkdir -p $HOST
OUT=$L/lbsllmw_$(date +%Y%m%d_%H%M%S)
# Old cells were pulled on 08-15/16 (native campaign) or 09-01/02 (the last wide campaign).
declare -A SRCDIR=([2026-08-15]=lbnat_20260815_145443 [2026-08-16]=lbnat_20260815_145443 \
                   [2026-09-01]=lbwide_20260901_015157 [2026-09-02]=lbwide_20260901_015157)
mg(){ case $1 in qasper) echo 128;; hotpotqa|2wikimqa|triviaqa) echo 32;; *) echo 64;; esac; }
adb_safe_shell "mkdir -p $OUT" < /dev/null
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
cleanup(){ adb_safe_shell "su -c 'echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM
for od in $(ls -d $OLD/streamingllm_* | sort); do
  CELL=$(basename $od); key=${CELL#streamingllm_}; task=${key%_*}; idx=${key##*_}
  D=$HOST/$CELL; [ -f "$D/gen.txt" ] && continue
  day=$(date -r $od/meta.json +%F); sd=${SRCDIR[$day]:-}
  [ -z "$sd" ] && { LOG "skip $CELL: no prompt source for $day"; continue; }
  OLDN=$(grep -oE '"n_prompt_tokens": *[0-9]+' $od/meta.json | grep -oE '[0-9]+$')
  mkdir -p "$D"; LOG "$CELL  ($sd, old n_prompt=$OLDN)"
  adb_safe_shell "mkdir -p $OUT/$CELL; LD_LIBRARY_PATH=$BIN timeout 1200 $BIN/eviction_bench \
    --prompt $L/$sd/${task}_${idx}.txt --prompt-id $CELL --eval-mode gen --max-tokens $(mg $task) \
    --ctx-size 16384 --model $M --seed 42 --threads 4 --n-gpu-layers 0 \
    --n-batch 512 --ubatch-size 64 --greedy --cache-type-k f16 --cache-type-v f16 \
    --policy streamingllm --n-sink 4 --k-nominal 2004 --sllm-window \
    --out-meta $OUT/$CELL/meta.json --out-gen $OUT/$CELL/gen.txt --out-csv /dev/null \
    > /dev/null 2> $OUT/$CELL/err" < /dev/null
  adb_safe_pull "$OUT/$CELL/gen.txt"   "$D/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "$OUT/$CELL/meta.json" "$D/meta.json" >/dev/null 2>&1
  NEWN=$(grep -oE '"n_prompt_tokens": *[0-9]+' $D/meta.json 2>/dev/null | grep -oE '[0-9]+$')
  [ "$NEWN" = "$OLDN" ] || LOG "  WARNING $CELL: n_prompt $NEWN differs from old $OLDN"
done
LOG LB_SLLMW_DONE
