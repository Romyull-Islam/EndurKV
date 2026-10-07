#!/bin/bash
# run_lb_keydiff.sh: KeyDiff on the phone LongBench set, using its own build (bin_cpu_kd).
# The prompt set is the (task, idx) pairs of the mukv_* cells in /tmp/lb_native, with the
# same settings as run_longbench_wide.sh and KeyDiff at its published budget of 2048.
# Existing cells are skipped, so the script resumes after a cut. No cooling gate (quality
# only), but a heat guard waits while DDR is above 52 C. Charging is off until exit.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-3C15B8003ZA00000}
export ADB_PORTS=${ADB_PORTS:-"5162 5161 5037"} ADB_CALL_TIMEOUT=${ADB_CALL_TIMEOUT:-1500}
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }

# Wait for the phone on one of the adb servers. 5161 is an ssh reverse tunnel, and a plain
# "adb devices" on a free port starts a local server there that blocks the tunnel. So only use
# ports that already have a listener, and kill a stray local adb server on a tunnel port.
port_ok(){
  local l; l=$(ss -ltnp 2>/dev/null | grep ":$1 ") || return 1
  if echo "$l" | grep -q '"adb"' && [ "$1" != 5037 ]; then
    LOG "stray local adb server on tunnel port $1; killing it"; adb -P $1 kill-server 2>/dev/null; return 1
  fi
  return 0
}
found=""
until [ -n "$found" ]; do
  for p in $ADB_PORTS; do
    port_ok $p || continue
    if ANDROID_ADB_SERVER_PORT=$p timeout 20 adb devices 2>/dev/null | grep -q "^$ANDROID_SERIAL[[:space:]]*device"; then found=$p; break; fi
  done
  [ -n "$found" ] || { LOG "waiting for $ANDROID_SERIAL"; sleep 60; }
done
export ANDROID_ADB_SERVER_PORT=$found
LOG "phone on adb port $found"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

BIN=/data/local/tmp/endurkv/bin_cpu_kd
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/longbench
OUT=/data/local/tmp/endurkv/logs/lbkd_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/lb_native
TASKS="hotpotqa qasper 2wikimqa triviaqa"
KD="--policy keydiff --n-sink 0 --k-nominal 2048 --compact-inplace"
mg(){ case $1 in qasper) echo 128;; hotpotqa|2wikimqa) echo 32;; triviaqa) echo 32;; *) echo 64;; esac; }

adb_safe_shell "mkdir -p $OUT" < /dev/null
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
cleanup(){ adb_safe_shell "su -c 'echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM

heat_guard(){
  local hot=0
  while [ $hot -lt 40 ]; do
    T=$(adb_safe_shell "cat /sys/class/thermal/thermal_zone47/temp" < /dev/null 2>/dev/null | tr -dc '0-9')
    [ -n "$T" ] || return
    [ "$((T/1000))" -le 52 ] && return
    LOG "  DDR $((T/1000))C > 52C, waiting"; sleep 60; hot=$((hot+1))
  done
}

run(){ # cell task idx
  local CELL=$1 task=$2 idx=$3
  local D=$HOST/$CELL; [ -s "$D/gen.txt" ] && { LOG "  $CELL cached"; return; }
  mkdir -p "$D"
  heat_guard
  adb push "$SRC/$task/prompt_${idx}.txt" "$OUT/${task}_${idx}.txt" < /dev/null >/dev/null 2>&1
  adb_safe_shell "mkdir -p $OUT/$CELL; LD_LIBRARY_PATH=$BIN timeout 1200 $BIN/eviction_bench \
    --prompt $OUT/${task}_${idx}.txt --prompt-id $CELL --eval-mode gen --max-tokens $(mg $task) \
    --ctx-size 16384 --model $M --seed 42 --threads 4 --n-gpu-layers 0 \
    --n-batch 512 --ubatch-size 64 --greedy --cache-type-k f16 --cache-type-v f16 $KD \
    --out-meta $OUT/$CELL/meta.json --out-gen $OUT/$CELL/gen.txt --out-csv /dev/null \
    > /dev/null 2> $OUT/$CELL/err" < /dev/null
  adb_safe_pull "$OUT/$CELL/gen.txt"   "$D/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "$OUT/$CELL/meta.json" "$D/meta.json" >/dev/null 2>&1
  [ -s "$D/gen.txt" ] && LOG "  $CELL ok  $(head -c 40 "$D/gen.txt" | tr '\n' ' ')" || LOG "  $CELL NO OUTPUT"
}

for task in $TASKS; do
  IDX=$(ls $HOST | grep "^mukv_${task}_" | sed "s/^mukv_${task}_//" | sort)
  LOG "=== $task: $(echo $IDX | wc -w) prompts, the set the other policies ran ==="
  for idx in $IDX; do
    run "keydiff_${task}_${idx}" $task $idx
  done
done
LOG "LB KEYDIFF DONE -> $HOST"
