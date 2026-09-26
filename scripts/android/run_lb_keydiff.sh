#!/bin/bash
# ============================================================================
# run_lb_keydiff.sh -- KeyDiff on the phone LongBench set, so Table 3's empty
# LongBench cells for KeyDiff can be filled. (2026-09-24)
#
# WHY. The LongBench-on-phone campaign (run_longbench_wide.sh, /tmp/lb_native)
# ran every policy on one build, and KeyDiff needs its own (bin_cpu_kd), which
# was ported after that campaign. So KeyDiff has no F1 and no cache share in
# Table 3. Thirty KeyDiff cells already exist (hotpotqa and qasper, 15 each, at
# k_nominal 2048, from run_kd_lb_headline.sh); the rest of the set is missing.
#
# WHAT. The SAME prompts the other policies ran: the (task, idx) pairs of the
# existing mukv_* cells in /tmp/lb_native define the set, so nothing depends on
# re-running the size filter. Same model, ctx, generation lengths, seed and
# threads as run_longbench_wide.sh; KeyDiff at its published budget of 2048
# with the flags Table 2's CPU row used (run_keydiff_full_ladder.sh). Cells
# that already have gen.txt are skipped, so the script resumes after a cut.
# Quality only: no cooling gate (F1 and live cells do not depend on clock), but
# a heat guard waits while DDR is above 52 C so an unattended run cannot cook
# the phone. Charging is off while it runs and restored at exit.
# ============================================================================
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-3C15B8003ZA00000}
export ADB_PORTS=${ADB_PORTS:-"5162 5161 5037"} ADB_CALL_TIMEOUT=${ADB_CALL_TIMEOUT:-1500}
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }

# wait for the phone on either adb server before sourcing the helper.
# 5161 is an ssh reverse tunnel to the laptop's adb server. If nothing listens there, a plain
# "adb devices" STARTS a local adb server on that port, which then blocks the tunnel from ever
# binding again (2026-09-24). So: only talk to a port something already listens on, and if the
# listener is a stray local adb server on the tunnel port, kill it and free the port.
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
