#!/bin/bash
# ============================================================================
# run_discharge_cycle2.sh -- second real-battery discharge cycle, a repeat of
# the paper's discharge run (2026-09-05/06, /tmp/discharge_final). (2026-09-24)
#
# WHY. The paper's discharge result is one cycle (123 requests, 91 to 12%).
# A second cycle under the same protocol turns it into a repeated measurement.
#
# WHAT. Same phone-resident loop (phone_discharge_loop.sh, detached, no host
# in the loop), same prompt (prompt_12k.txt, 9737 tokens, --ignore-eos), same
# floor (STOP_SOC 12), same seed: the shipped cost table
# ukv_sched_table.txt. Its only change since cycle 1 is q for the CPU K<1024
# rows (0.63/0.58 -> 0.74/0.74), which stays under every tier's floor, so the
# walk behaves the same. The scheduler is the current ukv_sched.sh (v2.7); for
# the default model its per-model table under $ROOT/tables is the shipped
# table, so that copy and its bias file are removed too, otherwise the loop
# would start from cycle 1's learned table and not from the cable seed.
#
# SEQUENCE. Wait for the phone; refuse if a bench is running; push scripts;
# charge to >= 90% with charging ON; reset tables; launch the loop detached
# under su with OUTD=$ROOT/discharge2; confirm it started; then print the
# instruction to UNPLUG THE CABLE (on the cable the USB port carries 5 W of
# every 6 W request, so the pack only discharges once the cable is out).
# The loop runs on the phone alone for ~37 h; pull_discharge_cycle2.sh
# collects the results when the phone is back on the cable.
# ============================================================================
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-3C15B8003ZA00000}
export ADB_PORTS=${ADB_PORTS:-"5162 5161 5037"} ADB_CALL_TIMEOUT=${ADB_CALL_TIMEOUT:-600}
LOG(){ echo "[$(date '+%F %H:%M:%S')] $*"; }
A=/home/mislam22/EndurKV_workspace/EndurKV/scripts/android
ROOT=/data/local/tmp/endurkv
OUTD=$ROOT/discharge2
MIN_SOC=${MIN_SOC:-90}
STOP_SOC=${STOP_SOC:-12}
MKEY=Llama-3.2-1B-Instruct-Q4_K_M

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
  [ -n "$found" ] || { LOG "waiting for $ANDROID_SERIAL"; sleep 120; }
done
export ANDROID_ADB_SERVER_PORT=$found
LOG "phone on adb port $found"
. $A/adb_resilient.sh
sh_su(){ adb_safe_shell "su -c '$1'" < /dev/null | tr -d '\r'; }

# refuse to start over another campaign or a still-running loop
BUSY=$(sh_su "pgrep -f [e]viction_bench; pgrep -f [p]hone_discharge_loop; pgrep -f [p]hone_discharge_start" | tr -d '\r' | xargs)
[ -n "$BUSY" ] && { LOG "phone busy (pids $BUSY); not starting"; exit 1; }
if [ "$(sh_su "[ -d $OUTD ] && echo yes" | xargs)" = yes ]; then
  LOG "$OUTD exists from an earlier attempt; moving it aside"
  sh_su "mv $OUTD $OUTD.bak_$(date +%Y%m%d_%H%M%S)"
fi

# scripts and the seed table
adb push $A/ukv_sched.sh $ROOT/ukv_sched.sh < /dev/null >/dev/null 2>&1
adb push $A/phone_discharge_loop.sh $ROOT/phone_discharge_loop.sh < /dev/null >/dev/null 2>&1
adb push $A/cool_gate.sh $ROOT/scripts/cool_gate.sh < /dev/null >/dev/null 2>&1
adb push $A/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
adb push $A/ukv_sched_table.txt $ROOT/ukv_sched_table.seed.txt < /dev/null >/dev/null 2>&1
sh_su "chmod 755 $ROOT/ukv_sched.sh $ROOT/phone_discharge_loop.sh $ROOT/scripts/cool_gate.sh /data/local/tmp/sample_sensors.sh"
sh_su "ls -la $ROOT/ukv_sched_table.seed.txt $ROOT/phone_discharge_loop.sh $ROOT/corpora/prompt_12k.txt" | sed 's/^/  /'

# the laptop port does not charge this phone (2026-09-25: switch on, USB_CDP 1.5 A, battery current
# 0, "Not charging"), so the wait for the start level lives ON THE PHONE: phone_discharge_start.sh
# charges on any charger, resets the tables to the seed at the moment it starts, and execs the loop.
adb push $A/phone_discharge_start.sh $ROOT/phone_discharge_start.sh < /dev/null >/dev/null 2>&1
sh_su "chmod 755 $ROOT/phone_discharge_start.sh; rm -f $ROOT/discharge2.start.log"
sh_su "cp $ROOT/tables/$MKEY.txt $ROOT/tables/$MKEY.txt.bak_cycle2_$(date +%Y%m%d_%H%M%S) 2>/dev/null; true"
# launch the loop on the phone, detached
sh_su "cd $ROOT; START_SOC=$MIN_SOC STOP_SOC=$STOP_SOC nohup sh $ROOT/phone_discharge_start.sh </dev/null >$ROOT/discharge2.nohup 2>&1 &"
sleep 20
PID=$(sh_su "pgrep -f [p]hone_discharge_start; pgrep -f [p]hone_discharge_loop" | tr -d '\r' | xargs)
sh_su "cat $ROOT/discharge2.start.log 2>/dev/null" | sed 's/^/  start.log: /'
if [ -n "$PID" ]; then
  LOG "DISCHARGE CYCLE 2 STARTER RUNNING on the phone (pid $PID): it charges to ${MIN_SOC}%, then starts the loop to ${STOP_SOC}%"
  LOG ">>> Unplug from the laptop, charge on a wall charger; at ${MIN_SOC}% the loop starts by itself; then unplug and leave it ~37 h. <<<"
else
  LOG "loop did not start; see $ROOT/discharge2.nohup"; sh_su "cat $ROOT/discharge2.nohup" | sed 's/^/  /'; exit 1
fi
