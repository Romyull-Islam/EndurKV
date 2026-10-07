#!/bin/bash
# run_discharge_cycle5.sh: start a real-battery discharge cycle, output in $ROOT/discharge5.
# Same phone-resident loop (phone_discharge_loop.sh), prompt (prompt_12k.txt, 9737 tokens,
# --ignore-eos), stop floor (STOP_SOC 12) and seed cost table as the earlier cycles.
# The phone charges to MIN_SOC, resets the tables to the seed and starts the loop itself.
# Then unplug: on the cable, USB carries 5 W of every 6 W request. The loop runs ~37 h.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-3C15B8003ZA00000}
export ADB_PORTS=${ADB_PORTS:-"5162 5161 5037"} ADB_CALL_TIMEOUT=${ADB_CALL_TIMEOUT:-600}
LOG(){ echo "[$(date '+%F %H:%M:%S')] $*"; }
A=/home/mislam22/EndurKV_workspace/EndurKV/scripts/android
ROOT=/data/local/tmp/endurkv
OUTD=$ROOT/discharge5
MIN_SOC=${MIN_SOC:-91}
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
adb push $A/ukv_sched_v2.3_backup.sh $ROOT/ukv_sched.sh < /dev/null >/dev/null 2>&1
adb push $A/phone_discharge_loop.sh $ROOT/phone_discharge_loop.sh < /dev/null >/dev/null 2>&1
adb push $A/cool_gate.sh $ROOT/scripts/cool_gate.sh < /dev/null >/dev/null 2>&1
adb push $A/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
adb push $A/ukv_sched_table.txt $ROOT/ukv_sched_table.seed.txt < /dev/null >/dev/null 2>&1
sh_su "chmod 755 $ROOT/ukv_sched.sh $ROOT/phone_discharge_loop.sh $ROOT/scripts/cool_gate.sh /data/local/tmp/sample_sensors.sh"
sh_su "ls -la $ROOT/ukv_sched_table.seed.txt $ROOT/phone_discharge_loop.sh $ROOT/corpora/prompt_12k.txt" | sed 's/^/  /'

# The laptop port does not charge this phone, so the wait for the start level runs on the
# phone: phone_discharge_start.sh charges on any charger, resets the tables and execs the loop.
adb push $A/phone_discharge_start5.sh $ROOT/phone_discharge_start.sh < /dev/null >/dev/null 2>&1
sh_su "chmod 755 $ROOT/phone_discharge_start.sh; rm -f $ROOT/discharge5.start.log"
sh_su "cp $ROOT/tables/$MKEY.txt $ROOT/tables/$MKEY.txt.bak_cycle2_$(date +%Y%m%d_%H%M%S) 2>/dev/null; true"
# launch the loop on the phone, detached
sh_su "cd $ROOT; START_SOC=$MIN_SOC STOP_SOC=$STOP_SOC nohup sh $ROOT/phone_discharge_start.sh </dev/null >$ROOT/discharge5.nohup 2>&1 &"
sleep 20
PID=$(sh_su "pgrep -f [p]hone_discharge_start; pgrep -f [p]hone_discharge_loop" | tr -d '\r' | xargs)
sh_su "cat $ROOT/discharge5.start.log 2>/dev/null" | sed 's/^/  start.log: /'
if [ -n "$PID" ]; then
  LOG "DISCHARGE CYCLE 5 STARTER RUNNING on the phone (pid $PID): it charges to ${MIN_SOC}%, then starts the loop to ${STOP_SOC}%"
  LOG ">>> Unplug from the laptop, charge on a wall charger; at ${MIN_SOC}% the loop starts by itself; then unplug and leave it ~37 h. <<<"
else
  LOG "loop did not start; see $ROOT/discharge5.nohup"; sh_su "cat $ROOT/discharge5.nohup" | sed 's/^/  /'; exit 1
fi
