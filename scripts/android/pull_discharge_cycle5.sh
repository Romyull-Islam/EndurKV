#!/bin/bash
# pull_discharge_cycle5.sh: collect the discharge5 cycle once the phone is back on the
# cable. Waits for the phone and prints progress. Once the loop has written DONE, pulls
# everything to /tmp/discharge5 and the archive. Safe to run while the loop is running.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-3C15B8003ZA00000}
export ADB_PORTS=${ADB_PORTS:-"5162 5161 5037"} ADB_CALL_TIMEOUT=${ADB_CALL_TIMEOUT:-600}
LOG(){ echo "[$(date '+%F %H:%M:%S')] $*"; }
A=/home/mislam22/EndurKV_workspace/EndurKV/scripts/android
ROOT=/data/local/tmp/endurkv; OUTD=$ROOT/discharge5
HOST=/tmp/discharge5; ARCH=/home/mislam22/EndurKV_workspace/tmp_archive/discharge5
found=""
until [ -n "$found" ]; do
  for p in $ADB_PORTS; do
    ss -ltnp 2>/dev/null | grep -q ":$p " || continue
    if ANDROID_ADB_SERVER_PORT=$p timeout 20 adb devices 2>/dev/null | grep -q "^$ANDROID_SERIAL[[:space:]]*device"; then found=$p; break; fi
  done
  [ -n "$found" ] || { LOG "waiting for $ANDROID_SERIAL"; sleep 300; }
done
export ANDROID_ADB_SERVER_PORT=$found
. $A/adb_resilient.sh
sh_su(){ adb_safe_shell "su -c '$1'" < /dev/null | tr -d '\r'; }
N=$(sh_su "ls -d $OUTD/dis_* 2>/dev/null | wc -l"); SOC=$(adb_safe_shell "dumpsys battery" < /dev/null | tr -d '\r' | awk '$1=="level:"{print $2}')
LOG "phone on port $found: $N requests so far, SoC $SOC%, loop pid: $(sh_su 'pgrep -f [p]hone_discharge_loop' | tr '\n' ' ')"
sh_su "tail -3 $OUTD/run.log" | sed 's/^/  /'
if ! sh_su "[ -f $OUTD/DONE ] && cat $OUTD/DONE" | grep -q .; then LOG "not finished; nothing pulled"; exit 0; fi
mkdir -p $HOST
adb pull $OUTD $HOST/ < /dev/null >/dev/null 2>&1
adb_safe_pull $ROOT/ukv_sched.log $HOST/ukv_sched.log >/dev/null 2>&1
adb_safe_pull $ROOT/tables/Llama-3.2-1B-Instruct-Q4_K_M.txt $HOST/ukv_sched_table.final.txt >/dev/null 2>&1
adb_safe_pull $ROOT/ukv_sched_table.seed.txt $HOST/ukv_sched_table.seed.txt >/dev/null 2>&1
mkdir -p $ARCH && rsync -a $HOST/ $ARCH/
LOG "pulled $(ls -d $HOST/discharge5/dis_* 2>/dev/null | wc -l) requests to $HOST (archived to $ARCH); DONE=$(cat $HOST/discharge5/DONE)"
echo "DISCHARGE5 PULLED"
