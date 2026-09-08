#!/bin/bash
# ============================================================================
# run_discharge_timeline.sh -- the energy-aware scheduler on a REAL discharge (2026-09-04)
#
# No forced battery state. Charging is switched off and the phone is discharged by a session of
# back-to-back requests through the scheduler (ukv_sched.sh v2.1), which reads the real battery
# level each time, picks its plan for the tier it finds (healthy above 50%, mid 21 to 50%, low
# 20% or below), applies the output cap when the caller left the length open (it does here),
# meters the request, learns its table and lets the two loops move the lever. The run stops at
# 12% and charging is switched back on.
#
# Every request logs: time, SoC, tier, lever and bias, plan, GPU clocks, output cap, measured
# energy and time, predicted energy and time, temperatures at start. That is the timeline figure:
# the plan and the energy per request stepping down as the real battery crosses 50% and 20%.
#
# Protocol: CPU caps pinned as in the proofs; a LIGHT cool gate between requests (DDR <= 42 C,
# battery <= 36 C) so the drain keeps moving while requests still start from a similar thermal
# state; --ignore-eos so each request decodes exactly its cap (the long-answer case, where the
# data lever shows). The table is reset to its seed and the bias file removed at the start.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
HOST=${HOST:-/tmp/discharge}; mkdir -p $HOST
CSV=$HOST/timeline.csv
SCHED=/data/local/tmp/endurkv/ukv_sched.sh
ROOT=/data/local/tmp/endurkv
CPU_PRIME=1497600; CPU_REST=1785600
STOP_SOC=${STOP_SOC:-12}
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/ukv_sched.sh $SCHED < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/ukv_sched_table.txt $ROOT/ukv_sched_table.seed.txt < /dev/null >/dev/null 2>&1
adb_safe_shell "su -c 'cp $ROOT/ukv_sched_table.seed.txt $ROOT/ukv_sched_table.txt; rm -f $ROOT/ukv_lever_bias.txt; chmod 666 $ROOT/ukv_sched_table.txt'" < /dev/null
SAVED=$(adb_safe_shell "su -c 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo \$(basename \$p) \$(cat \$p/scaling_min_freq) \$(cat \$p/scaling_max_freq); done'" < /dev/null | tr -d '\r')
echo "$SAVED" > $HOST/pre_run_cpu_caps.txt
restore_all(){
  echo "$SAVED" | while read pol mn mx; do
    [ -n "${pol:-}" ] && adb_safe_shell "su -c 'echo $mx > /sys/devices/system/cpu/cpufreq/$pol/scaling_max_freq; echo $mn > /sys/devices/system/cpu/cpufreq/$pol/scaling_min_freq'" < /dev/null
  done
  adb_safe_shell "su -c 'echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel'" < /dev/null
}
pin_cpu(){ adb_safe_shell "su -c 'echo $CPU_REST > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; echo $CPU_REST > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; echo $CPU_PRIME > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq; echo $CPU_PRIME > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq'" < /dev/null; }
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; pkill -f eviction_bench; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; restore_all; echo "[$(date +%T)] charging restored, caps restored"; }
trap cleanup EXIT INT TERM

battery(){ adb_safe_shell "dumpsys battery" < /dev/null | tr -d '\r' | awk '/level:/{l=$2} /status:/{s=$2} /USB powered:/{u=$3} END{print l, s, u}'; }
temps(){ adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r' | awk '{printf "%d %d", $1/1000, $2/1000}'; }

[ -f $CSV ] || echo "ts,n,soc,status,tier,lever,bias_before,bias_after,plan,K,gpu_mhz,decode_mhz,nout_cap,np,ns,pred_J,pred_s,meas_J,meas_s,pre_J,dec_J,tps,ddr_start,batt_start,batt_end,ddr_end,loop" > $CSV
n=$(ls -d $HOST/dis_* 2>/dev/null | wc -l)
fails=0
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
echo "[$(date +%T)] discharge timeline: start SoC $(battery | awk '{print $1}')%, stop at ${STOP_SOC}%"
while :; do
  read SOC STAT USB <<< "$(battery)"
  [ -z "${SOC:-}" ] && { fails=$((fails+1)); echo "[$(date +%T)] battery read failed ($fails)"; [ $fails -ge 5 ] && break; sleep 60; continue; }
  [ "$SOC" -le "$STOP_SOC" ] && { echo "[$(date +%T)] SoC $SOC% <= $STOP_SOC%: stop"; break; }
  # light gate: keep requests starting from a similar thermal state without stalling the drain
  adb_safe_shell "su -c 'COOL_DDR_MAX=42 COOL_BAT_MAX=36 COOL_CPU_MAX=50 COOL_MAX_ITERS=120; . $ROOT/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1 | sed 's/^/    /'
  pin_cpu
  n=$((n+1)); TAG=$(printf "dis_%03d" $n); D=$HOST/$TAG; mkdir -p $D
  read DDR0 BAT0 <<< "$(temps)"
  echo "[$(date +%T)] $TAG: SoC ${SOC}% status $STAT usb $USB ddr ${DDR0}C batt ${BAT0}C"
  adb_safe_shell "su -c 'rm -rf $ROOT/sched/$TAG; sh $SCHED --prompt $P --tag $TAG --ignore-eos'" < /dev/null 2>&1 | grep -E "^\[sched\]|walk" | sed 's/^/    /'
  adb_safe_shell "su -c 'echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk'" < /dev/null >/dev/null 2>&1
  for f in meta.json err sensors.csv; do adb_safe_pull "$ROOT/sched/$TAG/$f" "$D/$f" >/dev/null 2>&1; done
  L=$(adb_safe_shell "grep \"tag=$TAG \" $ROOT/ukv_sched.log | tail -1" < /dev/null | tr -d '\r'); echo "$L" > $D/sched_log.txt
  read DDR1 BAT1 <<< "$(temps)"
  if [ -f "$D/meta.json" ] && [ -n "$L" ]; then
    fails=0
    g(){ echo "$L" | sed -n "s/.* $1=\([^ ]*\).*/\1/p" | head -1; }
    BA=$(g bias | sed 's/.*->//'); BB=$(g bias | sed 's/->.*//')
    LOOP=$(echo "$L" | sed 's/.*; //')
    echo "$(date +%F_%T),$n,$SOC,$STAT,$(g tier),$(g lever),$BB,$BA,$(g plan),$(g K),$(g gpu_mhz),$(g decode_mhz),$(g nout_cap),$(g np),$(g ns),$(g pred_J),$(g pred_s),$(g meas_J),$(g meas_s),$(g pre_J),$(g dec_J),$(g tps),$DDR0,$BAT0,$BAT1,$DDR1,\"$LOOP\"" >> $CSV
    echo "    -> $(g plan) cap $(g nout_cap): $(g meas_J) J / $(g meas_s) s (pred $(g pred_J) J / $(g pred_s) s); $LOOP"
  else
    fails=$((fails+1)); echo "    [$TAG] FAILED ($fails in a row)"; [ $fails -ge 3 ] && { echo "three failures in a row: stop"; break; }
  fi
done
adb_safe_pull $ROOT/ukv_sched_table.txt $HOST/ukv_sched_table.learned.txt >/dev/null 2>&1
adb_safe_pull $ROOT/ukv_sched.log $HOST/ukv_sched.log >/dev/null 2>&1
echo DISCHARGE_DONE
