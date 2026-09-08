#!/bin/bash
# ============================================================================
# run_sched_proof.sh -- does the adaptive scheduler save energy when the battery is low,
# and give full performance when it is not? (2026-09-03)
#
# The scheduler (ukv_sched.sh, on the phone) reads the battery and the request, picks a
# plan from its measured cost table, runs it, measures the request's energy and updates
# the table. Here every arm sends the SAME prompt through the scheduler; only the battery
# state it is told differs (forced, since the phone sits at 100%). One arm is the control:
# the scheduler told it is on mains, which is the full-performance plan and equals "no
# adaptation" (GPU 1200 MHz, K=1024, 4096 tokens).
#     control        mains          -> gpu 1200, K 1024, 4096 tokens
#     sched_healthy  SoC 80         -> expected gpu 1200, 4096 tokens (full performance)
#     sched_mid      SoC 40         -> expected gpu 902, cap 1024 tokens
#     sched_low      SoC 15         -> expected gpu 902 or 726, cap 512 tokens
#     sched_low_nocap SoC 15, caller fixes 4096 tokens -> the clock saving alone, equal output
# --ignore-eos in every arm so the decoded length is exactly the cap and energy per token
# and per request are both comparable. n=2 per arm, interleaved.
#
# Matched conditions as in run_energy_aware_proof_v2.sh: CPU caps pinned (1497.6 / 1785.6
# MHz, min = max), cool gate DDR <= 35 C and battery <= 33 C, charging off during cells,
# scheduler pins the bench to the big cores at high priority and writes the GPU clock.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
HOST=${HOST:-/tmp/sched_proof}; mkdir -p $HOST
SCHED=/data/local/tmp/endurkv/ukv_sched.sh
CPU_PRIME=1497600; CPU_REST=1785600
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/ukv_sched.sh /data/local/tmp/endurkv/ukv_sched.sh < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
adb_safe_shell "su -c 'cp /data/local/tmp/endurkv/ukv_sched_table.txt /data/local/tmp/endurkv/ukv_sched_table.seed.txt'" < /dev/null
SAVED=$(adb_safe_shell "su -c 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo \$(basename \$p) \$(cat \$p/scaling_min_freq) \$(cat \$p/scaling_max_freq); done'" < /dev/null | tr -d '\r')
echo "$SAVED" > $HOST/pre_run_cpu_caps.txt
restore_all(){
  echo "$SAVED" | while read pol mn mx; do
    [ -n "${pol:-}" ] && adb_safe_shell "su -c 'echo $mx > /sys/devices/system/cpu/cpufreq/$pol/scaling_max_freq; echo $mn > /sys/devices/system/cpu/cpufreq/$pol/scaling_min_freq'" < /dev/null
  done
  adb_safe_shell "su -c 'echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel'" < /dev/null
}
pin_cpu(){ adb_safe_shell "su -c 'echo $CPU_REST > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; echo $CPU_REST > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; echo $CPU_PRIME > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq; echo $CPU_PRIME > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq'" < /dev/null; }
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; pkill -f eviction_bench; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; restore_all; }
trap cleanup EXIT INT TERM

settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    local _rd; _rd=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $_rd|awk '{print int($1/1000)}'); local _sb=$(echo $_rd|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"; START_TEMPS="ddr_c=${_sd} batt_c=${_sb}"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1; }

cell(){ # tag  scheduler-args
  local TAG=$1; shift; local ARGS="$*"
  local D=$HOST/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  pin_cpu
  echo "$START_TEMPS" > "$D/start_temps.txt"
  echo "[$(date +%H:%M:%S)] running $TAG: $ARGS"
  adb_safe_shell "su -c 'rm -rf /data/local/tmp/endurkv/sched/$TAG; sh $SCHED --prompt $P --tag $TAG --ignore-eos $ARGS'" < /dev/null 2>&1 | grep -E "^\[sched\]|chosen" | sed 's/^/    /'
  for f in meta.json err gen.txt sensors.csv; do adb_safe_pull "/data/local/tmp/endurkv/sched/$TAG/$f" "$D/$f" >/dev/null 2>&1; done
  adb_safe_shell "grep \"tag=$TAG \" /data/local/tmp/endurkv/ukv_sched.log | tail -1" < /dev/null > "$D/sched_log.txt"
  [ -f "$D/meta.json" ] && python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/clock_cell_report.py "$D" "$TAG" 2>/dev/null || echo "  [$TAG] FAILED"
}

for R in 1 2; do
  echo "=== replicate $R ==="
  cell control_r$R          --force-status charging --max-tokens 4096
  cell sched_healthy_r$R    --force-soc 80 --force-status discharging
  cell sched_mid_r$R        --force-soc 40 --force-status discharging
  cell sched_low_r$R        --force-soc 15 --force-status discharging
  cell sched_low_nocap_r$R  --force-soc 15 --force-status discharging --max-tokens 4096
done
adb_safe_pull /data/local/tmp/endurkv/ukv_sched_table.txt $HOST/ukv_sched_table.learned.txt >/dev/null 2>&1
adb_safe_pull /data/local/tmp/endurkv/ukv_sched_table.seed.txt $HOST/ukv_sched_table.seed.txt >/dev/null 2>&1
adb_safe_pull /data/local/tmp/endurkv/ukv_sched.log $HOST/ukv_sched.log >/dev/null 2>&1
echo "=== learned table vs seed ==="; diff $HOST/ukv_sched_table.seed.txt $HOST/ukv_sched_table.learned.txt | grep "^[<>]" | sed 's/^/  /'
echo SCHEDPROOF_DONE
