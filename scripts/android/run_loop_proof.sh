#!/bin/bash
# ============================================================================
# run_loop_proof.sh -- do the two loops and the lever actually work on the phone? (2026-09-04)
#
# The scheduler (ukv_sched.sh v2.1) picks a plan from its cost table for the battery tier,
# runs it, meters it, and lets two loops move the lever: the performance loop nudges it toward
# performance when the request ran over its time budget, the energy loop toward energy when it
# drew more than its predicted energy; when both budgets are met the bias decays back. None of
# this had ever fired in a measured run (the earlier proof lost every update to a key bug), so
# this campaign provokes each loop with a REAL disturbance and checks the response:
#
#   ctrl_mains     told it is on mains         -> full performance, the control
#   healthy        SoC 80                      -> 1200 MHz with decode at 902, 4096 tokens
#   mid_1          SoC 40                      -> 1200 MHz with decode at 726, cap 1024
#   mid_burn_1/2   SoC 40, four idle cores spinning during the request (another app) ->
#                  energy over budget -> energy loop, lever -0.1 each -> bias -0.2
#   mid_after_1/2  SoC 40, no disturbance      -> at L=0.3 the walk reaches gpu902; both
#                  budgets met -> the bias decays (-0.15, then -0.10) and the plan returns
#   low_1          SoC 15                      -> gpu 902, cap 512
#   low_cap_1/2/3  SoC 15, an external 726 MHz cap written 12 s into the request (the vendor
#                  thermal limiter) -> time over budget -> performance loop, +0.1 each -> +0.3
#   low_after_1/2  SoC 15, no disturbance      -> at L=0.3 the time budget rejects gpu902 and
#                  the scheduler takes 1200 with decode at 726, cap 1024; then the bias decays
#
# Same matched protocol as the other proofs: cool gate DDR <= 35 C, battery <= 33 C, charging
# off, CPU caps pinned, --ignore-eos so every request decodes exactly its cap. The table is reset
# to its seed and the bias file removed at the start.
# ============================================================================
set -u
export ADB_CALL_TIMEOUT=900   # the wrapper kills and RERUNS any adb shell call past this; a request runs 145 to 230 s
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
HOST=${HOST:-/tmp/loop_proof}; mkdir -p $HOST
SCHED=/data/local/tmp/endurkv/ukv_sched.sh
ROOT=/data/local/tmp/endurkv
CPU_PRIME=1497600; CPU_REST=1785600
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
burn_stop(){ adb_safe_shell "su -c 'pkill -f \"whi[l]e :; do :; done\"'" < /dev/null >/dev/null 2>&1; }
cleanup(){ burn_stop; adb_safe_shell "su -c 'pkill -f sample_sensors; pkill -f eviction_bench; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; restore_all; }
trap cleanup EXIT INT TERM

settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. $ROOT/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    local _rd; _rd=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $_rd|awk '{print int($1/1000)}'); local _sb=$(echo $_rd|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"; START_TEMPS="ddr_c=${_sd} batt_c=${_sb}"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1; }

cell(){ # tag  disturbance(none|cpuburn|gpucap726)  scheduler-args
  local TAG=$1; local DIST=$2; shift 2; local ARGS="$*"
  local D=$HOST/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  pin_cpu
  echo "$START_TEMPS" > "$D/start_temps.txt"
  echo "[$(date +%H:%M:%S)] running $TAG ($DIST): $ARGS"
  adb_safe_shell "su -c 'rm -rf $ROOT/sched/$TAG'" < /dev/null >/dev/null 2>&1
  launch(){ adb_safe_shell "su -c 'mkdir -p $ROOT/sched; nohup sh $SCHED --prompt $P --tag $TAG --ignore-eos $ARGS > $ROOT/sched/${TAG}.stderr 2>&1 &'" < /dev/null >/dev/null 2>&1; }
  waitdone(){ for i in $(seq 1 120); do sleep 10; adb_safe_shell "pgrep -f 'ukv_sched.s[h].*--tag $TAG ' >/dev/null && echo run || echo done" < /dev/null | grep -q done && return 0; done; echo "    [$TAG] still running after 20 min"; return 1; }
  case "$DIST" in
    cpuburn)
      # four spinning shells on cores 0 to 3 (the bench is pinned to 4 to 7): another app's load
      adb_safe_shell "su -c 'for i in 0 1 2 3; do nohup taskset \$((1<<i)) sh -c \"while :; do :; done\" >/dev/null 2>&1 & done'" < /dev/null >/dev/null 2>&1
      sleep 3; launch; waitdone; burn_stop;;
    gpucap726)
      # the request starts under the scheduler's 902 cap; 12 s in, an external limiter drops the GPU to 726
      launch; sleep 12
      adb_safe_shell "su -c 'echo 726000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; cat /sys/class/kgsl/kgsl-3d0/max_gpuclk'" < /dev/null | sed 's/^/    external cap now: /'
      waitdone;;
    *) launch; waitdone;;
  esac
  sleep 3
  adb_safe_shell "cat $ROOT/sched/${TAG}.stderr" < /dev/null 2>&1 | grep -E "^\[sched\]|walk" | sed 's/^/    /'
  adb_safe_shell "su -c 'echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk'" < /dev/null >/dev/null 2>&1
  for f in meta.json err gen.txt sensors.csv; do adb_safe_pull "$ROOT/sched/$TAG/$f" "$D/$f" >/dev/null 2>&1; done
  adb_safe_shell "grep \"tag=$TAG \" $ROOT/ukv_sched.log | tail -1" < /dev/null > "$D/sched_log.txt"
  adb_safe_shell "cat $ROOT/ukv_lever_bias.txt 2>/dev/null" < /dev/null > "$D/bias_after.txt"
  echo "    bias file after: $(tr '\n' ' ' < $D/bias_after.txt)"
  [ -f "$D/meta.json" ] && python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/clock_cell_report.py "$D" "$TAG" 2>/dev/null || echo "  [$TAG] FAILED"
}

cell ctrl_mains   none      --force-status charging --max-tokens 4096
cell healthy      none      --force-soc 80 --force-status discharging
cell mid_1        none      --force-soc 40 --force-status discharging
cell mid_burn_1   cpuburn   --force-soc 40 --force-status discharging
cell mid_burn_2   cpuburn   --force-soc 40 --force-status discharging
cell mid_after_1  none      --force-soc 40 --force-status discharging
cell mid_after_2  none      --force-soc 40 --force-status discharging
cell low_1        none      --force-soc 15 --force-status discharging
cell low_cap_1    gpucap726 --force-soc 15 --force-status discharging
cell low_cap_2    gpucap726 --force-soc 15 --force-status discharging
cell low_cap_3    gpucap726 --force-soc 15 --force-status discharging
cell low_after_1  none      --force-soc 15 --force-status discharging
cell low_after_2  none      --force-soc 15 --force-status discharging
adb_safe_pull $ROOT/ukv_sched_table.txt $HOST/ukv_sched_table.learned.txt >/dev/null 2>&1
adb_safe_pull $ROOT/ukv_sched.log $HOST/ukv_sched.log >/dev/null 2>&1
echo LOOPPROOF_DONE
