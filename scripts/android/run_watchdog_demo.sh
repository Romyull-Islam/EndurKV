#!/bin/bash
# ============================================================================
# WATCHDOG GLIDE DEMO (2026-07-18). Cold start -> muKV (fa-on) on Bonsai-8B with
# the FIXED watchdog (zones-by-name + battery/skin ladders). Bonsai heats the
# phone through battery 35/35.5/36, so the watchdog glides the big-core cap
# 1632 -> 1497 -> 1382 -> 1267 gradually. High-rate (0.5s) log of:
#   epoch, scaling_max (watchdog cap), cpu6 cur freq, battery_mc, skin_mc
# so we can plot the glide vs the temperatures that drove it.
# ============================================================================
set -u; export ANDROID_ADB_SERVER_PORT=5151
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
OUT_HOST=/tmp/wd_demo; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/wddemo_$(date +%s 2>/dev/null || echo run)
adb_safe_shell "mkdir -p $OUT" < /dev/null
SCR=/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad
adb push "$SCR/wikitext_16k_p12k_d4k.txt" "$OUT/prompt.txt" < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/preempt_throttle_watchdog_v2.sh \
         /data/local/tmp/preempt_throttle_watchdog_v2.sh < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/wd_logger.sh /data/local/tmp/wd_logger.sh < /dev/null >/dev/null 2>&1
MODEL=/data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf
CB=/data/local/tmp/endurkv/bin_cpu_v87
WD_STOP=/data/local/tmp/cpu_wd.stop
LOG_STOP=/data/local/tmp/wd_demo_log.stop

# charge >=90% then off, so battery temp reflects load not charging
adb_safe_shell "su -c 'echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
CT0=$(date +%s)
while true; do
  cap=$(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/capacity'" < /dev/null|tr -d '\r'); cap=${cap:-0}
  case "$cap" in ''|*[!0-9]*) cap=0;; esac
  [ "$cap" -ge 90 ] && break
  [ $(($(date +%s)-CT0)) -gt 1200 ] && break
  sleep 20
done
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null

# resolve battery/shell zones by name (same as the fixed watchdog)
BZ=$(adb_safe_shell "su -c 'for z in /sys/class/thermal/thermal_zone*; do [ \"\$(cat \$z/type 2>/dev/null)\" = battery ] && { echo \$z; break; }; done'" < /dev/null | tr -d '\r')
SZ=$(adb_safe_shell "su -c 'for z in /sys/class/thermal/thermal_zone*; do [ \"\$(cat \$z/type 2>/dev/null)\" = shell_front ] && { echo \$z; break; }; done'" < /dev/null | tr -d '\r')
DZ=$(adb_safe_shell "su -c 'for z in /sys/class/thermal/thermal_zone*; do [ \"\$(cat \$z/type 2>/dev/null)\" = ddr ] && { echo \$z; break; }; done'" < /dev/null | tr -d '\r')
echo "[zones] battery=$BZ shell=$SZ ddr=$DZ"

# COLD GATE: start below the first battery threshold (34.5C) so the run heats
# THROUGH 35/35.5/36 and shows the full glide from tier 0.
echo "[$(date +%H:%M:%S)] cooling to battery<34.5 ..."
T0=$(date +%s)
while true; do
  bt=$(adb_safe_shell "su -c 'cat $BZ/temp'" < /dev/null|tr -d '\r'); bt=${bt:-99000}
  case "$bt" in ''|*[!0-9]*) bt=99000;; esac
  [ "$bt" -lt 34500 ] && { echo "  [cold battery=$((bt/1000)).$(((bt%1000)/100))C]"; break; }
  [ $(($(date +%s)-T0)) -gt 2400 ] && { echo "  [cool timeout battery=$((bt/1000))C]"; break; }
  sleep 15
done

# start FIXED watchdog + high-rate logger
adb_safe_shell "su -c 'rm -f $WD_STOP $LOG_STOP; nohup sh /data/local/tmp/preempt_throttle_watchdog_v2.sh /data/local/tmp/cpu_wd_demo.log $WD_STOP >/dev/null 2>&1 &'" < /dev/null
adb_safe_shell "su -c 'nohup sh /data/local/tmp/wd_logger.sh $OUT/wd_trace.csv $LOG_STOP $BZ $SZ >/dev/null 2>&1 &'" < /dev/null
sleep 2

echo "[$(date +%H:%M:%S)] running Bonsai muKV with fixed watchdog (heats through the ladder)"
MU="--policy v1_fa2 --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --fa-on-evict"
adb_safe_shell "LD_LIBRARY_PATH=$CB $CB/eviction_bench --prompt $OUT/prompt.txt --prompt-id wddemo --eval-mode gen \
  --max-tokens 1024 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
  --k-nominal 1024 $MU --out-meta $OUT/gen.json --out-csv /dev/null --out-gen /dev/null > $OUT/gen.out 2> $OUT/gen.err" < /dev/null

# stop watchdog + logger, restore
adb_safe_shell "su -c 'touch $WD_STOP $LOG_STOP; sleep 2; for c in cpu0 cpu6; do cat /sys/devices/system/cpu/\$c/cpufreq/cpuinfo_max_freq > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
adb pull "$OUT/wd_trace.csv" "$OUT_HOST/wd_trace.csv" < /dev/null >/dev/null 2>&1
adb pull /data/local/tmp/cpu_wd_demo.log "$OUT_HOST/cpu_wd_demo.log" < /dev/null >/dev/null 2>&1
adb pull "$OUT/gen.err" "$OUT_HOST/gen.err" < /dev/null >/dev/null 2>&1
touch /tmp/wd_demo_DONE
echo "[$(date +%H:%M:%S)] WATCHDOG DEMO DONE -> $OUT_HOST"
echo "--- watchdog tier transitions ---"; grep -iE 'tier|engaged|MHz' "$OUT_HOST/cpu_wd_demo.log" 2>/dev/null | tail -12
