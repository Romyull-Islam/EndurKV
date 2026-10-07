#!/bin/bash
# Deployment-model CPU run: Llama-3.2-1B Q4_K_M, WikiText prompt + 4096 decode, ctx 16384, K=1024.
# Every cell has big cores (cpu6/7) capped at 1632 MHz, a sustainable clock, and a cool gate
# (charging off, big cores <=52 C) for equal thermal starts.
# Baselines (vanilla, SnapKV defaults, AdaKV) run without a watchdog. muKV adds the reduce-only
# watchdog v2, which steps the clock down from 1632 MHz before the kernel throttles.
# muKV uses state-swap here, fa-on-evict is the GPU path.
set -u; export ANDROID_ADB_SERVER_PORT=5150
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

OUT_HOST=/tmp/deploy_cpu; mkdir -p "$OUT_HOST"
TS=$(date +%Y%m%d_%H%M%S); OUT=/data/local/tmp/endurkv/logs/deploycpu_$TS
adb_safe_shell "mkdir -p $OUT" < /dev/null
SCR="${SCR:-$(cd "$(dirname "$0")/../.." && pwd)/eval_corpora}"
adb push "$SCR/wikitext_16k_p12k_d4k.txt" "$OUT/prompt.txt" < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/preempt_throttle_watchdog_v2.sh \
         /data/local/tmp/preempt_throttle_watchdog_v2.sh < /dev/null >/dev/null 2>&1

MODEL=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
CB=/data/local/tmp/endurkv/bin_cpu_sol2
WD_STOP=/data/local/tmp/cpu_wd.stop

# static 1632 MHz cap on big cores, watchdog off
platform_native_engine(){
  adb_safe_shell "su -c 'touch $WD_STOP; for c in cpu6 cpu7; do echo 1632000 > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done'" < /dev/null
}
# muKV: same cap plus the surface-aware watchdog v2
start_watchdog(){ local tag=$1
  adb_safe_shell "su -c 'rm -f $WD_STOP; nohup sh /data/local/tmp/preempt_throttle_watchdog_v2.sh /data/local/tmp/cpu_wd_$tag.log $WD_STOP >/dev/null 2>&1 &'" < /dev/null; }
stop_watchdog(){ adb_safe_shell "su -c 'touch $WD_STOP'" < /dev/null; }

coolcpu(){ adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; local T0=$(date +%s)
  while true; do c=$(adb_safe_shell "su -c 'm=0; for z in 0 5 10 17 24; do t=\$(cat /sys/class/thermal/thermal_zone\$z/temp 2>/dev/null); [ \$t -gt \$m ]&&m=\$t; done; echo \$((m/1000))'" < /dev/null|tr -d '\r'); c=${c:-99}
    [ "$c" -le 52 ] && { echo "  [cpu ${c}C]"; return; }; [ $(($(date +%s)-T0)) -gt 240 ] && { echo "  [cpu timeout ${c}C]"; return; }; sleep 8; done; }

# run <cell> <use_watchdog:0|1> <policy-args...>
run(){ local CELL=$1; local WD=$2; shift 2; local PD=$OUT/$CELL
  echo "[$(date +%H:%M:%S)] $CELL (watchdog=$WD)"
  platform_native_engine; coolcpu
  [ "$WD" = 1 ] && start_watchdog "$CELL"
  adb_safe_shell "mkdir -p $PD" < /dev/null
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  sleep 2; local t0=$(date +%s%N)
  adb_safe_shell "LD_LIBRARY_PATH=$CB $CB/eviction_bench --prompt $OUT/prompt.txt --prompt-id $CELL --eval-mode gen \
    --max-tokens 4096 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --k-nominal 1024 $* --out-meta $PD/meta.json --out-csv /dev/null --out-gen /dev/null > $PD/out 2> $PD/err" < /dev/null
  local t1=$(date +%s%N)
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  [ "$WD" = 1 ] && { stop_watchdog; adb pull /data/local/tmp/cpu_wd_$CELL.log "$OUT_HOST/${CELL}_wd.log" < /dev/null >/dev/null 2>&1; }
  mkdir -p "$OUT_HOST/$CELL"; echo "$(( (t1-t0)/1000000 ))" > "$OUT_HOST/$CELL/wall_ms"
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  echo "  [done $CELL] wall=$(( (t1-t0)/1000000000 ))s  $(grep -oE 'decode_tps=[0-9.]+|peak_kv=[0-9]+|prefill_ms=[0-9.]+' "$OUT_HOST/$CELL/err" 2>/dev/null|tr '\n' ' ')"
}

MU="--policy v1_fa2 --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

# baselines, no watchdog
run vanilla  0 --policy vanilla
run snapkv   0 --policy snapkv --obs-window 64 --n-sink 0
run adakv    0 --policy adakv  --n-sink 4 --obs-window 16
# muKV with watchdog
run mukv     1 $MU

# restore: watchdog off, native max clock, charging on
adb_safe_shell "su -c 'touch $WD_STOP; for c in cpu6 cpu7; do cat /sys/devices/system/cpu/\$c/cpufreq/cpuinfo_max_freq > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
touch /tmp/deploy_cpu_DONE; echo "[$(date +%H:%M:%S)] DEPLOY-CPU DONE  ->  $OUT_HOST"
