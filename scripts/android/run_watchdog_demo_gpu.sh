#!/bin/bash
# ============================================================================
# WATCHDOG DEMONSTRATION (GPU, Adreno 840) -- the load-bearing thermal test.
# Proves the surface-aware reduce-only watchdog PREVENTS the kernel throttle.
#
# Three cells, each from a settled-idle start (natural equilibrium), NATIVE GPU
# clock (uncapped -> heats), sustained 4096-token decode:
#   1. vanilla_nowd  : baseline, no watchdog        -> heats, kernel THROTTLES
#   2. mukv_nowd      : muKV fa-on-evict, NO watchdog -> muKV alone still throttles
#   3. mukv_wd        : muKV fa-on-evict, WATCHDOG v4  -> watchdog glides clock DOWN
#                       before onset -> NO throttle, temp bounded
# The (2)-vs-(3) contrast isolates the watchdog on the SAME policy; (1) is the
# baseline. The proof is in the GPU-clock trace (gpu_clk_hz): a THROTTLE is a
# sudden cliff; the WATCHDOG is a gradual glide that keeps temp under threshold.
#
# CAPTURES per cell (5 Hz sensors.csv): GPU/DDR/skin temps + gpu_clk_hz + tps.
# Watchdog cell also logs tier transitions (which sensor tripped, when).
# ============================================================================
set -u; export ANDROID_ADB_SERVER_PORT=5151
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
OUT_HOST=/tmp/wd_demo; mkdir -p "$OUT_HOST"
TS=$(date +%Y%m%d_%H%M%S); OUT=/data/local/tmp/endurkv/logs/wddemo_$TS
adb_safe_shell "mkdir -p $OUT" < /dev/null
SCR=/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad
adb push "$SCR/wikitext_16k_p12k_d4k.txt" "$OUT/prompt.txt" < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/gpu_watchdog_v4_surface_aware.sh /data/local/tmp/gpu_watchdog_v4.sh < /dev/null >/dev/null 2>&1
MODEL=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
VKB=/data/local/tmp/endurkv/bin_vulkan_new; VKLIB=/data/local/tmp/endurkv/bin_vulkan
WD_STOP=/data/local/tmp/gpu_wd.stop

start_wd(){ adb_safe_shell "su -c 'rm -f $WD_STOP; nohup sh /data/local/tmp/gpu_watchdog_v4.sh /data/local/tmp/gpu_wd_$1.log $WD_STOP >/dev/null 2>&1 &'" < /dev/null; }
stop_wd(){ adb_safe_shell "su -c 'touch $WD_STOP; sleep 1; echo 1200 > /sys/kernel/gpu/gpu_max_clock'" < /dev/null; }
# deep settle: GPU+DDR <=37C, stable >=90s, native GPU clock (uncapped, so it can heat)
settle(){ local TAG=$1
  # CRITICAL: kill any leftover bench/sensors first (a stray bench pins cores -> phone never cools)
  adb_safe_shell "su -c 'pkill -9 -f eviction_bench 2>/dev/null; pkill -9 -f sample_sensors 2>/dev/null; input keyevent 26 2>/dev/null; echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable; touch $WD_STOP; echo 1200 > /sys/kernel/gpu/gpu_max_clock; echo 160 > /sys/kernel/gpu/gpu_min_clock'" < /dev/null
  local T0=$(date +%s); local h1=999 h2=999 h3=999 h4=999 h5=999 h6=999 h7=999 h8=999 h9=999
  while true; do
    read g d <<<"$(adb_safe_shell "su -c 'g=0; for z in 36 40 44 46; do t=\$(cat /sys/class/thermal/thermal_zone\$z/temp); [ \$t -gt \$g ] && g=\$t; done; d=\$(cat /sys/class/thermal/thermal_zone47/temp); echo \$((g/1000)) \$((d/1000))'" < /dev/null|tr -d '\r')"
    g=${g:-99}; d=${d:-99}; local hot=$g; [ "$d" -gt "$hot" ] && hot=$d
    h9=$h8;h8=$h7;h7=$h6;h6=$h5;h5=$h4;h4=$h3;h3=$h2;h2=$h1;h1=$hot
    local mn=$h1 mx=$h1; for v in $h2 $h3 $h4 $h5 $h6 $h7 $h8 $h9; do [ "$v" -lt "$mn" ]&&mn=$v; [ "$v" -gt "$mx" ]&&mx=$v; done
    local el=$(( $(date +%s)-T0 ))
    if [ "$el" -ge 90 ] && [ $((mx-mn)) -le 2 ] && [ "$g" -le 37 ] && [ "$d" -le 37 ]; then echo "  [settled $TAG] gpu=$g ddr=$d (${el}s)"; return; fi
    [ "$el" -gt 1800 ] && { echo "  [settle timeout $TAG] gpu=$g ddr=$d (${el}s)"; return; }
    [ $((el % 300)) -lt 10 ] && echo "  [cooling $TAG] gpu=$g ddr=$d (${el}s)"; sleep 10
  done; }

# run <cell> <use_wd:0|1> <policy-args...>
run(){ local CELL=$1 WD=$2; shift 2; local PD=$OUT/$CELL
  echo "[$(date +%H:%M:%S)] $CELL (wd=$WD)"; settle "$CELL"
  [ "$WD" = 1 ] && start_wd "$CELL"
  adb_safe_shell "mkdir -p $PD" < /dev/null
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  sleep 2; local t0=$(date +%s%N)
  adb_safe_shell "LD_LIBRARY_PATH=$VKLIB $VKB/eviction_bench --prompt $OUT/prompt.txt --prompt-id $CELL --eval-mode gen \
    --max-tokens 4096 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --k-nominal 1024 $* --out-meta $PD/meta.json --out-csv $PD/steps.csv --out-gen /dev/null > $PD/out 2> $PD/err" < /dev/null
  local t1=$(date +%s%N)
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  [ "$WD" = 1 ] && { stop_wd; adb pull /data/local/tmp/gpu_wd_$CELL.log "$OUT_HOST/${CELL}_wd.log" < /dev/null >/dev/null 2>&1; }
  mkdir -p "$OUT_HOST/$CELL"; echo "$(( (t1-t0)/1000000 ))" > "$OUT_HOST/$CELL/wall_ms"
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  echo "  [done $CELL] wall=$(( (t1-t0)/1000000000 ))s $(grep -oE 'decode_tps=[0-9.]+' "$OUT_HOST/$CELL/err" 2>/dev/null|head -1)"
}
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
run vanilla_nowd 0 --policy vanilla
run mukv_nowd    0 $MU
run mukv_wd      1 $MU
adb_safe_shell "su -c 'touch $WD_STOP; sleep 1; echo 1200 > /sys/kernel/gpu/gpu_max_clock; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
touch /tmp/wd_demo_DONE; echo "[$(date +%H:%M:%S)] WATCHDOG DEMO DONE -> $OUT_HOST"
