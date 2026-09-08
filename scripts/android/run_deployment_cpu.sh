#!/bin/bash
# ============================================================================
# DEFINITIVE deployment-model CPU run  --  HotMobile thermal-aware KV paper
# (finalized 2026-07-16, "restart with correct run")
#
# EXPERIMENTAL PLATFORM (applied to EVERY cell equally, NOT a muKV optimization):
#   big cores cpu6/cpu7 statically capped at 1632 MHz -- the realistic *sustained*
#   mobile-inference clock (the 3.4 GHz native boost is not thermally sustainable
#   and would confound duration/energy). Same platform for baselines AND muKV.
#
# WATCHDOG MODEL (muKV-ONLY, per user):
#   - Baselines (vanilla, SnapKV, AdaKV): platform cap only, NO watchdog. The
#     phone's own thermal engine is free to throttle below 1632 under heat
#     (kernel cliff -> ~883 MHz). This is their canonical/default setup, unmodified.
#   - muKV: platform cap + surface-aware reduce-only watchdog v2
#     (preempt_throttle_watchdog_v2.sh). Reads DDR/CPU/skin(shell_front)/battery,
#     GLIDES the clock down (1632->1497->1382->1267) BEFORE the kernel cliff, and
#     NEVER raises it. This is muKV's thermal actuator -- the only asymmetry.
#
# CANONICAL SnapKV (exact FasterDecoding/SnapKV defaults, verified from
#   snapkv_utils.py -- NONE of muKV's tricks):
#     --policy snapkv --k-nominal 1024 --obs-window 64 --n-sink 0
#   window_size=64 (always kept), kernel_size=5, avgpool, per-head, top-(K-64)
#   from the prefix. On llama.cpp's sequence-level engine the per-head union barely
#   compacts (peak_kv stays near full) -- the on-device realizability point, as DATA.
#
# muKV (state-swap, CPU-optimal decode; NO fa-on-evict here -- that's the GPU path):
#     --policy v1_fa2 --n-sink 4 --adaptive-anchor --adaptive-rmin 32
#     --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70
#
# ALL cells: WikiText 9737-token prompt + 4096 decode, ctx 16384,
#   Llama-3.2-1B-Instruct Q4_K_M, 6 threads, k-nominal 1024, bin_cpu_sol2
#   (code-matched 07-16, canonical SnapKV). Cool-gate (charging off + wait
#   big-core <=52 C) before every cell for equal thermal starts.
#
# CAPTURES per cell: peak_kv (realizability), prefill ms, decode tps, wall s,
#   energy (sensors.csv), peak DDR/CPU/skin temps, watchdog tier log (muKV only).
# ============================================================================
set -u; export ANDROID_ADB_SERVER_PORT=5150
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

OUT_HOST=/tmp/deploy_cpu; mkdir -p "$OUT_HOST"
TS=$(date +%Y%m%d_%H%M%S); OUT=/data/local/tmp/endurkv/logs/deploycpu_$TS
adb_safe_shell "mkdir -p $OUT" < /dev/null
SCR=/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad
adb push "$SCR/wikitext_16k_p12k_d4k.txt" "$OUT/prompt.txt" < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/preempt_throttle_watchdog_v2.sh \
         /data/local/tmp/preempt_throttle_watchdog_v2.sh < /dev/null >/dev/null 2>&1

MODEL=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
CB=/data/local/tmp/endurkv/bin_cpu_sol2
WD_STOP=/data/local/tmp/cpu_wd.stop

# -- platform: static 1632 cap on big cores, watchdog OFF (baseline default state) --
platform_native_engine(){
  adb_safe_shell "su -c 'touch $WD_STOP; for c in cpu6 cpu7; do echo 1632000 > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done'" < /dev/null
}
# -- muKV thermal actuator: same 1632 cap, PLUS surface-aware watchdog v2 running --
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

# ---- baselines: canonical setup, NO watchdog (WD=0) ----
run vanilla  0 --policy vanilla
run snapkv   0 --policy snapkv --obs-window 64 --n-sink 0
run adakv    0 --policy adakv  --n-sink 4 --obs-window 16
# ---- muKV: surface-aware watchdog v2 (WD=1) ----
run mukv     1 $MU

# restore: watchdog off, native max clock, charging on
adb_safe_shell "su -c 'touch $WD_STOP; for c in cpu6 cpu7; do cat /sys/devices/system/cpu/\$c/cpufreq/cpuinfo_max_freq > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
touch /tmp/deploy_cpu_DONE; echo "[$(date +%H:%M:%S)] DEPLOY-CPU DONE  ->  $OUT_HOST"
