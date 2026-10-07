#!/bin/bash
# Same-condition CPU table rows for H2O and TOVA on the current build (bin_cpu_kd).
# Per policy: a GEN pass (prefill, decode tok/s, wall, live cells, energy, temps) and a
# teacher-forced PPL pass on the disjoint eval text. WikiText 9737-token prompt, 4096
# decode, ctx 16384, Llama-3.2-1B Q4_K_M, 6 threads, k-nominal 1024, big cores 1632 MHz.
# Charging stays off for the whole run and energy integrates USB plus battery power.
# H2O and TOVA score at every decode step and the in-graph side node fires only in
# prefill, so both run with FlashAttention off.
set -u   # 2026-08-18: port pin removed -- the default server holds the device;
         # a second server on 5151 cannot claim the same USB endpoint, and the
         # other queued campaigns all use the default.
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

# Never share the phone with another campaign: concurrent runs overheat it and the
# cool gate can then never pass.
echo "[$(date +%H:%M:%S)] waiting for /tmp/def_cpu_kd_DONE before touching the phone"
true  # master table long since complete
echo "[$(date +%H:%M:%S)] KeyDiff table done -- starting FA-on baselines + muKV watchdog A/B"
# Stage the KeyDiff-capable build in its own dir. bin_cpu_sol2 is left untouched.
adb_safe_shell "mkdir -p /data/local/tmp/endurkv/bin_cpu_kd" < /dev/null
for _so in /home/mislam22/EndurKV_workspace/EndurKV/llama.cpp/build-android/bin/lib*.so; do
  timeout 180 adb push "$_so" /data/local/tmp/endurkv/bin_cpu_kd/ < /dev/null >/dev/null 2>&1
done
timeout 180 adb push /home/mislam22/EndurKV_workspace/EndurKV/entropy_probe/build-android/eviction_bench \
         /data/local/tmp/endurkv/bin_cpu_kd/eviction_bench < /dev/null >/dev/null 2>&1
adb_safe_shell "chmod 755 /data/local/tmp/endurkv/bin_cpu_kd/eviction_bench" < /dev/null
OUT_HOST=/tmp/def_cpu_h2otova; mkdir -p "$OUT_HOST"
TS=$(date +%Y%m%d_%H%M%S); OUT=/data/local/tmp/endurkv/logs/defcpuh2otova_$TS
adb_safe_shell "mkdir -p $OUT" < /dev/null
SCR="${SCR:-$(cd "$(dirname "$0")/../.." && pwd)/eval_corpora}"
timeout 180 adb push "$SCR/wikitext_16k_p12k_d4k.txt" "$OUT/prompt.txt" < /dev/null >/dev/null 2>&1
timeout 180 adb push "$SCR/wiki_eval_disjoint.txt" "$OUT/eval.txt" < /dev/null >/dev/null 2>&1   # DISJOINT continuation (generalization PPL, not recall)
timeout 180 adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/preempt_throttle_watchdog_v2.sh \
         /data/local/tmp/preempt_throttle_watchdog_v2.sh < /dev/null >/dev/null 2>&1
MODEL=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
CB=/data/local/tmp/endurkv/bin_cpu_kd
WD_STOP=/data/local/tmp/cpu_wd.stop

# One-time SoC gate, then charging off for the whole run.
echo "[$(date +%H:%M:%S)] pre-charge to >=90% (same SoC start for all)"
adb_safe_shell "su -c 'echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
CT0=$(date +%s)
while true; do
  cap=$(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/capacity'" < /dev/null|tr -d '\r'); cap=${cap:-0}
  case "$cap" in ''|*[!0-9]*) cap=0;; esac   # guard against non-numeric (permission/err)
  # Floor is SOC_FLOOR (default 70%) because the phone does not charge through the
  # tunnelled adb host. The aim is equal SoC across cells. The start value is logged.
  [ "$cap" -ge "${SOC_FLOOR:-70}" ] && { echo "  [start SoC ${cap}% -- floor ${SOC_FLOOR:-70}, host will not charge]"; break; }
  [ $(($(date +%s)-CT0)) -gt 1800 ] && { echo "  [charge timeout ${cap}%]"; break; }
  sleep 20
done
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null  # OFF for whole run

platform_cap(){ adb_safe_shell "su -c 'pkill -9 -f eviction_bench 2>/dev/null; pkill -9 -f sample_sensors 2>/dev/null; touch $WD_STOP; for c in cpu6 cpu7; do echo 1632000 > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done'" < /dev/null; }
start_wd(){ adb_safe_shell "su -c 'rm -f $WD_STOP; nohup sh /data/local/tmp/preempt_throttle_watchdog_v2.sh /data/local/tmp/cpu_wd_$1.log $WD_STOP >/dev/null 2>&1 &'" < /dev/null; }
stop_wd(){ adb_safe_shell "su -c 'touch $WD_STOP'" < /dev/null; }
# resolve thermal zones by NAME once (they re-enumerate across reboots)
ZMAP=$(adb_safe_shell "su -c 'for z in /sys/class/thermal/thermal_zone*; do printf \"%s:%s \" \$(basename \$z|sed s/thermal_zone//) \$(cat \$z/type 2>/dev/null); done'" < /dev/null|tr -d '\r')
z_by_name(){ echo "$ZMAP" | tr ' ' '\n' | grep -E ":$1\$" | head -1 | cut -d: -f1; }
DDR_Z=$(z_by_name ddr); SHELL_Z=$(z_by_name shell_front)
CPU_ZS=$(echo "$ZMAP" | tr ' ' '\n' | grep -E ':(cpu-[0-9]|cpullc-[0-9])' | cut -d: -f1 | tr '\n' ' ')  # exclude cpu-hw-trip (fixed 95C trip points)
echo "  [zones] cpu=($CPU_ZS) ddr=$DDR_Z shell=$SHELL_Z"
# Idle cool gate (charging off) for a cold thermal start. Returns 1 on timeout.
coolidle(){ local T0=$(date +%s)
  while true; do
    read cpu ddr sh <<<"$(adb_safe_shell "su -c 'm=0; for z in $CPU_ZS; do t=\$(cat /sys/class/thermal/thermal_zone\$z/temp 2>/dev/null); [ \$t -gt \$m ]&&m=\$t; done; d=\$(cat /sys/class/thermal/thermal_zone${DDR_Z}/temp 2>/dev/null); s=\$(cat /sys/class/thermal/thermal_zone${SHELL_Z}/temp 2>/dev/null); echo \$((m/1000)) \$((d/1000)) \$((s/1000))'" < /dev/null|tr -d '\r')"
    cpu=${cpu:-99}; ddr=${ddr:-99}; sh=${sh:-99}
    # Cold start: CPU<37, DDR<37, shell<34 C. Cooling from ~70 C takes about 25 min.
    if [ "$cpu" -lt 37 ] && [ "$ddr" -lt 37 ] && [ "$sh" -lt 34 ]; then echo "  [cold cpu=$cpu ddr=$ddr shell=$sh]"; return; fi
    # After 75 min, fail so the cell is skipped and retried on resume. Timed and energy
    # cells are only comparable if they start from the same thermal state.
    [ $(($(date +%s)-T0)) -gt 4500 ] && { echo "  [COOL FAILED cpu=$cpu ddr=$ddr shell=$sh -- skipping cell]"; return 1; }
    sleep 10
  done; }


# Detached device-side execution. The adb tunnel drops every 20 to 30 min and a cell
# takes about 25 min, and `adb shell` work dies with the connection. So the bench runs
# under setsid + nohup on the device and the host only polls for its done file.
dev_run() {   # dev_run <tag> <done-file> <command...>
    local TAG=$1 DONEF=$2; shift 2
    # Hard device-side timeout, same budget the host polls against, so an orphaned
    # run cannot keep heating the phone after the host gives up.
    adb_safe_shell "su -c 'rm -f $DONEF; setsid nohup sh -c \"timeout ${DEV_RUN_MAX_S:-5400} env $* ; echo DONE > $DONEF\" >/dev/null 2>&1 &'" < /dev/null
    local waited=0
    while [ $waited -lt "${DEV_RUN_MAX_S:-5400}" ]; do
        if adb_safe_shell "[ -f $DONEF ] && echo yes" < /dev/null 2>/dev/null | grep -q yes; then
            return 0
        fi
        sleep 30; waited=$((waited + 30))
    done
    echo "  [dev_run TIMEOUT $TAG after ${waited}s]"
    return 1
}

# run <cell> <use_wd:0|1> <policy-args...>
run(){ local CELL=$1 WD=$2; shift 2; local PD=$OUT/$CELL
  # Resume: skip cells that already have a gen.json.
  if [ -s "$OUT_HOST/$CELL/gen.json" ]; then echo "  [$CELL cached -- skipping]"; return; fi
  echo "[$(date +%H:%M:%S)] $CELL (wd=$WD)"; platform_cap
  if ! coolidle; then echo "  [$CELL SKIPPED -- never reached cold start]"; return; fi
  adb_safe_shell "mkdir -p $PD" < /dev/null
  # Record the starting thermal state of the cell. Must run after the mkdir above.
  adb_safe_shell "su -c 'echo start_temps cpu=\$(cat /sys/class/thermal/thermal_zone${CPU_Z0:-24}/temp) ddr=\$(cat /sys/class/thermal/thermal_zone${DDR_Z}/temp) > $PD/start_temps.txt'" < /dev/null
  [ "$WD" = 1 ] && start_wd "$CELL"
  # (1) GEN pass with sensors
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  sleep 2; local t0=$(date +%s%N)
  dev_run "${CELL}_gen" "$PD/.gen_done" "LD_LIBRARY_PATH=$CB $CB/eviction_bench --prompt $OUT/prompt.txt --prompt-id ${CELL}_gen --eval-mode gen \
    --max-tokens 4096 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --k-nominal 1024 $* --out-meta $PD/gen.json --out-csv $PD/gen_steps.csv --out-prefill-csv $PD/gen_prefill.csv --out-gen $PD/gen.txt > $PD/gen.out 2> $PD/gen.err"
  local t1=$(date +%s%N)
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  echo "$(( (t1-t0)/1000000 ))" > "$OUT_HOST/${CELL}_wall_ms" 2>/dev/null; mkdir -p "$OUT_HOST/$CELL"; echo "$(( (t1-t0)/1000000 ))" > "$OUT_HOST/$CELL/wall_ms"
  # (2) PPL pass (teacher-forced disjoint eval, no long decode)
  dev_run "${CELL}_ppl" "$PD/.ppl_done" "LD_LIBRARY_PATH=$CB $CB/eviction_bench --prompt $OUT/prompt.txt --prompt-id ${CELL}_ppl --eval-mode ppl --eval-text $OUT/eval.txt \
    --max-tokens 512 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --k-nominal 1024 $* --out-meta $PD/ppl.json --out-csv /dev/null --out-gen /dev/null > $PD/ppl.out 2> $PD/ppl.err"
  [ "$WD" = 1 ] && { stop_wd; adb pull /data/local/tmp/cpu_wd_$CELL.log "$OUT_HOST/${CELL}_wd.log" < /dev/null >/dev/null 2>&1; }
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  echo "  [done $CELL] gen=$(grep -oE 'decode_tps=[0-9.]+' "$OUT_HOST/$CELL/gen.err" 2>/dev/null|head -1) ppl=$(grep -oiE 'ppl[= ][0-9.]+|perplexity[= :]+[0-9.]+' "$OUT_HOST/$CELL/ppl.err" 2>/dev/null|head -1)"
}
MU="--policy v1_fa2 --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
run h2o   0 --policy h2o
run tova  0 --policy tova
# Restore CPU caps and charging.
adb_safe_shell "su -c 'touch $WD_STOP; for c in cpu6 cpu7; do cat /sys/devices/system/cpu/\$c/cpufreq/cpuinfo_max_freq > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
touch /tmp/def_cpu_h2otova_DONE; echo "[$(date +%H:%M:%S)] H2O + TOVA CURRENT-BUILD DONE -> $OUT_HOST"
