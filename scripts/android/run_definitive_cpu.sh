#!/bin/bash
# ============================================================================
# DEFINITIVE same-condition CPU table (HotMobile) -- full columns per policy:
#   PPL | prefill | decode_tps | wall | energy | live_cells | peak temps
#
# SAME CONDITION FOR ALL (fixes the SoC energy confound):
#   - Charge to >=90% ONCE at the start, then charging stays OFF for the ENTIRE
#     run (never toggled -> no battery-recharge USB spikes). Cool between cells by
#     IDLE-WAIT (charging off). Every cell starts cool (<=50 C big-core) at a
#     similar, monotonically-slowly-declining SoC.
#   - Energy = integral of TOTAL system power = USB rail (V*I) + battery-discharge
#     power (I>0 * V). Split-robust: correct whether the load is served by USB,
#     battery, or both.
#   - Platform: big cores capped 1632 for ALL cells equally. Baselines native (no
#     watchdog); muKV + surface-aware watchdog v2 (muKV-only; dormant when cool).
#
# Per policy: (1) GEN pass -> prefill_ms, decode_tps, wall, live_cells, energy, temps
#             (2) PPL pass -> teacher-forced perplexity on disjoint wiki_eval_1k.txt
# WikiText 9737-token prompt + 4096 decode, ctx 16384, Llama-3.2-1B Q4_K_M, 6 thr,
# k-nominal 1024, bin_cpu_sol2 (canonical SnapKV). Canonical SnapKV = --policy snapkv
# --obs-window 64 --n-sink 0. muKV = state-swap mass-full.
# ============================================================================
set -u; export ANDROID_ADB_SERVER_PORT=5151
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
OUT_HOST=/tmp/def_cpu; mkdir -p "$OUT_HOST"
TS=$(date +%Y%m%d_%H%M%S); OUT=/data/local/tmp/endurkv/logs/defcpu_$TS
adb_safe_shell "mkdir -p $OUT" < /dev/null
SCR=/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad
adb push "$SCR/wikitext_16k_p12k_d4k.txt" "$OUT/prompt.txt" < /dev/null >/dev/null 2>&1
adb push "$SCR/wiki_eval_disjoint.txt" "$OUT/eval.txt" < /dev/null >/dev/null 2>&1   # DISJOINT continuation (generalization PPL, not recall)
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/preempt_throttle_watchdog_v2.sh \
         /data/local/tmp/preempt_throttle_watchdog_v2.sh < /dev/null >/dev/null 2>&1
MODEL=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
CB=/data/local/tmp/endurkv/bin_cpu_sol2
WD_STOP=/data/local/tmp/cpu_wd.stop

# --- one-time: charge to >=90%, then charging OFF for the whole run ---
echo "[$(date +%H:%M:%S)] pre-charge to >=90% (same SoC start for all)"
adb_safe_shell "su -c 'echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
CT0=$(date +%s)
while true; do
  cap=$(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/capacity'" < /dev/null|tr -d '\r'); cap=${cap:-0}
  case "$cap" in ''|*[!0-9]*) cap=0;; esac   # guard against non-numeric (permission/err)
  [ "$cap" -ge 90 ] && { echo "  [charged ${cap}%]"; break; }
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
# STRICT idle cool-gate (charging stays OFF): CPU<36 AND DDR<36 AND shell<34 for a genuinely
# cold thermal start. Timeout 600s (cooling from ~58C DDR takes several minutes).
coolidle(){ local T0=$(date +%s)
  while true; do
    read cpu ddr sh <<<"$(adb_safe_shell "su -c 'm=0; for z in $CPU_ZS; do t=\$(cat /sys/class/thermal/thermal_zone\$z/temp 2>/dev/null); [ \$t -gt \$m ]&&m=\$t; done; d=\$(cat /sys/class/thermal/thermal_zone${DDR_Z}/temp 2>/dev/null); s=\$(cat /sys/class/thermal/thermal_zone${SHELL_Z}/temp 2>/dev/null); echo \$((m/1000)) \$((d/1000)) \$((s/1000))'" < /dev/null|tr -d '\r')"
    cpu=${cpu:-99}; ddr=${ddr:-99}; sh=${sh:-99}
    # TRUE cold start (24C room dissipates heat-soak in ~26min): CPU<37/DDR<37/shell<34.
    # Long timeout because a bench heats to ~70C and cooling back to <37 takes ~25min.
    if [ "$cpu" -lt 37 ] && [ "$ddr" -lt 37 ] && [ "$sh" -lt 34 ]; then echo "  [cold cpu=$cpu ddr=$ddr shell=$sh]"; return; fi
    [ $(($(date +%s)-T0)) -gt 1800 ] && { echo "  [cool timeout cpu=$cpu ddr=$ddr shell=$sh]"; return; }
    sleep 10
  done; }

# run <cell> <use_wd:0|1> <policy-args...>
run(){ local CELL=$1 WD=$2; shift 2; local PD=$OUT/$CELL
  echo "[$(date +%H:%M:%S)] $CELL (wd=$WD)"; platform_cap; coolidle
  adb_safe_shell "mkdir -p $PD" < /dev/null
  [ "$WD" = 1 ] && start_wd "$CELL"
  # (1) GEN pass with sensors
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  sleep 2; local t0=$(date +%s%N)
  adb_safe_shell "LD_LIBRARY_PATH=$CB $CB/eviction_bench --prompt $OUT/prompt.txt --prompt-id ${CELL}_gen --eval-mode gen \
    --max-tokens 4096 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --k-nominal 1024 $* --out-meta $PD/gen.json --out-csv $PD/gen_steps.csv --out-prefill-csv $PD/gen_prefill.csv --out-gen $PD/gen.txt > $PD/gen.out 2> $PD/gen.err" < /dev/null
  local t1=$(date +%s%N)
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  echo "$(( (t1-t0)/1000000 ))" > "$OUT_HOST/${CELL}_wall_ms" 2>/dev/null; mkdir -p "$OUT_HOST/$CELL"; echo "$(( (t1-t0)/1000000 ))" > "$OUT_HOST/$CELL/wall_ms"
  # (2) PPL pass (teacher-forced disjoint eval, no long decode)
  adb_safe_shell "LD_LIBRARY_PATH=$CB $CB/eviction_bench --prompt $OUT/prompt.txt --prompt-id ${CELL}_ppl --eval-mode ppl --eval-text $OUT/eval.txt \
    --max-tokens 512 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --k-nominal 1024 $* --out-meta $PD/ppl.json --out-csv /dev/null --out-gen /dev/null > $PD/ppl.out 2> $PD/ppl.err" < /dev/null
  [ "$WD" = 1 ] && { stop_wd; adb pull /data/local/tmp/cpu_wd_$CELL.log "$OUT_HOST/${CELL}_wd.log" < /dev/null >/dev/null 2>&1; }
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  echo "  [done $CELL] gen=$(grep -oE 'decode_tps=[0-9.]+' "$OUT_HOST/$CELL/gen.err" 2>/dev/null|head -1) ppl=$(grep -oiE 'ppl[= ][0-9.]+|perplexity[= :]+[0-9.]+' "$OUT_HOST/$CELL/ppl.err" 2>/dev/null|head -1)"
}
MU="--policy v1_fa2 --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
# muKV-mass-full FIRST (the GO/NO-GO checkpoint), then baselines + swap alt.
run mukv_faon  1 $MU --fa-on-evict   # PRIMARY muKV-mass-full = fa-on + CPU defrag: ~vanilla TTFT + tight/fast decode
run vanilla    0 --policy vanilla
run snapkv     0 --policy snapkv --obs-window 64 --n-sink 0
run adakv      0 --policy adakv --n-sink 0 --obs-window 32   # Ada-SnapKV canonical: window 32, maxpool-7, alpha=0.2, NO sinks
run mukv_swap  1 $MU                 # ALTERNATIVE: state-swap (FA-off prefill capture -> FA-on decode)
# restore
adb_safe_shell "su -c 'touch $WD_STOP; for c in cpu6 cpu7; do cat /sys/devices/system/cpu/\$c/cpufreq/cpuinfo_max_freq > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
touch /tmp/def_cpu_DONE; echo "[$(date +%H:%M:%S)] DEFINITIVE CPU DONE -> $OUT_HOST"
