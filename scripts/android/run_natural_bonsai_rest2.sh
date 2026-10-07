#!/bin/bash
# Bonsai-8B CPU campaign under natural DVFS: no artificial clock cap, so the vendor
# DVFS and thermal engine govern every cell the same way from a cold start.
# Baselines run without a watchdog, muKV runs with watchdog v2.
# WikiText prompt plus 4096 decoded tokens, ctx 16384, K=1024, on the armv8.7-a build
# bin_cpu_v87, whose results are not comparable with the older armv8-a builds.
set -u; export ANDROID_ADB_SERVER_PORT=5151
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
OUT_HOST=/tmp/nat_bonsai; mkdir -p "$OUT_HOST"
TS=$(date +%Y%m%d_%H%M%S); OUT=/data/local/tmp/endurkv/logs/natbonsai_$TS
adb_safe_shell "mkdir -p $OUT" < /dev/null
SCR="${SCR:-$(cd "$(dirname "$0")/../.." && pwd)/eval_corpora}"
adb push "$SCR/wikitext_16k_p12k_d4k.txt" "$OUT/prompt.txt" < /dev/null >/dev/null 2>&1
adb push "$SCR/wiki_eval_disjoint.txt" "$OUT/eval.txt" < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/preempt_throttle_watchdog_v2.sh \
         /data/local/tmp/preempt_throttle_watchdog_v2.sh < /dev/null >/dev/null 2>&1
MODEL=/data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf
CB=/data/local/tmp/endurkv/bin_cpu_v87        # NEW armv8.7-a build
WD_STOP=/data/local/tmp/cpu_wd.stop

echo "[$(date +%H:%M:%S)] pre-charge to >=90% (same SoC start for all)"
adb_safe_shell "su -c 'echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
CT0=$(date +%s)
while true; do
  cap=$(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/capacity'" < /dev/null|tr -d '\r'); cap=${cap:-0}
  case "$cap" in ''|*[!0-9]*) cap=0;; esac
  [ "$cap" -ge 90 ] && { echo "  [charged ${cap}%]"; break; }
  [ $(($(date +%s)-CT0)) -gt 1800 ] && { echo "  [charge timeout ${cap}%]"; break; }
  sleep 20
done
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null

# Undo any leftover clock cap by restoring scaling_max to the vendor ceiling
# (the vendor clamps a cpuinfo_max write to 2438).
platform_natural(){ adb_safe_shell "su -c 'pkill -9 -f eviction_bench 2>/dev/null; pkill -9 -f sample_sensors 2>/dev/null; touch $WD_STOP; for c in cpu6 cpu7; do cat /sys/devices/system/cpu/\$c/cpufreq/cpuinfo_max_freq > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done'" < /dev/null; }
start_wd(){ adb_safe_shell "su -c 'rm -f $WD_STOP; nohup sh /data/local/tmp/preempt_throttle_watchdog_v2.sh /data/local/tmp/cpu_wd_$1.log $WD_STOP >/dev/null 2>&1 &'" < /dev/null; }
stop_wd(){ adb_safe_shell "su -c 'touch $WD_STOP'" < /dev/null; }
ZMAP=$(adb_safe_shell "su -c 'for z in /sys/class/thermal/thermal_zone*; do printf \"%s:%s \" \$(basename \$z|sed s/thermal_zone//) \$(cat \$z/type 2>/dev/null); done'" < /dev/null|tr -d '\r')
z_by_name(){ echo "$ZMAP" | tr ' ' '\n' | grep -E ":$1\$" | head -1 | cut -d: -f1; }
DDR_Z=$(z_by_name ddr); SHELL_Z=$(z_by_name shell_front)
CPU_ZS=$(echo "$ZMAP" | tr ' ' '\n' | grep -E ':(cpu-[0-9]|cpullc-[0-9])' | cut -d: -f1 | tr '\n' ' ')
echo "  [zones] cpu=($CPU_ZS) ddr=$DDR_Z shell=$SHELL_Z"
coolidle(){ local T0=$(date +%s)
  while true; do
    read cpu ddr sh <<<"$(adb_safe_shell "su -c 'm=0; for z in $CPU_ZS; do t=\$(cat /sys/class/thermal/thermal_zone\$z/temp 2>/dev/null); [ \$t -gt \$m ]&&m=\$t; done; d=\$(cat /sys/class/thermal/thermal_zone${DDR_Z}/temp 2>/dev/null); s=\$(cat /sys/class/thermal/thermal_zone${SHELL_Z}/temp 2>/dev/null); echo \$((m/1000)) \$((d/1000)) \$((s/1000))'" < /dev/null|tr -d '\r')"
    cpu=${cpu:-99}; ddr=${ddr:-99}; sh=${sh:-99}
    if [ "$cpu" -lt 37 ] && [ "$ddr" -lt 37 ] && [ "$sh" -lt 34 ]; then echo "  [cold cpu=$cpu ddr=$ddr shell=$sh]"; return; fi
    [ $(($(date +%s)-T0)) -gt 1800 ] && { echo "  [cool timeout cpu=$cpu ddr=$ddr shell=$sh]"; return; }
    sleep 10
  done; }

run(){ local CELL=$1 WD=$2; shift 2; local PD=$OUT/$CELL
  # Resume: skip cells whose ppl.json, written last, exists.
  [ -f "$OUT_HOST/$CELL/ppl.json" ] && { echo "[$(date +%H:%M:%S)] skip $CELL (already complete)"; return; }
  echo "[$(date +%H:%M:%S)] $CELL (wd=$WD)"; platform_natural; coolidle
  adb_safe_shell "mkdir -p $PD" < /dev/null
  [ "$WD" = 1 ] && start_wd "$CELL"
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  sleep 2; local t0=$(date +%s%N)
  # Run gen through a nohup device script so a tunnel drop does not kill it, then poll for DONE.
  cat > /tmp/rest2_bench.sh <<EOF
#!/system/bin/sh
rm -f $PD/gen.DONE
LD_LIBRARY_PATH=$CB $CB/eviction_bench --prompt $OUT/prompt.txt --prompt-id ${CELL}_gen --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 6 --n-gpu-layers 0 --greedy --k-nominal 1024 $* --out-meta $PD/gen.json --out-csv $PD/gen_steps.csv --out-prefill-csv $PD/gen_prefill.csv --out-gen /dev/null > $PD/gen.out 2> $PD/gen.err
touch $PD/gen.DONE
EOF
  adb push /tmp/rest2_bench.sh /data/local/tmp/rest2_bench.sh < /dev/null >/dev/null 2>&1
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/rest2_bench.sh >/dev/null 2>&1 &'" < /dev/null
  while :; do dn=$(adb_safe_shell "[ -f $PD/gen.DONE ] && echo Y || echo N" < /dev/null | tr -d '\r'); case "$dn" in *Y*) break;; esac; sleep 60; done
  local t1=$(date +%s%N)
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  mkdir -p "$OUT_HOST/$CELL"; echo "$(( (t1-t0)/1000000 ))" > "$OUT_HOST/$CELL/wall_ms"
  # PPL, same pattern
  cat > /tmp/rest2_ppl.sh <<EOF
#!/system/bin/sh
rm -f $PD/ppl.DONE
LD_LIBRARY_PATH=$CB $CB/eviction_bench --prompt $OUT/prompt.txt --prompt-id ${CELL}_ppl --eval-mode ppl --eval-text $OUT/eval.txt --max-tokens 512 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 6 --n-gpu-layers 0 --greedy --k-nominal 1024 $* --out-meta $PD/ppl.json --out-csv /dev/null --out-gen /dev/null > $PD/ppl.out 2> $PD/ppl.err
touch $PD/ppl.DONE
EOF
  adb push /tmp/rest2_ppl.sh /data/local/tmp/rest2_ppl.sh < /dev/null >/dev/null 2>&1
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/rest2_ppl.sh >/dev/null 2>&1 &'" < /dev/null
  while :; do dn=$(adb_safe_shell "[ -f $PD/ppl.DONE ] && echo Y || echo N" < /dev/null | tr -d '\r'); case "$dn" in *Y*) break;; esac; sleep 30; done
  [ "$WD" = 1 ] && { stop_wd; adb pull /data/local/tmp/cpu_wd_$CELL.log "$OUT_HOST/${CELL}_wd.log" < /dev/null >/dev/null 2>&1; }
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  echo "  [done $CELL] gen=$(grep -oE 'decode_tps=[0-9.]+' "$OUT_HOST/$CELL/gen.err" 2>/dev/null|head -1) ppl=$(grep -oiE 'ppl[= ][0-9.]+|perplexity[= :]+[0-9.]+' "$OUT_HOST/$CELL/ppl.err" 2>/dev/null|head -1)"
}
MU="--policy v1_fa2 --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
# One build, natural DVFS
run mukv_faon     1 $MU --fa-on-evict   # muKV-mass-full (Solution 2 fa-on-evict + CPU defrag), NEW watchdog active
run adakv         0 --policy adakv --n-sink 0 --obs-window 32
run streamingllm  0 --policy streamingllm --n-sink 4
run h2o           0 --policy h2o --n-sink 0 --obs-window 64
run tova          0 --policy tova
adb_safe_shell "su -c 'touch $WD_STOP; for c in cpu6 cpu7; do cat /sys/devices/system/cpu/\$c/cpufreq/cpuinfo_max_freq > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
touch /tmp/nat_bonsai_rest2_DONE; echo "[$(date +%H:%M:%S)] NATURAL BONSAI DONE -> $OUT_HOST"
