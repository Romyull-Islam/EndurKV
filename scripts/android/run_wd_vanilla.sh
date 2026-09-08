#!/bin/bash
# ============================================================================
# WATCHDOG PREEMPTION TEST (2026-07-19). Vanilla Bonsai-8B + PREEMPTIVE watchdog v3
# (ladders anchored at the MEASURED battery-50C vendor throttle).
#
# WHY vanilla: it reliably drives the battery to 50C and the vendor deep-throttles
# BOTH clusters to ~883 MHz (see /tmp/nat_bonsai/vanilla: 85.6min, 883 sawtooth 40-86min).
# This runs the SAME vanilla workload (4096 decode) WITH the new watchdog, whose 47-49C
# glide (1497->1017 MHz, below the vendor's flat 1498) is designed to bend the temp curve
# and keep the battery UNDER 50C so the 883 cliff never triggers.
# COMPARE against the no-watchdog baseline /tmp/nat_bonsai/vanilla.
# ============================================================================
set -u; export ANDROID_ADB_SERVER_PORT=5151
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
OUT_HOST=/tmp/wd_vanilla; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/wdvanilla_$(date +%s 2>/dev/null || echo run)
adb_safe_shell "mkdir -p $OUT" < /dev/null
SCR=/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad
adb push "$SCR/wikitext_16k_p12k_d4k.txt" "$OUT/prompt.txt" < /dev/null >/dev/null 2>&1
# push the NEW (reframed) watchdog
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/preempt_throttle_watchdog_v2.sh \
         /data/local/tmp/preempt_throttle_watchdog_v2.sh < /dev/null >/dev/null 2>&1
MODEL=/data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf
CB=/data/local/tmp/endurkv/bin_cpu_v87
WD_STOP=/data/local/tmp/cpu_wd.stop

# SKIP charge: phone is already cold (battery ~32C = matched to the no-wd baseline cold
# start); charging would heat the battery and break that match. Just disable charging so
# battery temp reflects load, not charging.
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
echo "  [skip charge; matched cold start, SoC $(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/capacity'" < /dev/null|tr -d '\r')%]"

# zone resolution (same as the systems campaign)
ZMAP=$(adb_safe_shell "su -c 'for z in /sys/class/thermal/thermal_zone*; do printf \"%s:%s \" \$(basename \$z|sed s/thermal_zone//) \$(cat \$z/type 2>/dev/null); done'" < /dev/null|tr -d '\r')
z_by_name(){ echo "$ZMAP" | tr ' ' '\n' | grep -E ":$1\$" | head -1 | cut -d: -f1; }
DDR_Z=$(z_by_name ddr); SHELL_Z=$(z_by_name shell_front)
CPU_ZS=$(echo "$ZMAP" | tr ' ' '\n' | grep -E ':(cpu-[0-9]|cpullc-[0-9])' | cut -d: -f1 | tr '\n' ' ')
echo "  [zones] cpu=($CPU_ZS) ddr=$DDR_Z shell=$SHELL_Z"

# restore vendor max first (undo any leftover cap), watchdog off
adb_safe_shell "su -c 'touch $WD_STOP; for c in cpu0 cpu6; do cat /sys/devices/system/cpu/\$c/cpufreq/cpuinfo_max_freq > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done'" < /dev/null

# COLD gate (match systems baseline: cpu<37 ddr<37 shell<34)
echo "[$(date +%H:%M:%S)] cooling to cpu<37/ddr<37/shell<34 ..."
T0=$(date +%s)
while true; do
  read cpu ddr sh <<<"$(adb_safe_shell "su -c 'm=0; for z in $CPU_ZS; do t=\$(cat /sys/class/thermal/thermal_zone\$z/temp 2>/dev/null); [ \$t -gt \$m ]&&m=\$t; done; d=\$(cat /sys/class/thermal/thermal_zone${DDR_Z}/temp 2>/dev/null); s=\$(cat /sys/class/thermal/thermal_zone${SHELL_Z}/temp 2>/dev/null); echo \$((m/1000)) \$((d/1000)) \$((s/1000))'" < /dev/null|tr -d '\r')"
  cpu=${cpu:-99}; ddr=${ddr:-99}; sh=${sh:-99}
  { [ "$cpu" -lt 37 ] && [ "$ddr" -lt 37 ] && [ "$sh" -lt 34 ]; } && { echo "  [cold cpu=$cpu ddr=$ddr shell=$sh]"; break; }
  [ $(($(date +%s)-T0)) -gt 1800 ] && { echo "  [cool timeout cpu=$cpu ddr=$ddr shell=$sh]"; break; }
  sleep 10
done

# start the NEW watchdog (writes its own tier log) + full sensor sampling (matches baseline)
adb_safe_shell "su -c 'rm -f $WD_STOP; nohup sh /data/local/tmp/preempt_throttle_watchdog_v2.sh /data/local/tmp/cpu_wd_vanilla.log $WD_STOP >/dev/null 2>&1 &'" < /dev/null
adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $OUT/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
sleep 2

echo "[$(date +%H:%M:%S)] vanilla Bonsai + PREEMPTIVE watchdog (4096 decode) -- nohup on-device (tunnel-resilient)"
# Write a device-side runner so the bench is DETACHED and survives a tunnel blip.
BENCH_SH=/tmp/wdvan_bench.sh
cat > "$BENCH_SH" <<EOF
#!/system/bin/sh
rm -f $OUT/gen.DONE
LD_LIBRARY_PATH=$CB $CB/eviction_bench --prompt $OUT/prompt.txt --prompt-id wdvanilla_gen --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 6 --n-gpu-layers 0 --greedy --k-nominal 1024 --policy vanilla --out-meta $OUT/gen.json --out-csv $OUT/gen_steps.csv --out-gen /dev/null > $OUT/gen.out 2> $OUT/gen.err
touch $OUT/gen.DONE
EOF
adb push "$BENCH_SH" /data/local/tmp/wdvan_bench.sh < /dev/null >/dev/null 2>&1
adb_safe_shell "su -c 'nohup sh /data/local/tmp/wdvan_bench.sh >/dev/null 2>&1 &'" < /dev/null
echo "[$(date +%H:%M:%S)] bench launched detached; polling on-device gen.DONE (survives tunnel drops) ..."
while true; do
  dn=$(adb_safe_shell "[ -f $OUT/gen.DONE ] && echo Y || echo N" < /dev/null | tr -d '\r')
  case "$dn" in *Y*) echo "[$(date +%H:%M:%S)] bench complete"; break;; esac
  sleep 60
done

# stop watchdog + sensors, restore, pull
adb_safe_shell "su -c 'touch $WD_STOP; pkill -f sample_sensors 2>/dev/null; sleep 2; for c in cpu0 cpu6; do cat /sys/devices/system/cpu/\$c/cpufreq/cpuinfo_max_freq > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
adb pull "$OUT/sensors.csv"   "$OUT_HOST/sensors.csv"   < /dev/null >/dev/null 2>&1
adb pull "$OUT/gen.json"      "$OUT_HOST/gen.json"      < /dev/null >/dev/null 2>&1
adb pull "$OUT/gen_steps.csv" "$OUT_HOST/gen_steps.csv" < /dev/null >/dev/null 2>&1
adb pull "$OUT/gen.err"       "$OUT_HOST/gen.err"       < /dev/null >/dev/null 2>&1
adb pull /data/local/tmp/cpu_wd_vanilla.log "$OUT_HOST/cpu_wd_vanilla.log" < /dev/null >/dev/null 2>&1
touch /tmp/wd_vanilla_DONE
echo "[$(date +%H:%M:%S)] WD-VANILLA DONE -> $OUT_HOST"
echo "--- watchdog tier transitions ---"; grep -iE 'PERF=|PREEMPTIVE|BATTERY|SKIN' "$OUT_HOST/cpu_wd_vanilla.log" 2>/dev/null | tail -15