#!/bin/bash
# ============================================================================
# run_cpu_cache_vs_clock.sh -- CPU proof of the two levers. (2026-08-11)
#
# THE CLAIM UNDER TEST, on CPU this time:
#   CACHE lever  -- shrinking K cuts DRAM traffic per token, so it cuts POWER without
#                   costing time. Measured on the GPU: 6.38 -> 5.58 -> 4.75 W (-26%).
#   CLOCK lever  -- capping frequency cuts power too, but buys it all back in runtime,
#                   so energy per request barely moves. Predicted from this phone's own
#                   CPU power law: dynamic power goes as f^1.10 (voltage is pinned at
#                   Vmin, so the textbook V^2 term never scales), hence E = P*t ~ f^0.10.
#                   A 40% clock cut should save ~5% energy and cost ~67% more time.
# Both are asserted from GPU data and a CPU power-law fit; neither has been measured
# end-to-end on the CPU. This measures both, on the same device, same model, same prompt.
#
# ARMS
#   A. cache: vanilla, then muKV at k-pct 20 / 10 / 5, CPU governor untouched.
#   B. clock: muKV held at k-pct 20, prime cores capped to a descending ladder.
# Only one variable moves per arm, so the two levers can be compared directly.
#
# ENERGY = USB rail + battery pack (rail alone undercounts 4-36%; the pack silently
# supplements it whenever SoC draw exceeds what USB delivers).
# 1024 generated tokens, not 4096: CPU decode is slow enough that 4096 would make each
# cell ~30 min. Rates (tok/s, W, mJ/token) are unaffected by the token count.
# ============================================================================
set -u
# FIXED 2026-08-11: use adb_safe_pull, never bare `adb pull`. adb_resilient.sh exports
# ANDROID_ADB_SERVER_PORT after probing for the device, which conflicts with an
# ADB_SERVER_SOCKET set by the caller -- so adb_safe_shell reached the phone and ran the
# benchmark while every bare `adb pull` silently retrieved nothing. The first cell of the
# tier sweep looked FAILED for exactly this reason although the run had completed on-device
# (decode_tps=33.0, meta.json present). adb_safe_pull uses the resolved port and retries.
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
DEV=/data/local/tmp/endurkv/logs/cpu_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/cpu_levers; mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace"
BIG="6 7"; LIT="0 4"
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/benchmarks/ppl/wiki_eval_disjoint.txt /data/local/tmp/endurkv/logs/eval_disjoint.txt < /dev/null >/dev/null 2>&1
restore_freq(){ adb_safe_shell "su -c 'for c in $BIG $LIT; do f=/sys/devices/system/cpu/cpu\$c/cpufreq; [ -d \$f ] && cat \$f/cpuinfo_max_freq > \$f/scaling_max_freq 2>/dev/null; done'" < /dev/null; }
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; restore_freq; }
trap cleanup EXIT INT TERM

settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    R=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $R|awk '{print int($1/1000)}'); local _sb=$(echo $R|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1; }

cell(){ # tag  flags  freq_khz(0 = leave governor alone)  [mode: gen|ppl]
  local TAG=$1 FLAGS=$2 FQ=$3 MODE=${4:-gen}
  local D=$HOST/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  if [ "$FQ" != "0" ]; then
    adb_safe_shell "su -c 'for c in $BIG; do echo $FQ > /sys/devices/system/cpu/cpu\$c/cpufreq/scaling_max_freq 2>/dev/null; done; cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq'" < /dev/null | tail -1 | sed 's/^/    prime cap = /'
  else
    restore_freq; echo "    prime cap = governor default"
  fi
  local EX="--eval-mode gen --max-tokens 1024 --ignore-eos"
  [ "$MODE" = ppl ] && EX="--eval-mode ppl --eval-text /data/local/tmp/endurkv/logs/eval_disjoint.txt"
  adb_safe_shell "su -c 'rm -f /data/local/tmp/cl_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/cl_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG (CPU) ..."
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG $EX --ctx-size 16384 \
    --seed 42 --threads 4 --n-gpu-layers 0 --greedy --cache-type-k f16 --cache-type-v f16 \
    $FLAGS --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null \
    > /dev/null 2> $DEV/$TAG.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null; restore_freq
  adb_safe_pull $DEV/$TAG.json "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull $DEV/$TAG.gen "$D/gen.txt" >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/cl_$TAG.csv" "$D/sensors.csv" >/dev/null 2>&1
  python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/clock_cell_report.py "$D" "$TAG" 2>/dev/null || echo "  [$TAG] FAILED"
}

echo "=== ARM A: CACHE lever (governor untouched) ==="
cell cpu_vanilla "--policy vanilla" 0
cell cpu_pct20   "$MU --k-pct 20"   0
cell cpu_pct10   "$MU --k-pct 10"   0
cell cpu_pct5    "$MU --k-pct 5"    0
echo "=== ARM B: CLOCK lever (muKV fixed at k-pct 20) ==="
cell cpu_clk_full "$MU --k-pct 20" 0
cell cpu_clk_2438 "$MU --k-pct 20" 2438400
cell cpu_clk_1996 "$MU --k-pct 20" 1996800
cell cpu_clk_1500 "$MU --k-pct 20" 1500000
echo "=== ARM C: QUALITY per cache tier (CPU, teacher-forced on the disjoint slice) ==="
cell cpu_ppl_vanilla "--policy vanilla" 0 ppl
cell cpu_ppl_pct20   "$MU --k-pct 20"   0 ppl
cell cpu_ppl_pct10   "$MU --k-pct 10"   0 ppl
cell cpu_ppl_pct5    "$MU --k-pct 5"    0 ppl
echo CPU_LEVERS_DONE
