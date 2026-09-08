#!/bin/bash
# ============================================================================
# run_clock_sweetspot.sh -- find the GPU clock cap that sheds heat without
# costing throughput. (2026-08-11, OnePlus 15 / Adreno 840)
#
# WHY THERE MIGHT BE A SWEET SPOT AT ALL. On the CPU, measured dynamic power on this phone
# follows P ~ f^1.10 (voltage is pinned at Vmin, so the textbook V^2 term never scales).
# Since time ~ 1/f for compute-bound work, energy goes as f^0.10 -- capping the clock 40%
# saves 4.7% energy and costs 67% more time. That makes clock capping a poor ENERGY lever
# for compute-bound phases.
#
# But muKV's DECODE is BANDWIDTH-bound, not compute-bound: each token reads weights + KV
# from DRAM, and that traffic does not get faster with a higher shader clock. If decode is
# genuinely memory-limited, its throughput should be nearly FLAT in GPU clock over some
# range, while power and temperature keep falling with it. That range is the sweet spot,
# and it cannot be predicted from the CPU exponent -- it has to be measured.
#
# WHAT IS SWEPT. gpu_max_clock over the Adreno's authorized rungs, muKV held fixed at its
# frozen config so the only variable is the clock. Reported per cap: decode tok/s, peak
# skin/DDR/battery, mean power, and energy per token (rail + PACK -- rail alone undercounts
# by 4-36% because the battery silently supplements it).
#
# The knee is where d(tok/s)/d(clock) is still ~0 but temperature has already dropped.
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
DEV=/data/local/tmp/endurkv/logs/clk_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/clock_sweep; mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --k-nominal 1024 --compact-inplace"
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1200 > /sys/kernel/gpu/gpu_max_clock; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
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

cell(){ local CLK=$1
  local D=$HOST/clk_$CLK; [ -f "$D/meta.json" ] && { echo "  [clk $CLK] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for clk=$CLK ..."; settle || { echo "  [SKIP-HOT] $CLK"; return; }
  # pin BOTH ends so the governor cannot wander: max=cap, min stays at the floor
  adb_safe_shell "su -c 'echo $CLK > /sys/kernel/gpu/gpu_max_clock; cat /sys/kernel/gpu/gpu_max_clock'" < /dev/null | tail -1 | sed 's/^/    cap set to /'
  adb_safe_shell "su -c 'rm -f /data/local/tmp/sc_$CLK.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/sc_$CLK.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running clk=$CLK ..."
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN ./eviction_bench --model $M --prompt $P \
    --prompt-id clk$CLK --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 \
    --seed 42 --threads 4 --n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 \
    $MU --n-batch 512 --n-ubatch 64 --out-meta $DEV/clk$CLK.json --out-gen $DEV/clk$CLK.gen \
    --out-csv /dev/null > /dev/null 2> $DEV/clk$CLK.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1200 > /sys/kernel/gpu/gpu_max_clock'" < /dev/null
  adb_safe_pull $DEV/clk$CLK.json "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull $DEV/clk$CLK.gen "$D/gen.txt" >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/sc_$CLK.csv" "$D/sensors.csv" >/dev/null 2>&1
  python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/clock_cell_report.py "$D" "$CLK" 2>/dev/null || echo "  [clk $CLK] FAILED"
}
for C in 1200 1050 902 826 726; do cell $C; done
echo CLKSWEEP_DONE
