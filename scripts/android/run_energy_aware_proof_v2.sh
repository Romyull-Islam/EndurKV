#!/bin/bash
# ============================================================================
# run_energy_aware_proof_v2.sh -- does the energy-aware controller save energy when
# the battery is low, under a matched protocol? (2026-09-02)
#
# THE CONTROLLER UNDER TEST acts per backend, because the measurements say the
# energy lever differs:
#     GPU : cap the GPU clock (1200 / 902 / 726 MHz), HOLD K = 1024.
#           The cache tier moved GPU energy per token <7% in three campaigns and
#           costs 6.6 F1 on qasper; the clock cut 335->281 mJ/token at K=1024 with
#           throughput unchanged (30.0 vs 30.3 tok/s).
#     CPU : shrink the cache (K = 1024 / 512 / 256), leave the clock alone.
#           Decode is attention-bound over live cells; the clock is not a lever there.
# The controller writes the GPU clock itself (su) and restores it on exit. The
# sampler records gpu_clk_hz every sample, so the action is visible in the data.
#
# SAME COMMAND LINE EVERY ARM. Only the battery state the controller sees differs.
# The phone sits at 100% SoC, so a battery level is simulated by moving the
# thresholds (the controller still reads the real battery every time):
#     healthy : thresholds  50/20  -> 100 is above both       -> level 0
#     mid     : thresholds 100/20  -> 100 is not above 100    -> level 1
#     low     : thresholds 100/100 -> 100 is above neither    -> level 2
# Charging is disabled by the cool gate, so the status is not "mains" and the
# SoC path is exercised. Each cell asserts the level the controller picked and,
# on GPU, that the clock write succeeded; otherwise the cell is discarded.
#
# MATCHED CONDITIONS:
#   * CPU caps: identical for every cell (prime 1497.6 MHz, rest 1785.6 MHz,
#     min = max), the watchdog's healthy rung, sustained without throttling in
#     the CPU soak. Pre-run caps are saved and restored on exit.
#   * GPU: the controller sets the clock; the script restores 1200 MHz and
#     max_pwrlevel 0 on exit so nothing leaks into later campaigns.
#   * Scheduler: `taskset f0 nice -n -20` (big cores, high priority) for the
#     bench, which removed the 39% dispatch bimodality.
#   * Temperature gate: cool_ddr36 plus a settle loop until DDR <= 35 C and
#     battery <= 33 C before every cell; start temperatures and the caps in
#     force are written to start_temps.txt in each cell.
#   * Energy = USB rail integral + battery pack coulomb delta. Replicates are
#     interleaved (r1 of every arm, then r2, then r3), n = 3 per arm, 18 cells.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN_GPU=/data/local/tmp/ukv                     # Vulkan build: GPU cells
BIN_CPU=/data/local/tmp/endurkv/bin_cpu_ea      # CPU-only build: CPU cells. The Vulkan build with
                                                # --n-gpu-layers 0 still opens the Adreno device,
                                                # reserves a 253 MiB compute buffer on it and decodes
                                                # at 6 tok/s instead of 23 (found 2026-09-03).
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
DEV=/data/local/tmp/endurkv/logs/eaproof2_$(date +%Y%m%d_%H%M%S)
HOST=${HOST:-/tmp/ea_proof_v2}; mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024 --energy-aware --ea-k 1024 512 256 --ea-gpu-mhz 1200 902 726"
PIN="taskset f0 nice -n -20"
CPU_PRIME=1497600   # policy6 (cpu6-7)
CPU_REST=1785600    # policy0 (cpu0-5)
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1

SAVED=$(adb_safe_shell "su -c 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo \$(basename \$p) \$(cat \$p/scaling_min_freq) \$(cat \$p/scaling_max_freq); done'" < /dev/null | tr -d '\r')
echo "pre-run CPU caps:"; echo "$SAVED" | sed 's/^/    /'; echo "$SAVED" > $HOST/pre_run_cpu_caps.txt

restore_all(){
  echo "$SAVED" | while read pol mn mx; do
    [ -n "${pol:-}" ] && adb_safe_shell "su -c 'echo $mx > /sys/devices/system/cpu/cpufreq/$pol/scaling_max_freq; echo $mn > /sys/devices/system/cpu/cpufreq/$pol/scaling_min_freq'" < /dev/null
  done
  adb_safe_shell "su -c 'echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel'" < /dev/null
}
pin_cpu(){
  adb_safe_shell "su -c 'echo $CPU_REST > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; echo $CPU_REST > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; echo $CPU_PRIME > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq; echo $CPU_PRIME > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq'" < /dev/null
}
gpu_default(){ adb_safe_shell "su -c 'echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel'" < /dev/null; }
cleanup(){
  adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
  restore_all
}
trap cleanup EXIT INT TERM

settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    local _rd   # local: a global here clobbered the replicate loop variable R (2026-09-03 bug)
    _rd=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $_rd|awk '{print int($1/1000)}'); local _sb=$(echo $_rd|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"
    START_TEMPS="ddr_c=${_sd} batt_c=${_sb}"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1; }

cell(){ # tag  ngl  thresholds  maxtok  expected_level
  local TAG=$1 NGL=$2 TH=$3 TOK=$4 EXP=$5
  local D=$HOST/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  pin_cpu; gpu_default        # the controller lowers the GPU clock itself when the arm calls for it
  local CAPS; CAPS=$(adb_safe_shell "su -c 'echo cpu0=\$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq) cpu6=\$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq) gpu_max=\$(cat /sys/class/kgsl/kgsl-3d0/max_gpuclk) gpu_tpl=\$(cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel)'" < /dev/null | tr -d '\r')
  echo "$START_TEMPS $CAPS" > "$D/start_temps.txt"; echo "    start: $START_TEMPS $CAPS"
  adb_safe_shell "su -c 'rm -f /data/local/tmp/ukv_ea_level /data/local/tmp/ea_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/ea_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG ..."
  # taskset f0 + nice -20 fixes the GPU dispatch bimodality but cuts CPU decode from 26.7 to
  # 6.2 tok/s and doubles CPU prefill time (measured 2026-09-03, same binary, same caps).
  # So the pin applies to GPU cells only; CPU cells run unpinned like the soak did.
  local BIN CELLPIN; if [ "$NGL" -gt 0 ]; then BIN=$BIN_GPU; CELLPIN=$PIN; else BIN=$BIN_CPU; CELLPIN=""; fi
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN $CELLPIN ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG --eval-mode gen --max-tokens $TOK --ignore-eos --ctx-size 16384 \
    --seed 42 --threads 4 --n-gpu-layers $NGL --greedy --cache-type-k f16 --cache-type-v f16 \
    $MU $TH --n-batch 512 --n-ubatch 64 --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen \
    --out-csv /dev/null > /dev/null 2> $DEV/$TAG.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null
  gpu_default
  adb_safe_pull "$DEV/$TAG.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.gen"  "$D/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.err"  "$D/err"       >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/ea_$TAG.csv" "$D/sensors.csv" >/dev/null 2>&1
  local EAL; EAL=$(grep -m1 'energy-aware\] soc' "$D/err" 2>/dev/null); echo "    $EAL"
  local GOT; GOT=$(echo "$EAL" | sed -n 's/.*-> level=\([0-9]\).*/\1/p')
  if [ "${GOT:-x}" != "$EXP" ]; then
    echo "  [$TAG] LEVEL MISMATCH: wanted $EXP got ${GOT:-none}; battery state drifted. Cell discarded."
    mv "$D" "$D.badlevel_$(date +%H%M%S)"; return
  fi
  if [ "$NGL" -gt 0 ] && ! echo "$EAL" | grep -q "write_rc=0"; then
    echo "  [$TAG] GPU CLOCK WRITE FAILED; cell discarded."
    mv "$D" "$D.badclock_$(date +%H%M%S)"; return
  fi
  [ -f "$D/meta.json" ] && python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/clock_cell_report.py "$D" "$TAG" 2>/dev/null || echo "  [$TAG] FAILED"
}

for R in 1 2 3; do
  echo "=== replicate $R: GPU, identical command line, only the battery state differs (controller caps the clock) ==="
  cell gpu_healthy_r$R 99 "--ea-soc-hi 50  --ea-soc-lo 20"  4096 0
  cell gpu_mid_r$R     99 "--ea-soc-hi 100 --ea-soc-lo 20"  4096 1
  cell gpu_low_r$R     99 "--ea-soc-hi 100 --ea-soc-lo 100" 4096 2
  echo "=== replicate $R: CPU, same three battery states (controller shrinks the cache) ==="
  cell cpu_healthy_r$R  0 "--ea-soc-hi 50  --ea-soc-lo 20"  1024 0
  cell cpu_mid_r$R      0 "--ea-soc-hi 100 --ea-soc-lo 20"  1024 1
  cell cpu_low_r$R      0 "--ea-soc-hi 100 --ea-soc-lo 100" 1024 2
done
python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/ea_proof_summary.py "$HOST"
echo EAPROOF2_DONE
