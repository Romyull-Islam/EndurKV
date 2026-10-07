#!/bin/bash
# Energy-aware controller test, n = 3 per arm, replicates interleaved.
# GPU arms: the controller caps the GPU clock (1200 / 902 / 726 MHz) at K = 1024.
# CPU arms: it shrinks the cache (K = 1024 / 512 / 256) at fixed clocks.
# The phone sits at 100% SoC, so battery levels are simulated with the thresholds
# (50/20 gives level 0, 100/20 level 1, 100/100 level 2). A cell is discarded if the
# controller picked another level or, on GPU, the clock write failed.
# CPU caps are fixed (prime 1497.6 MHz, rest 1785.6 MHz) and restored on exit.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN_GPU=/data/local/tmp/ukv                     # Vulkan build: GPU cells
BIN_CPU=/data/local/tmp/endurkv/bin_cpu_ea      # CPU-only build: CPU cells. The Vulkan build with
                                                # --n-gpu-layers 0 still opens the Adreno device,
                                                # reserves a 253 MiB compute buffer on it and decodes
                                                # at 6 tok/s instead of 23.
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
  # Pin GPU cells only. The pin removes GPU dispatch bimodality, but on CPU it cuts
  # decode from 26.7 to 6.2 tok/s and doubles prefill time.
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
  echo "replicate $R: GPU, identical command line, only the battery state differs (controller caps the clock)"
  cell gpu_healthy_r$R 99 "--ea-soc-hi 50  --ea-soc-lo 20"  4096 0
  cell gpu_mid_r$R     99 "--ea-soc-hi 100 --ea-soc-lo 20"  4096 1
  cell gpu_low_r$R     99 "--ea-soc-hi 100 --ea-soc-lo 100" 4096 2
  echo "replicate $R: CPU, same three battery states (controller shrinks the cache)"
  cell cpu_healthy_r$R  0 "--ea-soc-hi 50  --ea-soc-lo 20"  1024 0
  cell cpu_mid_r$R      0 "--ea-soc-hi 100 --ea-soc-lo 20"  1024 1
  cell cpu_low_r$R      0 "--ea-soc-hi 100 --ea-soc-lo 100" 1024 2
done
python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/ea_proof_summary.py "$HOST"
echo EAPROOF2_DONE
