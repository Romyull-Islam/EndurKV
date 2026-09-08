#!/bin/bash
# ============================================================================
# run_split_clock_proof.sh -- measure the phase-aware plans the lever scheduler predicts
# (2026-09-03). Prefill at 1200 MHz, decode capped at 902 or 726 MHz, K=1024, 4096 output
# tokens, the 9737-token prompt, n=3 each, same pinned protocol as run_energy_aware_proof_v2
# (CPU caps pinned, cool gate, taskset f0 nice -20, sampler v6 recording the GPU clock).
# The controller is not used: the clock caps are set explicitly (prefill by the script,
# decode by --gpu-mhz-decode in the engine), so the cells are the plans' true costs.
# Compare with /tmp/ea_proof_v2 gpu_healthy (1200/1200), gpu_mid (902/902), gpu_low (726/726).
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
DEV=/data/local/tmp/endurkv/logs/split_$(date +%Y%m%d_%H%M%S)
HOST=${HOST:-/tmp/split_proof}; mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024"
PIN="taskset f0 nice -n -20"
CPU_PRIME=1497600; CPU_REST=1785600
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
SAVED=$(adb_safe_shell "su -c 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo \$(basename \$p) \$(cat \$p/scaling_min_freq) \$(cat \$p/scaling_max_freq); done'" < /dev/null | tr -d '\r')
echo "$SAVED" > $HOST/pre_run_cpu_caps.txt
restore_all(){
  echo "$SAVED" | while read pol mn mx; do
    [ -n "${pol:-}" ] && adb_safe_shell "su -c 'echo $mx > /sys/devices/system/cpu/cpufreq/$pol/scaling_max_freq; echo $mn > /sys/devices/system/cpu/cpufreq/$pol/scaling_min_freq'" < /dev/null
  done
  adb_safe_shell "su -c 'echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel'" < /dev/null
}
pin_cpu(){ adb_safe_shell "su -c 'echo $CPU_REST > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; echo $CPU_REST > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; echo $CPU_PRIME > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq; echo $CPU_PRIME > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq'" < /dev/null; }
cleanup(){ adb_safe_shell "su -c 'pkill -f eviction_bench; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null; restore_all; }
trap cleanup EXIT INT TERM

settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    local _rd; _rd=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $_rd|awk '{print int($1/1000)}'); local _sb=$(echo $_rd|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"; START_TEMPS="ddr_c=${_sd} batt_c=${_sb}"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1; }

cell(){ # tag  prefill_mhz  decode_mhz
  local TAG=$1 PMHZ=$2 DMHZ=$3
  local D=$HOST/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  pin_cpu
  adb_safe_shell "su -c 'echo $((PMHZ*1000000)) > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel'" < /dev/null
  echo "$START_TEMPS prefill_mhz=$PMHZ decode_mhz=$DMHZ" > "$D/start_temps.txt"
  adb_safe_shell "su -c 'rm -f /data/local/tmp/split_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/split_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG (prefill $PMHZ MHz, decode $DMHZ MHz) ..."
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN $PIN ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 \
    --seed 42 --threads 4 --n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 \
    $MU --gpu-mhz-decode $DMHZ --n-batch 512 --n-ubatch 64 --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen \
    --out-csv /dev/null > /dev/null 2> $DEV/$TAG.err" < /dev/null
  adb_safe_shell "su -c 'echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk'" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null
  adb_safe_pull "$DEV/$TAG.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.err"  "$D/err"       >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/split_$TAG.csv" "$D/sensors.csv" >/dev/null 2>&1
  adb_safe_shell "grep -m1 'phase-clock' $DEV/$TAG.err" < /dev/null | sed 's/^/    /'
  [ -f "$D/meta.json" ] && python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/clock_cell_report.py "$D" "$TAG" 2>/dev/null || echo "  [$TAG] FAILED"
}

for R in 1 2 3; do
  echo "=== replicate $R ==="
  cell gpu1200d902_r$R 1200 902
  cell gpu1200d726_r$R 1200 726
done
echo SPLITPROOF_DONE
