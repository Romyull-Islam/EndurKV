#!/bin/bash
# ============================================================================
# run_energy_aware_proof.sh -- does the energy-aware controller actually work?
# (2026-08-11)
#
# THE QUESTION IS NOT "does a smaller cache use less energy" -- that is already measured.
# It is "does muKV READ THE BATTERY AND ADAPT, and does the adaptation show up in joules".
# So each arm runs the IDENTICAL command line; the only thing that differs is the battery
# state the controller sees. Nothing else is passed. If the arms differ in K, in energy,
# and in throughput, the loop is real and closed.
#
# HOW A BATTERY LEVEL IS SIMULATED WITHOUT DRAINING THE PHONE. The controller compares the
# measured state of charge against two thresholds. Moving the thresholds is exactly
# equivalent to moving the SoC across them, and it is the only way to exercise the low-SoC
# path on a phone sitting at 98%. The controller still READS the real battery every time --
# the reported soc is genuine -- only the tier boundaries move.
#     healthy  : thresholds 50/20  -> real SoC 98% is above both  -> level 0
#     mid      : thresholds 99/20  -> real SoC 98% falls between  -> level 1
#     low      : thresholds 99/99  -> real SoC 98% is below both  -> level 2
#
# BACKEND-AWARE, so both are measured: on GPU the tiers descend 20 -> 10 -> 5 because the
# GPU optimum is 20% and going lower costs throughput; on CPU all tiers sit at 5% because
# the CPU optimum IS the floor (24.02 tok/s at k-pct 5, monotonically better than 20%).
# The CPU arms should therefore show NO adaptation -- that is the correct behaviour, and
# showing it is the point: an energy-aware controller that degrades a backend which gains
# nothing from degrading would be a bug, not a feature.
#
# Cool gate + settle before every cell; energy = USB rail + coulomb counter.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
DEV=/data/local/tmp/endurkv/logs/eaproof_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/ea_proof; mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --energy-aware"
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
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

cell(){ # tag  ngl  thresholds  maxtok
  local TAG=$1 NGL=$2 TH=$3 TOK=$4
  local D=$HOST/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  adb_safe_shell "su -c 'rm -f /data/local/tmp/ukv_ea_level /data/local/tmp/ea_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/ea_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG ..."
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG --eval-mode gen --max-tokens $TOK --ignore-eos --ctx-size 16384 \
    --seed 42 --threads 4 --n-gpu-layers $NGL --greedy --cache-type-k f16 --cache-type-v f16 \
    $MU $TH --n-batch 512 --n-ubatch 64 --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen \
    --out-csv /dev/null > /dev/null 2> $DEV/$TAG.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null
  adb_safe_pull "$DEV/$TAG.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.gen"  "$D/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/ea_$TAG.csv" "$D/sensors.csv" >/dev/null 2>&1
  adb_safe_shell "grep -m1 'energy-aware\] soc' $DEV/$TAG.err" < /dev/null | sed 's/^/    /'
  [ -f "$D/meta.json" ] && python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/clock_cell_report.py "$D" "$TAG" 2>/dev/null || echo "  [$TAG] FAILED"
}

echo "=== GPU: identical command line, only the battery state the controller sees differs ==="
cell gpu_healthy 99 "--ea-soc-hi 50 --ea-soc-lo 20" 4096
cell gpu_mid     99 "--ea-soc-hi 99 --ea-soc-lo 20" 4096
cell gpu_low     99 "--ea-soc-hi 99 --ea-soc-lo 99" 4096
echo "=== CPU: same three battery states (expect NO adaptation -- the CPU optimum is the floor) ==="
cell cpu_healthy  0 "--ea-soc-hi 50 --ea-soc-lo 20" 1024
cell cpu_low      0 "--ea-soc-hi 99 --ea-soc-lo 99" 1024
echo EAPROOF_DONE
