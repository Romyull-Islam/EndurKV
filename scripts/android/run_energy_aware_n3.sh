#!/bin/bash
# ============================================================================
# run_energy_aware_n3.sh -- the energy-aware result at n=3, with the run-order
# confound removed. (2026-08-14)
#
# WHY THIS RE-RUN EXISTS. run_energy_aware_proof.sh established that the controller
# closes the loop: identical command line, only the battery state it reads differs, and K
# moved 1947 -> 974 -> 487 with total system energy 1552 -> 1473 -> 1414 J (-8.9%). Two
# defects in that campaign stop it being publishable, and BOTH are design, not luck:
#
#   1. n=1 per arm. No error bar, so -8.9% cannot be separated from run-to-run spread.
#      Six identical muKV arms in an earlier phone campaign spread 11.2% on energy, which
#      is LARGER than the effect being claimed here. n=1 is therefore not merely weak, it
#      is uninterpretable.
#   2. Run order was confounded with the treatment. The arms ran strictly
#      healthy -> mid -> low, which is also monotonically decreasing K AND monotonically
#      decreasing battery charge, so any drift across the session (thermal history, pack
#      voltage, background daemons) aliases directly onto the result. The one drift that
#      could be checked -- pack voltage 4.210 -> 4.202 V -- happens to oppose the observed
#      trend, so the effect was probably understated rather than manufactured, but
#      "probably" is not a control.
#
# THE FIX FOR (2) IS A ROTATION, NOT A SHUFFLE. Each arm appears exactly once in each
# ordinal position across the three repetitions:
#       rep1:  healthy  mid      low
#       rep2:  low      healthy  mid
#       rep3:  mid      low      healthy
# A random shuffle could by chance reproduce the original ordering; this cannot. Any
# monotone session drift now contributes equally to all three arms instead of loading onto
# one, so it inflates the variance rather than biasing the mean.
#
# WHAT IS *NOT* FIXED, AND MUST BE WRITTEN INTO THE CAPTION. These cells run with USB
# attached. The OnePlus charger driver refuses to suspend the USB input -- writes to
# /sys/class/power_supply/usb/input_current_limit read straight back to 1500000 and
# current_max is SELinux-denied even to root -- so a genuinely unplugged run is not
# possible on this device. Charging is disabled (status "Not charging"), so the pack is
# not being topped up, but the rail still carries most of the load.
# CONSEQUENCE FOR REPORTING: quote TOTAL system energy (rail + coulomb-counter), which is
# what the phone consumed doing the work and transfers to the unplugged case where the
# pack supplies all of it. Do NOT quote the battery-only delta (the -33% mAh figure): with
# the rail carrying a near-constant load, the pack sees only the peaks, so that number
# describes the plugged-in split rather than the workload.
#
# Everything else follows run_energy_aware_proof.sh exactly -- same binary at
# /data/local/tmp/ukv (deliberately NOT the freshly rebuilt one, so these reps stay
# comparable with the n=1 campaign), same cool gate, same threshold trick for simulating
# state of charge, same sensor sampler.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv_n3
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
DEV=/data/local/tmp/endurkv/logs/ean3_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/ea_n3; mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --energy-aware"
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM

# charging OFF for the whole campaign (project protocol for any timed/energy cell)
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null

settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    R=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    # NOTE: these MUST stay `local`. An earlier version assigned D=... without local and
    # bash dynamic scoping clobbered cell()'s $D, so every pull wrote into a directory
    # named after the DDR temperature. That bug cost a whole campaign.
    local _sd=$(echo $R|awk '{print int($1/1000)}'); local _sb=$(echo $R|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1; }

cell(){ # tag  thresholds
  local TAG=$1 TH=$2
  local D=$HOST/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  adb_safe_shell "su -c 'rm -f /data/local/tmp/ukv_ea_level /data/local/tmp/ea_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/ea_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG ..."
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 \
    --seed 42 --threads 4 --n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 \
    $MU $TH --n-batch 512 --n-ubatch 64 --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen \
    --out-csv /dev/null > /dev/null 2> $DEV/$TAG.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null
  adb_safe_pull "$DEV/$TAG.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.gen"  "$D/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/ea_$TAG.csv" "$D/sensors.csv" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.err" "$D/err" >/dev/null 2>&1   # carries the controller decision
  adb_safe_shell "grep -m1 'energy-aware\] soc' $DEV/$TAG.err" < /dev/null | sed 's/^/    /'
  [ -f "$D/meta.json" ] && echo "  [$TAG] ok" || echo "  [$TAG] FAILED"
}

TH_HEALTHY="--ea-soc-hi 50 --ea-soc-lo 20"   # real SoC sits above both  -> level 0, k-pct 20
TH_MID="--ea-soc-hi 99 --ea-soc-lo 20"       # real SoC falls between    -> level 1, k-pct 10
TH_LOW="--ea-soc-hi 99 --ea-soc-lo 99"       # real SoC sits below both  -> level 2, k-pct 5

echo "=== rep1: healthy, mid, low ==="
cell r1_healthy "$TH_HEALTHY"; cell r1_mid "$TH_MID"; cell r1_low "$TH_LOW"
echo "=== rep2: low, healthy, mid  (rotated) ==="
cell r2_low "$TH_LOW"; cell r2_healthy "$TH_HEALTHY"; cell r2_mid "$TH_MID"
echo "=== rep3: mid, low, healthy  (rotated) ==="
cell r3_mid "$TH_MID"; cell r3_low "$TH_LOW"; cell r3_healthy "$TH_HEALTHY"
echo EA_N3_DONE
