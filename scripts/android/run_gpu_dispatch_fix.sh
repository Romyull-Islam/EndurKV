#!/bin/bash
# ============================================================================
# run_gpu_dispatch_fix.sh -- diagnose AND attempt to fix the phone-GPU decode
# bimodality. (2026-08-14)
#
# WHAT IS KNOWN. Eight identical cells (/tmp/gpu_bimod) spread 28.20-39.21 tok/s, 39%,
# from an unchanged command line after an identical cool gate. The cause is NOT thermal
# and NOT the clock:
#     the fast cell ran at 976 MHz mean, LOWER than the slow cells' 996 MHz
#     thermal_pwrlevel was pinned at 5 in all eight
#     prefill was stable 129-136s; ALL variance was in decode (104-145s)
#     GPU busy fraction is the only correlate: 88.8% fast vs 79-82% slow
# At constant clock the GPU is simply idle more. Prefill is a few large kernels where
# dispatch cost amortises; decode is thousands of tiny ones where CPU-side submission
# latency dominates. So the submission thread is intermittently not keeping the GPU fed.
#
# THIS RUN TESTS THREE THINGS AT ONCE, which is why the arms are shaped as they are.
#
#   (1) WAS MY OWN INSTRUMENT THE CONFOUND? The bimodality campaign added a 2 Hz shell
#       loop doing five cats per tick to sample gpuclk. If dispatch latency is the
#       mechanism, that loop could have caused the starvation it was built to observe --
#       and the rates are suspicious: 7/8 slow there versus 3/9 slow in the energy
#       campaign, which had no such loop. NEITHER arm here runs it. sample_sensors.sh is
#       kept because it was present in every previous campaign, so it is a constant
#       rather than a variable.
#
#   (2) DOES CORE PLACEMENT EXPLAIN IT? The `pin` arm binds the process to the four
#       highest-clocked cores (taskset f0 = cpu4-7; cpu7 is the 4.61 GHz prime, the others
#       3.63 GHz) and raises priority (nice -20), so the submission thread cannot be
#       parked on a little core or descheduled behind background work. If pinning
#       collapses the spread, the mechanism is scheduling.
#
#   (3) IS IT A FIX? If `pin` is both faster AND tighter, it is not merely a diagnosis --
#       it should become the standard way every phone-GPU cell is run, which would make
#       muKV's speed numbers both larger and reproducible. That matters more than the
#       diagnosis: at a 39% spread no ratio below 1.39x is claimable, which currently puts
#       muKV's 1.23x on Llama-1B inside the noise (Phi-3's 2.81x survives).
#
# SCHED_FIFO was considered for the pin arm and rejected: a real-time process that spins
# can make the phone unrecoverable over adb, and nice -20 + taskset tests the same
# hypothesis without that risk.
#
# ARMS ALTERNATE base,pin,base,pin,... so any session drift is shared equally rather than
# loading onto one arm -- the same reasoning as the rotation in run_energy_aware_n3.sh.
# Config is the frozen muKV at k-pct 10, the operating point the tier sweep identified as
# optimal on both energy and retrieval.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv_n3
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
DEV=/data/local/tmp/endurkv/logs/dispatch_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/gpu_dispatch; mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-pct 10"
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null

settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    R=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $R|awk '{print int($1/1000)}'); local _sb=$(echo $R|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1; }

cell(){ # tag  launcher-prefix
  local TAG=$1 PRE=$2 D=$HOST/$1
  [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  adb_safe_shell "su -c 'rm -f /data/local/tmp/dp_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/dp_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG ($PRE) ..."
  adb_safe_shell "su -c 'cd $BIN && LD_LIBRARY_PATH=$BIN $PRE ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 \
    --seed 42 --threads 4 --n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 \
    $MU --n-batch 512 --n-ubatch 64 --out-meta $DEV/$TAG.json --out-gen /dev/null \
    --out-csv /dev/null > /dev/null 2> $DEV/$TAG.err'" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null
  adb_safe_pull "$DEV/$TAG.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/dp_$TAG.csv" "$D/sensors.csv" >/dev/null 2>&1
  T=$(grep -oE '"decode_tps": *[0-9.]+' $D/meta.json 2>/dev/null | grep -oE '[0-9.]+')
  echo "  [$TAG] tok/s=$T"
}

for i in 1 2 3 4; do
  cell base$i ""
  cell pin$i  "taskset f0 nice -n -20"
done
echo DISPATCH_DONE
