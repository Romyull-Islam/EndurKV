#!/bin/bash
# Overnight queue in dependency order, so campaigns never share the phone:
#   1. wait for the LongBench tier sweep (it uses bin_cpu_cur, so new builds wait)
#   2. KeyDiff campaign: NIAH (GPU), LongBench (CPU), pinned GPU speed n=3
#   3. controller-v2 validation: k-pct x GPU clock cap, pinned, cooled, n=3
# bin_cpu_cur is updated in place since the new code paths are dormant without their
# flags. The Vulkan build goes to a new dir and /data/local/tmp/ukv_n3 is left as is.
# Only the new arms run. They join existing cells in /tmp/niah_vs_sllm and /tmp/lb_native.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
WS=/home/mislam22/EndurKV_workspace
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }

# 0. Wait for the tier sweep.
LOG "waiting for LB_TIERS_DONE"
for i in $(seq 1 600); do
  grep -q "LB_TIERS_DONE" /tmp/lb_tiers.log 2>/dev/null && break
  sleep 60
done
grep -q "LB_TIERS_DONE" /tmp/lb_tiers.log || { LOG "tier sweep never finished; aborting chain"; exit 1; }
LOG "tier sweep done ($(ls /tmp/lb_tiers/*/gen.txt 2>/dev/null | wc -l) cells)"

# 1. Push new builds.
LOG "pushing KeyDiff builds"
for so in $WS/EndurKV/llama.cpp/build-android/bin/lib*.so; do
  adb push "$so" /data/local/tmp/endurkv/bin_cpu_cur/ < /dev/null >/dev/null 2>&1
done
adb push $WS/EndurKV/entropy_probe/build-android/eviction_bench \
    /data/local/tmp/endurkv/bin_cpu_cur/eviction_bench < /dev/null >/dev/null 2>&1
adb_safe_shell "mkdir -p /data/local/tmp/ukv_kd" < /dev/null
for so in $WS/EndurKV/llama.cpp/build-android-vulkan/bin/lib*.so; do
  adb push "$so" /data/local/tmp/ukv_kd/ < /dev/null >/dev/null 2>&1
done
adb push $WS/EndurKV/entropy_probe/build-android-vulkan/eviction_bench \
    /data/local/tmp/ukv_kd/eviction_bench < /dev/null >/dev/null 2>&1
adb_safe_shell "chmod 755 /data/local/tmp/endurkv/bin_cpu_cur/eviction_bench /data/local/tmp/ukv_kd/eviction_bench" < /dev/null

# On-device smoke: KeyDiff must score (not decline) on both backends before the long runs.
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
KD="--policy keydiff --n-sink 0 --k-nominal 2048 --compact-inplace"
for spec in "cpu:/data/local/tmp/endurkv/bin_cpu_cur:0" "gpu:/data/local/tmp/ukv_kd:99"; do
  IFS=: read -r name BINDIR NGL <<< "$spec"
  adb_safe_shell "cd $BINDIR && LD_LIBRARY_PATH=$BINDIR timeout 400 ./eviction_bench \
    --prompt $P --prompt-id kdsmoke_$name --eval-mode gen --max-tokens 8 --ctx-size 16384 \
    --model $M --seed 42 --threads 4 --n-gpu-layers $NGL --greedy \
    --cache-type-k f16 --cache-type-v f16 $KD --n-batch 512 --n-ubatch 64 \
    --out-meta /data/local/tmp/kdsmoke_$name.json --out-gen /dev/null --out-csv /dev/null \
    > /dev/null 2> /data/local/tmp/kdsmoke_$name.err" < /dev/null
  OK=$(adb_safe_shell "grep -c 'keydiff\] scored' /data/local/tmp/kdsmoke_$name.err" < /dev/null | tr -d '\r')
  LOG "keydiff smoke $name: scored-lines=$OK"
  [ "${OK:-0}" -ge 1 ] || { LOG "keydiff smoke FAILED on $name -- aborting chain"; exit 1; }
done

# 2. KeyDiff campaign.
# 2a. NIAH, phone GPU, same 14 stimuli as /tmp/niah_vs_sllm. No cool gate, since
#     timing from these cells is not used.
SRC=$WS/EndurKV/benchmarks/niah
DEV=/data/local/tmp/endurkv/logs/kd_$(date +%Y%m%d_%H%M%S)
adb_safe_shell "mkdir -p $DEV" < /dev/null
for f in $SRC/niah_L*_n0.txt; do adb push "$f" "$DEV/$(basename $f)" < /dev/null >/dev/null 2>&1; done
mkdir -p /tmp/niah_vs_sllm
for STIM in $(cd $SRC && ls niah_L*_n0.txt); do
  case "$STIM" in *_L8K_*) CTX=8192;; *) CTX=4096;; esac
  id="keydiff__${STIM%.txt}"; D=/tmp/niah_vs_sllm/$id
  [ -f "$D/meta.json" ] && continue
  mkdir -p "$D"
  LOG "NIAH $id"
  adb_safe_shell "cd /data/local/tmp/ukv_kd && LD_LIBRARY_PATH=/data/local/tmp/ukv_kd timeout 600 ./eviction_bench \
    --prompt $DEV/$STIM --prompt-id $id --eval-mode gen --max-tokens 64 --ignore-eos \
    --ctx-size $CTX --model $M --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 $KD --n-batch 512 --n-ubatch 64 \
    --out-meta $DEV/$id.json --out-gen $DEV/$id.gen --out-csv /dev/null \
    > /dev/null 2> $DEV/$id.err" < /dev/null
  adb_safe_pull "$DEV/$id.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$id.gen"  "$D/gen.txt"   >/dev/null 2>&1
done

# 2b. LongBench, phone CPU, keydiff arm only (joins /tmp/lb_native, no --ignore-eos).
for task in qasper hotpotqa; do
  case $task in qasper) MG=128;; hotpotqa) MG=32;; esac
  for i in $(seq 0 14); do
    idx=$(printf "%03d" $i)
    src=/tmp/longbench_adaptive_3x3/phi3_vanilla_${task}/prompt_${idx}.txt
    [ -f "$src" ] || continue
    adb push "$src" "$DEV/${task}_${idx}.txt" < /dev/null >/dev/null 2>&1
    CELL="keydiff_${task}_${idx}"; D=/tmp/lb_native/$CELL
    [ -f "$D/gen.txt" ] && continue
    mkdir -p "$D"
    LOG "LB $CELL"
    adb_safe_shell "mkdir -p $DEV/$CELL; LD_LIBRARY_PATH=/data/local/tmp/endurkv/bin_cpu_cur timeout 1200 \
      /data/local/tmp/endurkv/bin_cpu_cur/eviction_bench \
      --prompt $DEV/${task}_${idx}.txt --prompt-id $CELL --eval-mode gen --max-tokens $MG \
      --ctx-size 16384 --model $M --seed 42 --threads 4 --n-gpu-layers 0 \
      --n-batch 512 --ubatch-size 64 --greedy --cache-type-k f16 --cache-type-v f16 $KD \
      --out-meta $DEV/$CELL/meta.json --out-gen $DEV/$CELL/gen.txt --out-csv /dev/null \
      > /dev/null 2> $DEV/$CELL/err" < /dev/null
    adb_safe_pull "$DEV/$CELL/gen.txt"   "$D/gen.txt"   >/dev/null 2>&1
    adb_safe_pull "$DEV/$CELL/meta.json" "$D/meta.json" >/dev/null 2>&1
  done
done

# Shared cool gate for the timed cells below.
settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    R=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $R|awk '{print int($1/1000)}'); local _sb=$(echo $R|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1; }
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
trap 'adb_safe_shell "su -c \"pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable\"" < /dev/null' EXIT INT TERM
adb push $WS/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
PIN="taskset f0 nice -n -20"
P16=/data/local/tmp/endurkv/corpora/prompt_12k.txt

timed_cell(){ # dir tag clk_cap flags...
  local HOSTD=$1 TAG=$2 CAP=$3; shift 3
  local D=$HOSTD/$TAG; [ -f "$D/meta.json" ] && { LOG "$TAG cached"; return; }
  mkdir -p "$D"
  LOG "cooling for $TAG"; settle || { LOG "SKIP-HOT $TAG"; return; }
  if [ "$CAP" != "0" ]; then
    adb_safe_shell "su -c 'echo $CAP > /sys/class/kgsl/kgsl-3d0/max_gpuclk'" < /dev/null
  fi
  adb_safe_shell "su -c 'rm -f /data/local/tmp/tc_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/tc_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  LOG "running $TAG (cap=$CAP)"
  adb_safe_shell "su -c 'cd /data/local/tmp/ukv_kd && LD_LIBRARY_PATH=/data/local/tmp/ukv_kd $PIN ./eviction_bench \
    --model $M --prompt $P16 --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos \
    --ctx-size 16384 --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 $* --n-batch 512 --n-ubatch 64 \
    --out-meta $DEV/$TAG.json --out-gen /dev/null --out-csv /dev/null > /dev/null 2> $DEV/$TAG.err'" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk'" < /dev/null
  adb_safe_pull "$DEV/$TAG.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/tc_$TAG.csv" "$D/sensors.csv" >/dev/null 2>&1
  T=$(grep -oE '"decode_tps": *[0-9.]+' $D/meta.json 2>/dev/null | grep -oE '[0-9.]+')
  LOG "$TAG tok/s=$T"
}

# 2c. KeyDiff pinned GPU speed, n=3, same protocol as /tmp/sllm_faithful. Separate
#     dir because the binary differs.
mkdir -p /tmp/kd_speed
for r in 1 2 3; do
  timed_cell /tmp/kd_speed kd_r$r 0 $KD
done

# 3. controller-v2 validation: cache x clock, pinned, cooled, n=3.
# A: k-pct 20, uncapped. B: k-pct 20, cap 902 MHz. C: k-pct 10, cap 902 MHz (low battery).
# Interleaved A,B,C x3 so drift is shared. Caps apply to the whole run.
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace"
mkdir -p /tmp/ctrl_v2
for r in 1 2 3; do
  timed_cell /tmp/ctrl_v2 A_full_r$r   0          $MU --k-pct 20
  timed_cell /tmp/ctrl_v2 B_902_r$r    902000000  $MU --k-pct 20
  timed_cell /tmp/ctrl_v2 C_low_r$r    902000000  $MU --k-pct 10
done
LOG "NIGHT_CHAIN_DONE"
