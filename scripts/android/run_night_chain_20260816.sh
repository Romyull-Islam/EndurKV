#!/bin/bash
# ============================================================================
# run_night_chain_20260816.sh -- the overnight queue, in dependency order.
#
# WHY A CHAIN. Three campaigns want the phone tonight and they cannot overlap:
#   1. the LongBench tier sweep (already running, /tmp/lb_tiers) uses bin_cpu_cur,
#      so the NEW binaries (KeyDiff support) must not be pushed until it is done --
#      swapping a binary under a running campaign changes the engine mid-table,
#      the exact class of error this week kept finding in old data.
#   2. KeyDiff phone campaign: the mobile-positioned eviction baseline, implemented
#      and CUDA-verified 2026-08-16 (PPL 14.136 vs vanilla 14.361). Needs the new
#      builds on-device.
#   3. controller-v2 validation: re-validate the 902 MHz decode plateau PINNED at
#      n=3 (it is currently n=1, measured before the 36% dispatch bimodality was
#      understood), plus the composed low-battery package (cap 902 + k-pct 10).
#      Caps are WHOLE-RUN here: the decode-only phase cap needs a binary feature
#      (phase marker) that is deliberately not being added the same night the
#      binary already changed for KeyDiff. One change per build.
#
# BUILD-PROVENANCE RULES (why the push targets are what they are):
#   * bin_cpu_cur is UPDATED in place -- the CPU LongBench tables continue there,
#     and the added code paths (keydiff policy, keydiff_scores API, FA-off
#     kq_evict emission) are all dormant unless their flags are passed, so
#     existing-policy cells stay comparable. Verified: kq_evict on FA-off emits
#     only when g_endurkv_evict_obs_window > 0, which only --fa-on-evict sets.
#   * Vulkan goes to a NEW dir bin_vk_kd. /data/local/tmp/ukv_n3 is the provenance
#     of every pinned GPU number in the paper and stays byte-identical.
#
# CELL REUSE. KeyDiff is compared against cells that ALREADY exist on matching
# protocols (NIAH: /tmp/niah_vs_sllm vanilla/mukv/sfown; LongBench: /tmp/lb_native;
# pinned speed: /tmp/sllm_faithful). Only the keydiff arms run, plus the
# controller-v2 arms. Nothing valid is re-measured.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
WS=/home/mislam22/EndurKV_workspace
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }

# ── 0. wait for the tier sweep ────────────────────────────────────────────────
LOG "waiting for LB_TIERS_DONE"
for i in $(seq 1 600); do
  grep -q "LB_TIERS_DONE" /tmp/lb_tiers.log 2>/dev/null && break
  sleep 60
done
grep -q "LB_TIERS_DONE" /tmp/lb_tiers.log || { LOG "tier sweep never finished; aborting chain"; exit 1; }
LOG "tier sweep done ($(ls /tmp/lb_tiers/*/gen.txt 2>/dev/null | wc -l) cells)"

# ── 1. push new builds ────────────────────────────────────────────────────────
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

# on-device smoke: KeyDiff must score (not decline) on BOTH backends before the
# night is spent on it.
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

# ── 2. KeyDiff campaign ──────────────────────────────────────────────────────
# 2a. NIAH, phone GPU, same 14 stimuli as /tmp/niah_vs_sllm (no gate: retrieval
#     is thermally invariant; timing from these cells is never quoted).
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

# 2b. LongBench, phone CPU, keydiff arm only (joins /tmp/lb_native; no --ignore-eos).
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

# ── shared cool-gate for the timed cells below ───────────────────────────────
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

# 2c. KeyDiff pinned GPU speed, n=3 (joins /tmp/sllm_faithful's protocol; new dir
#     because the BINARY differs -- comparable, but the provenance split is explicit).
mkdir -p /tmp/kd_speed
for r in 1 2 3; do
  timed_cell /tmp/kd_speed kd_r$r 0 $KD
done

# ── 3. controller-v2 validation: cache x clock, pinned, cooled, n=3 ──────────
# arm A: healthy      -- k-pct 20, uncapped
# arm B: clock-only   -- k-pct 20, cap 902 MHz  (re-validates the plateau, pinned)
# arm C: composed low -- k-pct 10, cap 902 MHz  (the proposed low-battery package)
# Interleaved A,B,C x3 so drift is shared. Whole-run caps (see header).
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace"
mkdir -p /tmp/ctrl_v2
for r in 1 2 3; do
  timed_cell /tmp/ctrl_v2 A_full_r$r   0          $MU --k-pct 20
  timed_cell /tmp/ctrl_v2 B_902_r$r    902000000  $MU --k-pct 20
  timed_cell /tmp/ctrl_v2 C_low_r$r    902000000  $MU --k-pct 10
done
LOG "NIGHT_CHAIN_DONE"
