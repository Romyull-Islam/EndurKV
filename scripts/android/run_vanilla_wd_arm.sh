#!/bin/bash
# ============================================================================
# run_vanilla_wd_arm.sh -- 5th soak arm: VANILLA + vendor-anchored glide.
# The decisive watchdog demonstration: vanilla is the only arm that actually
# enters the cliff run-up (battery 48C, peak DDR 68.3C -- ABOVE the 65C kernel
# cliff). muKV arms can only show the ladder dormant; this one shows it bending
# a genuinely cliff-bound temperature curve, gradually, before the vendor trips.
# Two validated ladder anchors exist (matching the GPU soak's wd-early /
# wd-vendor pair): vendor-anchored (battery 47.0+, skin 50.0+) glides only in
# the cliff run-up; EARLY (battery 35.0+, skin 39.5+) glides during workloads
# that never near the cliff, so BOTH sensors demonstrably act. Same staircase,
# same code, only the anchor shifted via env (BAT_L0/SKIN_L0).
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
LOG "waiting for the early arm ..."
while [ ! -f /tmp/wd_early_DONE ]; do sleep 60; done
exec 9>/tmp/.endurkv_queue.lock; flock 9
CB=/data/local/tmp/endurkv/bin_cpu_kd
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
DEV=/data/local/tmp/cpusoak; HOST=/tmp/cpu_soak/vanilla_wdon
mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
[ -s "$HOST/iter_6.json" ] && { LOG "cached"; exit 0; }
LOG "cooling for vanilla_wdon ..."
CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
case "$CG" in *"cool ddr="*) LOG "  $CG";; *) LOG "  [SKIP-HOT]"; exit 1;; esac
adb_safe_shell "su -c 'rm -f /data/local/tmp/wd.stop; nohup env sh /data/local/tmp/endurkv/scripts/preempt_throttle_watchdog_v2.sh $DEV/vanilla_wdon_wd.log /data/local/tmp/wd.stop >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $DEV/vanilla_wdon.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
for i in 1 2 3 4 5 6; do
  LOG "  vanilla_wdon gen $i/6 (no cooling)"
  adb_safe_shell "rm -f $DEV/vw.done; setsid nohup sh -c \"timeout 7200 env LD_LIBRARY_PATH=$CB $CB/eviction_bench \
    --prompt $DEV/wt/prompt.txt --prompt-id vwd_$i --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 \
    --n-batch 512 --ubatch-size 64 --model $M --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --cache-type-k f16 --cache-type-v f16 --policy vanilla --k-nominal 1024 \
    --out-meta $DEV/vwd_$i.json --out-gen $DEV/vwd_$i.gen --out-csv /dev/null > /dev/null 2> $DEV/vwd_$i.err ; \
    echo DONE > $DEV/vw.done\" >/dev/null 2>&1 &" < /dev/null
  w=0
  while [ $w -lt 7400 ]; do
    adb_safe_shell "[ -f $DEV/vw.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 30; w=$((w+30)); done
  adb_safe_pull "$DEV/vwd_$i.json" "$HOST/iter_$i.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/vwd_$i.gen"  "$HOST/iter_$i.gen"  >/dev/null 2>&1
done
adb_safe_shell "su -c 'touch /data/local/tmp/wd.stop; pkill -f sample_sensors 2>/dev/null'" < /dev/null >/dev/null 2>&1
adb_safe_pull "$DEV/vanilla_wdon.csv"    "$HOST/sensors.csv" >/dev/null 2>&1
adb_safe_pull "$DEV/vanilla_wdon_wd.log" "$HOST/wd.log"      >/dev/null 2>&1
LOG "VANILLA_WD_DONE"
touch /tmp/vanilla_wd_DONE
