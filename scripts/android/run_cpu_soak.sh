#!/bin/bash
# ============================================================================
# run_cpu_soak.sh -- SUSTAINED CPU load: the missing watchdog demonstration.
# (2026-08-26)
#
# EVERY timed CPU cell so far is cool-gated and finishes before the thermal
# ladders can engage: wd logs end tier=0 in all of them, so the paper currently
# has NO CPU evidence for "the watchdog holds the clock by capping frequency
# BEFORE the vendor throttle trips". The GPU soak has it (+10.8% tok/s, throttle
# residency 41.0%->28.5%); this is the CPU counterpart.
#
# DESIGN. Per arm: ONE cool gate, then SIX back-to-back 9.7K+4K generations with
# NO cooling between them -- heat accumulates as it would in real use. Arms:
#   vanilla        : does the full cache deep-throttle under sustained load?
#   mukv_wdoff     : is eviction alone enough to stay out of throttle?
#   mukv_wdon      : does the preemptive ladder (preempt_throttle_watchdog_v2,
#                    name-resolved zones, post-07-18 fix) beat the vendor
#                    governor once heat has accumulated?
# Recorded per generation: tok/s, and at 5 Hz: cpu6 clock (883 MHz floor
# residency), DDR/CPU/battery temps -- enough for a per-generation trace figure.
# Threads 6, same build and prompt as the n=3 table so rows are comparable.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
LOG "waiting for the prefill diagnostic ..."
while [ ! -f /tmp/prefill_diag_DONE ]; do sleep 60; done
exec 9>/tmp/.endurkv_queue.lock; flock 9
LOG "starting CPU soak"
CB=/data/local/tmp/endurkv/bin_cpu_kd
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
DEV=/data/local/tmp/cpusoak; HOST=/tmp/cpu_soak
mkdir -p $HOST; adb_safe_shell "mkdir -p $DEV/wt" < /dev/null >/dev/null 2>&1
timeout 180 adb push /tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad/wikitext_16k_p12k_d4k.txt "$DEV/wt/prompt.txt" < /dev/null >/dev/null 2>&1
MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

arm(){
  local ARM=$1 WD=$2; shift 2
  [ -s "$HOST/$ARM/iter_6.json" ] && { LOG "$ARM cached"; return; }
  mkdir -p "$HOST/$ARM"
  LOG "cooling once for arm $ARM ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) LOG "  $CG";; *) LOG "  [SKIP-HOT] $ARM"; return;; esac
  if [ "$WD" = on ]; then
    adb_safe_shell "su -c 'rm -f /data/local/tmp/wd.stop; nohup sh /data/local/tmp/endurkv/scripts/preempt_throttle_watchdog_v2.sh $DEV/${ARM}_wd.log /data/local/tmp/wd.stop >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
  fi
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $DEV/$ARM.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
  for i in 1 2 3 4 5 6; do
    LOG "  $ARM gen $i/6 (no cooling)"
    adb_safe_shell "rm -f $DEV/$ARM.done; setsid nohup sh -c \"timeout 7200 env LD_LIBRARY_PATH=$CB $CB/eviction_bench \
      --prompt $DEV/wt/prompt.txt --prompt-id ${ARM}_$i --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 \
      --n-batch 512 --ubatch-size 64 --model $M --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
      --cache-type-k f16 --cache-type-v f16 $* \
      --out-meta $DEV/${ARM}_$i.json --out-gen $DEV/${ARM}_$i.gen --out-csv /dev/null > /dev/null 2> $DEV/${ARM}_$i.err ; \
      echo DONE > $DEV/$ARM.done\" >/dev/null 2>&1 &" < /dev/null
    local w=0
    while [ $w -lt 7400 ]; do
      adb_safe_shell "[ -f $DEV/$ARM.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
      sleep 30; w=$((w+30)); done
    adb_safe_pull "$DEV/${ARM}_$i.json" "$HOST/$ARM/iter_$i.json" >/dev/null 2>&1
    adb_safe_pull "$DEV/${ARM}_$i.gen"  "$HOST/$ARM/iter_$i.gen"  >/dev/null 2>&1
  done
  adb_safe_shell "su -c 'touch /data/local/tmp/wd.stop; pkill -f sample_sensors 2>/dev/null'" < /dev/null >/dev/null 2>&1
  adb_safe_pull "$DEV/$ARM.csv"      "$HOST/$ARM/sensors.csv" >/dev/null 2>&1
  adb_safe_pull "$DEV/${ARM}_wd.log" "$HOST/$ARM/wd.log"      >/dev/null 2>&1
  LOG "  [$ARM] arm complete"
}
arm vanilla    off --policy vanilla --k-nominal 1024
arm mukv_wdoff off $MU --k-nominal 1024
arm mukv_wdon  on  $MU --k-nominal 1024
LOG "CPU_SOAK_DONE"
touch /tmp/cpu_soak_DONE
