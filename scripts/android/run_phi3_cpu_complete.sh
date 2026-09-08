#!/bin/bash
# ============================================================================
# run_phi3_cpu_complete.sh -- Phi-3 phone-CPU table, CURRENT build, PUBLISHED
# configs. (2026-08-25)
#
# WHY. The existing Phi-3 CPU table (phone_7k_strict, 2026-07-05) is unusable
# twice over:
#   1. BUILD. It predates 2026-07-17, when the Android builds gained
#      dotprod/i8mm (armv8.7-a). Its tok/s and mWh are not comparable with any
#      current cell.
#   2. CONFIG. Every policy ran at a UNIFORM k_nominal=1024, n_sink=4 -- not the
#      setting each paper publishes. Verified from its own meta.json.
# This re-runs the same 7K prompt + 2048 decode so it supersedes that table
# directly, on bin_cpu_kd, with every baseline at ITS OWN published config --
# the same discipline the Llama-1B table uses.
#
# CONFIGS (as published): SnapKV window 32 / pool 7; Ada-KV 2048; TOVA 2048;
# H2O 20%-of-N (7542 tok -> 1508); StreamingLLM start 4 + recent 2000 = 2004;
# KeyDiff 2048 (its smallest published budget) with block-wise decode eviction
# B=128 and gather-compaction, which is intrinsic to KeyDiff, not a gift.
#
# muKV uses --compact-inplace: the state-API round-trip needs a second full
# cache and is OS-killed for Phi-3 at this size.
#
# GENERATIONS KEPT (--out-gen) so every cell is <unk>-gradeable, and BATTERY
# VOLTAGE recorded per cell: the cool gate fixes temperature but not electrical
# state, and a 4.45 V vs 3.79 V difference is invisible in the thermal log.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
exec 9>/tmp/.endurkv_queue.lock; flock -n 9 || { LOG "another campaign holds the lock"; exit 0; }
CB=/data/local/tmp/endurkv/bin_cpu_kd
M=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_7k.txt
DEV=/data/local/tmp/phi3cpu
HOST=/tmp/phi3_cpu_complete
mkdir -p $HOST; adb_safe_shell "mkdir -p $DEV" < /dev/null >/dev/null 2>&1
MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

cell(){
  local TAG=$1; shift
  [ -s "$HOST/$TAG/meta.json" ] && { LOG "$TAG cached"; return; }
  mkdir -p "$HOST/$TAG"
  LOG "cooling for $TAG ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) LOG "  $CG";; *) LOG "  [SKIP-HOT] $TAG"; return;; esac
  BV=$(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/voltage_now'" < /dev/null 2>/dev/null | tr -d ' \r')
  echo "batt_voltage_uv=$BV" > "$HOST/$TAG/start_power.txt"; LOG "  batt=${BV:-?} uV"
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $DEV/$TAG.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 10800 env LD_LIBRARY_PATH=$CB $CB/eviction_bench \
    --prompt $P --prompt-id $TAG --eval-mode gen --max-tokens 2048 --ignore-eos --ctx-size 16384 \
    --n-batch 512 --ubatch-size 64 --model $M --seed 42 --threads 4 --n-gpu-layers 0 --greedy \
    --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > $DEV/$TAG.out 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  local w=0
  while [ $w -lt 11000 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 30; w=$((w+30))
  done
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.json" "$HOST/$TAG/meta.json"   >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.gen"  "$HOST/$TAG/gen.txt"     >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.csv"  "$HOST/$TAG/sensors.csv" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.err"  "$HOST/$TAG/err.txt"     >/dev/null 2>&1
  [ -s "$HOST/$TAG/gen.txt" ] && LOG "  [$TAG] ok" || LOG "  [$TAG] NO OUTPUT"
}
LOG "=== Phi-3 phone CPU, 7K prompt + 2048 decode, bin_cpu_kd, published configs ==="
cell vanilla      --policy vanilla --k-nominal 1024
cell mukv         $MU --k-nominal 1024
cell keydiff2048  --policy keydiff --n-sink 0 --k-nominal 2048 --compact-inplace --keydiff-decode-block 128
cell snapkv       --policy snapkv --obs-window 32 --snapkv-kernel 7 --n-sink 0 --k-nominal 1024
cell adakv        --policy adakv --n-sink 0 --k-nominal 2048
cell tova         --policy tova --n-sink 0 --k-nominal 2048
cell h2o          --policy h2o --n-sink 0 --k-nominal 1508
cell streamingllm --policy streamingllm --n-sink 4 --k-nominal 2000
LOG "PHI3_CPU_COMPLETE_DONE"
touch /tmp/phi3_cpu_complete_DONE
