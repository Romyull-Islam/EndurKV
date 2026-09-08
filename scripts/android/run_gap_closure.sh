#!/bin/bash
# ============================================================================
# run_gap_closure.sh -- the last three coverage gaps. (2026-08-25)
# Waits for the Phi-3 CPU campaign; one campaign owns the phone at a time.
#
#  D. KeyDiff on GPU Llama-1B   -- the two GPU rows are otherwise not policy-
#     matched (GPU Phi-3 has KeyDiff, GPU Llama-1B does not).
#  E. KeyDiff on CPU Bonsai-8B  -- the only missing cell in that row.
#  F. gemma-2B CPU, published configs, current build -- same double defect the
#     Phi-3 CPU table had: pre-2026-07-17 binary AND a uniform k_nominal=1024
#     for every policy instead of each paper's own setting.
#
# H2O uses 20%-of-N resolved per prompt: gemma 6382 -> 1276.
# Generations kept and battery voltage recorded on every cell.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
LOG "waiting for the Phi-3 CPU campaign ..."
while [ ! -f /tmp/phi3_cpu_complete_DONE ]; do sleep 60; done
exec 9>/tmp/.endurkv_queue.lock; flock 9
LOG "starting gap closure"
CB=/data/local/tmp/endurkv/bin_cpu_kd
VK=/data/local/tmp/endurkv/bin_vk_cur
MOD=/data/local/tmp/endurkv/models
DEV=/data/local/tmp/gapclose
adb_safe_shell "mkdir -p $DEV" < /dev/null >/dev/null 2>&1
MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

# $1=host_root $2=tag $3=bin $4=model $5=prompt $6=maxtok $7=gpulayers  rest=flags
cell(){
  local ROOT=$1 TAG=$2 BIN=$3 MP=$4 PR=$5 MT=$6 GL=$7; shift 7
  mkdir -p "$ROOT/$TAG"
  [ -s "$ROOT/$TAG/meta.json" ] && { LOG "$TAG cached"; return; }
  LOG "cooling for $TAG ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) LOG "  $CG";; *) LOG "  [SKIP-HOT] $TAG"; return;; esac
  BV=$(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/voltage_now'" < /dev/null 2>/dev/null | tr -d ' \r')
  echo "batt_voltage_uv=$BV" > "$ROOT/$TAG/start_power.txt"
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $DEV/$TAG.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 10800 env LD_LIBRARY_PATH=$BIN $BIN/eviction_bench \
    --prompt $PR --prompt-id $TAG --eval-mode gen --max-tokens $MT --ignore-eos --ctx-size 16384 \
    --n-batch 512 --n-ubatch 64 --model $MP --seed 42 --threads 4 --n-gpu-layers $GL --greedy \
    --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > $DEV/$TAG.out 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  local w=0
  while [ $w -lt 11000 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 30; w=$((w+30)); done
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.json" "$ROOT/$TAG/meta.json"   >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.gen"  "$ROOT/$TAG/gen.txt"     >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.csv"  "$ROOT/$TAG/sensors.csv" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.err"  "$ROOT/$TAG/err.txt"     >/dev/null 2>&1
  [ -s "$ROOT/$TAG/gen.txt" ] && LOG "  [$TAG] ok" || LOG "  [$TAG] NO OUTPUT"
}

KD="--policy keydiff --n-sink 0 --k-nominal 2048 --compact-inplace --keydiff-decode-block 128"
# D. KeyDiff, GPU, Llama-1B (12K prompt + 4096, matching phone_gpu_16k)
cell /tmp/gap_gpu_llama1b keydiff2048 $VK $MOD/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
     /data/local/tmp/endurkv/corpora/prompt_12k.txt 4096 99 $KD
# E. KeyDiff, CPU, Bonsai-8B (matching nat_bonsai: 10074 prompt + 4096)
cell /tmp/gap_cpu_bonsai keydiff2048 $CB $MOD/Bonsai-8B-Q1_0.gguf \
     /data/local/tmp/endurkv/corpora/prompt_12k.txt 4096 0 $KD
# F. gemma-2B CPU, published configs (6382 prompt -> H2O 20% = 1276)
G=/tmp/gap_cpu_gemma; GM=$MOD/gemma-2-2b-it-Q4_K_M.gguf; GP=/data/local/tmp/endurkv/corpora/prompt_7k.txt
cell $G vanilla      $CB $GM $GP 2048 0 --policy vanilla --k-nominal 1024
cell $G mukv         $CB $GM $GP 2048 0 $MU --k-nominal 1024
cell $G keydiff2048  $CB $GM $GP 2048 0 $KD
cell $G snapkv       $CB $GM $GP 2048 0 --policy snapkv --obs-window 32 --snapkv-kernel 7 --n-sink 0 --k-nominal 1024
cell $G adakv        $CB $GM $GP 2048 0 --policy adakv --n-sink 0 --k-nominal 2048
cell $G tova         $CB $GM $GP 2048 0 --policy tova --n-sink 0 --k-nominal 2048
cell $G h2o          $CB $GM $GP 2048 0 --policy h2o --n-sink 0 --k-nominal 1276
cell $G streamingllm $CB $GM $GP 2048 0 --policy streamingllm --n-sink 4 --k-nominal 2000
LOG "GAP_CLOSURE_DONE"
touch /tmp/gap_closure_DONE
