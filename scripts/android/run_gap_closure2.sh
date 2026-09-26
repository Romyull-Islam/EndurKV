#!/bin/bash
# ============================================================================
# run_gap_closure2.sh -- the table gaps the 2026-09-20 audit found. (2026-09-20)
#
#  A. gemma-2-2b on the Adreno. The GPU table covers Llama-1B and Phi-3 only.
#     The paper currently says gemma is CPU-only because its context is 8192,
#     but gemma has its OWN 6382-token prompt (prompt_7k.txt), so the real
#     question is whether the Vulkan backend runs gemma-2 at all. Matched to
#     the gemma CPU row: same prompt, same 2048 generated, same budgets.
#     Every cell keeps its generation so the output can be checked for the
#     Bonsai failure mode (full speed, garbage tokens).
#  B. Bonsai-8B full cache on the KeyDiff build. gap_cpu_bonsai/keydiff2048
#     has no same-campaign no-eviction reference, so that pair of CPU cells
#     is currently blank.
#  C. KeyDiff on the needle grid, all four models: see run_niah_keydiff.sh,
#     which this script launches last.
#
# Charging is disabled by cool_gate.sh on every cell; the phone stays plugged in.
# Resumable: a cell whose meta.json already exists is skipped.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%F' '%H:%M:%S)] $*"; }
exec 9>/tmp/.endurkv_queue.lock; flock 9
LOG "gap closure 2 start"
CB=/data/local/tmp/endurkv/bin_cpu_kd          # KeyDiff-capable CPU build
VK=/data/local/tmp/endurkv/bin_vk_cur          # current Vulkan build
MOD=/data/local/tmp/endurkv/models
DEV=/data/local/tmp/gapclose2
adb_safe_shell "mkdir -p $DEV" < /dev/null >/dev/null 2>&1
MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
KD="--policy keydiff --n-sink 0 --k-nominal 2048 --compact-inplace --keydiff-decode-block 128"

# $1=host_root $2=tag $3=bin $4=model $5=prompt $6=maxtok $7=gpulayers $8=ctx  rest=flags
cell(){
  local ROOT=$1 TAG=$2 BIN=$3 MP=$4 PR=$5 MT=$6 GL=$7 CTX=$8; shift 8
  mkdir -p "$ROOT/$TAG"
  [ -s "$ROOT/$TAG/meta.json" ] && { LOG "$TAG cached"; return; }
  LOG "cooling for $TAG ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) LOG "  $CG";; *) LOG "  [SKIP-HOT] $TAG"; return;; esac
  # cool_gate.sh's charging_restore() turns charging back on when it exits; the cell
  # must be metered with it off (see phone-energy-cooling-gate).
  adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null >/dev/null 2>&1
  BV=$(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/voltage_now'" < /dev/null 2>/dev/null | tr -d ' \r')
  echo "batt_voltage_uv=$BV" > "$ROOT/$TAG/start_power.txt"
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $DEV/$TAG.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 10800 env LD_LIBRARY_PATH=$BIN $BIN/eviction_bench \
    --prompt $PR --prompt-id $TAG --eval-mode gen --max-tokens $MT --ignore-eos --ctx-size $CTX \
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
  if [ -s "$ROOT/$TAG/gen.txt" ]; then
    LOG "  [$TAG] ok  $(head -c 90 "$ROOT/$TAG/gen.txt" | tr '\n' ' ')"
  else
    LOG "  [$TAG] NO OUTPUT"
  fi
}

# --- A. gemma-2-2b on the Adreno, matched to its CPU row -------------------
# prompt_7k.txt is 6382 gemma tokens; H2O's 20% of N is 1276, as on the CPU.
G=/tmp/gap_gpu_gemma; GM=$MOD/gemma-2-2b-it-Q4_K_M.gguf; GP=/data/local/tmp/endurkv/corpora/prompt_7k.txt
# probe first: 64 tokens, ctx 16384. If the Vulkan backend cannot run gemma-2 at all,
# this costs two minutes instead of an hour, and its gen.txt shows whether the
# output is real text or the Bonsai non-finite-logit failure.
cell $G probe_vanilla $VK $GM $GP 64 99 16384 --policy vanilla --k-nominal 1024
if [ ! -s $G/probe_vanilla/gen.txt ]; then
  LOG "gemma GPU probe produced no output; skipping the gemma GPU block"
else
  cell $G vanilla      $VK $GM $GP 2048 99 16384 --policy vanilla --k-nominal 1024
  cell $G mukv         $VK $GM $GP 2048 99 16384 $MU --k-nominal 1024
  cell $G streamingllm $VK $GM $GP 2048 99 16384 --policy streamingllm --n-sink 4 --compact-inplace --k-nominal 2000
  cell $G snapkv       $VK $GM $GP 2048 99 16384 --policy snapkv --obs-window 32 --snapkv-kernel 7 --n-sink 0 --k-nominal 1024
  cell $G adakv        $VK $GM $GP 2048 99 16384 --policy adakv --n-sink 0 --k-nominal 2048
  cell $G tova         $VK $GM $GP 2048 99 16384 --policy tova --n-sink 0 --k-nominal 2048
  cell $G h2o          $VK $GM $GP 2048 99 16384 --policy h2o --n-sink 0 --k-nominal 1276
  cell $G keydiff2048  $VK $GM $GP 2048 99 16384 $KD
fi

# --- B. Bonsai-8B full cache on the KeyDiff build --------------------------
# Pairs with the existing /tmp/gap_cpu_bonsai/keydiff2048 (10074 prompt + 4096).
cell /tmp/gap_cpu_bonsai vanilla $CB $MOD/Bonsai-8B-Q1_0.gguf \
     /data/local/tmp/endurkv/corpora/prompt_12k.txt 4096 0 16384 --policy vanilla --k-nominal 1024

LOG "GAP_CLOSURE2_DONE"
touch /tmp/gap_closure2_DONE
# --- C. KeyDiff on the needle grid, all four models ------------------------
nohup bash /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/run_niah_keydiff.sh \
  > /tmp/niah_keydiff.log 2>&1 &
LOG "needle KeyDiff campaign launched -> /tmp/niah_keydiff.log"
