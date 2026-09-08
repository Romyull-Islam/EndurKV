#!/bin/bash
# ============================================================================
# run_longbench_phone_parity.sh -- phone-CPU LongBench parity subset (2026-08-02)
#
# WHY A SUBSET, AND WHY CPU.
# LongBench F1 is a property of (model, policy), not of silicon, so the full
# 4-model x 7-policy x 100-prompt grid is measured on the RTX at ~2s/cell. This
# script exists to prove the PHONE agrees with those numbers -- the same role the
# Jetson-vs-phone equivalence check plays for the eviction selection (14/14, alpha
# identical to 3 d.p.). A disagreement here would mean the RTX grid cannot stand
# in for the device, and we need to know that.
#
# CPU, not GPU, on measured grounds:
#   - Adreno gives no prefill advantage: 70.6 vs 67.6 tok/s on Llama-1B (noise),
#     and 18.3 vs 25.6 tok/s on Phi-3 -- i.e. the GPU is 29% SLOWER there.
#   - The phone GPU only runs 2 of our 4 models; Gemma-2B and Bonsai-8B abort
#     under FA-on with vk::DeviceLostError.
#   - LongBench generates <=128 tokens, so it is prefill-dominated and decode --
#     the only axis on which the GPU or eviction wins -- barely registers.
# The phone CPU is also where the 392-cell NIAH quality table was measured, so
# this subset lands in the same place as the rest of our quality evidence.
#
# NO COOL GATE, deliberately: this is a quality cell, greedy + fixed seed, so
# thermal state changes how fast a token appears, never which token. Nothing
# timed is reported from these cells. (Every timed/energy cell elsewhere keeps
# the DDR<=35C / batt<=33C gate.)  NO watchdog on any policy for the same reason.
# ============================================================================
set -u
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

LB=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/longbench
# 2026-08-02: v87 -> v88. v87 predates the 08-01 SnapKV window/kernel wiring and
# rejects --snapkv-kernel outright, which killed every SnapKV cell. v88 is the same
# armv8.7-a config (i8mm/dotprod) rebuilt from current source, pushed WITH all five
# .so deps (incl. the newly-required libmtmd.so) -- pushing a binary against stale
# libs is what produced 12 empty bin_vulkan cells previously. Fresh OUT_HOST so no
# v87 cell is ever mixed with a v88 cell in one table.
CB=/data/local/tmp/endurkv/bin_cpu_v88
OUT_HOST=/tmp/lb_phone_v88; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/lbphone_$(date +%Y%m%d_%H%M%S)
N=${N:-5}                      # prompts per task -- parity check, not a ranking
adb_safe_shell "mkdir -p $OUT/p" < /dev/null
declare -A MODELS=(
  [llama1b]=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  [phi3]=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf )
declare -A MAXGEN=( [hotpotqa]=32 [qasper]=128 )
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
flags_for(){ case "$1" in
  vanilla) echo "--policy vanilla" ;;
  mukv)    echo "$MU" ;;
  snapkv)  echo "--policy snapkv --obs-window 16 --snapkv-kernel 5 --n-sink 0" ;;
esac; }

for MT in llama1b phi3; do
 for TASK in hotpotqa qasper; do
  for i in $(seq -f "%03g" 0 $((N-1))); do
   SRC=$LB/$TASK/trunc_16384/$MT/prompt_$i.txt
   [ -f "$SRC" ] || continue
   adb push "$SRC" "$OUT/p/${MT}_${TASK}_$i.txt" < /dev/null >/dev/null 2>&1
   for POL in vanilla mukv snapkv; do
    ID="${MT}__${POL}__${TASK}__${i}"; PD=$OUT/$ID
    [ -f "$OUT_HOST/$ID/gen.txt" ] && continue
    adb_safe_shell "mkdir -p $PD" < /dev/null
    adb_safe_shell "LD_LIBRARY_PATH=$CB timeout ${TMO:-3600} $CB/eviction_bench \
      --prompt $OUT/p/${MT}_${TASK}_$i.txt --prompt-id $ID --eval-mode gen \
      --max-tokens ${MAXGEN[$TASK]} --ctx-size 16384 --model ${MODELS[$MT]} --seed 42 \
      --threads 6 --n-gpu-layers 0 --greedy --k-nominal 1024 $(flags_for $POL) \
      --out-meta $PD/meta.json --out-gen $PD/gen.txt --out-csv /dev/null \
      > $PD/out 2> $PD/err" < /dev/null
    adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
    echo "[$(date +%H:%M:%S)] $ID -> $(head -c 48 "$OUT_HOST/$ID/gen.txt" 2>/dev/null | tr '\n' ' ')"
   done
  done
 done
done
echo "LB_PHONE_DONE -> $OUT_HOST"
python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/longbench_score.py \
  --runs "$OUT_HOST" --gold "$LB/gold.json" --json-out "$OUT_HOST/scores.json"
