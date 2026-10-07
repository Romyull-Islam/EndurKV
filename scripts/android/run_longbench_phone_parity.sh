#!/bin/bash
# run_longbench_phone_parity.sh: small LongBench subset on the phone CPU, to check that the
# phone gives the same F1 as the full grid measured on the RTX host.
# CPU because LongBench is prefill-dominated (128 tokens or fewer generated), where the Adreno
# GPU gives no advantage, and the GPU cannot run every model.
# No cool gate and no watchdog: greedy decoding with a fixed seed makes the tokens independent
# of thermal state, and nothing timed is reported from these cells.
set -u
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

LB=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/longbench
# bin_cpu_v88 is an armv8.7-a build (i8mm, dotprod) that accepts --snapkv-kernel. Push it with
# all its .so files, a binary against stale libs gives empty cells. Separate OUT_HOST per build.
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
