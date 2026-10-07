#!/bin/bash
# LongBench QA quality sweep: official prompts, generation lengths and token F1
# (scripts/longbench_score.py) against benchmarks/longbench/gold.json.
# No cool gate and no watchdog. Greedy decoding with a fixed seed is deterministic,
# so clocks and thermal state change speed but not the emitted tokens or F1.
# Runs on the host CPU by default (set BIN/THREADS to change). Prompts are
# middle-truncated (truncate_longbench.py) so the question at the tail survives.
set -u
LB=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/longbench
BIN=${BIN:-/home/mislam22/EndurKV_workspace/EndurKV/entropy_probe/build-host/eviction_bench}
CTX=${CTX:-16384}
THREADS=${THREADS:-16}
NGL=${NGL:-0}
N=${N:-50}                       # prompts per task
OUT=${OUT:-/tmp/longbench_host}
TASKS=${TASKS:-"hotpotqa qasper"}
POLICIES=${POLICIES:-"vanilla mukv snapkv"}
MODELS_LIST=${MODELS_LIST:-"llama1b phi3"}
mkdir -p "$OUT"

# Same four models as the phone NIAH table
declare -A MODEL=(
  [llama1b]=/home/mislam22/EndurKV_workspace/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  [phi3]=/home/mislam22/EndurKV_workspace/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
  [gemma2b]=/home/mislam22/EndurKV_workspace/models/gemma-2-2b-it-Q4_K_M.gguf
  [bonsai8b]=/home/mislam22/EndurKV_workspace/models/bonsai/Bonsai-8B-Q1_0.gguf )
# Official dataset2maxlen for the F1-scored tasks. Summarization needs ROUGE-L
# and is left out.
declare -A MAXGEN=( [hotpotqa]=32 [qasper]=128 [multifieldqa_en]=64 [2wikimqa]=32 [triviaqa]=32 )

MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

flags_for(){ case "$1" in
  vanilla) echo "--policy vanilla" ;;
  mukv)    echo "$MU" ;;
  # SnapKV window/kernel differ per benchmark in the paper (NIAH 16/5, LongBench
  # 32/7), so each setting gets its own name.
  snapkv)      echo "--policy snapkv --obs-window 16 --snapkv-kernel 5 --n-sink 0" ;;   # paper NIAH setting
  snapkv_lb)   echo "--policy snapkv --obs-window 32 --snapkv-kernel 7 --n-sink 0" ;;   # paper LongBench setting
  snapkv_repo) echo "--policy snapkv --obs-window 32 --snapkv-kernel 5 --n-sink 0" ;;   # FasterDecoding default (w32/k5, verified from snapkv_utils.py 2026-08-04)
  h2o)     echo "--policy h2o --n-sink 0 --obs-window 64" ;;
  tova)    echo "--policy tova" ;;
  streamingllm) echo "--policy streamingllm --n-sink 4" ;;
  adakv)   echo "--policy adakv --n-sink 0 --obs-window 32" ;;
esac; }

t0=$(date +%s); done_n=0
for MT in $MODELS_LIST; do
 for TASK in $TASKS; do
  PDIR=$LB/$TASK/trunc_${CTX}/$MT
  [ -d "$PDIR" ] || { echo "[skip] no truncated prompts: $PDIR"; continue; }
  for POL in $POLICIES; do
   for i in $(seq -f "%03g" 0 $((N-1))); do
    P=$PDIR/prompt_$i.txt; [ -f "$P" ] || continue
    CELL=$OUT/${MT}__${POL}__${TASK}__${i}
    # Resume on meta.json, which is written only on success. A crashed cell can
    # leave an empty gen.txt, which would score as F1=0.
    [ -f "$CELL/meta.json" ] && continue                    # resume
    mkdir -p "$CELL"
    # no --ignore-eos: LongBench stops at EOS or maxlen, whichever comes first
    timeout ${TMO:-1800} "$BIN" --prompt "$P" --prompt-id "${MT}_${POL}_${TASK}_$i" \
      --eval-mode gen --max-tokens ${MAXGEN[$TASK]} --ctx-size $CTX \
      --model "${MODEL[$MT]}" --seed 42 --threads $THREADS --n-gpu-layers $NGL --greedy \
      --k-nominal ${KBUD:-1024} $(flags_for $POL) \
      --out-meta "$CELL/meta.json" --out-gen "$CELL/gen.txt" --out-csv /dev/null \
      > "$CELL/out" 2> "$CELL/err"
    done_n=$((done_n+1))
    if [ $((done_n % 25)) -eq 0 ]; then
      echo "[$(date +%H:%M:%S)] $done_n cells, $(( ($(date +%s)-t0)/60 ))min elapsed (now: $MT/$POL/$TASK/$i)"
    fi
   done
   echo "[$(date +%H:%M:%S)] === $MT / $TASK / $POL complete ==="
  done
 done
done
echo "[$(date +%H:%M:%S)] LONGBENCH_DONE -> $OUT  ($done_n cells, $(( ($(date +%s)-t0)/60 ))min)"
python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/longbench_score.py \
  --runs "$OUT" --gold "$LB/gold.json" --json-out "$OUT/scores.json"
