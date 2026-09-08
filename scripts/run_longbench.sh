#!/bin/bash
# ============================================================================
# run_longbench.sh -- LongBench QA quality sweep.  (2026-08-02)
#
# WHY THIS EXISTS. SnapKV, Ada-KV, PyramidKV and RocketKV all report LongBench;
# our draft had only a "preliminary sample-0 F1", which is the single biggest
# reviewer-visible gap. This runs the real thing: official prompts, official
# generation lengths, official token-F1 (scripts/longbench_score.py), scored
# against gold answers recovered into benchmarks/longbench/gold.json.
#
# THIS IS A QUALITY MEASUREMENT, NOT A TIMED ONE -- and that distinction drives
# two deliberate choices:
#   1. NO cool gate, and cells run back-to-back. Greedy decoding with a fixed
#      seed is deterministic: DVFS/throttling changes how FAST a token is
#      produced, never WHICH token. So thermal state cannot move F1, and the
#      6-8h of cooling a gated sweep would cost buys nothing. Nothing timed is
#      reported from these cells -- the timing tables come from the gated phone
#      runs, and this script's prefill/decode fields are logged for triage only.
#   2. NO watchdog on ANY policy, including muKV. The watchdog only reduces
#      clocks; it cannot change emitted tokens. Running it here would slow the
#      sweep without changing a single F1. (The watchdog stays muKV-only in
#      every table where it CAN matter, i.e. the thermal/energy ones.)
#
# DEVICE. Defaults to the host CPU (i9-14900K): a 1B/3.8B Q4 prefill of ~15K
# tokens takes seconds here vs minutes on the phone, and F1 is a property of
# the policy + model, not the silicon. This mirrors how SnapKV et al. report
# LongBench on a server and device costs separately. Set BIN/THREADS to run it
# elsewhere; a phone subset is run separately to show device parity.
#
# Prompts are the middle-truncated set (truncate_longbench.py) so the QUESTION
# at the tail of each LongBench prompt always survives -- tail-clipping would
# delete it and flatten every policy's F1 to noise.
# ============================================================================
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

# ADDED 2026-08-02: gemma2b + bonsai8b. The prompt sets and gold answers were
# generated for all four models but this map still held only two, so those two
# invocations died on an unbound key under `set -u` and silently produced no
# cells. All four are the same models as the phone NIAH table, which is the point
# -- the two quality benchmarks must share one model set to be comparable.
declare -A MODEL=(
  [llama1b]=/home/mislam22/EndurKV_workspace/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  [phi3]=/home/mislam22/EndurKV_workspace/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
  [gemma2b]=/home/mislam22/EndurKV_workspace/models/gemma-2-2b-it-Q4_K_M.gguf
  [bonsai8b]=/home/mislam22/EndurKV_workspace/models/bonsai/Bonsai-8B-Q1_0.gguf )
# official dataset2maxlen. 2026-08-02: extended from 2 to 5 F1-scored tasks so the
# suite spans single-doc QA, multi-doc QA and few-shot rather than 2 tasks of 16.
# Summarization (gov_report/qmsum/multi_news) needs ROUGE-L and is left out.
declare -A MAXGEN=( [hotpotqa]=32 [qasper]=128 [multifieldqa_en]=64 [2wikimqa]=32 [triviaqa]=32 )

MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

flags_for(){ case "$1" in
  vanilla) echo "--policy vanilla" ;;
  mukv)    echo "$MU" ;;
  # SnapKV retunes window/kernel PER BENCHMARK (paper: NIAH 16/5, LongBench 32/7,
  # Command-R 64/13; FasterDecoding repo default 64/5). Until 2026-08-01 both were
  # hardcoded at 64/5 and --obs-window was inert, so "which SnapKV" a row measured
  # was not controlled. These three names make the choice explicit and comparable.
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
    # RESUME GUARD (fixed 2026-08-02): key on meta.json, NOT gen.txt. A cell that
    # crashed (CUDA OOM under GPU contention, ctx-alloc failure, timeout) still
    # leaves an EMPTY gen.txt behind, so a gen.txt-keyed resume treated crashed
    # cells as complete and never retried them -- and the scorer then read the
    # empty prediction as F1=0. meta.json is written only on success.
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
