#!/bin/bash
# ============================================================================
# run_longbench_native_budgets.sh -- LongBench with EVERY policy at its own
# published budget, at a sample size that can support a claim. (2026-08-15)
#
# WHY THE EXISTING LONGBENCH TABLE CANNOT BE USED. /tmp/phone_longbench_full has two
# independent defects, and the second is worse than the first:
#   1. Every policy ran at --k-nominal 1024, including StreamingLLM (own budget 2004),
#      SnapKV/Ada-KV/TOVA (2048) and H2O (a RATIO, 20% of N, not an absolute at all).
#      Forcing one K onto every baseline deletes the policy being compared.
#   2. It completed 2-3 samples per cell. Scored on matched sample IDs the comparison
#      collapses to n=1-2, where Phi-3 shows the FULL CACHE at 0.00 F1 while muKV scores
#      47.62 on a single draw, and Gemma-2 scores 0.00 for every policy including vanilla.
#      Those are not results. No budget correction can rescue a table that thin, which is
#      why this re-runs everything rather than patching StreamingLLM alone.
#
# BUDGETS, each from the policy's own paper/repo, matching run_64k_native_budgets.sh:
#   vanilla       full cache
#   muKV          K=1024, its frozen deployment budget, + in-place compaction
#   SnapKV        2048, obs-window 32, kernel 5, no sinks
#   Ada-KV        2048 total, adaptively allocated across heads
#   TOVA          2048 multi-state
#   StreamingLLM  start_size 4 + recent_size 2000 = 2004
#                 (mit-han-lab/streaming-llm, run_streaming_llama.py argparse defaults)
#   H2O           20% of N -- a RATIO, so K is computed per prompt from that prompt's
#                 measured token count, not fixed. This is the only policy whose budget
#                 changes per sample, and flattening it to a constant is what the old
#                 campaign did.
#
# TWO PASSES, and the first is not overhead. Pass 1 runs vanilla, which the table needs
# anyway, and its meta.json yields n_prompt_tokens per sample -- exactly what H2O's ratio
# needs. --k-pct exists in the binary but is gated to the muKV path, so resolving H2O's
# budget host-side from measured lengths is both correct and avoids changing engine code
# in the middle of a measurement campaign.
#
# LLAMA-3.2-1B ONLY, deliberately. It is the model the rest of the corrected phone table
# uses, and the only one of the three that produces non-degenerate F1 on these tasks
# (vanilla 42.42 hotpotqa / 30.43 qasper, against gemma2b 0.00 and phi3 0.00). Adding
# models that score zero for every policy including the ceiling adds rows, not evidence.
# Gemma-2 additionally has n_ctx_train 8192 against this campaign's ctx 16384.
#
# NO COOL GATE: F1 is decided by the keep-set under greedy decoding, both deterministic.
# NO TIMING OR ENERGY MAY BE QUOTED FROM THESE CELLS.
#
# --ignore-eos REMOVED 2026-08-16. It was copied in from the throughput harnesses, where a
# FIXED token count is exactly what you want so that tok/s is comparable across cells. It
# is wrong for QA scoring: the model emits the correct answer, hits EOS, and then keeps
# generating because we told it to, so token-F1 divides the right answer by a prediction
# 3-5x too long and precision collapses. Measured cost of the flag on the cells already
# collected -- same generations, scored with and without truncation at the first EOS:
#     vanilla  hotpotqa  15.80 -> 36.00      muKV hotpotqa  15.25 -> 33.85
#     vanilla  qasper    15.08 -> 20.29      muKV qasper    13.20 -> 17.32
# The truncated values line up with the RTX table (42.14 / 40.72 / 17.55 / 17.06), which
# is how the flag was caught: CPU and CUDA should not disagree by 2.5x on the same greedy
# decode of the same prompt, and they did not -- the harness did.
#
# The cells ALREADY COLLECTED remain usable without a re-run: generation is greedy and
# causal, so the token sequence before EOS is identical whether or not we kept going.
# Truncating at the first EOS post-hoc is exactly equivalent to having stopped there.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/endurkv/bin_cpu_cur
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
OUT=/data/local/tmp/endurkv/logs/lbnat_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/lb_native; mkdir -p $HOST
N=${N_SAMPLES:-15}
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024"
adb_safe_shell "mkdir -p $OUT" < /dev/null
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
cleanup(){ adb_safe_shell "su -c 'echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM
for task in qasper hotpotqa; do
  for i in $(seq 0 $((N-1))); do
    idx=$(printf "%03d" $i)
    src=/tmp/longbench_adaptive_3x3/phi3_vanilla_${task}/prompt_${idx}.txt
    [ -f "$src" ] && adb push "$src" "$OUT/${task}_${idx}.txt" < /dev/null >/dev/null 2>&1
  done
done
mg(){ case $1 in qasper) echo 128;; hotpotqa) echo 32;; *) echo 64;; esac; }

run(){ # cell task idx flags...
  local CELL=$1 task=$2 idx=$3; shift 3
  local D=$HOST/$CELL; [ -f "$D/gen.txt" ] && return
  mkdir -p "$D"
  adb_safe_shell "mkdir -p $OUT/$CELL; LD_LIBRARY_PATH=$BIN timeout 1200 $BIN/eviction_bench \
    --prompt $OUT/${task}_${idx}.txt --prompt-id $CELL --eval-mode gen --max-tokens $(mg $task) \
    --ctx-size 16384 --model $M --seed 42 --threads 4 --n-gpu-layers 0 \
    --n-batch 512 --ubatch-size 64 --greedy --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $OUT/$CELL/meta.json --out-gen $OUT/$CELL/gen.txt --out-csv /dev/null \
    > /dev/null 2> $OUT/$CELL/err" < /dev/null
  adb_safe_pull "$OUT/$CELL/gen.txt"   "$D/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "$OUT/$CELL/meta.json" "$D/meta.json" >/dev/null 2>&1
}

echo "=== PASS 1: vanilla (also yields n_prompt_tokens for H2O's ratio) ==="
for task in qasper hotpotqa; do
  for i in $(seq 0 $((N-1))); do
    idx=$(printf "%03d" $i)
    echo "[$(date +%H:%M:%S)] vanilla_${task}_${idx}"
    run "vanilla_${task}_${idx}" $task $idx --policy vanilla
  done
done

echo "=== PASS 2: every other policy at its own budget ==="
for task in qasper hotpotqa; do
  for i in $(seq 0 $((N-1))); do
    idx=$(printf "%03d" $i)
    NP=$(python3 -c "
import json,re,os
p='$HOST/vanilla_${task}_${idx}/meta.json'
print(json.loads(re.sub(r':\s*-?nan\b',': NaN',open(p).read())).get('n_prompt_tokens',0) if os.path.exists(p) else 0)" 2>/dev/null)
    [ "${NP:-0}" -lt 100 ] && { echo "  skip ${task}_${idx}: no vanilla token count"; continue; }
    H2OK=$(( NP / 5 ))    # 20% of N, H2O's published ratio
    echo "[$(date +%H:%M:%S)] ${task}_${idx}  N=$NP  H2O K=$H2OK"
    run "mukv_${task}_${idx}"         $task $idx $MU
    run "snapkv_${task}_${idx}"       $task $idx --policy snapkv --obs-window 32 --snapkv-kernel 5 --n-sink 0 --k-nominal 2048
    run "adakv_${task}_${idx}"        $task $idx --policy adakv --obs-window 32 --n-sink 0 --k-nominal 2048
    run "tova_${task}_${idx}"         $task $idx --policy tova --k-nominal 2048
    run "streamingllm_${task}_${idx}" $task $idx --policy streamingllm --n-sink 4 --k-nominal 2004
    run "h2o_${task}_${idx}"          $task $idx --policy h2o --obs-window 64 --n-sink 0 --k-nominal $H2OK
  done
done
echo LB_NATIVE_DONE
