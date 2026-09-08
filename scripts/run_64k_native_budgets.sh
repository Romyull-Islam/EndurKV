#!/bin/bash
# ============================================================================
# run_64k_native_budgets.sh -- each baseline at ITS OWN published budget rule.
# (2026-08-07)
#
# CORRECTS A BROKEN EXPERIMENT. run_64k_nolimit.sh forced --k-nominal 65536 on every
# policy to ask "what does it compress on its own?". That was wrong: for these methods
# the BUDGET RULE *is* the policy, so overriding K did not reveal their behaviour, it
# deleted it. Every baseline dutifully reported 100% retention -- an artifact of the
# harness, not a property of SnapKV/H2O/TOVA/StreamingLLM. This script instead gives
# each method the budget its own paper/repo specifies.
#
# WHERE EACH DEFAULT COMES FROM:
#   SnapKV   max_capacity_prompt=2048, window=32, kernel=5, avgpool.
#            From init_snapkv() in snapkv_utils.py (archived at
#            benchmarks/snapkv_utils_reference.py) -- the integration path their
#            harness actually runs. NOTE the SnapKVCluster CLASS default is different
#            (window=64, max_capacity=320); the integration default is the operative
#            one. ABSOLUTE, does not scale with prompt length.
#   Ada-KV   same total budget as SnapKV (2048), allocated adaptively ACROSS heads
#            instead of uniformly. ABSOLUTE.
#   H2O      budget is a RATIO of sequence length: 20% total (10% heavy hitters +
#            10% recent) is the paper's standard setting. This is the ONLY baseline
#            here whose budget scales with context -- the distinction the no-limit
#            run was trying and failing to expose.
#   TOVA     fixed multi-state size; 2048 is a representative published operating
#            point (their paper sweeps cache size). ABSOLUTE.
#   StreamingLLM  start_size=4 sinks + a fixed recent window; 2048 total. ABSOLUTE
#            and purely structural -- no attention scoring at all.
#   muKV     reported twice: at NO limit (K=n_ctx, so only its mass gate acts) and at
#            its frozen deployment budget K=1024, so the comparison shows both what it
#            does unconstrained and what it does as configured.
#
# f16 EVERYWHERE: per-head evictors need FA-off to read attention weights and
# llama.cpp requires flash-attention for a quantized V (SnapKV core-dumps at q8_0).
# 512 generated tokens: FA-off decode at 64K is slow; tok/s is a rate so it is
# unaffected, and wall-clock is comparable within this table only.
# ============================================================================
set -u
cd /home/mislam22/EndurKV_workspace
B=EndurKV/entropy_probe/build-pc-cuda/eviction_bench
P=EndurKV/benchmarks/ctx_sweep/llama1b_57344tok.txt
# CHANGED 2026-08-13: was wiki_eval_disjoint_long.txt, which overlaps this 57344-token
# prompt 70/119 (200-char windows). See run_64k_compaction_matrix.sh for the full note.
# All 64K PPL measured before this date is void.
E=EndurKV/benchmarks/ppl/wiki_eval_disjoint_64k.txt
M=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
O=/tmp/rtx64k_native; mkdir -p $O
# H2O: 20% of the 57118-token prompt
H2OK=11424

spec(){ case "$1" in
  vanilla)      echo "--policy vanilla --k-nominal 65536" ;;
  mukv_nolimit) echo "$MU --compact-inplace --k-nominal 65536" ;;
  mukv_k1024)   echo "$MU --compact-inplace --k-nominal 1024" ;;
  snapkv)       echo "--policy snapkv --obs-window 32 --snapkv-kernel 5 --n-sink 0 --k-nominal 2048" ;;
  adakv)        echo "--policy adakv  --obs-window 32 --n-sink 0 --k-nominal 2048" ;;
  h2o)          echo "--policy h2o    --obs-window 64 --n-sink 0 --k-nominal $H2OK" ;;
  tova)         echo "--policy tova   --k-nominal 2048" ;;
  streamingllm) echo "--policy streamingllm --n-sink 4 --k-nominal 2048" ;;
esac; }

for POL in vanilla mukv_nolimit mukv_k1024 snapkv adakv h2o tova streamingllm; do
  for MODE in gen ppl; do
    D=$O/${MODE}_$POL; [ -f $D/meta.json ] && { echo "  [$MODE/$POL] cached"; continue; }
    mkdir -p $D
    EX="--eval-mode gen --max-tokens 512 --ignore-eos"
    [ $MODE = ppl ] && EX="--eval-mode ppl --eval-text $E"
    timeout 5400 $B --prompt $P --prompt-id nb_$POL $EX --ctx-size 65536 --model $M \
      --seed 42 --threads 8 --n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 \
      $(spec $POL) --out-meta $D/meta.json --out-gen /dev/null --out-csv /dev/null \
      >/dev/null 2>$D/err
    echo "  [$MODE/$POL] $([ -f $D/meta.json ] && echo ok || echo "FAILED: $(tail -1 $D/err|cut -c1-60)")"
  done
done
echo NATIVE_DONE
