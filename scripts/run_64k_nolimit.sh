#!/bin/bash
# ============================================================================
# run_64k_nolimit.sh -- "does the policy compress ANYTHING on its own?" (2026-08-07)
#
# THE QUESTION. Every eviction policy here is usually run with a budget K, and the
# resulting compression is then reported as the policy's achievement. But K is an
# INPUT. This experiment removes it: give every policy K = n_ctx (no limit) and see
# what each one still chooses to drop. Whatever survives that is the policy's own
# intrinsic compression; everything else was the budget doing the work.
#
# f16 FOR EVERY POLICY, and this is forced, not preferred. The per-head evictors
# (SnapKV, Ada-KV, H2O, TOVA) must run FA-OFF to read attention weights, and
# llama.cpp requires flash-attention for a quantized V -- SnapKV core-dumps inside
# llama_decode at q8_0. So the whole table runs f16 or the ratios divide a valid run
# by a broken one, which is exactly how an earlier phone-GPU campaign was wasted.
# NOTE: this makes the numbers here NOT directly comparable to the q8_0 64K matrix.
#
# 512 GENERATED TOKENS, not 4096. FA-off decode at 64K is slow enough that 4096 tokens
# across six arms is hours. decode_tps is a rate so it is unaffected; the wall-clock
# column is only comparable WITHIN this table, where every arm generates the same 512.
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
O=/tmp/rtx64k_nolimit; mkdir -p $O

flags(){ case "$1" in
  vanilla)      echo "--policy vanilla" ;;
  mukv)         echo "$MU --compact-inplace" ;;
  snapkv)       echo "--policy snapkv --obs-window 32 --snapkv-kernel 5 --n-sink 0" ;;
  adakv)        echo "--policy adakv --n-sink 0 --obs-window 32" ;;
  h2o)          echo "--policy h2o --n-sink 0 --obs-window 64" ;;
  tova)         echo "--policy tova" ;;
  streamingllm) echo "--policy streamingllm --n-sink 4" ;;
esac; }

for POL in vanilla mukv snapkv adakv h2o tova streamingllm; do
  for MODE in gen ppl; do
    D=$O/${MODE}_$POL; [ -f $D/meta.json ] && { echo "  [$MODE/$POL] cached"; continue; }
    mkdir -p $D
    EX="--eval-mode gen --max-tokens 512 --ignore-eos"
    [ $MODE = ppl ] && EX="--eval-mode ppl --eval-text $E"
    timeout 5400 $B --prompt $P --prompt-id nl_$POL $EX --ctx-size 65536 --model $M \
      --seed 42 --threads 8 --n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 \
      --k-nominal 65536 $(flags $POL) \
      --out-meta $D/meta.json --out-gen /dev/null --out-csv /dev/null >/dev/null 2>$D/err
    echo "  [$MODE/$POL] $([ -f $D/meta.json ] && echo ok || echo "FAILED: $(tail -1 $D/err|cut -c1-60)")"
  done
done
echo NOLIMIT_DONE
