#!/bin/bash
# 64K run with no budget (K = n_ctx) for every policy, to see what each one drops
# on its own.
# All policies use an f16 cache: the per-head evictors run FA-off to read attention
# weights, and llama.cpp needs flash attention for a quantized V. So these numbers
# are not directly comparable with the q8_0 64K matrix. 512 tokens are generated
# because FA-off decode at 64K is slow, so wall time compares only within this table.
set -u
cd /home/mislam22/EndurKV_workspace
B=EndurKV/entropy_probe/build-pc-cuda/eviction_bench
P=EndurKV/benchmarks/ctx_sweep/llama1b_57344tok.txt
# Eval slice disjoint from the 57344-token prompt. wiki_eval_disjoint_long.txt overlaps it.
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
