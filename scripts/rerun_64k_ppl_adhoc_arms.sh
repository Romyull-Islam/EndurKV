#!/bin/bash
# rerun_64k_ppl_adhoc_arms.sh: disjoint-text PPL for three 64K arms in /tmp/rtx64k_native
# that no campaign script loops over, so rerun_64k_ppl_clean.sh misses them:
#   streamingllm_fa  StreamingLLM, flash-attention on, not compacted
#   sllm_compact     StreamingLLM, flash-attention on, compacted
#   mukv_matched     muKV at K=2542, about the 2048 cells StreamingLLM keeps
# The plain streamingllm arm in that dir ran on an older build with flash-attention off,
# so it is a different build from these rows.
set -u
cd /home/mislam22/EndurKV_workspace || exit 1
B=EndurKV/entropy_probe/build-pc-cuda/eviction_bench
P=EndurKV/benchmarks/ctx_sweep/llama1b_57344tok.txt
E=EndurKV/benchmarks/ppl/wiki_eval_disjoint_64k.txt   # 0/119 and 0/399 vs P, asserted 08-13
M=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
O=/tmp/rtx64k_native

# Flags taken from each arm's meta.json and its llama_context flash_attn line.
spec(){ case "$1" in
  streamingllm_fa) echo "--policy streamingllm --n-sink 4 --k-nominal 2048 --fa-on-evict" ;;
  sllm_compact)    echo "--policy streamingllm --n-sink 4 --k-nominal 2048 --fa-on-evict --compact-inplace" ;;
  mukv_matched)    echo "$MU --k-nominal 2542 --compact-inplace" ;;
esac; }

for POL in streamingllm_fa sllm_compact mukv_matched; do
  D=$O/ppl_$POL
  [ -f $D/meta.json ] && { echo "  [ppl/$POL] cached"; continue; }
  mkdir -p $D
  timeout 3600 $B --prompt $P --prompt-id ah_$POL --eval-mode ppl --eval-text $E \
    --ctx-size 65536 --model $M --seed 42 --threads 8 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 $(spec $POL) \
    --out-meta $D/meta.json --out-gen /dev/null --out-csv /dev/null >/dev/null 2>$D/err
  echo "  [ppl/$POL] $([ -f $D/meta.json ] && echo ok || echo "FAILED: $(tail -1 $D/err|cut -c1-60)")"
done
echo ADHOC_PPL_DONE
