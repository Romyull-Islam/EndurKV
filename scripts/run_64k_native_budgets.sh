#!/bin/bash
# 64K-context run (Llama-3.2-1B, RTX) with each baseline at its own published budget rule:
#   SnapKV 2048 (window 32, kernel 5, avgpool, from init_snapkv() in snapkv_utils.py)
#   Ada-KV 2048 total across heads, TOVA 2048, StreamingLLM 4 sinks + recent = 2048
#   H2O 20% of the prompt (10% heavy + 10% recent), muKV at K=n_ctx and at K=1024.
# f16 K/V everywhere: per-head evictors need FA-off, and quantized V needs FA in llama.cpp.
# 512 generated tokens because FA-off decode at 64K is slow, wall time compares within this table only.
set -u
cd /home/mislam22/EndurKV_workspace
B=EndurKV/entropy_probe/build-pc-cuda/eviction_bench
P=EndurKV/benchmarks/ctx_sweep/llama1b_57344tok.txt
# PPL eval text must not overlap the 57344-token prompt (wiki_eval_disjoint_long.txt does).
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
