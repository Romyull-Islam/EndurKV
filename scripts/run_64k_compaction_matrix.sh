#!/bin/bash
# 64K-context compaction matrix on the desktop GPU (Llama-3.2-1B, CUDA build).
# Eviction alone (seq_rm) leaves survivors in place, so decode still scans to the
# highest occupied cell. Compaction makes them contiguous. Two compaction modes:
#   roundtrip: state get/set through a second context, peak memory 2x the cache
#   inplace:   endurkv_compact_seq() packs survivors inside the existing tensors
# Both should give identical keep-sets and PPL. K=65536 leaves the budget to the
# mass gate alone. Each arm runs gen (4096 tokens) and ppl on a WikiText slice
# disjoint from the prompt (see benchmarks/ppl/README.md).
set -u
cd /home/mislam22/EndurKV_workspace
B=EndurKV/entropy_probe/build-pc-cuda/eviction_bench
P=EndurKV/benchmarks/ctx_sweep/llama1b_57344tok.txt
# Eval text is raw[252000:276000], disjoint from this 57344-token prompt.
# wiki_eval_disjoint_long.txt overlaps this prompt and must not be used here.
E=EndurKV/benchmarks/ppl/wiki_eval_disjoint_64k.txt
M=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
OUT=${OUT:-/tmp/rtx64k_matrix}; mkdir -p $OUT

# arm := tag:kflags:compaction-flag
ARMS="
vanilla:--policy vanilla:--no-defrag
mukv_k1024_none:$MU --k-nominal 1024:--no-defrag
mukv_k1024_rt:$MU --k-nominal 1024:--force-defrag
mukv_k1024_ip:$MU --k-nominal 1024:--compact-inplace
mukv_k8192_none:$MU --k-nominal 8192:--no-defrag
mukv_k8192_rt:$MU --k-nominal 8192:--force-defrag
mukv_k8192_ip:$MU --k-nominal 8192:--compact-inplace
mukv_kfree_none:$MU --k-nominal 65536:--no-defrag
mukv_kfree_ip:$MU --k-nominal 65536:--compact-inplace
"

run(){ # tag polflags cflag mode
  local TAG=$1 POL=$2 CF=$3 MODE=$4
  local D=$OUT/${MODE}_$TAG
  [ -f "$D/meta.json" ] && { echo "  [$MODE/$TAG] cached"; return; }
  mkdir -p $D
  local EXTRA="--eval-mode gen --max-tokens 4096 --ignore-eos"
  [ "$MODE" = "ppl" ] && EXTRA="--eval-mode ppl --eval-text $E"
  timeout 3000 $B --prompt $P --prompt-id $TAG $EXTRA --ctx-size 65536 \
    --model $M --seed 42 --threads 8 --n-gpu-layers 99 --greedy \
    --cache-type-k q8_0 --cache-type-v q8_0 $POL $CF \
    --out-meta $D/meta.json --out-gen $D/gen.txt --out-csv /dev/null \
    > $D/out 2> $D/err
  [ -f "$D/meta.json" ] && echo "  [$MODE/$TAG] ok" || echo "  [$MODE/$TAG] FAILED: $(tail -1 $D/err | cut -c1-70)"
}

echo "$ARMS" | while IFS=: read -r TAG POL CF; do
  [ -z "$TAG" ] && continue
  for MODE in gen ppl; do run "$TAG" "$POL" "$CF" "$MODE"; done
done
echo "MATRIX_DONE -> $OUT"
