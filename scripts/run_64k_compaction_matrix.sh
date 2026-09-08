#!/bin/bash
# ============================================================================
# run_64k_compaction_matrix.sh -- the full 64K three-way compaction comparison.
# (2026-08-07, RTX 4500 Ada)
#
# WHAT THIS ANSWERS. muKV's speedup has always been reported as one number, but it
# comes from TWO mechanisms that are easy to conflate:
#   (a) EVICTION      -- seq_rm marks cells free. It reduces what is RETAINED, but
#                        the survivors do not move, so attention still scans to the
#                        highest occupied index and decode stays O(N). On its own
#                        this is worth ~1.03x.
#   (b) COMPACTION    -- the survivors are made CONTIGUOUS, so decode becomes O(K).
#                        This is where the speedup actually lives.
# Every arm below therefore reports both, and the table states which one ran.
#
# TWO COMPACTION MODES, which is the point of this matrix:
#   roundtrip : llama_state_seq_get_data -> a SECOND context -> _set_data. Correct
#               and quality-neutral, but peak memory is 2x the cache. That 2x is what
#               makes Phi-3 infeasible at 16K on the phone and bounds 64K everywhere.
#   inplace   : endurkv_compact_seq() slides the survivors down into a dense prefix
#               INSIDE the tensors prefill already allocated, in chunks. Peak memory
#               does not rise at all. Added 2026-08-07 precisely to remove that bound.
# They must produce IDENTICAL keep-sets and therefore identical PPL. The table exists
# so that can be checked rather than asserted.
#
# K IS SWEPT, INCLUDING UNLIMITED. --k-nominal 65536 lets the mass gate decide the
# budget by itself; the arm is here to show that at 64K the gate does NOT self-limit
# (it keeps ~93% of cells), i.e. K is doing the work, not the gate.
#
# BOTH MODES PER ARM. gen (4096 tokens) gives prefill/decode/wall/tps/memory; ppl on a
# DISJOINT WikiText slice gives quality. The slice must not overlap the prompt -- an
# earlier version did, which turned a prediction test into a recall test and produced
# a meaningless vanilla PPL of 1.03. See benchmarks/ppl/README.md.
# ============================================================================
set -u
cd /home/mislam22/EndurKV_workspace
B=EndurKV/entropy_probe/build-pc-cuda/eviction_bench
P=EndurKV/benchmarks/ctx_sweep/llama1b_57344tok.txt
# CHANGED 2026-08-13: was wiki_eval_disjoint_long.txt, which is NOT disjoint from this
# 57344-token prompt -- 70/119 200-char and 233/399 60-char windows appear verbatim in it.
# The "_long" slice was verified disjoint against the 12K phone prompt, and that property
# was assumed to carry over to the 64K prompt. It does not: the 64K prompt is 251502 chars
# of the same WikiText stream and simply reaches far enough to swallow the slice. Every
# 64K PPL number measured before this date is therefore part recall, not prediction, and
# is void. wiki_eval_disjoint_64k.txt is raw[252000:276000], re-asserted at 0/119 and
# 0/399 against this prompt on 2026-08-13.
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
