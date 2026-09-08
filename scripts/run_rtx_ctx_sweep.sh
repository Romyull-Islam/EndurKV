#!/bin/bash
# ============================================================================
# run_rtx_ctx_sweep.sh -- RTX 4500 Ada context sweep + compaction A/B (2026-08-02)
#
# WHY THIS EXISTS. Reviewers (and our own supervisor) ask why server KV papers
# report 2-4x while our phone numbers are 1.1-2x. The answer is the decode
# roofline, and this experiment tests it rather than asserting it.
#
# Decode is memory-bandwidth-bound: every step reads ALL weights once plus the
# live KV. So the speedup ceiling from compressing KV by a factor r is
#       S = (W + KV) / (W + KV/r)
# and KV grows with context while W does not. The prediction is therefore sharp:
# S must RISE with context length, approaching r asymptotically. If our measured
# S tracks that curve, the modest phone numbers are explained by KV share -- not
# by a weak implementation -- and the same policy would show server-class
# speedups at server context lengths. If it does NOT track, the roofline story
# in the paper is wrong and we need to know before a reviewer finds out.
#
# Phi-3 is the vehicle: 32 layers, no GQA, so it has the largest KV per token of
# our models (~208 KB/token at q8_0) and reaches interesting KV shares within
# 24 GB. Llama-1B is included at the low end as the contrast case (23% KV share
# at 16K -> ceiling only ~1.26x).
#
# Also runs the compaction A/B at each point. On the phone, muKV WITHOUT
# compaction measured SLOWER than vanilla (30.50 vs 33.99 tps) because seq_rm
# frees cells without moving survivors, so attention still scans to the highest
# occupied index and a 10x smaller cache buys nothing. We assert that this is a
# backend-independent property of the mechanism, so it must reproduce on CUDA.
#
# Decode length is fixed at 4096 tokens at every context so the fixed
# post-prefill cost (~2.9s vanilla / ~0.2s muKV on Adreno) is amortised the same
# way everywhere -- that cost is what inflated the earlier 256-token runs to a
# spurious 2.12x, and holding it constant is what makes these points comparable.
# ============================================================================
set -u
B=/home/mislam22/EndurKV_workspace/EndurKV/entropy_probe/build-pc-cuda/eviction_bench
P=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/decode_16k
OUT=${OUT:-/tmp/rtx_ctx_sweep}; mkdir -p "$OUT"
GEN=${GEN:-4096}
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
declare -A M=( [llama1b]=/home/mislam22/EndurKV_workspace/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
               [phi3]=/home/mislam22/EndurKV_workspace/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf )

# Prompts are token-EXACT WikiText slices (benchmarks/ctx_sweep, built and verified
# with each model's own tokenizer). Prompt length must scale WITH ctx: reusing a
# 16K prompt at ctx 65536 would leave the live cache at ~20K cells and the sweep
# would measure nothing, since KV share depends on cells actually occupied, not on
# the allocation. prompt + 4096 generated = the target cell count at each point.
prompt_for(){ echo "/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/ctx_sweep/$1_${2}tok.txt"; }

cell(){ local TAG=$1 MT=$2 CTX=$3 PT=$4; shift 4; local D=$OUT/$TAG
  [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  timeout ${TMO:-3600} $B --prompt "$(prompt_for $MT $PT)" --prompt-id "$TAG" \
    --eval-mode gen --max-tokens $GEN --ignore-eos --ctx-size $CTX \
    --model "${M[$MT]}" --seed 42 --threads 8 --n-gpu-layers 99 --greedy \
    --k-nominal ${KBUD:-1024} --cache-type-k q8_0 --cache-type-v q8_0 "$@" \
    --out-meta "$D/meta.json" --out-gen /dev/null --out-csv /dev/null > "$D/out" 2> "$D/err"
  python3 - "$D" "$TAG" <<'PY'
import json,re,sys,os
f=os.path.join(sys.argv[1],'meta.json')
if not os.path.exists(f):
    e=os.path.join(sys.argv[1],'err')
    print("  [%-26s] FAILED %s"%(sys.argv[2], open(e,errors='replace').read().strip().split('\n')[-1][:60] if os.path.exists(e) else '?')); raise SystemExit
s=open(f).read(); s=re.sub(r':\s*-?nan\b',': NaN',s); s=re.sub(r':\s*-?inf\b',': Infinity',s); j=json.loads(s)
print("  [%-26s] ctx=%6d prefill=%7.1fs decode=%7.1fs tps=%7.2f ret=%8.1fMiB cells=%6d compact=%s"%(
  sys.argv[2], j['ctx_size'], j['prefill_ms']/1000, j['decode_ms']/1000, j.get('decode_tps') or 0,
  (j.get('retained_kv_bytes') or 0)/1048576.0, j.get('peak_kv_cells') or 0, j.get('compaction_applied')))
PY
}

# ctx : prompt tokens  (prompt + 4096 generated = live cells at that point)
for MT in phi3 llama1b; do
  for pair in 8192:4096 16384:12288 32768:28672 65536:57344; do
    CTX=${pair%%:*}; PT=${pair##*:}
    echo "[$(date +%H:%M:%S)] === $MT ctx=$CTX prompt=${PT}tok (gen $GEN) ==="
    cell ${MT}_ctx${CTX}_vanilla    $MT $CTX $PT --policy vanilla
    cell ${MT}_ctx${CTX}_mukv       $MT $CTX $PT $MU --force-defrag
    cell ${MT}_ctx${CTX}_mukv_nodfg $MT $CTX $PT $MU --no-defrag
  done
done
echo "RTX_CTX_SWEEP_DONE -> $OUT"
