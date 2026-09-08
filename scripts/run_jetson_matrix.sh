#!/bin/bash
# ============================================================================
# run_jetson_matrix.sh -- Jetson Orin NX, same protocol as the phone. (2026-08-05)
#
# WHY THE JETSON MATTERS. The two closest related systems -- KVSwap (MobiSys) and
# MobiLoRA (ACL, with Honor) -- are both branded "mobile" but evaluate on Jetson
# Orin boards, not phones. Measuring muKV on an Orin lets us compare against them
# on THEIR hardware class instead of arguing across devices, and it separates two
# things our phone results conflate: what is due to being memory-bound on a small
# device, and what is due to the Adreno/Vulkan backend specifically.
#
# PROTOCOL IS THE PHONE'S, deliberately: 12K WikiText prompt + 4096 generated,
# ctx 16384, batch 1, greedy, seed 42, f16 K/V for every policy. Identical inputs
# are what make the two devices comparable at all.
#
# f16, NOT q8_0. Quantized KV is numerically broken on the phone's Adreno build
# (random tokens; NLL above ln(vocab)). The Jetson is CUDA and q8_0 is fine there,
# but using f16 keeps this table on the same footing as the corrected phone table
# and as LongBench -- and it is the only format all policies can share, since a
# per-head evictor needs FA-off and llama.cpp requires FA for a quantized V.
#
# CTXJ/PROMPTJ let the context be reduced when the board is co-tenanted. A
# co-tenant grew to 9.7 GB mid-setup and a 512 MiB KV allocation began failing
# outright (NvMapMemAllocInternalTagged error 12), so 16K is not always feasible.
# An 8K run is not directly comparable to the phone's 16K table and must be
# labelled as such -- but the POLICY ORDERING on Orin-class hardware is still
# the point, since that is the class KVSwap and MobiLoRA evaluate on.
#
# LLAMA-1B ONLY. A co-tenant process holds ~8.5 GB of the board's 15 GB. Phi-3
# needs 2.4 GB of weights plus 6.4 GB of f16 KV at 16K -- and compaction transiently
# needs a SECOND full context -- so it cannot fit in the ~5 GB available. Running it
# anyway would either OOM or silently fall back to the sparse path, which is exactly
# the failure that produced a bogus 13x on the RTX. Phi-3 is deferred, not dropped.
# ============================================================================
set -u
H=orin-nx
RB=/home/romyull/ukv/code/entropy_probe/build-jetson-cuda/eviction_bench
RM=/home/romyull/ukv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
RP=${RPJ:-/home/romyull/ukv/prompt_12k.txt}
OUT_HOST=/tmp/jetson_matrix; mkdir -p "$OUT_HOST"
ROUT=/home/romyull/ukv/logs/jm_$(date +%Y%m%d_%H%M%S)
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

ssh -o BatchMode=yes $H "mkdir -p $ROUT" 2>/dev/null
# push the SAME prompt the phone used, so the workloads are identical
scp -o BatchMode=yes ${PROMPTJ:-/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/ctx_sweep/llama1b_12288tok.txt} $H:$RP >/dev/null 2>&1

flags_for(){ case "$1" in
  vanilla)    echo "--policy vanilla" ;;
  mukv_dfg)   echo "$MU --force-defrag" ;;
  mukv_nodfg) echo "$MU --no-defrag" ;;
  # WikiText has no published SnapKV setting, so the baseline gets the shipped
  # FasterDecoding default (window 32, avgpool-5) -- verified from snapkv_utils.py.
  snapkv)     echo "--policy snapkv --obs-window 32 --snapkv-kernel 5 --n-sink 0" ;;
esac; }

cell(){ local POL=$1; local TAG="llama1b_$POL"
  [ -f "$OUT_HOST/$TAG/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$OUT_HOST/$TAG"
  echo "[$(date +%H:%M:%S)] $TAG ..."
  ssh -o BatchMode=yes $H "mkdir -p $ROUT/$TAG; cd /home/romyull/ukv && timeout ${TMO:-9000} $RB \
    --prompt $RP --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos \
    --ctx-size ${CTXJ:-16384} --model $RM --seed 42 --threads 6 --n-gpu-layers 99 --greedy \
    --k-nominal 1024 --cache-type-k f16 --cache-type-v f16 $(flags_for $POL) \
    --out-meta $ROUT/$TAG/meta.json --out-gen $ROUT/$TAG/gen.txt --out-csv /dev/null \
    > $ROUT/$TAG/out 2> $ROUT/$TAG/err" 2>/dev/null
  scp -o BatchMode=yes -r $H:$ROUT/$TAG/* "$OUT_HOST/$TAG/" >/dev/null 2>&1
  python3 - "$OUT_HOST/$TAG" "$TAG" <<'PY'
import json,re,sys,os
f=os.path.join(sys.argv[1],'meta.json')
if not os.path.exists(f):
    e=os.path.join(sys.argv[1],'err')
    print("  [%-20s] FAILED %s"%(sys.argv[2], open(e,errors='replace').read().strip().split('\n')[-1][:70] if os.path.exists(e) else '?')); raise SystemExit
s=open(f).read(); s=re.sub(r':\s*-?nan\b',': NaN',s); s=re.sub(r':\s*-?inf\b',': Infinity',s); j=json.loads(s)
print("  [%-20s] prefill=%7.1fs decode=%7.1fs wall=%7.1fs tps=%7.2f cells=%6.0f compact=%s"%(
  sys.argv[2],j['prefill_ms']/1000,j['decode_ms']/1000,j['total_ms']/1000,j.get('decode_tps') or 0,
  j['retained_kv_bytes']/(16*8*64*2*2.0), j.get('compaction_applied')))
PY
}
# correctness gate FIRST: a timed run whose output is wrong is worthless, and that
# is exactly how the phone GPU q8_0 pass wasted a full campaign.
echo "=== correctness check (expected: Miller v. California) ==="
scp -o BatchMode=yes /home/mislam22/EndurKV_workspace/EndurKV/benchmarks/longbench/hotpotqa/trunc_16384/llama1b/prompt_000.txt $H:/home/romyull/ukv/sanity.txt >/dev/null 2>&1
ssh -o BatchMode=yes $H "cd /home/romyull/ukv && $RB --prompt sanity.txt --prompt-id sanity --eval-mode gen \
  --max-tokens 24 --ctx-size ${CTXJ:-16384} --model $RM --seed 42 --threads 6 --n-gpu-layers 99 --greedy \
  --policy vanilla --cache-type-k f16 --cache-type-v f16 --out-meta /tmp/s.json --out-gen /dev/stdout \
  --out-csv /dev/null 2>/dev/null" 2>/dev/null | head -2

for POL in vanilla mukv_dfg mukv_nodfg snapkv; do cell "$POL"; done
echo "JETSON_MATRIX_DONE -> $OUT_HOST"
