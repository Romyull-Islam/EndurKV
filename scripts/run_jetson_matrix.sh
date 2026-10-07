#!/bin/bash
# Jetson Orin NX matrix with the phone protocol: 12K WikiText prompt + 4096
# generated tokens, ctx 16384, greedy, seed 42, Llama-3.2-1B. KVSwap and
# MobiLoRA evaluate on Orin boards, so this compares on their hardware class.
# f16 K/V for every policy: per-head evictors need FA-off, and llama.cpp needs
# FA for a quantized V. CTXJ/PROMPTJ shrink the context when other jobs on the
# board leave too little memory. Label such runs, they do not match the 16K table.
# Phi-3 does not fit next to the other jobs, since compaction briefly needs a
# second full context.
set -u
H=orin-nx
RB=/home/romyull/ukv/code/entropy_probe/build-jetson-cuda/eviction_bench
RM=/home/romyull/ukv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
RP=${RPJ:-/home/romyull/ukv/prompt_12k.txt}
OUT_HOST=/tmp/jetson_matrix; mkdir -p "$OUT_HOST"
ROUT=/home/romyull/ukv/logs/jm_$(date +%Y%m%d_%H%M%S)
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

ssh -o BatchMode=yes $H "mkdir -p $ROUT" 2>/dev/null
# Push the same prompt the phone used.
scp -o BatchMode=yes ${PROMPTJ:-/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/ctx_sweep/llama1b_12288tok.txt} $H:$RP >/dev/null 2>&1

flags_for(){ case "$1" in
  vanilla)    echo "--policy vanilla" ;;
  mukv_dfg)   echo "$MU --force-defrag" ;;
  mukv_nodfg) echo "$MU --no-defrag" ;;
  # SnapKV has no published WikiText setting, so use the FasterDecoding default
  # (window 32, avgpool 5) from snapkv_utils.py.
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
# Check output correctness before any timed run.
echo "correctness check (expected: Miller v. California)"
scp -o BatchMode=yes /home/mislam22/EndurKV_workspace/EndurKV/benchmarks/longbench/hotpotqa/trunc_16384/llama1b/prompt_000.txt $H:/home/romyull/ukv/sanity.txt >/dev/null 2>&1
ssh -o BatchMode=yes $H "cd /home/romyull/ukv && $RB --prompt sanity.txt --prompt-id sanity --eval-mode gen \
  --max-tokens 24 --ctx-size ${CTXJ:-16384} --model $RM --seed 42 --threads 6 --n-gpu-layers 99 --greedy \
  --policy vanilla --cache-type-k f16 --cache-type-v f16 --out-meta /tmp/s.json --out-gen /dev/stdout \
  --out-csv /dev/null 2>/dev/null" 2>/dev/null | head -2

for POL in vanilla mukv_dfg mukv_nodfg snapkv; do cell "$POL"; done
echo "JETSON_MATRIX_DONE -> $OUT_HOST"
