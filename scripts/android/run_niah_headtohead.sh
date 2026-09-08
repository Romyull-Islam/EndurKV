#!/bin/bash
# ============================================================================
# NIAH quality head-to-head: muKV vs canonical SnapKV vs vanilla  (HotMobile)
# ----------------------------------------------------------------------------
# The defensibility test. On llama.cpp's sequence-level engine canonical SnapKV
# cannot compact (per-head union -> near-full), so it "retrieves" trivially by
# keeping almost everything. muKV compacts to a real budget and must KEEP the
# needle via its demand-aware selection + alpha-gate. So we report BOTH:
#     (1) needle hit  (gen contains "mango sorbet" or "bi-rite", case-insensitive)
#     (2) physical live cells after prefill eviction  (peak_kv - evicted_prefill)
# The story is accuracy AT a given physical cache size -- muKV hits at ~K cells,
# SnapKV only hits because it stays near-full (no efficiency).
#
# TIGHT budget K=256 (stress the gate) over the 14 mango/Bi-Rite needle files
# (4K & 8K contexts x 7 depths). CPU, 64 greedy tokens, no cool-gate (retrieval
# correctness is thermally invariant). Vanilla = full-cache accuracy ceiling.
#
# Canonical SnapKV: --policy snapkv --obs-window 64 --n-sink 0 (exact defaults).
# muKV (CPU state-swap): --policy v1_fa2 --n-sink 4 --adaptive-anchor
#   --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70
# ============================================================================
set -u; export ANDROID_ADB_SERVER_PORT=5151
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

KBUD=${1:-256}                      # tight budget (override: run_niah_headtohead.sh 512)
OUT_HOST=/tmp/niah_h2h_k${KBUD}; mkdir -p "$OUT_HOST"
TS=$(date +%Y%m%d_%H%M%S); OUT=/data/local/tmp/endurkv/logs/niah_h2h_$TS
adb_safe_shell "mkdir -p $OUT" < /dev/null
NIAH_SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/niah
MODEL=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
CB=/data/local/tmp/endurkv/bin_cpu_sol2

# push the 14 needle stimuli
for f in "$NIAH_SRC"/niah_L*_n0.txt; do adb push "$f" "$OUT/$(basename "$f")" < /dev/null >/dev/null 2>&1; done
STIMS=$(cd "$NIAH_SRC" && ls niah_L*_n0.txt)

# cell <policy-tag> <stim> <ctx> <policy-args...>
cell(){ local POL=$1 STIM=$2 CTX=$3; shift 3; local id="${POL}__${STIM%.txt}"; local PD=$OUT/$id
  adb_safe_shell "mkdir -p $PD" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$CB timeout 240 $CB/eviction_bench --prompt $OUT/$STIM --prompt-id $id --eval-mode gen \
    --max-tokens 64 --ignore-eos --ctx-size $CTX --model $MODEL --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --k-nominal $KBUD $* --out-meta $PD/meta.json --out-csv /dev/null --out-gen $PD/gen.txt >/dev/null 2>&1" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
}
MU="--policy v1_fa2 --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

i=0; n=$(echo "$STIMS" | wc -w)
for STIM in $STIMS; do
  i=$((i+1))
  case "$STIM" in *_L8K_*) CTX=8192;; *) CTX=4096;; esac
  echo "[$(date +%H:%M:%S)] ($i/$n) $STIM ctx=$CTX"
  cell vanilla "$STIM" "$CTX" --policy vanilla
  cell snapkv  "$STIM" "$CTX" --policy snapkv --obs-window 64 --n-sink 0
  cell mukv    "$STIM" "$CTX" $MU
done
touch /tmp/niah_h2h_DONE; echo "[$(date +%H:%M:%S)] NIAH H2H DONE (K=$KBUD) -> $OUT_HOST"

# ---- host-side scoring: needle hit + live cells, per policy ----
python3 - "$OUT_HOST" <<'PY'
import json,re,sys,glob,os
root=sys.argv[1]
def live(meta):
    try:
        d=json.loads(re.sub(r'\bnan\b','null',open(meta).read()))
        pk=d.get('peak_kv_cells');ev=d.get('evicted_prefill')
        return pk, (pk-ev if pk is not None and ev is not None else None)
    except: return None,None
agg={}
for d in sorted(glob.glob(os.path.join(root,'*__*'))):
    base=os.path.basename(d); pol=base.split('__')[0]
    gen=os.path.join(d,'gen.txt'); meta=os.path.join(d,'meta.json')
    txt=open(gen).read().lower() if os.path.exists(gen) else ''
    hit=int(('mango sorbet' in txt) or ('bi-rite' in txt) or ('bi rite' in txt))
    pk,lv=live(meta)
    a=agg.setdefault(pol,{'hit':0,'n':0,'live':[]})
    a['hit']+=hit; a['n']+=1
    if lv is not None: a['live'].append(lv)
print(f"\n{'policy':<10}{'NIAH hit':<12}{'mean live cells':<18}{'note'}")
for pol in ['vanilla','snapkv','mukv']:
    if pol not in agg: continue
    a=agg[pol]; ml=sum(a['live'])/len(a['live']) if a['live'] else float('nan')
    note={'vanilla':'full-cache ceiling','snapkv':'per-head union (near-full=no compaction)','mukv':'compacted to budget'}[pol]
    print(f"{pol:<10}{a['hit']}/{a['n']:<10}{ml:<18.0f}{note}")
PY
