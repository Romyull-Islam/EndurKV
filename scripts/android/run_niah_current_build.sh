#!/bin/bash
# ============================================================================
# run_niah_current_build.sh -- rebuild the NIAH grid on ONE build. (2026-08-22)
#
# WHY. The 7-policy NIAH grid (June, /tmp/phone_niah_full) and the 4-policy grid
# (August, /tmp/niah_vs_sllm) DISAGREE ON VANILLA: 13/14 vs 14/14, same stimuli,
# same scorer, differing only at L8K@17%. So they cannot be merged, and the
# "*vanilla misses it too" annotation -- which was the basis for calling that
# depth a model limit and for saying muKV-mass exceeds the ceiling -- does not
# hold in the newer build. The June grid also records peak_kv_cells rather than
# retained_kv_bytes, so it cannot carry retention at all.
#
# This run puts the missing policies into the CURRENT build so one table is one
# build, with live-cell retention on every row.
#
# CONFIGURATIONS. Each baseline as its own paper describes, per KeyDiff App. A.1:
#   SnapKV / Ada-KV  score ONCE at end of prefill from a windowed side matmul,
#                    so they run FA-ON via the side node with their own windows.
#   TOVA / H2O       score at EVERY decode step, which the side node cannot serve
#                    (it fires only when q->ne[1] > 1), so they stay FA-off. That
#                    is a property of their designs, not a handicap we impose.
# Budget K=1024 for these four -- their papers specify no NIAH budget, and matching
# muKV's nominal budget makes any retention difference attributable to
# realizability, which is the effect under test.
#
# KeyDiff additionally runs at 6144, its OWN published NIAH setting (their Fig. 6
# uses a 6K budget with B=128). On these stimuli (3122 / 6099 tokens) 6144 never
# evicts, so that row is expected to equal vanilla -- it is included precisely to
# show that their published NIAH configuration does not compress at these lengths.
#
# No cool gate: NIAH is a retrieval measurement and no timing or energy from these
# cells is ever quoted. Timed cells always cool-gate; these do not need to.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
WS=/home/mislam22/EndurKV_workspace
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
BIN=/data/local/tmp/ukv_kd
SRC=$WS/EndurKV/benchmarks/niah
DEV=/data/local/tmp/endurkv/logs/niahcur_$(date +%Y%m%d_%H%M%S)
adb_safe_shell "mkdir -p $DEV" < /dev/null
for f in $SRC/niah_L*_n0.txt; do timeout 120 adb push "$f" "$DEV/$(basename $f)" < /dev/null >/dev/null 2>&1; done

run_pol(){   # run_pol <tag> <policy-args...>
  local TAG=$1; shift
  for STIM in $(cd $SRC && ls niah_L*_n0.txt); do
    case "$STIM" in *L8K*) CTX=8192;; *) CTX=4096;; esac
    local id="${TAG}__${STIM%.txt}"; local D=/tmp/niah_vs_sllm/$id
    [ -f "$D/meta.json" ] && continue
    mkdir -p "$D"; LOG "$id"
    adb_safe_shell "su -c 'rm -f $DEV/$id.done; setsid nohup sh -c \"timeout 900 env LD_LIBRARY_PATH=$BIN $BIN/eviction_bench \
      --prompt $DEV/$STIM --prompt-id $id --eval-mode gen --max-tokens 64 --ignore-eos \
      --ctx-size $CTX --model $M --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
      --cache-type-k f16 --cache-type-v f16 $* --n-batch 512 --n-ubatch 64 \
      --out-meta $DEV/$id.json --out-gen $DEV/$id.gen --out-csv /dev/null > /dev/null 2> $DEV/$id.err ; \
      echo DONE > $DEV/$id.done\" >/dev/null 2>&1 &'" < /dev/null
    local w=0
    while [ $w -lt 900 ]; do
      adb_safe_shell "[ -f $DEV/$id.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
      sleep 15; w=$((w+15))
    done
    adb_safe_pull "$DEV/$id.json" "$D/meta.json" >/dev/null 2>&1
    adb_safe_pull "$DEV/$id.gen"  "$D/gen.txt"   >/dev/null 2>&1
  done
}
run_pol snapkv1024  --policy snapkv --fa-on-evict --obs-window 64 --n-sink 0 --k-nominal 1024 --compact-inplace
run_pol adakv1024   --policy adakv  --fa-on-evict --obs-window 32 --n-sink 0 --k-nominal 1024 --compact-inplace
run_pol tova1024    --policy tova   --k-nominal 1024
run_pol h2o1024     --policy h2o    --k-nominal 1024
run_pol keydiff6144 --policy keydiff --n-sink 0 --k-nominal 6144 --compact-inplace --keydiff-decode-block 128
LOG "NIAH_CURRENT_BUILD_DONE"
touch /tmp/niah_current_DONE
