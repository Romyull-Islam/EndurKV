#!/bin/bash
# NIAH runs for SnapKV, Ada-KV, TOVA, H2O and KeyDiff on the current build, so the
# table comes from one build with live-cell retention on every row.
# SnapKV and Ada-KV score once at the end of prefill and run FA-on through the side
# node. TOVA and H2O score every decode step, which the side node cannot serve, so
# they run FA-off. These four use K=1024, the muKV budget. KeyDiff uses 6144, its
# published NIAH setting, which never evicts at these prompt lengths.
# No cool gate, since no timing or energy is reported from these cells.
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
