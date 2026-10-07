#!/bin/bash
# Prefill diagnostic, Phi-3 CPU, 6 threads: why evicting policies (even StreamingLLM,
# which installs no eval callback) prefill slower than vanilla.
# Each policy runs with round-trip compaction, in-place compaction and --no-defrag.
#   vanilla == nodefrag  : the cost is compaction
#   nodefrag == compact  : the cost is eviction/selection
#   vanilla_ctl differs from earlier vanilla runs : the cost is run order
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
LOG "waiting for the n=3 headline ..."
while [ ! -f /tmp/n3_headline_v2_DONE ]; do sleep 60; done
exec 9>/tmp/.endurkv_queue.lock; flock 9
LOG "starting prefill diagnostic"
CB=/data/local/tmp/endurkv/bin_cpu_kd
M=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_7k.txt
DEV=/data/local/tmp/prediag; HOST=/tmp/prefill_diag
mkdir -p $HOST; adb_safe_shell "mkdir -p $DEV" < /dev/null >/dev/null 2>&1
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
cell(){
  local TAG=$1; shift
  [ -s "$HOST/$TAG/meta.json" ] && { LOG "$TAG cached"; return; }
  mkdir -p "$HOST/$TAG"
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) : ;; *) LOG "  [SKIP-HOT] $TAG"; return;; esac
  # 64 decode tokens only, this measures prefill
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 5400 env LD_LIBRARY_PATH=$CB $CB/eviction_bench \
    --prompt $P --prompt-id $TAG --eval-mode gen --max-tokens 64 --ignore-eos --ctx-size 16384 \
    --n-batch 512 --ubatch-size 64 --model $M --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > $DEV/$TAG.out 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  local w=0
  while [ $w -lt 5600 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 20; w=$((w+20)); done
  adb_safe_pull "$DEV/$TAG.json" "$HOST/$TAG/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.err"  "$HOST/$TAG/err.txt"   >/dev/null 2>&1
  PF=$(python3 -c "
import json,re,sys
try:
  j=json.loads(re.sub(r':\s*-?nan\b',': NaN',open('$HOST/$TAG/meta.json').read())); print('%.0f'%(j['prefill_ms']/1000))
except Exception: print('--')" 2>/dev/null)
  LOG "  [$TAG] prefill=${PF}s"
}
# Round-trip compaction needs 2x the cache but is fast, in-place stays within the
# prefill allocation but does many small memcpys. These arms measure that trade-off.
# vanilla runs mid-queue (4 of 8) so it is not biased by running first. KeyDiff reads
# no attention and installs no callback, so it brackets the cost from the other side.
cell mukv_roundtrip     $MU --k-nominal 1024 --force-defrag
cell keydiff_inplace    --policy keydiff --n-sink 0 --k-nominal 2048 --compact-inplace --keydiff-decode-block 128
cell mukv_inplace       $MU --k-nominal 1024 --compact-inplace
cell vanilla_ctl        --policy vanilla --k-nominal 1024
cell mukv_nocompact     $MU --k-nominal 1024 --no-defrag
cell keydiff_nocompact  --policy keydiff --n-sink 0 --k-nominal 2048 --no-defrag --keydiff-decode-block 128
cell sllm_roundtrip     --policy streamingllm --n-sink 4 --k-nominal 2000 --force-defrag
cell sllm_nocompact     --policy streamingllm --n-sink 4 --k-nominal 2000 --no-defrag
LOG "PREFILL_DIAG_DONE"
touch /tmp/prefill_diag_DONE
