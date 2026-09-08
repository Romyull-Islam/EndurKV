#!/bin/bash
# ============================================================================
# run_prefill_diag.sh -- why is FA-on prefill +23% over vanilla? (2026-08-26)
#
# ANOMALY. On Phi-3 CPU, vanilla prefills in 381 s but EVERY evicting policy
# takes ~470 s (+23%), including StreamingLLM -- which reads no attention, no
# key geometry, installs no eval-callback (positional_policy => cb_eval NULL,
# same as vanilla), and runs the same FA-on graph with identical n_batch /
# n_ubatch / n_ctx / KV dtype. In-place compaction is logged at 43 ms, three
# orders of magnitude too small to explain a 90 s gap. So the published claim
# "+1.2% prefill for in-graph scoring" does not hold on this model and we do
# not yet know what the cost actually is.
#
# THREADS=6 (was 4 in the killed first attempt). The thread count is now known to
# scale throughput almost linearly (observed 6/4 = 1.46 vs ideal 1.50), so an arm
# at the wrong count would swamp the effect being measured.
#
# DESIGN. Two arms per policy, everything else fixed:
#   *_defrag   : as shipped (evict + compact)
#   *_nodefrag : --no-defrag (evict, do NOT compact)
# plus a vanilla control re-run at the SAME point in the queue, because vanilla
# was cell #1 of the previous campaign and run-order is a live confound.
#
#   vanilla == nodefrag  -> the cost is COMPACTION
#   nodefrag == defrag   -> the cost is EVICTION/selection, not compaction
#   vanilla_ctl != 381 s -> the cost is RUN ORDER, not the policy at all
# ============================================================================
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
  # only 64 decode tokens: this measures PREFILL, so decode is a formality
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
# THREE compaction arms for muKV, not two. Table 1's muKV never passed
# --compact-inplace and fell through to the ROUND-TRIP path: its prefill was
# +3.5% over vanilla, matching the paper's "+1.2%" claim. Adding
# --compact-inplace this session took prefill to +22%. So in-place is NOT
# strictly better than round-trip -- it trades prefill time for peak memory:
#   round-trip : 2x cache (OS-killed for Phi-3@16K), logged 330 ms, cheap
#   in-place   : never exceeds prefill's allocation, but ~20k small memcpys
# These arms measure that trade-off directly instead of inferring it.
# ORDER: vanilla runs MID-QUEUE (position 4 of 8), not first -- a first-slot
# vanilla is the exact bias this whole investigation started from. KeyDiff arms
# added: it pays the LARGEST prefill premium (+34%) while reading no attention,
# holding no callback and emitting no side node, so it brackets the mechanism
# from the other side.
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
