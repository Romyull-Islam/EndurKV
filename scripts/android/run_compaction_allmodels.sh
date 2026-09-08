#!/bin/bash
# ============================================================================
# run_compaction_allmodels.sh -- does compaction preserve output on EVERY model?
# (2026-08-28)
#
# The gap. The byte-identity check (compacted vs uncompacted muKV at a FIXED
# keep-set -> identical generated text) was run on Llama-3.2-1B / hotpotqa only,
# n=15. The paper cannot claim it for Phi-3, gemma-2 (interleaved SWA) or
# Bonsai-8B (1-bit) without running it there, and gemma is the interesting case:
# its iSWA cache is the one layout where a sliding compaction could plausibly
# disturb positions.
#
# The test. For each model, the same prompt is run twice with an identical policy
# and budget, differing ONLY in --compact-inplace vs --no-defrag. Same seed,
# greedy. If compaction is position-preserving the two generations must be
# byte-identical; any difference is a real defect, not a quality trade.
#
# Scope. 5 hotpotqa prompts x 2 arms x 3 models = 30 cells. Quality-only, so no
# cool gate (nothing timed is quoted from these). Ordered fastest model first so
# an interruption still leaves complete models behind.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
LOG "waiting for the prefill decomposition ..."
while [ ! -f /tmp/prefill_decompose_DONE ]; do sleep 60; done
exec 9>/tmp/.endurkv_queue.lock; flock 9
CB=/data/local/tmp/endurkv/bin_cpu_kd
MOD=/data/local/tmp/endurkv/models
DEV=/data/local/tmp/cmpall; HOST=/tmp/compaction_allmodels
mkdir -p $HOST; adb_safe_shell "mkdir -p $DEV" < /dev/null >/dev/null 2>&1
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

run(){   # $1=model_tag $2=gguf $3=idx $4=arm $5=extra-flag
  local TAG="$1_$3_$4"
  [ -s "$HOST/$TAG/gen.txt" ] && { LOG "$TAG cached"; return; }
  mkdir -p "$HOST/$TAG"
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 5400 env LD_LIBRARY_PATH=$CB $CB/eviction_bench \
    --prompt $DEV/hp_$3.txt --prompt-id $TAG --eval-mode gen --max-tokens 32 --ctx-size 16384 \
    --n-batch 512 --ubatch-size 64 --model $2 --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --cache-type-k f16 --cache-type-v f16 $MU --k-nominal 1024 $5 \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > /dev/null 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  local w=0
  while [ $w -lt 5500 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 20; w=$((w+20)); done
  adb_safe_pull "$DEV/$TAG.gen"  "$HOST/$TAG/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.json" "$HOST/$TAG/meta.json" >/dev/null 2>&1
  LOG "  $TAG done"
}
for i in 000 001 002 003 004; do
  src=/tmp/longbench_adaptive_3x3/phi3_vanilla_hotpotqa/prompt_$i.txt
  [ -f "$src" ] && timeout 180 adb push "$src" "$DEV/hp_$i.txt" < /dev/null >/dev/null 2>&1
done
# REORDERED 2026-08-29: gemma is already answered (2/2 pairs differ; in-place declines
# to round-trip on its interleaved-SWA cache every time). Phi-3 and Bonsai both ENGAGE
# in-place and are the models that decide whether the compaction claim generalises, so
# they run first. Completed cells are cached and skipped, so nothing is repeated.
for spec in "phi3:$MOD/Phi-3-mini-128k-instruct-Q4_K_M.gguf" "bonsai:$MOD/Bonsai-8B-Q1_0.gguf" "gemma:$MOD/gemma-2-2b-it-Q4_K_M.gguf"; do
  tag="${spec%%:*}"; gguf="${spec#*:}"
  LOG "=== model $tag ==="
  for i in 000 001 002 003 004; do
    run "$tag" "$gguf" "$i" compact   "--compact-inplace"
    run "$tag" "$gguf" "$i" nocompact "--no-defrag"
  done
done
LOG "COMPACTION_ALLMODELS_DONE"
touch /tmp/compaction_allmodels_DONE
