#!/bin/bash
# Checks that in-place compaction preserves output on Phi-3, Bonsai-8B and gemma-2:
# each hotpotqa prompt runs with --compact-inplace and with --no-defrag, same policy,
# budget and greedy seed. The two generations should be byte-identical.
# 5 prompts x 2 arms x 3 models. No cool gate, since nothing is timed.
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
# Phi-3 and Bonsai first, since in-place compaction engages on them but declines on
# the gemma interleaved-SWA cache. Completed cells are skipped.
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
