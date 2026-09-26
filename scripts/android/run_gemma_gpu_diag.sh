#!/bin/bash
# ============================================================================
# run_gemma_gpu_diag.sh -- why gemma-2-2b does not run on the Adreno. (2026-09-20)
#
# The 6382-token gemma GPU cell died with
#     vk::DeviceLostError: vk::Queue::submit: ErrorDeviceLost
# inside llama_decode, AFTER the graph reserved cleanly (948 nodes, 2 splits,
# 63 MiB compute buffer). So it is not an allocation or context-size failure.
# The paper currently claims gemma is CPU-only because its context is 8192,
# which this shows is the wrong reason. Three cells settle what the right one is:
#
#   D1 tiny prompt, ctx 4096, 16 tokens. Still lost  -> gemma's graph breaks the
#      driver, independent of length. Survives -> length or ctx is involved.
#   D2 the real 6382-token prompt at ctx 8192, 16 tokens. Separates "long prompt"
#      from "16384 ctx".
#   D3 Llama-1B on the same build, same GPU, right after. Proves the device is
#      healthy and D1/D2 are about gemma, not about a wedged GPU.
#
# Cheap by design: 16 generated tokens each, no cool gate, since a crash or a
# clean load does not depend on temperature. Runs GPU-only, so it must not
# overlap the needle campaign.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%F' '%H:%M:%S)] $*"; }
VK=/data/local/tmp/endurkv/bin_vk_cur
MOD=/data/local/tmp/endurkv/models
DEV=/data/local/tmp/gemmadiag
OUT=/tmp/gemma_gpu_diag; mkdir -p $OUT
adb_safe_shell "mkdir -p $DEV" < /dev/null >/dev/null 2>&1

# a short prompt: first 40 lines of the 7k corpus, a few hundred tokens
adb_safe_shell "head -c 1200 /data/local/tmp/endurkv/corpora/prompt_7k.txt > $DEV/tiny.txt; wc -c $DEV/tiny.txt" < /dev/null

# $1=tag $2=model $3=prompt $4=ctx $5=maxtok
diag(){
  local TAG=$1 MP=$2 PR=$3 CTX=$4 MT=$5
  mkdir -p "$OUT/$TAG"
  [ -s "$OUT/$TAG/meta.json" ] && { LOG "$TAG cached"; return; }
  LOG "running $TAG (ctx=$CTX)"
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 900 env LD_LIBRARY_PATH=$VK $VK/eviction_bench \
    --prompt $PR --prompt-id $TAG --eval-mode gen --max-tokens $MT --ignore-eos --ctx-size $CTX \
    --n-batch 512 --n-ubatch 64 --model $MP --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 --policy vanilla --k-nominal 1024 \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > $DEV/$TAG.out 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  local w=0
  while [ $w -lt 960 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 15; w=$((w+15)); done
  adb_safe_pull "$DEV/$TAG.json" "$OUT/$TAG/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.gen"  "$OUT/$TAG/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.err"  "$OUT/$TAG/err.txt"   >/dev/null 2>&1
  if [ -s "$OUT/$TAG/gen.txt" ]; then
    LOG "  [$TAG] RAN: $(head -c 80 "$OUT/$TAG/gen.txt" | tr '\n' ' ')"
  else
    LOG "  [$TAG] FAILED: $(grep -oE 'vk::[A-Za-z]+Error[^ ]*|ErrorDeviceLost|error[^\"]{0,60}' "$OUT/$TAG/err.txt" 2>/dev/null | tail -1)"
  fi
}

GM=$MOD/gemma-2-2b-it-Q4_K_M.gguf
LM=$MOD/Llama-3.2-1B-Instruct-Q4_K_M.gguf
diag gemma_tiny_ctx4096  $GM $DEV/tiny.txt 4096 16
diag gemma_6382_ctx8192  $GM /data/local/tmp/endurkv/corpora/prompt_7k.txt 8192 16
diag llama_control       $LM /data/local/tmp/endurkv/corpora/prompt_7k.txt 8192 16
LOG "GEMMA GPU DIAG DONE -> $OUT"
touch /tmp/gemma_gpu_diag_DONE
