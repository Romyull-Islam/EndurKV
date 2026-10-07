#!/bin/bash
# run_gemma_gpu_mitigate.sh: workarounds for Gemma-2 failing on the Adreno GPU. It emits
# "<pad>" on a 223-token prompt and hits vk::DeviceLostError on a 6382-token one, while
# Llama runs on the same build. 16 tokens per cell:
#   M1 ubatch16   smaller dispatches (n-ubatch 16)
#   M2 partial    half the layers on the GPU, to isolate one offloaded op
#   M3 mid2k      a ~2000-token prompt, to look for a size threshold
#   M4 cpu_tiny   control: the 223-token prompt on the CPU. If it also emits <pad>,
#                 that output is not a GPU defect.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%F' '%H:%M:%S)] $*"; }
VK=/data/local/tmp/endurkv/bin_vk_cur
CB=/data/local/tmp/endurkv/bin_cpu_kd
MOD=/data/local/tmp/endurkv/models
GM=$MOD/gemma-2-2b-it-Q4_K_M.gguf
DEV=/data/local/tmp/gemmadiag
OUT=/tmp/gemma_gpu_mitigate; mkdir -p $OUT
adb_safe_shell "mkdir -p $DEV" < /dev/null >/dev/null 2>&1
# a ~2000-token probe: roughly a third of the 6382-token corpus
adb_safe_shell "head -c 8000 /data/local/tmp/endurkv/corpora/prompt_7k.txt > $DEV/mid2k.txt" < /dev/null >/dev/null 2>&1

# $1=tag $2=bin $3=prompt $4=ctx $5=gpulayers  rest=extra flags
m(){
  local TAG=$1 BIN=$2 PR=$3 CTX=$4 GL=$5; shift 5
  mkdir -p "$OUT/$TAG"
  [ -s "$OUT/$TAG/meta.json" ] && { LOG "$TAG cached"; return; }
  LOG "running $TAG"
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 1200 env LD_LIBRARY_PATH=$BIN $BIN/eviction_bench \
    --prompt $PR --prompt-id $TAG --eval-mode gen --max-tokens 16 --ignore-eos --ctx-size $CTX \
    --model $GM --seed 42 --threads 4 --n-gpu-layers $GL --greedy \
    --cache-type-k f16 --cache-type-v f16 --policy vanilla --k-nominal 1024 $* \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > $DEV/$TAG.out 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  local w=0
  while [ $w -lt 1260 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 15; w=$((w+15)); done
  adb_safe_pull "$DEV/$TAG.json" "$OUT/$TAG/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.gen"  "$OUT/$TAG/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.err"  "$OUT/$TAG/err.txt"   >/dev/null 2>&1
  if [ -s "$OUT/$TAG/gen.txt" ]; then
    LOG "  [$TAG] RAN: $(head -c 100 "$OUT/$TAG/gen.txt" | tr '\n' ' ')"
  else
    LOG "  [$TAG] FAILED: $(grep -oE 'vk::[A-Za-z]+Error|ErrorDeviceLost' "$OUT/$TAG/err.txt" 2>/dev/null | tail -1)"
  fi
}

P7=/data/local/tmp/endurkv/corpora/prompt_7k.txt
m M1_ubatch16  $VK $P7           8192 99 --n-batch 128 --n-ubatch 16
m M2_partial   $VK $P7           8192 14 --n-batch 512 --n-ubatch 64
m M3_mid2k     $VK $DEV/mid2k.txt 8192 99 --n-batch 512 --n-ubatch 64
m M4_cpu_tiny  $CB $DEV/tiny.txt  4096  0 --n-batch 512 --n-ubatch 64
LOG "GEMMA MITIGATION SWEEP DONE -> $OUT"
touch /tmp/gemma_mitigate_DONE
