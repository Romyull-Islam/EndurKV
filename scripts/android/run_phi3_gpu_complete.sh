#!/bin/bash
# Phi-3 phone-GPU policy table (12K prompt + 4096 decode, ctx 16K, bin_vk_cur).
# FA-off attention capture gives corrupt output on Adreno at Phi-3's head_dim 96, so the
# build's FA-off gate moves SnapKV/Ada-KV to the in-graph side node (they score once after
# prefill) and refuses H2O/TOVA, which re-score every step. H2O/TOVA still run so the refusal is logged.
# muKV uses --compact-inplace: round-trip needs a second 6144 MiB cache and is OS-killed at 16K.
# Generations are kept so every cell can be checked for <unk> output.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
VK=/data/local/tmp/endurkv/bin_vk_cur
M=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
GEN=${GEN:-4096}
DEV=/data/local/tmp/phi3gpu
HOST=/tmp/phi3_gpu_complete
mkdir -p $HOST
adb_safe_shell "mkdir -p $DEV" < /dev/null >/dev/null 2>&1

MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

cell(){
  local TAG=$1; shift
  [ -s "$HOST/$TAG/meta.json" ] && { LOG "$TAG cached"; return; }
  mkdir -p "$HOST/$TAG"
  LOG "cooling for $TAG ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) LOG "  $CG";; *) LOG "  [SKIP-HOT] $TAG"; return;; esac
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $DEV/$TAG.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"LD_LIBRARY_PATH=$VK timeout 7200 $VK/eviction_bench \
    --prompt $P --prompt-id $TAG --eval-mode gen --max-tokens $GEN --ignore-eos --ctx-size 16384 \
    --n-batch 512 --n-ubatch 64 --model $M --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > $DEV/$TAG.out 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  local w=0
  while [ $w -lt 7500 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 30; w=$((w+30))
  done
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null >/dev/null 2>&1
  for e in json gen err out csv; do
    adb_safe_pull "$DEV/$TAG.$e" "$HOST/$TAG/$( [ $e = csv ] && echo sensors.csv || echo $e.txt )" >/dev/null 2>&1
  done
  [ -s "$HOST/$TAG/json.txt" ] && mv "$HOST/$TAG/json.txt" "$HOST/$TAG/meta.json"
  [ -s "$HOST/$TAG/gen.txt" ] && LOG "  [$TAG] ok  $(head -c 46 "$HOST/$TAG/gen.txt" | tr '\n' ' ')" \
    || LOG "  [$TAG] NO OUTPUT -- $(grep -m1 '^\[gate\]' "$HOST/$TAG/err.txt" 2>/dev/null | cut -c1-92)"
}

LOG "=== Phi-3 phone GPU, 12K prompt + ${GEN} decode = 16K ctx, build bin_vk_cur ==="
cell vanilla       --policy vanilla
cell mukv          $MU --k-nominal 1024
cell keydiff2048   --policy keydiff --n-sink 0 --k-nominal 2048 --compact-inplace --keydiff-decode-block 128
cell snapkv        --policy snapkv --obs-window 16 --snapkv-kernel 5 --n-sink 0 --k-nominal 1024
cell adakv         --policy adakv  --n-sink 0 --k-nominal 1024
cell streamingllm  --policy streamingllm --n-sink 4 --k-nominal 2000
cell h2o           --policy h2o  --n-sink 0 --k-nominal 1024
cell tova          --policy tova --n-sink 0 --k-nominal 1024
LOG "PHI3_GPU_COMPLETE_DONE"
touch /tmp/phi3_gpu_complete_DONE
