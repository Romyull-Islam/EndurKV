#!/bin/bash
# ============================================================================
# run_phi3_gpu_complete.sh -- the COMPLETE Phi-3 phone-GPU policy table. (2026-08-25)
#
# WHY THIS EXISTS. The phone_gpu_16k Phi-3 rows are unpublishable: five baselines
# (SnapKV, Ada-KV, H2O, TOVA, StreamingLLM) emitted 3.65-8.73 <unk> per 100 chars
# because the FA-off kq_soft_max capture is numerically broken on Adreno when
# head_dim is neither 64 nor 128 (Phi-3 = 96; Llama-1B = 64 and is unaffected).
# Their tok/s numbers measured a computation that was producing garbage.
#
# WHAT CHANGED. bin_vk_cur (host build 2026-08-22, pushed 08-24) carries the
# FA-off viability gate: on GPU with an unsupported head_dim it auto-promotes
# SnapKV/Ada-KV to the in-graph side node (they score ONCE at end of prefill, so
# this is plumbing, not a policy change) and REFUSES H2O/TOVA (they re-score every
# decode step; under --fa-on-evict H2O evicts 0 cells and is vanilla wearing an
# H2O label). Every cell records fa_off_gate in meta.json.
#
# H2O and TOVA ARE STILL ATTEMPTED so the refusal is recorded as evidence rather
# than an absence. They exit 2 with no meta.json; the scorer reports them as
# driver-refused and cites their CPU numbers instead.
#
# muKV USES --compact-inplace, NOT the round-trip. Round-trip needs a second
# 6144 MiB cache and is OS-killed at 16K on this phone: phi3_mukv_dfg{,_r2,_r3}
# in phone_gpu_16k are empty files whose logs stop mid-KV-allocation.
#
# GENERATIONS ARE KEPT (--out-gen), so every cell can be <unk>-graded. The old
# CPU runs wrote /dev/null and are permanently unverifiable; that is fixed here.
# ============================================================================
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
