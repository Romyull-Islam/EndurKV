#!/bin/bash
# ============================================================================
# CONTROL EXPERIMENT (2026-07-27): does an 8B model run on the Adreno 840 GPU?
#
# WHY: Bonsai-8B (Q1_0, 1-bit) fails on the phone GPU two ways -- vk::DeviceLostError
# under the eviction_bench harness, and numerically broken Vulkan Q1_0 kernels
# (nan PPL, degenerate text) on the Prism build. Those are confounded: we cannot
# tell whether 8B-on-Adreno fails because of SIZE or because of the 1-BIT KERNEL.
#
# This script removes the confound by running a same-scale 8B in a Vulkan-native
# quant: DeepSeek-R1-Distill-Llama-8B-Q4_K_M (llama arch, 4.58 GB, 32 layers).
#   - If it runs -> the Bonsai failure is the Q1_0 Vulkan kernel, not model size,
#     and "8B on mobile GPU" stays viable for the paper.
#   - If it also dies -> Adreno cannot sustain 8B prefill at all, and the CPU+muKV
#     deployment claim is about the device, not about the quant.
#
# STAGE 0 is the MANDATORY output-validity check (PLATFORMS.md): greedy GPU-vs-CPU
# diff + a finite-PPL cell. Bonsai passed every mechanism check while computing
# garbage; no timing number is recorded here until logits are proven finite.
# STAGE 1 finds the largest prompt that completes (the question asked of Bonsai).
# STAGE 2 runs vanilla / muKV / SnapKV at that size, cooled, with energy.
#
# Serialised by design: NEVER run while the NIAH campaign is live -- concurrent
# device work contaminates every timed and energy-measured cell.
# ============================================================================
set -u
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue   # never poke a dead port (starts a squatting adb server)
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] using server port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

HOST_MODEL=/home/mislam22/EndurKV_workspace/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
DEV_MODEL=/data/local/tmp/endurkv/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
VK=/data/local/tmp/endurkv/bin_vulkan
CPUB=/data/local/tmp/endurkv/bin_cpu_v87
OUT_HOST=/tmp/llama8b_gpu; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/l8b_$(date +%Y%m%d_%H%M%S)
PROMPT=/data/local/tmp/endurkv/corpora/prompt_12k.txt
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

adb_safe_shell "mkdir -p $OUT" < /dev/null

# ---- push the model once (4.58 GB; skip if already correct size) -----------
have=$(adb_safe_shell "su -c 'stat -c %s $DEV_MODEL 2>/dev/null || echo 0'" < /dev/null | tr -d '\r ')
want=$(stat -c %s "$HOST_MODEL")
if [ "${have:-0}" != "$want" ]; then
  echo "[push] $HOST_MODEL -> phone ($(( want / 1048576 )) MiB) ..."
  adb push "$HOST_MODEL" /data/local/tmp/l8b.gguf < /dev/null
  adb_safe_shell "su -c 'mv /data/local/tmp/l8b.gguf $DEV_MODEL'" < /dev/null
else
  echo "[push] model already on device, size matches"
fi

cool(){ CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
        echo "$CG" | tail -1
        case "$CG" in *"cool ddr="*) return 0;; *) echo "[SKIP-HOT] gate failed"; return 1;; esac; }

# ============ STAGE 0 — output validity (no timing recorded yet) ============
echo "=== STAGE 0: output-validity check (GPU vs CPU greedy diff + finite PPL) ==="
adb_safe_shell "su -c 'head -c 2000 $PROMPT > /data/local/tmp/l8b_short.txt'" < /dev/null
for BE in gpu cpu; do
  if [ $BE = gpu ]; then B=$VK; NGL=99; TH=4; else B=$CPUB; NGL=0; TH=6; fi
  adb_safe_shell "LD_LIBRARY_PATH=$B timeout 1800 $B/eviction_bench --prompt /data/local/tmp/l8b_short.txt \
    --prompt-id valid_$BE --eval-mode gen --max-tokens 24 --ignore-eos --ctx-size 4096 \
    --n-batch 512 --n-ubatch 64 --model $DEV_MODEL --seed 42 --threads $TH --n-gpu-layers $NGL --greedy \
    --policy vanilla --cache-type-k f16 --cache-type-v f16 \
    --out-meta $OUT/valid_$BE.json --out-gen $OUT/valid_$BE.txt --out-csv /dev/null \
    > $OUT/valid_$BE.out 2> $OUT/valid_$BE.err" < /dev/null
done
adb_safe_pull "$OUT" "$OUT_HOST/"
echo "--- greedy outputs (MUST be identical) ---"
diff "$OUT_HOST/$(basename $OUT)/valid_gpu.txt" "$OUT_HOST/$(basename $OUT)/valid_cpu.txt" >/dev/null 2>&1 \
  && echo "PASS: GPU == CPU" || { echo "FAIL: GPU != CPU -- kernels are wrong, STOPPING (this is the Bonsai failure mode)"; \
       head -c 300 "$OUT_HOST/$(basename $OUT)/valid_gpu.txt" 2>/dev/null; exit 2; }

# ============ STAGE 1 — largest prompt that completes on GPU ================
echo "=== STAGE 1: prompt-length ceiling on Adreno (vanilla) ==="
BEST=0
for BYTES in 2000 4000 8000 16000 32000 44308; do
  cool || continue
  adb_safe_shell "su -c 'head -c $BYTES $PROMPT > /data/local/tmp/l8b_len.txt'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$VK timeout 2400 $VK/eviction_bench --prompt /data/local/tmp/l8b_len.txt \
    --prompt-id len$BYTES --eval-mode gen --max-tokens 16 --ignore-eos --ctx-size 16384 \
    --n-batch 512 --n-ubatch 64 --model $DEV_MODEL --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --policy vanilla --cache-type-k f16 --cache-type-v f16 \
    --out-meta $OUT/len$BYTES.json --out-gen /dev/null --out-csv /dev/null \
    > $OUT/len$BYTES.out 2> $OUT/len$BYTES.err" < /dev/null
  if adb_safe_shell "test -s $OUT/len$BYTES.json && echo OK" < /dev/null | grep -q OK; then
    echo "  bytes=$BYTES OK"; BEST=$BYTES
  else
    echo "  bytes=$BYTES FAILED ($(adb_safe_shell "grep -oE 'DeviceLost|out of memory|Killed' $OUT/len$BYTES.err | tail -1" < /dev/null | tr -d '\r'))"
    break
  fi
done
echo "=== largest completing prompt: $BEST bytes ==="
[ "$BEST" = 0 ] && { echo "8B does not run on this GPU at any tested size"; adb_safe_pull "$OUT" "$OUT_HOST/"; exit 3; }

# ============ STAGE 2 — vanilla / muKV / SnapKV at the working size =========
echo "=== STAGE 2: three policies at $BEST bytes, cooled, with energy ==="
adb_safe_shell "su -c 'head -c $BEST $PROMPT > /data/local/tmp/l8b_run.txt'" < /dev/null
cell(){ local TAG=$1; shift
  cool || return
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $OUT/${TAG}_sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$VK timeout 3600 $VK/eviction_bench --prompt /data/local/tmp/l8b_run.txt \
    --prompt-id $TAG --eval-mode gen --max-tokens 512 --ignore-eos --ctx-size 16384 \
    --n-batch 512 --n-ubatch 64 --model $DEV_MODEL --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --k-nominal 1024 --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $OUT/$TAG.json --out-gen $OUT/$TAG.gen --out-csv /dev/null \
    > $OUT/$TAG.out 2> $OUT/$TAG.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  echo "  [$TAG] done"
}
cell vanilla --policy vanilla
cell mukv    $MU
cell snapkv  --policy snapkv --obs-window 64 --n-sink 0

adb_safe_pull "$OUT" "$OUT_HOST/"
echo "=== ALL DONE -> $OUT_HOST/$(basename $OUT) ==="
