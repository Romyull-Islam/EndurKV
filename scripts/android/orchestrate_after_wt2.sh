#!/bin/bash
# Orchestrator: after WT2-PPL finishes, validate Vulkan with llama-bench,
# decide CPU vs GPU based on validation result, then launch the full 3-policy
# 8B sweep with whichever backend wins.
set -e
export PATH=/home/mislam22/tools/platform-tools:$PATH
unset ANDROID_ADB_SERVER_PORT

LOG=/home/mislam22/EndurKV_workspace/EndurKV/figures/orchestrate.log
echo "[orch] starting at $(date)" | tee -a "$LOG"

MODEL=models/Llama-3.1-8B-Instruct-Q4_K_M.gguf
PHONE_ROOT=/data/local/tmp/endurkv

# ------------------------------------------------------------------------
# STEP 1: wait for WT2-PPL
# ------------------------------------------------------------------------
echo "[orch] waiting for llama-perplexity (WT2-PPL) to finish ..." | tee -a "$LOG"
while adb shell "ps -A | grep -q llama-perplexity" 2>/dev/null; do
    sleep 60
done
echo "[orch] WT2-PPL done at $(date)" | tee -a "$LOG"

# Kill the old queue (PID 22691) that would start A/B #4 K=2048 with old binary
kill 22691 2>/dev/null && echo "[orch] killed old queue PID 22691" | tee -a "$LOG" \
                       || echo "[orch] queue already exited" | tee -a "$LOG"
sleep 2

# Pull WT2 PPL output
echo "[orch] pulling WT2 PPL output ..." | tee -a "$LOG"
mkdir -p /home/mislam22/EndurKV_workspace/phone-logs/wt2_ppl
adb pull $PHONE_ROOT/logs/wt2_ppl/ /home/mislam22/EndurKV_workspace/phone-logs/ 2>&1 | tail -3 | tee -a "$LOG"
echo "[orch] WT2 PPL final estimate:" | tee -a "$LOG"
grep -E "Final estimate|estimate.*PPL" /home/mislam22/EndurKV_workspace/phone-logs/wt2_ppl/ppl_output.txt 2>/dev/null | tee -a "$LOG"

# ------------------------------------------------------------------------
# STEP 2: validate Vulkan (push GPU stack, run llama-bench briefly)
# ------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[orch] === STEP 2: VALIDATE VULKAN ===" | tee -a "$LOG"

VK_BIN=/home/mislam22/EndurKV_workspace/EndurKV/llama.cpp/build-android-vulkan/bin

# Create a vulkan-test dir on phone (don't overwrite CPU binaries yet)
adb shell "mkdir -p $PHONE_ROOT/vk_test"
for f in llama-bench libllama.so libggml.so libggml-base.so libggml-cpu.so libggml-vulkan.so; do
    adb push $VK_BIN/$f $PHONE_ROOT/vk_test/ 2>&1 | tail -1 | tee -a "$LOG"
done
adb shell "chmod 755 $PHONE_ROOT/vk_test/llama-bench"

# Push libomp.so too if needed
adb shell "[ -f $PHONE_ROOT/bin/libomp.so ] && cp $PHONE_ROOT/bin/libomp.so $PHONE_ROOT/vk_test/ || true"

# Run quick benchmark: 8B model, prefill 512 tokens + decode 16 tokens, -ngl 99 (all on GPU)
echo "[orch] running Vulkan llama-bench (pp512 + tg16, all GPU layers) ..." | tee -a "$LOG"
adb shell "
cd $PHONE_ROOT
LD_LIBRARY_PATH=vk_test vk_test/llama-bench \
  -m $MODEL \
  -p 512 -n 16 \
  -ngl 99 \
  -t 4 \
  --output md 2>&1
" > /tmp/vk_bench.out 2>&1
cat /tmp/vk_bench.out | tee -a "$LOG"

# Decide: did Vulkan work?
if grep -qE "(pp512|tg16).*[0-9]+\.[0-9]+" /tmp/vk_bench.out; then
    USE_VULKAN=1
    echo "[orch] ✓ Vulkan validated — will rebuild + run sweep on GPU" | tee -a "$LOG"
else
    USE_VULKAN=0
    echo "[orch] ✗ Vulkan failed (no benchmark numbers); falling back to CPU" | tee -a "$LOG"
fi

# ------------------------------------------------------------------------
# STEP 3: deploy whichever backend + launch full sweep
# ------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[orch] === STEP 3: DEPLOY + LAUNCH SWEEP (use_vulkan=$USE_VULKAN) ===" | tee -a "$LOG"

if [ "$USE_VULKAN" = "1" ]; then
    # Rebuild eviction_bench against Vulkan llama.cpp
    echo "[orch] rebuilding eviction_bench linked against Vulkan libllama ..." | tee -a "$LOG"
    cd /home/mislam22/EndurKV_workspace/EndurKV/entropy_probe
    rm -rf build-android-vulkan
    mkdir build-android-vulkan
    cd build-android-vulkan
    cmake .. \
      -DCMAKE_TOOLCHAIN_FILE=/home/mislam22/tools/ndk/android-ndk-r27c/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-28 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CPP_DIR=/home/mislam22/EndurKV_workspace/EndurKV/llama.cpp \
      -DLLAMA_BUILD_DIR=/home/mislam22/EndurKV_workspace/EndurKV/llama.cpp/build-android-vulkan \
      2>&1 | tail -5 | tee -a "$LOG"
    cmake --build . --target eviction_bench -j 12 2>&1 | tail -5 | tee -a "$LOG"
    STRIP=/home/mislam22/tools/ndk/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip
    $STRIP eviction_bench
    ls -lh eviction_bench | tee -a "$LOG"

    # Push Vulkan libs + new eviction_bench to phone's bin/
    echo "[orch] pushing Vulkan stack to phone ..." | tee -a "$LOG"
    for f in libllama.so libggml.so libggml-base.so libggml-cpu.so libggml-vulkan.so; do
        adb push /home/mislam22/EndurKV_workspace/EndurKV/llama.cpp/build-android-vulkan/bin/$f \
                $PHONE_ROOT/bin/$f 2>&1 | tail -1
    done
    adb push /home/mislam22/EndurKV_workspace/EndurKV/entropy_probe/build-android-vulkan/eviction_bench \
            $PHONE_ROOT/bin/eviction_bench 2>&1 | tail -1
    adb shell "chmod 755 $PHONE_ROOT/bin/eviction_bench"
    # Save the original CPU binary as eviction_bench_cpu for fallback
    adb shell "cp $PHONE_ROOT/bin/eviction_bench $PHONE_ROOT/bin/eviction_bench_vulkan" 2>&1
    export N_GPU_LAYERS=99
else
    # Just push CPU eviction_bench (already up to date with top-p)
    echo "[orch] pushing CPU eviction_bench with top-p sampling ..." | tee -a "$LOG"
    adb push /home/mislam22/EndurKV_workspace/EndurKV/entropy_probe/build-android/eviction_bench \
            $PHONE_ROOT/bin/eviction_bench 2>&1 | tail -1
    adb shell "chmod 755 $PHONE_ROOT/bin/eviction_bench"
    export N_GPU_LAYERS=0
fi

# Push cool-then-run wrapper
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/phone_cool_then_run.sh \
        $PHONE_ROOT/scripts/ 2>&1 | tail -1
adb shell "chmod 755 $PHONE_ROOT/scripts/phone_cool_then_run.sh"

# ------------------------------------------------------------------------
# STEP 4: launch the full sweep
# ------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[orch] === STEP 4: LAUNCH FULL SWEEP (8B) ===" | tee -a "$LOG"
N_GPU_LAYERS=$N_GPU_LAYERS \
MODEL=$MODEL \
bash /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/phone_full_sweep_3policy.sh 2>&1 | tee -a "$LOG"

echo "[orch] FULL SWEEP DONE at $(date)" | tee -a "$LOG"
