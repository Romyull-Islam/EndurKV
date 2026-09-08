#!/usr/bin/env bash
# build_llama_android_vulkan.sh — cross-compile llama.cpp for Android with Vulkan GPU.
# Output in: $LLAMA_DIR/build-android-vulkan/
# Standard llama.cpp practice for Adreno GPUs (Snapdragon 8 Elite Gen 5).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ANDROID_NDK="${ANDROID_NDK:-/home/mislam22/tools/ndk/android-ndk-r27c}"
LLAMA_DIR="$WORKSPACE/EndurKV/llama.cpp"
LLAMA_BUILD="$LLAMA_DIR/build-android-vulkan"

if [ ! -d "$ANDROID_NDK/build/cmake" ]; then
    echo "ERROR: Android NDK not found at $ANDROID_NDK" >&2; exit 1
fi

echo "[build-vk] workspace: $WORKSPACE"
echo "[build-vk] NDK:       $ANDROID_NDK"
echo "[build-vk] target:    $LLAMA_BUILD"

rm -rf "$LLAMA_BUILD/CMakeCache.txt" "$LLAMA_BUILD/CMakeFiles"
mkdir -p "$LLAMA_BUILD"
cd "$LLAMA_BUILD"

VK_SYSROOT="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot"
# glslc lives in NDK's shader-tools — put it on PATH so FindVulkan can find it
export PATH="$ANDROID_NDK/shader-tools/linux-x86_64:$PATH"

cmake "$LLAMA_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_CURL=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_VULKAN=ON \
  -DVulkan_INCLUDE_DIR="$VK_SYSROOT/usr/include" \
  -DVulkan_LIBRARY="$VK_SYSROOT/usr/lib/aarch64-linux-android/28/libvulkan.so"

cmake --build . --target llama-completion llama-perplexity llama-bench -j"$(nproc)"

echo ""
echo "[build-vk] DONE. Binaries:"
ls -lh "$LLAMA_BUILD/bin/" | grep -E "llama-(cli|perplexity|bench|completion)|libllama|libggml" | head -15
file "$LLAMA_BUILD/bin/llama-bench" 2>&1 | head -1
