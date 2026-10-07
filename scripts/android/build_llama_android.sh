#!/usr/bin/env bash
# Cross-compile llama-cli, llama-perplexity and llama-bench (plus libllama and
# libggml) for aarch64 Android into $LLAMA_BUILD/bin/. Used for the vanilla
# llama.cpp baseline on the phone.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ANDROID_NDK="${ANDROID_NDK:-/home/mislam22/tools/ndk/android-ndk-r27c}"
LLAMA_DIR="$WORKSPACE/EndurKV/llama.cpp"
LLAMA_BUILD="$LLAMA_DIR/build-android"

if [ ! -d "$ANDROID_NDK/build/cmake" ]; then
    echo "ERROR: Android NDK r27c not found at $ANDROID_NDK" >&2
    exit 1
fi

echo "[build] workspace: $WORKSPACE"
echo "[build] NDK:       $ANDROID_NDK"
echo "[build] target:    $LLAMA_BUILD"
echo "[build] (clean reconfigure to fix stale Windows cache)"

# Remove a stale CMake cache with Windows paths
rm -rf "$LLAMA_BUILD/CMakeCache.txt" "$LLAMA_BUILD/CMakeFiles"

mkdir -p "$LLAMA_BUILD"
cd "$LLAMA_BUILD"

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
  -DGGML_OPENMP=OFF

# Build the three tools we need for vanilla baseline measurement
cmake --build . --target llama-cli llama-perplexity llama-bench -j"$(nproc)"

echo ""
echo "[build] DONE. Binaries:"
ls -lh "$LLAMA_BUILD/bin/" | grep -E "llama-(cli|perplexity|bench)|libllama|libggml" | head -10
echo ""
echo "[build] verify ARM:"
file "$LLAMA_BUILD/bin/llama-cli" "$LLAMA_BUILD/bin/llama-perplexity" 2>&1 | head -2
