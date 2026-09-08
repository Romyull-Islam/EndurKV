#!/bin/bash
# Cross-build eviction_bench for Android phones with Adreno GPU (Vulkan).
# Run on a Linux host with the Android NDK. Profile A of PLATFORMS.md.
# Flags extracted from the validated 07-21 build (SHA e97a23f8… on OnePlus 15).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NDK="${ANDROID_NDK_ROOT:-$HOME/tools/ndk/android-ndk-r27c}"
[ -f "$NDK/build/cmake/android.toolchain.cmake" ] || { echo "NDK not found at $NDK (set ANDROID_NDK_ROOT)"; exit 1; }

cmake -S "$ROOT/llama.cpp" -B "$ROOT/llama.cpp/build-android-vulkan" \
      -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
      -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/llama.cpp/build-android-vulkan" -j"$(nproc)"

cmake -S "$ROOT/entropy_probe" -B "$ROOT/entropy_probe/build-android-vulkan" \
      -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
      -DLLAMA_CPP_DIR="$ROOT/llama.cpp" \
      -DLLAMA_BUILD_DIR="$ROOT/llama.cpp/build-android-vulkan"
cmake --build "$ROOT/entropy_probe/build-android-vulkan" -j"$(nproc)" --target eviction_bench

BIN="$ROOT/entropy_probe/build-android-vulkan/eviction_bench"
echo "OK: $BIN"
sha256sum "$BIN"
cat <<'DEPLOY'
Deploy (binary AND libs — never the binary alone; see PLATFORMS.md):
  adb push <root>/llama.cpp/build-android-vulkan/bin/*.so /data/local/tmp/endurkv/bin_vulkan/
  adb push <root>/entropy_probe/build-android-vulkan/eviction_bench /data/local/tmp/endurkv/bin_vulkan_new/
Then verify: bash scripts/android/assert_binary_current.sh entropy_probe/build-android-vulkan \
  /data/local/tmp/endurkv/bin_vulkan_new/eviction_bench
DEPLOY
