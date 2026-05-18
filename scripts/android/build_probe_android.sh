#!/usr/bin/env bash
# build_probe_android.sh — cross-compile entropy_probe (and friends) for aarch64.
#
# Run from anywhere; resolves the workspace via $WORKSPACE or by walking up
# from the script's location.  Produces:
#   $WORKSPACE/EndurKV/entropy_probe/build-android/entropy_probe
#   $WORKSPACE/EndurKV/entropy_probe/build-android/attention_probe
#   $WORKSPACE/EndurKV/entropy_probe/build-android/prune_probe
#
# Prereqs:
#   * Android NDK r27c at $WORKSPACE/toolchain/android-ndk-r27c/
#     (override via $ANDROID_NDK if installed elsewhere)
#   * llama.cpp Android build already done at
#     $WORKSPACE/EndurKV/llama.cpp/build-android/ (see build_llama_android.sh)
#   * cmake + ninja on PATH.
#
# Output binaries link against ../../llama.cpp/build-android/bin/lib*.so via
# $ORIGIN-rooted rpath, so deployment is: copy the four .so files into the
# same dir as the binary on /data/local/tmp/endurkv/bin/.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ANDROID_NDK="${ANDROID_NDK:-$WORKSPACE/toolchain/android-ndk-r27c}"
LLAMA_DIR="$WORKSPACE/EndurKV/llama.cpp"
LLAMA_BUILD="$LLAMA_DIR/build-android"
PROBE_SRC="$WORKSPACE/EndurKV/entropy_probe"
PROBE_BUILD="$PROBE_SRC/build-android"

if [ ! -d "$ANDROID_NDK/build/cmake" ]; then
    echo "ERROR: Android NDK not found at $ANDROID_NDK" >&2
    echo "Set \$ANDROID_NDK or place NDK r27c at $ANDROID_NDK" >&2
    exit 1
fi
if [ ! -f "$LLAMA_BUILD/bin/libllama.so" ]; then
    echo "ERROR: $LLAMA_BUILD/bin/libllama.so not found." >&2
    echo "Run scripts/android/build_llama_android.sh first." >&2
    exit 1
fi

echo "[probe-android] workspace:    $WORKSPACE"
echo "[probe-android] ndk:          $ANDROID_NDK"
echo "[probe-android] llama build:  $LLAMA_BUILD"
echo "[probe-android] probe build:  $PROBE_BUILD"

mkdir -p "$PROBE_BUILD"
cd "$PROBE_BUILD"

cmake "$PROBE_SRC" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DLLAMA_CPP_DIR="$LLAMA_DIR" \
  -DLLAMA_BUILD_DIR="$LLAMA_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PROBE=ON

ninja -j8

echo
echo "[probe-android] artifacts:"
ls -lh entropy_probe attention_probe prune_probe 2>/dev/null || true
file entropy_probe 2>/dev/null || true
