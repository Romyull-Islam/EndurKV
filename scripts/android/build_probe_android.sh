#!/usr/bin/env bash
# build_probe_android.sh: cross-compile entropy_probe, attention_probe and prune_probe for
# aarch64 into $WORKSPACE/EndurKV/entropy_probe/build-android/. Runs from anywhere.
# Needs Android NDK r27c ($ANDROID_NDK, default $WORKSPACE/toolchain/android-ndk-r27c),
# a finished llama.cpp Android build (build_llama_android.sh), and cmake plus ninja.
# The binaries load llama.cpp's lib*.so through an $ORIGIN rpath, so deploy those .so
# files next to the binary in /data/local/tmp/endurkv/bin/.

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

cmake "$PROBE_SRC" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DLLAMA_CPP_DIR="$LLAMA_DIR" \
  -DLLAMA_BUILD_DIR="$LLAMA_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PROBE=ON

cmake --build . -j8

echo
echo "[probe-android] artifacts:"
ls -lh entropy_probe attention_probe prune_probe controller_probe 2>/dev/null || true
file entropy_probe 2>/dev/null || true
