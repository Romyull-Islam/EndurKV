#!/usr/bin/env bash
# Phase B — build entropy_probe (probe-on variant) into entropy_probe/build/.
# Run from the EndurKV root:  bash scripts/04_build_probe.sh

set -e
cd "$(dirname "$0")/.."

ROOT="$(pwd)"
SRC="$ROOT/entropy_probe"
BUILD="$SRC/build"

echo "=== configure (ENABLE_PROBE=ON) ==="
cmake -S "$SRC" -B "$BUILD" \
    -DLLAMA_CPP_DIR="$ROOT/llama.cpp" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_PROBE=ON

echo
echo "=== build ==="
cmake --build "$BUILD" -j

echo
echo "=== result ==="
ls -lh "$BUILD/entropy_probe"
file "$BUILD/entropy_probe"
echo
echo "Linked libraries:"
ldd "$BUILD/entropy_probe" | grep -E "llama|ggml" || true
