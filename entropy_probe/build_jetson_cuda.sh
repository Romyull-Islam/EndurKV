#!/bin/bash
# Build eviction_bench for NVIDIA Jetson Orin (CUDA, SM_87). Run ON the Jetson.
# Layout: <root>/llama.cpp and <root>/entropy_probe side by side (as in ~/ukv/code).
# Profile B of PLATFORMS.md. Verified 2026-07-24 on Orin NX 16GB, JetPack R36.4.7, CUDA 12.6.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH=/usr/local/cuda/bin:$PATH

cmake -S "$ROOT/llama.cpp" -B "$ROOT/llama.cpp/build-jetson-cuda" \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/llama.cpp/build-jetson-cuda" -j"$(nproc)"

cmake -S "$ROOT/entropy_probe" -B "$ROOT/entropy_probe/build-jetson-cuda" \
      -DLLAMA_CPP_DIR="$ROOT/llama.cpp" \
      -DLLAMA_BUILD_DIR="$ROOT/llama.cpp/build-jetson-cuda"
cmake --build "$ROOT/entropy_probe/build-jetson-cuda" -j"$(nproc)" --target eviction_bench

echo "OK: $ROOT/entropy_probe/build-jetson-cuda/eviction_bench"
sha256sum "$ROOT/entropy_probe/build-jetson-cuda/eviction_bench"
