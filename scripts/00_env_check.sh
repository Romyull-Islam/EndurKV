#!/usr/bin/env bash
# Phase 0 environment probe for EndurKV.
# Run from the EndurKV root:  bash scripts/00_env_check.sh
# Then paste the entire output back to Claude.

set +e
sep() { printf '\n=== %s ===\n' "$1"; }

sep "System"
uname -a
echo "HOSTNAME=$(hostname)"
echo "KERNEL=$(uname -r)"
echo "PWD=$(pwd)"
echo "HOME=$HOME"

sep "C/C++ toolchain"
command -v gcc   && gcc   --version | head -1
command -v g++   && g++   --version | head -1
command -v clang && clang --version | head -1 || echo "clang: not found (ok if gcc works)"

sep "CMake / make / git / pkg-config"
command -v cmake      && cmake      --version | head -1
command -v make       && make       --version | head -1
command -v git        && git        --version
command -v pkg-config && pkg-config --version

sep "Python"
command -v python3 && python3 --version
python3 -c "import sys; print('exec:', sys.executable)"
python3 -m pip --version 2>&1 | head -1
echo "-- packages --"
python3 - <<'PY'
import importlib
for m in ["numpy","pandas","matplotlib","scipy","seaborn","tqdm","datasets","huggingface_hub"]:
    try:
        x = importlib.import_module(m)
        print(f"  {m}: {getattr(x,'__version__','?')}")
    except ImportError:
        print(f"  {m}: MISSING")
PY

sep "Disk space"
df -h "$HOME" . 2>/dev/null | awk 'NR<=3'
echo "Free in $(pwd): $(df -h . | awk 'NR==2{print $4}')"

sep "CPU / memory"
echo "nproc=$(nproc)"
grep -m1 "model name" /proc/cpuinfo
free -h | head -2

sep "GPU"
command -v nvcc       && nvcc       --version | tail -2 || echo "nvcc: not in PATH"
command -v nvidia-smi && nvidia-smi -L          || echo "nvidia-smi: not in PATH"

sep "llama.cpp state"
if [ -d llama.cpp/.git ]; then
    echo "commit: $(git -C llama.cpp rev-parse HEAD)"
    echo "branch: $(git -C llama.cpp rev-parse --abbrev-ref HEAD)"
    echo "subject: $(git -C llama.cpp log -1 --oneline)"
    echo "remote: $(git -C llama.cpp remote -v | head -1)"
    echo "dirty (first 5 lines):"
    git -C llama.cpp status --short | head -5
    echo "build dirs present:"
    ls -d llama.cpp/build* 2>/dev/null || echo "  (none)"
else
    echo "llama.cpp not present at ./llama.cpp"
fi

sep "Models"
if [ -d models ]; then
    ls -la models/ 2>/dev/null | head -20
else
    echo "no ./models dir yet"
fi

sep "Done"
echo "Paste this entire output back to Claude (everything from the first '=== System ===' to here)."
