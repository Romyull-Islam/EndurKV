#!/usr/bin/env bash
# Phase 0 — create project venv and install missing Python deps.
# Run from the EndurKV root:  bash scripts/01_install_python_deps.sh
# Idempotent. Safe to re-run.

set -e
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    echo "Creating venv at ./.venv (with --system-site-packages so we inherit numpy/pandas/scipy/tqdm)..."
    python3 -m venv --system-site-packages .venv
else
    echo "Reusing existing ./.venv"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --quiet --upgrade pip
echo "Installing matplotlib, seaborn, datasets, huggingface_hub ..."
python -m pip install --quiet matplotlib seaborn datasets huggingface_hub

echo
echo "=== Resolved package versions ==="
python - <<'PY'
import importlib
for m in ["numpy","pandas","scipy","tqdm","matplotlib","seaborn","datasets","huggingface_hub"]:
    try:
        x = importlib.import_module(m)
        print(f"  {m:18s} {getattr(x,'__version__','?')}")
    except Exception as e:
        print(f"  {m:18s} FAILED ({e})")
PY

echo
echo "Activate with:  source .venv/bin/activate"
