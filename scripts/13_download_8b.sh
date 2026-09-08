#!/usr/bin/env bash
# Download Llama-3.1-8B-Instruct Q4_K_M GGUF (~5 GB).
# Run from EndurKV root:  bash scripts/13_download_8b.sh
set -e
cd "$(dirname "$0")/.."

mkdir -p models
MODEL_PATH=models/Llama-3.1-8B-Instruct-Q4_K_M.gguf
PRIMARY_URL=https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

if [ -f "$MODEL_PATH" ] && [ -s "$MODEL_PATH" ]; then
    echo "Model already present: $MODEL_PATH"
    ls -lh "$MODEL_PATH"
    [ -f "${MODEL_PATH}.sha256" ] && cat "${MODEL_PATH}.sha256" || sha256sum "$MODEL_PATH" | tee "${MODEL_PATH}.sha256"
    exit 0
fi

echo "Downloading from $PRIMARY_URL ..."
echo "  (~5 GB.  If this 401s, the bartowski mirror has gone gated; tell Claude.)"
wget --progress=bar:force -O "${MODEL_PATH}.tmp" "$PRIMARY_URL"
mv "${MODEL_PATH}.tmp" "$MODEL_PATH"

echo
ls -lh "$MODEL_PATH"
echo "Computing sha256..."
sha256sum "$MODEL_PATH" | tee "${MODEL_PATH}.sha256"
echo "Done."
