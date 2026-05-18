#!/usr/bin/env bash
# Phase 0 — download Llama 3.2 1B Instruct Q4_K_M GGUF.
# Run from the EndurKV root:  bash scripts/02_download_model.sh
# Idempotent. Skips if the file already exists with the right name.

set -e
cd "$(dirname "$0")/.."

mkdir -p models
MODEL_PATH=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
PRIMARY_URL=https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf

if [ -f "$MODEL_PATH" ] && [ -s "$MODEL_PATH" ]; then
    echo "Model already present: $MODEL_PATH"
    ls -lh "$MODEL_PATH"
    if [ -f "${MODEL_PATH}.sha256" ]; then
        cat "${MODEL_PATH}.sha256"
    else
        echo "Computing sha256..."
        sha256sum "$MODEL_PATH" | tee "${MODEL_PATH}.sha256"
    fi
    exit 0
fi

echo "Downloading from $PRIMARY_URL ..."
echo "(If this 401s, the bartowski mirror has gone gated; tell Claude and we'll switch source.)"
wget --progress=bar:force -O "${MODEL_PATH}.tmp" "$PRIMARY_URL"
mv "${MODEL_PATH}.tmp" "$MODEL_PATH"

echo
echo "Downloaded:"
ls -lh "$MODEL_PATH"
echo "Computing sha256..."
sha256sum "$MODEL_PATH" | tee "${MODEL_PATH}.sha256"
echo "Done."
