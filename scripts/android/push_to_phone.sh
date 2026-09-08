#!/usr/bin/env bash
# push_to_phone.sh — push the vanilla llama.cpp deployment + bench scripts to the phone.
#
# Run from your LAPTOP (not this host PC) — the host where adb is connected to the OnePlus.
# This script is a TEMPLATE — copy it to wherever adb lives, edit ARTIFACTS to be the
# local path where you rsync'd the host PC's phone-deploy/ tree.
#
# Expected phone layout after push (under /data/local/tmp/endurkv/):
#   bin/                 llama-completion, llama-perplexity, llama-bench
#                        libllama.so, libggml.so, libggml-base.so, libggml-cpu.so
#                        libomp.so, attention_probe (for v1 capture later)
#   scripts/             phone_bench_vanilla.sh, phone_bench_perplexity.sh,
#                        sample_sensors.sh
#   models/              (you populate this — gguf files)
#   prompts/             (you populate this — *.txt files extracted from JSONL)
#   corpora/             (optional — wiki.test.raw for perplexity)
#   logs/                (empty, fills during runs)
set -e

PHONE_ROOT="/data/local/tmp/endurkv"
ARTIFACTS="${ARTIFACTS:-./phone-deploy-stage}"

echo "[push] target: $PHONE_ROOT"
echo "[push] from:   $ARTIFACTS"

# Verify phone is connected
adb wait-for-device
adb shell "id" || { echo "no adb device"; exit 1; }

# Setup root
adb shell "mkdir -p $PHONE_ROOT/{bin,scripts,models,prompts,corpora,logs}"

# Push binaries — resolve symlinks before copying (-L)
echo "[push] binaries ..."
for f in llama-completion llama-perplexity llama-bench attention_probe \
         libllama.so libggml.so libggml-base.so libggml-cpu.so libomp.so; do
    if [ -f "$ARTIFACTS/bin/$f" ] || [ -L "$ARTIFACTS/bin/$f" ]; then
        adb push -p "$ARTIFACTS/bin/$f" "$PHONE_ROOT/bin/$f"
        adb shell "chmod 755 $PHONE_ROOT/bin/$f"
    fi
done

echo "[push] scripts ..."
for f in phone_bench_vanilla.sh phone_bench_perplexity.sh sample_sensors.sh \
         run_one_prompt.sh; do
    if [ -f "$ARTIFACTS/scripts/$f" ]; then
        adb push -p "$ARTIFACTS/scripts/$f" "$PHONE_ROOT/scripts/$f"
        adb shell "chmod 755 $PHONE_ROOT/scripts/$f"
    fi
done

# Optional: models, prompts, corpora — only push if present locally
for sub in models prompts corpora; do
    if [ -d "$ARTIFACTS/$sub" ] && [ -n "$(ls -A "$ARTIFACTS/$sub" 2>/dev/null)" ]; then
        echo "[push] $sub ..."
        for f in "$ARTIFACTS/$sub"/*; do
            adb push -p "$f" "$PHONE_ROOT/$sub/$(basename "$f")"
        done
    fi
done

# Verify
echo ""
echo "[push] DONE — on-device contents:"
adb shell "ls -lh $PHONE_ROOT/bin/" | head -15
adb shell "ls $PHONE_ROOT/scripts/"
adb shell "file $PHONE_ROOT/bin/llama-completion" 2>&1 | head -1

echo ""
echo "==============================================================="
echo "Next on phone (over adb shell):"
echo ""
echo "  adb shell"
echo "  cd $PHONE_ROOT"
echo "  sh scripts/phone_bench_vanilla.sh \\"
echo "      --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \\"
echo "      --prompt prompts/narrativeqa_lc_01.txt \\"
echo "      --prompt-id narrativeqa_lc_01 \\"
echo "      --ctx-size 4096 \\"
echo "      --max-tokens 64 \\"
echo "      --out-dir logs/vanilla_run_\$(date +%s)"
echo ""
echo "Pull results back with:"
echo "  adb pull $PHONE_ROOT/logs ./phone-logs/"
echo "==============================================================="
