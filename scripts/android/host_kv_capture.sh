#!/bin/bash
# Host-side K/V capture: runs attention_probe on x86_64 host with
# ATTNPROBE_CAPTURE_KV=1 so each prompt yields BOTH .attn.bin and .kv.bin.
#
# Usage: host_kv_capture.sh <model_name> [<prompt_id>]
#   model_name in {phi3, mistral, qwen2, gemma2, r1distill}
#   prompt_id defaults to narrativeqa_pub_001
#
# Output goes to /home/mislam22/EndurKV_workspace/logs/host_kv_<model>/
set -e

MODEL="${1:?Usage: $0 <model_name> [<prompt_id>]}"
PROMPT_ID="${2:-narrativeqa_pub_001}"

case "$MODEL" in
  phi3)      MODEL_PATH=/home/mislam22/EndurKV_workspace/models/Phi-3-mini-4k-instruct-Q4_K_M.gguf ;;
  mistral)   MODEL_PATH=/home/mislam22/EndurKV_workspace/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf ;;
  qwen2)     MODEL_PATH=/home/mislam22/EndurKV_workspace/models/Qwen2-7B-Instruct-Q4_K_M.gguf ;;
  gemma2)    MODEL_PATH=/home/mislam22/EndurKV_workspace/models/gemma-2-2b-it-Q4_K_M.gguf ;;
  r1distill) MODEL_PATH=/home/mislam22/EndurKV_workspace/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf ;;
  *) echo "unknown model: $MODEL" >&2; exit 1 ;;
esac

PROMPT_TXT="/tmp/${PROMPT_ID}.txt"
if [ ! -f "$PROMPT_TXT" ]; then
  # extract from JSONL
  /home/mislam22/EndurKV_workspace/.venv/bin/python -c "
import json
target='$PROMPT_ID'
with open('/home/mislam22/EndurKV_workspace/prompts/prompts_pub_longbench.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        if rec['prompt_id'] == target:
            with open('$PROMPT_TXT','w') as out: out.write(rec['prompt_text'])
            print(f'wrote $PROMPT_TXT ({len(rec[\"prompt_text\"])} chars)')
            break
"
fi

OUT_DIR=/home/mislam22/EndurKV_workspace/logs/host_kv_${MODEL}
mkdir -p "$OUT_DIR"

PROBE=/home/mislam22/EndurKV_workspace/EndurKV/entropy_probe/build-host/attention_probe
LIBS=/home/mislam22/EndurKV_workspace/EndurKV/llama.cpp/build/bin

echo "[host_kv] model=$MODEL prompt=$PROMPT_ID out=$OUT_DIR"
echo "[host_kv] starting at $(date)"
START=$(date +%s)

ATTNPROBE_CAPTURE_KV=1 LD_LIBRARY_PATH="$LIBS" "$PROBE" \
  --model "$MODEL_PATH" \
  --prompt-file "$PROMPT_TXT" \
  --prompt-id "$PROMPT_ID" \
  --max-tokens 64 \
  --output "$OUT_DIR/${PROMPT_ID}.entropy.csv" \
  --output-attn "$OUT_DIR/${PROMPT_ID}.attn.bin" \
  --seed 42 \
  --n-gpu-layers 0

END=$(date +%s)
echo "[host_kv] done in $((END-START))s"
ls -lah "$OUT_DIR"
