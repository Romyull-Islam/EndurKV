#!/usr/bin/env python3
"""Wrap each LongBench prompt in each model's chat template, written to
prompts_chat/<model_tag>/<prompt_id>.txt. Instruct models given raw text produce degenerate output."""
import sys
from pathlib import Path

SRC = Path("/home/mislam22/EndurKV_workspace/phone-deploy/prompts/longbench")
DST = Path("/home/mislam22/EndurKV_workspace/EndurKV/prompts_chat")

TEMPLATES = {
    "Llama-3.2-1B": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "Gemma-2-2B":   "<bos><start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n",
    "Phi-3-128k":   "<|user|>\n{p}<|end|>\n<|assistant|>\n",
}

src_files = sorted(SRC.glob("*.txt"))
if not src_files:
    print(f"no prompts in {SRC}", file=sys.stderr); sys.exit(1)

for model_tag, tmpl in TEMPLATES.items():
    out_dir = DST / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in src_files:
        prompt_text = src.read_text(errors='replace')
        # Strip any trailing blank
        prompt_text = prompt_text.rstrip() + "\n"
        wrapped = tmpl.format(p=prompt_text)
        out = out_dir / src.name
        out.write_text(wrapped)
    print(f"wrote {len(src_files)} prompts to {out_dir}")
