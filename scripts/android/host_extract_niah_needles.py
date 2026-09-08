#!/usr/bin/env python3
"""Extract per-prompt needle positions from NIAH JSONL prompts.

Output schema (one JSON record per line):
  {"prompt_id": ..., "needle_char_offset": int, "prompt_chars": int,
   "needle_fraction": float in [0,1], "needle_phrase": str}

The downstream simulator multiplies `needle_fraction` by the captured n_kv
(from the .attn.bin sidecar header) to get the absolute KV index, giving
~1-2% char-vs-token positional accuracy. That's well within the granularity
needed for K-budget eviction hit-rate analysis (eviction operates on
positions, not subwords).
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True,
                    help="NIAH prompts JSONL (e.g. prompts/prompts_pub_niah.jsonl)")
    ap.add_argument("--out", required=True,
                    help="Output JSONL with per-prompt needle metadata")
    args = ap.parse_args()

    src = Path(args.prompts)
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 1

    n_total = n_found = n_missing = 0
    with src.open() as f_in, Path(args.out).open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            pid = r.get("prompt_id")
            prompt = r.get("prompt_text", "")
            meta = r.get("_meta", {}) or {}
            needle_phrase = meta.get("needle_phrase")
            if not needle_phrase:
                # Fall back to ground_truth substring if no explicit needle phrase
                needle_phrase = r.get("ground_truth", "")
            if not needle_phrase:
                n_missing += 1
                continue
            char_offset = prompt.find(needle_phrase)
            if char_offset < 0:
                n_missing += 1
                continue
            prompt_chars = len(prompt)
            needle_fraction = char_offset / prompt_chars if prompt_chars else 0.0
            f_out.write(json.dumps({
                "prompt_id": pid,
                "needle_char_offset": char_offset,
                "needle_char_end": char_offset + len(needle_phrase),
                "prompt_chars": prompt_chars,
                "needle_fraction": needle_fraction,
                "needle_phrase": needle_phrase,
                "depth_fraction": meta.get("depth_fraction"),
            }) + "\n")
            n_found += 1

    print(f"[needles] processed {n_total} prompts, found needles in {n_found}, "
          f"missing in {n_missing}. Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
