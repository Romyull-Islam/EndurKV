#!/usr/bin/env python3
"""Strict scorer for the phone needle grid. A hit names the flavor (mango) in the answer span
and does not then disclaim it. Naming only the shop (Bi-Rite) is a miss, unlike the runner's
loose 'mango sorbet|bi-rite' grep. Prints hits per model and policy, mean live cache, and disagreements.
Usage: score_niah_strict.py [ROOT]
"""
import json
import os
import re
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/niah_tableC_k1024_ctx16384"
MODELS = ("phi3", "llama1b", "gemma2b", "bonsai8b")
POLICIES = ("vanilla", "mukv", "snapkv", "adakv", "tova", "h2o", "streamingllm", "keydiff",
            # published budgets: StreamingLLM 4+2000, Ada-KV and TOVA 2048, H2O 20% of the prompt
            "sllm2004", "adakv2048", "tova2048", "h2opub")

# --ignore-eos makes models keep generating after their turn terminator, so only the
# span before the first terminator is scored as the answer.
END = re.compile(r"<end_of_turn>|<\|eot_id\|>|<\|end\|>|<\|im_end\|>|<\|endoftext\|>|<pad>", re.I)
FLAVOR = re.compile(r"mango", re.I)
# disclaimer inside the answer span (the passage "does not specify" the flavor), counted as a miss
DISCLAIM = re.compile(
    r"(does not|doesn't|didn't|do not)\s+\w{0,12}\s*(specify|mention|say|state|provide|give|indicate|contain)"
    r"|trick question",
    re.I,
)


def answer_span(text):
    """The model's actual answer: everything before its first turn terminator."""
    m = END.search(text)
    return text[: m.start()] if m else text


def verdict(text):
    """(hit, why). Judged only on the answer span."""
    span = answer_span(text)
    if not FLAVOR.search(span):
        return False, "flavor not named in the answer"
    if DISCLAIM.search(span):
        return False, "named the flavor then disclaimed it"
    return True, "hit"


# meta.json's head_dim is n_embd / n_head, but Gemma-2 sets its head size to 256 (not 288),
# which would make its cell counts 11% low.
HEAD_DIM = {"gemma2b": 256}


def cells_of(meta, model=None):
    rb, hd, nl, nk = (meta.get(k) for k in ("retained_kv_bytes", "head_dim", "n_layers", "n_kv_heads"))
    hd = HEAD_DIM.get(model, hd)
    if not (rb and hd and nl and nk):
        return None
    return rb / (2 * 2 * hd * nl * nk)


hits = defaultdict(int)
total = defaultdict(int)
cells = defaultdict(list)
flagged = []

for d in sorted(os.listdir(ROOT)):
    parts = d.split("__")
    if len(parts) < 3:
        continue
    model, policy = parts[0], parts[1]
    gen = os.path.join(ROOT, d, "gen.txt")
    # meta.json is written last, so cells cut off mid-run are not scored
    if not (os.path.exists(gen) and os.path.exists(os.path.join(ROOT, d, "meta.json"))):
        continue
    text = open(gen, errors="ignore").read()
    ok, why = verdict(text)
    total[(model, policy)] += 1
    if ok:
        hits[(model, policy)] += 1
    # cells where the loose regex and the strict rule disagree
    loose = bool(re.search(r"mango sorbet|bi-rite", text, re.I))
    if loose != ok:
        flagged.append((d, "loose=HIT strict=miss" if loose else "loose=miss strict=HIT", why,
                        " ".join(text.split())[:110]))
    m = os.path.join(ROOT, d, "meta.json")
    if os.path.exists(m):
        try:
            c = cells_of(json.loads(re.sub(r":\s*-?nan\b", ": NaN", open(m).read())), model)
            if c:
                cells[policy].append(c)
        except Exception:
            pass

print(f"{'policy':14s} " + " ".join(f"{m:>9s}" for m in MODELS) + "   total     cells")
for pol in POLICIES:
    row = [hits[(m, pol)] for m in MODELS]
    n = [total[(m, pol)] for m in MODELS]
    if not sum(n):
        continue
    c = sum(cells[pol]) / len(cells[pol]) if cells[pol] else float("nan")
    print(f"{pol:14s} " + " ".join(f"{h:5d}/{t:<3d}" for h, t in zip(row, n))
          + f"  {sum(row):3d}/{sum(n):<3d} {c:9.0f}")

print(f"\ncells where the loose and strict rules disagree ({len(flagged)}):")
for d, kind, why, snippet in flagged:
    print(f"  {d}\n      {kind} ({why})\n      {snippet}")
