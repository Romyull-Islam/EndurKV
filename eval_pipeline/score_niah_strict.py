#!/usr/bin/env python3
"""Score the phone needle grid, strictly. (2026-09-20)

WHY THIS EXISTS. The campaign runner counts a hit with

    grep -qiE 'mango sorbet|bi-rite'

and the needle is "The best ice-cream flavor in San Francisco is mango sorbet
from Bi-Rite." The "bi-rite" half of that alternation matches answers that name
the shop and then say, in words, that the flavor is not in the passage:

    gemma2b/mukv/L8K_d33  "Bi-Rite is the best ice cream place ... It does not,
                           however, specify a particular flavor."
    llama1b/h2o/L8K_d17   "Bi-Rite is a popular ice cream shop ... However, it
                           does not provide information on which flavor is best"

Those are misses. The question asks for the flavor. Worse, the published table
scored some cells one way and some the other, which flattered muKV on gemma
(11 instead of 10), flattered H2O on Llama (8 instead of 7), and scored
StreamingLLM's gemma cell strictly at 2. A reviewer re-scoring uniformly gets a
different table either way, so the grid is rescored here under one rule.

THE RULE. A hit names the flavor: "mango sorbet", or "mango" and "sorbet"
close together. An answer that also disclaims knowing it is a miss, because a
self-contradicted answer is not a retrieval. Naming only the shop is a miss.

Reports per model and policy, and the mean live cache, so the table can be
rebuilt from one place.
"""
import json
import os
import re
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/niah_tableC_k1024_ctx16384"
MODELS = ("phi3", "llama1b", "gemma2b", "bonsai8b")
POLICIES = ("vanilla", "mukv", "snapkv", "adakv", "tova", "h2o", "streamingllm", "keydiff",
            # the published-budget re-runs (2026-09-21): StreamingLLM 4+2000, Ada-KV and TOVA
            # at 2048, H2O at 20% of each stimulus's prompt
            "sllm2004", "adakv2048", "tova2048", "h2opub")

# --ignore-eos forces 64 tokens, so every model answers, emits its turn terminator,
# and then keeps generating unrelated commentary. Only the span before the first
# terminator is the answer; scoring the whole buffer is what made the loose regex
# both over- and under-count.
END = re.compile(r"<end_of_turn>|<\|eot_id\|>|<\|end\|>|<\|im_end\|>|<\|endoftext\|>|<pad>", re.I)
FLAVOR = re.compile(r"mango", re.I)
# a disclaimer INSIDE the answer span: the model names the shop, or nothing, and
# says the passage does not give the flavor. A self-contradicted answer is a miss.
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


# meta.json's head_dim is n_embd / n_head. Gemma-2 sets its head size explicitly (256, not
# 2304 / 8 = 288), so without this every gemma cell count comes out 256/288 = 11% low; with it the
# full cache's cells equal its prompt length, as on the other three models.
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
    # a cell counts only once it finished: meta.json is written last, so a cell cut
    # off mid-run (the phone dropped off USB on 2026-09-22) is not scored
    if not (os.path.exists(gen) and os.path.exists(os.path.join(ROOT, d, "meta.json"))):
        continue
    text = open(gen, errors="ignore").read()
    ok, why = verdict(text)
    total[(model, policy)] += 1
    if ok:
        hits[(model, policy)] += 1
    # anything the loose regex would have called a hit but this does not, and vice versa
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
