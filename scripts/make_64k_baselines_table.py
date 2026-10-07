#!/usr/bin/env python3
# Table for the 64K baseline campaigns (desktop CUDA, so no energy column):
#   nolimit: every policy gets K = n_ctx, native: each policy at its own published budget.
# retain and cmpct show whether a policy actually freed and compacted cache.
# PPL is read only from ppl_* cells. gen_* meta.json also has "perplexity", but that is
# the model's perplexity on its own generated tokens, not on the held-out text.
import json, os, re, sys, math

# Llama-3.2-1B: 16 layers x 8 KV heads x 64 dim, K and V.
CELL_F16 = 16 * 8 * 64 * 2 * 2.0
CELL_Q80 = 16 * 8 * 64 * 2 * (34.0 / 32.0)

CAMPAIGNS = [
    ("/tmp/rtx64k_nolimit", "NO LIMIT -- every policy gets K = n_ctx (65536); what it still drops is its own",
     [("vanilla", "vanilla (full cache)"), ("mukv", r"muKV"), ("snapkv", "SnapKV"),
      ("adakv", "Ada-KV"), ("h2o", "H2O"), ("tova", "TOVA"),
      ("streamingllm", "StreamingLLM")]),
    ("/tmp/rtx64k_native", "NATIVE BUDGETS -- every policy at ITS OWN paper's operating point",
     [("vanilla", "vanilla (full cache)"), ("mukv_nolimit", r"muKV  K=free"),
      ("mukv_k1024", r"muKV  K=1024"), ("mukv_matched", r"muKV  K=matched"),
      ("snapkv", "SnapKV  2048/head"), ("adakv", "Ada-KV  2048"),
      ("h2o", "H2O  20% of N"), ("tova", "TOVA  2048"),
      ("streamingllm", "StreamingLLM  2048"), ("streamingllm_fa", "StreamingLLM (FA-on)"),
      ("sllm_compact", "StreamingLLM +compact")]),
]


def load(p):
    if not os.path.exists(p):
        return None
    s = open(p).read()
    s = re.sub(r':\s*-?nan\b', ': NaN', s)
    s = re.sub(r':\s*-?inf\b', ': Infinity', s)
    try:
        return json.loads(s)
    except Exception:
        return None


def ppl_of(base, tag):
    """Held-out perplexity from the ppl_ cell, not gen_'s self-perplexity."""
    j = load(os.path.join(base, "ppl_" + tag, "meta.json"))
    if not j:
        return None
    p = j.get("perplexity")
    if not p and j.get("mean_nll") is not None:
        try:
            p = math.exp(j["mean_nll"])
        except OverflowError:
            return None
    return p


HDR = ("%-24s %8s %8s %8s %8s %8s %7s %8s %7s %8s %7s %6s %8s" %
       ("policy", "PPL", "prefill", "decode", "wall", "tok/s", "dec x",
        "cells", "kept%", "evicted", "retain", "cmpct", "peakRSS"))

for base, title, arms in CAMPAIGNS:
    if not os.path.isdir(base):
        continue
    print("%s" % title)
    print("    %s   (Llama-3.2-1B Q4_K_M, ctx 65536, 57118-token prompt)" % os.path.basename(base))
    print(HDR)
    print("-" * len(HDR))
    v = load(os.path.join(base, "gen_vanilla", "meta.json"))
    vdec = v["decode_ms"] if v else None
    # cell size differs per campaign (f16 vs q8_0), infer it from vanilla
    cell = CELL_F16
    if v and v.get("retained_kv_bytes"):
        est = v["retained_kv_bytes"] / max(v.get("n_prompt_tokens") or 1, 1)
        cell = CELL_Q80 if abs(est - CELL_Q80) < abs(est - CELL_F16) else CELL_F16
    for tag, lab in arms:
        g = load(os.path.join(base, "gen_" + tag, "meta.json"))
        if not g:
            continue
        p = ppl_of(base, tag)
        cells = (g.get("retained_kv_bytes") or 0) / cell
        vc = (v.get("retained_kv_bytes") or 1) / cell if v else 1
        print("%-24s %8s %7.1fs %7.1fs %7.1fs %8.2f %6.2fx %8.0f %6.1f%% %8d %7.2f %6s %7.2fG" % (
            lab,
            ("%.3f" % p) if p else "  --  ",
            g["prefill_ms"] / 1000, g["decode_ms"] / 1000, g["total_ms"] / 1000,
            g.get("decode_tps") or 0,
            (vdec / g["decode_ms"]) if vdec else 0,
            cells, 100.0 * cells / max(vc, 1),
            g.get("evicted_prefill") or 0,
            g.get("mean_retention_ratio") if g.get("mean_retention_ratio") is not None else -1,
            "yes" if g.get("compaction_applied") else "NO",
            (g.get("peak_rss_kb") or 0) / 1048576.0))

print()
print("retain = mean_retention_ratio: 1.00 means the policy freed NOTHING despite its budget.")
print("cmpct  = engine actually made survivors contiguous. Small nominal budget + cmpct=NO")
print("         means the policy pays eviction's cost and collects none of its benefit.")
print("PPL    = held-out, wiki_eval_disjoint_64k.txt, asserted 0/119 disjoint from the prompt.")
