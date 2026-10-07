#!/usr/bin/env python3
# Runs the SnapKV selection rule in HuggingFace and counts the cache entries left
# under two storage layouts, from the same keep-sets: per KV head (HuggingFace) and
# one cell array shared by all heads and layers (llama.cpp, where a cell survives
# if any selector keeps it).
import argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="microsoft/Phi-3-mini-128k-instruct")
ap.add_argument("--prompt", required=True)
ap.add_argument("--budget", type=int, default=1024)   # SnapKV max_capacity_prompt
ap.add_argument("--window", type=int, default=32)     # observation window
ap.add_argument("--kernel", type=int, default=7)      # avgpool kernel
ap.add_argument("--max-tokens", type=int, default=12288)
ap.add_argument("--out", default="/tmp/hf_snapkv.json")
a = ap.parse_args()

# No trust_remote_code: the bundled Phi-3 code calls DynamicCache.from_legacy_cache(),
# which transformers 5.x removed. The built-in Phi3 model is used instead.
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(
    a.model, dtype=torch.float16, device_map="cuda",
    attn_implementation="eager")     # eager returns the attention weights,
model.eval()                          # which SnapKV needs

text = open(a.prompt, errors="replace").read()
ids = tok(text, return_tensors="pt").input_ids[:, :a.max_tokens].cuda()
N = ids.shape[1]
print("model=%s  prompt=%d tokens  budget K=%d  window=%d kernel=%d"
      % (a.model, N, a.budget, a.window, a.kernel))

with torch.no_grad():
    out = model(ids, output_attentions=True, use_cache=True)

L = len(out.attentions)
_, H, _, _ = out.attentions[0].shape
HKV = getattr(model.config, "num_key_value_heads", H)
GRP = H // HKV
print("layers=%d  query heads=%d  KV heads=%d  (GQA group=%d)" % (L, H, HKV, GRP))
# SnapKV selects per query head but the cache is stored per KV head, so even in
# HuggingFace the GRP query heads sharing a KV head are unioned (4 on Llama-3.2-1B).

W = min(a.window, N)
prefix = N - W
K_prefix = max(0, min(a.budget - W, prefix))
pool = torch.nn.AvgPool1d(a.kernel, stride=1, padding=a.kernel // 2)

hf_entries = []             # what HF physically stores: per KV head, union over its GQA group
union_per_layer = []        # what llama.cpp must hold: union over ALL heads, per layer
for l in range(L):
    attn = out.attentions[l][0]                       # [H, q, kv]
    obs = attn[:, -W:, :prefix].mean(dim=1)           # observation-window attention
    scored = pool(obs.unsqueeze(1).float()).squeeze(1)[:, :prefix]
    layer_union = set(range(prefix, N))                # window kept by every head
    head_keep = []
    for h in range(H):
        idx = torch.topk(scored[h], min(K_prefix, prefix)).indices.tolist()
        head_keep.append(set(idx))
        layer_union.update(idx)
    union_per_layer.append(len(layer_union))
    for g in range(HKV):                               # HF: one tensor per KV head
        grp = set()
        for h in range(g * GRP, (g + 1) * GRP):
            grp |= head_keep[h]
        hf_entries.append(len(grp) + W)

seq_union = max(union_per_layer)   # llama.cpp frees a cell only if NO layer keeps it
hf_entries_per_head = sum(hf_entries) / len(hf_entries)

res = dict(model=a.model, prompt_tokens=N, budget=a.budget, layers=L, heads=H, kv_heads=HKV, gqa_group=GRP,
           hf_mean_entries_per_head=hf_entries_per_head,
           hf_reduction=N / hf_entries_per_head,
           llamacpp_union_cells=seq_union,
           llamacpp_reduction=N / seq_union,
           union_per_layer_min=min(union_per_layer), union_per_layer_max=max(union_per_layer))
json.dump(res, open(a.out, "w"), indent=1)

print("SAME selection, two storage layouts")
print("HuggingFace  (per-KV-head tensors): %8.1f entries/head -> %5.2fx reduction"
      % (hf_entries_per_head, N / hf_entries_per_head))
print("llama.cpp    (shared cell array): %8d cells        -> %5.2fx reduction"
      % (seq_union, N / seq_union))
print("union per layer: min %d, max %d of %d prompt tokens"
      % (min(union_per_layer), max(union_per_layer), N))
print("-> the budget is realizable per-head; the union is what a sequence-level cache must keep")
