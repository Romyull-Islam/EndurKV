#!/usr/bin/env python3
"""Cells the allocator keeps against the budget a policy asks for, phone GPU.

Job: show in one picture that a per-head budget does not materialize on a shared cell
array. Form: lines over the nominal budget K, one per policy, against the "budget
honoured" diagonal. Per-head policies share one hue family (they fail the same way);
the two sequence-level policies get their own hues. Markers differ per policy so the
chart reads without colour, and every line is labelled at its right end.

Data: /tmp/claims_data.json (claims_data.py), realized_cells, each cell read from the
run's own [cache] retained_kv line (32 KiB per cell, Llama-3.2-1B f16), 9741-cell prompt,
Adreno 840, phone queue of 2026-09-17/18. StreamingLLM sits on the diagonal at
every budget (its published 4+2000 keeps 2004).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open("/tmp/claims_data.json"))["summary"]["realized_cells"]
N_FULL = 9741

INK, MUTED, GRID, AXIS = "#1a1a1a", "#6b6b6b", "#e6e6e6", "#b8b8b8"
PH = {"snapkv": ("#8c2d04", "X", "SnapKV"), "h2o": ("#cc4c02", "s", "H2O"),
      "tova": ("#ec7014", "^", "TOVA"), "adakv": ("#fe9929", "D", "Ada-KV")}
SEQ = {"sllm": ("#0072B2", "o", "StreamingLLM"), "mukv": ("#009E73", "o", "$\\mu$KV")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "axes.labelsize": 7.4,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "pdf.fonttype": 42})

fig, ax = plt.subplots(figsize=(3.33, 1.85))
ks = [256, 512, 1024, 2048]
ax.plot([0, 2600], [0, 2600], color="#8a8a8a", lw=0.9, ls=(0, (3, 2)), zorder=4)
ax.text(2590, 2780, "cells = $K$", color=MUTED, fontsize=6.2, ha="right", va="bottom")
ax.axhline(N_FULL, color=AXIS, lw=0.7, zorder=1)
ax.text(40, N_FULL + 150, "full cache, 9741 cells", color=MUTED, fontsize=6.2, va="bottom")

def series(key):
    xs, ys = [], []
    for k in ks:
        v = S.get(f"{key}_k{k}")
        if v: xs.append(k); ys.append(v)
    o = sorted(zip(xs, ys)); return [a for a, b in o], [b for a, b in o]

ends = {}
for key, (c, m, lab) in {**PH, **SEQ}.items():
    xs, ys = series(key)
    ax.plot(xs, ys, color=c, lw=1.3, marker=m, ms=3.6, mec="white", mew=0.5, zorder=3)
    ends[key] = (xs[-1], ys[-1], c, lab)

# right-end labels, nudged apart where lines end close together
nudge = {"snapkv": 250, "h2o": 150, "tova": 0, "adakv": -250, "sllm": -120, "mukv": -300}
for key, (x, y, c, lab) in ends.items():
    ax.text(x + 55, y + nudge[key], lab, color=c, fontsize=6.4, va="center", ha="left",
            weight="bold" if key == "mukv" else "normal")

ax.set_xlim(0, 2600); ax.set_ylim(0, 10600)
ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000]); ax.set_yticklabels(["0", "2K", "4K", "6K", "8K", "10K"])
ax.set_xlabel("budget asked for, $K$ cells"); ax.set_ylabel("cells live after prefill")
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, length=2)
fig.tight_layout(pad=0.3)
out = os.path.join(HERE, "fig_budget_sweep")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
print("wrote", out, {k: series(k) for k in list(PH) + list(SEQ)})
