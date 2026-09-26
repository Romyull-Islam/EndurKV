#!/usr/bin/env python3
"""Why server policies fail on a phone, in one column figure. (2026-09-24)

Two panels side by side, one per engine property, replacing the separate Figure 1 (backend
inversion) and Figure 3 (budget sweep) of the 6-page paper. Same data and encodings as
fig_backend_inversion.py and fig_budget_sweep.py; merged because the two floats cost 3.25 in
of column height and two captions, and the CFP counts references inside six pages.
  (a) decode speed of each published policy on the phone CPU and on the Adreno GPU, as a
      multiple of no eviction on the same backend (log axis; the sign flips with the backend).
  (b) cells the allocator actually keeps against the budget asked for, phone GPU: per-head
      policies keep 2.7 to 7.5 times their budget, sequence-level ones keep at most theirs.
Data: (a) the ratios of Tables 1 and 2, listed in R below; (b) /tmp/claims_data.json, written by
claims_data.py (summary.realized_cells).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID, AXIS = "#1a1a1a", "#6b6b6b", "#e6e6e6", "#b8b8b8"
CPU, GPU = "#0072B2", "#D55E00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.6, "axes.labelsize": 6.6,
                     "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "pdf.fonttype": 42})

fig, (a, b) = plt.subplots(1, 2, figsize=(3.33, 1.40), gridspec_kw=dict(width_ratios=[1.0, 1.05]))
fig.subplots_adjust(left=0.13, right=0.985, top=0.895, bottom=0.25, wspace=0.62)

# ---- (a) backend inversion: paired dots per policy, CPU and GPU, log x
# (policy, CPU decode speed as a multiple of the full cache, GPU the same), read from the paper's
# tables: CPU = Table 2's Llama-1B column, GPU = Table 1's matched-budget rows. Until 2026-09-25 the
# muKV CPU point was the July campaign's run (run_natural_cpu.sh, /tmp/nat_cpu, every policy on the
# same bin_cpu_v87 build): 24.21 / 5.02 = 4.82, a valid same-build ratio. Table 2 instead reports the
# current muKV build against its own same-build full cache (/tmp/qres_cpu, 23.755 / 5.459 = 4.35),
# so the figure now plots 4.35 to match the table and the muKV the paper describes.
R = [("$\\mu$KV", 4.35, 1.19), ("SnapKV", 1.35, 0.16), ("Ada-KV", 1.33, 0.16),
     ("TOVA", 1.19, 0.22), ("H2O", 1.16, 0.14)]
a.axvspan(0.08, 1.0, color="#f6f2ef", zorder=0)
a.axvline(1.0, color="#8f8f8f", lw=0.8, ls="--", zorder=1)
for i, (lab, c, g) in enumerate(R):
    y = len(R) - 1 - i
    xc = c
    a.plot([g, xc], [y, y], color="#d6d6d6", lw=1.4, zorder=2, solid_capstyle="round")
    a.scatter([g], [y], s=22, marker="X", color=GPU, edgecolor="white", lw=0.6, zorder=4)
    a.scatter([xc], [y], s=22, marker="o", color=CPU, edgecolor="white", lw=0.6, zorder=4)
a.set_yticks(range(len(R))); a.set_yticklabels([r[0] for r in R][::-1], fontsize=6.4)
a.set_xscale("log"); a.set_xticks([0.1, 0.25, 0.5, 1, 2, 5]); a.set_xticklabels(["0.1", "", "0.5", "1", "2", "5"])
a.set_xlim(0.08, 6.5); a.set_ylim(-0.45, len(R) - 0.4)
a.set_xlabel("decode speed, × no eviction")
a.text(0.93, len(R) - 0.45, "slower than\nthe full cache", fontsize=5.8, color=MUTED, ha="right", va="top", linespacing=1.2)
# direct labels on the top row instead of a legend: nothing left to collide with
top=len(R)-1
a.text(R[0][2], top-0.48, "GPU", color=GPU, fontsize=5.6, ha="center", va="top")
a.text(R[0][1], top-0.48, "CPU", color=CPU, fontsize=5.6, ha="center", va="top")
a.set_title("(a) same policy, two backends", loc="left", fontsize=6.3, color=INK, pad=3)
a.grid(axis="x", color=GRID, lw=0.5); a.set_axisbelow(True)
for s in ("top", "right", "left"): a.spines[s].set_visible(False)

# ---- (b) budget sweep: cells kept against the budget asked for, phone GPU
S = json.load(open("/tmp/claims_data.json"))["summary"]["realized_cells"]
N_FULL = 9741
PH = {"snapkv": ("#8c2d04", "X", "SnapKV"), "h2o": ("#cc4c02", "s", "H2O"),
      "tova": ("#ec7014", "^", "TOVA"), "adakv": ("#fe9929", "D", "Ada-KV")}
SEQ = {"sllm": ("#0072B2", "o", "SLLM"), "mukv": ("#009E73", "o", "$\\mu$KV")}
ks = [256, 512, 1024, 2048]
b.plot([0, 2600], [0, 2600], color="#8a8a8a", lw=0.8, ls=(0, (3, 2)), zorder=4)
b.text(2450, 2950, "cells = $K$", color=MUTED, fontsize=5.6, ha="right", va="bottom")
b.axhline(N_FULL, color=AXIS, lw=0.7, zorder=1)
b.text(40, N_FULL + 200, "full cache, 9741", color=MUTED, fontsize=5.6, va="bottom")
def series(key):
    pts = sorted((k, S[f"{key}_k{k}"]) for k in ks if S.get(f"{key}_k{k}"))
    return [p[0] for p in pts], [p[1] for p in pts]
ends = {}
for key, (c, m, lab) in {**PH, **SEQ}.items():
    xs, ys = series(key)
    b.plot(xs, ys, color=c, lw=1.1, marker=m, ms=3.0, mec="white", mew=0.4, zorder=3)
    ends[key] = (xs[-1], ys[-1], c, lab)
nudge = {"snapkv": 300, "h2o": 200, "tova": 0, "adakv": -320, "sllm": -80, "mukv": -420}
for key, (x, y, c, lab) in ends.items():
    b.text(x + 60, y + nudge[key], lab, color=c, fontsize=5.6, va="center", ha="left",
           weight="bold" if key == "mukv" else "normal")
b.set_xlim(0, 3150); b.set_ylim(0, 10800)
b.set_xticks([512, 1024, 2048]); b.set_xticklabels(["512", "1K", "2K"]); b.set_xticks([256], minor=True)
b.set_yticks([0, 2000, 4000, 6000, 8000, 10000]); b.set_yticklabels(["0", "2K", "4K", "6K", "8K", "10K"])
b.set_xlabel("budget asked for, $K$ cells"); b.set_ylabel("cells live after prefill", labelpad=1)
b.set_title("(b) budget vs cells kept", loc="left", fontsize=6.3, color=INK, pad=3)
b.grid(axis="y", color=GRID, lw=0.5); b.set_axisbelow(True)
for s in ("top", "right"): b.spines[s].set_visible(False)

for ax in (a, b):
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=2)

out = os.path.join(HERE, "fig_why_col")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=600)
print("wrote", out)
