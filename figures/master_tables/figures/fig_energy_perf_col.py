#!/usr/bin/env python3
"""Column-width version of the energy / performance figure for the HotMobile paper.

Job: answer "how energy and performance affect each other" in one column, where performance is
time AND accuracy. Two stacked panels, energy on both y axes.
  (a) time: the same request at pinned GPU clocks. Walking down the curve saves energy and costs
      time, but only to a point: 826 MHz is slower than 902 and no cheaper, so the useful range ends
      at the 902 MHz elbow. The clock does not change the answer.
  (b) accuracy: muKV at K = 1024 against its own full cache on four models. The cache saves 40 to
      66% of the energy, runs faster, and keeps 92.5 to 100% of LongBench accuracy.

Data: energy_perf_data.json (energy_perf_data.py). Panel (a) is the finer ladder of 2026-09-22:
five clocks, n=3 each, all in one session, because absolute joules drift a few percent between
sessions and a spliced curve would show a step no run measured. Paper style: DejaVu Sans 7 pt.
"""
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "energy_perf_data.json")))
INK, MUTED, GRID, AXIS = "#1a1a1a", "#6b6b6b", "#ececec", "#b8b8b8"
Q1C, Q2C, MU = "#0072B2", "#D55E00", "#009E73"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.0, "axes.labelsize": 7.2,
                     "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "pdf.fonttype": 42})

# ---- clock points: the finer ladder of 2026-09-22, five clocks measured in ONE session (n=3 each).
# A clock curve has to come from one session: the earlier 24-pull ladder sat about 5% lower in
# absolute joules, so splicing the two would invent a step that no run measured.
src = D["ladder_fine"] if "ladder_fine" in D else D["ladder"]
pts = {p["mhz"]: dict(T=p["time_s"], E=p["energy_J"], n=p["n"], sd=p["energy_sd"]) for p in src["points"]}
clocks = sorted(pts)
thr = {m: 3600 / pts[m]["T"] for m in clocks}
MID = src.get("rule_with_fine_rungs", {}).get("mid", "")
mid_mhz = int(MID.replace("gpu", "")) if MID.startswith("gpu") else None

fig, (a, b) = plt.subplots(2, 1, figsize=(3.33, 2.20))
fig.subplots_adjust(left=0.135, right=0.97, top=0.93, bottom=0.15, hspace=0.85)

# (a) time
xs = [thr[m] for m in clocks]; ys = [pts[m]["E"] for m in clocks]
a.plot(xs, ys, color=INK, lw=1.4, zorder=3)
# spread across the three runs of each clock: 2 J at 1200, 19 to 33 J below it
a.errorbar(xs, ys, yerr=[pts[m]["sd"] for m in clocks], fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=2, zorder=3)
a.scatter(xs, ys, s=18, color=INK, edgecolor="white", linewidth=0.6, zorder=4)
if mid_mhz in pts:                                   # the rung the mid tier's walk stops at
    a.scatter([thr[mid_mhz]], [pts[mid_mhz]["E"]], s=64, facecolor="none", edgecolor=MU, linewidth=1.3, zorder=5)
for m in clocks:                                      # every clock label above its point, clear of the axis
    a.annotate(f"{m}", (thr[m], pts[m]["E"]), xytext=(0, 11), textcoords="offset points", ha="center", va="bottom",
               fontsize=5.8, color=MUTED)
lo, k, hi = clocks[0], 902, clocks[-1]
e_hi = 100 * (pts[hi]["E"] / pts[k]["E"] - 1); g_hi = 100 * (thr[hi] / thr[k] - 1)
e_lo = 100 * (pts[lo]["E"] / pts[k]["E"] - 1); g_lo = 100 * (thr[k] / thr[lo] - 1)
a.axvline(thr[k], color=AXIS, lw=0.7, ls=(0, (3, 2)), zorder=1)
a.text(thr[k] - 0.12, max(ys) + 18, "elbow", color=MUTED, fontsize=6, va="top", ha="right")
# "the clock does not change the answer" lives in the caption: the panel has no room for it
a.set_xlim(min(xs) - 1.0, max(xs) + 1.2); a.set_ylim(min(ys) - 45, max(ys) + 75)
a.set_xlabel("throughput (requests per hour)  \u2192 faster")
a.set_ylabel("energy (J)")
a.set_title("(a) time: the GPU clock, \u03bcKV at K = 1024", loc="left", fontsize=7.2, color=INK, pad=4)

# (b) accuracy
pm = D["accuracy"]["per_model"]
short = {"Llama-3.2-1B": "Llama-1B", "Phi-3-mini": "Phi-3", "gemma-2-2b": "Gemma-2B", "Bonsai-8B": "Bonsai-8B"}
off = {"Llama-3.2-1B": (0, 8, "center"), "Phi-3-mini": (5, -1, "left"), "gemma-2-2b": (5, 0, "left"), "Bonsai-8B": (5, 0, "left")}
b.axhline(100, color=AXIS, lw=0.7, ls=(0, (3, 2)), zorder=1)
b.scatter([100], [100], s=34, facecolor="white", edgecolor=INK, linewidth=1.1, zorder=4)
b.annotate("full cache", (100, 100), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=6.2)
for p in pm:
    x, y = p["energy_pct"], p["acc_kept"]
    b.scatter([x], [y], s=26, color=MU, edgecolor="white", linewidth=0.6, zorder=5)
    dx, dy, ha = off[p["model"]]
    b.annotate(f"{short[p['model']]}, {100 / p['time_pct']:.1f}× faster", (x, y), xytext=(dx, dy),
               textcoords="offset points", ha=ha, va="center", fontsize=6, color=INK)
lo_e = 100 - max(p["energy_pct"] for p in pm); hi_e = 100 - min(p["energy_pct"] for p in pm)
lo_a = min(p["acc_kept"] for p in pm)
b.set_xlim(25, 108); b.set_ylim(90.5, 103)
b.set_xlabel("energy per request (% of full cache)  ← cheaper")
b.set_ylabel("LongBench (%)")
b.set_title("(b) accuracy: the cache, μKV K = 1024 vs full", loc="left", fontsize=7.2, color=INK, pad=4)

for ax in (a, b):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=2)

out = os.path.join(HERE, "fig_energy_perf_col")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("clocks", clocks, "n", [pts[m]["n"] for m in clocks], "mid rung", mid_mhz, "-> wrote", out)
