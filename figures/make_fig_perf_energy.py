#!/usr/bin/env python3
# ============================================================================
# make_fig_perf_energy.py -- phone GPU: throughput, energy and memory together.
#                            (2026-08-02)
#
# WHY ONE FIGURE INSTEAD OF THREE. Prior KV-eviction papers report throughput and
# memory; none of SnapKV / Ada-KV / H2O / TOVA / PyramidKV reports ENERGY. On a
# phone that is the omission that matters -- energy sets temperature and
# temperature sets the throttle that caps sustained throughput. All three axes
# therefore belong in one view.
#
# FORM. Everything is normalised so that >1 means BETTER THAN THE FULL CACHE on
# that axis, and all three bars share one meaning ("times better"), so a single
# log axis carries them without a second scale. A Pareto scatter was tried first
# and rejected: with 4-6 points per model the labels collide and the reader still
# has to do the division in their head.
#
# THE POINT THE FIGURE MAKES. muKV clears 1.0 on all three axes simultaneously;
# SnapKV falls below 1.0 on two of three -- it pays the eviction cost and collects
# neither the speed nor the energy, while still holding most of the cache.
#
# Measured, Adreno 840, 12K-token WikiText prompt + 4096 generated, ctx 16384,
# batch 1, cool gate (DDR<=35C, batt<=33C) before every cell, USB-rail energy.
# n=1 per cell; repeats are in flight (device run-to-run spread measured at 9.9%).
# ============================================================================
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# One hue per AXIS (the quantity being measured), fixed order, never cycled.
AX = [("throughput", "#0072B2"), ("energy efficiency", "#009E73"), ("cache reduction", "#CC79A7")]
ORDER = ["muKV", "muKV no-compact", "SnapKV"]
SHOW = {"muKV": r"$\mu$KV", "muKV no-compact": r"$\mu$KV" "\n" "no compaction", "SnapKV": "SnapKV"}

rows = json.load(open("/tmp/phone_gpu_summary.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True)

for ax, mt, disp in zip(axes, ["llama1b", "phi3"], ["Llama-3.2-1B", "Phi-3-mini"]):
    rs = {r["pol"]: r for r in rows if r["model"] == mt}
    v = rs["vanilla"]
    vE = v["mwh"] / (v["gen"] / 1000.0)
    vE_abs = v["mwh"] * 3600.0 / v["gen"]          # mJ per generated token
    pols = [p for p in ORDER if p in rs]
    w, xs = 0.26, np.arange(len(pols))
    for k, (name, col) in enumerate(AX):
        vals = []
        for p in pols:
            r = rs[p]
            if name == "throughput":
                vals.append(r["tps"] / v["tps"])
            elif name == "energy efficiency":
                vals.append(vE / (r["mwh"] / (r["gen"] / 1000.0)))
            else:
                vals.append(v["cells"] / r["cells"])
        b = ax.bar(xs + (k - 1) * w, vals, w * 0.92, color=col,
                   label=name if mt == "llama1b" else None, zorder=3,
                   edgecolor="white", linewidth=0.6)
        for rect, val in zip(b, vals):
            ax.annotate(("%.2f" % val) if val < 10 else ("%.1f" % val),
                        (rect.get_x() + rect.get_width() / 2,
                         val * (1.06 if val >= 1 else 1.06)),
                        ha="center", va="bottom", fontsize=6.3,
                        color=col if val >= 1 else "#B00000",
                        fontweight="bold" if val < 1 else "normal", zorder=4)
    # the break-even line: below it, the policy is worse than not evicting at all
    ax.axhline(1.0, color="#333333", lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([SHOW[p] for p in pols], fontsize=7.4)
    # ABSOLUTE energy per generated token under each group: a ratio alone hides the
    # scale, and mJ/token is the number a phone actually budgets against.
    for xi, p in zip(xs, pols):
        mj = rs[p]["mwh"] * 3600.0 / rs[p]["gen"]
        ax.annotate("%.0f mJ/tok" % mj, (xi, -0.235), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=6.3, color=AX[1][1])
    ax.annotate("full cache: %.0f mJ/tok" % vE_abs, (0.5, -0.40), xycoords="axes fraction",
                ha="center", va="top", fontsize=6.6, color="#444444")
    ax.set_ylim(0.075, 26)
    ax.set_title(disp, fontsize=8.4)
    ax.grid(True, axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

axes[0].set_ylabel(r"$\times$ better than the full cache")
# the dashed line + the y-axis label already say "full cache = 1.0"; a text tag
# there only collided with the first bar's value label
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7.4, loc="lower center",
           bbox_to_anchor=(0.5, 1.005), ncol=3, handlelength=1.1, columnspacing=1.6)
fig.text(0.5, -0.34,
         "All three axes normalised so $>\\!1$ is better than the full cache. "
         "$\\mu$KV clears 1.0 on every axis; SnapKV falls below it on two of three\n"
         "— it pays the eviction cost and collects neither the speed nor the energy. "
         "No prior KV-eviction paper reports the energy axis at all.",
         ha="center", fontsize=6.9, color="#333333", linespacing=1.4)
fig.subplots_adjust(wspace=0.08)
for ext in ("pdf", "png"):
    fig.savefig("/home/mislam22/EndurKV_workspace/EndurKV/figures/fig_perf_energy." + ext, dpi=400)
print("wrote fig_perf_energy.pdf / .png")
