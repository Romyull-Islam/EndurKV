#!/usr/bin/env python3
"""Column-width thermal figure: (a) battery temperature with the vendor's 50 C throttle trigger,
(b) prime-core clock. Bonsai-8B on the phone CPU from a cold start, full cache vs muKV, same build
and prompt, no watchdog. Data in /tmp/nat_bonsai (restore from tmp_archive after a reboot).
fig_bonsai_thermal.py is the four-panel version with skin and DDR."""
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [("full cache", "/tmp/nat_bonsai/vanilla", "#888888"),
        (r"$\mu$KV", "/tmp/nat_bonsai/mukv_faon", "#0072B2")]
# Only eviction arms. The watchdog arm (/tmp/wd_vanilla) is not plotted.
INK, MUTED, GRID, AXIS, HOT = "#1a1a1a", "#6b6b6b", "#ececec", "#b8b8b8", "#c0392b"


def load(path):
    sr = list(csv.DictReader(open(f"{path}/sensors.csv", "rb").read().decode("utf-8", "replace").splitlines()))

    def col(n):
        out = []
        for r in sr:
            try:
                out.append(float(r.get(n, "")))
            except ValueError:
                out.append(np.nan)
        return np.array(out)

    t = (col("monotonic_s") - col("monotonic_s")[0]) / 60.0
    g = json.loads(re.sub(r"\binf|\bnan|-nan", "null", open(f"{path}/gen.json").read()))
    return dict(t=t, prime=col("cpu6_freq_hz") / 1e3, bat=col("battery_temp_mc") / 1000.0,
                wall=(g["prefill_ms"] + g["decode_ms"]) / 60000.0)


def smooth(y, w):
    y = np.asarray(y, float)
    if np.isnan(y).any():
        y = np.where(np.isnan(y), np.nanmean(y), y)
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w) / w, mode="valid")[:len(y)]


data = {lbl: load(p) for lbl, p, _ in RUNS}
xmax = max(d["wall"] for d in data.values()) * 1.03

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.0, "axes.labelsize": 7.2,
                     "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "pdf.fonttype": 42,
                     "legend.frameon": False, "axes.linewidth": 0.6})
fig, (a, b) = plt.subplots(2, 1, figsize=(3.33, 1.30), sharex=True)
fig.subplots_adjust(left=0.155, right=0.985, top=0.965, bottom=0.215, hspace=0.20)

for lbl, path, color in RUNS:
    d = data[lbl]
    lw = 1.2 if "mu" in lbl else 1.0
    m = d["t"] <= d["wall"]                       # the sampler outlives the run; plot only the run
    a.plot(d["t"][m], smooth(d["bat"], 7)[m], color=color, lw=lw, label=lbl)
    # The governor oscillates every sample, so plot a 27-sample (13 s) mean to show the sustained level.
    b.plot(d["t"][m], smooth(d["prime"], 27)[m], color=color, lw=lw)

a.axhline(50, color=HOT, ls=(0, (2, 2)), lw=0.7)
a.text(xmax * 0.015, 50.4, "50 °C vendor throttle trigger", fontsize=6, color=HOT, va="bottom")
a.set_ylabel("battery (°C)", fontsize=6.0, labelpad=1)
a.set_ylim(31, 54)


b.axhline(883, color=HOT, ls=(0, (2, 2)), lw=0.7)
b.text(2.5, 905, "883 MHz throttle floor", fontsize=6, color=HOT, va="bottom", ha="left")   # left: the traces sit above 1200 there
b.set_ylabel("prime clock (MHz)")
b.set_ylim(820, 1760)
b.set_xlabel("elapsed time (min)")
b.legend(handles=[plt.Line2D([], [], color=c, lw=1.2, label=l) for l, _, c in RUNS],
         loc="upper right", ncol=2, fontsize=6.0, handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)

# mark where each arm finishes
for lbl, _, color in RUNS:
    d = data[lbl]
    a.plot([d["wall"]], [np.interp(d["wall"], d["t"], smooth(d["bat"], 7))], marker="o", ms=3.2,
           color=color, mec="white", mew=0.6, zorder=5)
a.annotate("μKV ends here,\n6 °C below the trigger", (data[r"$\mu$KV"]["wall"], 44.2),
           xytext=(6, -2), textcoords="offset points", ha="left", va="top", fontsize=6.2,
           color="#0072B2", linespacing=1.1)

for ax in (a, b):
    ax.set_xlim(0, xmax)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=2)

out = os.path.join(HERE, "fig_thermal_col")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("arms:", {k: round(v["wall"], 1) for k, v in data.items()}, "-> wrote", out)
