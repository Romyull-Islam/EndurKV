#!/usr/bin/env python3
"""muKV: what saving energy does to performance, where performance is time AND accuracy.

Panel A, time. muKV at K = 1024 on the Adreno, the same request (9737-token prompt, 1024
output tokens, Llama-3.2-1B) at three pinned GPU clocks, 24 cooled repeats. x is throughput
(3600 / whole-request time), y is energy for the whole request. Walking down the curve is Q1
(save energy, lose speed), walking up is Q2 (gain speed, spend energy). The clock does not
change the answer: the same text came out at 902, 1200 and with a decode cap.

Panel B, accuracy. muKV at K = 1024 against its own full cache, four models. x is energy per
request on the phone CPU, y is LongBench accuracy (five tasks, token-F1), both as a percent
of the full cache. The move left is Q1 (muKV saves energy and keeps most of the accuracy);
the move back right is Q2 (what the last few points of accuracy cost in energy).

Sources: energy_perf_data.json (energy_perf_data.py). Accuracy is the paper's four-model
LongBench table (RTX 4500, same model and budget); energy and time are phone runs.
Palette: Okabe-Ito, validated for colour-vision deficiency.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "energy_perf_data.json")))

INK, BODY, MUTED, GRID, AXIS = "#1a1a1a", "#34414a", "#5f6b73", "#e6e9eb", "#b8c0c6"
Q1C, Q2C, CAPC = "#0072B2", "#D55E00", "#009E73"

# the finer ladder of 2026-09-22: five clocks measured in one session. A clock curve must come from
# one session, since absolute joules drift a few percent between them.
SRC = D["ladder_fine"] if "ladder_fine" in D else D["ladder"]
P = {p["mhz"]: p for p in SRC["points"]}
MIDRUNG = int(SRC.get("rule_with_fine_rungs", {}).get("mid", "gpu967").replace("gpu", ""))
thr = {m: 3600 / p["time_s"] for m, p in P.items()}
E = {m: p["energy_J"] for m, p in P.items()}
sd = {m: p["energy_sd"] for m, p in P.items()}
ddr = {m: p["ddr_peak"] for m, p in P.items()}
cap = D["cost_table"]["steps"][0]                     # decode clock capped at 902 MHz, from 1200
cap_thr = thr[1200] / (1 + cap["time_cost"] / 100)
cap_E = E[1200] * (1 - cap["energy_saved"] / 100)


def step(a, b):
    """Down the curve from a to b: time cost %, energy saved %, rate."""
    t = 100 * (P[b]["time_s"] / P[a]["time_s"] - 1)
    e = 100 * (1 - E[b] / E[a])
    return t, e, e / t


LOW = min(P)                                        # 826 MHz: the rung below the elbow
t12, e12, r12 = step(1200, 902)
t97, e97, r97 = step(902, LOW)
g_hi = 100 * (thr[1200] / thr[902] - 1)             # throughput gained going up
g_lo = 100 * (thr[902] / thr[LOW] - 1)
up_hi = 100 * (E[1200] / E[902] - 1)                # energy it costs
up_lo = 100 * (E[902] / E[LOW] - 1)                 # negative: below the elbow the trade reverses
e_mid = 100 * (1 - E[MIDRUNG] / E[1200]); t_mid = 100 * (P[MIDRUNG]["time_s"] / P[1200]["time_s"] - 1)
TIERS = D["cost_table"]["tiers"]

ACC = D["accuracy"]
MU = "#009E73"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 11.5,
                     "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})
fig = plt.figure(figsize=(12.1, 5.0))
axA = fig.add_axes([0.06, 0.13, 0.405, 0.74])
axB = fig.add_axes([0.575, 0.13, 0.41, 0.74])
fig.text(0.06, 0.965, "A.  Time: the GPU clock", fontsize=13.5, color=INK, weight="bold", va="top")
fig.text(0.06, 0.915, "\u03bcKV at K = 1024, same request, Adreno 840", fontsize=11, color=MUTED, va="top")
fig.text(0.575, 0.965, "B.  Accuracy: the cache budget", fontsize=13.5, color=INK, weight="bold", va="top")
fig.text(0.575, 0.915, "\u03bcKV at K = 1024 against its own full cache, four models", fontsize=11, color=MUTED, va="top")
import numpy as np

# ------------------------------------------------------------------ panel A: time
clocks = sorted(P)
xs = [thr[m] for m in clocks]; ys = [E[m] for m in clocks]
axA.plot(xs, ys, color=INK, lw=2.4, zorder=3, solid_capstyle="round")
axA.errorbar(xs, ys, yerr=[sd[m] for m in clocks], fmt="none", ecolor=MUTED, elinewidth=1, capsize=3, zorder=4)
axA.scatter(xs, ys, s=80, color=INK, edgecolor="white", linewidth=1.5, zorder=5)
axA.scatter([thr[MIDRUNG]], [E[MIDRUNG]], s=230, facecolor="none", edgecolor=CAPC, linewidth=2.2, zorder=6)
for m in clocks:                                     # the ringed mid rung is named in its callout
    if m == MIDRUNG:
        continue
    axA.annotate(f"{m}", (thr[m], E[m]), xytext=(0, -18), textcoords="offset points",
                 ha="center", va="top", fontsize=10.5, color=MUTED)
def walk(ax, x0, y0, x1, y1, off, col):
    p0 = ax.transData.transform((x0, y0)); p1 = ax.transData.transform((x1, y1))
    d = p1 - p0; n = np.array([-d[1], d[0]]) / np.hypot(*d)
    q0 = ax.transData.inverted().transform(p0 + off * n + 0.14 * d)
    q1 = ax.transData.inverted().transform(p0 + off * n + 0.86 * d)
    ax.add_patch(FancyArrowPatch(q0, q1, arrowstyle="-|>", mutation_scale=17, color=col, lw=2.2, zorder=7))
axA.set_xlim(16.2, 28.7); axA.set_ylim(540, 985)
walk(axA, thr[902], E[902], thr[1200], E[1200], +20, Q2C)
walk(axA, thr[1200], E[1200], thr[902], E[902], +20, Q1C)
axA.text(16.5, 978, f"Q2  gain speed:\n+{g_hi:.0f}% throughput\ncosts +{up_hi:.0f}% energy", ha="left", va="top",
         fontsize=11.5, color=INK, weight="bold", linespacing=1.25)
axA.text(23.9, 745, f"Q1  save energy:\n\u2212{e12:.0f}% energy\ncosts +{t12:.0f}% time", ha="left", va="top",
         fontsize=11.5, color=INK, weight="bold", linespacing=1.25)
axA.annotate(f"{LOW} MHz: {g_lo:.0f}% slower\nthan 902 and no cheaper", (thr[LOW], E[LOW] - 6), xytext=(16.5, 640),
             textcoords="data", ha="left", va="center", fontsize=10.5, color=INK, linespacing=1.2,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkA=4, shrinkB=2))
axA.annotate(f"where the mid tier stops, {MIDRUNG} MHz:\n\u2212{e_mid:.0f}% energy for +{t_mid:.0f}% time",
             (thr[MIDRUNG], E[MIDRUNG] + 6), xytext=(16.45, 880), textcoords="data",
             ha="left", va="top", fontsize=10.5, color=CAPC, weight="bold", linespacing=1.2,
             arrowprops=dict(arrowstyle="-", color=CAPC, lw=0.9, shrinkA=6, shrinkB=8))
axA.axvline(thr[902], color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
axA.text(thr[902] + 0.18, 592, "elbow", color=MUTED, fontsize=10.5, va="bottom", ha="left")
axA.text(28.5, 548, "accuracy unchanged: the same answer\nat 902, 1200 and with a decode cap",
         ha="right", va="bottom", fontsize=10, color=MU, weight="bold", linespacing=1.2)
axA.set_xlabel("throughput (requests per hour)  \u2192 faster")
axA.set_ylabel("energy per request (J)")

# ------------------------------------------------------------------ panel B: accuracy
pm = ACC["per_model"]
axB.axhline(100, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
axB.axvline(100, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
axB.scatter([100], [100], s=150, facecolor="white", edgecolor=INK, linewidth=2, zorder=5)
axB.annotate("full cache\n(each model)", (100, 100), xytext=(0, 13), textcoords="offset points",
             ha="center", va="bottom", fontsize=10.5, color=INK, linespacing=1.15)
lab_off = {"Llama-3.2-1B": (-8, -16, "right"), "Phi-3-mini": (10, 6, "left"),
           "gemma-2-2b": (-10, 0, "right"), "Bonsai-8B": (10, -4, "left")}
for a in pm:
    x, y = a["energy_pct"], a["acc_kept"]
    axB.scatter([x], [y], s=110, color=MU, edgecolor="white", linewidth=1.5, zorder=6)
    dx, dy, ha = lab_off[a["model"]]
    txt = f"{a['model']}\n{y:.1f}% acc, {100 / a['time_pct']:.1f}\u00d7 faster"
    if a["model"] == "Llama-3.2-1B":   # crowded next to Phi-3: label in the free space above, with a leader
        axB.annotate(txt, (x, y), xytext=(21.5, 106.4), textcoords="data", ha="left", va="center", fontsize=10,
                     color=INK, linespacing=1.15, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkA=2, shrinkB=5))
        continue
    axB.annotate(txt, (x, y), xytext=(dx, dy),
                 textcoords="offset points", ha=ha, va="center", fontsize=10, color=INK, linespacing=1.15)
e_lo, e_hi = min(a["energy_pct"] for a in pm), max(a["energy_pct"] for a in pm)
a_lo = min(a["acc_kept"] for a in pm); a_hi = max(a["acc_kept"] for a in pm)
b_lo = min(a["buy_back_energy_pct"] for a in pm); b_hi = max(a["buy_back_energy_pct"] for a in pm)
axB.add_patch(FancyArrowPatch((90.5, 101.6), (73, 101.6), arrowstyle="-|>", mutation_scale=17, color=Q1C, lw=2.2, zorder=4))
axB.add_patch(FancyArrowPatch((73, 98.4), (90.5, 98.4), arrowstyle="-|>", mutation_scale=17, color=Q2C, lw=2.2, zorder=4))
axB.text(74, 102.3, f"Q1  save energy with \u03bcKV:\n{100 - e_hi}\u2013{100 - e_lo}% less energy\n"
         f"{a_lo:.1f}\u2013{min(a_hi, 100):.0f}% of accuracy kept\nand faster, not slower",
         ha="center", va="bottom", fontsize=11, color=INK, weight="bold", linespacing=1.2)
axB.text(83, 97.8, f"Q2  win back the last\n0\u2013{100 - a_lo:.1f}% of accuracy:\n+{b_lo} to +{b_hi}% energy",
         ha="center", va="top", fontsize=11, color=INK, weight="bold", linespacing=1.2)
b = ACC["below_1024"]
axB.text(21, 89.15, f"Below K = 1024 the saving stops:\n{-b['K512']:.1f}% more at K = 512, {-b['K256']:.1f}% at K = 256",
         ha="left", va="bottom", fontsize=10.5, color=MUTED, linespacing=1.2)
axB.set_xlim(20, 112); axB.set_ylim(88.5, 108.5)
axB.set_xlabel("energy per request (% of full cache)  \u2190 cheaper")
axB.set_ylabel("accuracy (% of full cache)")

for ax in (axA, axB):
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=3)

out = os.path.join(HERE, "fig_energy_perf_curve")
fig.savefig(out + ".png", dpi=300)
fig.savefig(out + ".pdf")
print(f"up the curve: 902->1200 +{g_hi:.1f}% throughput, +{up_hi:.1f}% energy;  {LOW}->902 +{g_lo:.1f}%, {up_lo:+.1f}%")
print(f"down the curve: 1200->902 saves {e12:.1f}% for +{t12:.1f}% time (rate {r12:.2f}); {LOW} saves {e97:.1f}% for +{t97:.1f}% (rate {r97:.2f})")
print(f"decode cap: -{cap['energy_saved']}% energy, -{100 * (1 - cap_thr / thr[1200]):.1f}% throughput (rate {cap['rate']})")
print("wrote", out)
