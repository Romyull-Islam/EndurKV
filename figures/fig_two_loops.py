#!/usr/bin/env python3
"""Two coupled loops, performance (P) and energy (E), each with its own control, and the shared
region where reducing one sacrifices the other. Panel (a) formalizes the sketch. Panels (b) and
(c) put every measured knob step on the (time change, energy change) plane: the slope of a step
is the exchange rate the loops trade at, and the lever thresholds are lines through the origin.

Sources: scripts/android/ukv_sched_table.txt (per-token costs, 9737 prompt + 1024 output),
the clock campaign (CPU 883 to 1632 MHz), the six-generation soak (full cache against K=1024),
and the data lever (4096 against 1024 output tokens at 1200 MHz).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

BLUE, AQUA, RED, PURPLE, GREEN, ORANGE = "#2a78d6", "#1b9aa8", "#e34948", "#5b2a86", "#3aa41c", "#e08a1e"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ---- measured steps: whole request, 9737 prompt + 1024 output unless stated ----------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "energy_rl"))
import sched_policy as sp
rows = {r["plan"]: r for r in sp.load_table()}
def ET(plan, n_out=1024, n_p=9737):
    r = rows[plan]
    return (n_p * r["e_pre"] + n_out * r["e_dec"]) / 1000, (n_p * r["t_pre"] + n_out * r["t_dec"]) / 1000
def step(a, b, **kw):
    (Ea, Ta), (Eb, Tb) = a, b
    return dict(dT=100 * (Tb / Ta - 1), dE=100 * (Eb / Ea - 1), **kw)

big = [
    step(ET("gpu1200_k1024"), ET("gpu902_k1024"), label="GPU clock 1200 to 902\nwhole request", who="shared", ha="right", dx=-5, dy=6),
    step(ET("gpu902_k1024"), ET("gpu726_k1024"), label="GPU clock 902 to 726:\n3% energy for 21% time", who="poor", ha="left", dx=5, dy=8),
    step(ET("gpu1200_k1024", 4096), ET("gpu1200_k1024", 1024), label="output cap 4096 to 1024\ncosts answer length, not speed", who="aligned", ha="left", dx=5, dy=-1),
    dict(dT=100 * (7.36 / 13.44 - 1), dE=100 * (412 / 444 - 1), label="CPU clock 883 to 1632\nenergy flat within noise", who="P", ha="center", dx=0, dy=12),
    dict(dT=100 * (1 / 4.6 - 1), dE=-62.0, label="CPU cache full to 1024\nquality 39.3 to 29.8 F1 hotpotqa", who="aligned", ha="left", dx=5, dy=6),
]
small = [
    step(ET("gpu1200_k1024"), ET("gpu1200d902_k1024"), label="decode 1200 to 902", who="E", ha="right", dx=-5, dy=0),
    step(ET("gpu1200d902_k1024"), ET("gpu1200d726_k1024"), label="decode 902 to 726", who="E", ha="right", dx=-5, dy=0),
    step(ET("cpu_k1024"), ET("cpu_k512"), label="CPU cache 1024 to 512\nquality 1.00 to 0.63", who="Q", ha="center", dx=0, dy=-11),
    step(ET("cpu_k512"), ET("cpu_k256"), label="CPU cache 512 to 256\nquality 0.63 to 0.58", who="Q", ha="center", dx=0, dy=11),
]
col = {"shared": PURPLE, "poor": ORANGE, "E": GREEN, "P": RED, "aligned": BLUE, "Q": AQUA}
levers = ((1.5, "lever 1 (healthy): 1.5", (0, (4, 2))), (0.8, "lever 0.5 (mid): 0.8", (0, (2, 2))), (0.5, "lever 0 (low): 0.5", (0, (1, 1.5))))

fig = plt.figure(figsize=(7.2, 3.35))
gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.0, 0.80], left=0.012, right=0.99, top=0.9, bottom=0.215, wspace=0.34)

# ---- (a) the two loops --------------------------------------------------------------------
a = fig.add_subplot(gs[0]); a.set_axis_off(); a.set_xlim(0, 10); a.set_ylim(0, 10)
a.set_title("(a) two loops, one shared region", loc="left", fontsize=7.6)
a.add_patch(Ellipse((4.0, 6.55), 7.0, 3.5, fill=False, lw=1.4, ec=RED, zorder=2))
a.add_patch(Ellipse((5.0, 4.15), 7.0, 3.5, fill=False, lw=1.4, ec=GREEN, zorder=2))
a.add_patch(Ellipse((4.5, 5.35), 3.5, 1.5, fill=True, fc="#efe9f7", ec=PURPLE, lw=1.1, zorder=3))
a.text(0.15, 9.75, "cutting f2 inside the shared region costs P one for one, and\nraising f1 there costs E. Outside it, each loop moves freely.",
       fontsize=5.9, color=INK, va="top")
a.text(0.15, 8.75, "P  performance loop", color=RED, fontsize=7.2, weight="bold", va="center")
a.text(0.15, 8.4, "keeps time inside its budget", color=RED, fontsize=5.9, va="center")
a.text(0.15, 1.15, "E  energy loop", color=GREEN, fontsize=7.2, weight="bold", va="center")
a.text(0.15, 0.8, "takes every saving the lever allows", color=GREEN, fontsize=5.9, va="center")
# f1 on the P loop, f2 on the E loop
a.text(7.75, 7.05, "f1", color=RED, fontsize=8, weight="bold", ha="center", va="center")
a.annotate("", (7.75, 8.45), (7.75, 7.45), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0))
a.text(8.05, 7.95, "raise", color=RED, fontsize=5.9, va="center")
a.text(7.6, 6.65, "P-only knobs\nCPU clock\ncache vs full", color=RED, fontsize=5.5, ha="left", va="top", linespacing=1.15)
a.text(6.9, 2.35, "f2", color=GREEN, fontsize=8, weight="bold", ha="center", va="center")
a.annotate("", (6.9, 1.0), (6.9, 1.95), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.0))
a.text(6.65, 1.5, "cut", color=GREEN, fontsize=5.9, va="center", ha="right")
a.text(7.55, 2.3, "E-only knobs\ndecode clock\noutput cap when\nno length is asked", color=GREEN, fontsize=5.5, ha="left", va="top", linespacing=1.15)
# the shared region and the lever
a.text(4.5, 5.62, "shared", color=PURPLE, fontsize=6.5, weight="bold", ha="center", va="center")
a.text(4.5, 5.12, "GPU prefill clock\nwanted answer length", color=PURPLE, fontsize=5.2, ha="center", va="center", linespacing=1.1)
a.plot([4.5, 4.5], [4.6, 4.0], color=PURPLE, lw=0.8, ls=(0, (2, 2)), zorder=2)
a.text(4.5, 3.92, "lever L, set by battery tier:\nthe exchange rate a shared\nknob may be cut at", color=PURPLE, fontsize=5.3, ha="center", va="top", linespacing=1.1)

# ---- (b) the large steps ------------------------------------------------------------------
b = fig.add_subplot(gs[1])
b.set_title("(b) the large steps", loc="left", fontsize=7.6)
c = fig.add_subplot(gs[2])
c.set_title("(c) the small steps", loc="left", fontsize=7.6)
for ax in (b, c):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1); ax.axvline(0, color=AXIS, lw=0.6, zorder=1)
    ax.set_xlabel("time change per request (%)")
b.set_ylabel("energy change per request (%)")
xs = np.linspace(0, 34, 40)
for lam, lab, ls in levers:
    b.plot(xs, -lam * xs, color=PURPLE, lw=0.7, ls=ls, zorder=2, alpha=0.8)
    b.text(35, -lam * 34, lab.split(":")[0], color=PURPLE, fontsize=5.0, va="center", ha="left")
for s in big:
    b.scatter(s["dT"], s["dE"], s=28, color=col[s["who"]], edgecolor=SURF, linewidth=0.6, zorder=4)
    b.annotate(s["label"], (s["dT"], s["dE"]), xytext=(s["dx"], s["dy"]), textcoords="offset points", fontsize=5.3, color=col[s["who"]], ha=s["ha"], va="center", linespacing=1.1)
for s in small:
    b.scatter(s["dT"], s["dE"], s=9, color=MUTED, zorder=3)
b.add_patch(plt.Rectangle((-3.6, -6.6), 7.9, 9.8, fill=False, ec=MUTED, lw=0.6, zorder=3)); b.text(5.0, -6.6, "(c)", color=MUTED, fontsize=5.3, va="bottom", ha="left")
b.set_xlim(-88, 70); b.set_ylim(-72, 15)
b.text(-86, 14.5, "left of zero: faster and cheaper", color=BLUE, fontsize=5.4, va="top")
b.text(69, -71, "right of zero: energy bought with time", color=PURPLE, fontsize=5.4, va="bottom", ha="right")

# ---- (c) the small steps -------------------------------------------------------------------
xi = np.linspace(0, 3.4, 20)
for lam, lab, ls in levers:
    c.plot(xi, -lam * xi, color=PURPLE, lw=0.7, ls=ls, alpha=0.8, zorder=2)
    c.text(3.5, -lam * 3.4, f"{lam}", color=PURPLE, fontsize=5.2, va="center", ha="left")
for s in small:
    c.scatter(s["dT"], s["dE"], s=26, color=col[s["who"]], edgecolor=SURF, linewidth=0.6, zorder=4)
    c.annotate(s["label"], (s["dT"], s["dE"]), xytext=(s["dx"], s["dy"]), textcoords="offset points", fontsize=5.2, color=col[s["who"]], ha=s["ha"], va="center", linespacing=1.1)
c.set_xlim(-3.6, 4.3); c.set_ylim(-6.6, 3.2)
c.text(-3.5, -6.5, "a shared step is taken only if its point\nlies below the lever line of the tier", color=PURPLE, fontsize=5.2, va="bottom", linespacing=1.1)

fig.text(0.012, 0.018, "OnePlus 15, Llama-3.2-1B. GPU and split-clock steps from the cost table (9737 prompt + 1024 output tokens, Adreno 840); CPU clock from the 883 to 1632 MHz campaign;\n"
         "full cache from the six-generation CPU soak; output cap from the 4096 against 1024 token cells at 1200 MHz. Lever lines: energy saved per unit of time given up, by tier.",
         fontsize=5.7, color=MUTED, va="bottom")
out = os.path.join(os.path.dirname(__file__), "fig_two_loops")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
print("wrote", out)
for s in big + small:
    lam = (-s["dE"] / s["dT"]) if s["dT"] > 0 else float("nan")
    print(f"  {s['label'].splitlines()[0]:36s} dT {s['dT']:+6.1f}%  dE {s['dE']:+6.1f}%  exchange rate {lam:5.2f}  [{s['who']}]")
