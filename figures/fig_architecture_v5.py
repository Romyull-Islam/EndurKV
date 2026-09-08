#!/usr/bin/env python3
# ============================================================================
# fig_architecture_v5.py -- the muKV pipeline as one left-to-right pass.
# (2026-08-17, replaces fig_architecture_v4; geometry reworked same night after
# a render check caught arrays overflowing their stage boxes)
#
# WHY THE REDRAW. v4 was an inventory: ten boxes, three edge styles, two nested
# containers and a seven-item legend, with the decode path and the thermal loop
# given equal visual weight -- the reader could not answer "what does muKV DO?"
# in five seconds. It also advertised a stale mechanism: its "serialize ->
# restore" arrow is the state ROUND-TRIP, which the paper now reports as
# OS-killed at 16K; the shipped mechanism is the in-place slide.
#
# v5 is the canon form (TinyMem / PagedAttention style): ONE pipeline, five
# stages, each annotated with its MEASURED cost or payoff, and the cell array --
# the data structure the paper is about -- drawn at the two points where it
# changes (select, slide). The thermal/energy machinery is a thin gray strip at
# reduced weight: it is an independent outer loop and the geometry now says so.
# ============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import numpy as np

MUKV, GOOD, INK = "#D55E00", "#009E73", "#222222"

fig, ax = plt.subplots(figsize=(11.4, 3.9))
ax.set_xlim(0, 23.2); ax.set_ylim(-0.1, 7.4); ax.axis("off")

BY, BH = 3.15, 3.0          # stage box y / height

def stage(x, w, title, sub, fc="white", ec=INK):
    ax.add_patch(FancyBboxPatch((x, BY), w, BH, boxstyle="round,pad=0.08",
                                facecolor=fc, edgecolor=ec, lw=1.2))
    ax.text(x + w/2, BY + BH - 0.42, title, ha="center", fontsize=9,
            fontweight="bold", color=INK)
    ax.text(x + w/2, BY + BH - 0.82, sub, ha="center", va="top", fontsize=6.6,
            color="#444444")
    return x + w

def arrow(x0, x1, label=None):
    y = BY + BH/2 + 0.15
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=13, color=INK, lw=1.3))
    if label:
        ax.text((x0+x1)/2, y + 0.28, label, ha="center", fontsize=6.3, color=INK)

def cells(x0, y0, keep, w=0.215, gap=0.045, h=0.34):
    for i, k in enumerate(keep):
        xx = x0 + i * (w + gap)
        ax.add_patch(Rectangle((xx, y0), w, h,
                     facecolor=MUKV if k else "white",
                     edgecolor=MUKV if k else "#BBBBBB", lw=0.5))
    return x0 + len(keep) * (w + gap) - gap

G = 0.85                     # inter-stage gap

# 1 prompt
e = stage(0.3, 2.5, "prompt", "N tokens\n(9.7k–57k here)")
arrow(e + 0.05, e + G - 0.05)

# 2 FA-on prefill + side tap
x2 = e + G
e2 = stage(x2, 4.1, "FlashAttention prefill", "unmodified llama.cpp graph")
chip_x, chip_w = x2 + 0.10, 3.9
ax.add_patch(FancyBboxPatch((chip_x, 2.02), chip_w, 0.92, boxstyle="round,pad=0.06",
                            facecolor="#FBE8DC", edgecolor=MUKV, lw=1.2))
ax.text(chip_x + chip_w/2, 2.48, "kq_evict side node — scores from the\nlast chunk; FA never turns off",
        ha="center", va="center", fontsize=6.3, color=MUKV)
ax.annotate("", xy=(chip_x + chip_w/2, BY - 0.02), xytext=(chip_x + chip_w/2, 2.96),
            arrowprops=dict(arrowstyle="-|>", color=MUKV, lw=1.2))
ax.text(chip_x + chip_w/2, 1.62, "cost: +1.2% prefill (SnapKV pays +55% for this signal)",
        ha="center", fontsize=6.2, color=MUKV)
arrow(e2 + 0.05, e2 + G - 0.05, "scores")

# 3 select — 12-cell array inside the box
x3 = e2 + G
W3 = 3.9
e3 = stage(x3, W3, "select ONE keep-set", "budget gate + α-split;\nshared by all heads & layers")
rng = np.random.default_rng(4)
keep = np.zeros(12, dtype=bool); keep[rng.choice(12, 5, replace=False)] = True; keep[0] = True
cells(x3 + 0.39, 3.42, keep)
arrow(e3 + 0.05, e3 + G - 0.05, "K idx")

# 4 in-place slide — sparse above, dense below, slide arrows between
x4 = e3 + G
W4 = 4.0
e4 = stage(x4, W4, "in-place slide", "survivors → dense prefix")
ax0 = x4 + 0.42
cells(ax0, 4.28, keep)
kept_idx = np.where(keep)[0]
for i, src in enumerate(kept_idx):
    ax.add_patch(FancyArrowPatch((ax0 + src*0.26 + 0.10, 4.24),
                                 (ax0 + i*0.26 + 0.10, 3.84),
                 arrowstyle="->", mutation_scale=5, color="#999999", lw=0.7))
cells(ax0, 3.44, np.ones(len(kept_idx), dtype=bool))
ax.text(x4 + W4/2, 2.55, "peak memory never rises —\nthe 2× round-trip is OS-killed at 16K",
        ha="center", fontsize=6.2, color="#444444")
arrow(e4 + 0.05, e4 + G - 0.05)

# 5 decode
x5 = e4 + G
W5 = 3.7
stage(x5, W5, "FlashAttention decode", "reads K cells only\n(723 of 9741 = 7.4%)",
      fc="#E7F4EF", ec=GOOD)
ax.text(x5 + W5/2, 2.62, "1.20× GPU · 3.18× CPU\n−8% → −32% energy",
        ha="center", fontsize=7.0, color=GOOD, fontweight="bold")

# outer loop strip
ax.add_patch(FancyBboxPatch((0.3, 0.28), 22.4, 0.95, boxstyle="round,pad=0.06",
                            facecolor="#F4F4F4", edgecolor="#AAAAAA", lw=0.9))
ax.text(0.75, 0.75, "outer loop (independent):", fontsize=7, color="#555555",
        fontweight="bold", va="center")
ax.text(4.55, 0.75, "skin·battery·DDR·CPU @2 Hz → reduce-only clock watchdog → DVFS cap"
        "    ·    battery SoC → cache tier (§energy-aware)",
        fontsize=6.6, color="#555555", va="center")
ax.text(22.45, 0.75, "never touches the keep-set", fontsize=6.2, color="#888888",
        va="center", ha="right", style="italic")

ax.set_title(r"$\mu$KV in the decode path: score inside the FA graph, select once at "
             "sequence level, slide in place, decode over K cells",
             fontsize=9.5, loc="left", pad=6)
fig.tight_layout()
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_architecture_v5"
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf")
