#!/usr/bin/env python3
# ============================================================================
# make_fig_arch_compare.py -- shoulder-to-shoulder architecture comparison.
#                             (2026-08-02)
#
# The mechanism difference, drawn as two pipelines side by side, because the
# numbers alone do not explain WHY per-head eviction fails on a phone.
#
# Every quantity annotated on the figure is measured, not illustrative:
#   per-head union    SnapKV keeps 6592/9744 Llama-1B cells and 9468/11158 Phi-3
#                     cells at the SAME K=1024 budget muKV uses (phone GPU, 16K).
#   selector count    16 layers x 8 KV heads = 128 (Llama-1B); 32 x 32 = 1024
#                     (Phi-3). Retention rises with selector count, 67.7% -> 84.9%.
#   muKV retention    970 cells, i.e. its budget, on both models.
#   FA-off cost       SnapKV must read attention weights, which llama.cpp exposes
#                     only with flash-attention OFF -- and quantized V requires FA,
#                     so a per-head evictor also forfeits q8_0 KV (1.88x bytes/cell).
# ============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.2,
    "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# Okabe-Ito, colour-blind safe. Colour encodes ROLE, not rank.
BLUE, VERM, GRAY, LGRAY, GREEN = "#0072B2", "#D55E00", "#5A5A5A", "#DDDDDD", "#009E73"

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.72))


def stage(ax, x, y, w, h, title, sub, fc, ec, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + h * 0.24, sub, ha="center", va="center",
            fontsize=6.6, color=tc, linespacing=1.25)


def arrow(ax, x, y0, y1, color="#333333"):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=color, shrinkA=0, shrinkB=0))


def cachebar(ax, x, y, w, h, frac_live, label, color):
    """Draw the physical cache: `frac_live` of the span actually occupied."""
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="#999999", lw=0.7))
    if label.startswith("SPARSE"):
        # survivors scattered across the whole span -- attention still scans to
        # the highest occupied index, so the cost stays O(N)
        n = 26
        for i in range(n):
            if i % 3 == 0:
                ax.add_patch(Rectangle((x + w * i / n, y), w / n * 0.85, h, fc=color, ec="none"))
    else:
        ax.add_patch(Rectangle((x, y), w * frac_live, h, fc=color, ec="none"))
    ax.text(x + w / 2, y + h + 0.022, label, ha="center", va="bottom",
            fontsize=6.4, color="#333333")


for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ---------------- LEFT: per-head evictors ----------------
ax = axes[0]
ax.text(0.5, 0.975, "Per-head evictors  (SnapKV, Ada-KV, H2O, TOVA)",
        ha="center", va="top", fontsize=8.4, fontweight="bold", color=VERM)
stage(ax, 0.08, 0.775, 0.84, 0.115, "Prefill with FLASH-ATTENTION OFF",
      "attention weights must be materialised to be read", "#FBE3D6", VERM)
arrow(ax, 0.5, 0.775, 0.715)
stage(ax, 0.08, 0.595, 0.84, 0.115, "Top-$K$ per (layer, head)",
      "each of 128–1024 selectors picks its own $K$ tokens", "white", "#999999")
arrow(ax, 0.5, 0.595, 0.535)
stage(ax, 0.08, 0.415, 0.84, 0.115, "UNION across heads and layers",
      "a cell survives if ANY selector keeps it", "#FBE3D6", VERM)
arrow(ax, 0.5, 0.415, 0.355)
stage(ax, 0.08, 0.235, 0.84, 0.115, r"$\mathtt{seq\_rm}$ marks cells free",
      "survivors are never moved", "white", "#999999")
cachebar(ax, 0.13, 0.088, 0.74, 0.05, 1.0, "SPARSE: decode scans to the last live cell", VERM)
ax.text(0.5, 0.045, "retains 67.7–84.9 % of cells at the same $K$",
        ha="center", va="top", fontsize=7.0, color=VERM, fontweight="bold")

# ---------------- RIGHT: muKV ----------------
ax = axes[1]
ax.text(0.5, 0.975, r"$\mu$KV  (ours)", ha="center", va="top",
        fontsize=8.4, fontweight="bold", color=BLUE)
stage(ax, 0.08, 0.775, 0.84, 0.115, "Prefill with FLASH-ATTENTION ON",
      r"in-graph $\mathtt{kq\_evict}$ side node scores inside FA", "#D9ECF7", BLUE)
arrow(ax, 0.5, 0.775, 0.715)
stage(ax, 0.08, 0.595, 0.84, 0.115, "One keep-set per layer",
      "scored on aggregated attention mass, not per head", "white", "#999999")
arrow(ax, 0.5, 0.595, 0.535)
stage(ax, 0.08, 0.415, 0.84, 0.115, "NO union to take",
      "the selected set is already sequence-level", "#D9ECF7", BLUE)
arrow(ax, 0.5, 0.415, 0.355)
stage(ax, 0.08, 0.235, 0.84, 0.115, "COMPACTION: state round-trip",
      "survivors moved together; 32 ms – 6.1 s once", "#D9ECF7", BLUE)
cachebar(ax, 0.13, 0.088, 0.74, 0.05, 0.093, "DENSE: decode scans $K$ cells, not $N$", BLUE)
ax.text(0.5, 0.045, "retains 8.7–10.0 % $=$ its budget; keeps q8_0 KV",
        ha="center", va="top", fontsize=7.0, color=BLUE, fontweight="bold")

fig.subplots_adjust(wspace=0.06)
for ext in ("pdf", "png"):
    fig.savefig("/home/mislam22/EndurKV_workspace/EndurKV/figures/fig_arch_compare." + ext, dpi=400)
print("wrote fig_arch_compare.pdf / .png")
