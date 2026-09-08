#!/usr/bin/env python3
"""Draw the 4-axis design-space slide for the dissertation deck.

Shows TOVA's eviction problem decomposed into four independent axes:
  1. SIGNAL    — feature per position
  2. SELECTION — top-K rule
  3. BUDGET    — K_h per head  (← v1 lives here)
  4. INTER-HEAD — coupling between heads

Color code:
  gray   = TOVA's default (baseline)
  green  = explored / done by our work (only the BUDGET axis so far)
  orange = open / queued for exploration in this study
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures/architecture_fig")
OUT.mkdir(parents=True, exist_ok=True)

# Layout: 4 columns, slide aspect ratio
fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis("off")

# Colors
GRAY = "#9aa0a6"
GREEN = "#34a853"
ORANGE = "#fb8c00"
NAVY = "#1a3661"
LIGHT_GRAY = "#f1f3f4"
LIGHT_GREEN = "#e6f4ea"
LIGHT_ORANGE = "#fef3e6"

# Title
ax.text(7.5, 8.1, "TOVA Eviction Decomposed — Four Independent Axes",
        ha="center", va="center", fontsize=18, fontweight="bold", color=NAVY)
ax.text(7.5, 7.65,
        "v1 / EndurKV-Evict occupies the BUDGET axis only.  Three other axes are open novelty space.",
        ha="center", va="center", fontsize=11, color="#4a4a4a", style="italic")

# Four columns
COL_X = [0.5, 4.0, 7.5, 11.0]   # left edge
COL_W = 3.3
HDR_Y = 6.95
BOX1_Y = 5.4   # TOVA default
BOX1_H = 1.0
BOX2_Y = 1.5   # Open / done
BOX2_H = 3.6


def draw_header(x, title, subtitle, color=NAVY):
    ax.add_patch(FancyBboxPatch((x, HDR_Y), COL_W, 0.55,
                                boxstyle="round,pad=0.03",
                                facecolor=color, edgecolor=color, linewidth=0))
    ax.text(x + COL_W/2, HDR_Y + 0.33, title,
            ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    ax.text(x + COL_W/2, HDR_Y - 0.18, subtitle,
            ha="center", va="center", fontsize=9, color="#4a4a4a", style="italic")


def draw_default_box(x, items):
    # TOVA's default — gray
    ax.add_patch(FancyBboxPatch((x, BOX1_Y), COL_W, BOX1_H,
                                boxstyle="round,pad=0.04",
                                facecolor=LIGHT_GRAY, edgecolor=GRAY, linewidth=1.3))
    ax.text(x + COL_W/2, BOX1_Y + BOX1_H - 0.2, "TOVA default",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=GRAY)
    y = BOX1_Y + BOX1_H - 0.5
    for it in items:
        ax.text(x + 0.15, y, "•", ha="left", va="center", fontsize=11, color=GRAY)
        ax.text(x + 0.35, y, it, ha="left", va="center", fontsize=9.5, color="#3a3a3a")
        y -= 0.28


def draw_status_box(x, status, items, header_text):
    """status='done' (green) or 'open' (orange)"""
    if status == "done":
        face, edge, head_color = LIGHT_GREEN, GREEN, GREEN
    else:
        face, edge, head_color = LIGHT_ORANGE, ORANGE, ORANGE
    ax.add_patch(FancyBboxPatch((x, BOX2_Y), COL_W, BOX2_H,
                                boxstyle="round,pad=0.04",
                                facecolor=face, edgecolor=edge, linewidth=1.5))
    ax.text(x + COL_W/2, BOX2_Y + BOX2_H - 0.25, header_text,
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=head_color)
    y = BOX2_Y + BOX2_H - 0.65
    for it in items:
        ax.text(x + 0.15, y, "•", ha="left", va="top", fontsize=11, color=head_color)
        ax.text(x + 0.35, y, it, ha="left", va="top", fontsize=9.5, color="#1c1c1c",
                wrap=True)
        # account for multi-line items
        lines = it.count("\n") + 1
        y -= 0.32 * lines + 0.05


# ---- Column 1: SIGNAL ----
x = COL_X[0]
draw_header(x, "1. SIGNAL", "feature per position")
draw_default_box(x, ["a[h, i] = current attn", "(plain TOVA scoring)"])
draw_status_box(x, "open", [
    "Smoothed attention\n  (heat-equation diffusion)",
    "Spectral FFT energy\n  (low-freq persistence)",
    "K-graph centrality\n  (PageRank on K-sim)",
    "Cross-step variance⁻¹",
    "Cross-head consensus\n  vote count",
], "open ← we go next")

# ---- Column 2: SELECTION ----
x = COL_X[1]
draw_header(x, "2. SELECTION", "top-K rule")
draw_default_box(x, ["argmax-K of score", "(hard top-K)"])
draw_status_box(x, "open", [
    "Threshold-based\n  (variable count / head)",
    "Wave-propagated argmax\n  (diffuse score → select)",
    "Gumbel-soft sampling\n  (stochastic top-K)",
    "Graph-connected subgraph\n  (k-core selection)",
    "Compressed-sensing\n  reconstruction",
], "open ← we go next")

# ---- Column 3: BUDGET (our work lives here) ----
x = COL_X[2]
draw_header(x, "3. BUDGET", "K_h per head", color=GREEN)
draw_default_box(x, ["K_h = K  (fixed for all)", "(no per-head adaptation)"])
draw_status_box(x, "done", [
    "✓ v1 (linear gate, α=1.3, β=0.6)\n   final headline policy",
    "✓ swept 69 gate shapes ×\n   25 (α, β) × 7 thresholds",
    "✓ signal/selection axes:\n   negative result — TOVA's\n   max_a + argmax is optimal",
    "Result on Llama (720 cells):\n   median −11% Δ KL vs TOVA\n   93% per-cell wins\n   cache× = 1.02 (neutral)",
], "✓ explored (this paper)")

# ---- Column 4: INTER-HEAD ----
x = COL_X[3]
draw_header(x, "4. INTER-HEAD", "coupling between heads")
draw_default_box(x, ["each head decides alone", "(no information exchange)"])
draw_status_box(x, "open", [
    "Cross-head consensus\n  (k-core graph mechanism)",
    "Mean attention across heads",
    "Mean-field coupling",
    "Eigenvector centrality\n  across head-similarity",
    "Shared budget pool\n  within each layer",
], "open ← we go next")

# Bottom bar — pivot A teaser
ax.add_patch(FancyBboxPatch((0.5, 0.35), 14.0, 0.85,
                            boxstyle="round,pad=0.06",
                            facecolor="#fff7e0", edgecolor="#f5b912", linewidth=1.5))
ax.text(7.5, 0.95, "Composition strategy:  best signal  ×  best selection  ×  best budget  ×  best inter-head  =  full novelty stack",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="#5a3a00")
ax.text(7.5, 0.55, "This composed policy is the base for Pivot A — thermal-coupled adaptive cache "
                   "(α, β modulated by junction temp + battery SoC, on real phone silicon).",
        ha="center", va="center", fontsize=9.5, style="italic", color="#5a3a00")

plt.tight_layout()
plt.savefig(OUT / "design_space_4axis.png", dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.savefig(OUT / "design_space_4axis.pdf", bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"saved {OUT}/design_space_4axis.png and .pdf")
