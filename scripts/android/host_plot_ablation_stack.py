#!/usr/bin/env python3
"""
PLOT 6: Counter-factual ablation - each mechanism removed from v1_FA2-stack.

Compares 5 cells (Phi-3 long-decode protocol) showing how each layer of
EndurKV control affects peak DDR temperature and throttle outcomes:

  - Wave-4 vanilla              (no eviction, no control)
  - Wave-4 v1 K=512             (eviction only, no control)
  - Wave-6 v1_fa bounded        (+ state-swap, no control)
  - Wave-8 v1_fa2 selective     (+ smarter anchoring, no control)
  - Wave-9 v1_fa2-stack         (+ Q8 K + watchdog + adaptive K) -- only one
                                that avoids kernel throttle.

Output: figures/relationship_plots/06_ablation_stack.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_FIG = "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/06_ablation_stack.png"

# Ordered from baseline (top of stack added last) to fully-stacked (bottom).
# We plot top-to-bottom so the most-stacked variant sits at the top of the bar
# chart. Each row records:
#   label, mechanism_delta, peak_ddr_C, throttle_bool
CELLS = [
    {
        "label":  "Wave-4 vanilla",
        "delta":  "(no eviction, no control)",
        "ddr":    62.9,
        "throttle": True,
    },
    {
        "label":  "Wave-4 v1 K=512",
        "delta":  "+ eviction (K=512)",
        "ddr":    54.4,
        "throttle": False,
    },
    {
        "label":  "Wave-6 v1_fa bounded",
        "delta":  "+ file-backed state-swap",
        "ddr":    65.6,
        "throttle": True,
    },
    {
        "label":  "Wave-8 v1_fa2 selective",
        "delta":  "+ selective anchoring",
        "ddr":    72.9,
        "throttle": True,
    },
    {
        "label":  "Wave-9 v1_fa2-stack",
        "delta":  "+ Q8 K + watchdog + adaptive K",
        "ddr":    64.1,
        "throttle": False,
    },
]

THROTTLE_COLOR = "#d62728"   # red
NO_THROTTLE_COLOR = "#2ca02c"  # green


def main():
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

    # We want most-stacked at TOP of plot, so reverse for matplotlib's y order.
    rows = list(reversed(CELLS))
    n = len(rows)

    y_pos = np.arange(n)
    ddrs = [r["ddr"] for r in rows]
    colors = [THROTTLE_COLOR if r["throttle"] else NO_THROTTLE_COLOR for r in rows]

    # Build y-axis labels combining variant name + mechanism delta.
    ytick_labels = [f"{r['label']}\n{r['delta']}" for r in rows]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(y_pos, ddrs, color=colors, edgecolor="black",
                   linewidth=1.0, height=0.65, alpha=0.88)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ytick_labels, fontsize=10)
    ax.set_xlabel("Peak DDR temperature (deg C)", fontsize=11)
    ax.set_xlim(0, max(ddrs) * 1.22)
    ax.axvline(70.0, color="#555555", linestyle=":", linewidth=1,
               label="70 deg C (typical throttle threshold)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # Annotate each bar with the delta from the PREVIOUS condition (in the
    # original stacking order = CELLS order). Build a lookup from row label.
    prev_ddr_by_label = {}
    for i, c in enumerate(CELLS):
        if i == 0:
            prev_ddr_by_label[c["label"]] = None
        else:
            prev_ddr_by_label[c["label"]] = CELLS[i - 1]["ddr"]

    for bar, r in zip(bars, rows):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2.0
        prev = prev_ddr_by_label[r["label"]]
        throttle_tag = "throttle" if r["throttle"] else "no throttle"
        if prev is None:
            annot = f"{x:.1f} deg C   (baseline, {throttle_tag})"
        else:
            d = x - prev
            sign = "+" if d >= 0 else ""
            annot = f"{x:.1f} deg C   ({sign}{d:.1f} vs prev, {throttle_tag})"
        ax.text(x + 0.6, y, annot, va="center", ha="left",
                fontsize=9.5, fontweight="bold",
                color="#222222")

    # Legend: throttle yes/no + threshold line
    legend_handles = [
        mpatches.Patch(facecolor=THROTTLE_COLOR, edgecolor="black",
                       label="Kernel throttle event: YES"),
        mpatches.Patch(facecolor=NO_THROTTLE_COLOR, edgecolor="black",
                       label="Kernel throttle event: NO"),
    ]
    leg1 = ax.legend(handles=legend_handles, loc="lower right",
                     fontsize=10, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_title(
        "Mechanism-by-mechanism ablation: each layer of EndurKV control "
        "affects peak DDR\n"
        "Phi-3 long-decode protocol; bars colored by kernel throttle outcome",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    print(f"[ok] wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
