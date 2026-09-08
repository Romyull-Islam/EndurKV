#!/usr/bin/env python3
"""perhead_v1 architecture — minimalist vertical flow.

Design rules:
  * Single column, top-to-bottom flow.
  * Each stage is one row: numbered circle + box on the LEFT, small icon on the RIGHT.
  * No formulas, no equations, no overlapping text anywhere.
  * Generous whitespace between rows.
  * One color per stage (orange / blue / gold / green / red).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle


def draw_grid(ax, x0, y0, w, h, n_rows, n_cols, color_fn, pad=0.10):
    """Draw n_rows × n_cols cells inside (x0,y0,w,h)."""
    cell_w = w / n_cols
    cell_h = h / n_rows
    inset = min(cell_w, cell_h) * pad
    for r in range(n_rows):
        for c in range(n_cols):
            x = x0 + c * cell_w + inset
            y = y0 + (n_rows - 1 - r) * cell_h + inset
            fc = color_fn(r, c)
            if fc is None:
                continue
            ax.add_patch(Rectangle(
                (x, y), cell_w - 2 * inset, cell_h - 2 * inset,
                facecolor=fc, edgecolor="none",
            ))


def main() -> int:
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures/architecture_fig")
    out_dir.mkdir(exist_ok=True, parents=True)

    fig = plt.figure(figsize=(11, 14), facecolor="white", dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ─── Title ───
    ax.text(0.5, 0.965,
            "perhead_v1 (EndurKV-Evict)",
            fontsize=24, weight="bold", ha="center", color="#0f2e57")
    ax.text(0.5, 0.935,
            "Per-Head Adaptive KV Cache Eviction — Architecture",
            fontsize=14, ha="center", color="#555")

    # ─── 5 stages ───
    stages = [
        ("1", "Full KV cache in RAM",        "#fff0e8", "#d24a1d"),
        ("2", "Compute per-head attention",  "#e7f1ff", "#1f5fa8"),
        ("3", "Spread gate (per head)",      "#fff4d6", "#c18a0c"),
        ("4", "Adaptive top-K selection",    "#e3f6e3", "#2c8b3b"),
        ("5", "Compressed KV cache",         "#fcdbdb", "#a31616"),
    ]
    n_stages = len(stages)
    top_y       = 0.89
    stage_h     = 0.12
    gap         = 0.030
    box_x0      = 0.05
    box_w       = 0.46
    circle_cx   = 0.10
    label_x     = 0.32
    icon_x0     = 0.61
    icon_w      = 0.34

    stage_y_centers = []

    for i, (tag, label, fc, ec) in enumerate(stages):
        y_top = top_y - i * (stage_h + gap)
        y_bot = y_top - stage_h
        y_c   = (y_top + y_bot) / 2
        stage_y_centers.append((y_top, y_bot, y_c))

        # Box (left)
        box = FancyBboxPatch(
            (box_x0, y_bot + 0.004), box_w, stage_h - 0.008,
            boxstyle="round,pad=0.004,rounding_size=0.014",
            linewidth=2.2, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(box)

        # Numbered circle
        cr = 0.040
        circ = Circle((circle_cx, y_c), cr,
                      facecolor="white", edgecolor=ec, linewidth=2.6)
        ax.add_patch(circ)
        ax.text(circle_cx, y_c, tag,
                ha="center", va="center",
                fontsize=22, weight="bold", color=ec)

        # Stage label
        ax.text(label_x, y_c, label,
                ha="center", va="center",
                fontsize=15, weight="bold", color=ec)

        # Arrow to the next stage (vertical, in the gap)
        if i < n_stages - 1:
            y_arrow_top = y_bot - 0.002
            y_arrow_bot = y_bot - gap + 0.002
            arr = FancyArrowPatch(
                (label_x, y_arrow_top), (label_x, y_arrow_bot),
                arrowstyle="-|>", mutation_scale=24,
                linewidth=2.4, color="#444",
            )
            ax.add_patch(arr)

    # ─── Icons (one per stage, right side) ───
    rng = np.random.default_rng(7)
    icon_pad = 0.014

    # Stage 1: full grid (all cells filled — every position stored)
    y_top, y_bot, y_c = stage_y_centers[0]
    draw_grid(ax, icon_x0, y_bot + icon_pad, icon_w, stage_h - 2 * icon_pad,
              n_rows=8, n_cols=20,
              color_fn=lambda r, c: "#4a90c2")

    # Stage 2: heatmap (attention peaks + low background)
    y_top, y_bot, y_c = stage_y_centers[1]
    peaks = {(0, 4), (1, 17), (2, 11), (3, 2), (5, 14), (6, 8)}
    def attn_color(r, c):
        if (r, c) in peaks:           return "#08306b"
        if any((r, c + dc) in peaks for dc in (-1, 1)):  return "#3a7bb0"
        if any((r, c + dc) in peaks for dc in (-2, 2)):  return "#a6c7e3"
        return "#eaf2fa"
    draw_grid(ax, icon_x0, y_bot + icon_pad, icon_w, stage_h - 2 * icon_pad,
              n_rows=8, n_cols=20, color_fn=attn_color)

    # Stage 3: spread-gate curve (line plot drawn directly in the main ax)
    y_top, y_bot, y_c = stage_y_centers[2]
    curve_x0 = icon_x0 + 0.015
    curve_x1 = icon_x0 + icon_w - 0.010
    curve_y0 = y_bot + icon_pad + 0.012
    curve_y1 = y_top - icon_pad - 0.012
    # data domain x ∈ [0,1] (max_a), y = 1.3 − 0.6·clip((x−0.4)/0.4, 0, 1)
    n = 60
    xs = np.linspace(0, 1, n)
    ys = 1.3 - 0.6 * np.clip((xs - 0.4) / 0.4, 0, 1)
    x_fig = curve_x0 + xs * (curve_x1 - curve_x0)
    y_fig = curve_y0 + ((ys - 0.70) / 0.60) * (curve_y1 - curve_y0)
    # Background regions
    x_lo = curve_x0 + 0.0 * (curve_x1 - curve_x0)
    x_mid = curve_x0 + 0.4 * (curve_x1 - curve_x0)
    x_hi_start = curve_x0 + 0.8 * (curve_x1 - curve_x0)
    x_hi = curve_x1
    ax.add_patch(Rectangle((x_lo, curve_y0), x_mid - x_lo, curve_y1 - curve_y0,
                           facecolor="#e0f3e0", edgecolor="none", alpha=0.6))
    ax.add_patch(Rectangle((x_hi_start, curve_y0), x_hi - x_hi_start, curve_y1 - curve_y0,
                           facecolor="#fadcdc", edgecolor="none", alpha=0.6))
    ax.plot(x_fig, y_fig, color="#c18a0c", linewidth=3.0, solid_capstyle="round")
    # Tiny axis labels (only at corners, no numbers)
    ax.text(curve_x0 + 0.02, curve_y1 - 0.008,
            "1.3", fontsize=8, ha="left", va="top", color="#888")
    ax.text(curve_x1 - 0.005, curve_y0 + 0.005,
            "0.7", fontsize=8, ha="right", va="bottom", color="#888")
    ax.text((curve_x0 + curve_x1) / 2, curve_y0 - 0.008,
            "head sharpness  →", fontsize=8.5,
            ha="center", va="top", color="#555", style="italic")
    ax.text(curve_x0 - 0.012, (curve_y0 + curve_y1) / 2,
            "multiplier", fontsize=8.5,
            ha="right", va="center", color="#555", style="italic", rotation=90)

    # Stage 4: per-head bars (varying heights, colored by sharp/medium/diffuse)
    y_top, y_bot, y_c = stage_y_centers[3]
    bar_heights = [0.40, 0.45, 0.55, 0.95, 1.00, 0.95, 0.85, 0.50]
    bar_colors  = ["#a31616", "#a31616", "#7f7f7f", "#2c8b3b",
                   "#2c8b3b", "#2c8b3b", "#2c8b3b", "#7f7f7f"]
    n_bars = len(bar_heights)
    bar_area_x0 = icon_x0 + 0.020
    bar_area_w  = icon_w - 0.040
    bar_w = bar_area_w / n_bars * 0.70
    bar_gap = bar_area_w / n_bars * 0.30
    base_y  = y_bot + icon_pad + 0.014
    max_h   = stage_h - 2 * icon_pad - 0.024
    # Light gray dashed baseline for K_nominal
    ax.plot([bar_area_x0 - 0.005, bar_area_x0 + bar_area_w + 0.005],
            [base_y + 0.70 * max_h] * 2,
            linestyle=":", color="#999", linewidth=1.2)
    for i, (bh, bc) in enumerate(zip(bar_heights, bar_colors)):
        bx = bar_area_x0 + i * (bar_w + bar_gap)
        ax.add_patch(Rectangle((bx, base_y), bar_w, bh * max_h,
                               facecolor=bc, edgecolor="black", linewidth=0.6))
    ax.text((bar_area_x0 + bar_area_w / 2), base_y - 0.012,
            "heads (H0…H7)", fontsize=8.5,
            ha="center", va="top", color="#555", style="italic")

    # Stage 5: evicted grid (~40% kept, 60% evicted with subtle hatch)
    y_top, y_bot, y_c = stage_y_centers[4]
    kept_rng = np.random.default_rng(3)
    keep_rate = [0.25, 0.30, 0.45, 0.55, 0.65, 0.60, 0.50, 0.30]
    def evict_color(r, c):
        return "#d24a1d" if kept_rng.random() < keep_rate[r] else "#eaeaea"
    draw_grid(ax, icon_x0, y_bot + icon_pad, icon_w, stage_h - 2 * icon_pad,
              n_rows=8, n_cols=20, color_fn=evict_color)

    # ─── Bottom: 3-step summary bar ───
    y_summary = 0.045
    box_h_sum = 0.045
    summary_color = "#0f2e57"
    pieces = [
        ("100% cache",        "Input",   "#fff0e8", "#d24a1d"),
        ("head-adaptive eviction", "Process", "#fff4d6", "#c18a0c"),
        ("~35% cache, 96% mass retained", "Output",  "#e3f6e3", "#2c8b3b"),
    ]
    sb_x0 = 0.07
    sb_total_w = 0.86
    sb_gap = 0.030
    sb_w = (sb_total_w - 2 * sb_gap) / 3
    for i, (txt, lab, fc, ec) in enumerate(pieces):
        x = sb_x0 + i * (sb_w + sb_gap)
        rect = FancyBboxPatch((x, y_summary), sb_w, box_h_sum,
                              boxstyle="round,pad=0.004,rounding_size=0.012",
                              linewidth=1.6, edgecolor=ec, facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + sb_w / 2, y_summary + box_h_sum * 0.72,
                lab, fontsize=10, weight="bold", color=ec, ha="center")
        ax.text(x + sb_w / 2, y_summary + box_h_sum * 0.30,
                txt, fontsize=10.5, color="#222", ha="center")
        if i < len(pieces) - 1:
            arr = FancyArrowPatch(
                (x + sb_w + 0.003, y_summary + box_h_sum / 2),
                (x + sb_w + sb_gap - 0.003, y_summary + box_h_sum / 2),
                arrowstyle="-|>", mutation_scale=18,
                linewidth=1.8, color="#444",
            )
            ax.add_patch(arr)

    out_path = out_dir / "perhead_v1_clean.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
