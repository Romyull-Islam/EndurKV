#!/usr/bin/env python3
"""
Wave-11 Phi-3 Pareto frontier figure.

3-axis Pareto bubble plot:
  x: total_latency_s (lower better)
  y: mean_PPL (lower better)
  bubble color: peak_DDR_°C (lower better, viridis)
  bubble size: swap_MB + 1 (lower better)

Uses 4 complete policies (vanilla, h2o, v1_fa2_stack, tova) plus
streamingllm annotated as "in progress".
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


OUT = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/eval_plots/wave11_phi3_pareto.png"
)


# Real measured Phi-3 K=512 values (per task spec)
POLICIES = [
    # name,        lat_s,  PPL,   DDR_C, swap_MB, complete
    ("vanilla",      1036.0, 5.71,  64.8,  158.0, True),
    ("h2o",          1524.0, 5.77,  62.9,    0.0, True),
    ("v1_fa2_stack",  832.0, 6.39,  66.0,   58.0, True),
    ("tova",         1180.0, 5.92,  57.0,    0.0, True),  # tova lat approx
    # streamingllm — partial run, annotate only (no bubble plotted in metric space)
    ("streamingllm",  None,  None,  None,   None, False),
]


def pareto_set(points):
    """Return indices of non-dominated points (all axes: lower is better)."""
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j <= i on all axes and < on at least one
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def main() -> None:
    complete = [p for p in POLICIES if p[5]]
    names = [p[0] for p in complete]
    lat = np.array([p[1] for p in complete], dtype=float)
    ppl = np.array([p[2] for p in complete], dtype=float)
    ddr = np.array([p[3] for p in complete], dtype=float)
    swp = np.array([p[4] for p in complete], dtype=float)

    # 4-D Pareto over (lat, ppl, ddr, swap)
    pts4 = np.stack([lat, ppl, ddr, swp], axis=1)
    pareto_idx = pareto_set(pts4)
    pareto_names = [names[i] for i in pareto_idx]

    # ---- figure --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=130)

    # Bubble size: encode swap_MB+1 with sqrt scaling so swap=0 is still visible.
    base = 80.0
    bubble_size = base + 16.0 * np.sqrt(swp + 1.0) * 6.0

    norm = Normalize(vmin=float(np.min(ddr)) - 1.0, vmax=float(np.max(ddr)) + 1.0)
    cmap = cm.get_cmap("viridis")

    sc = ax.scatter(
        lat,
        ppl,
        s=bubble_size,
        c=ddr,
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidths=1.2,
        alpha=0.92,
        zorder=3,
    )

    # Annotate each complete policy
    label_offsets = {
        "vanilla":      ( 18,  10),
        "h2o":          ( 18, -18),
        "v1_fa2_stack": (-12,  18),
        "tova":         ( 18,  12),
    }
    for i, name in enumerate(names):
        dx, dy = label_offsets.get(name, (10, 10))
        tag = name
        if i in pareto_idx:
            tag = name + "  *"
        ax.annotate(
            tag,
            xy=(lat[i], ppl[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold" if i in pareto_idx else "normal",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="0.4",
                lw=0.6,
                alpha=0.85,
            ),
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6),
            zorder=4,
        )
        # Per-axis "wins": annotate which metric this policy is best at
        wins = []
        if lat[i] == lat.min():
            wins.append("min lat")
        if ppl[i] == ppl.min():
            wins.append("min PPL")
        if ddr[i] == ddr.min():
            wins.append("min DDR")
        if swp[i] == swp.min():
            wins.append("min swap")
        if wins:
            ax.annotate(
                " | ".join(wins),
                xy=(lat[i], ppl[i]),
                xytext=(dx, dy - 18),
                textcoords="offset points",
                fontsize=8,
                color="#1a5e1a",
                fontstyle="italic",
                zorder=4,
            )

    # ---- Pareto frontier curve (in lat / PPL plane) --------------------
    # Sort Pareto-front policies by latency and draw a step line for visual cue.
    pareto_pts = sorted(
        [(lat[i], ppl[i], names[i]) for i in pareto_idx], key=lambda t: t[0]
    )
    if len(pareto_pts) >= 2:
        xs = [p[0] for p in pareto_pts]
        ys = [p[1] for p in pareto_pts]
        ax.plot(
            xs,
            ys,
            linestyle="--",
            color="crimson",
            linewidth=1.6,
            alpha=0.7,
            zorder=2,
            label=f"Pareto frontier ({len(pareto_idx)}/{len(names)})",
        )

    # ---- streamingllm "in progress" annotation -------------------------
    # Place in the upper-right margin of the data area as a callout.
    x_lo, x_hi = lat.min(), lat.max()
    y_lo, y_hi = ppl.min(), ppl.max()
    x_pad = 0.08 * (x_hi - x_lo + 1.0)
    y_pad = 0.10 * (y_hi - y_lo + 1.0)
    ax.set_xlim(x_lo - x_pad - 60, x_hi + x_pad + 120)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad + 0.10)

    ax.text(
        0.98,
        0.02,
        "streamingllm: in progress\n(partial run — not plotted)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#7a3b00",
        bbox=dict(
            boxstyle="round,pad=0.4",
            fc="#fff5e6",
            ec="#cc8400",
            lw=1.0,
        ),
        zorder=5,
    )

    # ---- axis cosmetics -----------------------------------------------
    ax.set_xlabel("total_latency_s  (lower is better)", fontsize=11)
    ax.set_ylabel("mean_PPL  (lower is better)", fontsize=11)
    ax.set_title(
        "Phi-3 K=512 Pareto: no single policy dominates",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.grid(True, linestyle=":", alpha=0.5, zorder=1)

    # Colorbar for DDR
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("peak_DDR_°C  (lower better)", fontsize=10)

    # Bubble-size legend (swap_MB)
    ref_swaps = [0, 60, 160]
    legend_handles = []
    for s in ref_swaps:
        sz = base + 16.0 * np.sqrt(s + 1.0) * 6.0
        h = ax.scatter(
            [], [],
            s=sz,
            facecolor="lightgray",
            edgecolor="black",
            linewidths=1.0,
            label=f"swap = {s} MB",
        )
        legend_handles.append(h)
    # Frontier line legend
    frontier_line, = ax.plot(
        [], [], linestyle="--", color="crimson", linewidth=1.6,
        label=f"Pareto frontier ({len(pareto_idx)}/{len(names)})",
    )
    legend_handles.append(frontier_line)
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
        title="bubble size  =  swap_MB+1",
        title_fontsize=9,
    )
    leg.set_zorder(6)

    # Subtitle line under axes describing the Pareto winners
    win_lat = names[int(np.argmin(lat))]
    win_ppl = names[int(np.argmin(ppl))]
    win_ddr = names[int(np.argmin(ddr))]
    subtitle = (
        f"latency winner: {win_lat}   |   "
        f"PPL winner: {win_ppl}   |   "
        f"DDR winner: {win_ddr}   |   "
        f"Pareto set: {{{', '.join(pareto_names)}}}"
    )
    fig.text(
        0.5,
        0.005,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.2",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"[OK] wrote {OUT}")
    print(f"[OK] Pareto set: {pareto_names}")


if __name__ == "__main__":
    main()
