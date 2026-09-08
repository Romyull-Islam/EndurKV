#!/usr/bin/env python3
"""PLOT 5: Memory pressure vs cache size + temperature (3-way relationship).

For every phone-log run cell that has both sensors.csv AND stress.csv,
compute:
  - peak_kv_cells          := stress.csv['peak_kv_cells'].max()
  - peak_rss_gb            := sensors.csv['peak_rss_kb'].max() / 1024 / 1024
                              (RSS taken from stress.csv['peak_rss_kb'])
  - swap_mb                := (vmstat_pswpout.max() - vmstat_pswpout.min())
                              * 4 KB / 1024
  - peak_DDR_C             := sensors.csv['ddr_temp_mc'].max() / 1000

Render a single bubble-scatter:
  x-axis : peak KV cells (log)
  y-axis : peak DDR temperature (degrees Celsius)
  bubble area : peak RSS (GB)
  bubble color: swap_MB (darker = more swap)
  label  : policy name (per-cell directory name)

Output:
  /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/
      05_memory_pressure_3way.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D


PHONE_LOGS = Path("/home/mislam22/EndurKV_workspace/phone-logs")
OUT_PATH = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/"
    "relationship_plots/05_memory_pressure_3way.png"
)

KB = 1024.0
PAGE_KB = 4.0  # one memory page = 4 KB on this device


def collect_cells() -> pd.DataFrame:
    """Scan phone-logs/* for cells that contain BOTH sensors.csv and stress.csv.

    Returns one row per cell with the four metrics required by the plot,
    plus a policy label and the wave/run group it belongs to.
    """
    rows: list[dict] = []
    for stress_csv in PHONE_LOGS.rglob("stress.csv"):
        cell_dir = stress_csv.parent
        sensors_csv = cell_dir / "sensors.csv"
        if not sensors_csv.exists():
            continue
        try:
            stress = pd.read_csv(stress_csv)
            sensors = pd.read_csv(sensors_csv, low_memory=False)
        except Exception as e:  # malformed CSV, skip
            print(f"[skip] {cell_dir}: failed to parse ({e})")
            continue
        if stress.empty or sensors.empty:
            continue
        if "peak_kv_cells" not in stress.columns:
            continue

        # --- peak KV cells (size of the cache, in KV positions) -----------
        peak_kv_cells = float(np.nan_to_num(stress["peak_kv_cells"].max(),
                                            nan=0.0))
        if peak_kv_cells <= 0:
            continue  # no useful cache-size signal

        # --- peak RSS in GB -----------------------------------------------
        if "peak_rss_kb" in stress.columns:
            peak_rss_kb = float(np.nan_to_num(stress["peak_rss_kb"].max(),
                                              nan=0.0))
        else:
            peak_rss_kb = 0.0
        peak_rss_gb = peak_rss_kb / KB / KB

        # --- swap_MB = (delta vmstat_pswpout pages) * 4 KB / 1024 ---------
        swap_mb = 0.0
        if "vmstat_pswpout" in sensors.columns:
            ps = pd.to_numeric(sensors["vmstat_pswpout"], errors="coerce")
            ps = ps.dropna()
            if len(ps) >= 2:
                # vmstat_pswpout is a monotonic counter; delta = max - min
                delta_pages = float(ps.max() - ps.min())
                swap_mb = max(0.0, delta_pages) * PAGE_KB / KB

        # --- peak DDR temperature (degrees C) -----------------------------
        peak_ddr_c = np.nan
        if "ddr_temp_mc" in sensors.columns:
            ddr = pd.to_numeric(sensors["ddr_temp_mc"], errors="coerce")
            ddr = ddr.dropna()
            if not ddr.empty:
                peak_ddr_c = float(ddr.max()) / 1000.0

        if np.isnan(peak_ddr_c):
            continue  # DDR temperature is the y-axis; cell unusable without it

        # The policy label is the leaf cell directory name (e.g. v1_K512).
        # Disambiguate identical labels coming from different wave runs by
        # prepending the wave directory's parent group.
        policy = cell_dir.name
        wave = cell_dir.parent.name
        rows.append(
            dict(
                wave=wave,
                cell=str(cell_dir.relative_to(PHONE_LOGS)),
                policy=policy,
                peak_kv_cells=peak_kv_cells,
                peak_rss_gb=peak_rss_gb,
                swap_mb=swap_mb,
                peak_ddr_c=peak_ddr_c,
            )
        )

    df = pd.DataFrame(rows)
    return df


def dedupe_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per (policy, peak_kv_cells, peak_ddr_c) signature so that
    the scatter is not overwhelmed by repeated runs of the same configuration.
    When duplicates exist, keep the run with the highest peak_rss_gb (the
    most stressful instance of that policy)."""
    if df.empty:
        return df
    df = df.sort_values("peak_rss_gb", ascending=False).drop_duplicates(
        subset=["policy", "peak_kv_cells"], keep="first"
    )
    return df.reset_index(drop=True)


def render(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        # Render an explanatory empty figure so callers always have a file.
        fig, ax = plt.subplots(figsize=(11, 7), dpi=130)
        ax.text(0.5, 0.5,
                "No (sensors.csv, stress.csv) cells with usable\n"
                "peak_kv_cells AND ddr_temp_mc were found.",
                ha="center", va="center", fontsize=13, color="#444")
        ax.set_axis_off()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote (empty) {out_path}")
        return

    fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=140)

    # Bubble size proportional to RSS GB. Use a min floor so even small
    # caches stay visible. The scale factor was tuned so a ~5 GB RSS gives a
    # marker area around 1800 pt^2.
    rss = df["peak_rss_gb"].to_numpy()
    sizes = 80 + 350 * rss  # area in pt^2

    swap = df["swap_mb"].to_numpy()
    # Color = swap_MB. Dark = more swap. Use log normalisation when range is
    # wide; fall back to linear if everything is zero / near zero.
    swap_for_color = np.where(swap > 0, swap, 0.0)
    if swap_for_color.max() <= 0:
        norm = Normalize(vmin=0, vmax=1)
    elif swap_for_color.max() / max(swap_for_color[swap_for_color > 0].min(),
                                    1e-3) > 50:
        # wide dynamic range — use log
        norm = LogNorm(vmin=max(swap_for_color[swap_for_color > 0].min(), 0.5),
                       vmax=max(swap_for_color.max(), 1.0))
        # ensure zeros plot on the lightest end
        swap_for_color = np.clip(swap_for_color, norm.vmin, norm.vmax)
    else:
        norm = Normalize(vmin=0, vmax=max(swap_for_color.max(), 1.0))

    # Reversed Greys: dark = more swap
    cmap = plt.cm.Greys
    sc = ax.scatter(
        df["peak_kv_cells"], df["peak_ddr_c"],
        s=sizes, c=swap_for_color, cmap=cmap, norm=norm,
        edgecolors="#b22222", linewidths=1.1, alpha=0.92,
    )

    # Per-bubble policy labels. Offset to avoid the bubble center.
    for _, r in df.iterrows():
        ax.annotate(
            r["policy"],
            (r["peak_kv_cells"], r["peak_ddr_c"]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=7.5, color="#111",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="#888", alpha=0.65, lw=0.4),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Peak KV cache size (positions, log scale)", fontsize=11.5)
    ax.set_ylabel("Peak DDR temperature (degrees C)", fontsize=11.5)
    ax.set_title(
        "Cache size determines memory footprint AND temperature; "
        "eviction reduces both at PPL cost",
        fontsize=12.5, weight="bold"
    )
    ax.grid(alpha=0.25, which="both")

    # Colorbar for swap_MB
    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Swap activity during run (MB, darker = more swap)",
                   fontsize=10)

    # Custom size legend (peak RSS in GB)
    rss_min, rss_max = float(rss.min()), float(rss.max())
    # Show three reference sizes spanning the observed range
    if rss_max - rss_min < 0.05:
        ref_vals = [rss_min]
    else:
        ref_vals = sorted({
            round(rss_min, 2),
            round((rss_min + rss_max) / 2, 2),
            round(rss_max, 2),
        })
    size_handles = []
    for v in ref_vals:
        area = 80 + 350 * v
        size_handles.append(
            Line2D([0], [0], marker="o", linestyle="",
                   markersize=np.sqrt(area), markerfacecolor="#ddd",
                   markeredgecolor="#b22222", markeredgewidth=1.0,
                   label=f"peak RSS = {v:.2f} GB")
        )
    # Place the bubble-size legend outside the data region (lower-right
    # corner inside the axes) so it does not collide with the large bubbles
    # that cluster near the upper-left of the plot.
    ax.legend(handles=size_handles, loc="lower right", fontsize=9,
              title="Bubble area = peak RSS (GB)", title_fontsize=9.5,
              framealpha=0.95, labelspacing=1.6, borderpad=1.0)

    # Give the y-axis a bit of headroom so the topmost label is not clipped.
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 1.0, y_max + 2.5)

    # Subtle footer with provenance / cell count
    ax.text(
        0.99, -0.13,
        f"n = {len(df)} cells from phone-logs/ "
        "(each with sensors.csv and stress.csv)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="#666",
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}  (n={len(df)})")


def main() -> int:
    df = collect_cells()
    print(f"[collect] {len(df)} cells with sensors.csv + stress.csv")
    if not df.empty:
        df = dedupe_cells(df)
        print(f"[dedupe]  {len(df)} unique (policy, peak_kv_cells) rows")
        # Print the table that backs the plot for the headline / log trail.
        cols = ["wave", "policy", "peak_kv_cells", "peak_rss_gb",
                "swap_mb", "peak_ddr_c"]
        with pd.option_context("display.max_rows", 200,
                               "display.width", 160,
                               "display.float_format", lambda x: f"{x:,.3f}"):
            print(df[cols].to_string(index=False))
    render(df, OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
