#!/usr/bin/env python3
"""PLOT 14: Memory utilization (RSS + swap) vs SoC + memory (DDR) temperature.

For every cell directory under phone-logs/* that contains BOTH a stress.csv
(per-iter probe summary) and a sensors.csv (high-frequency thermal trace),
compute the per-cell aggregate:

  - RSS_gb       := stress.csv['peak_rss_kb'].max() / (1024 * 1024)
  - swap_MB      := (vmstat_pswpout.max() - vmstat_pswpout.min()) * 4 KB / 1024
                    (vmstat_pswpout is a monotonic page-counter; one page = 4 KB)
  - SoC_temp_C   := max over all CPU thermal zones (cpu*_temp_mc) / 1000
                    (same definition used by host_master_table_4policy.py
                    for the "CPU peak" column in the master tables)
  - DDR_temp_C   := sensors.csv['ddr_temp_mc'].max() / 1000

Renders a 2x2 panel figure:
   (top-left)  RSS_gb  vs SoC_temp_C   -- colored by policy family
   (top-right) RSS_gb  vs DDR_temp_C   -- colored by policy family
   (bot-left)  swap_MB vs SoC_temp_C   -- colored by policy family
   (bot-right) swap_MB vs DDR_temp_C   -- colored by policy family

Each panel overlays a linear regression line and annotates the R^2.
Cells with extreme swap (Wave-7's 1552 MB run, Wave-3's 515 MB run) are
explicitly labelled as outliers in the swap panels.

Output:
   /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/
       14_memutil_vs_temp.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHONE_LOGS = Path("/home/mislam22/EndurKV_workspace/phone-logs")
OUT_DIR    = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures/"
                  "relationship_plots")
OUT_FIG    = OUT_DIR / "14_memutil_vs_temp.png"
OUT_SCHEMA = OUT_DIR / "14_memutil_vs_temp.schema.json"

KB = 1024.0
PAGE_KB = 4.0   # one VM page == 4 KB on the test device

# Policy family -> color (kept in sync with host_plot_cache_temp_scatter.py)
POLICY_COLOR = {
    "vanilla":          "#d62728",   # red
    "llamacpp_stock":   "#7f7f7f",   # grey
    "tova":             "#9467bd",   # purple
    "pyramid":          "#bcbd22",   # olive
    "h2o":              "#e377c2",   # pink
    "v1":               "#1f77b4",   # blue
    "v1_fa":            "#17becf",   # cyan
    "v1_fa2":           "#ff7f0e",   # orange
    "v1_fa2_selective": "#8c564b",   # brown
    "v1_fa2_stack":     "#2ca02c",   # green
    "other":            "#444444",
}

POLICY_ORDER = [
    "vanilla", "llamacpp_stock", "tova", "pyramid", "h2o",
    "v1", "v1_fa", "v1_fa2", "v1_fa2_selective", "v1_fa2_stack", "other",
]


def policy_family(cell_name: str) -> str:
    """Map a leaf cell directory name to a policy family key."""
    n = cell_name.lower()
    if "v1_fa2_stack" in n:        return "v1_fa2_stack"
    if "v1_fa2_selective" in n:    return "v1_fa2_selective"
    if "v1_fa2" in n:              return "v1_fa2"
    if "v1_fa" in n:               return "v1_fa"
    if "tova" in n:                return "tova"
    if "pyramid" in n:             return "pyramid"
    if "h2o" in n:                 return "h2o"
    if "llamacpp" in n:            return "llamacpp_stock"
    if "vanilla" in n:             return "vanilla"
    if n.startswith("v1") or "_v1_" in n or "v1_k" in n:
        return "v1"
    if n.startswith("k") and any(c.isdigit() for c in n):
        return "v1"
    return "other"


def max_cpu_temp_c(sensors: pd.DataFrame) -> float:
    """Peak SoC temperature in degrees C, computed as the maximum over all
    CPU thermal zones present in the sensors dataframe.

    Excludes hardware-trip pseudo-zones (cpu-hw-trip-*) per
    host_master_table_4policy.py."""
    cpu_cols = [c for c in sensors.columns
                if c.startswith("cpu") and c.endswith("_temp_mc")
                and "hw-trip" not in c]
    if not cpu_cols:
        return float("nan")
    block = sensors[cpu_cols].apply(pd.to_numeric, errors="coerce")
    # Per-row max -> overall max. Ignore zero/None readings.
    block = block.where(block > 0)
    if block.dropna(how="all").empty:
        return float("nan")
    return float(block.max(axis=1).max()) / 1000.0


def collect_cells() -> pd.DataFrame:
    """Walk phone-logs/* and return one row per cell with the four metrics
    needed by the plot."""
    rows: list[dict] = []
    for stress_csv in PHONE_LOGS.rglob("stress.csv"):
        cell_dir = stress_csv.parent
        sensors_csv = cell_dir / "sensors.csv"
        if not sensors_csv.exists():
            continue
        try:
            stress  = pd.read_csv(stress_csv)
            sensors = pd.read_csv(sensors_csv, low_memory=False)
        except Exception as e:
            print(f"[skip] {cell_dir}: parse failed ({e})", file=sys.stderr)
            continue
        if stress.empty or sensors.empty:
            continue

        # peak RSS in GB --------------------------------------------------
        if "peak_rss_kb" not in stress.columns:
            continue
        peak_rss_kb = float(np.nan_to_num(stress["peak_rss_kb"].max(),
                                          nan=0.0))
        if peak_rss_kb <= 0:
            continue
        rss_gb = peak_rss_kb / KB / KB

        # swap_MB ---------------------------------------------------------
        swap_mb = 0.0
        if "vmstat_pswpout" in sensors.columns:
            ps = pd.to_numeric(sensors["vmstat_pswpout"],
                               errors="coerce").dropna()
            if len(ps) >= 2:
                swap_mb = max(0.0, float(ps.max() - ps.min())) * PAGE_KB / KB

        # SoC temperature (peak across CPU thermal zones) ----------------
        soc_c = max_cpu_temp_c(sensors)

        # DDR temperature -------------------------------------------------
        ddr_c = float("nan")
        if "ddr_temp_mc" in sensors.columns:
            d = pd.to_numeric(sensors["ddr_temp_mc"],
                              errors="coerce").dropna()
            if not d.empty:
                ddr_c = float(d.max()) / 1000.0

        if np.isnan(soc_c) or np.isnan(ddr_c):
            continue

        wave = cell_dir.parent.name
        cell = cell_dir.name
        rows.append(dict(
            wave=wave,
            cell=cell,
            family=policy_family(cell),
            rss_gb=rss_gb,
            swap_mb=swap_mb,
            soc_c=soc_c,
            ddr_c=ddr_c,
        ))
    df = pd.DataFrame(rows)
    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (wave, cell) -- when the same exact directory shows up
    twice (rare nested copies), keep the heavier run."""
    if df.empty:
        return df
    df = df.sort_values("rss_gb", ascending=False).drop_duplicates(
        subset=["wave", "cell"], keep="first")
    return df.reset_index(drop=True)


def fit_line(x: np.ndarray, y: np.ndarray):
    """Return (slope, intercept, r2) for y ~ slope*x + intercept.

    r2 is the squared Pearson correlation. NaN if fewer than 2 points or
    if x has no variance."""
    if len(x) < 2 or np.ptp(x) <= 0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return float(slope), float(intercept), float(r * r)


def draw_panel(ax, df: pd.DataFrame, x_col: str, y_col: str,
               x_label: str, y_label: str, title: str,
               annotate_outliers: bool):
    """Render a single scatter+regression panel and return the fit dict."""
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    # Scatter, colored by policy family
    seen = {}
    for fam in POLICY_ORDER:
        sel = df["family"] == fam
        if not sel.any():
            continue
        ax.scatter(df.loc[sel, x_col], df.loc[sel, y_col],
                   s=42, alpha=0.85,
                   color=POLICY_COLOR.get(fam, "#444"),
                   edgecolors="white", linewidths=0.6,
                   label=f"{fam} (n={int(sel.sum())})")
        seen[fam] = int(sel.sum())

    # Regression line + R^2
    slope, intercept, r2 = fit_line(x, y)
    if not np.isnan(slope):
        xfit = np.linspace(x.min(), x.max(), 100)
        yfit = intercept + slope * xfit
        ax.plot(xfit, yfit, color="black", lw=1.6, ls="--",
                label=f"fit: y={intercept:.2f}+{slope:.3f}x  R^2={r2:.3f}")

    # Outlier annotations -- the two extreme-swap cells called out by the
    # caller (Wave-7's 1552 MB run and Wave-3's 515 MB run). Annotated on
    # the swap panels only.
    annotated: list[dict] = []
    if annotate_outliers:
        # Match by wave-prefix AND a target swap_mb (with a 50-MB tolerance
        # so re-runs of the same cell are still picked up correctly).
        targets = [
            ("wave7", 1552.0, "Wave-7  v1_fa2  (1552 MB swap)"),
            ("wave3", 515.0,  "Wave-3  v1_fa_K512  (515 MB swap)"),
        ]
        for wave_prefix, target_mb, label_text in targets:
            mask = df["wave"].str.startswith(wave_prefix) & \
                (df["swap_mb"].between(target_mb - 50, target_mb + 50))
            sub = df[mask]
            if sub.empty:
                continue
            # If multiple, take the one closest to the target swap value
            r = sub.iloc[(sub["swap_mb"] - target_mb).abs().argsort()].iloc[0]
            xx = float(r[x_col]); yy = float(r[y_col])
            ax.annotate(
                label_text, (xx, yy),
                textcoords="offset points", xytext=(12, 14),
                fontsize=8, color="#7a1f1f", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#7a1f1f", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5",
                          ec="#7a1f1f", lw=0.7, alpha=0.92),
            )
            annotated.append(dict(wave=str(r["wave"]), cell=str(r["cell"]),
                                  swap_mb=float(r["swap_mb"]),
                                  x=xx, y=yy, label=label_text))

    ax.set_xlabel(x_label, fontsize=10.5)
    ax.set_ylabel(y_label, fontsize=10.5)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.grid(alpha=0.3)

    return dict(
        x_col=x_col, y_col=y_col,
        n=int(len(df)),
        slope=slope, intercept=intercept, r2=r2,
        outliers=annotated,
        family_counts=seen,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = collect_cells()
    print(f"[collect] {len(df)} cells with stress.csv + sensors.csv",
          file=sys.stderr)
    if df.empty:
        print("No data; aborting.", file=sys.stderr)
        return 1
    df = dedupe(df)
    print(f"[dedupe]  {len(df)} unique (wave, cell) rows", file=sys.stderr)

    # Quick provenance dump so the schema can carry headline numbers
    with pd.option_context("display.max_rows", 200,
                           "display.width", 160,
                           "display.float_format", lambda x: f"{x:,.3f}"):
        print(df[["wave", "cell", "family",
                  "rss_gb", "swap_mb", "soc_c", "ddr_c"]].to_string(index=False),
              file=sys.stderr)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), dpi=140)
    fits: dict[str, dict] = {}
    fits["rss_vs_soc"] = draw_panel(
        axes[0][0], df, "soc_c", "rss_gb",
        "SoC peak temperature (deg C)", "Peak RSS (GB)",
        "RSS vs SoC temperature",
        annotate_outliers=False,
    )
    fits["rss_vs_ddr"] = draw_panel(
        axes[0][1], df, "ddr_c", "rss_gb",
        "DDR peak temperature (deg C)", "Peak RSS (GB)",
        "RSS vs DRAM temperature",
        annotate_outliers=False,
    )
    fits["swap_vs_soc"] = draw_panel(
        axes[1][0], df, "soc_c", "swap_mb",
        "SoC peak temperature (deg C)", "Swap activity (MB)",
        "Swap vs SoC temperature",
        annotate_outliers=True,
    )
    fits["swap_vs_ddr"] = draw_panel(
        axes[1][1], df, "ddr_c", "swap_mb",
        "DDR peak temperature (deg C)", "Swap activity (MB)",
        "Swap vs DRAM temperature",
        annotate_outliers=True,
    )

    # Single shared legend (policy color key) below the figure
    handles, labels = axes[0][0].get_legend_handles_labels()
    # de-dup while preserving order
    keep = []
    seen_labels = set()
    for h, l in zip(handles, labels):
        # drop the per-panel fit lines from the global legend
        if l.startswith("fit:"):
            continue
        # collapse the "(n=...)" trailing counts so the global legend isn't
        # tied to one panel's counts (they're identical anyway after dedup).
        base = l.split(" (n=")[0]
        if base in seen_labels:
            continue
        seen_labels.add(base)
        keep.append((h, base))
    if keep:
        fig.legend([h for h, _ in keep], [l for _, l in keep],
                   loc="lower center", ncol=min(len(keep), 6),
                   fontsize=9, frameon=True,
                   bbox_to_anchor=(0.5, -0.005),
                   title="Policy family", title_fontsize=9.5)

    fig.suptitle(
        "Memory utilization correlates with both SoC and DRAM temperature",
        fontsize=14, weight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {OUT_FIG}", file=sys.stderr)

    # PLOT_SCHEMA sidecar
    schema = dict(
        figure_name="14_memutil_vs_temp",
        file_path=str(OUT_FIG),
        title="Memory utilization correlates with both SoC and DRAM temperature",
        layout=dict(rows=2, cols=2),
        panels=[
            dict(position="top-left",  **fits["rss_vs_soc"]),
            dict(position="top-right", **fits["rss_vs_ddr"]),
            dict(position="bot-left",  **fits["swap_vs_soc"]),
            dict(position="bot-right", **fits["swap_vs_ddr"]),
        ],
        n_cells=int(len(df)),
        source="phone-logs/<wave>/<cell>/{sensors.csv,stress.csv}",
        metrics={
            "rss_gb":  "max(stress.peak_rss_kb) / 1024 / 1024",
            "swap_mb": "(max-min)(sensors.vmstat_pswpout) * 4 KB / 1024",
            "soc_c":   "max over CPU thermal zones (cpu*_temp_mc) / 1000",
            "ddr_c":   "max(sensors.ddr_temp_mc) / 1000",
        },
    )
    OUT_SCHEMA.write_text(json.dumps(schema, indent=2))
    print(f"[plot] wrote {OUT_SCHEMA}", file=sys.stderr)

    # Print PLOT_SCHEMA to stdout (per task instruction)
    print("PLOT_SCHEMA")
    print(json.dumps(schema, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
