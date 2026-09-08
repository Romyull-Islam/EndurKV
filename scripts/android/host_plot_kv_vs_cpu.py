#!/usr/bin/env python3
"""
PLOT 13 - KV cache growth vs CPU temperature (Wave-8 race-to-idle finding).

Two-panel figure built from the wave10 K-sweep (K256 / K384 / K1024), where
the only knob varied between cells is the eviction budget K -- everything
else (model, dataset, hardware pinning, watchdog) is identical.

Top panel  : time-series. Cache size (n_kv_cells, per-decode-step) over wall
             time overlaid with the big-core CPU temperature for each policy.
Bottom panel: scatter of CPU temp (C) versus the live cache size (cells)
              with a linear regression line and R^2 per policy plus an
              all-data fit.

CPU sensor column choice
------------------------
Per task spec, CPU temp is the per-row MAX across the big-core cluster:
    cpu-1-0-0_temp_mc, cpu-1-0-1_temp_mc,
    cpu-1-1-0_temp_mc, cpu-1-1-1_temp_mc,
    cpullc-1-0_temp_mc, cpullc-1-1_temp_mc
Single best column for plotting is cpu-1-0-0_temp_mc (prime-core sensor that
tracks decoding load); we plot that one as the line on the top panel and use
the per-row MAX for the scatter so the relationship is robust to which
big-core happens to be loaded.

Output
------
    EndurKV/figures/relationship_plots/13_kv_cache_vs_cpu_temp.png
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------- config -----------------------------------------

WAVE10_DIR = Path(
    "/home/mislam22/EndurKV_workspace/phone-logs/wave10_ksweep_1780815847"
)
OUT_PATH = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/"
    "13_kv_cache_vs_cpu_temp.png"
)

# Policy cells: (dir_name, display_label, color)
# Smaller K = more aggressive eviction -> smaller cache -> faster decode.
POLICIES = [
    ("K256",  "K=256  (aggressive eviction)",   "#d62728"),  # red
    ("K384",  "K=384  (moderate)",              "#1f77b4"),  # blue
    ("K1024", "K=1024 (loose, near-vanilla)",   "#2ca02c"),  # green
]

# Big-core CPU temperature columns (per task spec).
CPU_BIG_COLS = [
    "cpu-1-0-0_temp_mc",
    "cpu-1-0-1_temp_mc",
    "cpu-1-1-0_temp_mc",
    "cpu-1-1-1_temp_mc",
    "cpullc-1-0_temp_mc",
    "cpullc-1-1_temp_mc",
]
CPU_PRIMARY_COL = "cpu-1-0-0_temp_mc"


# ---------------------------- helpers -----------------------------------------

def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_sensors(cell_dir: Path):
    """Return arrays (t_s, cpu_primary_C, cpu_max_C) with t_s rebased to 0."""
    t = []
    cpu_primary = []
    cpu_cols_vals = {c: [] for c in CPU_BIG_COLS}
    with open(cell_dir / "sensors.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t.append(_to_float(row.get("monotonic_s")))
            cpu_primary.append(
                _to_float(row.get(CPU_PRIMARY_COL)) / 1000.0
            )
            for c in CPU_BIG_COLS:
                cpu_cols_vals[c].append(_to_float(row.get(c)) / 1000.0)
    t_arr = np.asarray(t, dtype=float)
    cpu_primary_arr = np.asarray(cpu_primary, dtype=float)
    stack = np.vstack(
        [np.asarray(cpu_cols_vals[c], dtype=float) for c in CPU_BIG_COLS]
    )
    # Per-row max across the big-core cluster; ignore NaN.
    with np.errstate(invalid="ignore"):
        cpu_max_arr = np.nanmax(stack, axis=0)
    if t_arr.size:
        t_arr = t_arr - t_arr[0]
    # Filter obviously bad reads (<10 C, >110 C).
    bad_primary = (cpu_primary_arr < 10) | (cpu_primary_arr > 110)
    bad_max = (cpu_max_arr < 10) | (cpu_max_arr > 110)
    cpu_primary_arr = np.where(bad_primary, np.nan, cpu_primary_arr)
    cpu_max_arr = np.where(bad_max, np.nan, cpu_max_arr)
    return t_arr, cpu_primary_arr, cpu_max_arr


def load_iter_starts(cell_dir: Path):
    """Return {iter:int -> t_elapsed_s:float} from stress.csv."""
    starts = {}
    p = cell_dir / "stress.csv"
    if not p.exists():
        return starts
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                starts[int(row["iter"])] = float(row["t_elapsed_s"])
            except (TypeError, ValueError, KeyError):
                continue
    return starts


def load_cache_series(cell_dir: Path):
    """Return (t_arr_s, n_kv_arr) concatenated across iters, sorted by time.

    steps.csv wall_us is microseconds since iter's prefill start.
    stress.csv t_elapsed_s gives iter start relative to run start (same
    reference as sensors monotonic_s - monotonic_s[0]).
    """
    iter_start = load_iter_starts(cell_dir)
    t_all, kv_all = [], []
    for it, t_iter in sorted(iter_start.items()):
        steps_path = cell_dir / f"iter{it:04d}" / "steps.csv"
        if not steps_path.exists():
            continue
        with open(steps_path, newline="") as fh:
            for row in csv.DictReader(fh):
                w = _to_float(row.get("wall_us"))
                n = _to_float(row.get("n_kv_cells"))
                if not (np.isfinite(w) and np.isfinite(n)):
                    continue
                t_all.append(t_iter + w / 1.0e6)
                kv_all.append(n)
    if not t_all:
        return np.array([]), np.array([])
    t_arr = np.asarray(t_all, dtype=float)
    n_arr = np.asarray(kv_all, dtype=float)
    order = np.argsort(t_arr)
    return t_arr[order], n_arr[order]


def resample_to_sensor_time(
    sensor_t: np.ndarray,
    cache_t: np.ndarray,
    cache_n: np.ndarray,
) -> np.ndarray:
    """Interpolate cache_n onto sensor_t (left/right held at edges)."""
    if cache_t.size == 0 or sensor_t.size == 0:
        return np.full_like(sensor_t, np.nan, dtype=float)
    return np.interp(sensor_t, cache_t, cache_n,
                     left=cache_n[0], right=cache_n[-1])


def linreg(x: np.ndarray, y: np.ndarray):
    """Return (slope, intercept, r2, n) with NaN-safe filtering."""
    mask = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[mask], y[mask]
    if xs.size < 3:
        return float("nan"), float("nan"), float("nan"), int(xs.size)
    slope, intercept = np.polyfit(xs, ys, 1)
    yhat = slope * xs + intercept
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2), int(xs.size)


# ------------------------------ plot ------------------------------------------

def main() -> int:
    if not WAVE10_DIR.exists():
        print(f"ERROR: missing wave10 dir: {WAVE10_DIR}", file=sys.stderr)
        return 1
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    for cell, label, color in POLICIES:
        cell_dir = WAVE10_DIR / cell
        if not cell_dir.exists():
            print(f"WARN: missing cell {cell_dir}", file=sys.stderr)
            continue
        t_s, cpu_prim, cpu_max = load_sensors(cell_dir)
        cache_t, cache_n = load_cache_series(cell_dir)
        cache_resamp = resample_to_sensor_time(t_s, cache_t, cache_n)
        data[cell] = dict(
            label=label, color=color,
            t=t_s, cpu_primary=cpu_prim, cpu_max=cpu_max,
            cache_t=cache_t, cache_n=cache_n,
            cache_at_sensor=cache_resamp,
        )

    # Stats summary
    stats_lines = []
    for cell, d in data.items():
        cmax = d["cpu_max"]
        cprim = d["cpu_primary"]
        if cmax.size and np.isfinite(cmax).any():
            peak_max = float(np.nanmax(cmax))
            mean_max = float(np.nanmean(cmax))
        else:
            peak_max = float("nan"); mean_max = float("nan")
        if cprim.size and np.isfinite(cprim).any():
            peak_prim = float(np.nanmax(cprim))
            mean_prim = float(np.nanmean(cprim))
        else:
            peak_prim = float("nan"); mean_prim = float("nan")
        kv = d["cache_n"]
        peak_kv = float(np.nanmax(kv)) if kv.size else float("nan")
        stats_lines.append(
            f"  {cell:6s}  peak_kv={peak_kv:5.0f}  "
            f"cpu_max peak={peak_max:5.2f}C mean={mean_max:5.2f}C  "
            f"cpu-1-0-0 peak={peak_prim:5.2f}C mean={mean_prim:5.2f}C"
        )
    for ln in stats_lines:
        print(ln)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 9.8),
        gridspec_kw={"hspace": 0.30, "height_ratios": [1.0, 1.05]},
    )
    ax_ts, ax_sc = axes

    # ============ TOP: time-series ============
    ax_ts_cpu = ax_ts.twinx()
    for cell, d in data.items():
        c, lbl = d["color"], d["label"]
        if d["cache_t"].size:
            ax_ts.plot(
                d["cache_t"], d["cache_n"],
                color=c, lw=1.1, alpha=0.85,
                label=f"{lbl}  cache",
            )
        if d["t"].size:
            ax_ts_cpu.plot(
                d["t"], d["cpu_primary"],
                color=c, lw=1.3, alpha=0.95, linestyle="--",
                label=f"{lbl}  CPU temp",
            )

    ax_ts.set_ylabel("KV cache size (n_kv_cells)")
    ax_ts_cpu.set_ylabel("CPU big-core temp (deg C) - cpu-1-0-0")
    ax_ts.set_xlabel("Wall time since cell start (s)")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.set_title(
        "Top: cache growth (solid) and big-core CPU temp (dashed) per policy"
    )

    # Combined legend (cache solid + cpu dashed)
    h1, l1 = ax_ts.get_legend_handles_labels()
    h2, l2 = ax_ts_cpu.get_legend_handles_labels()
    ax_ts.legend(h1 + h2, l1 + l2, loc="lower right",
                 fontsize=8.5, framealpha=0.9, ncol=2)

    # ============ BOTTOM: scatter + regression ============
    all_x, all_y = [], []
    for cell, d in data.items():
        c, lbl = d["color"], d["label"]
        x = d["cache_at_sensor"]
        y = d["cpu_max"]
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            continue
        x_m, y_m = x[mask], y[mask]

        ax_sc.scatter(
            x_m, y_m, s=8, alpha=0.30, color=c,
            edgecolors="none", rasterized=True,
        )

        slope, intercept, r2, n_pts = linreg(x_m, y_m)
        if np.isfinite(slope):
            xfit = np.linspace(float(x_m.min()), float(x_m.max()), 200)
            yfit = slope * xfit + intercept
            ax_sc.plot(
                xfit, yfit, color=c, lw=2.0,
                label=(
                    f"{lbl}  slope={slope*100:+.2f} C / 100 cells  "
                    f"R^2={r2:.2f}  (n={n_pts:,})"
                ),
            )

        all_x.append(x_m); all_y.append(y_m)

    # Global fit (all policies aggregated)
    if all_x:
        X = np.concatenate(all_x); Y = np.concatenate(all_y)
        g_slope, g_intercept, g_r2, g_n = linreg(X, Y)
        if np.isfinite(g_slope):
            xg = np.linspace(float(X.min()), float(X.max()), 200)
            yg = g_slope * xg + g_intercept
            ax_sc.plot(
                xg, yg, color="black", lw=2.2, ls="--",
                label=(
                    f"ALL policies  slope={g_slope*100:+.2f} C / 100 cells  "
                    f"R^2={g_r2:.2f}  (n={g_n:,})"
                ),
            )
        print(
            f"[fit-global] slope={g_slope:.4f}  intercept={g_intercept:.2f}  "
            f"R2={g_r2:.3f}  n={g_n}"
        )

    ax_sc.set_xlabel("KV cache size at sensor sample (cells)")
    ax_sc.set_ylabel("Big-core CPU temp (deg C) - max across cluster")
    ax_sc.set_title(
        "Bottom: scatter of CPU temp vs live cache size with per-policy "
        "linear fit"
    )
    ax_sc.grid(True, alpha=0.3)
    ax_sc.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        "Race-to-idle effect: aggressive eviction can raise CPU temp via "
        "higher throughput",
        fontsize=13.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
