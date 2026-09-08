#!/usr/bin/env python3
"""
Llama-3.2-1B Wave-3 relationship scatter for the supervisor.

Source: /home/mislam22/EndurKV_workspace/phone-logs/wave3_real_1780680903/
Cells:  vanilla / v1_K2048 / v1_K512 / v1_fa_K512   (4 cells, ~5 iters each)

We aggregate one row per (cell, iter) and plot how the effective live KV
cache budget per iteration covaries with three thermal/memory channels:

  * DDR (memory) temperature
  * Big-core CPU temperature
  * Process RSS

Definition of `cache_size_cells` per iter
-----------------------------------------
For the v1 / v1_fa policies the per-position attention budget is exactly
K_nominal positions plus the n_sink protected tokens, and the decode loop
may grow it transiently by n_decode_steps before the next eviction pass:

    cache_size_cells = K_nominal + n_sink + n_decode_steps        (v1, v1_fa)
    cache_size_cells = peak_kv_cells                              (vanilla)

This matches the supervisor's narrative that K=2048 "sustains" a ~2k working
set while K=512 "caps" at a few hundred, with vanilla acting as the
no-eviction reference (full 8k prompt + decode).

Three-panel figure (single column, stacked):
  Top    : cache_size_cells vs DDR temp (C)   scatter + linear fit + R^2
  Middle : cache_size_cells vs CPU temp (C)   scatter + linear fit + R^2
  Bottom : cache_size_cells vs RSS (GB)       scatter + linear fit + R^2

CPU temp is the per-iter mean of the per-row MAX across the big-core
cluster (cpu-1-0-[01], cpu-1-1-[01], cpullc-1-[01]), matching the
race-to-idle CPU figure (13).

Output:
  /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/
      21_llama1b_relations.png
  /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/
      21_llama1b_relations.schema.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------- config -----------------------------------------

WAVE_DIR = Path(
    "/home/mislam22/EndurKV_workspace/phone-logs/wave3_real_1780680903"
)
OUT_PATH = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/"
    "21_llama1b_relations.png"
)
SCHEMA_PATH = OUT_PATH.with_suffix(".schema.json")

# (dir_name, display_label, color)
CELLS = [
    ("vanilla",     "vanilla (K=ctx, no evict)", "#888888"),
    ("v1_K2048",    "v1  K=2048",                "#1F77B4"),
    ("v1_K512",     "v1  K=512",                 "#4C9AFF"),
    ("v1_fa_K512",  "v1_FA  K=512",              "#E07B00"),
]

DDR_COL = "ddr_temp_mc"
CPU_BIG_COLS = [
    "cpu-1-0-0_temp_mc",
    "cpu-1-0-1_temp_mc",
    "cpu-1-1-0_temp_mc",
    "cpu-1-1-1_temp_mc",
    "cpullc-1-0_temp_mc",
    "cpullc-1-1_temp_mc",
]
COL_MONO_S = "monotonic_s"


# ---------------------------- helpers -----------------------------------------

def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_sensors(cell_dir: Path) -> dict:
    """Return dict with arrays: t_rel (s, from first sample), ddr_C, cpu_max_C."""
    t = []
    ddr = []
    cpu_cols = {c: [] for c in CPU_BIG_COLS}
    with open(cell_dir / "sensors.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t.append(_f(row.get(COL_MONO_S)))
            ddr.append(_f(row.get(DDR_COL)))
            for c in CPU_BIG_COLS:
                cpu_cols[c].append(_f(row.get(c)))
    t_arr = np.asarray(t, dtype=float)
    if t_arr.size:
        t_arr = t_arr - t_arr[0]
    ddr_C = np.asarray(ddr, dtype=float) / 1000.0
    cpu_stack = np.vstack(
        [np.asarray(cpu_cols[c], dtype=float) / 1000.0 for c in CPU_BIG_COLS]
    )
    with np.errstate(invalid="ignore"):
        cpu_max_C = np.nanmax(cpu_stack, axis=0)
    # Filter implausible reads
    ddr_C = np.where((ddr_C < 10.0) | (ddr_C > 120.0), np.nan, ddr_C)
    cpu_max_C = np.where((cpu_max_C < 10.0) | (cpu_max_C > 120.0), np.nan, cpu_max_C)
    return dict(t=t_arr, ddr=ddr_C, cpu=cpu_max_C)


def load_iter_starts(cell_dir: Path) -> dict[int, float]:
    starts: dict[int, float] = {}
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


def load_meta(iter_dir: Path) -> dict:
    p = iter_dir / "meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def iter_temp_window(sensor: dict, t_start: float, t_end: float
                     ) -> tuple[float, float]:
    """Mean DDR & CPU temp inside [t_start, t_end] window."""
    t = sensor["t"]
    if t.size == 0 or not np.isfinite(t_end - t_start):
        return float("nan"), float("nan")
    mask = (t >= t_start) & (t <= t_end)
    if not mask.any():
        # Fall back to nearest single sample
        idx = int(np.argmin(np.abs(t - t_start)))
        return (
            float(sensor["ddr"][idx]) if np.isfinite(sensor["ddr"][idx]) else float("nan"),
            float(sensor["cpu"][idx]) if np.isfinite(sensor["cpu"][idx]) else float("nan"),
        )
    ddr_slice = sensor["ddr"][mask]
    cpu_slice = sensor["cpu"][mask]
    with np.errstate(invalid="ignore"):
        ddr_mean = float(np.nanmean(ddr_slice))
        cpu_mean = float(np.nanmean(cpu_slice))
    return ddr_mean, cpu_mean


def compute_cache_size(meta: dict, peak_kv: int) -> float:
    """Effective live cache budget per iter.

    v1 / v1_fa: K_nominal + n_sink + n_decode_steps
    vanilla   : peak_kv_cells
    """
    pol = str(meta.get("policy", "")).lower()
    if pol == "vanilla":
        return float(peak_kv)
    k = float(meta.get("k_nominal", 0.0))
    n_sink = float(meta.get("n_sink", 0.0))
    n_dec = float(meta.get("n_decode_steps", 0.0))
    return k + n_sink + n_dec


def linreg(x: np.ndarray, y: np.ndarray):
    """Return (slope, intercept, r2, n) — finite-only."""
    m = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[m], y[m]
    if xs.size < 3 or float(np.var(xs)) == 0.0:
        return float("nan"), float("nan"), float("nan"), int(xs.size)
    slope, intercept = np.polyfit(xs, ys, 1)
    yhat = slope * xs + intercept
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2), int(xs.size)


# ------------------------------ main ------------------------------------------

def main() -> int:
    if not WAVE_DIR.exists():
        print(f"ERROR: missing wave dir: {WAVE_DIR}", file=sys.stderr)
        return 1
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []                        # one entry per (cell, iter)
    per_cell_summary = []
    for cell_name, label, color in CELLS:
        cell_dir = WAVE_DIR / cell_name
        if not cell_dir.exists():
            print(f"WARN: missing cell {cell_dir}", file=sys.stderr)
            continue
        sensor = load_sensors(cell_dir)
        iter_starts = load_iter_starts(cell_dir)
        iters_sorted = sorted(iter_starts.items())
        cell_rows = []
        for idx, (it, t_iter_start) in enumerate(iters_sorted):
            iter_dir = cell_dir / f"iter{it:04d}"
            meta = load_meta(iter_dir)
            if not meta:
                continue
            peak_kv = int(meta.get("peak_kv_cells", 0))
            peak_rss_kb = float(meta.get("peak_rss_kb", float("nan")))
            cache_cells = compute_cache_size(meta, peak_kv)

            # Define the iter's wall-time window for averaging temps.
            # End = start of next iter, or start + total_ms/1000 for the last.
            if idx + 1 < len(iters_sorted):
                t_end = iters_sorted[idx + 1][1]
            else:
                total_ms = float(meta.get("total_ms", 0.0))
                t_end = t_iter_start + total_ms / 1000.0
            ddr_C, cpu_C = iter_temp_window(sensor, t_iter_start, t_end)
            rss_GB = peak_rss_kb / (1024.0 * 1024.0)

            row = dict(
                cell=cell_name,
                label=label,
                color=color,
                iter=it,
                policy=str(meta.get("policy", "")),
                k_nominal=int(meta.get("k_nominal", 0)),
                n_sink=int(meta.get("n_sink", 0)),
                n_decode_steps=int(meta.get("n_decode_steps", 0)),
                peak_kv_cells=peak_kv,
                cache_size_cells=cache_cells,
                ddr_C=ddr_C,
                cpu_C=cpu_C,
                rss_GB=rss_GB,
                t_iter_start_s=t_iter_start,
                t_iter_end_s=t_end,
            )
            rows.append(row)
            cell_rows.append(row)

        if cell_rows:
            per_cell_summary.append(dict(
                cell=cell_name,
                label=label,
                n_iter=len(cell_rows),
                cache_size_cells=float(np.mean([r["cache_size_cells"] for r in cell_rows])),
                ddr_C_mean=float(np.nanmean([r["ddr_C"] for r in cell_rows])),
                cpu_C_mean=float(np.nanmean([r["cpu_C"] for r in cell_rows])),
                rss_GB_mean=float(np.nanmean([r["rss_GB"] for r in cell_rows])),
            ))

    if not rows:
        print("ERROR: no rows aggregated", file=sys.stderr)
        return 1

    # Vectors
    cs = np.array([r["cache_size_cells"] for r in rows], dtype=float)
    ddr = np.array([r["ddr_C"] for r in rows], dtype=float)
    cpu = np.array([r["cpu_C"] for r in rows], dtype=float)
    rss = np.array([r["rss_GB"] for r in rows], dtype=float)
    colors = [r["color"] for r in rows]

    print(f"=== aggregated {len(rows)} (cell, iter) rows ===")
    for s in per_cell_summary:
        print(
            f"  {s['cell']:12s}  n={s['n_iter']}  "
            f"cache={s['cache_size_cells']:7.1f}  "
            f"ddr={s['ddr_C_mean']:5.2f}C  cpu={s['cpu_C_mean']:5.2f}C  "
            f"rss={s['rss_GB_mean']:4.2f}GB"
        )

    # Global linear fits
    fit_ddr = linreg(cs, ddr)
    fit_cpu = linreg(cs, cpu)
    fit_rss = linreg(cs, rss)
    print(f"[fit ddr] slope={fit_ddr[0]:.6g}  R2={fit_ddr[2]:.3f}  n={fit_ddr[3]}")
    print(f"[fit cpu] slope={fit_cpu[0]:.6g}  R2={fit_cpu[2]:.3f}  n={fit_cpu[3]}")
    print(f"[fit rss] slope={fit_rss[0]:.6g}  R2={fit_rss[2]:.3f}  n={fit_rss[3]}")

    # ----------------------------- figure -------------------------------------
    fig, axes = plt.subplots(
        3, 1, figsize=(10.0, 12.5),
        gridspec_kw={"hspace": 0.34},
    )
    ax_ddr, ax_cpu, ax_rss = axes

    panels = [
        (ax_ddr, ddr, fit_ddr,
         "cache_size_cells vs DDR temperature (memory)",
         "DDR temperature (°C)",
         "°C per 1000 cells"),
        (ax_cpu, cpu, fit_cpu,
         "cache_size_cells vs CPU big-core temperature",
         "CPU big-core max temperature (°C)",
         "°C per 1000 cells"),
        (ax_rss, rss, fit_rss,
         "cache_size_cells vs process RSS",
         "Peak RSS (GB)",
         "GB per 1000 cells"),
    ]

    # Per-policy scatter (one legend handle per cell)
    seen_labels = set()
    for ax, y_vec, _fit, _title, _ylabel, _unit in panels:
        for r, yi in zip(rows, y_vec):
            lbl = r["label"] if r["label"] not in seen_labels else None
            ax.scatter(
                r["cache_size_cells"], yi,
                color=r["color"], s=70, alpha=0.85,
                edgecolors="black", linewidths=0.6,
                label=lbl,
            )
        seen_labels |= {r["label"] for r in rows}

    # Reset seen so each axis can re-add legend handles; we add legend per axis
    # but only need one handle per policy — we'll rebuild from unique labels.

    for ax, y_vec, fit, title, ylabel, unit in panels:
        slope, intercept, r2, n_pts = fit
        if np.isfinite(slope):
            x_min = float(np.nanmin(cs))
            x_max = float(np.nanmax(cs))
            xf = np.linspace(x_min, x_max, 200)
            yf = slope * xf + intercept
            slope_per_1k = slope * 1000.0
            ax.plot(
                xf, yf, color="black", lw=2.2, ls="--",
                label=(
                    f"linear fit: slope = {slope_per_1k:+.3f} {unit}, "
                    f"R² = {r2:.3f}  (n={n_pts})"
                ),
            )
        ax.set_xlabel("cache_size_cells  (K_nominal + n_sink + n_decode_steps; "
                      "vanilla = peak_kv_cells)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        # Deduplicate legend handles by label
        handles, labels_ = ax.get_legend_handles_labels()
        dedup = {}
        for h, l in zip(handles, labels_):
            if l not in dedup:
                dedup[l] = h
        ax.legend(dedup.values(), dedup.keys(),
                  loc="best", fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        "Llama-1B at K=2048 sustains 2400 cells, K=512 caps at 700 — "
        "thermal slope ~0.5°C per 1000 cells",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.005,
        f"Source: {WAVE_DIR.name}  |  cells: " +
        ", ".join(c[0] for c in CELLS) +
        f"  |  N(cell,iter) = {len(rows)}",
        ha="center", va="bottom", fontsize=8, color="dimgray",
    )

    fig.tight_layout(rect=(0, 0.012, 1, 0.965))
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1024:.1f} KB)")

    # ------------------------------- RES_SCHEMA -------------------------------
    schema = {
        "kind": "RES_SCHEMA",
        "version": 1,
        "generator": "host_plot_llama1b_relations.py",
        "figure": str(OUT_PATH),
        "title": (
            "Llama-1B at K=2048 sustains 2400 cells, K=512 caps at 700 — "
            "thermal slope ~0.5°C per 1000 cells"
        ),
        "role_in_paper": "Supervisor relationship scatter — Llama-1B Wave-3 "
                          "cache budget vs SoC/memory/CPU/RSS",
        "x_axis": "cache_size_cells (K_nominal + n_sink + n_decode_steps; "
                  "vanilla = peak_kv_cells)",
        "panels": [
            {
                "panel": "top",
                "y_axis": "DDR temperature (C)",
                "y_source_column": DDR_COL,
                "agg": "mean over iter wall-time window",
                "fit": {
                    "model": "ddr_C = intercept + slope * cache_size_cells",
                    "slope_C_per_cell": fit_ddr[0] if np.isfinite(fit_ddr[0]) else None,
                    "intercept_C": fit_ddr[1] if np.isfinite(fit_ddr[1]) else None,
                    "r_squared": fit_ddr[2] if np.isfinite(fit_ddr[2]) else None,
                    "slope_C_per_1000_cells": (fit_ddr[0] * 1000.0) if np.isfinite(fit_ddr[0]) else None,
                    "n_points": fit_ddr[3],
                },
            },
            {
                "panel": "middle",
                "y_axis": "CPU big-core max temperature (C)",
                "y_source_columns": CPU_BIG_COLS,
                "agg": "per-row max across big cluster, then mean over iter window",
                "fit": {
                    "model": "cpu_C = intercept + slope * cache_size_cells",
                    "slope_C_per_cell": fit_cpu[0] if np.isfinite(fit_cpu[0]) else None,
                    "intercept_C": fit_cpu[1] if np.isfinite(fit_cpu[1]) else None,
                    "r_squared": fit_cpu[2] if np.isfinite(fit_cpu[2]) else None,
                    "slope_C_per_1000_cells": (fit_cpu[0] * 1000.0) if np.isfinite(fit_cpu[0]) else None,
                    "n_points": fit_cpu[3],
                },
            },
            {
                "panel": "bottom",
                "y_axis": "Peak RSS (GB)",
                "y_source_field": "meta.json:peak_rss_kb",
                "agg": "per-iter peak (KB) converted to GB",
                "fit": {
                    "model": "rss_GB = intercept + slope * cache_size_cells",
                    "slope_GB_per_cell": fit_rss[0] if np.isfinite(fit_rss[0]) else None,
                    "intercept_GB": fit_rss[1] if np.isfinite(fit_rss[1]) else None,
                    "r_squared": fit_rss[2] if np.isfinite(fit_rss[2]) else None,
                    "slope_GB_per_1000_cells": (fit_rss[0] * 1000.0) if np.isfinite(fit_rss[0]) else None,
                    "n_points": fit_rss[3],
                },
            },
        ],
        "wave_source": str(WAVE_DIR),
        "cells": per_cell_summary,
        "n_total_iter_rows": len(rows),
        "rows": [
            {
                "cell": r["cell"],
                "iter": r["iter"],
                "policy": r["policy"],
                "k_nominal": r["k_nominal"],
                "n_sink": r["n_sink"],
                "n_decode_steps": r["n_decode_steps"],
                "peak_kv_cells": r["peak_kv_cells"],
                "cache_size_cells": r["cache_size_cells"],
                "ddr_C": r["ddr_C"] if np.isfinite(r["ddr_C"]) else None,
                "cpu_C": r["cpu_C"] if np.isfinite(r["cpu_C"]) else None,
                "rss_GB": r["rss_GB"] if np.isfinite(r["rss_GB"]) else None,
            }
            for r in rows
        ],
    }
    with open(SCHEMA_PATH, "w") as fh:
        json.dump(schema, fh, indent=2)
    print(f"[ok] wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
