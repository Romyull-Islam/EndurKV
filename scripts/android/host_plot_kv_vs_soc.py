#!/usr/bin/env python3
"""
KV cache growth vs SoC temperature.

Sensor selection (per dataset note):
  * Primary SoC = column socd_temp_mc (the dedicated SoC-die thermal zone,
    millidegrees C).  In some phone-log captures this column is dead (all 0).
    When that happens we fall back to the documented skin-style proxy
    shell_front_temp_mc, so the figure always carries a meaningful curve.
  * The figure annotates which sensor was used per cell.
  * skin_temp_dc / soc_pmic_temp / msm_therm_aps / package_temp do NOT exist
    in these CSVs and are intentionally not consulted.
  * Memory temperature column referenced elsewhere is ddr_temp_mc, but this
    figure only needs SoC.

Cells used:
  * Wave-4 long-decode trio   : vanilla, v1_K512, v1_fa_K512
  * Wave-9 stack overlay      : v1_fa2_stack

Figure: 2 panels
  - Top    : time-series, KV cache size (left y, log) + SoC temp (right y, °C)
             for every policy.
  - Bottom : scatter cache_size vs SoC temp across all samples, with linear
             regression line and R^2.

Output: figures/relationship_plots/11_kv_cache_vs_soc_temp.png

Stdlib + numpy + matplotlib only (matches the rest of the repo).
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

PHONE_LOGS = Path("/home/mislam22/EndurKV_workspace/phone-logs")
WAVE4_DIR = PHONE_LOGS / "wave4_longdecode_1780750084"
WAVE9_DIR = PHONE_LOGS / "wave9_v1fa2_stack_1780796320"

OUT_PATH = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/"
    "11_kv_cache_vs_soc_temp.png"
)
SCHEMA_PATH = OUT_PATH.with_suffix(".schema.json")

# (cell_dir, label, color, source_wave_label)
CELLS = [
    (WAVE4_DIR / "vanilla",        "vanilla (W4)",      "#d62728", "wave4"),
    (WAVE4_DIR / "v1_K512",        "v1 K=512 (W4)",     "#1f77b4", "wave4"),
    (WAVE4_DIR / "v1_fa_K512",     "v1_fa K=512 (W4)",  "#2ca02c", "wave4"),
    (WAVE9_DIR / "v1_fa2_stack",   "v1_fa2_stack (W9)", "#ff7f0e", "wave9"),
]

# Columns
COL_TIME = "monotonic_s"
COL_SOC_PRIMARY = "socd_temp_mc"
COL_SOC_FALLBACK = "shell_front_temp_mc"


# ---------------------------- helpers -----------------------------------------

def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _is_dead(values: np.ndarray) -> bool:
    """A sensor column is 'dead' if it's all-NaN, all-zero, or essentially
    constant (range < 0.1°C i.e. < 100 milli-deg)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    nonzero = finite[finite > 0]
    if nonzero.size < 5:
        return True
    return float(nonzero.max() - nonzero.min()) < 100.0


def load_sensors(cell_dir: Path):
    """Return (t_s_from_cell_start, soc_c, used_col_name)."""
    t, soc_primary, soc_fallback = [], [], []
    with open(cell_dir / "sensors.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t.append(_to_float(row.get(COL_TIME)))
            soc_primary.append(_to_float(row.get(COL_SOC_PRIMARY)))
            soc_fallback.append(_to_float(row.get(COL_SOC_FALLBACK)))
    t = np.asarray(t, dtype=float)
    soc_primary = np.asarray(soc_primary, dtype=float)
    soc_fallback = np.asarray(soc_fallback, dtype=float)
    if t.size:
        t = t - t[0]

    if not _is_dead(soc_primary):
        soc_c = soc_primary / 1000.0
        used = COL_SOC_PRIMARY
    else:
        soc_c = soc_fallback / 1000.0
        used = COL_SOC_FALLBACK + " (fallback; socd_temp_mc dead)"
    # Filter obviously bogus readings (sensor glitches)
    bad = (soc_c < 10.0) | (soc_c > 110.0) | ~np.isfinite(soc_c)
    soc_c[bad] = np.nan
    return t, soc_c, used


def load_cache_series(cell_dir: Path):
    """Per-step (t_s, n_kv) concatenated across iters, aligned to run wall-time.

    steps.csv wall_us  : microseconds since that iter's prefill start.
    stress.csv t_elapsed_s : iter start relative to run start (same reference
                             as sensors' monotonic_s - monotonic_s[0]).
    """
    stress_p = cell_dir / "stress.csv"
    if not stress_p.exists():
        return np.array([]), np.array([])

    iter_start = {}
    with open(stress_p, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                iter_start[int(row["iter"])] = float(row["t_elapsed_s"])
            except (TypeError, ValueError, KeyError):
                continue

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


def resample_cache_at_times(sample_times: np.ndarray,
                            cache_t: np.ndarray,
                            cache_n: np.ndarray) -> np.ndarray:
    """Step-wise resample of cache_n at sample_times.  Uses last-known cache
    size at each sensor sample (held constant between decode steps).  Samples
    earlier than the first cache event are returned as NaN."""
    out = np.full_like(sample_times, np.nan, dtype=float)
    if cache_t.size == 0:
        return out
    # cache_t is sorted (load_cache_series did argsort).
    idx = np.searchsorted(cache_t, sample_times, side="right") - 1
    valid = idx >= 0
    out[valid] = cache_n[idx[valid]]
    return out


# ------------------------------- main -----------------------------------------

def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for cell_dir, label, color, wave_label in CELLS:
        if not cell_dir.exists():
            print(f"WARN: missing cell {cell_dir}", file=sys.stderr)
            continue
        t_s, soc_c, soc_used = load_sensors(cell_dir)
        cache_t, cache_n = load_cache_series(cell_dir)
        data.append(dict(
            cell=cell_dir.name, label=label, color=color, wave=wave_label,
            t=t_s, soc=soc_c, cache_t=cache_t, cache_n=cache_n,
            soc_col=soc_used,
        ))

    if not data:
        print("ERROR: no cells loaded", file=sys.stderr)
        return 1

    # ---------- aggregate scatter data: align cache and SoC at sensor times ---
    scat_x, scat_y, scat_c, scat_lbl = [], [], [], []
    for d in data:
        if d["t"].size == 0 or d["cache_t"].size == 0:
            continue
        kv_at_sensor = resample_cache_at_times(d["t"], d["cache_t"], d["cache_n"])
        ok = np.isfinite(kv_at_sensor) & np.isfinite(d["soc"]) & (kv_at_sensor > 0)
        scat_x.append(kv_at_sensor[ok])
        scat_y.append(d["soc"][ok])
        scat_c.extend([d["color"]] * int(ok.sum()))
        scat_lbl.extend([d["label"]] * int(ok.sum()))

    scat_x = np.concatenate(scat_x) if scat_x else np.array([])
    scat_y = np.concatenate(scat_y) if scat_y else np.array([])

    # Linear regression: y = a + b * x  (cache size linear, per user spec).
    slope = intercept = r2 = float("nan")
    if scat_x.size >= 2:
        slope, intercept = np.polyfit(scat_x, scat_y, 1)
        y_pred = intercept + slope * scat_x
        ss_res = float(np.sum((scat_y - y_pred) ** 2))
        ss_tot = float(np.sum((scat_y - scat_y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # -------------------------------- figure ---------------------------------
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 10))

    # ------- TOP: time-series, cache (left log) + SoC (right linear) ----------
    ax_top_soc = ax_top.twinx()

    for d in data:
        if d["cache_t"].size:
            ax_top.plot(d["cache_t"], d["cache_n"],
                        color=d["color"], lw=1.3, alpha=0.85,
                        label=f"cache: {d['label']}")
        if d["t"].size and np.any(np.isfinite(d["soc"])):
            ax_top_soc.plot(d["t"], d["soc"],
                            color=d["color"], lw=1.1, alpha=0.85,
                            linestyle="--",
                            label=f"SoC: {d['label']}")

    ax_top.set_yscale("log")
    ax_top.set_xlabel("Wall time since cell start (s)")
    ax_top.set_ylabel("KV cache size (cells, log)")
    ax_top_soc.set_ylabel("SoC temperature (°C)")
    ax_top.grid(True, which="both", alpha=0.25)

    # Two legends: solid (cache) on the left, dashed (SoC) on the right.
    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_top_soc.get_legend_handles_labels()
    leg1 = ax_top.legend(h1, l1, loc="upper left",
                         fontsize=8, framealpha=0.9, title="solid = cache")
    ax_top_soc.legend(h2, l2, loc="lower right",
                      fontsize=8, framealpha=0.9, title="dashed = SoC °C")
    ax_top.add_artist(leg1)

    used_cols = {d["soc_col"] for d in data}
    used_summary = " | ".join(sorted(used_cols))
    ax_top.text(
        0.01, 0.02, f"SoC sensor(s): {used_summary}",
        transform=ax_top.transAxes, ha="left", va="bottom", fontsize=7,
        color="0.25",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85),
    )

    # ------- BOTTOM: scatter + linear fit -------
    for d in data:
        if d["t"].size == 0 or d["cache_t"].size == 0:
            continue
        kv_at_sensor = resample_cache_at_times(d["t"], d["cache_t"], d["cache_n"])
        ok = np.isfinite(kv_at_sensor) & np.isfinite(d["soc"]) & (kv_at_sensor > 0)
        if not ok.any():
            continue
        ax_bot.scatter(kv_at_sensor[ok], d["soc"][ok],
                       s=10, alpha=0.35, color=d["color"],
                       edgecolors="none", rasterized=True,
                       label=f"{d['label']}  (n={int(ok.sum()):,})")

    if scat_x.size >= 2 and np.isfinite(slope):
        xfit = np.linspace(float(scat_x.min()), float(scat_x.max()), 200)
        yfit = intercept + slope * xfit
        ax_bot.plot(xfit, yfit, color="black", lw=2.2,
                    label=(f"fit: SoC = {intercept:.2f} + "
                           f"{slope*1e3:.3f}·(cache/1000)   "
                           f"R²={r2:.3f}"))

    ax_bot.set_xlabel("KV cache size (cells)")
    ax_bot.set_ylabel("SoC temperature (°C)")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc="lower right", fontsize=8, framealpha=0.9,
                  markerscale=2)
    ax_bot.set_title(f"Aggregate scatter (N = {scat_x.size:,} samples)")

    r2_str = f"{r2:.3f}" if np.isfinite(r2) else "n/a"
    fig.suptitle(
        f"KV cache size drives SoC surface temperature (R² = {r2_str})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    schema = {
        "figure_name": OUT_PATH.name,
        "file_path": str(OUT_PATH),
        "panels": [
            {
                "panel": "top",
                "kind": "time_series_dual_axis",
                "x": "wall_time_s_since_cell_start",
                "y_left": "kv_cache_size_cells_log",
                "y_right": "soc_temp_C",
                "series": [
                    {"cell": d["cell"], "label": d["label"],
                     "color": d["color"], "wave": d["wave"],
                     "soc_column": d["soc_col"]}
                    for d in data
                ],
            },
            {
                "panel": "bottom",
                "kind": "scatter_with_linear_fit",
                "x": "kv_cache_size_cells",
                "y": "soc_temp_C",
                "n_samples": int(scat_x.size),
                "fit": {
                    "model": "soc_C = intercept + slope * cache_cells",
                    "intercept_C": float(intercept) if np.isfinite(intercept)
                                   else None,
                    "slope_C_per_cell": float(slope) if np.isfinite(slope)
                                        else None,
                    "r_squared": float(r2) if np.isfinite(r2) else None,
                },
            },
        ],
        "sources": {
            "wave4_dir": str(WAVE4_DIR),
            "wave9_dir": str(WAVE9_DIR),
            "soc_primary_column": COL_SOC_PRIMARY,
            "soc_fallback_column": COL_SOC_FALLBACK,
        },
        "relationship": (
            "KV cache size (cells) -> SoC surface temperature (°C). "
            "Aggregate linear regression across Wave-4 long-decode "
            "(vanilla, v1_K512, v1_fa_K512) and Wave-9 (v1_fa2_stack) cells "
            f"yields R² = {r2_str}."
        ),
    }
    with open(SCHEMA_PATH, "w") as fh:
        json.dump(schema, fh, indent=2)

    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.1f} KB)")
    print(f"wrote {SCHEMA_PATH}")
    print(f"  N={scat_x.size}  slope={slope:.6g}  "
          f"intercept={intercept:.3f}  R²={r2:.4f}")
    for d in data:
        print(f"  [{d['cell']}] SoC col = {d['soc_col']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
