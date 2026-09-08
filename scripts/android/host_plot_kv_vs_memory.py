#!/usr/bin/env python3
"""
PLOT 12: KV cache growth vs DDR (memory) temperature.

The DDR temperature sensor (ddr_temp_mc, millidegrees C) is the ONLY
DRAM-side temperature in the sensor file, so it is the cleanest signal
for "memory got hotter because the working set grew".

Inputs (K-sweep / long-decode cells):
  - Wave-4  : phone-logs/wave4_longdecode_1780750084/{vanilla, v1_K512, v1_fa_K512}
  - Wave-9  : phone-logs/wave9_v1fa2_stack_1780796320/v1_fa2_stack
  - Wave-10 : phone-logs/wave10_ksweep_1780815847/{K256, K384, K1024}

Each cell provides:
  - sensors.csv        : wall_clock_s / monotonic_s + ddr_temp_mc
  - stress.csv         : per-iter t_elapsed_s (relative to that cell's sensor t0)
  - iter*/steps.csv    : per-step wall_us (us since iter's prefill start) + n_kv_cells

Output: figures/relationship_plots/12_kv_cache_vs_memory_temp.png

Two-panel figure:
  Top    : time-series of cache size (left axis) + DDR temp (right axis),
           one line-pair per policy. Horizontal dashed line at 65 C
           marking the kernel DDR throttle trip.
  Bottom : scatter of cache_size vs DDR_temp_C across ALL samples from
           ALL cells, with linear regression, R^2 in the title, and a
           "thermal danger zone" annotation for the DDR > 60 C cluster.

stdlib + numpy + matplotlib only (consistent with the rest of the repo;
the K-sweep version uses pandas, but we keep this script lean).
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WAVE4_DIR = Path(
    "/home/mislam22/EndurKV_workspace/phone-logs/wave4_longdecode_1780750084"
)
WAVE9_DIR = Path(
    "/home/mislam22/EndurKV_workspace/phone-logs/wave9_v1fa2_stack_1780796320"
)
WAVE10_DIR = Path(
    "/home/mislam22/EndurKV_workspace/phone-logs/wave10_ksweep_1780815847"
)

OUT_PATH = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/"
    "12_kv_cache_vs_memory_temp.png"
)
SCHEMA_PATH = OUT_PATH.with_suffix(".schema.json")

DDR_THROTTLE_C = 65.0   # kernel DDR throttle trip
DDR_DANGER_C   = 60.0   # "thermal danger zone" lower bound

# (wave_label, cell_dir, display_label, color)
CELLS = [
    ("wave4",  WAVE4_DIR  / "vanilla",       "W4 vanilla",       "#d62728"),  # red
    ("wave4",  WAVE4_DIR  / "v1_K512",       "W4 v1 K=512",      "#1f77b4"),  # blue
    ("wave4",  WAVE4_DIR  / "v1_fa_K512",    "W4 v1_fa K=512",   "#2ca02c"),  # green
    ("wave9",  WAVE9_DIR  / "v1_fa2_stack",  "W9 v1_fa2_stack",  "#9467bd"),  # purple
    ("wave10", WAVE10_DIR / "K256",          "W10 K=256",        "#17becf"),  # cyan
    ("wave10", WAVE10_DIR / "K384",          "W10 K=384",        "#bcbd22"),  # olive
    ("wave10", WAVE10_DIR / "K1024",         "W10 K=1024",       "#e377c2"),  # pink
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_sensors(cell_dir: Path):
    """Return (t_s, ddr_c) where t_s = monotonic_s - monotonic_s[0]."""
    t, ddr = [], []
    path = cell_dir / "sensors.csv"
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t.append(_to_float(row.get("monotonic_s")))
            ddr.append(_to_float(row.get("ddr_temp_mc")) / 1000.0)
    t   = np.asarray(t,   dtype=float)
    ddr = np.asarray(ddr, dtype=float)
    if t.size:
        t = t - t[0]
    return t, ddr


def load_cache_series(cell_dir: Path):
    """Per-step (t_s, n_kv) concatenated across iters, in cell-wall time.

    steps.csv wall_us is microseconds since that iter's prefill start;
    stress.csv t_elapsed_s gives that iter's offset relative to the cell's
    sensor t0 (the first sensors monotonic_s sample). So a step's absolute
    cell-wall time is t_iter + wall_us/1e6.
    """
    iter_start = {}
    with open(cell_dir / "stress.csv", newline="") as fh:
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


def steady_state_mean(t, y, tail_frac=0.6):
    if t.size == 0:
        return float("nan")
    cutoff = t[-1] * (1.0 - tail_frac)
    mask = (t >= cutoff) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    return float(np.nanmean(y[mask]))


def join_cache_to_sensors(cache_t, cache_n, sens_t, sens_ddr):
    """For each cache sample, look up the nearest-in-time DDR temp.

    Returns matched arrays (n_kv, ddr_c) of equal length.
    """
    if cache_t.size == 0 or sens_t.size == 0:
        return np.array([]), np.array([])
    # sensors are uniformly sampled and sorted; cache samples are sorted.
    idx = np.searchsorted(sens_t, cache_t)
    idx = np.clip(idx, 0, sens_t.size - 1)
    # nudge left when that's strictly closer in time
    left_idx = np.clip(idx - 1, 0, sens_t.size - 1)
    use_left = (idx > 0) & (
        np.abs(sens_t[left_idx] - cache_t) < np.abs(sens_t[idx] - cache_t)
    )
    idx[use_left] = left_idx[use_left]
    ddr_at_cache = sens_ddr[idx]
    mask = np.isfinite(cache_n) & np.isfinite(ddr_at_cache)
    return cache_n[mask], ddr_at_cache[mask]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cells_data = []
    missing = []
    for wave, cell_dir, label, color in CELLS:
        if not cell_dir.exists():
            missing.append(str(cell_dir))
            continue
        sens_t, sens_ddr = load_sensors(cell_dir)
        cache_t, cache_n = load_cache_series(cell_dir)
        kv_matched, ddr_matched = join_cache_to_sensors(
            cache_t, cache_n, sens_t, sens_ddr
        )
        cells_data.append(dict(
            wave=wave, label=label, color=color, src=str(cell_dir),
            sens_t=sens_t, sens_ddr=sens_ddr,
            cache_t=cache_t, cache_n=cache_n,
            kv_matched=kv_matched, ddr_matched=ddr_matched,
            n_sens=int(sens_t.size), n_cache=int(cache_t.size),
            n_matched=int(kv_matched.size),
            ddr_steady_mean=steady_state_mean(sens_t, sens_ddr),
            peak_kv=float(np.nanmax(cache_n)) if cache_n.size else float("nan"),
            peak_ddr=float(np.nanmax(sens_ddr)) if sens_ddr.size else float("nan"),
        ))

    if missing:
        for m in missing:
            print(f"WARN: missing cell dir: {m}", file=sys.stderr)

    if not cells_data:
        print("ERROR: no cell data loaded", file=sys.stderr)
        return 1

    # Wave-4 swing used in the title: peak-DDR delta between the hottest
    # Wave-4 cell and the coolest (= K=512 budgeted v1). On this device that
    # peak-vs-peak gap is the headline 8.5 C number (62.9 C vanilla -
    # 54.4 C v1_K512); steady-state means show a smaller ~5.6 C swing.
    swing_c = float("nan")
    steady_swing_c = float("nan")
    by_label = {c["label"]: c for c in cells_data}
    wave4_cells = [c for c in cells_data if c["wave"] == "wave4"]
    if len(wave4_cells) >= 2:
        peaks  = [c["peak_ddr"]        for c in wave4_cells if np.isfinite(c["peak_ddr"])]
        steady = [c["ddr_steady_mean"] for c in wave4_cells
                  if np.isfinite(c["ddr_steady_mean"])]
        if peaks:
            swing_c = max(peaks) - min(peaks)
        if steady:
            steady_swing_c = max(steady) - min(steady)

    # ----- Pool ALL matched samples for the global scatter / regression -----
    all_kv  = np.concatenate([c["kv_matched"]  for c in cells_data]) \
        if any(c["kv_matched"].size for c in cells_data) else np.array([])
    all_ddr = np.concatenate([c["ddr_matched"] for c in cells_data]) \
        if any(c["ddr_matched"].size for c in cells_data) else np.array([])

    slope = intercept = r2 = float("nan")
    if all_kv.size >= 2:
        coeffs = np.polyfit(all_kv, all_ddr, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        pred = np.polyval(coeffs, all_kv)
        ss_res = float(np.sum((all_ddr - pred) ** 2))
        ss_tot = float(np.sum((all_ddr - np.mean(all_ddr)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # ---------------------------------------------------------------------
    # Figure: 2 stacked panels
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(12.0, 10.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.28)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # Title
    if np.isfinite(r2) and np.isfinite(swing_c):
        title = (
            f"KV cache size is the binding lever for DRAM temperature "
            f"(R² = {r2:.2f}, {abs(swing_c):.1f}°C swing Wave-4)"
        )
    elif np.isfinite(r2):
        title = (
            f"KV cache size is the binding lever for DRAM temperature "
            f"(R² = {r2:.2f})"
        )
    else:
        title = "KV cache size is the binding lever for DRAM temperature"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    # -------------------------------------------------------------------
    # TOP PANEL: time-series, cache + DDR per cell
    # -------------------------------------------------------------------
    ax_top.set_title(
        "Cache size and DDR temperature evolve together "
        "(solid = KV cache; dashed = DDR temp)",
        fontsize=11,
    )
    ax_top_r = ax_top.twinx()

    for c in cells_data:
        col = c["color"]
        if c["cache_t"].size:
            ax_top.plot(
                c["cache_t"], c["cache_n"],
                color=col, lw=1.3, alpha=0.95, label=c["label"],
            )
        if c["sens_t"].size:
            ax_top_r.plot(
                c["sens_t"], c["sens_ddr"],
                color=col, lw=1.1, alpha=0.75, linestyle="--",
            )

    ax_top_r.axhline(
        DDR_THROTTLE_C, color="black", lw=1.1, linestyle=":",
        alpha=0.85,
    )
    ax_top_r.text(
        0.995, DDR_THROTTLE_C + 0.25,
        f"DDR throttle trip ({DDR_THROTTLE_C:.0f}°C)",
        transform=ax_top_r.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=8, color="black",
    )

    ax_top.set_xlabel("Wall time since cell start (s)")
    ax_top.set_ylabel("KV cache (cells)")
    ax_top_r.set_ylabel("DDR temperature (°C)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(
        loc="upper left", ncol=2, fontsize=8, framealpha=0.9,
        title="cache traces", title_fontsize=8,
    )

    # -------------------------------------------------------------------
    # BOTTOM PANEL: scatter + regression
    # -------------------------------------------------------------------
    ax_bot.set_title(
        "Scatter: every (n_kv_cells, DDR °C) sample across "
        "Wave-4 / Wave-9 / Wave-10 cells",
        fontsize=11,
    )

    for c in cells_data:
        if c["kv_matched"].size == 0:
            continue
        ax_bot.scatter(
            c["kv_matched"], c["ddr_matched"],
            s=8, alpha=0.45, color=c["color"], label=c["label"],
            edgecolors="none",
        )

    if np.isfinite(slope):
        x_line = np.linspace(float(all_kv.min()), float(all_kv.max()), 200)
        y_line = slope * x_line + intercept
        ax_bot.plot(
            x_line, y_line,
            color="black", lw=1.8, linestyle="-",
            label=(
                f"OLS fit: DDR = {slope:.4f}·n_kv + "
                f"{intercept:.1f}°C   (R² = {r2:.2f})"
            ),
        )

    # 65 C kernel throttle reference line
    ax_bot.axhline(
        DDR_THROTTLE_C, color="black", lw=1.1, linestyle=":", alpha=0.85,
    )
    ax_bot.text(
        0.005, DDR_THROTTLE_C + 0.25,
        f"DDR throttle trip ({DDR_THROTTLE_C:.0f}°C)",
        transform=ax_bot.get_yaxis_transform(),
        ha="left", va="bottom", fontsize=8, color="black",
    )

    # Thermal danger-zone shading + annotation (DDR > 60 C cluster)
    if all_ddr.size:
        ax_bot.axhspan(
            DDR_DANGER_C, max(float(all_ddr.max()) + 1.5, DDR_THROTTLE_C + 1.5),
            color="red", alpha=0.07, zorder=0,
        )
        danger_mask = all_ddr > DDR_DANGER_C
        if danger_mask.any():
            x_mid = float(np.median(all_kv[danger_mask]))
            y_top = float(np.max(all_ddr[danger_mask]))
            ax_bot.annotate(
                "thermal danger zone\n(DDR > 60°C)",
                xy=(x_mid, y_top),
                xytext=(x_mid, y_top + 1.2),
                ha="center", va="bottom",
                fontsize=10, color="#8b0000", fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white", ec="#8b0000", alpha=0.85,
                ),
                arrowprops=dict(
                    arrowstyle="->", color="#8b0000", lw=1.0,
                ),
            )

    ax_bot.set_xlabel("KV cache size (n_kv_cells)")
    ax_bot.set_ylabel("DDR temperature (°C)")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=2)

    # Pad y so the danger-zone annotation never gets clipped
    if all_ddr.size:
        y_lo = min(float(all_ddr.min()) - 1.0, 40.0)
        y_hi = max(float(all_ddr.max()) + 3.5, DDR_THROTTLE_C + 2.0)
        ax_bot.set_ylim(y_lo, y_hi)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- schema sidecar ----------------
    import json
    schema = {
        "kind": "ANALYSIS_SCHEMA",
        "name": OUT_PATH.stem,
        "figure": str(OUT_PATH),
        "ddr_temp_sensor_column": "ddr_temp_mc",
        "ddr_throttle_c": DDR_THROTTLE_C,
        "ddr_danger_c": DDR_DANGER_C,
        "wave_sources": {
            "wave4":  str(WAVE4_DIR),
            "wave9":  str(WAVE9_DIR),
            "wave10": str(WAVE10_DIR),
        },
        "regression": {
            "model": "OLS: DDR_C = slope * n_kv_cells + intercept",
            "slope_c_per_cell": slope,
            "intercept_c": intercept,
            "r2": r2,
            "n_samples": int(all_kv.size),
        },
        "wave4_peak_ddr_swing_c": swing_c,
        "wave4_steady_state_ddr_swing_c": steady_swing_c,
        "per_cell": [
            {
                "wave": c["wave"], "label": c["label"], "src_dir": c["src"],
                "n_sensor_samples": c["n_sens"],
                "n_cache_samples":  c["n_cache"],
                "n_matched_samples": c["n_matched"],
                "ddr_steady_mean_c": c["ddr_steady_mean"],
                "peak_kv_cells": c["peak_kv"],
                "peak_ddr_c":   c["peak_ddr"],
            }
            for c in cells_data
        ],
        "missing_cell_dirs": missing,
    }
    with open(SCHEMA_PATH, "w") as fh:
        json.dump(schema, fh, indent=2)

    # --------------- console summary --------------
    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1024:.1f} KB)")
    print(f"wrote {SCHEMA_PATH}")
    print(f"  global OLS: slope={slope:.4f} C/cell, "
          f"intercept={intercept:.2f} C, R^2={r2:.3f}, n={all_kv.size}")
    if np.isfinite(swing_c):
        print(f"  Wave-4 peak-DDR swing (hottest - coolest cell) = "
              f"{swing_c:+.2f} C")
    if np.isfinite(steady_swing_c):
        print(f"  Wave-4 steady-state DDR swing                  = "
              f"{steady_swing_c:+.2f} C")
    for c in cells_data:
        print(
            f"  [{c['wave']:>6s}] {c['label']:<18s} "
            f"n_sens={c['n_sens']:>5d} n_cache={c['n_cache']:>6d} "
            f"n_matched={c['n_matched']:>6d} "
            f"ddr_steady={c['ddr_steady_mean']:.2f} C  "
            f"peak_kv={c['peak_kv']:.0f}  peak_ddr={c['peak_ddr']:.1f} C"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
