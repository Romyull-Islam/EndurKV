#!/usr/bin/env python3
"""
Wave-4 (Phi-3 long-decode) smoking-gun plot:
    cache growth -> DDR temp -> CPU temp time-series, three eviction variants.

For each cell (vanilla, v1_K512, v1_fa_K512):
  - sensors.csv      : monotonic_s, ddr_temp_mc, cpullc-0-0_temp_mc, cpu6_freq_hz
  - stress.csv       : per-iter t_elapsed_s, peak_kv_cells
  - iter*/steps.csv  : per-step wall_us, n_kv_cells

Output: figures/relationship_plots/01_cache_vs_temp_timeseries.png

Uses only stdlib + numpy + matplotlib (no pandas) to match the rest of the repo.
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

WAVE4_DIR = Path(
    "/home/mislam22/EndurKV_workspace/phone-logs/wave4_longdecode_1780750084"
)
OUT_PATH = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/"
    "01_cache_vs_temp_timeseries.png"
)

VARIANTS = [
    ("vanilla",     "vanilla",       "#d62728"),  # red
    ("v1_K512",     "v1 (K=512)",    "#1f77b4"),  # blue
    ("v1_fa_K512",  "v1_fa (K=512)", "#2ca02c"),  # green
]

DDR_THROTTLE_C = 65.0  # kernel DDR throttle trigger


# ---------------------------- helpers -----------------------------------------

def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_sensors(cell_dir: Path):
    """Return arrays (t_s, ddr_c, cpu_c, cpu6_mhz) with t_s = monotonic - t0."""
    t, ddr, cpu, cpu6 = [], [], [], []
    with open(cell_dir / "sensors.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t.append(_to_float(row.get("monotonic_s")))
            ddr.append(_to_float(row.get("ddr_temp_mc")) / 1000.0)
            cpu.append(_to_float(row.get("cpullc-0-0_temp_mc")) / 1000.0)
            cpu6.append(_to_float(row.get("cpu6_freq_hz")) / 1000.0)
    t   = np.asarray(t,   dtype=float)
    ddr = np.asarray(ddr, dtype=float)
    cpu = np.asarray(cpu, dtype=float)
    cpu6 = np.asarray(cpu6, dtype=float)
    if t.size:
        t = t - t[0]
    return t, ddr, cpu, cpu6


def load_cache_series(cell_dir: Path):
    """Per-step (t_s, n_kv) concatenated across iters, aligned to run-wall-time.

    steps.csv wall_us = microseconds since that iter's prefill start.
    stress.csv t_elapsed_s gives the iter's start relative to run start, which
    is the same reference as sensors' (monotonic_s - monotonic_s[0]).
    """
    # Read stress.csv to get per-iter t_elapsed_s
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
    """Mean of y over the last tail_frac of the run."""
    if t.size == 0:
        return float("nan")
    cutoff = t[-1] * (1.0 - tail_frac)
    mask = (t >= cutoff) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    return float(np.nanmean(y[mask]))


# ------------------------------- plot -----------------------------------------

def main() -> int:
    if not WAVE4_DIR.exists():
        print(f"ERROR: missing wave4 dir: {WAVE4_DIR}", file=sys.stderr)
        return 1
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    for cell, label, color in VARIANTS:
        cell_dir = WAVE4_DIR / cell
        if not cell_dir.exists():
            print(f"WARN: missing cell {cell_dir}", file=sys.stderr)
            continue
        t_s, ddr_c, cpu_c, cpu6_mhz = load_sensors(cell_dir)
        cache_t, cache_n = load_cache_series(cell_dir)
        data[cell] = dict(
            label=label, color=color,
            t=t_s, ddr=ddr_c, cpu=cpu_c, cpu6=cpu6_mhz,
            cache_t=cache_t, cache_n=cache_n,
        )

    # Steady-state DDR means (last 60% of each cell's sensor trace).
    ddr_means = {
        cell: steady_state_mean(d["t"], d["ddr"]) for cell, d in data.items()
    }
    swing_c = float("nan")
    if "vanilla" in ddr_means and "v1_K512" in ddr_means:
        swing_c = ddr_means["vanilla"] - ddr_means["v1_K512"]

    # ---------- figure: 3 stacked panels, shared x-axis ----------
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9.5), sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    ax_cache, ax_ddr, ax_cpu = axes

    for cell, d in data.items():
        c, lbl = d["color"], d["label"]

        if d["cache_t"].size:
            ax_cache.plot(d["cache_t"], d["cache_n"],
                          color=c, lw=1.3, alpha=0.9, label=lbl)

        if d["t"].size:
            ax_ddr.plot(d["t"], d["ddr"], color=c, lw=1.2,
                        alpha=0.9, label=lbl)
            ax_cpu.plot(d["t"], d["cpu"], color=c, lw=1.2,
                        alpha=0.9, label=lbl)

    # Row 1 cosmetics
    ax_cache.set_ylabel("KV cache (cells)")
    if np.isfinite(swing_c):
        title = (
            "Cache growth drives DDR temperature: same workload, "
            f"different eviction → {abs(swing_c):.1f}°C swing"
        )
    else:
        title = (
            "Cache growth drives DDR temperature: same workload, "
            "different eviction"
        )
    ax_cache.set_title(title)
    ax_cache.grid(True, alpha=0.3)
    ax_cache.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # Row 2 cosmetics
    ax_ddr.axhline(DDR_THROTTLE_C, color="black", lw=1.0,
                   linestyle="--", alpha=0.7)
    ax_ddr.text(
        0.99, DDR_THROTTLE_C + 0.4,
        f"DDR throttle ({DDR_THROTTLE_C:.0f}°C)",
        transform=ax_ddr.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=8, color="black", alpha=0.8,
    )
    ax_ddr.set_ylabel("DDR temp (°C)")
    ax_ddr.grid(True, alpha=0.3)
    ax_ddr.legend(loc="lower right", framealpha=0.9, fontsize=9)

    if ddr_means:
        annot = "  |  ".join(
            f"{data[cell]['label']}: {v:.1f}°C"
            for cell, v in ddr_means.items() if np.isfinite(v)
        )
        ax_ddr.text(
            0.01, 0.97, f"steady-state DDR mean: {annot}",
            transform=ax_ddr.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                      alpha=0.9),
        )

    # Row 3 cosmetics
    ax_cpu.set_ylabel("CPU LLC-0-0 temp (°C)")
    ax_cpu.set_xlabel("Wall time since cell start (s)")
    ax_cpu.grid(True, alpha=0.3)
    ax_cpu.legend(loc="lower right", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.1f} KB)")
    for cell, v in ddr_means.items():
        print(f"  DDR steady-state mean[{cell}] = {v:.2f} C")
    if np.isfinite(swing_c):
        print(f"  vanilla - v1_K512 DDR swing = {swing_c:+.2f} C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
