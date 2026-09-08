#!/usr/bin/env python3
"""
host_plot_ksweep.py

Generates three paper-ready figures from the K-sweep measurement data:
  1. ksweep_pareto.png            — 3-panel: K vs mean tps, K vs PPL, K vs peak DDR temp
  2. ksweep_trajectories.png      — 2x2 grid of per-iter tps trajectories with DDR overlay
  3. ksweep_2axis_pareto.png      — scatter of mean tps vs PPL, sized by peak DDR, with Pareto frontier

By default the script LOADS metrics from
``phone-logs/wave10_ksweep_*/K{nominal}/iter*/meta.json`` (per-iter tps + ppl)
and ``sensors.csv`` (per-iter DDR temperature), so the figures are reproducible
from the raw data. If no wave10 ksweep directory is present, it falls back to
the FALLBACK_CELLS literal embedded below (kept only as a last resort so the
dissertation figures can still be regenerated if the raw logs are unavailable;
a warning is printed so reviewers know they are looking at frozen data).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHONE_LOGS_ROOT = Path(
    os.environ.get("PHONE_LOGS_ROOT", "/home/mislam22/EndurKV_workspace/phone-logs")
)


# ---------------------------------------------------------------------------
# Fallback K-sweep measurement data (frozen snapshot of on-device runs).
# Used ONLY when phone-logs/wave10_ksweep_*/K* is unavailable. Prefer the
# live-loading path (see load_ksweep_cells) so figures track the raw data.
# ---------------------------------------------------------------------------
FALLBACK_CELLS = [
    {
        "K": 1024,
        "n_iters": 10,
        "iter1_tps": 7.105,
        "last_iter_tps": 5.87,
        "mean_tps": 6.1985,
        "decay_pct": 17.3821,
        "ppl": 1.826802,
        "per_iter_tps": [7.105, 6.793, 6.652, 6.565, 6.095, 5.861, 5.747, 5.642, 5.655, 5.870],
        "per_iter_ddr": [31, 50, 54, 57, 59, 57, 60, 59, 57, 59],
        "evicted_total": 508641,
        "peak_ddr_c": 63.7,
        "peak_cpu_c": 68.7,
        "peak_rss_gb": 14.4956,
        "swap_mb": 119.18,
        "watchdog_tier1_count": 415,
        "watchdog_tier2_count": 412,
        "watchdog_tier3_count": 0,
    },
    {
        "K": 512,
        "n_iters": 10,
        "iter1_tps": 7.382,
        "last_iter_tps": 4.601,
        "mean_tps": 6.089,
        "decay_pct": 37.6727,
        "ppl": 2.168611,
        "per_iter_tps": [7.382, 6.723, 6.418, 6.113, 6.399, 6.347, 6.170, 4.569, 6.168, 4.601],
        "per_iter_ddr": [37, 54, 57, 58, 58, 57, 58, 59, 54, 59],
        "evicted_total": 1212705,
        "peak_ddr_c": 64.1,
        "peak_cpu_c": 66.8,
        "peak_rss_gb": 13.681,
        "swap_mb": 0,
        "watchdog_tier1_count": 0,
        "watchdog_tier2_count": 0,
        "watchdog_tier3_count": 0,
    },
    {
        "K": 384,
        "n_iters": 12,
        "iter1_tps": 7.759,
        "last_iter_tps": 6.599,
        "mean_tps": 7.0521,
        "decay_pct": 14.9504,
        "ppl": 2.12272,
        "per_iter_tps": [7.759, 7.343, 7.383, 7.389, 7.223, 6.958, 6.916, 6.799, 6.859, 6.750, 6.647, 6.599],
        "per_iter_ddr": [37, 52, 55, 57, 58, 59, 57, 58, 58, 57, 59, 59],
        "evicted_total": 1430516,
        "peak_ddr_c": 63.3,
        "peak_cpu_c": 67.9,
        "peak_rss_gb": 14.4125,
        "swap_mb": 21.48,
        "watchdog_tier1_count": 260,
        "watchdog_tier2_count": 255,
        "watchdog_tier3_count": 0,
    },
    {
        "K": 256,
        "n_iters": 12,
        "iter1_tps": 8.004,
        "last_iter_tps": 6.805,
        "mean_tps": 7.1735,
        "decay_pct": 14.98,
        "ppl": 2.094768,
        "per_iter_tps": [8.004, 7.547, 7.512, 7.471, 7.270, 7.076, 6.862, 6.913, 6.845, 6.861, 6.916, 6.805],
        "per_iter_ddr": [35, 52, 56, 58, 59, 59, 59, 59, 59, 59, 59, 57],
        "evicted_total": 1656897,
        "peak_ddr_c": 64.1,
        "peak_cpu_c": 67.2,
        "peak_rss_gb": 14.5346,
        "swap_mb": 25.25,
        "watchdog_tier1_count": 304,
        "watchdog_tier2_count": 299,
        "watchdog_tier3_count": 0,
    },
]


OUT_DIR = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures/thermal_plots")


# ---------------------------------------------------------------------------
# Live data loader — reads phone-logs/wave10_ksweep_*/K*/iter*/meta.json and
# phone-logs/wave10_ksweep_*/K*/sensors.csv into the same Cells shape as
# FALLBACK_CELLS. This is the preferred path; FALLBACK_CELLS is a last resort.
# ---------------------------------------------------------------------------
def _read_meta(meta_path: Path) -> Dict[str, Any]:
    try:
        with meta_path.open("r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _read_sensor_ddr(sensors_path: Path) -> List[float]:
    """Returns sequence of peak DDR temperatures in °C, one per ~second tick.

    Used only to overlay a coarse DDR trace on per-iter trajectories.
    """
    if not sensors_path.is_file():
        return []
    out: List[float] = []
    try:
        with sensors_path.open("r") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                val = row.get("ddr_temp_mc") or row.get("ddr_temp_c")
                if val is None:
                    continue
                try:
                    fv = float(val)
                except (TypeError, ValueError):
                    continue
                # Convert milli-degrees -> degrees if needed.
                if fv > 200:
                    fv = fv / 1000.0
                out.append(fv)
    except OSError:
        return []
    return out


def _peak_ddr_c(sensors_path: Path) -> Optional[float]:
    trace = _read_sensor_ddr(sensors_path)
    return max(trace) if trace else None


def load_ksweep_cells(phone_logs_root: Path) -> List[Dict[str, Any]]:
    """Discover wave10 K-sweep cells under phone-logs and aggregate per-K
    metrics from iter*/meta.json + sensors.csv. Returns [] when no run dir is
    available so the caller can fall back to FALLBACK_CELLS.
    """
    run_dirs = sorted(glob.glob(str(phone_logs_root / "wave10_ksweep_*")))
    if not run_dirs:
        return []
    run_dir = Path(run_dirs[-1])  # use the most recent run

    cells: List[Dict[str, Any]] = []
    for kdir in sorted(run_dir.glob("K*"), key=lambda p: p.name):
        name = kdir.name
        if not name.startswith("K"):
            continue
        try:
            K = int(name[1:])
        except ValueError:
            continue
        iters = sorted(kdir.glob("iter*"))
        per_iter_tps: List[float] = []
        ppl_val: Optional[float] = None
        evicted_total = 0
        for it in iters:
            m = _read_meta(it / "meta.json")
            if not m:
                continue
            tps = m.get("decode_tps") or m.get("mean_tps")
            try:
                if tps is not None:
                    per_iter_tps.append(float(tps))
            except (TypeError, ValueError):
                pass
            if ppl_val is None:
                p = m.get("perplexity")
                if p is not None:
                    try:
                        ppl_val = float(p)
                    except (TypeError, ValueError):
                        pass
            ev = m.get("evicted_total_decode") or m.get("evicted_total")
            if ev:
                try:
                    evicted_total += int(ev)
                except (TypeError, ValueError):
                    pass

        if not per_iter_tps:
            print(f"[warn] {kdir} — no per-iter tps in meta.json; skipping", file=sys.stderr)
            continue

        peak_ddr = _peak_ddr_c(kdir / "sensors.csv")
        if peak_ddr is None:
            print(f"[warn] {kdir} — no DDR trace; using 0", file=sys.stderr)
            peak_ddr = 0.0

        ddr_trace = _read_sensor_ddr(kdir / "sensors.csv")
        # Down-sample DDR trace to one sample per iteration (uniform stride).
        if ddr_trace and per_iter_tps:
            step = max(1, len(ddr_trace) // len(per_iter_tps))
            per_iter_ddr = [ddr_trace[min(len(ddr_trace) - 1, i * step)]
                            for i in range(len(per_iter_tps))]
        else:
            per_iter_ddr = [peak_ddr] * len(per_iter_tps)

        mean_tps = sum(per_iter_tps) / len(per_iter_tps)
        decay_pct = ((per_iter_tps[0] - per_iter_tps[-1]) / per_iter_tps[0] * 100.0
                     if per_iter_tps[0] > 0 else 0.0)
        cells.append({
            "K": K,
            "n_iters": len(per_iter_tps),
            "iter1_tps": per_iter_tps[0],
            "last_iter_tps": per_iter_tps[-1],
            "mean_tps": mean_tps,
            "decay_pct": decay_pct,
            "ppl": ppl_val if ppl_val is not None else float("nan"),
            "per_iter_tps": per_iter_tps,
            "per_iter_ddr": per_iter_ddr,
            "evicted_total": evicted_total,
            "peak_ddr_c": peak_ddr,
            "peak_cpu_c": 0.0,   # not aggregated here; thermal_full has it
            "peak_rss_gb": 0.0,
            "swap_mb": 0.0,
            "watchdog_tier1_count": 0,
            "watchdog_tier2_count": 0,
            "watchdog_tier3_count": 0,
            "source_dir": str(kdir),
        })

    return cells


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------
def pareto_front_max_min(points):
    """
    Returns indices of Pareto-optimal points where we want to MAXIMIZE the
    first coordinate (mean_tps) and MINIMIZE the second (PPL).

    A point i is dominated iff there exists j such that
        mean_tps[j] >= mean_tps[i]  AND  ppl[j] <= ppl[i]
    with at least one strict inequality.
    """
    nd_idx = []
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            if (xj >= xi and yj <= yi) and (xj > xi or yj < yi):
                dominated = True
                break
        if not dominated:
            nd_idx.append(i)
    return nd_idx


def pareto_front_3d(points):
    """
    3D Pareto: maximize mean_tps, minimize PPL, minimize peak_DDR_C.
    points: list of (mean_tps, ppl, peak_ddr).
    """
    nd_idx = []
    for i, (xi, yi, zi) in enumerate(points):
        dominated = False
        for j, (xj, yj, zj) in enumerate(points):
            if i == j:
                continue
            if (xj >= xi and yj <= yi and zj <= zi) and (xj > xi or yj < yi or zj < zi):
                dominated = True
                break
        if not dominated:
            nd_idx.append(i)
    return nd_idx


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_three_panel_pareto(cells, out_path: Path):
    cells_sorted = sorted(cells, key=lambda c: c["K"])
    Ks = np.array([c["K"] for c in cells_sorted])
    mean_tps = np.array([c["mean_tps"] for c in cells_sorted])
    ppl = np.array([c["ppl"] for c in cells_sorted])
    peak_ddr = np.array([c["peak_ddr_c"] for c in cells_sorted])

    nd_3d = set(pareto_front_3d(list(zip(mean_tps, ppl, peak_ddr))))

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True)

    panels = [
        (axes[0], mean_tps, "Mean decode tps (tok/s)", "tab:blue", False),
        (axes[1], ppl, "Perplexity (PPL)", "tab:orange", True),
        (axes[2], peak_ddr, "Peak DDR temperature (C)", "tab:red", True),
    ]

    for ax, ys, ylabel, color, lower_is_better in panels:
        ax.plot(Ks, ys, "-o", color=color, linewidth=2.0, markersize=8, label=ylabel)
        # Overlay Pareto-optimal points
        for i, k in enumerate(Ks):
            if i in nd_3d:
                ax.scatter([k], [ys[i]], s=180, facecolors="none",
                           edgecolors="green", linewidths=2.2, zorder=5,
                           label="Pareto-optimal" if i == list(nd_3d)[0] else None)
            ax.annotate(f"K={k}", (k, ys[i]),
                        textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.35)
        arrow = "(lower is better)" if lower_is_better else "(higher is better)"
        ax.set_title(f"{ylabel} vs K  {arrow}", fontsize=10)

    axes[-1].set_xlabel("Top-K (log scale)")
    fig.suptitle("EndurKV K-sweep: throughput, quality, and thermal cost",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def plot_trajectories(cells, out_path: Path):
    cells_sorted = sorted(cells, key=lambda c: c["K"])
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=False)
    axes = axes.flatten()

    for ax, c in zip(axes, cells_sorted):
        iters = np.arange(1, c["n_iters"] + 1)
        tps = np.array(c["per_iter_tps"])
        ddr = np.array(c["per_iter_ddr"])

        l1, = ax.plot(iters, tps, "-o", color="tab:blue",
                      linewidth=2.0, markersize=6, label="tps (tok/s)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("tps (tok/s)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(True, alpha=0.35)

        ax2 = ax.twinx()
        l2, = ax2.plot(iters, ddr, "--s", color="tab:red",
                       linewidth=1.6, markersize=5, alpha=0.85, label="DDR temp (C)")
        ax2.set_ylabel("DDR temp (C)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        ax.set_title(
            f"K={c['K']}  | mean tps={c['mean_tps']:.2f}  "
            f"PPL={c['ppl']:.3f}  decay={c['decay_pct']:.1f}%",
            fontsize=10,
        )
        ax.legend(handles=[l1, l2], loc="lower left", fontsize=8)

    fig.suptitle("EndurKV K-sweep: per-iteration throughput trajectories with DDR overlay",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def plot_2axis_pareto(cells, out_path: Path):
    cells_sorted = sorted(cells, key=lambda c: c["K"])
    mean_tps = np.array([c["mean_tps"] for c in cells_sorted])
    ppl = np.array([c["ppl"] for c in cells_sorted])
    peak_ddr = np.array([c["peak_ddr_c"] for c in cells_sorted])
    Ks = [c["K"] for c in cells_sorted]

    # Pareto frontier in (max mean_tps, min PPL)
    nd_idx = pareto_front_max_min(list(zip(mean_tps, ppl)))

    # Bubble sizes scale with peak DDR temperature.
    sizes = (peak_ddr - peak_ddr.min() + 1.0) ** 2 * 18 + 240

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    sc = ax.scatter(mean_tps, ppl, s=sizes, c=peak_ddr,
                    cmap="autumn_r", edgecolors="black",
                    linewidths=1.2, alpha=0.92, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Peak DDR temp (C)")

    for i, k in enumerate(Ks):
        ax.annotate(f"K={k}",
                    (mean_tps[i], ppl[i]),
                    textcoords="offset points",
                    xytext=(12, 8),
                    fontsize=10, fontweight="bold")

    # Pareto frontier line (sorted by mean_tps ascending)
    if len(nd_idx) >= 2:
        nd_sorted = sorted(nd_idx, key=lambda i: mean_tps[i])
        ax.plot(mean_tps[nd_sorted], ppl[nd_sorted],
                "--", color="green", linewidth=2.0,
                label="Pareto frontier", zorder=2)
    elif len(nd_idx) == 1:
        i = nd_idx[0]
        ax.scatter([mean_tps[i]], [ppl[i]], s=420, facecolors="none",
                   edgecolors="green", linewidths=2.5, label="Pareto-optimal")

    ax.set_xlabel("Mean decode throughput (tok/s)  (higher is better)")
    ax.set_ylabel("Perplexity (PPL)  (lower is better)")
    ax.set_title("EndurKV K-sweep: quality vs throughput Pareto plot\n"
                 "(bubble area / color = peak DDR temperature)",
                 fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer live data from phone-logs; fall back to the frozen FALLBACK_CELLS
    # if no wave10_ksweep_* directory is available locally.
    CELLS = load_ksweep_cells(PHONE_LOGS_ROOT)
    if CELLS:
        print(f"[ok] loaded {len(CELLS)} K-sweep cells from {PHONE_LOGS_ROOT}")
    else:
        print(
            "[warn] no wave10_ksweep_* run found under "
            f"{PHONE_LOGS_ROOT}; using FALLBACK_CELLS (frozen dissertation data)",
            file=sys.stderr,
        )
        CELLS = FALLBACK_CELLS

    plot_three_panel_pareto(CELLS, OUT_DIR / "ksweep_pareto.png")
    plot_trajectories(CELLS, OUT_DIR / "ksweep_trajectories.png")
    plot_2axis_pareto(CELLS, OUT_DIR / "ksweep_2axis_pareto.png")

    # Echo Pareto verdict for the dissertation log
    cells_sorted = sorted(CELLS, key=lambda c: c["K"])
    mean_tps = [c["mean_tps"] for c in cells_sorted]
    ppl = [c["ppl"] for c in cells_sorted]
    peak_ddr = [c["peak_ddr_c"] for c in cells_sorted]
    Ks = [c["K"] for c in cells_sorted]

    nd_2d = pareto_front_max_min(list(zip(mean_tps, ppl)))
    nd_3d = pareto_front_3d(list(zip(mean_tps, ppl, peak_ddr)))

    print("\n=== Pareto summary ===")
    print(f"K values         : {Ks}")
    print(f"mean_tps         : {mean_tps}")
    print(f"PPL              : {ppl}")
    print(f"peak_DDR_C       : {peak_ddr}")
    print(f"2D non-dominated : {[Ks[i] for i in nd_2d]}  (tps vs PPL)")
    print(f"3D non-dominated : {[Ks[i] for i in nd_3d]}  (tps vs PPL vs DDR)")


if __name__ == "__main__":
    main()
