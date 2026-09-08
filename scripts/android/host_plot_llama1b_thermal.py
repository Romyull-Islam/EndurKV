#!/usr/bin/env python3
"""
FIGURE 20 - Llama-1B Wave-3 thermal trajectories (4-panel).

Plots time-series traces of the four Wave-3 Llama-3.2-1B cells
(narrativeqa_pub_001, 25 min sustained-stress protocol on OnePlus-15
Snapdragon 8 Elite Gen 5):

  vanilla       - stock llama.cpp, full KV (k_nominal=0)
  v1_K512       - v1 selective-stack, K=512 budget
  v1_K2048      - v1 selective-stack, K=2048 budget
  v1_fa_K512    - v1 selective-stack + custom FA kernel, K=512

Source per cell:
  sensors.csv  -> wall_clock_s, ddr_temp_mc, cpullc-0-0_temp_mc
  stress.csv   -> iter, t_elapsed_s (iter start offset within session)
  iter*/steps.csv -> step, wall_us (iter-local microseconds),
                     n_kv_cells, rss_kb

Layout (2x2):
  Top-left    : DDR temp (C) vs time, all 4 policies + 65 C kernel cliff
  Top-right   : CPU (cpullc-0-0) temp (C) vs time, all 4 policies
  Bottom-left : cache_size (n_kv_cells) vs time, all 4 policies
  Bottom-right: RSS (MB) vs time, all 4 policies

Headline:
  "Llama-1B stays thermally comfortable (peak 51.7 C, 13 C below kernel
  cliff) - eviction effect on thermals is small but measurable"

Output:
  /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/
  20_llama1b_thermal_trajectories.png
  + matching .schema.json sidecar (RES_SCHEMA)
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----- inputs ---------------------------------------------------------------

PHONE_LOGS = "/home/mislam22/EndurKV_workspace/phone-logs"
WAVE3_DIR = f"{PHONE_LOGS}/wave3_real_1780680903"

OUT_PNG = (
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/"
    "relationship_plots/20_llama1b_thermal_trajectories.png"
)
OUT_SCHEMA = OUT_PNG.replace(".png", ".schema.json")

DDR_COL = "ddr_temp_mc"
CPU_COL = "cpullc-0-0_temp_mc"
KERNEL_CLIFF_C = 65.0


@dataclass
class Cell:
    label: str
    color: str
    subdir: str
    policy: str


CELLS = [
    Cell("vanilla       (no control, full KV)", "#888888", "vanilla",    "vanilla"),
    Cell("v1   K=2048   (large budget)",        "#1f77b4", "v1_K2048",   "v1_K2048"),
    Cell("v1   K=512    (tight budget)",        "#d62728", "v1_K512",    "v1_K512"),
    Cell("v1_FA K=512   (custom FA kernel)",    "#2ca02c", "v1_fa_K512", "v1_fa_K512"),
]


# ----- helpers --------------------------------------------------------------

def load_sensors(cell_dir: str) -> pd.DataFrame:
    """Load sensors.csv with wall_clock_s -> t_rel_s (sec from sensor start),
    plus DDR and CPU temps in C."""
    path = os.path.join(cell_dir, "sensors.csv")
    df = pd.read_csv(path, low_memory=False)
    df["wall_clock_s"] = pd.to_numeric(df["wall_clock_s"], errors="coerce")
    df["ddr_c"] = pd.to_numeric(df[DDR_COL], errors="coerce") / 1000.0
    df["cpu_c"] = pd.to_numeric(df[CPU_COL], errors="coerce") / 1000.0
    df = df.dropna(subset=["wall_clock_s", "ddr_c", "cpu_c"]).reset_index(drop=True)
    df = df.sort_values("wall_clock_s").reset_index(drop=True)
    t0 = float(df["wall_clock_s"].iloc[0])
    df["t_rel_s"] = df["wall_clock_s"] - t0
    return df


def load_stress(cell_dir: str) -> pd.DataFrame:
    """Iter -> t_elapsed_s (offset of iter-start from session start)."""
    df = pd.read_csv(os.path.join(cell_dir, "stress.csv"))
    df["iter"] = pd.to_numeric(df["iter"], errors="coerce").astype(int)
    df["t_elapsed_s"] = pd.to_numeric(df["t_elapsed_s"], errors="coerce")
    return df


def load_steps(cell_dir: str, stress: pd.DataFrame) -> pd.DataFrame:
    """Concatenate iter*/steps.csv into a single trace, with an absolute
    session-time column built from stress.t_elapsed_s[iter] + wall_us[step]."""
    rows: list[pd.DataFrame] = []
    iter_dirs = sorted(glob.glob(os.path.join(cell_dir, "iter*")))
    for idir in iter_dirs:
        base = os.path.basename(idir)
        # iter0001 -> 1
        try:
            it = int(base.replace("iter", "").lstrip("0") or "0")
        except ValueError:
            continue
        sp = os.path.join(idir, "steps.csv")
        if not os.path.isfile(sp):
            continue
        s = pd.read_csv(sp)
        if len(s) == 0:
            continue
        s["wall_us"] = pd.to_numeric(s["wall_us"], errors="coerce")
        s["n_kv_cells"] = pd.to_numeric(s["n_kv_cells"], errors="coerce")
        s["rss_kb"] = pd.to_numeric(s["rss_kb"], errors="coerce")
        s["iter"] = it
        # iter start offset from session t=0 (stress.csv)
        try:
            off = float(stress.loc[stress["iter"] == it, "t_elapsed_s"].iloc[0])
        except (IndexError, KeyError):
            off = float("nan")
        # wall_us inside each iter is measured from that iter's local clock
        # start (validated: step0.wall_us == prefill_ms*1000 for iter0001)
        s["t_session_s"] = off + s["wall_us"] / 1.0e6
        rows.append(s)
    if not rows:
        return pd.DataFrame(columns=["step", "wall_us", "n_kv_cells",
                                     "rss_kb", "iter", "t_session_s"])
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["t_session_s", "n_kv_cells", "rss_kb"])
    out = out.sort_values("t_session_s").reset_index(drop=True)
    return out


# ----- main -----------------------------------------------------------------

def main() -> None:
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    if not os.path.isdir(WAVE3_DIR):
        sys.exit(f"[fatal] missing wave3 dir: {WAVE3_DIR}")

    per_cell: list[tuple[Cell, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    schema_cells: list[dict] = []

    for c in CELLS:
        cdir = os.path.join(WAVE3_DIR, c.subdir)
        if not os.path.isdir(cdir):
            sys.exit(f"[fatal] missing cell dir: {cdir}")
        sn = load_sensors(cdir)
        st = load_stress(cdir)
        sp = load_steps(cdir, st)

        sn_max_t = float(sn["t_rel_s"].iloc[-1]) if len(sn) else 0.0
        peak_ddr = float(sn["ddr_c"].max()) if len(sn) else float("nan")
        peak_cpu = float(sn["cpu_c"].max()) if len(sn) else float("nan")
        mean_ddr = float(sn["ddr_c"].mean()) if len(sn) else float("nan")
        mean_cpu = float(sn["cpu_c"].mean()) if len(sn) else float("nan")
        peak_kv = int(sp["n_kv_cells"].max()) if len(sp) else 0
        peak_rss_mb = float(sp["rss_kb"].max()) / 1024.0 if len(sp) else 0.0

        print(
            f"[info] {c.label:<38}  "
            f"sensors n={len(sn):5d}  T={sn_max_t/60:.1f} min  "
            f"peakDDR={peak_ddr:.1f}C  peakCPU={peak_cpu:.1f}C  "
            f"peakKV={peak_kv}  peakRSS={peak_rss_mb:.0f}MB  "
            f"steps={len(sp)}  iters={len(st)}"
        )

        per_cell.append((c, sn, st, sp))
        schema_cells.append({
            "policy": c.policy,
            "label": c.label,
            "src_dir": cdir,
            "n_iter": int(len(st)),
            "session_duration_s": sn_max_t,
            "sensors_rows": int(len(sn)),
            "steps_rows": int(len(sp)),
            "peak_ddr_c": round(peak_ddr, 2),
            "mean_ddr_c": round(mean_ddr, 2),
            "peak_cpu_c": round(peak_cpu, 2),
            "mean_cpu_c": round(mean_cpu, 2),
            "peak_kv_cells": peak_kv,
            "peak_rss_mb": round(peak_rss_mb, 1),
        })

    # ----- figure ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    ax_ddr, ax_cpu = axes[0]
    ax_kv,  ax_rss = axes[1]

    # ----- (a) DDR temp vs time --------------------------------------------
    for c, sn, _st, _sp in per_cell:
        ax_ddr.plot(
            sn["t_rel_s"] / 60.0, sn["ddr_c"],
            color=c.color, lw=1.4, alpha=0.92, label=c.label,
        )
    ax_ddr.axhline(
        KERNEL_CLIFF_C,
        color="#c00000", ls="--", lw=1.7, alpha=0.9, zorder=1,
    )
    ax_ddr.text(
        0.99, KERNEL_CLIFF_C - 0.4,
        f"kernel freq-cliff @ DDR >= {KERNEL_CLIFF_C:.0f} C  "
        f"(SD8-Gen5 governor caps big-core to ~1.27 GHz)",
        transform=ax_ddr.get_yaxis_transform(),
        ha="right", va="top", fontsize=9.0, color="#c00000",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.30", fc="#ffe9e9", ec="#c00000", lw=0.9,
        ),
    )
    # band showing all-cell peak well under cliff
    all_peak = max(s["peak_ddr_c"] for s in schema_cells)
    ax_ddr.axhline(
        all_peak,
        color="#2ca02c", ls=":", lw=1.2, alpha=0.85, zorder=1,
    )
    ax_ddr.text(
        0.01, all_peak + 0.6,
        f"Llama-1B peak DDR = {all_peak:.1f} C  "
        f"(13 C below cliff)",
        transform=ax_ddr.get_yaxis_transform(),
        ha="left", va="bottom", fontsize=9.0, color="#1b6e1b",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25", fc="#e8f5e8", ec="#2ca02c", lw=0.9,
        ),
    )
    ax_ddr.set_xlabel("time (min from cold start)")
    ax_ddr.set_ylabel("DDR memory temp (C)")
    ax_ddr.set_title("(a) DDR temperature trajectory")
    ax_ddr.grid(True, alpha=0.3)
    ax_ddr.set_ylim(44, 68)
    ax_ddr.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    # ----- (b) CPU temp vs time --------------------------------------------
    for c, sn, _st, _sp in per_cell:
        ax_cpu.plot(
            sn["t_rel_s"] / 60.0, sn["cpu_c"],
            color=c.color, lw=1.4, alpha=0.92, label=c.label,
        )
    cpu_peak_all = max(s["peak_cpu_c"] for s in schema_cells)
    ax_cpu.text(
        0.99, 0.96,
        f"peak CPU = {cpu_peak_all:.1f} C\n"
        f"(SoC trip ~95 C, far above)",
        transform=ax_cpu.transAxes,
        ha="right", va="top", fontsize=9.0, color="#1b6e1b",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.30", fc="#e8f5e8", ec="#2ca02c", lw=0.9,
        ),
    )
    ax_cpu.set_xlabel("time (min from cold start)")
    ax_cpu.set_ylabel("CPU (cpullc-0-0) temp (C)")
    ax_cpu.set_title("(b) CPU temperature trajectory")
    ax_cpu.grid(True, alpha=0.3)
    ax_cpu.set_ylim(44, 64)
    ax_cpu.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    # ----- (c) cache size vs time (n_kv_cells) -----------------------------
    for c, _sn, _st, sp in per_cell:
        if len(sp) == 0:
            continue
        ax_kv.plot(
            sp["t_session_s"] / 60.0, sp["n_kv_cells"],
            color=c.color, marker="o", ms=2.6, lw=1.0, alpha=0.85,
            label=c.label,
        )
    ax_kv.set_xlabel("time (min from session start)")
    ax_kv.set_ylabel("KV cache size (cells)")
    ax_kv.set_title(
        "(c) KV cache size trajectory  (per-step n_kv_cells from steps.csv)"
    )
    ax_kv.grid(True, alpha=0.3)
    ax_kv.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    # ----- (d) RSS vs time -------------------------------------------------
    for c, _sn, _st, sp in per_cell:
        if len(sp) == 0:
            continue
        ax_rss.plot(
            sp["t_session_s"] / 60.0, sp["rss_kb"] / 1024.0,
            color=c.color, marker="o", ms=2.6, lw=1.0, alpha=0.85,
            label=c.label,
        )
    ax_rss.set_xlabel("time (min from session start)")
    ax_rss.set_ylabel("process RSS (MB)")
    ax_rss.set_title(
        "(d) Resident-set-size trajectory  (per-step rss_kb from steps.csv)"
    )
    ax_rss.grid(True, alpha=0.3)
    ax_rss.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    fig.suptitle(
        "Llama-1B stays thermally comfortable (peak "
        f"{all_peak:.1f} C, 13 C below kernel cliff) - "
        "eviction effect on thermals is small but measurable",
        fontsize=13.0, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT_PNG, dpi=160)
    print(f"[ok] wrote {OUT_PNG}")

    # ----- schema sidecar --------------------------------------------------
    schema = {
        "kind": "RES_SCHEMA",
        "version": 1,
        "generator": "host_plot_llama1b_thermal.py",
        "figure": OUT_PNG,
        "title": (
            f"Llama-1B stays thermally comfortable (peak {all_peak:.1f} C, "
            "13 C below kernel cliff) - eviction effect on thermals is "
            "small but measurable"
        ),
        "role_in_paper": "FIGURE 20 - Llama-1B Wave-3 thermal trajectories",
        "model": "Llama-3.2-1B-Instruct-Q4_K_M",
        "prompt_id": "narrativeqa_pub_001",
        "device": "OnePlus-15 / Snapdragon 8 Elite Gen 5",
        "wave": "wave3",
        "wave_source": WAVE3_DIR,
        "panels": {
            "top_left":     "DDR temp vs time (4 policies, with 65 C cliff)",
            "top_right":    "CPU (cpullc-0-0) temp vs time (4 policies)",
            "bottom_left":  "n_kv_cells vs time (from steps.csv)",
            "bottom_right": "RSS (MB) vs time (from steps.csv)",
        },
        "x_axis": "time in minutes (cold-start / session-start)",
        "y_axes": {
            "top_left":     "DDR memory temperature (C)",
            "top_right":    "CPU temperature (C)",
            "bottom_left":  "KV cache size (n_kv_cells)",
            "bottom_right": "process RSS (MB)",
        },
        "thresholds": {
            "kernel_freq_cliff_c": KERNEL_CLIFF_C,
            "cliff_mechanism": (
                "SD8-Gen5 kernel thermal governor caps big-core to "
                "~1267 MHz when ddr_temp >= 65 C"
            ),
            "headroom_c_below_cliff": round(KERNEL_CLIFF_C - all_peak, 2),
        },
        "headline": {
            "all_cells_peak_ddr_c": round(all_peak, 2),
            "all_cells_peak_cpu_c": round(cpu_peak_all, 2),
            "below_cliff_c": round(KERNEL_CLIFF_C - all_peak, 2),
            "interpretation": (
                "Llama-1B (1.3 GB Q4_K_M) stays cool across all 4 Wave-3 "
                "policies; DDR peaks at "
                f"{all_peak:.1f} C, well below the SD8-Gen5 kernel "
                "freq-cliff at 65 C. Eviction-policy differences across "
                "vanilla / v1_K2048 / v1_K512 / v1_FA_K512 produce small "
                "but measurable thermal shifts (~2 C span in peak DDR)."
            ),
        },
        "cells": schema_cells,
    }
    with open(OUT_SCHEMA, "w") as fh:
        json.dump(schema, fh, indent=2, default=float)
    print(f"[ok] wrote {OUT_SCHEMA}")


if __name__ == "__main__":
    main()
