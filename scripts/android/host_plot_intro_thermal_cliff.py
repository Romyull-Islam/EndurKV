#!/usr/bin/env python3
"""
FIGURE 1 (intro) - Thermal-cliff motivator.

Tells the dissertation's opening story in two panels:
  Top    - per-iteration decode tps for three Wave-3..10 runs:
             Wave-4 vanilla (red)     -> drops 6.05 -> 4.08 over 9 iters
             Wave-8 v1_fa2 selective  -> 9 flat iters, iter-10 kernel cliff
             Wave-9 v1_fa2 stack      -> 10 iters >= 4.6 tps via watchdog
  Bottom - DDR-temp trajectory per iter aligned on the same x-axis (iter idx).
           Horizontal annotated line at 65 C marks the SD8-Gen5 kernel
           freq-cliff trip point.

Sources (ALL Wave-3..10 only; no Wave-11):
  Wave-4 vanilla:      /home/mislam22/EndurKV_workspace/phone-logs/
                       wave4_longdecode_1780750084/vanilla
  Wave-8 v1_fa2 sel.:  /home/mislam22/EndurKV_workspace/phone-logs/
                       wave8_v1fa2_sel_1780788550/v1_fa2_selective
  Wave-9 v1_fa2 stack: /home/mislam22/EndurKV_workspace/phone-logs/
                       wave9_v1fa2_stack_1780796320/v1_fa2_stack

Each cell ships its own stress.csv (iter, t_elapsed_s, decode_tps, ...) and
sensors.csv (wall_clock_s, ddr_temp_mc, ...). We use stress.csv for tps
and per-iter wall-clock alignment, and join in DDR by sampling the sensors
trace at each iteration's start/end window.

Output:
  /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/
  19_intro_thermal_cliff.png
  + matching .schema.json sidecar (RES_SCHEMA)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----- inputs ---------------------------------------------------------------

PHONE_LOGS = "/home/mislam22/EndurKV_workspace/phone-logs"

W4_DIR = f"{PHONE_LOGS}/wave4_longdecode_1780750084/vanilla"
W8_DIR = f"{PHONE_LOGS}/wave8_v1fa2_sel_1780788550/v1_fa2_selective"
W9_DIR = f"{PHONE_LOGS}/wave9_v1fa2_stack_1780796320/v1_fa2_stack"

OUT_PNG = (
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/"
    "relationship_plots/19_intro_thermal_cliff.png"
)
OUT_SCHEMA = OUT_PNG.replace(".png", ".schema.json")

DDR_COL = "ddr_temp_mc"
KERNEL_CLIFF_C = 65.0


@dataclass
class Cell:
    label: str
    color: str
    src_dir: str
    wave: str
    policy: str


CELLS = [
    Cell(
        label="W4 vanilla (no control, no FA)",
        color="#d62728",
        src_dir=W4_DIR,
        wave="wave4",
        policy="vanilla",
    ),
    Cell(
        label="W8 v1_fa2 selective (no watchdog)",
        color="#e9b40a",  # yellow
        src_dir=W8_DIR,
        wave="wave8",
        policy="v1_fa2_selective",
    ),
    Cell(
        label="W9 v1_fa2 stack (closed-loop watchdog)",
        color="#2ca02c",
        src_dir=W9_DIR,
        wave="wave9",
        policy="v1_fa2_stack",
    ),
]


# ----- helpers --------------------------------------------------------------

def load_stress(src_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(src_dir, "stress.csv"))
    df["iter"] = pd.to_numeric(df["iter"], errors="coerce").astype(int)
    df["t_elapsed_s"] = pd.to_numeric(df["t_elapsed_s"], errors="coerce")
    df["decode_tps"] = pd.to_numeric(df["decode_tps"], errors="coerce")
    return df


def load_sensors(src_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(src_dir, "sensors.csv"), low_memory=False)
    df["wall_clock_s"] = pd.to_numeric(df["wall_clock_s"], errors="coerce")
    df["ddr_c"] = pd.to_numeric(df[DDR_COL], errors="coerce") / 1000.0
    df = df.dropna(subset=["wall_clock_s", "ddr_c"]).reset_index(drop=True)
    return df


def attach_ddr(stress: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    """For each iter, take the median DDR temp during the iteration window."""
    t0 = sensors["wall_clock_s"].iloc[0]
    sensors = sensors.copy()
    sensors["t_rel_s"] = sensors["wall_clock_s"] - t0

    # iteration boundaries on the same elapsed-since-run-start clock.
    starts = stress["t_elapsed_s"].to_numpy()
    # end of iter i is start of iter i+1; last iter goes to sensors end.
    ends = np.concatenate([starts[1:], [sensors["t_rel_s"].iloc[-1]]])

    ddr_med = np.full(len(stress), np.nan)
    ddr_peak = np.full(len(stress), np.nan)
    ddr_end = np.full(len(stress), np.nan)
    for i, (s, e) in enumerate(zip(starts, ends)):
        sl = sensors[(sensors["t_rel_s"] >= s) & (sensors["t_rel_s"] < e)]
        if len(sl) == 0:
            continue
        ddr_med[i] = float(sl["ddr_c"].median())
        ddr_peak[i] = float(sl["ddr_c"].max())
        # snapshot near end of iter (last 10% of window) for trajectory
        cut = s + 0.9 * (e - s)
        tail = sl[sl["t_rel_s"] >= cut]
        if len(tail) == 0:
            tail = sl.tail(max(1, len(sl) // 10))
        ddr_end[i] = float(tail["ddr_c"].median())

    out = stress.copy()
    out["ddr_med_c"] = ddr_med
    out["ddr_peak_c"] = ddr_peak
    out["ddr_end_c"] = ddr_end
    return out


# ----- main -----------------------------------------------------------------

def main() -> None:
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    rows: list[dict] = []
    per_cell: list[tuple[Cell, pd.DataFrame]] = []
    for c in CELLS:
        if not os.path.isdir(c.src_dir):
            sys.exit(f"[fatal] missing src dir: {c.src_dir}")
        st = load_stress(c.src_dir)
        sn = load_sensors(c.src_dir)
        df = attach_ddr(st, sn)
        per_cell.append((c, df))
        print(
            f"[info] {c.label:<42}  n_iter={len(df):>2}  "
            f"tps={df['decode_tps'].iloc[0]:.2f} -> "
            f"{df['decode_tps'].iloc[-1]:.2f}  "
            f"DDR(end iter1)={df['ddr_end_c'].iloc[0]:.1f}C "
            f"-> (end last)={df['ddr_end_c'].iloc[-1]:.1f}C  "
            f"peak DDR={df['ddr_peak_c'].max():.1f}C"
        )
        rows.append(
            {
                "wave": c.wave,
                "policy": c.policy,
                "label": c.label,
                "src_dir": c.src_dir,
                "n_iter": int(len(df)),
                "iters": df["iter"].astype(int).tolist(),
                "decode_tps_per_iter": df["decode_tps"].round(3).tolist(),
                "ddr_med_c_per_iter": np.round(
                    df["ddr_med_c"].to_numpy(), 2
                ).tolist(),
                "ddr_peak_c_per_iter": np.round(
                    df["ddr_peak_c"].to_numpy(), 2
                ).tolist(),
                "ddr_end_c_per_iter": np.round(
                    df["ddr_end_c"].to_numpy(), 2
                ).tolist(),
                "mean_tps": float(df["decode_tps"].mean()),
                "tps_first": float(df["decode_tps"].iloc[0]),
                "tps_last": float(df["decode_tps"].iloc[-1]),
                "tps_drop_pct": (
                    float(
                        100.0
                        * (
                            df["decode_tps"].iloc[0]
                            - df["decode_tps"].iloc[-1]
                        )
                        / max(df["decode_tps"].iloc[0], 1e-9)
                    )
                ),
                "peak_ddr_c": float(df["ddr_peak_c"].max()),
            }
        )

    # ----- figure ----------------------------------------------------------
    fig, (ax_t, ax_d) = plt.subplots(
        2, 1, figsize=(11, 8.5), sharex=True
    )

    # ----- TOP: per-iter tps trajectory ------------------------------------
    for c, df in per_cell:
        ax_t.plot(
            df["iter"], df["decode_tps"],
            color=c.color, marker="o", ms=6.5, lw=2.0, label=c.label,
        )

    # callouts: vanilla collapse, w8 cliff, w9 sustain
    w4 = per_cell[0][1]
    w8 = per_cell[1][1]
    w9 = per_cell[2][1]

    # 1) vanilla collapse: arrow from iter1 to iter9
    drop_pct_v = (
        100.0
        * (w4["decode_tps"].iloc[0] - w4["decode_tps"].iloc[-1])
        / w4["decode_tps"].iloc[0]
    )
    ax_t.annotate(
        f"vanilla collapses {drop_pct_v:.0f}% across 9 iters\n"
        f"({w4['decode_tps'].iloc[0]:.1f} -> "
        f"{w4['decode_tps'].iloc[-1]:.1f} tps)",
        xy=(w4["iter"].iloc[-1], w4["decode_tps"].iloc[-1]),
        xytext=(2.0, 5.40),
        fontsize=10, color="#d62728", fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.30", fc="#fdecec", ec="#d62728", lw=0.9,
        ),
        arrowprops=dict(
            arrowstyle="->", color="#d62728", lw=1.2, shrinkA=2, shrinkB=2,
        ),
    )

    # 2) wave-8 iter-10 cliff
    w8_cliff_iter = int(w8.loc[w8["decode_tps"].idxmin(), "iter"])
    w8_cliff_tps = float(w8["decode_tps"].min())
    w8_pre_tps = float(w8.loc[w8["iter"] == w8_cliff_iter - 1, "decode_tps"].iloc[0])
    ax_t.annotate(
        f"W8 iter-{w8_cliff_iter} kernel cliff\n"
        f"({w8_pre_tps:.1f} -> {w8_cliff_tps:.1f} tps, "
        f"{100*(w8_pre_tps - w8_cliff_tps)/w8_pre_tps:.0f}% drop)",
        xy=(w8_cliff_iter, w8_cliff_tps),
        xytext=(6.4, 3.20),
        fontsize=9.5, color="#8a6d00", fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.30", fc="#fff7d6", ec="#8a6d00", lw=0.9,
        ),
        arrowprops=dict(
            arrowstyle="->", color="#8a6d00", lw=1.2, shrinkA=2, shrinkB=2,
        ),
    )

    # 3) wave-9 sustain band
    ax_t.axhspan(
        float(w9["decode_tps"].min()), float(w9["decode_tps"].max()),
        color="#2ca02c", alpha=0.07, zorder=0,
    )
    ax_t.text(
        0.98, 0.96,
        f"W9 holds {w9['decode_tps'].min():.1f}-"
        f"{w9['decode_tps'].max():.1f} tps for 10 iters\n"
        f"(watchdog pre-empts kernel cliff)",
        transform=ax_t.transAxes,
        ha="right", va="top", fontsize=9.5,
        color="#1b6e1b", fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.35", fc="#e8f5e8", ec="#2ca02c", lw=0.9,
        ),
    )

    ax_t.set_ylabel("decode throughput (tokens / s)")
    ax_t.set_title(
        "Per-iteration decode tps - vanilla collapse, "
        "W8 kernel cliff, W9 watchdog sustain"
    )
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="lower left", fontsize=9.0, framealpha=0.93)
    ax_t.set_ylim(2.5, 8.2)
    ax_t.set_xticks(range(1, 12))
    ax_t.set_xlim(0.5, 11.5)

    # ----- BOTTOM: DDR temp per iter ---------------------------------------
    for c, df in per_cell:
        ax_d.plot(
            df["iter"], df["ddr_end_c"],
            color=c.color, marker="s", ms=6.0, lw=2.0,
            label=f"{c.label}  (peak {df['ddr_peak_c'].max():.1f} C)",
        )
        # also show the per-iter peak as a faint trace
        ax_d.plot(
            df["iter"], df["ddr_peak_c"],
            color=c.color, lw=0.9, ls=":", alpha=0.55,
        )

    # 65 C kernel cliff line (data coords; place label just above the line)
    ax_d.axhline(
        KERNEL_CLIFF_C,
        color="#c00000", ls="--", lw=1.8, alpha=0.9, zorder=1,
    )
    ax_d.text(
        0.99, KERNEL_CLIFF_C + 0.6,
        f"kernel freq-cliff trip @ DDR >= {KERNEL_CLIFF_C:.0f} C  "
        f"(SD8-Gen5 governor caps big-core to ~1.27 GHz)",
        transform=ax_d.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=9.5, color="#c00000",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.35", fc="#ffe9e9", ec="#c00000", lw=0.9,
        ),
    )

    # mark wave-8 cliff iter on bottom panel too
    w8_cliff_ddr = float(
        w8.loc[w8["iter"] == w8_cliff_iter, "ddr_peak_c"].iloc[0]
    )
    ax_d.scatter(
        [w8_cliff_iter], [w8_cliff_ddr],
        s=180, facecolors="none", edgecolors="#8a6d00", lw=2.2, zorder=4,
    )
    ax_d.annotate(
        f"W8 iter-{w8_cliff_iter}: DDR peaks "
        f"{w8_cliff_ddr:.1f} C\nkernel forces freq cliff",
        xy=(w8_cliff_iter, w8_cliff_ddr),
        xytext=(w8_cliff_iter - 4.0, w8_cliff_ddr + 3.0),
        fontsize=9.0, color="#8a6d00", fontweight="bold",
        arrowprops=dict(
            arrowstyle="->", color="#8a6d00", lw=1.2, shrinkA=4, shrinkB=4,
        ),
    )

    ax_d.set_xlabel("iteration index (each iter = 2048-token decode)")
    ax_d.set_ylabel("DDR memory temp (C)")
    ax_d.set_title(
        "DDR temperature per iter - solid: end-of-iter, dotted: in-iter peak"
    )
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="lower right", fontsize=8.5, framealpha=0.93)
    ax_d.set_xticks(range(1, 12))
    ax_d.set_xlim(0.5, 11.5)
    ax_d.set_ylim(48, 78)

    fig.suptitle(
        "On-device LLM inference hits thermal cliff at iter 6-10 "
        "without closed-loop control",
        fontsize=13.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT_PNG, dpi=160)
    print(f"[ok] wrote {OUT_PNG}")

    # ----- schema sidecar -------------------------------------------------
    schema = {
        "kind": "RES_SCHEMA",
        "version": 1,
        "generator": "host_plot_intro_thermal_cliff.py",
        "figure": OUT_PNG,
        "title": (
            "On-device LLM inference hits thermal cliff at iter 6-10 "
            "without closed-loop control"
        ),
        "role_in_paper": "FIGURE 1 (intro motivator)",
        "x_axis": "iteration index (1..N)",
        "y_axes": {
            "top": "decode tokens / s",
            "bottom": "DDR memory temperature (C)",
        },
        "thresholds": {
            "kernel_freq_cliff_c": KERNEL_CLIFF_C,
            "cliff_mechanism": (
                "SD8-Gen5 kernel thermal governor caps big-core to "
                "~1267 MHz when ddr_temp >= 65 C"
            ),
        },
        "waves_used": ["wave4", "wave8", "wave9"],
        "waves_excluded": ["wave11"],
        "wave_sources": {
            "wave4": W4_DIR,
            "wave8": W8_DIR,
            "wave9": W9_DIR,
        },
        "story": {
            "vanilla_collapse_pct": rows[0]["tps_drop_pct"],
            "wave8_cliff_iter": int(w8_cliff_iter),
            "wave8_cliff_tps_drop_pct": float(
                100.0 * (w8_pre_tps - w8_cliff_tps) / w8_pre_tps
            ),
            "wave9_min_tps": float(w9["decode_tps"].min()),
            "wave9_max_tps": float(w9["decode_tps"].max()),
            "wave9_holds_tps_floor": float(w9["decode_tps"].min()),
        },
        "cells": rows,
    }
    with open(OUT_SCHEMA, "w") as fh:
        json.dump(schema, fh, indent=2, default=float)
    print(f"[ok] wrote {OUT_SCHEMA}")


if __name__ == "__main__":
    main()
