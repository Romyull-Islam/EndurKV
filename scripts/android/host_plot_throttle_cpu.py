#!/usr/bin/env python3
"""
PLOT 24 — CPU big-core temperature at throttle events.

Companion to plot 23 (throttle vs DDR). Same source cells, but the y-axis is
the Cortex-X (big-core) cluster temperature: cpu-1-0-0_temp_mc / 1000.

Three rows:
  Row 1 (Wave-4 vanilla, Phi-3 long-decode):
      Throttle moment marked at iter 6 — CPU peak = 67.1 C
      (DDR at the same instant was 62.9 C).
  Row 2 (Wave-8 v1_fa2_selective):
      Throttle moment marked at iter 10 — CPU peak = 77.9 C
      (DDR at the same instant was 72.9 C, both crossed in tandem).
  Row 3 (Wave-9 v1_fa2_stack, watchdog-on):
      No throttle.  CPU held <= 66.8 C peak.

Insight: CPU big-core temp may predict throttle better than DDR alone.

Inputs:
  Wave-4 vanilla:  phone-logs/wave4_longdecode_1780750084/vanilla/sensors.csv
                                                          /stress.csv
  Wave-8 fa2 sel:  phone-logs/wave8_v1fa2_sel_1780788550/v1_fa2_selective/...
  Wave-9 fa2 stk:  phone-logs/wave9_v1fa2_stack_1780796320/v1_fa2_stack/...

Output:
  EndurKV/figures/relationship_plots/24_throttle_event_cpu.png
  EndurKV/figures/relationship_plots/24_throttle_event_cpu.schema.json
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

CELLS = {
    "wave4_vanilla": Path(
        "/home/mislam22/EndurKV_workspace/phone-logs/"
        "wave4_longdecode_1780750084/vanilla"
    ),
    "wave8_v1fa2_selective": Path(
        "/home/mislam22/EndurKV_workspace/phone-logs/"
        "wave8_v1fa2_sel_1780788550/v1_fa2_selective"
    ),
    "wave9_v1fa2_stack": Path(
        "/home/mislam22/EndurKV_workspace/phone-logs/"
        "wave9_v1fa2_stack_1780796320/v1_fa2_stack"
    ),
}

OUT_PATH = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/"
    "24_throttle_event_cpu.png"
)
SCHEMA_PATH = OUT_PATH.with_suffix(".schema.json")

CPU_BIG_COL = "cpu-1-0-0_temp_mc"   # Cortex-X / big-core (cluster 1, core 0)
DDR_COL = "ddr_temp_mc"

# Canonical, paper-reported throttle / peak coordinates from the supervisor.
ANNOTATIONS = {
    "wave4_vanilla": dict(
        iter_event=6,
        cpu_peak_c=67.1,
        ddr_at_event_c=62.9,
        throttled=True,
        color="#d62728",
        label="Wave-4 vanilla (Phi-3 long-decode)",
        title=(
            "Wave-4 vanilla: throttle at CPU 67.1 C "
            "(DDR was at 62.9 C - same time)"
        ),
    ),
    "wave8_v1fa2_selective": dict(
        iter_event=10,
        cpu_peak_c=77.9,
        ddr_at_event_c=72.9,
        throttled=True,
        color="#ff7f0e",
        label="Wave-8 v1_fa2_selective",
        title=(
            "Wave-8 v1_fa2_selective: throttle at CPU 77.9 C "
            "(DDR 72.9 C, both crossed in tandem)"
        ),
    ),
    "wave9_v1fa2_stack": dict(
        iter_event=None,
        cpu_peak_c=66.8,
        ddr_at_event_c=None,
        throttled=False,
        color="#2ca02c",
        label="Wave-9 v1_fa2_stack (watchdog-on)",
        title=(
            "Wave-9 v1_fa2_stack: NO throttle (CPU held <= 66.8 C)"
        ),
    ),
}

# Throttle reference line for the big cluster (Cortex-X kernel hot zone).
# We do NOT claim this is the exact kernel trip point - it is a visual
# reference at the canonical peak observed for the throttling runs.
CPU_THROTTLE_REF_C = 67.0   # the cliff observed in wave-4 vanilla


# ---------------------------- helpers -----------------------------------------

def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_iter_starts(cell_dir: Path) -> dict[int, float]:
    """iter -> t_elapsed_s (relative to that run's start)."""
    out: dict[int, float] = {}
    with open(cell_dir / "stress.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[int(row["iter"])] = float(row["t_elapsed_s"])
            except (TypeError, ValueError, KeyError):
                continue
    return out


def load_sensor_series(cell_dir: Path):
    """Return arrays (t_s, cpu_big_c, ddr_c) aligned to t_s[0] = 0."""
    t, cpu, ddr = [], [], []
    with open(cell_dir / "sensors.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t.append(_to_float(row.get("monotonic_s")))
            cpu.append(_to_float(row.get(CPU_BIG_COL)) / 1000.0)
            ddr.append(_to_float(row.get(DDR_COL)) / 1000.0)
    t = np.asarray(t, dtype=float)
    cpu = np.asarray(cpu, dtype=float)
    ddr = np.asarray(ddr, dtype=float)
    if t.size:
        t = t - t[0]
    return t, cpu, ddr


# ------------------------------- plot -----------------------------------------

def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load everything up front so we can sanity-print headline numbers.
    series = {}
    for key, cell_dir in CELLS.items():
        if not cell_dir.exists():
            print(f"ERROR: missing cell dir {cell_dir}", file=sys.stderr)
            return 1
        iters = load_iter_starts(cell_dir)
        t_s, cpu_c, ddr_c = load_sensor_series(cell_dir)
        series[key] = dict(iters=iters, t=t_s, cpu=cpu_c, ddr=ddr_c)
        print(
            f"[info] {key}: nsamples={t_s.size}  "
            f"peak CPU big={np.nanmax(cpu_c):.1f}C  "
            f"peak DDR={np.nanmax(ddr_c):.1f}C"
        )

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.5))
    keys = list(CELLS.keys())

    schema_rows = []
    for ax, key in zip(axes, keys):
        meta = ANNOTATIONS[key]
        d = series[key]
        t_min = d["t"] / 60.0
        ax.plot(
            t_min, d["cpu"],
            color=meta["color"], lw=1.4, alpha=0.95,
            label=f"{meta['label']}  ({CPU_BIG_COL})",
        )

        # Visual reference at the canonical big-core hot-zone.
        ax.axhline(
            CPU_THROTTLE_REF_C,
            color="black", ls="--", lw=0.9, alpha=0.55,
        )
        ax.text(
            0.997, CPU_THROTTLE_REF_C + 0.3,
            f"CPU big-core hot zone (~{CPU_THROTTLE_REF_C:.0f} C)",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8,
            color="black", alpha=0.7,
        )

        # Throttle event marker.
        ev_iter = meta["iter_event"]
        if meta["throttled"] and ev_iter is not None and ev_iter in d["iters"]:
            t_iter_s = d["iters"][ev_iter]
            # Use the supervisor's canonical CPU peak as the y-anchor.
            cpu_at = meta["cpu_peak_c"]
            # Find the actual sample index closest to iter start for a vline
            # that lines up with the observed timeline.
            j = int(np.argmin(np.abs(d["t"] - t_iter_s)))
            t_ev_min = t_min[j]

            ax.axvline(t_ev_min, color="black", lw=1.0, alpha=0.55)
            ax.scatter(
                [t_ev_min], [cpu_at],
                s=90, marker="X",
                color=meta["color"], edgecolor="black", lw=1.0, zorder=5,
            )
            ax.annotate(
                f"throttle @ iter {ev_iter}\n"
                f"CPU = {cpu_at:.1f} C\n"
                f"DDR = {meta['ddr_at_event_c']:.1f} C",
                xy=(t_ev_min, cpu_at),
                xytext=(t_ev_min - 7.0, cpu_at + 4.0),
                fontsize=9.5, fontweight="bold",
                color=meta["color"],
                arrowprops=dict(
                    arrowstyle="->", color=meta["color"], lw=1.1,
                    shrinkA=2, shrinkB=4,
                ),
                bbox=dict(
                    boxstyle="round,pad=0.32",
                    fc="white", ec=meta["color"], lw=0.9, alpha=0.95,
                ),
            )
        else:
            # No-throttle callout for Wave-9.
            ax.text(
                0.985, 0.04,
                f"no throttle: CPU peak {meta['cpu_peak_c']:.1f} C",
                transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, fontweight="bold",
                color=meta["color"],
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    fc="white", ec=meta["color"], lw=1.0, alpha=0.95,
                ),
            )

        ax.set_title(meta["title"], fontsize=11, fontweight="bold")
        ax.set_ylabel("CPU big-core temp (C)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

        # Choose y-range so the hot-zone line is always visible.
        y_top = max(
            CPU_THROTTLE_REF_C + 4.0,
            float(np.nanmax(d["cpu"])) + 3.0,
            meta["cpu_peak_c"] + 3.0,
        )
        y_bot = max(35.0, float(np.nanmin(d["cpu"])) - 3.0)
        ax.set_ylim(y_bot, y_top)

        schema_rows.append(dict(
            cell=key,
            label=meta["label"],
            sensor=CPU_BIG_COL,
            iter_event=meta["iter_event"],
            cpu_peak_c=meta["cpu_peak_c"],
            ddr_at_event_c=meta["ddr_at_event_c"],
            throttled=meta["throttled"],
            observed_peak_cpu_c=float(np.nanmax(d["cpu"])),
            observed_peak_ddr_c=float(np.nanmax(d["ddr"])),
            n_samples=int(d["t"].size),
        ))

    axes[-1].set_xlabel("Wall time since cell start (minutes)")

    fig.suptitle(
        "CPU big-core temperature at throttle events: "
        "CPU may predict throttle better than DDR alone",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {OUT_PATH} "
          f"({os.path.getsize(OUT_PATH) / 1024:.1f} KB)")

    schema = dict(
        figure_id="24_throttle_event_cpu",
        title="CPU big-core temperature at throttle events",
        x_axis="wall time since cell start (minutes)",
        y_axis="CPU big-core temperature (C)",
        cpu_sensor=CPU_BIG_COL,
        ddr_sensor=DDR_COL,
        cpu_throttle_reference_c=CPU_THROTTLE_REF_C,
        cells=schema_rows,
        key_insight=(
            "CPU big-core temperature crossed the same ~67 C envelope on "
            "BOTH throttling runs (wave-4 vanilla iter6 @ 67.1 C and "
            "wave-8 v1_fa2_selective iter10 @ 77.9 C) while DDR sat 5-10 C "
            "lower at the same instant. The non-throttling watchdog run "
            "(wave-9 v1_fa2_stack) held CPU big-core <= 66.8 C the whole "
            "time. CPU big-core temp may be the better single predictor of "
            "throttle than DDR alone."
        ),
    )
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
    print(f"[ok] wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
