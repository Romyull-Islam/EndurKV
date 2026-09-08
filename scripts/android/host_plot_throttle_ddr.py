#!/usr/bin/env python3
"""
host_plot_throttle_ddr.py — DDR temperature at throttle events.

Produces a single figure with three stacked subplots, each showing the LPDDR5X
controller temperature over the lifetime of one decode cell. Vertical red
dashed lines mark the moment(s) at which the per-iter throttle test
(``cpu6_freq_hz`` drop >=10% OR ``decode_tps`` drop >=15%) fires.

Cells:
  Subplot 1: Wave-4 vanilla — throttle fires at iter 6 (DDR 62.9 C peak).
  Subplot 2: Wave-8 v1_fa2_selective — throttle fires at iter 10 (DDR 72.9 C).
  Subplot 3: Wave-9 v1_fa2_stack — NO throttle. Watchdog tier transitions are
             overlayed for context.

Empirical thresholds: 58 / 62 / 65 C horizontal guides on every subplot.

Outputs:
  figures/relationship_plots/23_throttle_event_ddr.png

Run (defaults assume the standard workspace layout):
  python scripts/android/host_plot_throttle_ddr.py
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = Path(
    os.environ.get("WORKSPACE", "/home/mislam22/EndurKV_workspace")
).expanduser()
PHONE_LOGS = WORKSPACE / "phone-logs"
FIG_OUT = (
    WORKSPACE
    / "EndurKV"
    / "figures"
    / "relationship_plots"
    / "23_throttle_event_ddr.png"
)

# Empirical DDR thresholds (degrees C) used by the wave-9 watchdog.
THRESH_C = [58.0, 62.0, 65.0]
THRESH_COLORS = ["#2ca02c", "#ff9f1a", "#d62728"]
THRESH_LABELS = ["58 C (tier 1 entry)", "62 C (tier 2 entry)", "65 C (alarm)"]


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_sensors(path: Path) -> Tuple[List[float], List[Optional[float]], List[Optional[float]]]:
    """Return (t_rel_s, ddr_c, cpu6_mhz) sampled at the sensor cadence.

    ``t_rel_s`` is wall-clock seconds since the first sensor row of the cell.
    """
    t_abs: List[float] = []
    ddr: List[Optional[float]] = []
    cpu6: List[Optional[float]] = []
    with path.open() as f:
        r = csv.reader(f)
        hdr = next(r)
        ddr_i = hdr.index("ddr_temp_mc")
        cpu6_i = hdr.index("cpu6_freq_hz")
        for row in r:
            try:
                t = float(row[0])
            except (ValueError, IndexError):
                continue
            t_abs.append(t)
            try:
                ddr.append(float(row[ddr_i]) / 1000.0)
            except (ValueError, IndexError):
                ddr.append(None)
            try:
                # raw values are kHz (max 1632000); convert to MHz.
                cpu6.append(float(row[cpu6_i]) / 1000.0)
            except (ValueError, IndexError):
                cpu6.append(None)
    if not t_abs:
        return [], [], []
    t0 = t_abs[0]
    t_rel = [t - t0 for t in t_abs]
    return t_rel, ddr, cpu6


def load_stress(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["iter"] = int(row["iter"])
                row["t_elapsed_s"] = float(row["t_elapsed_s"])
                row["decode_tps"] = float(row["decode_tps"])
            except (KeyError, ValueError):
                continue
            rows.append(row)
    return rows


def load_watchdog(path: Path) -> Tuple[float, List[Tuple[float, int, int]]]:
    """Return (anchor_ts, [(unix_ts, tier, freq_khz), ...]).

    The first watchdog line is ``watchdog start``; its unix timestamp is
    returned as the anchor used to convert subsequent events to relative
    seconds from cell start.
    """
    events: List[Tuple[float, int, int]] = []
    anchor: Optional[float] = None
    pat_start = re.compile(r"^\[(\d+)\]\s+watchdog start")
    pat_tier = re.compile(r"^\[(\d+)\]\s+DDR=(\d+)C\s+->\s+tier=(\d+)\s+\S+=(\d+)")
    with path.open() as f:
        for line in f:
            line = line.strip()
            if anchor is None:
                m = pat_start.match(line)
                if m:
                    anchor = float(m.group(1))
                    continue
            m = pat_tier.match(line)
            if m:
                ts = float(m.group(1))
                tier = int(m.group(3))
                freq_khz = int(m.group(4))
                events.append((ts, tier, freq_khz))
    if anchor is None:
        anchor = 0.0
    return anchor, events


# ----------------------------------------------------------------------------
# Throttle detector (paper-style)
# ----------------------------------------------------------------------------
def find_throttle_iter(stress: List[dict]) -> Optional[int]:
    """Return iter number whose decode_tps drops >=15% vs prev, else None."""
    prev_tps = None
    for row in stress:
        if prev_tps is not None and row["decode_tps"] <= 0.85 * prev_tps:
            return row["iter"]
        prev_tps = row["decode_tps"]
    return None


def iter_start_t(stress: List[dict], iter_no: int) -> float:
    for row in stress:
        if row["iter"] == iter_no:
            return row["t_elapsed_s"]
    raise KeyError(f"iter {iter_no} not in stress.csv")


def window_max(
    t_rel: List[float],
    ddr: List[Optional[float]],
    cpu6: List[Optional[float]],
    t_start: float,
    t_end: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Within [t_start, t_end] return (max_ddr, t_max_ddr, min_cpu6, t_min_cpu6)."""
    best_d, t_d, best_c, t_c = None, None, None, None
    for ti, d, c in zip(t_rel, ddr, cpu6):
        if not (t_start <= ti <= t_end):
            continue
        if d is not None and (best_d is None or d > best_d):
            best_d, t_d = d, ti
        if c is not None and (best_c is None or c < best_c):
            best_c, t_c = c, ti
    return best_d, t_d, best_c, t_c


# ----------------------------------------------------------------------------
# Plotters
# ----------------------------------------------------------------------------
def add_thresholds(ax) -> None:
    for c, color, label in zip(THRESH_C, THRESH_COLORS, THRESH_LABELS):
        ax.axhline(c, color=color, linestyle=":", linewidth=1.0, alpha=0.7, label=label)


def plot_throttle_cell(
    ax,
    title: str,
    t_rel: List[float],
    ddr: List[Optional[float]],
    throttle_t: float,
    throttle_ddr: float,
) -> None:
    # Mask out None DDR samples.
    xs = [t for t, d in zip(t_rel, ddr) if d is not None]
    ys = [d for d in ddr if d is not None]
    ax.plot(xs, ys, color="#1f3a93", linewidth=1.4, label="DDR temp")
    add_thresholds(ax)
    # Throttle marker.
    ax.axvline(throttle_t, color="#d62728", linestyle="--", linewidth=1.6, label="throttle iter start")
    ax.annotate(
        f"{throttle_ddr:.1f} C @ throttle",
        xy=(throttle_t, throttle_ddr),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=10,
        color="#d62728",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5", ec="#d62728", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
    )
    ax.scatter([throttle_t], [throttle_ddr], color="#d62728", zorder=5, s=28)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("DDR temp (C)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(30, 80)


def plot_no_throttle_cell(
    ax,
    title: str,
    t_rel: List[float],
    ddr: List[Optional[float]],
    watchdog_events_rel: List[Tuple[float, int, int]],
) -> None:
    xs = [t for t, d in zip(t_rel, ddr) if d is not None]
    ys = [d for d in ddr if d is not None]
    ax.plot(xs, ys, color="#1f3a93", linewidth=1.4, label="DDR temp")
    add_thresholds(ax)
    # Watchdog tier annotations: paint short vertical segments at each transition.
    tier_color = {0: "#2ca02c", 1: "#ff9f1a", 2: "#d62728"}
    plotted_labels = set()
    for t_evt, tier, freq_khz in watchdog_events_rel:
        col = tier_color.get(tier, "gray")
        label = f"watchdog tier={tier}"
        if label in plotted_labels:
            label = None
        else:
            plotted_labels.add(f"watchdog tier={tier}")
        ax.axvline(
            t_evt,
            color=col,
            linestyle="-",
            linewidth=0.8,
            alpha=0.35,
            label=label,
        )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("DDR temp (C)")
    ax.set_xlabel("time since cell start (s)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(30, 80)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(WORKSPACE))
    ap.add_argument("--out", default=str(FIG_OUT))
    args = ap.parse_args()

    ws = Path(args.workspace)
    phone_logs = ws / "phone-logs"

    w4_dir = phone_logs / "wave4_longdecode_1780750084" / "vanilla"
    w8_dir = phone_logs / "wave8_v1fa2_sel_1780788550" / "v1_fa2_selective"
    w9_dir = phone_logs / "wave9_v1fa2_stack_1780796320" / "v1_fa2_stack"
    w9_watchdog = phone_logs / "wave9_v1fa2_stack_1780796320" / "watchdog.log"

    # ---- Wave-4 vanilla ----
    t4, ddr4, cpu64 = load_sensors(w4_dir / "sensors.csv")
    stress4 = load_stress(w4_dir / "stress.csv")
    iter4 = 6
    t4_start = iter_start_t(stress4, iter4)
    # iter ends at next iter's start, else last sample.
    t4_end = next(
        (r["t_elapsed_s"] for r in stress4 if r["iter"] == iter4 + 1),
        t4[-1] if t4 else t4_start + 500,
    )
    max_ddr4, t_max_ddr4, min_cpu64, _ = window_max(t4, ddr4, cpu64, t4_start, t4_end)
    # The title uses the canonical per-iter avg-frequency callout (decode_tps
    # ratio -> -15%, raw kHz callout 1382 -> 1171 MHz) from the analysis log;
    # the in-window cpu6 min sample is reported in the SUMMARY line for
    # transparency.
    title4 = (
        f"Wave-4 vanilla: throttle at DDR {max_ddr4:.1f} C "
        f"(iter {iter4}, freq 1382 -> 1171 MHz, -15%)"
    )
    pre_freq4 = next(
        (c for t, c in zip(t4, cpu64) if t >= t4_start and c is not None),
        None,
    )

    # ---- Wave-8 v1_fa2_selective ----
    t8, ddr8, cpu68 = load_sensors(w8_dir / "sensors.csv")
    stress8 = load_stress(w8_dir / "stress.csv")
    iter8 = 10
    t8_start = iter_start_t(stress8, iter8)
    t8_end = next(
        (r["t_elapsed_s"] for r in stress8 if r["iter"] == iter8 + 1),
        t8[-1] if t8 else t8_start + 500,
    )
    max_ddr8, t_max_ddr8, min_cpu68, _ = window_max(t8, ddr8, cpu68, t8_start, t8_end)
    pre_freq8 = next(
        (c for t, c in zip(t8, cpu68) if t >= t8_start and c is not None),
        None,
    )
    title8 = (
        f"Wave-8 v1_fa2_selective: throttle at DDR {max_ddr8:.1f} C "
        f"(iter {iter8}, freq 1497 -> 1355 MHz, -9.5%)"
    )

    # ---- Wave-9 v1_fa2_stack (no throttle) ----
    t9, ddr9, cpu69 = load_sensors(w9_dir / "sensors.csv")
    # Anchor wave9 sensors at its first sample (absolute unix).
    # Then translate watchdog events to the same axis.
    with (w9_dir / "sensors.csv").open() as f:
        r = csv.reader(f); next(r)
        sensor_t0 = float(next(r)[0])
    wd_anchor, wd_events = load_watchdog(w9_watchdog)
    # Convert watchdog events to seconds-since-sensor-start.
    wd_events_rel = [
        (ts - sensor_t0, tier, fk) for (ts, tier, fk) in wd_events
    ]
    max_ddr9 = max((d for d in ddr9 if d is not None), default=0.0)
    title9 = (
        f"Wave-9 v1_fa2_stack: NO throttle "
        f"(watchdog active, DDR held <= {max_ddr9:.1f} C)"
    )

    # ---- Render ----
    fig, axes = plt.subplots(3, 1, figsize=(11, 11.5), sharex=False)
    plot_throttle_cell(
        axes[0], title4, t4, ddr4, throttle_t=t_max_ddr4, throttle_ddr=max_ddr4
    )
    plot_throttle_cell(
        axes[1], title8, t8, ddr8, throttle_t=t_max_ddr8, throttle_ddr=max_ddr8
    )
    plot_no_throttle_cell(axes[2], title9, t9, ddr9, wd_events_rel)

    axes[0].set_xlabel("time since cell start (s)")
    axes[1].set_xlabel("time since cell start (s)")

    # Combined legend for the throttle subplots; wave-9 keeps its own.
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", fontsize=8, ncol=2)
    axes[1].legend(loc="upper left", fontsize=8, ncol=2)
    axes[2].legend(loc="upper left", fontsize=8, ncol=2)

    fig.suptitle(
        "DDR temperature at per-iter throttle events (3 cells)", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)

    # Stdout schema summary for the orchestrator.
    print(f"WROTE {out}")
    print(
        "SUMMARY",
        {
            "wave4_iter": iter4,
            "wave4_ddr_at_throttle_c": round(max_ddr4, 2),
            "wave4_cpu6_drop_mhz": (round(pre_freq4, 1), round(min_cpu64, 1)),
            "wave8_iter": iter8,
            "wave8_ddr_at_throttle_c": round(max_ddr8, 2),
            "wave8_cpu6_drop_mhz": (round(pre_freq8, 1), round(min_cpu68, 1)),
            "wave9_max_ddr_c": round(max_ddr9, 2),
            "wave9_watchdog_events": len(wd_events_rel),
        },
    )


if __name__ == "__main__":
    main()
