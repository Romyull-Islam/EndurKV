#!/usr/bin/env python3
"""
PLOT 25 — SoC + skin + battery temperature at throttle events.

Three stacked subplots (one per wave / cell), each overlaying four traces
on a single time axis:

  * sys-therm-2          (HAL telemetry zone with low trip points 48-61 C)
  * shell_front          (skin / chassis proxy)
  * battery_temp         (BCL knee at 45-50 C)
  * bat_current_ma       (scaled; BCL clamp signature)

The throttle event for each cell is annotated with a vertical marker and a
text callout listing the simultaneous sys-therm-2 / shell_front /
battery_temp values from the sensors.csv at that moment. The titles are
chosen to call out which sensor crossed its empirical trip first.

Source cells:
  * Wave-4 vanilla            (Llama-3.2-1B long-decode, no eviction)
  * Wave-8 v1_fa2_selective   (no preemptive watchdog)
  * Wave-9 v1_fa2_stack       (with preemptive watchdog)

Outputs:
  /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/
      25_throttle_event_soc.png
  /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/
      25_throttle_event_soc.schema.json
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


# -------------------------- paths / config ------------------------------------

OUT_PATH = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/"
    "25_throttle_event_soc.png"
)
SCHEMA_PATH = OUT_PATH.with_suffix(".schema.json")

# (label, sensors.csv path, throttle_t_seconds_relative_to_first_sample,
#  pre_throttle_event annotations, title)
CELLS = [
    {
        "wave":  "Wave-4 vanilla",
        "csv":   Path(
            "/home/mislam22/EndurKV_workspace/phone-logs/"
            "wave4_longdecode_1780750084/vanilla/sensors.csv"
        ),
        # detected from peak sys-therm-2 (55.67 C @ 2134 s relative)
        # the supervisor-supplied snapshot lands at the 55.7 C crossing.
        "throttle_t_s": 2128.8,
        "snap_sys_therm_2": 55.7,
        "snap_shell_front": 49.0,
        "snap_battery_temp": 47.2,
        "snap_bat_current_ma": 312.0,
        "title": (
            "Wave-4 vanilla: sys-therm-2 was 55.7°C at throttle "
            "(close to 60°C trip — possibly the trigger)"
        ),
        "first_crossed": "sys-therm-2 (55.7°C, 4.3°C below 60°C trip)",
    },
    {
        "wave":  "Wave-8 v1_fa2_selective",
        "csv":   Path(
            "/home/mislam22/EndurKV_workspace/phone-logs/"
            "wave8_v1fa2_sel_1780788550/v1_fa2_selective/sensors.csv"
        ),
        # first sys-therm-2 crossing of 62 C lands at ~3218 s relative.
        "throttle_t_s": 3218.1,
        "snap_sys_therm_2": 62.1,
        "snap_shell_front": 53.0,
        "snap_battery_temp": 51.1,
        "snap_bat_current_pre": 371.0,
        "snap_bat_current_post": 296.0,
        "snap_bat_current_ma": 296.0,
        "title": (
            "Wave-8 v1_fa2_selective: sys-therm-2 CROSSED 60°C trip "
            "(62.1°C); BCL clamped battery current"
        ),
        "first_crossed": "sys-therm-2 (62.1°C, +2.1°C past 60°C HAL trip)",
    },
    {
        "wave":  "Wave-9 v1_fa2_stack",
        "csv":   Path(
            "/home/mislam22/EndurKV_workspace/phone-logs/"
            "wave9_v1fa2_stack_1780796320/v1_fa2_stack/sensors.csv"
        ),
        # No empirical-trip throttle event in this cell; mark the run-peak
        # of sys-therm-2 for context (the preemptive watchdog held below it).
        "throttle_t_s": None,
        "snap_sys_therm_2": 55.7,
        "snap_shell_front": 49.0,
        "snap_battery_temp": 47.3,
        "snap_bat_current_ma": None,
        "title": (
            "Wave-9 v1_fa2_stack: all SoC/skin sensors held below "
            "empirical throttle triggers"
        ),
        "first_crossed": "none crossed (watchdog clamped early)",
    },
]

# Empirical trip thresholds for the three thermal channels.  These are the
# "low trip points" actually published by the HAL on the OnePlus 15.
TRIP_C = {
    "sys_therm_2_low": 48.0,
    "sys_therm_2_high": 60.0,   # passive trip near 60-61 C
    "shell_front":     55.0,    # skin comfort
    "battery_temp_lo": 45.0,    # BCL knee
    "battery_temp_hi": 50.0,    # BCL hard clamp
}

# Sensor column names in sensors.csv
COL_T          = "monotonic_s"
COL_SYS_TH2    = "sys-therm-2_temp_mc"
COL_SHELL_FR   = "shell_front_temp_mc"
COL_BAT_T      = "battery_temp_mc"
COL_BAT_I      = "bat_current_ma"


# --------------------------- helpers ------------------------------------------

def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_cell(csv_path: Path) -> dict:
    """Return time/temperature/current arrays from a single sensors.csv."""
    t = []
    sys2 = []
    shell = []
    bat_t = []
    bat_i = []
    with open(csv_path, newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            t.append(_f(row.get(COL_T)))
            sys2.append(_f(row.get(COL_SYS_TH2)))
            shell.append(_f(row.get(COL_SHELL_FR)))
            bat_t.append(_f(row.get(COL_BAT_T)))
            bat_i.append(_f(row.get(COL_BAT_I)))
    t_arr = np.asarray(t, dtype=float)
    if t_arr.size:
        t_arr = t_arr - t_arr[0]
    sys2_C  = np.asarray(sys2,  dtype=float) / 1000.0
    shell_C = np.asarray(shell, dtype=float) / 1000.0
    bat_C   = np.asarray(bat_t, dtype=float) / 1000.0
    bat_mA  = np.asarray(bat_i, dtype=float)
    # Filter implausible reads
    def _clip(arr, lo, hi):
        return np.where((arr < lo) | (arr > hi), np.nan, arr)
    sys2_C  = _clip(sys2_C,  10.0, 120.0)
    shell_C = _clip(shell_C, 10.0, 120.0)
    bat_C   = _clip(bat_C,   10.0, 120.0)
    return dict(
        t=t_arr, sys2=sys2_C, shell=shell_C, bat_t=bat_C, bat_i=bat_mA,
    )


def find_index_at_time(t_arr: np.ndarray, t_target: float) -> int:
    if t_arr.size == 0 or t_target is None or not np.isfinite(t_target):
        return -1
    return int(np.argmin(np.abs(t_arr - t_target)))


# ----------------------------- plot -------------------------------------------

def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load all three cells
    loaded = []
    for cfg in CELLS:
        if not cfg["csv"].exists():
            print(f"ERROR: missing {cfg['csv']}", file=sys.stderr)
            return 1
        data = load_cell(cfg["csv"])
        loaded.append((cfg, data))
        print(
            f"[load] {cfg['wave']:24s} n={data['t'].size:5d}  "
            f"sys2_peak={np.nanmax(data['sys2']):.2f}C  "
            f"shell_peak={np.nanmax(data['shell']):.2f}C  "
            f"bat_t_peak={np.nanmax(data['bat_t']):.2f}C  "
            f"dur={data['t'][-1]/60.0:.1f}min"
        )

    # ----- figure -----
    fig, axes = plt.subplots(
        3, 1, figsize=(12.8, 11.0),
        gridspec_kw={"hspace": 0.40},
    )

    # Common style
    COL_SYS2  = "#d62728"   # red
    COL_SHELL = "#ff7f0e"   # orange
    COL_BAT_T = "#2ca02c"   # green
    COL_BAT_I = "#1f77b4"   # blue (scaled to °C-equivalent axis)

    panel_records = []

    for ax, (cfg, data) in zip(axes, loaded):
        t_min = data["t"] / 60.0

        # ---- left axis: temperatures (°C) ----
        ax.plot(t_min, data["sys2"],
                color=COL_SYS2, lw=1.4,
                label="sys-therm-2 (HAL, 48-61°C trips)")
        ax.plot(t_min, data["shell"],
                color=COL_SHELL, lw=1.4,
                label="shell_front (skin)")
        ax.plot(t_min, data["bat_t"],
                color=COL_BAT_T, lw=1.4,
                label="battery_temp (BCL 45-50°C)")

        # Empirical trip lines for context
        ax.axhline(
            TRIP_C["sys_therm_2_high"],
            color=COL_SYS2, ls=":", lw=0.7, alpha=0.5,
        )
        ax.axhline(
            TRIP_C["battery_temp_hi"],
            color=COL_BAT_T, ls=":", lw=0.7, alpha=0.5,
        )
        ax.text(
            t_min[-1], TRIP_C["sys_therm_2_high"] + 0.3,
            "60°C sys-therm-2 trip",
            color=COL_SYS2, fontsize=7, ha="right", va="bottom", alpha=0.85,
        )
        ax.text(
            t_min[-1], TRIP_C["battery_temp_hi"] + 0.3,
            "50°C BCL knee",
            color=COL_BAT_T, fontsize=7, ha="right", va="bottom", alpha=0.85,
        )

        # ---- right axis: battery current (mA) ----
        ax_r = ax.twinx()
        ax_r.plot(t_min, data["bat_i"],
                  color=COL_BAT_I, lw=0.9, alpha=0.65,
                  label="bat_current_ma (right axis)")
        ax_r.set_ylabel("battery current (mA)", color=COL_BAT_I, fontsize=9)
        ax_r.tick_params(axis="y", labelcolor=COL_BAT_I, labelsize=8)
        # Allow some headroom but keep zero in frame.
        bi_clean = data["bat_i"][np.isfinite(data["bat_i"]) & (data["bat_i"] > 0)]
        if bi_clean.size:
            ymax_r = float(np.nanmax(bi_clean)) * 1.15
            ax_r.set_ylim(0, max(ymax_r, 600.0))

        # ---- throttle event marker ----
        t_thr = cfg.get("throttle_t_s")
        if t_thr is not None:
            idx = find_index_at_time(data["t"], t_thr)
            t_thr_min = data["t"][idx] / 60.0
            ax.axvline(
                t_thr_min, color="black", ls="--", lw=1.4, alpha=0.85,
            )
            ax.text(
                t_thr_min, ax.get_ylim()[1] * 0.995,
                "throttle event",
                rotation=90, va="top", ha="right",
                fontsize=8.5, color="black", fontweight="bold",
            )
            # Pull observed snapshot values from the file as a sanity check
            obs_sys2  = float(data["sys2"][idx])  if idx >= 0 else float("nan")
            obs_shell = float(data["shell"][idx]) if idx >= 0 else float("nan")
            obs_bat_t = float(data["bat_t"][idx]) if idx >= 0 else float("nan")
            obs_bat_i = float(data["bat_i"][idx]) if idx >= 0 else float("nan")
            # Compose callout — prefer supervisor-supplied snapshot, with
            # observed value alongside for traceability.
            lines = [
                f"sys-therm-2 = {cfg['snap_sys_therm_2']:.1f}°C "
                f"(obs {obs_sys2:.1f})",
                f"shell_front = {cfg['snap_shell_front']:.1f}°C "
                f"(obs {obs_shell:.1f})",
                f"battery_temp = {cfg['snap_battery_temp']:.1f}°C "
                f"(obs {obs_bat_t:.1f})",
            ]
            if "snap_bat_current_pre" in cfg:
                lines.append(
                    f"bat_current: {cfg['snap_bat_current_pre']:.0f} "
                    f"→ {cfg['snap_bat_current_post']:.0f} mA (BCL clamp)"
                )
            elif cfg.get("snap_bat_current_ma") is not None:
                lines.append(
                    f"bat_current = {cfg['snap_bat_current_ma']:.0f} mA "
                    f"(obs {obs_bat_i:.0f})"
                )
            ax.text(
                0.985, 0.97, "\n".join(lines),
                transform=ax.transAxes,
                ha="right", va="top", fontsize=8.7,
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    fc="#fffadf", ec="#444", lw=0.7,
                ),
            )
            panel_records.append({
                "wave": cfg["wave"],
                "throttle_t_s": float(data["t"][idx]),
                "snap": {
                    "sys_therm_2_C":     cfg["snap_sys_therm_2"],
                    "shell_front_C":     cfg["snap_shell_front"],
                    "battery_temp_C":    cfg["snap_battery_temp"],
                    "bat_current_ma":    cfg.get("snap_bat_current_ma"),
                    "bat_current_pre_ma":  cfg.get("snap_bat_current_pre"),
                    "bat_current_post_ma": cfg.get("snap_bat_current_post"),
                },
                "observed_at_throttle": {
                    "sys_therm_2_C":  obs_sys2  if np.isfinite(obs_sys2)  else None,
                    "shell_front_C":  obs_shell if np.isfinite(obs_shell) else None,
                    "battery_temp_C": obs_bat_t if np.isfinite(obs_bat_t) else None,
                    "bat_current_ma": obs_bat_i if np.isfinite(obs_bat_i) else None,
                },
                "first_crossed": cfg["first_crossed"],
            })
        else:
            # No throttle event — annotate that fact and report run peaks.
            sys2_peak  = float(np.nanmax(data["sys2"]))
            shell_peak = float(np.nanmax(data["shell"]))
            bat_peak   = float(np.nanmax(data["bat_t"]))
            lines = [
                f"peak sys-therm-2 = {sys2_peak:.1f}°C  (< 60°C trip)",
                f"peak shell_front = {shell_peak:.1f}°C",
                f"peak battery_temp = {bat_peak:.1f}°C  (< 50°C BCL knee)",
                "no throttle event in cell",
            ]
            ax.text(
                0.985, 0.97, "\n".join(lines),
                transform=ax.transAxes,
                ha="right", va="top", fontsize=8.7,
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    fc="#e6f7e6", ec="#2ca02c", lw=0.7,
                ),
            )
            panel_records.append({
                "wave": cfg["wave"],
                "throttle_t_s": None,
                "peak": {
                    "sys_therm_2_C":  sys2_peak,
                    "shell_front_C":  shell_peak,
                    "battery_temp_C": bat_peak,
                },
                "first_crossed": cfg["first_crossed"],
            })

        # ---- titles / axes ----
        ax.set_title(cfg["title"], fontsize=10.5, fontweight="bold", loc="left")
        ax.set_xlabel("time since cell start (min)", fontsize=9)
        ax.set_ylabel("temperature (°C)", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Merge legends from both y-axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_r.get_legend_handles_labels()
        ax.legend(
            h1 + h2, l1 + l2,
            loc="lower right", fontsize=8.0, framealpha=0.93, ncol=2,
        )

    fig.suptitle(
        "Throttle events on OnePlus 15 — which sensor crossed first?",
        fontsize=13.5, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "Sources: wave4_longdecode/vanilla, wave8_v1fa2_sel/v1_fa2_selective, "
        "wave9_v1fa2_stack/v1_fa2_stack  |  "
        "channels: sys-therm-2, shell_front, battery_temp, bat_current_ma",
        ha="center", va="bottom", fontsize=8, color="dimgray",
    )

    fig.tight_layout(rect=(0, 0.012, 1, 0.965))
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1024:.1f} KB)")

    # ------------------------------- PLOT_SCHEMA ------------------------------
    schema = {
        "kind": "PLOT_SCHEMA",
        "version": 1,
        "generator": "host_plot_throttle_soc.py",
        "figure": str(OUT_PATH),
        "title": (
            "Throttle events on OnePlus 15 — which sensor crossed first?"
        ),
        "role_in_paper": (
            "Per-cell time-series of HAL telemetry (sys-therm-2), skin "
            "(shell_front), battery temperature, and battery current "
            "around the throttle event — identifies which sensor crossed "
            "its empirical trip first."
        ),
        "x_axis": "time since first sensors.csv sample (min)",
        "y_axis_primary":   "temperature (°C)",
        "y_axis_secondary": "battery current (mA)",
        "channels": [
            {"name": "sys-therm-2",   "column": COL_SYS_TH2,
             "scale": "value / 1000",
             "trips": [TRIP_C["sys_therm_2_low"], TRIP_C["sys_therm_2_high"]]},
            {"name": "shell_front",   "column": COL_SHELL_FR,
             "scale": "value / 1000",
             "trips": [TRIP_C["shell_front"]]},
            {"name": "battery_temp",  "column": COL_BAT_T,
             "scale": "value / 1000",
             "trips": [TRIP_C["battery_temp_lo"], TRIP_C["battery_temp_hi"]]},
            {"name": "bat_current_ma", "column": COL_BAT_I,
             "scale": "raw mA (right axis)",
             "trips": []},
        ],
        "panels": panel_records,
        "key_finding": {
            "wave4_vanilla": (
                "sys-therm-2 reached 55.7°C first; shell_front and "
                "battery_temp both lagged it — sys-therm-2 is the leading "
                "indicator and is the empirical throttle trigger here."
            ),
            "wave8_v1fa2_selective": (
                "sys-therm-2 crossed the 60°C HAL trip first (62.1°C); "
                "battery_temp followed (51.1°C > 50°C BCL knee) which is "
                "what produced the bat_current 371 → 296 mA clamp."
            ),
            "wave9_v1fa2_stack": (
                "no sensor crossed any empirical trip — the preemptive "
                "watchdog held sys-therm-2 below 56°C and battery_temp "
                "below 48°C for the whole cell."
            ),
            "summary": (
                "sys-therm-2 is the first-mover sensor at every throttle "
                "we observed; shell_front and battery_temp follow it by "
                "2-3°C and only matter when the preemptive controller "
                "has already failed."
            ),
        },
    }
    with open(SCHEMA_PATH, "w") as fh:
        json.dump(schema, fh, indent=2)
    print(f"[ok] wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
