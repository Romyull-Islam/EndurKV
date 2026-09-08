#!/usr/bin/env python3
"""
PLOT 4: K-sweep ablation -- cache budget -> DDR/CPU temperature curves.

Data: Wave-10 K-sweep at phone-logs/wave10_ksweep_1780815847/
Each K subdirectory contains sensors.csv (high-rate thermals) and
stress.csv (per-iter llama-cli summary). The shared watchdog.log
contains every DDR-driven DVFS tier transition with unix-second
timestamps so we can slice tier-1/tier-2 hits per K-phase by time.

NOTE: The task lists K=256/384/512/1024, but the Wave-10 run on
disk only contains 256/384/1024 (K=512 was skipped on the device).
We plot the three K-values that were actually run.

3-panel figure:
  Panel 1: K vs peak DDR (bar, error bars from per-iter ddr_max)
  Panel 2: K vs peak CPU (bar, error bars from per-iter cpu_max)
  Panel 3: K vs mean decode_tps (from stress.csv)

Each bar annotated with K-value and tier-1 transition count from
watchdog.log restricted to that K's time window.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WAVE_DIR = Path("/home/mislam22/EndurKV_workspace/phone-logs/wave10_ksweep_1780815847")
OUT_PATH = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/04_ksweep_thermal_curves.png"
)

K_VALUES = [256, 384, 1024]  # K=512 not present in this Wave-10 run.
K_DIRS = {k: WAVE_DIR / f"K{k}" for k in K_VALUES}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sensors(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["wall_clock_s"] = pd.to_numeric(df["wall_clock_s"], errors="coerce")
    df = df.dropna(subset=["wall_clock_s"])
    # DDR temp (already a single column).
    df["ddr_c"] = pd.to_numeric(df["ddr_temp_mc"], errors="coerce") / 1000.0
    # CPU peak across every cpu-* zone (skip cpullc, cpu-hw-trip).
    cpu_cols = [
        c for c in df.columns
        if re.match(r"^cpu-\d-\d-\d_temp_mc$", c)
    ]
    cpu_arr = df[cpu_cols].apply(pd.to_numeric, errors="coerce").to_numpy() / 1000.0
    df["cpu_max_c"] = np.nanmax(cpu_arr, axis=1)
    return df[["wall_clock_s", "ddr_c", "cpu_max_c"]].copy()


def load_stress(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def per_iter_peaks(sensors: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    """For each iter row, slice sensors by wall-clock window [t0, t1] and
    return that iter's peak DDR and peak CPU temperature in C.

    iter timing comes from stress.csv via t_elapsed_s anchored at the
    first sensors wall_clock_s in the run (close enough -- iter 1 starts
    a few seconds after sensor capture begins; we use the iter-end of
    the previous row as the window start)."""
    if sensors.empty or stress.empty:
        return pd.DataFrame(columns=["iter", "ddr_peak_c", "cpu_peak_c"])
    t_origin = float(sensors["wall_clock_s"].iloc[0])
    starts = stress["t_elapsed_s"].astype(float).to_numpy()
    # next iter's start = end of current iter; final iter ends at last sample.
    last_t = float(sensors["wall_clock_s"].iloc[-1]) - t_origin
    ends = np.concatenate([starts[1:], [last_t]])
    rows = []
    for i, (s, e) in enumerate(zip(starts, ends), start=1):
        mask = (sensors["wall_clock_s"] >= t_origin + s) & (
            sensors["wall_clock_s"] < t_origin + e
        )
        sub = sensors.loc[mask]
        if sub.empty:
            continue
        rows.append({
            "iter": i,
            "ddr_peak_c": float(np.nanmax(sub["ddr_c"])),
            "cpu_peak_c": float(np.nanmax(sub["cpu_max_c"])),
        })
    return pd.DataFrame(rows)


WATCHDOG_RE = re.compile(
    r"^\[(?P<ts>\d+)\]\s+DDR=(?P<ddr>\d+)C\s+->\s+tier=(?P<tier>\d+)"
)

def parse_watchdog(log_path: Path):
    """Return list of (unix_ts:int, tier:int) for every tier transition."""
    events = []
    with open(log_path) as fh:
        for line in fh:
            m = WATCHDOG_RE.match(line)
            if not m:
                continue
            events.append((int(m["ts"]), int(m["tier"])))
    return events


def tier_counts_in_window(events, t0: float, t1: float):
    """Count tier-1, tier-2 transitions that fall in [t0, t1]."""
    n1 = sum(1 for ts, tier in events if t0 <= ts <= t1 and tier == 1)
    n2 = sum(1 for ts, tier in events if t0 <= ts <= t1 and tier == 2)
    return n1, n2


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def aggregate():
    events = parse_watchdog(WAVE_DIR / "watchdog.log")
    summary = []
    for k in K_VALUES:
        d = K_DIRS[k]
        sensors = load_sensors(d / "sensors.csv")
        stress = load_stress(d / "stress.csv")
        peaks = per_iter_peaks(sensors, stress)
        t0 = float(sensors["wall_clock_s"].iloc[0])
        t1 = float(sensors["wall_clock_s"].iloc[-1])
        n_tier1, n_tier2 = tier_counts_in_window(events, t0, t1)
        summary.append({
            "K": k,
            "n_iter": len(peaks),
            "ddr_peak_mean": peaks["ddr_peak_c"].mean(),
            "ddr_peak_std": peaks["ddr_peak_c"].std(ddof=0),
            "ddr_peak_max": peaks["ddr_peak_c"].max(),
            "cpu_peak_mean": peaks["cpu_peak_c"].mean(),
            "cpu_peak_std": peaks["cpu_peak_c"].std(ddof=0),
            "cpu_peak_max": peaks["cpu_peak_c"].max(),
            "mean_decode_tps": float(stress["decode_tps"].mean()),
            "std_decode_tps": float(stress["decode_tps"].std(ddof=0)),
            "tier1_transitions": n_tier1,
            "tier2_transitions": n_tier2,
            "t_start_unix": t0,
            "t_end_unix": t1,
        })
    return pd.DataFrame(summary)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(df: pd.DataFrame):
    df = df.sort_values("K").reset_index(drop=True)
    x_labels = [str(k) for k in df["K"]]
    x_pos = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    fig.suptitle(
        "K-budget ablation: smaller cache -> cooler chip + more eviction events",
        fontsize=14, fontweight="bold",
    )

    # Palette: warm-to-cool from largest-K to smallest-K (bigger cache = hotter).
    color_map = {256: "#2c7fb8", 384: "#7fbf7b", 1024: "#d7301f"}
    bar_colors = [color_map[k] for k in df["K"]]

    # ---------- Panel 1: peak DDR --------------------------------------
    ax = axes[0]
    bars = ax.bar(
        x_pos, df["ddr_peak_mean"], yerr=df["ddr_peak_std"],
        capsize=6, color=bar_colors, edgecolor="black", linewidth=0.6,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("K (recent-token budget, log-spaced)")
    ax.set_ylabel("Peak DDR temperature (degC)  per-iter mean +/- s.d.")
    ax.set_title("Panel 1: peak DDR vs K")
    ax.grid(True, axis="y", alpha=0.3)
    # Annotate K + tier-1 transitions over each bar.
    for bar, (_, row) in zip(bars, df.iterrows()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + (row["ddr_peak_std"] if not np.isnan(row["ddr_peak_std"]) else 0) + 0.4,
            f"K={int(row['K'])}\n{int(row['tier1_transitions'])} tier-1\n"
            f"{int(row['tier2_transitions'])} tier-2",
            ha="center", va="bottom", fontsize=9,
        )
    # Pad y-axis so the annotations fit.
    ymax = (df["ddr_peak_mean"] + df["ddr_peak_std"].fillna(0)).max()
    ax.set_ylim(0, ymax * 1.22)

    # ---------- Panel 2: peak CPU --------------------------------------
    ax = axes[1]
    bars = ax.bar(
        x_pos, df["cpu_peak_mean"], yerr=df["cpu_peak_std"],
        capsize=6, color=bar_colors, edgecolor="black", linewidth=0.6,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("K (recent-token budget, log-spaced)")
    ax.set_ylabel("Peak CPU temperature (degC)  per-iter mean +/- s.d.")
    ax.set_title("Panel 2: peak CPU vs K")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, (_, row) in zip(bars, df.iterrows()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + (row["cpu_peak_std"] if not np.isnan(row["cpu_peak_std"]) else 0) + 0.4,
            f"K={int(row['K'])}\n{int(row['tier1_transitions'])} tier-1",
            ha="center", va="bottom", fontsize=9,
        )
    ymax = (df["cpu_peak_mean"] + df["cpu_peak_std"].fillna(0)).max()
    ax.set_ylim(0, ymax * 1.22)

    # ---------- Panel 3: decode tps (context) --------------------------
    ax = axes[2]
    bars = ax.bar(
        x_pos, df["mean_decode_tps"], yerr=df["std_decode_tps"],
        capsize=6, color=bar_colors, edgecolor="black", linewidth=0.6,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("K (recent-token budget, log-spaced)")
    ax.set_ylabel("Mean decode throughput (tok/s)  across iters")
    ax.set_title("Panel 3: mean decode tps vs K (context)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, (_, row) in zip(bars, df.iterrows()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + (row["std_decode_tps"] if not np.isnan(row["std_decode_tps"]) else 0) + 0.06,
            f"K={int(row['K'])}\n{row['mean_decode_tps']:.2f} tok/s",
            ha="center", va="bottom", fontsize=9,
        )
    ymax = (df["mean_decode_tps"] + df["std_decode_tps"].fillna(0)).max()
    ax.set_ylim(0, ymax * 1.25)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)


def main():
    df = aggregate()
    # Print summary so the CLI run is auditable.
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("[wave10 K-sweep summary]")
    print(df.to_string(index=False))
    plot(df)
    print(f"[wrote] {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
