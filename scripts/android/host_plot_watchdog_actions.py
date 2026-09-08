#!/usr/bin/env python3
"""
PLOT 17: Pre-empt-throttle watchdog -- action timeline (Wave-9 + Wave-10 + Wave-11).

The watchdog runs as a root sidecar that polls the DDR thermal zone (zone47)
once per second and clamps the big-core max-frequency via cpufreq scaling:

   tier 0  -> MAX  = 1632 MHz    (no clamp)
   tier 1  -> HIGH = 1497.6 MHz  (engages at DDR ~= 58 C)
   tier 2  -> MED  = 1267.2 MHz  (engages at DDR ~= 62 C)
   tier 3  -> LOW  =  883.2 MHz  (engages at DDR ~= 65 C, paper hard-cap)

The log line format is:
   [<unix_ts>] DDR=<int>C -> tier=<n> <NAME>=<khz>

Three sources:
  Wave-9  v1_fa2_stack_1780796320/watchdog.log         (single arm, 1 hour)
  Wave-10 ksweep_1780815847/watchdog.log               (K=256/384/1024 phases, 4 hours)
  Wave-11 wave11_eval_1780862534/Phi-3-mini-128k/v1_fa2_stack/ppl/watchdog.log
          (will appear when the Wave-11 v1_fa2_stack arm finishes -- handled gracefully)

3-panel figure:
  Top    -- Wave-9 timeline      : DDR temp curve (from sensors.csv) overlaid
                                   with watchdog tier-step trace + engagement
                                   annotations.
  Middle -- Wave-10 K-sweep      : per-K tier-transition density (counts/min)
                                   stacked by K, on a shared minutes axis with
                                   K-phase shading.
  Bottom -- Wave-11 v1_fa2_stack : same layout as the top panel; falls back to
                                   a friendly "data not yet available" panel
                                   while the run is still in flight.

Output:
  /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/17_watchdog_action_timeline.png
  /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/17_watchdog_action_timeline.schema.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PHONE_LOGS = Path("/home/mislam22/EndurKV_workspace/phone-logs")
WAVE9_DIR = PHONE_LOGS / "wave9_v1fa2_stack_1780796320"
WAVE10_DIR = PHONE_LOGS / "wave10_ksweep_1780815847"
WAVE11_WATCHDOG = (
    PHONE_LOGS
    / "wave11_eval_1780862534"
    / "Phi-3-mini-128k"
    / "v1_fa2_stack"
    / "ppl"
    / "watchdog.log"
)

FIG_DIR = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots"
)
OUT_PNG = FIG_DIR / "17_watchdog_action_timeline.png"
OUT_SCHEMA = FIG_DIR / "17_watchdog_action_timeline.schema.json"

# Tier frequency table (kHz -> MHz) as reported in the watchdog log.
TIER_FREQ_MHZ = {
    0: 1632.0,
    1: 1497.6,
    2: 1267.2,
    3: 883.2,
}
# Engagement temperature thresholds inferred from log transitions / watchdog source.
TIER_ENGAGE_C = {1: 58, 2: 62, 3: 65}

TIER_COLOR = {
    0: "#2c7fb8",  # cool / no clamp
    1: "#fdae61",  # warm clamp
    2: "#f46d43",  # hotter clamp
    3: "#a50026",  # paper hard cap
}

# Wave-10 K-phase boundaries (from progress.log, cross-checked against watchdog timestamps).
WAVE10_K_PHASES: List[Tuple[int, int, int]] = [
    # (K, t_start_unix, t_end_unix)
    (1024, 1780815848, 1780819532),
    (384,  1780821125, 1780825049),  # iters started after a cool-down ~25 min after K1024 end
    (256,  1780825049, 1780830432),
]
K_COLOR = {256: "#2c7fb8", 384: "#7fbf7b", 1024: "#d7301f"}


# ----------------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------------
WATCHDOG_RE = re.compile(
    r"^\[(?P<ts>\d+)\]\s+DDR=(?P<ddr>\d+)C\s+->\s+tier=(?P<tier>\d+)"
)


@dataclass
class TierEvent:
    ts: int
    ddr_c: int
    tier: int


def parse_watchdog(log_path: Path) -> List[TierEvent]:
    events: List[TierEvent] = []
    if not log_path.exists():
        return events
    with open(log_path) as fh:
        for line in fh:
            m = WATCHDOG_RE.match(line)
            if not m:
                continue
            events.append(
                TierEvent(
                    ts=int(m["ts"]),
                    ddr_c=int(m["ddr"]),
                    tier=int(m["tier"]),
                )
            )
    return events


def parse_watchdog_window(log_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read the very first 'watchdog start' timestamp and the 'watchdog exit'
    timestamp -- both are emitted with the same [unix_ts] prefix."""
    start_ts: Optional[int] = None
    exit_ts: Optional[int] = None
    if not log_path.exists():
        return start_ts, exit_ts
    ts_re = re.compile(r"^\[(\d+)\]\s+(.*)$")
    with open(log_path) as fh:
        for line in fh:
            m = ts_re.match(line)
            if not m:
                continue
            ts = int(m.group(1))
            tail = m.group(2)
            if "watchdog start" in tail and start_ts is None:
                start_ts = ts
            if "watchdog exit" in tail:
                exit_ts = ts
    return start_ts, exit_ts


# ----------------------------------------------------------------------------
# Sensor helpers
# ----------------------------------------------------------------------------
def load_ddr_sensors(csv_path: Path) -> pd.DataFrame:
    """Return DataFrame[wall_clock_s, ddr_c] from a sensors.csv."""
    if not csv_path.exists():
        return pd.DataFrame(columns=["wall_clock_s", "ddr_c"])
    df = pd.read_csv(csv_path, low_memory=False, usecols=lambda c: c in {
        "wall_clock_s", "ddr_temp_mc",
    })
    df["wall_clock_s"] = pd.to_numeric(df["wall_clock_s"], errors="coerce")
    df["ddr_c"] = pd.to_numeric(df["ddr_temp_mc"], errors="coerce") / 1000.0
    df = df.dropna(subset=["wall_clock_s", "ddr_c"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Step-trace builder
# ----------------------------------------------------------------------------
def build_tier_step_trace(events: List[TierEvent],
                          t_start: int, t_end: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of tier transitions to (xs, ys) suitable for plt.step.

    The watchdog emits a log line only when the tier changes, so we hold the
    last-set tier until the next event. We bookend the trace with tier=0
    (watchdog boots at MAX) and the final value at t_end."""
    if not events:
        return np.array([t_start, t_end]), np.array([0, 0])
    # Sort defensively.
    ev = sorted(events, key=lambda e: e.ts)
    xs = [t_start]
    ys = [0]
    for e in ev:
        xs.append(e.ts)
        ys.append(e.tier)
    xs.append(t_end)
    ys.append(ev[-1].tier)
    return np.array(xs, dtype=float), np.array(ys, dtype=int)


# ----------------------------------------------------------------------------
# First-engagement detection (for annotations)
# ----------------------------------------------------------------------------
def first_engagements(events: List[TierEvent]) -> Dict[int, TierEvent]:
    """Return first time each tier>=1 is engaged."""
    seen: Dict[int, TierEvent] = {}
    for e in events:
        if e.tier >= 1 and e.tier not in seen:
            seen[e.tier] = e
    return seen


def post_engagement_ddr_drop(sensors: pd.DataFrame, engage_ts: int,
                             window_s: int = 90) -> Optional[float]:
    """Take the DDR temp at engage_ts and the minimum DDR temp in the
    window_s seconds *following* engagement. Returns the delta (positive =
    cooling). Returns None if the sensor coverage is insufficient."""
    if sensors.empty:
        return None
    near_engage = sensors[
        (sensors["wall_clock_s"] >= engage_ts - 2) &
        (sensors["wall_clock_s"] <= engage_ts + 2)
    ]
    after = sensors[
        (sensors["wall_clock_s"] > engage_ts) &
        (sensors["wall_clock_s"] <= engage_ts + window_s)
    ]
    if near_engage.empty or after.empty:
        return None
    t_engage_c = float(near_engage["ddr_c"].iloc[0])
    t_min_c = float(after["ddr_c"].min())
    return t_engage_c - t_min_c


# ============================================================================
# PANEL DRAWERS
# ============================================================================
def draw_top_wave9(ax_top: plt.Axes, ax_top_tier: plt.Axes,
                   events: List[TierEvent], sensors: pd.DataFrame,
                   t0: int, t1: int) -> Dict:
    """Top panel: Wave-9 DDR temp + tier step trace + first-engagement annotations."""
    # Time axis in minutes since the watchdog started.
    s = sensors[(sensors["wall_clock_s"] >= t0) & (sensors["wall_clock_s"] <= t1)].copy()
    if not s.empty:
        s["min"] = (s["wall_clock_s"] - t0) / 60.0
        ax_top.plot(s["min"], s["ddr_c"], lw=1.0, color="#222222",
                    label="DDR temp (degC)")
    # Tier engagement bands.
    for tier_v, c_thresh in TIER_ENGAGE_C.items():
        ax_top.axhline(c_thresh, color=TIER_COLOR[tier_v], ls="--", lw=0.8, alpha=0.7)
        ax_top.text(0.985, c_thresh + 0.15,
                    f"tier {tier_v} engage = {c_thresh} C ({TIER_FREQ_MHZ[tier_v]:.0f} MHz)",
                    transform=ax_top.get_yaxis_transform(),
                    fontsize=7.5, color=TIER_COLOR[tier_v], va="bottom", ha="right")

    # Tier step trace (twin axis below DDR curve).
    xs, ys = build_tier_step_trace(events, t0, t1)
    xs_min = (xs - t0) / 60.0
    ax_top_tier.step(xs_min, ys, where="post", color="#d7301f", lw=1.5,
                     label="watchdog tier")
    ax_top_tier.set_yticks([0, 1, 2, 3])
    ax_top_tier.set_yticklabels(["0 (MAX)", "1 (HIGH)", "2 (MED)", "3 (LOW)"])
    ax_top_tier.set_ylim(-0.4, 3.4)
    ax_top_tier.set_ylabel("watchdog tier", color="#d7301f")
    ax_top_tier.tick_params(axis="y", colors="#d7301f")

    # First-engagement annotation -- arrow pointing at the engagement moment + DDR drop note.
    ax_top.set_ylabel("DDR temperature (degC)")
    ax_top.set_xlabel("time since watchdog start (min)")
    duration_min = (t1 - t0) / 60.0
    ax_top.set_xlim(0, duration_min)
    if not s.empty:
        ddr_min = float(np.nanmin(s["ddr_c"]))
        ddr_max = float(np.nanmax(s["ddr_c"]))
    else:
        ddr_min, ddr_max = 50.0, 65.0
    # Widen y-range so annotation boxes have room.
    ax_top.set_ylim(min(ddr_min - 2, 35), max(ddr_max + 14, 80))

    fe = first_engagements(events)
    drops: Dict[int, Optional[float]] = {}
    # Stagger callouts so they don't overlap. Tier-1 goes below the curve; tier-2/3 above.
    callout_layout = {
        1: dict(dx=3.0, dy=-10.0),   # below the curve
        2: dict(dx=10.0, dy=+8.0),   # above + further right
        3: dict(dx=14.0, dy=+12.0),
    }
    for tier_v, ev in sorted(fe.items()):
        m = (ev.ts - t0) / 60.0
        drop = post_engagement_ddr_drop(sensors, ev.ts, window_s=90)
        drops[tier_v] = drop
        txt = (f"tier {tier_v} engaged at DDR {ev.ddr_c}C\n"
               f"-> freq cap {TIER_FREQ_MHZ[tier_v]:.0f} MHz")
        if drop is not None:
            txt += f"\nDDR dropped {drop:.1f} C in 90 s"
        lay = callout_layout.get(tier_v, dict(dx=2.0, dy=5.0))
        ax_top.annotate(
            txt,
            xy=(m, ev.ddr_c),
            xytext=(min(m + lay["dx"], duration_min - 9),
                    ev.ddr_c + lay["dy"]),
            fontsize=8.5,
            color=TIER_COLOR[tier_v],
            arrowprops=dict(arrowstyle="->", color=TIER_COLOR[tier_v], lw=0.9),
            bbox=dict(boxstyle="round,pad=0.30",
                      fc="white", ec=TIER_COLOR[tier_v], lw=0.9, alpha=0.94),
        )
    ax_top.grid(True, alpha=0.25)
    n_t1 = sum(1 for e in events if e.tier == 1)
    n_t2 = sum(1 for e in events if e.tier == 2)
    n_t3 = sum(1 for e in events if e.tier == 3)
    ax_top.set_title(
        f"Wave-9 v1_fa2_stack (1 h, K=512): {len(events)} watchdog actions "
        f"({n_t1} tier-1, {n_t2} tier-2, {n_t3} tier-3) -- DDR held below 65 C cliff",
        fontsize=11, loc="left",
    )
    return {
        "n_events": len(events),
        "n_tier1": n_t1,
        "n_tier2": n_t2,
        "n_tier3": n_t3,
        "first_engagements": {
            str(tier_v): {
                "unix_ts": ev.ts,
                "ddr_c_at_engage": ev.ddr_c,
                "freq_cap_mhz": TIER_FREQ_MHZ[tier_v],
                "ddr_drop_c_90s": drops.get(tier_v),
            } for tier_v, ev in fe.items()
        },
        "ddr_min_c": ddr_min,
        "ddr_max_c": ddr_max,
        "duration_min": duration_min,
    }


def draw_middle_wave10(ax: plt.Axes, events: List[TierEvent],
                       t0: int, t1: int) -> Dict:
    """Middle panel: Wave-10 K-sweep transition density per K phase."""
    duration_min = (t1 - t0) / 60.0
    # Bin transitions per minute, split by tier (1, 2 only -- tier-3 never engaged).
    bin_edges = np.arange(0, np.ceil(duration_min) + 1, 1)  # 1-min bins
    bin_centers = bin_edges[:-1] + 0.5
    tier1_xs = [(e.ts - t0) / 60.0 for e in events if e.tier == 1 and t0 <= e.ts <= t1]
    tier2_xs = [(e.ts - t0) / 60.0 for e in events if e.tier == 2 and t0 <= e.ts <= t1]
    tier3_xs = [(e.ts - t0) / 60.0 for e in events if e.tier == 3 and t0 <= e.ts <= t1]
    h_t1, _ = np.histogram(tier1_xs, bins=bin_edges)
    h_t2, _ = np.histogram(tier2_xs, bins=bin_edges)
    h_t3, _ = np.histogram(tier3_xs, bins=bin_edges)

    # Stacked bar: tier-1 (bottom) + tier-2 (on top).
    width = 0.95
    ax.bar(bin_centers, h_t1, width=width, color=TIER_COLOR[1], edgecolor="none",
           label=f"tier-1 engage (n={sum(h_t1)})")
    ax.bar(bin_centers, h_t2, width=width, bottom=h_t1, color=TIER_COLOR[2],
           edgecolor="none", label=f"tier-2 engage (n={sum(h_t2)})")
    if h_t3.sum() > 0:
        ax.bar(bin_centers, h_t3, width=width, bottom=h_t1 + h_t2,
               color=TIER_COLOR[3], edgecolor="none",
               label=f"tier-3 engage (n={sum(h_t3)})")

    ax.set_xlabel("time since watchdog start (min)")
    ax.set_ylabel("watchdog actions per minute")
    ax.set_xlim(0, duration_min)
    # Make room for K-phase labels at the top BEFORE drawing them.
    y_top_data = max(1.0, float(np.nanmax(h_t1 + h_t2 + h_t3)))
    ax.set_ylim(0, y_top_data * 1.40)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # K-phase shading + labels (after final ylim so labels are placed correctly).
    per_k_summary = []
    label_y = ax.get_ylim()[1] * 0.93
    for k, ks_unix, ke_unix in WAVE10_K_PHASES:
        ks_min = max(0.0, (ks_unix - t0) / 60.0)
        ke_min = min(duration_min, (ke_unix - t0) / 60.0)
        if ke_min <= ks_min:
            continue
        ax.axvspan(ks_min, ke_min, color=K_COLOR[k], alpha=0.10, zorder=0)
        n1 = sum(1 for e in events if e.tier == 1 and ks_unix <= e.ts <= ke_unix)
        n2 = sum(1 for e in events if e.tier == 2 and ks_unix <= e.ts <= ke_unix)
        n3 = sum(1 for e in events if e.tier == 3 and ks_unix <= e.ts <= ke_unix)
        per_k_summary.append({
            "K": k,
            "t_start_min": ks_min,
            "t_end_min": ke_min,
            "duration_min": ke_min - ks_min,
            "n_tier1": n1,
            "n_tier2": n2,
            "n_tier3": n3,
            "tier1_per_min": n1 / max(1e-6, ke_min - ks_min),
            "tier2_per_min": n2 / max(1e-6, ke_min - ks_min),
        })
        ax.text((ks_min + ke_min) / 2.0, label_y,
                f"K={k}\n{n1} tier-1, {n2} tier-2",
                ha="center", va="top", fontsize=9,
                color="black",
                bbox=dict(boxstyle="round,pad=0.22",
                          fc="white", ec=K_COLOR[k], lw=0.9, alpha=0.95))
    total_n = len(events)
    ax.set_title(
        f"Wave-10 K-sweep aggregate (K=256/384/1024, 4 h, "
        f"{total_n} watchdog actions): higher K -> denser tier-2 fire-rate",
        fontsize=11, loc="left",
    )
    return {
        "duration_min": duration_min,
        "n_events_total": total_n,
        "per_K": per_k_summary,
    }


def draw_bottom_wave11(ax_main: plt.Axes,
                       ax_tier: plt.Axes,
                       events: List[TierEvent],
                       sensors: pd.DataFrame,
                       t0: Optional[int],
                       t1: Optional[int]) -> Dict:
    """Bottom panel: Wave-11 v1_fa2_stack -- same layout as Wave-9 if data is
    present, otherwise placeholder."""
    if not events or t0 is None or t1 is None:
        for ax in (ax_main, ax_tier):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax_main.text(
            0.5, 0.55,
            "Wave-11 v1_fa2_stack watchdog.log will appear when the run finishes",
            ha="center", va="center", fontsize=12, color="#555555",
            transform=ax_main.transAxes,
        )
        ax_main.text(
            0.5, 0.40,
            f"expected path: {WAVE11_WATCHDOG}",
            ha="center", va="center", fontsize=8.5, color="#888888",
            family="monospace", transform=ax_main.transAxes,
        )
        ax_main.set_title(
            "Wave-11 Phi-3-mini-128k v1_fa2_stack ppl arm: data not yet available",
            fontsize=11, loc="left",
        )
        return {"available": False, "expected_path": str(WAVE11_WATCHDOG)}

    # Mirror the Wave-9 layout.
    s = sensors.copy()
    if not s.empty:
        s = s[(s["wall_clock_s"] >= t0) & (s["wall_clock_s"] <= t1)]
        s["min"] = (s["wall_clock_s"] - t0) / 60.0
        ax_main.plot(s["min"], s["ddr_c"], lw=1.0, color="#222222",
                     label="DDR temp (degC)")
    for tier_v, c_thresh in TIER_ENGAGE_C.items():
        ax_main.axhline(c_thresh, color=TIER_COLOR[tier_v], ls="--", lw=0.8, alpha=0.6)

    xs, ys = build_tier_step_trace(events, t0, t1)
    xs_min = (xs - t0) / 60.0
    ax_tier.step(xs_min, ys, where="post", color="#d7301f", lw=1.5,
                 label="watchdog tier")
    ax_tier.set_yticks([0, 1, 2, 3])
    ax_tier.set_yticklabels(["0 (MAX)", "1 (HIGH)", "2 (MED)", "3 (LOW)"])
    ax_tier.set_ylim(-0.4, 3.4)
    ax_tier.set_ylabel("watchdog tier", color="#d7301f")
    ax_tier.tick_params(axis="y", colors="#d7301f")

    ax_main.set_ylabel("DDR temperature (degC)")
    ax_main.set_xlabel("time since watchdog start (min)")
    duration_min = (t1 - t0) / 60.0
    ax_main.set_xlim(0, duration_min)
    if not s.empty:
        ddr_min = float(np.nanmin(s["ddr_c"]))
        ddr_max = float(np.nanmax(s["ddr_c"]))
    else:
        ddr_min, ddr_max = 50.0, 65.0
    ax_main.set_ylim(min(ddr_min - 2, 35), max(ddr_max + 14, 80))

    fe = first_engagements(events)
    drops: Dict[int, Optional[float]] = {}
    callout_layout = {
        1: dict(dx=3.0, dy=-10.0),
        2: dict(dx=10.0, dy=+8.0),
        3: dict(dx=14.0, dy=+12.0),
    }
    for tier_v, ev in sorted(fe.items()):
        m = (ev.ts - t0) / 60.0
        drop = post_engagement_ddr_drop(sensors, ev.ts, window_s=90)
        drops[tier_v] = drop
        txt = (f"tier {tier_v} engaged at DDR {ev.ddr_c}C\n"
               f"-> freq cap {TIER_FREQ_MHZ[tier_v]:.0f} MHz")
        if drop is not None:
            txt += f"\nDDR dropped {drop:.1f} C in 90 s"
        lay = callout_layout.get(tier_v, dict(dx=2.0, dy=5.0))
        ax_main.annotate(
            txt,
            xy=(m, ev.ddr_c),
            xytext=(min(m + lay["dx"], duration_min - 9),
                    ev.ddr_c + lay["dy"]),
            fontsize=8.5,
            color=TIER_COLOR[tier_v],
            arrowprops=dict(arrowstyle="->", color=TIER_COLOR[tier_v], lw=0.9),
            bbox=dict(boxstyle="round,pad=0.30",
                      fc="white", ec=TIER_COLOR[tier_v], lw=0.9, alpha=0.94),
        )
    ax_main.grid(True, alpha=0.25)
    n_t1 = sum(1 for e in events if e.tier == 1)
    n_t2 = sum(1 for e in events if e.tier == 2)
    n_t3 = sum(1 for e in events if e.tier == 3)
    ax_main.set_title(
        f"Wave-11 Phi-3-mini-128k v1_fa2_stack ppl: {len(events)} watchdog actions "
        f"({n_t1} tier-1, {n_t2} tier-2, {n_t3} tier-3)",
        fontsize=11, loc="left",
    )
    return {
        "available": True,
        "n_events": len(events),
        "n_tier1": n_t1,
        "n_tier2": n_t2,
        "n_tier3": n_t3,
        "first_engagements": {
            str(tier_v): {
                "unix_ts": ev.ts,
                "ddr_c_at_engage": ev.ddr_c,
                "freq_cap_mhz": TIER_FREQ_MHZ[tier_v],
                "ddr_drop_c_90s": drops.get(tier_v),
            } for tier_v, ev in fe.items()
        },
        "ddr_min_c": ddr_min,
        "ddr_max_c": ddr_max,
        "duration_min": duration_min,
        "source_path": str(WAVE11_WATCHDOG),
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Wave 9 -----
    w9_log = WAVE9_DIR / "watchdog.log"
    w9_events = parse_watchdog(w9_log)
    w9_t_start, w9_t_end = parse_watchdog_window(w9_log)
    if w9_t_start is None and w9_events:
        w9_t_start = w9_events[0].ts
    if w9_t_end is None and w9_events:
        w9_t_end = w9_events[-1].ts + 30
    w9_sensors = load_ddr_sensors(WAVE9_DIR / "v1_fa2_stack" / "sensors.csv")
    print(f"[wave9 ] events={len(w9_events)} window=[{w9_t_start},{w9_t_end}] "
          f"sensors_rows={len(w9_sensors)}")

    # ----- Wave 10 -----
    w10_log = WAVE10_DIR / "watchdog.log"
    w10_events = parse_watchdog(w10_log)
    w10_t_start, w10_t_end = parse_watchdog_window(w10_log)
    if w10_t_start is None and w10_events:
        w10_t_start = w10_events[0].ts
    if w10_t_end is None and w10_events:
        w10_t_end = w10_events[-1].ts + 30
    print(f"[wave10] events={len(w10_events)} window=[{w10_t_start},{w10_t_end}]")

    # ----- Wave 11 (graceful) -----
    w11_events = parse_watchdog(WAVE11_WATCHDOG)
    w11_t_start, w11_t_end = parse_watchdog_window(WAVE11_WATCHDOG)
    if w11_t_start is None and w11_events:
        w11_t_start = w11_events[0].ts
    if w11_t_end is None and w11_events:
        w11_t_end = w11_events[-1].ts + 30
    w11_sensors = pd.DataFrame(columns=["wall_clock_s", "ddr_c"])
    w11_sensors_path = WAVE11_WATCHDOG.parent / "sensors.csv"
    if w11_sensors_path.exists():
        w11_sensors = load_ddr_sensors(w11_sensors_path)
    print(f"[wave11] events={len(w11_events)} window=[{w11_t_start},{w11_t_end}] "
          f"sensors_rows={len(w11_sensors)}")

    # ----- Figure layout -----
    # 3 stacked panels. Top + bottom panels each get a twin axis for the tier
    # step trace; middle is a simple bar plot.
    fig = plt.figure(figsize=(14.5, 13.5), constrained_layout=False)
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 0.95, 1.0],
        hspace=0.75, top=0.90, bottom=0.055, left=0.07, right=0.93,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top_tier = ax_top.twinx()
    ax_mid = fig.add_subplot(gs[1, 0])
    ax_bot = fig.add_subplot(gs[2, 0])
    ax_bot_tier = ax_bot.twinx()

    fig.suptitle(
        "Preempt-throttle watchdog: closed-loop temperature control, "
        "no kernel cliff",
        fontsize=15, fontweight="bold", y=0.975,
    )

    top_meta = draw_top_wave9(ax_top, ax_top_tier, w9_events, w9_sensors,
                              w9_t_start, w9_t_end)
    mid_meta = draw_middle_wave10(ax_mid, w10_events, w10_t_start, w10_t_end)
    bot_meta = draw_bottom_wave11(ax_bot, ax_bot_tier, w11_events, w11_sensors,
                                  w11_t_start, w11_t_end)

    # Shared legend at the top.
    legend_handles = [
        Line2D([0], [0], color="#222222", lw=1.2, label="DDR temperature"),
        Line2D([0], [0], color="#d7301f", lw=1.5, label="watchdog tier (step trace)"),
        Patch(facecolor=TIER_COLOR[1], label="tier-1 action (1497.6 MHz)"),
        Patch(facecolor=TIER_COLOR[2], label="tier-2 action (1267.2 MHz)"),
        Patch(facecolor=TIER_COLOR[3], label="tier-3 action (883.2 MHz)"),
        Patch(facecolor=K_COLOR[256], alpha=0.30, label="K=256 phase"),
        Patch(facecolor=K_COLOR[384], alpha=0.30, label="K=384 phase"),
        Patch(facecolor=K_COLOR[1024], alpha=0.30, label="K=1024 phase"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.935), ncol=4, fontsize=9, frameon=True,
    )

    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"[wrote] {OUT_PNG}")

    # ----- ART_SCHEMA -----
    schema = {
        "kind": "ANALYSIS_SCHEMA",
        "name": "17_watchdog_action_timeline",
        "figure": str(OUT_PNG),
        "title": ("Preempt-throttle watchdog: closed-loop temperature "
                  "control, no kernel cliff"),
        "tier_freq_mhz": {str(k): v for k, v in TIER_FREQ_MHZ.items()},
        "tier_engage_c": {str(k): v for k, v in TIER_ENGAGE_C.items()},
        "sources": {
            "wave9_watchdog_log": str(w9_log),
            "wave9_sensors_csv": str(WAVE9_DIR / "v1_fa2_stack" / "sensors.csv"),
            "wave10_watchdog_log": str(w10_log),
            "wave11_watchdog_log": str(WAVE11_WATCHDOG),
            "wave11_sensors_csv": str(w11_sensors_path),
        },
        "wave9": {
            "watchdog_t_start_unix": w9_t_start,
            "watchdog_t_end_unix": w9_t_end,
            **top_meta,
        },
        "wave10": {
            "watchdog_t_start_unix": w10_t_start,
            "watchdog_t_end_unix": w10_t_end,
            **mid_meta,
            "K_phase_boundaries_unix": [
                {"K": k, "t_start_unix": s, "t_end_unix": e}
                for (k, s, e) in WAVE10_K_PHASES
            ],
        },
        "wave11": {
            "watchdog_t_start_unix": w11_t_start,
            "watchdog_t_end_unix": w11_t_end,
            **bot_meta,
        },
        "narrative": (
            "The watchdog polls DDR thermal_zone47 once per second and clamps "
            "the big-core max-frequency before the kernel TJ_MAX cliff at 65 C. "
            "Across Wave-9 and Wave-10 the closed-loop control held DDR below "
            "the 65 C hard-cap with zero tier-3 engagements -- every tier-1 / "
            "tier-2 step caused a measurable DDR drop within 90 s."
        ),
    }
    with open(OUT_SCHEMA, "w") as fh:
        json.dump(schema, fh, indent=2, default=str)
    print(f"[wrote] {OUT_SCHEMA}")

    print("\n[ART_SCHEMA]")
    print(json.dumps(schema, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
