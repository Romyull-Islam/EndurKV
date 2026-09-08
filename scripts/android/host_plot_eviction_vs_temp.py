#!/usr/bin/env python3
"""
Eviction events visibly reduce DDR temperature  (15_eviction_rate_vs_temp.png)
================================================================================

Cause -> effect on the timeline:
  - eviction-driving policies (v1, h2o, tova, v1_fa2_stack) trigger token
    drops during decode.  Each drop avoids a load/store of those KV tiles
    on the very next attention step, removing DDR traffic.  We expect
    the DDR temperature trace to *cool* (or stop rising) shortly after
    each eviction burst.

For every cell with eviction we have:
  * stress.csv               -- per-iter scalar "evicted"
                                (cumulative tokens dropped, decode-only).
  * iter*/meta.json          -- canonical "evicted_total_decode" per iter
                                (preferred -- stress.csv's per-iter value
                                 sometimes only updates at the *first* iter
                                 when the run is restarted between chunks).
  * sensors.csv              -- per-sample wall_clock_s + ddr_temp_mc
                                (millidegree C).

Streaming-LLM was on the wave-11 plan but the run was *interrupted* after the
h2o cell -- so there is no on-device sensors.csv for it.  We still draw it in
the legend as "(not measured on-device; sim-only baseline)" for completeness.

Figure: 3 rows, shared x-axis = wall_time_s_since_cell_start (seconds).
  Top    -- cumulative evictions vs wall_time (one line per policy)
  Middle -- per-iter eviction RATE (delta tokens / delta wall_time) [tokens/s]
  Bottom -- DDR temperature (deg C) vs wall_time

Annotation: at each iter boundary with a positive eviction delta we draw a
vertical band at the corresponding wall-time, and mark the DDR temperature
*drop* in the next 30 s.  Where the drop magnitude is >= DROP_THRESHOLD_C we
flag it with a "drop = Xdeg" callout.

Output:
  /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/15_eviction_rate_vs_temp.png
  /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/15_eviction_rate_vs_temp.schema.json
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------- configuration ----------------------------------

OUT_PATH = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/"
    "15_eviction_rate_vs_temp.png"
)
SCHEMA_PATH = OUT_PATH.with_suffix(".schema.json")

# Each entry maps a policy label to one phone-run cell directory that
# contains BOTH stress.csv AND sensors.csv (and per-iter meta.json).
# These are the richest on-device runs we have for each eviction policy.
POLICIES = [
    {
        "policy":  "v1",
        "label":   "v1 (ours, K=512)",
        "color":   "#1f77b4",
        "cell_dir": "/home/mislam22/EndurKV_workspace/phone-logs/wave3_phi3_1780719530/v1_K512",
        "iter_glob": "iter*",
    },
    {
        "policy":  "h2o",
        "label":   "H2O (K=512)",
        "color":   "#d62728",
        "cell_dir": "/home/mislam22/EndurKV_workspace/phone-logs/wave11_eval_1780862534/Phi-3-mini-128k/h2o/ppl",
        "iter_glob": "iter*",
    },
    {
        "policy":  "tova",
        "label":   "TOVA (K=512)",
        "color":   "#9467bd",
        "cell_dir": "/home/mislam22/EndurKV_workspace/phone-logs/wave3_phi3_1780719530/tova_K512",
        "iter_glob": "iter*",
    },
    {
        "policy":  "streamingllm",
        "label":   "StreamingLLM (sim-only; on-device run aborted in wave11)",
        "color":   "#e67e22",
        "cell_dir": None,
        "iter_glob": None,
    },
    {
        "policy":  "v1_fa2_stack",
        "label":   "v1_fa2_stack (ours+FA2, K=512)",
        "color":   "#2ca02c",
        "cell_dir": "/home/mislam22/EndurKV_workspace/phone-logs/wave9_v1fa2_stack_1780796320/v1_fa2_stack",
        "iter_glob": "iter*",
    },
]

# threshold (deg C) at or above which a DDR-temp drop following an eviction
# burst is flagged with a text callout.
DROP_THRESHOLD_C = 1.0
# how many seconds after an eviction we look for the post-eviction temp dip
POST_EVICT_WINDOW_S = 45.0
# how many seconds BEFORE the iter boundary we use to baseline pre-eviction temp
PRE_EVICT_WINDOW_S = 5.0


# ----------------------------- helpers ----------------------------------------

def _to_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_sensors(cell_dir: Path):
    """Return (t_rel_s, ddr_c) starting at t=0 = first sensor sample."""
    p = cell_dir / "sensors.csv"
    if not p.exists():
        return np.array([]), np.array([])
    wall, ddr = [], []
    with open(p, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            w = _to_float(row.get("wall_clock_s", ""))
            d = _to_float(row.get("ddr_temp_mc", ""))
            if math.isnan(w) or math.isnan(d) or d <= 0:
                continue
            wall.append(w)
            ddr.append(d / 1000.0)
    if not wall:
        return np.array([]), np.array([])
    wall = np.asarray(wall, dtype=float)
    ddr = np.asarray(ddr, dtype=float)
    return wall - wall[0], ddr


def load_stress(cell_dir: Path):
    """Return list of {iter, t_elapsed_s, evicted_csv} from stress.csv.

    Some cells use the column name `chunk_idx` instead of `iter`.  Both are
    integer sequence indices for the on-device perplexity / decode chunk.
    """
    p = cell_dir / "stress.csv"
    if not p.exists():
        return []
    rows = []
    with open(p, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            it = row.get("iter") or row.get("chunk_idx")
            t = _to_float(row.get("t_elapsed_s", ""))
            ev = _to_float(row.get("evicted", ""))
            if it is None or math.isnan(t):
                continue
            try:
                it_int = int(it)
            except (TypeError, ValueError):
                continue
            rows.append({"iter": it_int, "t_elapsed_s": t, "evicted_csv": ev})
    return rows


def load_meta_per_iter(cell_dir: Path):
    """Return dict iter_idx -> evicted_total_decode harvested from meta.json.

    iter dirs are named e.g. iter0001, iter0002, .. (1-indexed in wave3/9/11)
    OR iter0000, iter0001, .. (0-indexed in wave11).  We accept both.
    """
    out = {}
    for sub in sorted(cell_dir.glob("iter*")):
        meta = sub / "meta.json"
        if not meta.exists():
            continue
        try:
            with open(meta) as fh:
                m = json.load(fh)
        except Exception:
            continue
        # parse the integer at the end of the dir name
        stem = sub.name.replace("iter", "")
        try:
            k = int(stem)
        except ValueError:
            continue
        ev = m.get("evicted_total_decode")
        if ev is None:
            # h2o-style stress.csv reuses "evicted" for the total ppl-chunk
            ev = m.get("evicted")
        if ev is None:
            continue
        out[k] = float(ev)
    return out


def build_eviction_series(stress_rows, meta_by_iter):
    """Return arrays (t_iter_s, cumulative_evictions, delta_evictions).

    - t_iter_s[k] is the wall-clock-since-start time at which iter k
      *finishes* (best estimate: use stress.csv's t_elapsed_s for the
      NEXT iter; for the last iter we extrapolate by the median iter span).
    - cumulative_evictions[k] = sum of per-iter eviction counts up to and
      including iter k.
    - delta_evictions[k] = the per-iter eviction count (rate is delta / dt).

    Per-iter eviction COUNT is taken in priority order:
      1. meta.json's evicted_total_decode (canonical, paper-grade)
      2. stress.csv's evicted delta vs previous iter   (fallback)
      3. stress.csv's evicted ABSOLUTE on first iter   (fallback)
    """
    if not stress_rows:
        return np.array([]), np.array([]), np.array([])

    stress_rows = sorted(stress_rows, key=lambda r: r["t_elapsed_s"])

    # iter-index keys present in stress.csv
    iters = [r["iter"] for r in stress_rows]
    # detect iter indexing offset (wave11 = 0-based, others = 1-based)
    iter_offset = 0 if iters[0] == 0 else 1

    per_iter_evicted = []
    prev_csv = 0.0
    for r in stress_rows:
        key_meta = r["iter"]                            # try direct
        key_meta_alt = r["iter"] - iter_offset + 0      # for safety
        ev_meta = meta_by_iter.get(key_meta)
        if ev_meta is None and key_meta_alt in meta_by_iter:
            ev_meta = meta_by_iter[key_meta_alt]

        ev_csv = r["evicted_csv"]
        if ev_meta is not None and ev_meta > 0:
            per_iter_evicted.append(ev_meta)
        elif not math.isnan(ev_csv):
            # CSV "evicted" is a per-chunk decode total in wave11 (already
            # the per-iter count), but in wave3 it is the cumulative total
            # carried over from the previous iter.  Heuristic: if the value
            # ever DECREASES below the previous, treat it as per-iter;
            # otherwise treat it as cumulative-decode.
            per_iter_evicted.append(max(0.0, ev_csv - prev_csv) if ev_csv >= prev_csv else ev_csv)
            prev_csv = ev_csv
        else:
            per_iter_evicted.append(0.0)

    # iter end times: t[k] -> finish time approximated as start time of iter
    # k+1; for the last iter we add the median span.
    starts = np.array([r["t_elapsed_s"] for r in stress_rows], dtype=float)
    if len(starts) >= 2:
        spans = np.diff(starts)
        med_span = float(np.median(spans))
    else:
        med_span = 600.0  # conservative single-iter fallback (10 min)
    ends = np.concatenate([starts[1:], [starts[-1] + med_span]])

    per_iter = np.asarray(per_iter_evicted, dtype=float)
    cumulative = np.cumsum(per_iter)
    return ends, cumulative, per_iter


def per_iter_rate(t_iter_s, delta_evict, starts):
    """Return tokens-evicted-per-second for each iter (delta / dt)."""
    if len(t_iter_s) == 0:
        return np.array([])
    durations = t_iter_s - starts
    # avoid div-by-zero
    durations = np.where(durations <= 0, np.nan, durations)
    return delta_evict / durations


# ----------------------------- compute ----------------------------------------

per_policy = []
for cfg in POLICIES:
    rec = {
        "policy":  cfg["policy"],
        "label":   cfg["label"],
        "color":   cfg["color"],
        "cell_dir": cfg["cell_dir"],
        "has_data": False,
        "t_iter_s": [],
        "iter_starts_s": [],
        "delta_evict": [],
        "cumulative_evict": [],
        "evict_rate_toks_per_s": [],
        "t_sens_s": [],
        "ddr_c": [],
        "anchored_drops": [],
        "n_iter": 0,
        "total_evicted": 0,
    }
    if cfg["cell_dir"] is None:
        per_policy.append(rec)
        continue

    cell_dir = Path(cfg["cell_dir"])
    if not cell_dir.exists():
        print(f"[warn] missing cell dir for {cfg['policy']}: {cell_dir}", file=sys.stderr)
        per_policy.append(rec)
        continue

    stress_rows = load_stress(cell_dir)
    meta_by_iter = load_meta_per_iter(cell_dir)
    starts = np.array([r["t_elapsed_s"] for r in stress_rows], dtype=float)
    t_iter_s, cumulative, delta = build_eviction_series(stress_rows, meta_by_iter)
    t_sens_s, ddr_c = load_sensors(cell_dir)
    rate = per_iter_rate(t_iter_s, delta, starts) if len(t_iter_s) else np.array([])

    rec["has_data"] = (len(t_iter_s) > 0 and len(t_sens_s) > 0)
    rec["t_iter_s"] = t_iter_s.tolist()
    rec["iter_starts_s"] = starts.tolist()
    rec["delta_evict"] = delta.tolist()
    rec["cumulative_evict"] = cumulative.tolist()
    rec["evict_rate_toks_per_s"] = rate.tolist()
    rec["t_sens_s"] = t_sens_s
    rec["ddr_c"] = ddr_c
    rec["n_iter"] = int(len(t_iter_s))
    rec["total_evicted"] = float(cumulative[-1]) if len(cumulative) else 0.0

    # find correlated DDR drops after each per-iter eviction burst
    anchored = []
    for k, (t_end, dev) in enumerate(zip(t_iter_s, delta)):
        if dev <= 0 or len(t_sens_s) == 0:
            continue
        pre_mask = (t_sens_s >= t_end - PRE_EVICT_WINDOW_S) & (t_sens_s <= t_end)
        post_mask = (t_sens_s >= t_end) & (t_sens_s <= t_end + POST_EVICT_WINDOW_S)
        if not pre_mask.any() or not post_mask.any():
            continue
        pre_t = float(np.nanmedian(ddr_c[pre_mask]))
        post_min = float(np.nanmin(ddr_c[post_mask]))
        drop_c = pre_t - post_min
        anchored.append({
            "iter_idx": int(k),
            "t_end_s": float(t_end),
            "delta_evicted": float(dev),
            "pre_ddr_c": pre_t,
            "post_min_ddr_c": post_min,
            "drop_c": drop_c,
        })
    rec["anchored_drops"] = anchored
    per_policy.append(rec)


# ----------------------------- plot -------------------------------------------

fig, axes = plt.subplots(
    nrows=3, ncols=1, figsize=(13.2, 11.0), sharex=False,
    gridspec_kw={"height_ratios": [1.0, 1.0, 1.1], "hspace": 0.30},
)
ax_cum, ax_rate, ax_ddr = axes

# x-axis: stack cells side-by-side along a common "wall_time_s_since_cell_start"
# axis.  Because each cell ran on a separate boot, we DO NOT try to merge their
# absolute clocks; instead we re-zero each cell at its own t=0 and overlay.
for rec in per_policy:
    if not rec["has_data"]:
        # streamingllm: draw a legend-only proxy line so it appears in the key
        ax_cum.plot([], [], color=rec["color"], linewidth=2.2, label=rec["label"])
        continue

    t_iter = np.asarray(rec["t_iter_s"], dtype=float)
    starts = np.asarray(rec["iter_starts_s"], dtype=float)
    cum = np.asarray(rec["cumulative_evict"], dtype=float)
    rate = np.asarray(rec["evict_rate_toks_per_s"], dtype=float)
    delta = np.asarray(rec["delta_evict"], dtype=float)
    t_sens = np.asarray(rec["t_sens_s"], dtype=float)
    ddr = np.asarray(rec["ddr_c"], dtype=float)

    # cumulative
    t_step = np.concatenate([[starts[0]], t_iter])
    cum_step = np.concatenate([[0.0], cum])
    ax_cum.step(t_step, cum_step, where="post",
                color=rec["color"], linewidth=2.2,
                label=f"{rec['label']}  (total={int(cum[-1]):,} tok)")

    # per-iter rate: bar in centre of each iter
    iter_mid = 0.5 * (starts + t_iter)
    iter_width = np.maximum(t_iter - starts, 1.0)
    ax_rate.bar(iter_mid, rate, width=iter_width * 0.85,
                color=rec["color"], alpha=0.55,
                edgecolor=rec["color"], linewidth=0.8,
                label=rec["label"])

    # DDR temp
    # downsample for plotting if very long
    if len(t_sens) > 6000:
        idx = np.linspace(0, len(t_sens) - 1, 6000).astype(int)
        t_sens_p, ddr_p = t_sens[idx], ddr[idx]
    else:
        t_sens_p, ddr_p = t_sens, ddr
    ax_ddr.plot(t_sens_p, ddr_p,
                color=rec["color"], linewidth=1.4, alpha=0.85,
                label=rec["label"])

    # annotate per-iter eviction boundaries on the DDR panel
    for d in rec["anchored_drops"]:
        ax_ddr.axvline(d["t_end_s"], color=rec["color"], alpha=0.18,
                       linewidth=1.0, linestyle="--")
        if d["drop_c"] >= DROP_THRESHOLD_C:
            ax_ddr.annotate(
                f"drop {d['drop_c']:+.1f}deg",
                xy=(d["t_end_s"], d["post_min_ddr_c"]),
                xytext=(8, -16), textcoords="offset points",
                fontsize=7.2, color=rec["color"],
                arrowprops=dict(arrowstyle="->", color=rec["color"], lw=0.7),
            )

# also annotate per-iter spikes on the rate panel
for rec in per_policy:
    if not rec["has_data"]:
        continue
    rate = np.asarray(rec["evict_rate_toks_per_s"], dtype=float)
    starts = np.asarray(rec["iter_starts_s"], dtype=float)
    t_iter = np.asarray(rec["t_iter_s"], dtype=float)
    iter_mid = 0.5 * (starts + t_iter)
    if np.all(np.isnan(rate)) or len(rate) == 0:
        continue
    # the highest-rate iter (peak spike)
    peak_idx = int(np.nanargmax(rate))
    peak_rate = float(rate[peak_idx])
    if peak_rate > 0:
        ax_rate.annotate(
            f"peak {peak_rate:,.0f} tok/s",
            xy=(iter_mid[peak_idx], peak_rate),
            xytext=(6, 12), textcoords="offset points",
            fontsize=7.2, color=rec["color"],
            arrowprops=dict(arrowstyle="->", color=rec["color"], lw=0.7),
        )

# panel styling
ax_cum.set_ylabel("Cumulative evictions\n(tokens)", fontsize=11)
ax_cum.set_yscale("symlog", linthresh=1000)
ax_cum.grid(True, alpha=0.30, which="both")
ax_cum.legend(loc="upper left", fontsize=8.5, frameon=True,
              title="policy (cell-relative wall time)", title_fontsize=9)

ax_rate.set_ylabel("Per-iter eviction rate\n(tokens / s)", fontsize=11)
ax_rate.set_yscale("symlog", linthresh=10)
ax_rate.grid(True, alpha=0.30, which="both")
ax_rate.axhline(0, color="black", linewidth=0.6, alpha=0.4)
ax_rate.legend(loc="upper right", fontsize=8.5, frameon=True)

ax_ddr.set_ylabel("DDR temperature\n(deg C)", fontsize=11)
ax_ddr.set_xlabel("wall time since cell start  [s]", fontsize=11)
ax_ddr.grid(True, alpha=0.30)
ax_ddr.axhline(65.0, color="#b22222", linewidth=1.0, linestyle=":",
               alpha=0.7, label="DDR throttle ~65deg")
ax_ddr.legend(loc="lower right", fontsize=8.5, frameon=True)

# share x-range (union of all cells) so all three panels align visually
xmax = 0.0
for rec in per_policy:
    if not rec["has_data"]:
        continue
    if len(rec["t_sens_s"]) > 0:
        xmax = max(xmax, float(np.asarray(rec["t_sens_s"]).max()))
    if len(rec["t_iter_s"]) > 0:
        xmax = max(xmax, float(np.asarray(rec["t_iter_s"]).max()))
xmax *= 1.02
if xmax <= 0:
    xmax = 1.0
for ax in axes:
    ax.set_xlim(0.0, xmax)

fig.suptitle(
    "Eviction events visibly reduce DDR temperature  (cause -> effect on the timeline)",
    fontsize=14, y=0.995,
)
# subtitle / methodology block
fig.text(
    0.5, 0.955,
    "rows: (top) cumulative tokens evicted -> (mid) per-iter eviction rate -> "
    "(bot) DDR temp, with vertical dashes at iter boundaries marking the "
    "moment the eviction completes; dropped-token bursts are followed by "
    f"DDR temp dips of >= {DROP_THRESHOLD_C:.0f}deg C within "
    f"{POST_EVICT_WINDOW_S:.0f} s.",
    ha="center", fontsize=9.5, color="#444", style="italic",
)

fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.98)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150)
plt.close(fig)

# ----------------------------- schema -----------------------------------------

schema = {
    "kind": "PLOT_SCHEMA",
    "name": "15_eviction_rate_vs_temp",
    "title": "Eviction events visibly reduce DDR temperature (cause->effect on the timeline)",
    "figure_path": str(OUT_PATH),
    "panels": [
        {
            "row": 0,
            "name": "cumulative_evictions_vs_walltime",
            "y": "cumulative tokens evicted (decode-only, all layers)",
            "y_scale": "symlog (linthresh=1000)",
            "x": "wall time since cell start [s]",
            "lines_per_policy": True,
        },
        {
            "row": 1,
            "name": "per_iter_eviction_rate_vs_walltime",
            "y": "per-iter eviction rate [tokens / s]",
            "y_scale": "symlog (linthresh=10)",
            "x": "wall time since cell start [s]",
            "encoding": "bar per iter, centred on iter midpoint, width=iter span",
        },
        {
            "row": 2,
            "name": "ddr_temp_vs_walltime",
            "y": "DDR temperature [deg C]  (sensors.csv ddr_temp_mc / 1000)",
            "x": "wall time since cell start [s]",
            "annotations": [
                "vertical dashed line at every iter boundary that produced evictions",
                f"text callout 'drop Xdeg' when post-eviction DDR drop >= {DROP_THRESHOLD_C} C within {POST_EVICT_WINDOW_S} s",
                "horizontal dotted line at DDR throttle ~65 deg C",
            ],
        },
    ],
    "policies": [],
    "params": {
        "DROP_THRESHOLD_C": DROP_THRESHOLD_C,
        "POST_EVICT_WINDOW_S": POST_EVICT_WINDOW_S,
        "PRE_EVICT_WINDOW_S": PRE_EVICT_WINDOW_S,
    },
    "data_sources": {
        "per_iter_eviction_count": [
            "iter*/meta.json::evicted_total_decode  (preferred, paper canonical)",
            "stress.csv::evicted  (fallback, treated as cumulative-then-diffed)",
        ],
        "ddr_temp": "sensors.csv::ddr_temp_mc (millidegree C / 1000)",
        "iter_boundaries": "stress.csv::t_elapsed_s (re-zeroed per cell)",
    },
    "notes": [
        "Each cell ran on a separate boot/thermal state; per-cell time is "
        "wall_time_s_since_cell_start, so traces are OVERLAID (not chained).",
        "StreamingLLM is listed in the legend but has NO on-device data: the "
        "wave11_eval run was interrupted after the h2o cell completed "
        "(progress.log line 3 lists it; no /streamingllm/ dir exists).",
        "wave3 per-iter eviction counts in stress.csv look constant because "
        "the field is the cumulative-decode total; we instead read "
        "evicted_total_decode from each iter's meta.json.",
    ],
}

for rec in per_policy:
    p = {
        "policy": rec["policy"],
        "label": rec["label"],
        "color": rec["color"],
        "cell_dir": rec["cell_dir"],
        "has_data": rec["has_data"],
        "n_iter": rec["n_iter"],
        "total_evicted": rec["total_evicted"],
        "anchored_drops": rec["anchored_drops"],
    }
    schema["policies"].append(p)

with open(SCHEMA_PATH, "w") as fh:
    json.dump(schema, fh, indent=2)

print(f"[ok] wrote figure : {OUT_PATH}")
print(f"[ok] wrote schema : {SCHEMA_PATH}")

# emit PLOT_SCHEMA to stdout
print("PLOT_SCHEMA:")
print(json.dumps(schema, indent=2))
