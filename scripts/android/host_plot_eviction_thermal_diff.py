#!/usr/bin/env python3
"""
PLOT 16: Eviction's thermal payoff -- before/after eviction thermal differential.

Hold the workload constant (Phi-3-mini long-decode, Wave-4 scenario) and ask:
how much hotter does the chip run when we let the KV cache grow vs when we
clamp it with an eviction policy?

Data layout (everything is logged at ~30 Hz via the sensors probe, with
per-iter stress.csv summaries and meta.json from llama-cli):

  vanilla (no eviction):
    phone-logs/wave4_longdecode_1780750084/vanilla/
        sensors.csv  -- DDR/CPU thermal stream
        stress.csv   -- iter timing + peak_kv_cells
        iter*/meta.json
  v1 K=512 (aggressive spread-gate eviction):
    phone-logs/wave4_longdecode_1780750084/v1_K512/
        same files
  v1_fa2_stack K=512 (FA2 + stacked inflight eviction, Wave-9):
    phone-logs/wave9_v1fa2_stack_1780796320/v1_fa2_stack/
  h2o K=512 (sink + recency, Wave-11 Phi-3 ppl run):
    phone-logs/wave11_eval_1780862534/Phi-3-mini-128k/h2o/ppl/

For each arm we compute peak DDR, mean DDR over the decode segment (drop the
last ~5 s of each iter as cooldown), peak CPU, and steady-state KV cells from
meta.json. We then express vanilla as the eviction-off baseline and the other
three arms as the eviction-on alternatives. The differential against vanilla
goes in the title.

2-panel figure:
  Left:  bar chart -- peak DDR for {vanilla, v1, v1_fa2_stack, h2o}
  Right: bar chart -- peak CPU for the same arms
  Each bar annotated with the policy mechanism string.

Output:
  figures/relationship_plots/16_eviction_thermal_diff.png
  figures/relationship_plots/16_eviction_thermal_diff.schema.json (PLOT_SCHEMA)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Arm spec
# ---------------------------------------------------------------------------
ARMS = [
    {
        "key": "vanilla",
        "label": "vanilla",
        "wave": "wave4_longdecode_1780750084",
        "dir": Path("/home/mislam22/EndurKV_workspace/phone-logs/"
                    "wave4_longdecode_1780750084/vanilla"),
        "mechanism": "no eviction\n(cache grows 500→2311)",
        "color": "#d7301f",
    },
    {
        "key": "v1",
        "label": "v1 K=512",
        "wave": "wave4_longdecode_1780750084",
        "dir": Path("/home/mislam22/EndurKV_workspace/phone-logs/"
                    "wave4_longdecode_1780750084/v1_K512"),
        "mechanism": "spread-gate K=512\n(cache capped at 748)",
        "color": "#2c7fb8",
    },
    {
        "key": "v1_fa2_stack",
        "label": "v1_fa2_stack K=512",
        "wave": "wave9_v1fa2_stack_1780796320",
        "dir": Path("/home/mislam22/EndurKV_workspace/phone-logs/"
                    "wave9_v1fa2_stack_1780796320/v1_fa2_stack"),
        "mechanism": "FA2 + stacked-inflight\nK=512",
        "color": "#7fbf7b",
    },
    {
        "key": "h2o",
        "label": "h2o K=512",
        "wave": "wave11_eval_1780862534",
        "dir": Path("/home/mislam22/EndurKV_workspace/phone-logs/"
                    "wave11_eval_1780862534/Phi-3-mini-128k/h2o/ppl"),
        "mechanism": "sink+recency K=512",
        "color": "#fdae61",
    },
]

OUT_PNG = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/"
    "relationship_plots/16_eviction_thermal_diff.png"
)
OUT_SCHEMA = OUT_PNG.with_suffix(".schema.json")

# Trim the last `COOLDOWN_S` seconds of each iter window because llama-cli has
# finished and the chip is bleeding heat into the heatsink (would underestimate
# the load-time mean).
COOLDOWN_S = 5.0


# ---------------------------------------------------------------------------
# Sensor loading
# ---------------------------------------------------------------------------
def load_sensors(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["wall_clock_s"] = pd.to_numeric(df["wall_clock_s"], errors="coerce")
    df = df.dropna(subset=["wall_clock_s"])
    df["ddr_c"] = pd.to_numeric(df["ddr_temp_mc"], errors="coerce") / 1000.0
    cpu_cols = [c for c in df.columns
                if re.match(r"^cpu-\d-\d-\d_temp_mc$", c)]
    if not cpu_cols:
        raise SystemExit(f"no cpu-*-*-*_temp_mc columns in {csv_path}")
    cpu_arr = (df[cpu_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
               / 1000.0)
    df["cpu_max_c"] = np.nanmax(cpu_arr, axis=1)
    return df[["wall_clock_s", "ddr_c", "cpu_max_c"]].reset_index(drop=True)


def load_stress(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def iter_windows(sensors: pd.DataFrame, stress: pd.DataFrame):
    """Yield (iter_idx, t_start_unix, t_end_unix) tuples for each iter row.

    stress.t_elapsed_s is anchored at the start of the run; we add it to the
    first sensors wall_clock_s to project into unix seconds. The iter ends at
    the next iter's start, or the last sensor sample for the final iter.
    """
    if sensors.empty or stress.empty:
        return
    t_origin = float(sensors["wall_clock_s"].iloc[0])
    starts = stress["t_elapsed_s"].astype(float).to_numpy()
    last_t = float(sensors["wall_clock_s"].iloc[-1])
    ends = np.concatenate([starts[1:] + t_origin, [last_t]])
    starts = starts + t_origin
    for i, (s, e) in enumerate(zip(starts, ends), start=1):
        yield i, s, e


def per_iter_thermal(arm: dict):
    """Return dataframe of per-iter ddr_peak / ddr_decode_mean / cpu_peak."""
    sensors = load_sensors(arm["dir"] / "sensors.csv")
    stress = load_stress(arm["dir"] / "stress.csv")
    rows = []
    for i, s, e in iter_windows(sensors, stress):
        # Trim cooldown tail.
        e_eff = max(s + 1.0, e - COOLDOWN_S)
        mask = (sensors["wall_clock_s"] >= s) & (sensors["wall_clock_s"] < e_eff)
        sub = sensors.loc[mask]
        if sub.empty:
            continue
        rows.append({
            "iter": i,
            "ddr_peak_c": float(np.nanmax(sub["ddr_c"])),
            "ddr_mean_decode_c": float(np.nanmean(sub["ddr_c"])),
            "cpu_peak_c": float(np.nanmax(sub["cpu_max_c"])),
            "cpu_mean_decode_c": float(np.nanmean(sub["cpu_max_c"])),
            "n_samples": int(len(sub)),
        })
    return pd.DataFrame(rows)


def load_kv_summary(arm: dict):
    """Steady-state KV from meta.json files (peak_kv_cells, evicted_total_decode)."""
    cells = []
    evicts = []
    policy = None
    k_nom = None
    for meta_path in sorted(arm["dir"].glob("iter*/meta.json")):
        try:
            m = json.loads(meta_path.read_text())
        except Exception:
            continue
        if m.get("peak_kv_cells") is not None:
            cells.append(int(m["peak_kv_cells"]))
        if m.get("evicted_total_decode") is not None:
            evicts.append(int(m["evicted_total_decode"]))
        policy = m.get("policy", policy)
        k_nom = m.get("k_nominal", k_nom)
    return {
        "policy_meta": policy,
        "k_nominal": k_nom,
        "peak_kv_cells_mean": float(np.mean(cells)) if cells else float("nan"),
        "peak_kv_cells_max": int(max(cells)) if cells else None,
        "evicted_mean": float(np.mean(evicts)) if evicts else float("nan"),
        "n_iter_meta": len(cells),
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def aggregate():
    arm_rows = []
    per_iter_all = {}
    for arm in ARMS:
        if not (arm["dir"] / "sensors.csv").exists():
            print(f"[warn] sensors missing for {arm['key']}: {arm['dir']}",
                  file=sys.stderr)
            continue
        per_iter = per_iter_thermal(arm)
        kv = load_kv_summary(arm)
        per_iter_all[arm["key"]] = per_iter
        if per_iter.empty:
            print(f"[warn] no iter windows for {arm['key']}", file=sys.stderr)
            continue
        arm_rows.append({
            "arm": arm["key"],
            "label": arm["label"],
            "mechanism": arm["mechanism"],
            "color": arm["color"],
            "wave": arm["wave"],
            "n_iter": int(len(per_iter)),
            "ddr_peak_max_c": float(per_iter["ddr_peak_c"].max()),
            "ddr_peak_mean_c": float(per_iter["ddr_peak_c"].mean()),
            "ddr_peak_std_c": float(per_iter["ddr_peak_c"].std(ddof=0)),
            "ddr_mean_decode_c": float(per_iter["ddr_mean_decode_c"].mean()),
            "ddr_mean_decode_std_c": float(
                per_iter["ddr_mean_decode_c"].std(ddof=0)),
            "cpu_peak_max_c": float(per_iter["cpu_peak_c"].max()),
            "cpu_peak_mean_c": float(per_iter["cpu_peak_c"].mean()),
            "cpu_peak_std_c": float(per_iter["cpu_peak_c"].std(ddof=0)),
            "cpu_mean_decode_c": float(per_iter["cpu_mean_decode_c"].mean()),
            **kv,
        })
    return pd.DataFrame(arm_rows), per_iter_all


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(df: pd.DataFrame, deltas: dict):
    # Preserve ARMS ordering.
    order = [a["key"] for a in ARMS]
    df = df.set_index("arm").reindex([k for k in order if k in df["arm"].tolist()
                                      if False] + order).dropna(how="all")
    # The reindex trick above gets fussy; just sort by ARMS order directly.
    df = df.copy()
    df["__order__"] = df.index.map({k: i for i, k in enumerate(order)})
    df = df.sort_values("__order__").drop(columns="__order__")

    n = len(df)
    x = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))

    title = (
        f"Eviction's thermal payoff: ΔDDR = "
        f"{deltas['ddr_peak_v1_vs_vanilla']:.1f}°C at K=512 "
        f"vs vanilla (Wave-4)"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ---------- Left: peak DDR -----------------------------------------
    ax = axes[0]
    bars = ax.bar(
        x, df["ddr_peak_max_c"], color=df["color"],
        edgecolor="black", linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=0, fontsize=9)
    ax.set_ylabel("Peak DDR temperature (°C)")
    ax.set_title("Left: peak DDR per arm  (workload = Phi-3 long-decode)")
    ax.grid(True, axis="y", alpha=0.3)
    # Annotate each bar with mechanism + value.
    ymax_l = float(df["ddr_peak_max_c"].max())
    for bar, (_, row) in zip(bars, df.iterrows()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.5,
            f"{h:.1f}°C\n{row['mechanism']}",
            ha="center", va="bottom", fontsize=8.5,
        )
    ax.set_ylim(0, ymax_l * 1.28)
    # Dotted vanilla reference line.
    if "vanilla" in df.index:
        v = float(df.loc["vanilla", "ddr_peak_max_c"])
        ax.axhline(v, color="#d7301f", linestyle="--", alpha=0.5, linewidth=1.0)
        ax.text(n - 0.4, v + 0.2, f"vanilla = {v:.1f}°C",
                fontsize=8, color="#d7301f", ha="right")

    # ---------- Right: peak CPU ----------------------------------------
    ax = axes[1]
    bars = ax.bar(
        x, df["cpu_peak_max_c"], color=df["color"],
        edgecolor="black", linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=0, fontsize=9)
    ax.set_ylabel("Peak CPU temperature (°C)")
    ax.set_title("Right: peak CPU per arm")
    ax.grid(True, axis="y", alpha=0.3)
    ymax_r = float(df["cpu_peak_max_c"].max())
    for bar, (_, row) in zip(bars, df.iterrows()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.5,
            f"{h:.1f}°C\n{row['mechanism']}",
            ha="center", va="bottom", fontsize=8.5,
        )
    ax.set_ylim(0, ymax_r * 1.28)
    if "vanilla" in df.index:
        v = float(df.loc["vanilla", "cpu_peak_max_c"])
        ax.axhline(v, color="#d7301f", linestyle="--", alpha=0.5, linewidth=1.0)
        ax.text(n - 0.4, v + 0.2, f"vanilla = {v:.1f}°C",
                fontsize=8, color="#d7301f", ha="right")

    # Footer summarising the deltas.
    foot = (
        f"ΔDDR_peak (v1 vs vanilla)  = "
        f"{deltas['ddr_peak_v1_vs_vanilla']:+.1f}°C   "
        f"ΔDDR_mean (decode)         = "
        f"{deltas['ddr_mean_v1_vs_vanilla']:+.1f}°C   "
        f"Δcache_steady (peak_kv)    = "
        f"{deltas['cache_v1_vs_vanilla']:+.0f} cells"
    )
    fig.text(0.5, 0.012, foot, ha="center", fontsize=9, family="monospace")

    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def compute_deltas(df: pd.DataFrame):
    """v1 (Wave-4 K=512) is the named eviction arm for the title delta."""
    if "vanilla" not in df["arm"].tolist() or "v1" not in df["arm"].tolist():
        raise SystemExit("vanilla and v1 arms are both required")
    van = df.set_index("arm").loc["vanilla"]
    v1 = df.set_index("arm").loc["v1"]
    return {
        "ddr_peak_v1_vs_vanilla":  float(van["ddr_peak_max_c"]   - v1["ddr_peak_max_c"]),
        "ddr_mean_v1_vs_vanilla":  float(van["ddr_mean_decode_c"] - v1["ddr_mean_decode_c"]),
        "cpu_peak_v1_vs_vanilla":  float(van["cpu_peak_max_c"]   - v1["cpu_peak_max_c"]),
        "cache_v1_vs_vanilla":     float(van["peak_kv_cells_mean"] - v1["peak_kv_cells_mean"]),
    }


def emit_schema(df: pd.DataFrame, per_iter_all: dict, deltas: dict):
    sources = {
        a["key"]: {
            "wave": a["wave"],
            "sensors": str(a["dir"] / "sensors.csv"),
            "stress":  str(a["dir"] / "stress.csv"),
            "meta_glob": str(a["dir"] / "iter*/meta.json"),
        }
        for a in ARMS
    }
    per_arm = []
    for _, row in df.iterrows():
        per_arm.append({k: (None if (isinstance(v, float) and np.isnan(v)) else
                            (v.item() if hasattr(v, "item") else v))
                        for k, v in row.to_dict().items()})
    per_cell = []
    for arm_key, pi in per_iter_all.items():
        for _, r in pi.iterrows():
            per_cell.append({
                "arm": arm_key,
                "iter": int(r["iter"]),
                "ddr_peak_c": float(r["ddr_peak_c"]),
                "ddr_mean_decode_c": float(r["ddr_mean_decode_c"]),
                "cpu_peak_c": float(r["cpu_peak_c"]),
                "cpu_mean_decode_c": float(r["cpu_mean_decode_c"]),
                "n_samples": int(r["n_samples"]),
            })
    schema = {
        "kind": "PLOT_SCHEMA",
        "name": "16_eviction_thermal_diff",
        "model": "Phi-3-mini-128k",
        "scenario": "long-decode (Wave-4 workload)",
        "cooldown_trim_s": COOLDOWN_S,
        "sources": sources,
        "deltas_v1_vs_vanilla": deltas,
        "per_arm": per_arm,
        "per_cell": per_cell,
        "figure": str(OUT_PNG),
    }
    OUT_SCHEMA.write_text(json.dumps(schema, indent=2))
    print(f"wrote {OUT_SCHEMA}")


def main():
    df, per_iter_all = aggregate()
    if df.empty:
        raise SystemExit("no arms could be aggregated; check phone-logs paths")
    deltas = compute_deltas(df)
    print(df.to_string(index=False))
    print("deltas:", json.dumps(deltas, indent=2))
    plot(df, deltas)
    emit_schema(df, per_iter_all, deltas)


if __name__ == "__main__":
    main()
