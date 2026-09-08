#!/usr/bin/env python3
"""Compare thermal behavior of vanilla vs h2o on Phi-3 (Wave-11)."""
import os
import re
import json
import glob
import math
import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/mislam22/EndurKV_workspace/phone-logs/wave11_eval_1780862534/Phi-3-mini-128k"
OUT_PNG = "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/10_phi3_vanilla_vs_h2o_thermal.png"
OUT_JSON = "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/10_phi3_vanilla_vs_h2o_thermal.schema.json"

ADB = "/home/mislam22/tools/platform-tools/adb"


def load_meta(p):
    """meta.json may contain bare `inf` (invalid JSON). Sanitize."""
    raw = open(p).read()
    raw = re.sub(r":\s*-?inf\b", ": null", raw, flags=re.IGNORECASE)
    raw = re.sub(r":\s*nan\b", ": null", raw, flags=re.IGNORECASE)
    return json.loads(raw)


def phone_mtime(remote_path):
    out = subprocess.run([ADB, "shell", f"stat -c %Y {remote_path}"], capture_output=True, text=True).stdout.strip()
    if not out or not out.isdigit():
        return None
    return int(out)


def collect_cell(cell_dir, phone_root):
    iters = sorted(glob.glob(os.path.join(cell_dir, "iter*")))
    cells = []
    for it in iters:
        meta_path = os.path.join(it, "meta.json")
        if not os.path.exists(meta_path):
            continue
        meta = load_meta(meta_path)
        iter_name = os.path.basename(it)
        remote_meta = f"{phone_root}/{iter_name}/meta.json"
        mtime = phone_mtime(remote_meta)
        total_ms = meta.get("total_ms")
        if mtime is None or total_ms is None:
            continue
        end_wall = float(mtime)
        start_wall = end_wall - (total_ms / 1000.0)
        # decode segment ~ start + prefill_ms
        prefill_ms = meta.get("prefill_ms", 0.0) or 0.0
        decode_start = start_wall + (prefill_ms / 1000.0)
        cells.append({
            "iter": iter_name,
            "start_wall": start_wall,
            "end_wall": end_wall,
            "decode_start_wall": decode_start,
            "prefill_ms": prefill_ms,
            "total_ms": total_ms,
            "n_decode_steps": meta.get("n_decode_steps"),
            "perplexity": meta.get("perplexity"),
            "peak_kv_cells": meta.get("peak_kv_cells"),
            "evicted_total_decode": meta.get("evicted_total_decode", 0),
            "policy": meta.get("policy"),
        })
    return cells


def load_sensors(p):
    df = pd.read_csv(p, low_memory=False)
    # Convert temps from milli-C to C
    for col in ["ddr_temp_mc", "cpu-0-5-1_temp_mc", "cpu-0-4-1_temp_mc"]:
        if col in df.columns:
            df[col.replace("_mc", "_c")] = df[col] / 1000.0
    # Cpu peak = max across cpu_* columns per row, excluding hw-trip static points (95C)
    cpu_cols = [c for c in df.columns
                if c.startswith("cpu-") and c.endswith("_temp_mc") and "hw-trip" not in c]
    df["cpu_peak_c"] = df[cpu_cols].max(axis=1) / 1000.0
    df["ddr_c"] = df["ddr_temp_mc"] / 1000.0
    df["mem_avail_gb"] = df["mem_avail_kb"] / (1024.0 * 1024.0)
    df["swap_mb"] = df["vmstat_pswpout"] / 256.0  # 4KB pages -> MB (pages * 4 / 1024)
    df = df.sort_values("wall_clock_s").reset_index(drop=True)
    return df


def summarize_cell(sensors, cell):
    """Per-cell metrics over [start, end] and decode-only mean DDR."""
    mask_all = (sensors["wall_clock_s"] >= cell["start_wall"]) & (sensors["wall_clock_s"] <= cell["end_wall"])
    mask_dec = (sensors["wall_clock_s"] >= cell["decode_start_wall"]) & (sensors["wall_clock_s"] <= cell["end_wall"])
    sub = sensors[mask_all]
    decode = sensors[mask_dec]
    if len(sub) == 0:
        return None
    swap_total = 0
    if len(sub) > 0:
        # vmstat_pswpout is cumulative; use diff
        sw0 = sub["vmstat_pswpout"].iloc[0]
        sw1 = sub["vmstat_pswpout"].iloc[-1]
        swap_total = (sw1 - sw0) * 4 / 1024.0  # pages*4KB -> MB
    return {
        "iter": cell["iter"],
        "policy": cell["policy"],
        "n_decode_steps": cell["n_decode_steps"],
        "perplexity": cell["perplexity"],
        "peak_kv_cells": cell["peak_kv_cells"],
        "evicted_total_decode": cell["evicted_total_decode"],
        "peak_ddr_c": float(sub["ddr_c"].max()),
        "mean_ddr_c": float(sub["ddr_c"].mean()),
        "mean_ddr_decode_c": float(decode["ddr_c"].mean()) if len(decode) else float("nan"),
        "peak_cpu_c": float(sub["cpu_peak_c"].max()),
        "mean_cpu_c": float(sub["cpu_peak_c"].mean()),
        "swap_mb_total": float(swap_total),
        "min_mem_avail_gb": float(sub["mem_avail_gb"].min()),
        "n_samples": int(len(sub)),
        "duration_s": float(cell["end_wall"] - cell["start_wall"]),
    }


def main():
    cfgs = [
        ("vanilla", os.path.join(ROOT, "vanilla", "ppl"),
         "/data/local/tmp/endurkv/logs/wave11_eval_1780862534/Phi-3-mini-128k/vanilla/ppl"),
        ("h2o", os.path.join(ROOT, "h2o", "ppl"),
         "/data/local/tmp/endurkv/logs/wave11_eval_1780862534/Phi-3-mini-128k/h2o/ppl"),
    ]
    cells_by = {}
    sensors_by = {}
    for name, local, remote in cfgs:
        sensors_csv = os.path.join(local, "sensors.csv")
        print(f"[{name}] loading {sensors_csv} ...")
        sensors_by[name] = load_sensors(sensors_csv)
        cells_by[name] = collect_cell(local, remote)
        print(f"[{name}] cells: {len(cells_by[name])}")

    # ---------- Per-cell summaries ----------
    rows = []
    for name, cells in cells_by.items():
        for c in cells:
            s = summarize_cell(sensors_by[name], c)
            if s is None:
                continue
            s["arm"] = name
            rows.append(s)
    summary = pd.DataFrame(rows)
    print(summary[["arm","iter","peak_ddr_c","mean_ddr_decode_c","peak_cpu_c","swap_mb_total","min_mem_avail_gb","perplexity","peak_kv_cells","evicted_total_decode"]].to_string(index=False))

    # ---------- Comparison tables ----------
    agg = summary.groupby("arm").agg(
        n_cells=("iter","count"),
        peak_ddr_c=("peak_ddr_c","max"),
        mean_peak_ddr_c=("peak_ddr_c","mean"),
        mean_ddr_decode_c=("mean_ddr_decode_c","mean"),
        peak_cpu_c=("peak_cpu_c","max"),
        mean_swap_mb=("swap_mb_total","mean"),
        total_swap_mb=("swap_mb_total","sum"),
        min_mem_avail_gb=("min_mem_avail_gb","min"),
        mean_perplexity=("perplexity","mean"),
        mean_peak_kv_cells=("peak_kv_cells","mean"),
        mean_evicted=("evicted_total_decode","mean"),
    ).reset_index()
    print("\n=== Per-arm aggregate ===")
    print(agg.to_string(index=False))

    # ---------- Plot ----------
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = {"vanilla":"#1f77b4", "h2o":"#2ca02c"}

    # Align time at vanilla's first cell start = 0
    t0 = min(c["start_wall"] for c in cells_by["vanilla"])
    for name, sub in sensors_by.items():
        t = sub["wall_clock_s"].values - t0
        # Trim to start a bit before first cell
        first_start = min((c["start_wall"] for c in cells_by[name]), default=t0) - t0 - 30
        last_end = max((c["end_wall"] for c in cells_by[name]), default=t0) - t0 + 30
        m = (t >= first_start) & (t <= last_end)
        axes[0].plot(t[m]/60.0, sub["ddr_c"].values[m], color=colors[name], lw=1.0, label=f"{name} DDR")
        axes[1].plot(t[m]/60.0, sub["cpu_peak_c"].values[m], color=colors[name], lw=1.0, label=f"{name} CPU peak")

    # Cell shading (just for h2o to highlight progress relative)
    for name, cells in cells_by.items():
        col = colors[name]
        for c in cells:
            t_lo = (c["start_wall"] - t0)/60.0
            t_hi = (c["end_wall"] - t0)/60.0
            for ax in axes:
                ax.axvspan(t_lo, t_hi, alpha=0.05, color=col)

    axes[0].set_ylabel("DDR temp (C)")
    axes[1].set_ylabel("Peak CPU temp (C)")
    axes[1].set_xlabel("Time since vanilla start (min)")
    axes[0].set_title("Phi-3-mini-128k Wave-11: vanilla (blue, full run) vs H2O K=512 (green, partial)\nDDR + CPU thermal time series")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    # Annotate H2O current-status
    h2o_last = max(c["end_wall"] for c in cells_by["h2o"]) - t0
    axes[0].axvline(h2o_last/60.0, color="green", ls="--", lw=0.7, alpha=0.5)
    axes[0].text(h2o_last/60.0, axes[0].get_ylim()[1]*0.98, "  H2O current", color="green", fontsize=8, va="top")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved: {OUT_PNG}")

    # ---------- Schema ----------
    def jsonable(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o

    out = {
        "kind": "ANALYSIS_SCHEMA",
        "name": "10_phi3_vanilla_vs_h2o_thermal",
        "model": "Phi-3-mini-128k",
        "wave": "wave11_eval_1780862534",
        "sources": {
            "vanilla_sensors": os.path.join(ROOT, "vanilla", "ppl", "sensors.csv"),
            "h2o_sensors": os.path.join(ROOT, "h2o", "ppl", "sensors.csv"),
        },
        "per_cell": json.loads(summary.to_json(orient="records")),
        "per_arm": json.loads(agg.to_json(orient="records")),
        "figure": OUT_PNG,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=jsonable)
    print(f"Saved: {OUT_JSON}")

    # ---------- Headline ----------
    v = agg[agg["arm"]=="vanilla"].iloc[0]
    h = agg[agg["arm"]=="h2o"].iloc[0]
    print("\n=== HEADLINE ===")
    print(f"vanilla peak DDR={v.peak_ddr_c:.1f}C, mean-decode DDR={v.mean_ddr_decode_c:.2f}C, n_cells={int(v.n_cells)}")
    print(f"h2o     peak DDR={h.peak_ddr_c:.1f}C, mean-decode DDR={h.mean_ddr_decode_c:.2f}C, n_cells={int(h.n_cells)} (partial)")
    print(f"DELTA peak DDR (h2o - vanilla)         = {h.peak_ddr_c - v.peak_ddr_c:+.1f}C")
    print(f"DELTA mean-decode DDR (h2o - vanilla)  = {h.mean_ddr_decode_c - v.mean_ddr_decode_c:+.2f}C")
    print(f"DELTA peak CPU (h2o - vanilla)         = {h.peak_cpu_c - v.peak_cpu_c:+.1f}C")
    print(f"DELTA min mem_avail (h2o - vanilla)    = {h.min_mem_avail_gb - v.min_mem_avail_gb:+.2f}GB")


if __name__ == "__main__":
    main()
