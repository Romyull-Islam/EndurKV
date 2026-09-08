#!/usr/bin/env python3
"""Aggregate per-run outputs (meta.json + steps.csv + sensors.csv) into a single
row per (model, policy, K, prompt). Derives:
  - energy_mj  ≈ integral of (V × I) over decode wall time, in mJ
  - mwh_per_1k_tokens = energy_mj / 3600 × 1000 / n_decode_steps
  - mean / peak junction temperature (max over all *_temp_mc columns)
  - mean / max throttle level (max over cpu*_cool_state)
  - prefill PSI / RSS / mem averages

Usage:
    host_aggregate_full.py --in-dir <pulled-logs-dir> --out-csv full_results.csv
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np


def aggregate_one(run_dir: Path):
    meta_p = run_dir / "meta.json"
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text())
    except Exception:
        return None
    row = dict(meta)
    row["_run_dir"] = str(run_dir)

    # sensors
    sens_p = run_dir / "sensors.csv"
    if sens_p.exists() and sens_p.stat().st_size > 200:
        try:
            sens = pd.read_csv(sens_p)
            if len(sens) > 1:
                # Pick out thermal cols
                temp_cols = [c for c in sens.columns if c.endswith("_temp_mc")]
                if temp_cols:
                    # Filter sensor-rest values (-273000 = unset, 0 = bcl/trip not active)
                    valid = sens[temp_cols].mask(sens[temp_cols] <= 0)
                    row["mean_junction_temp_c"] = float(valid.mean().max()) / 1000.0
                    row["peak_junction_temp_c"] = float(valid.max().max()) / 1000.0
                # Battery
                if "bat_voltage_mv" in sens.columns and "bat_current_ma" in sens.columns:
                    v = sens["bat_voltage_mv"].astype(float)
                    i = sens["bat_current_ma"].astype(float)
                    # current is typically negative when discharging
                    instant_mw = (v * i).abs() / 1000.0   # mW
                    # estimate dt from monotonic timestamps
                    if "monotonic_s" in sens.columns and len(sens) >= 2:
                        dt_s = float(sens["monotonic_s"].max() - sens["monotonic_s"].min())
                    else:
                        dt_s = len(sens) / 10.0  # 10 Hz default
                    avg_mw = float(instant_mw.mean())
                    energy_mj = avg_mw * dt_s
                    row["avg_power_mw"] = avg_mw
                    row["energy_mj"] = energy_mj
                    if meta.get("n_decode_steps", 0) > 0:
                        # mWh per 1K tokens: energy_mWh / n_tokens × 1000
                        energy_mwh = energy_mj / 3600.0 / 1000.0  # mJ → mWh
                        row["mwh_per_1k_tokens"] = energy_mwh / meta["n_decode_steps"] * 1000.0
                    row["bat_voltage_mean_v"] = float(v.mean()) / 1000.0
                    row["bat_current_mean_ma"] = float(i.mean())
                # Throttling
                cool_cols = [c for c in sens.columns if c.startswith("cpu") and c.endswith("_cool_state")]
                if cool_cols:
                    row["mean_throttle_level"] = float(sens[cool_cols].astype(float).mean().max())
                    row["peak_throttle_level"] = float(sens[cool_cols].astype(float).max().max())
                # GPU busy fraction (rough)
                if "gpu_busy_us" in sens.columns and "gpu_total_us" in sens.columns:
                    busy_d = sens["gpu_busy_us"].diff().clip(lower=0)
                    tot_d  = sens["gpu_total_us"].diff().clip(lower=0)
                    if tot_d.sum() > 0:
                        row["gpu_busy_frac"] = float(busy_d.sum() / tot_d.sum())
                # Mem
                if "mem_avail_kb" in sens.columns:
                    row["mem_avail_min_mb"] = float(sens["mem_avail_kb"].min()) / 1024.0
                row["sensors_n_samples"] = len(sens)
        except Exception as e:
            row["_sensors_err"] = str(e)

    # per-step
    steps_p = run_dir / "steps.csv"
    if steps_p.exists() and steps_p.stat().st_size > 100:
        try:
            steps = pd.read_csv(steps_p)
            if len(steps) > 1:
                row["per_step_mean_nll"] = float(steps.get("nll", pd.Series([np.nan])).mean())
                row["per_step_p50_lat_us"] = float(steps["wall_us"].diff().median())
                row["per_step_p95_lat_us"] = float(steps["wall_us"].diff().quantile(0.95))
                row["per_step_max_lat_us"] = float(steps["wall_us"].diff().max())
                row["per_step_evicted_total"] = int(steps.get("evicted_this_step", pd.Series([0])).sum())
        except Exception:
            pass

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-csv", default="phone_results_full.csv")
    args = ap.parse_args()

    # Find every dir that contains meta.json
    in_dir = Path(args.in_dir)
    run_dirs = sorted({p.parent for p in in_dir.rglob("meta.json")})
    rows = [r for r in (aggregate_one(d) for d in run_dirs) if r]
    if not rows:
        print(f"no runs found in {in_dir}"); return 1
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(df)} rows)")

    # Cross-policy summary
    print("\n=== Cross-policy headline (mean across prompts/K within each policy) ===")
    agg = df.groupby("policy").agg(
        n=("prompt_id","count"),
        prefill_ms=("prefill_ms","mean"),
        decode_tps=("decode_tps","mean"),
        peak_kv_mb=("peak_kv_mb","mean"),
        peak_rss_mb=("peak_rss_kb", lambda s: s.mean()/1024),
        perplexity=("perplexity","mean"),
        mean_temp_c=("mean_junction_temp_c","mean"),
        peak_temp_c=("peak_junction_temp_c","mean"),
        peak_throttle=("peak_throttle_level","mean"),
        mwh_per_1k=("mwh_per_1k_tokens","mean"),
        evicted_total=("per_step_evicted_total","mean"),
    ).reset_index().sort_values("decode_tps", ascending=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(agg.to_string(index=False))

    # Per-cell side-by-side
    print("\n=== Per-cell side-by-side ===")
    for key, grp in df.groupby(["prompt_id","k_nominal"]):
        pid, k = key
        print(f"\n  {pid} K={k}:")
        sub = grp[["policy","prefill_ms","decode_tps","peak_kv_mb","peak_rss_kb",
                   "perplexity","peak_junction_temp_c","mwh_per_1k_tokens",
                   "per_step_evicted_total"]].sort_values("decode_tps", ascending=False)
        print(sub.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
