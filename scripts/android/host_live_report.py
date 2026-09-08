#!/usr/bin/env python3
"""Live report — scans pulled phone logs and prints comparison table.
Run repeatedly while sweep is in progress to see results land.

Usage:
    while sleep 60; do
        python host_live_report.py --in-dir <sweep_dir>
    done
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np


def load_runs(in_dir):
    rows = []
    for mf in Path(in_dir).rglob("meta.json"):
        try:
            d = json.loads(mf.read_text())
        except Exception:
            continue
        # Pull replicate number from path: .../<policy>/K<K>/<prompt>/rep<N>/meta.json
        parts = list(mf.parts)
        try:
            rep = int(parts[parts.index([p for p in parts if p.startswith("rep")][0])].replace("rep",""))
        except Exception:
            rep = 0
        d["_rep"] = rep
        d["_path"] = str(mf)
        rows.append(d)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    args = ap.parse_args()

    df = load_runs(args.in_dir)
    if df.empty:
        print(f"no runs yet in {args.in_dir}")
        return 0

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    print("=" * 100)
    print(f"LIVE REPORT — {len(df)} runs collected   ({pd.Timestamp.now().strftime('%H:%M:%S')})")
    print("=" * 100)

    # ----- Per-policy mean+std across all replicates -----
    print("\n=== Per-policy summary (all cells, mean ± std) ===\n")
    cols = ["prefill_ms","decode_tps","peak_kv_mb","peak_rss_kb",
            "perplexity","mean_mass_retained","mean_retention_ratio",
            "mean_eviction_efficiency","evicted_total_decode"]
    cols = [c for c in cols if c in df.columns]
    grp = df.groupby("policy")[cols].agg(['mean','std','count'])
    print(grp.round(3).to_string())

    # ----- Per-(policy, K) breakdown -----
    print("\n=== Per-(policy × K) breakdown ===\n")
    if "k_nominal" in df.columns:
        agg = (df.groupby(["policy","k_nominal"])
                 .agg(n=("prompt_id","count"),
                      prefill_ms=("prefill_ms","mean"),
                      prefill_std=("prefill_ms","std"),
                      decode_tps=("decode_tps","mean"),
                      decode_std=("decode_tps","std"),
                      perplexity=("perplexity","mean"),
                      ppl_std=("perplexity","std"),
                      mass=("mean_mass_retained","mean"),
                      retention=("mean_retention_ratio","mean"),
                      efficiency=("mean_eviction_efficiency","mean"),
                      eff_std=("mean_eviction_efficiency","std"))
                 .reset_index()
                 .sort_values(["k_nominal","policy"]))
        print(agg.round(3).to_string(index=False))

    # ----- Per-(policy, prompt) — what answer were the policies generating -----
    print("\n=== Per-(prompt, policy) effective-KV ===\n")
    if "prompt_id" in df.columns:
        slim = (df.groupby(["prompt_id","policy","k_nominal"])
                  .agg(n=("_rep","count"),
                       decode_tps=("decode_tps","mean"),
                       perplexity=("perplexity","mean"),
                       efficiency=("mean_eviction_efficiency","mean"))
                  .reset_index()
                  .sort_values(["prompt_id","k_nominal","policy"]))
        print(slim.round(3).to_string(index=False))

    # ----- Progress -----
    if "policy" in df.columns and "k_nominal" in df.columns and "prompt_id" in df.columns:
        total_cells_done = df.groupby(["policy","k_nominal","prompt_id"]).size().shape[0]
        print(f"\n=== Coverage: {total_cells_done} distinct (policy, K, prompt) cells filled ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
