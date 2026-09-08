#!/usr/bin/env python3
"""Aggregate per-run meta.json files pulled from the phone into a single CSV
and a side-by-side comparison table.

Usage:
    host_aggregate_phone_logs.py --in-dir <pulled-logs-dir> [--out-csv path]
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="directory containing meta.json files (any depth)")
    ap.add_argument("--out-csv", default="phone_results.csv")
    args = ap.parse_args()

    rows = []
    for mf in Path(args.in_dir).rglob("*.meta.json"):
        try:
            d = json.loads(mf.read_text())
            d["_path"] = str(mf)
            rows.append(d)
        except Exception as e:
            print(f"skip {mf}: {e}")
    if not rows:
        print("no meta.json found"); return 1
    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} runs across {df.policy.nunique()} policies, "
          f"{df.model.nunique()} models, {df.prompt_id.nunique()} prompts")
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")

    # Pivot: per (model, prompt_id, K) → policies columns
    print("\n=== Per-(prompt, K) comparison ===")
    for keyset, grp in df.groupby(["model", "prompt_id", "k_nominal"]):
        m, p, k = keyset
        print(f"\n  {Path(m).stem}  prompt={p}  K={k}:")
        sub = grp[["policy","prefill_ms","decode_tps","peak_kv_cells","peak_rss_kb","evicted_prefill"]]
        sub = sub.sort_values("policy")
        print(sub.to_string(index=False))

    # Side-by-side summary
    print("\n=== POLICY HEADLINE ===")
    agg = df.groupby("policy").agg(
        n=("prompt_id","count"),
        prefill_ms_mean=("prefill_ms","mean"),
        decode_tps_mean=("decode_tps","mean"),
        peak_kv_mean=("peak_kv_cells","mean"),
        peak_rss_mb_mean=("peak_rss_kb", lambda s: s.mean()/1024),
        evicted_prefill_mean=("evicted_prefill","mean"),
    ).reset_index().sort_values("decode_tps_mean", ascending=False)
    print(agg.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
