"""Per-head Pareto plotter.

Reads a `<study>_perhead/pareto_summary.csv` from host_simulate_eviction_perhead.py
and produces:

  1. `perhead_pareto.png`  — KL vs avg_actual_K curves for all per-head policies
                              (highlights v1 + endurkv variants vs baselines)
  2. `perhead_matched_K.png` — bar chart of mean_KL at the K_nominal closest to
                                a target budget, grouped by policy family
                                (per-head TOVA family, AdaKV family, EndurKV family)
  3. `perhead_summary.csv` — pivot table of mean_KL by (policy, K_nominal)

Usage:
  python host_plot_perhead.py <perhead_dir> [--target-K 128]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Style: family-coloured palette
PALETTE = {
    "full":                ("black",       ":",  1.0, "Full cache (ref)"),
    "local":               ("gray",        ":",  1.0, "Local window"),
    "perhead_tova":        ("tab:blue",    "--", 1.5, "TOVA (per-head)"),
    "perhead_v1":          ("tab:red",     "-",  2.5, "EndurKV-Evict v1 (per-head)"),
    "adakv":               ("tab:purple",  "-",  1.3, "AdaKV"),
    "headkv":              ("tab:orange",  "-",  1.3, "HeadKV"),
    "duoattention":        ("tab:brown",   "-",  1.3, "DuoAttention"),
    "endurkv_perhead":     ("tab:cyan",    "-",  1.2, "EndurKV (consensus)"),
    "endurkv_disagree":    ("tab:olive",   "-",  1.2, "EndurKV (disagree)"),
    "perhead_volatility":  ("tab:pink",    "-",  1.2, "EndurKV (volatility)"),
    "perhead_svd":         ("tab:green",   "-",  1.2, "EndurKV (SVD)"),
    "endurkv_perhead_v2":  ("crimson",     "-",  2.0, "EndurKV-Evict v2 (combined)"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("perhead_dir", help="Path to *_perhead dir containing pareto_summary.csv")
    ap.add_argument("--target-K", type=int, default=128,
                    help="K_nominal for the matched-K bar chart")
    args = ap.parse_args()

    d = Path(args.perhead_dir)
    summary_csv = d / "pareto_summary.csv"
    if not summary_csv.exists():
        print(f"ERROR: {summary_csv} not found"); return 1
    df = pd.read_csv(summary_csv)

    # Style assertion: required columns
    for c in ("policy", "K_nominal", "mean_kl", "avg_actual_K"):
        if c not in df.columns:
            print(f"ERROR: column {c!r} missing from {summary_csv}"); return 1

    # ============ Plot 1: Pareto curve =============
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — abort"); return 1

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for pol, g in df.groupby("policy"):
        if pol not in PALETTE:
            continue
        color, ls, lw, lbl = PALETTE[pol]
        g_sorted = g.sort_values("avg_actual_K")
        ax.plot(g_sorted["avg_actual_K"], g_sorted["mean_kl"],
                color=color, linestyle=ls, linewidth=lw,
                marker="o", markersize=4, label=lbl, alpha=0.9)
    ax.set_xlabel("avg actual K per head (tokens)", fontsize=10)
    ax.set_ylabel("mean KL (full vs evicted, per-head average)", fontsize=10)
    ax.set_title(f"Per-head Pareto — {d.name}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.legend(fontsize=8, loc="upper right", ncol=1, framealpha=0.85)
    fig.tight_layout()
    out_png = d / "perhead_pareto.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")
    plt.close(fig)

    # ============ Plot 2: Matched-K bar chart =============
    target = args.target_K
    closest_K = df["K_nominal"].iloc[(df["K_nominal"] - target).abs().argsort()[:1]].iloc[0]
    sub = df[df["K_nominal"] == closest_K].copy()
    if sub.empty:
        print(f"No data at K_nominal closest to {target}")
        return 0
    # Sort by KL ascending (best first)
    sub = sub.sort_values("mean_kl")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE.get(p, ("lightgray", "-", 1.0, p))[0] for p in sub["policy"]]
    labels = [PALETTE.get(p, ("lightgray", "-", 1.0, p))[3] for p in sub["policy"]]
    bars = ax.barh(range(len(sub)), sub["mean_kl"], color=colors)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean KL (lower = better)", fontsize=10)
    ax.set_title(f"Per-head policy ranking at K={closest_K}", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    # Annotate values
    for i, (b, v, k) in enumerate(zip(bars, sub["mean_kl"], sub["avg_actual_K"])):
        ax.text(v + 0.02, i, f"  {v:.3f} (K={k:.0f})", va="center", fontsize=7)
    fig.tight_layout()
    out_bar = d / f"perhead_matched_K{closest_K}.png"
    fig.savefig(out_bar, dpi=140, bbox_inches="tight")
    print(f"wrote {out_bar}")
    plt.close(fig)

    # ============ CSV pivot for paper inclusion =============
    pivot = df.pivot_table(index="policy", columns="K_nominal",
                           values=["mean_kl", "avg_actual_K"],
                           aggfunc="mean")
    pivot.to_csv(d / "perhead_pivot.csv")
    print(f"wrote {d / 'perhead_pivot.csv'}")

    # Headline numbers
    print(f"\n=== matched-K (K_nominal={closest_K}) ranking ===")
    for _, r in sub.iterrows():
        lbl = PALETTE.get(r["policy"], (None, None, None, r["policy"]))[3]
        print(f"  {lbl:<32}  KL={r['mean_kl']:.3f}  avg_K={r['avg_actual_K']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
