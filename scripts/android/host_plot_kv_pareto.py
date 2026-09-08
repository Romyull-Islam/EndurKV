#!/usr/bin/env python3
"""Pareto plots for KV cache eviction policies.

Reads pareto_summary.csv produced by host_simulate_kv_baselines.py and renders
one Pareto curve per policy. Field-standard format used by KVzip / PruLong /
R-KV: x-axis = actual cache budget (avg_actual_K), y-axis = quality metric.

Three plot variants:
  * attn_mass_retained  — fraction of attention mass retained (higher better)
  * mean_kl             — KL divergence (lower better)
  * needle_hit_rate     — NIAH answer position retention (higher better; only
                          rendered if at least one row has non-NaN values)

Each variant emits a PNG; the script also prints a critical-KV-footprint table
(smallest actual_K at which each policy hits F=90% of the best policy's
quality at full cache).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Color/marker per policy — stable across plots.
STYLE = {
    "perhead_v1":   ("#d62728", "o", "perhead_v1 (ours)"),
    "perhead_tova": ("#1f77b4", "s", "TOVA"),
    "kvzip_approx": ("#2ca02c", "^", "KVzip-decode-approx"),
    "laprox":       ("#ff7f0e", "D", "LaProx"),
    "rkv":          ("#9467bd", "P", "R-KV (λ=0.1)"),
    "keydiff":      ("#8c564b", "X", "KeyDiff"),
    "perhead_tova_spread": ("#17becf", "v", "perhead_tova_spread"),
}


def pareto_frontier(xs, ys, x_better: str = "lower", y_better: str = "higher"):
    """Return the indices of the points lying on the Pareto frontier.

    A point dominates another iff it is no worse on both axes and strictly
    better on at least one. With x_better='lower' and y_better='higher', this
    means: sort by x ascending (ties broken by 'better' y), sweep, and keep
    only points whose y strictly improves the running best.

    Returns (frontier_indices, dominated_indices) as Python lists of ints into
    the original arrays.
    """
    n = len(xs)
    if n == 0:
        return [], []
    idx = list(range(n))
    # Sort by x in the direction such that earlier == "lower-cost" candidates.
    idx.sort(key=lambda i: (xs[i] if x_better == "lower" else -xs[i],
                            -(ys[i] if y_better == "higher" else -ys[i])))
    frontier = []
    dominated = []
    best_y = None
    for i in idx:
        yi = ys[i]
        if best_y is None:
            frontier.append(i)
            best_y = yi
            continue
        if y_better == "higher":
            if yi > best_y:
                frontier.append(i)
                best_y = yi
            else:
                dominated.append(i)
        else:  # lower y is better
            if yi < best_y:
                frontier.append(i)
                best_y = yi
            else:
                dominated.append(i)
    return frontier, dominated


def plot_pareto(df: pd.DataFrame, metric: str, ylabel: str, higher_is_better: bool,
                out_path: Path, title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    policies = sorted(df["policy"].unique(),
                      key=lambda p: list(STYLE.keys()).index(p) if p in STYLE else 99)
    for pol in policies:
        sub = df[df["policy"] == pol].sort_values("avg_actual_K").reset_index(drop=True)
        if sub[metric].isna().all():
            continue
        color, marker, label = STYLE.get(pol, ("#777777", "o", pol))
        xs = sub["avg_actual_K"].tolist()
        ys = sub[metric].tolist()
        # Filter to the Pareto frontier (lower budget + better quality
        # dominates). Plot frontier as solid line + bold markers and the
        # dominated points as faded scatter for honesty.
        front_idx, dom_idx = pareto_frontier(
            xs, ys,
            x_better="lower",
            y_better=("higher" if higher_is_better else "lower"),
        )
        front_x = [xs[i] for i in front_idx]
        front_y = [ys[i] for i in front_idx]
        # Sort frontier by x for a clean line.
        order = sorted(range(len(front_x)), key=lambda k: front_x[k])
        front_x = [front_x[k] for k in order]
        front_y = [front_y[k] for k in order]
        ax.plot(front_x, front_y, color=color, marker=marker,
                label=label, linewidth=1.7, markersize=7)
        if dom_idx:
            dom_x = [xs[i] for i in dom_idx]
            dom_y = [ys[i] for i in dom_idx]
            ax.scatter(dom_x, dom_y, color=color, marker=marker,
                       alpha=0.25, s=35, edgecolors="none")
    ax.set_xlabel("Actual KV cache budget (avg positions per head)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    if higher_is_better:
        ax.set_ylim(top=min(1.005, max(ax.get_ylim()[1], 1.0)))
    title = f"Pareto frontier: {metric}"
    if title_suffix: title += f"  ({title_suffix})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def critical_footprint(df: pd.DataFrame, metric: str, threshold: float,
                       higher_is_better: bool) -> pd.DataFrame:
    """Smallest actual_K where each policy meets threshold."""
    rows = []
    for pol in sorted(df["policy"].unique()):
        sub = df[df["policy"] == pol].sort_values("avg_actual_K")
        if higher_is_better:
            ok = sub[sub[metric] >= threshold]
        else:
            ok = sub[sub[metric] <= threshold]
        if ok.empty:
            rows.append({"policy": pol, "critical_K": float("nan"),
                         "metric_at_critical": float("nan")})
        else:
            r = ok.iloc[0]
            rows.append({"policy": pol,
                         "critical_K": float(r["avg_actual_K"]),
                         "metric_at_critical": float(r[metric])})
    return pd.DataFrame(rows).sort_values("critical_K")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True,
                    help="pareto_summary.csv from host_simulate_kv_baselines.py")
    ap.add_argument("--out-dir", default="",
                    help="Output dir for PNGs (defaults to summary's parent)")
    ap.add_argument("--mass-threshold", type=float, default=0.95,
                    help="Critical-footprint threshold for attn_mass_retained")
    ap.add_argument("--needle-threshold", type=float, default=0.90,
                    help="Critical-footprint threshold for needle_hit_rate")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.summary).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mass-retained plot (always available)
    plot_pareto(df, "attn_mass_retained",
                "Attention mass retained (fraction)",
                higher_is_better=True,
                out_path=out_dir / "pareto_attn_mass.png",
                title_suffix="higher is better")

    # KL plot
    plot_pareto(df, "mean_kl",
                "Mean KL divergence",
                higher_is_better=False,
                out_path=out_dir / "pareto_mean_kl.png",
                title_suffix="lower is better")

    # Needle hit rate plot — only if non-NaN data exists
    if "needle_hit_rate" in df.columns and df["needle_hit_rate"].notna().any():
        plot_pareto(df, "needle_hit_rate",
                    "NIAH needle hit rate (kept-head fraction)",
                    higher_is_better=True,
                    out_path=out_dir / "pareto_needle_hit.png",
                    title_suffix="higher is better")

    # Print critical-footprint tables
    print("\n=== Critical KV footprint (attn_mass_retained ≥ "
          f"{args.mass_threshold}) ===")
    print(critical_footprint(df, "attn_mass_retained",
                              args.mass_threshold, True).to_string(index=False))

    if "needle_hit_rate" in df.columns and df["needle_hit_rate"].notna().any():
        print("\n=== Critical KV footprint (needle_hit_rate ≥ "
              f"{args.needle_threshold}) ===")
        print(critical_footprint(df, "needle_hit_rate",
                                  args.needle_threshold, True).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
