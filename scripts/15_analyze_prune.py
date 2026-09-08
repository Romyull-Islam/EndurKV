#!/usr/bin/env python3
"""
Phase E.2 — analyze prune_probe output.

Reads logs/prune_full.csv (or LOG_SUBDIR override).

Computes the headline test of the safety-gate hypothesis:
    Spearman(H_nats,  KL_after_prune)   per K value, per task, and overall.
    Hypothesis confirmed if H_t  positively correlates with KL_t.

Produces:
    figures/prune_01_h_vs_kl_scatter.{pdf,png}    main scatter (with binned mean)
    figures/prune_02_kl_distribution_by_K.{pdf,png}   violin of KL by K
    figures/prune_03_per_task_correlation.{pdf,png}   bar chart, per-task ρ at each K
    figures/prune_summary.json                       all numerical results

Usage:
    LOG_SUBDIR=prune       python3 scripts/15_analyze_prune.py
    LOG_SUBDIR=prune_8b    python3 scripts/15_analyze_prune.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT_FOR_VENV = Path(__file__).resolve().parents[1]
_VENV_DIR = _ROOT_FOR_VENV / ".venv"
_VENV_PY  = _VENV_DIR / "bin" / "python3"
if _VENV_PY.exists():
    try:
        _under_venv = Path(sys.prefix).resolve() == _VENV_DIR.resolve()
    except OSError:
        _under_venv = False
    if not _under_venv:
        os.execv(str(_VENV_PY), [str(_VENV_PY), __file__, *sys.argv[1:]])

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT     = Path(__file__).resolve().parents[1]
LOG_SUB  = os.environ.get("LOG_SUBDIR", "prune")
SUFFIX   = "" if LOG_SUB == "prune" else f"_{LOG_SUB.replace('prune_', '')}"
PRUNE_CSV = ROOT / "logs" / (f"{LOG_SUB}_full.csv" if LOG_SUB != "prune" else "prune_full.csv")
PRUNE_DIR = ROOT / "logs" / LOG_SUB
FIG_DIR   = ROOT / "figures"
SUMMARY   = FIG_DIR / f"prune_summary{SUFFIX}.json"


def main() -> int:
    if not PRUNE_CSV.exists():
        print(f"ERROR: {PRUNE_CSV} missing — run scripts/14_run_prune_study.py first.", file=sys.stderr)
        return 1
    FIG_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(PRUNE_CSV)
    print(f"loaded {len(df)} rows from {PRUNE_CSV.name}")
    print(f"tasks: {sorted(df['task'].unique())}, prompts: {df['prompt_id'].nunique()}")

    # Drop the placeholder step-0 rows (kl_nats == 0 because there was no prior decode).
    df = df[df["step_index"] >= 1].copy()
    print(f"after dropping step==0 placeholders: {len(df)} rows")

    Ks = sorted(df["K"].unique().tolist())
    print(f"K values present: {Ks}")

    summary = {
        "n_rows": int(len(df)),
        "n_prompts": int(df["prompt_id"].nunique()),
        "K_values": [int(k) for k in Ks],
        "by_K": {},
        "by_K_and_task": {},
    }

    # === 01 — H vs KL scatter, one panel per K, with binned mean overlay ===
    n_K = len(Ks)
    fig, axes = plt.subplots(1, n_K, figsize=(5 * n_K, 4.4), sharey=True)
    if n_K == 1: axes = [axes]
    for ax, K in zip(axes, Ks):
        sub = df[df["K"] == K]
        if len(sub) < 5:
            ax.set_title(f"K={K} (n={len(sub)} — too few)")
            continue
        # raw scatter
        ax.scatter(sub["H_nats"], sub["kl_nats"], s=8, alpha=0.25, color="#444")

        # binned mean
        bins = np.unique(np.quantile(sub["H_nats"], np.linspace(0, 1, 11)))
        if len(bins) >= 3:
            bin_idx = np.digitize(sub["H_nats"], bins[1:-1])
            xs, ys, lo, hi = [], [], [], []
            for b in range(int(bin_idx.max()) + 1):
                sel = bin_idx == b
                if sel.sum() < 3: continue
                xs.append(float(np.median(sub["H_nats"].values[sel])))
                ys.append(float(np.mean(sub["kl_nats"].values[sel])))
                lo.append(float(np.percentile(sub["kl_nats"].values[sel], 25)))
                hi.append(float(np.percentile(sub["kl_nats"].values[sel], 75)))
            xs = np.array(xs); ys = np.array(ys); lo = np.array(lo); hi = np.array(hi)
            ax.fill_between(xs, lo, hi, color="#1f5f9b", alpha=0.20)
            ax.plot(xs, ys, color="#0b3d72", linewidth=2.2, marker="o", markersize=6)

        rho, p = stats.spearmanr(sub["H_nats"], sub["kl_nats"])
        r,   pp = stats.pearsonr(sub["H_nats"],  sub["kl_nats"])
        summary["by_K"][str(K)] = {
            "n": int(len(sub)),
            "spearman_rho": float(rho), "spearman_p": float(p),
            "pearson_r":    float(r),   "pearson_p":  float(pp),
            "kl_mean":   float(sub["kl_nats"].mean()),
            "kl_median": float(sub["kl_nats"].median()),
            "kl_p95":    float(sub["kl_nats"].quantile(0.95)),
        }
        ax.set_xlabel("output entropy  H  (nats)")
        ax.set_ylabel("KL(P_full ‖ P_pruned)  (nats)")
        ax.set_title(f"K = {K}     ρ = {rho:+.3f}   p = {p:.1e}    n = {len(sub)}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Direct test of the safety-gate hypothesis:\n"
                 "high entropy ⇔ high KL after pruning K oldest KV entries",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"prune_01_h_vs_kl_scatter{SUFFIX}.pdf")
    fig.savefig(FIG_DIR / f"prune_01_h_vs_kl_scatter{SUFFIX}.png", dpi=200)
    plt.close(fig)

    # === 02 — KL distribution by K (violin) ===
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [df.loc[df["K"] == K, "kl_nats"].values for K in Ks]
    parts = ax.violinplot(data, positions=range(len(Ks)), showmedians=True, widths=0.8)
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([f"K={K}" for K in Ks])
    ax.set_ylabel("KL(P_full ‖ P_pruned)  (nats, log scale)")
    ax.set_yscale("symlog", linthresh=0.001)
    ax.set_title("KL after pruning K oldest KV entries (all decode steps)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"prune_02_kl_distribution_by_K{SUFFIX}.pdf")
    fig.savefig(FIG_DIR / f"prune_02_kl_distribution_by_K{SUFFIX}.png", dpi=200)
    plt.close(fig)

    # === 03 — per-task per-K correlation ===
    tasks = sorted(df["task"].unique())
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(tasks) + 2), 4.5))
    width = 0.8 / max(1, len(Ks))
    x = np.arange(len(tasks))
    palette = {16: "#1f77b4", 64: "#ff7f0e", 256: "#2ca02c"}
    for i, K in enumerate(Ks):
        rhos = []
        for t in tasks:
            s = df[(df["K"] == K) & (df["task"] == t)]
            if len(s) < 5:
                rhos.append(np.nan); continue
            rho, p = stats.spearmanr(s["H_nats"], s["kl_nats"])
            rhos.append(rho)
            summary["by_K_and_task"].setdefault(str(K), {})[t] = {
                "n": int(len(s)),
                "spearman_rho": float(rho), "spearman_p": float(p),
            }
        offset = (i - (len(Ks) - 1) / 2.0) * width
        ax.bar(x + offset, rhos, width, label=f"K={K}",
               color=palette.get(K, None), edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Spearman ρ   (H_nats  vs.  KL_nats)")
    ax.set_title("Per-task safety-gate validation: H predicts KL after pruning")
    ax.legend(title="K oldest evicted")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"prune_03_per_task_correlation{SUFFIX}.pdf")
    fig.savefig(FIG_DIR / f"prune_03_per_task_correlation{SUFFIX}.png", dpi=200)
    plt.close(fig)

    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    # === Headline print ===
    print()
    print("=" * 72)
    print("HEADLINE — direct safety-gate test:  H_nats  vs  KL_nats (after prune)")
    print("=" * 72)
    for K in Ks:
        s = summary["by_K"][str(K)]
        sig = "***" if s["spearman_p"] < 1e-3 else ("**" if s["spearman_p"] < 0.01 else ("*" if s["spearman_p"] < 0.05 else "n.s."))
        print(f"  K={K:<4}  ρ={s['spearman_rho']:+.3f}  p={s['spearman_p']:.2e}  n={s['n']}  "
              f"KL_mean={s['kl_mean']:.4f}  KL_median={s['kl_median']:.4f}  KL_p95={s['kl_p95']:.4f}  {sig}")
    print("=" * 72)
    print(f"figures: {FIG_DIR}/prune_*{SUFFIX}.{{pdf,png}}")
    print(f"summary: {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
