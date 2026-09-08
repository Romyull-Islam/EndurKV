#!/usr/bin/env python3
"""Hyperparameter sweep for the perhead_v1 spread gate.

Spread gate (current paper default):
    norm_h = clip((max_a_h - thresh_low) / (thresh_high - thresh_low), 0, 1)
    mult_h = alpha - beta * norm_h
    K_h    = round(K_nominal * mult_h)

Parameters being swept (`--mode quick` is Path A; `--mode full` is Path B):
    alpha       : ceiling of mult_h    (current paper default = 1.3)
    beta        : range of mult_h       (current paper default = 0.6)
                  → mult_h ∈ [alpha - beta, alpha]
    thresh_low  : where the gate starts to bite (default = 0.4)
    thresh_high : where the gate saturates    (default = 0.8)

Outputs:
    sweep_<mode>_results.csv  — per-(config, dir, K) KL + mass retained + win vs TOVA
    sweep_<mode>_heatmap.png  — 2D contour (alpha × beta) of mean KL improvement
    sweep_<mode>_robustness.png — band of (alpha, beta) where we still beat TOVA by ≥5%
"""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import (
    POLICIES, load_attn_perhead, load_kv_sidecar, simulate,
    policy_perhead_tova,
)


def make_parameterized_perhead_v1(alpha: float, beta: float,
                                  thresh_low: float = 0.4,
                                  thresh_high: float = 0.8):
    """Construct a perhead_v1 policy with custom spread-gate parameters."""
    band = thresh_high - thresh_low
    if band <= 0:
        raise ValueError(f"thresh_high must exceed thresh_low ({thresh_high} > {thresh_low})")

    def policy(attn_ph_layer_step, K, **kw):
        nh, nk = attn_ph_layer_step.shape
        if nk <= K:
            return np.ones((nh, nk), dtype=bool)
        m = np.zeros((nh, nk), dtype=bool)
        for h in range(nh):
            a = attn_ph_layer_step[h]
            max_a = float(a.max())
            norm = max(0.0, min(1.0, (max_a - thresh_low) / band))
            mult = alpha - beta * norm
            K_t = max(1, min(nk, int(round(K * mult))))
            idx = np.argpartition(-a, K_t)[:K_t]
            m[h, idx] = True
        return m
    return policy


# ── Capture dirs to sweep on ─────────────────────────────────────────────
LONG_CTX_DIRS = [
    "/home/mislam22/EndurKV_workspace/logs/study_phone_phi3_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_mistral_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_qwen2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_gemma2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_r1distill_longbench",
]
NIAH_DIRS = [
    "/home/mislam22/EndurKV_workspace/logs/study_phone_phi3_niah",
]
BUDGETS = [128, 256, 512, 1024]


def evaluate_config(alpha: float, beta: float, thresh_low: float, thresh_high: float,
                    dirs: list, budgets: list, max_prompts_per_dir: int = 2):
    """Evaluate one (alpha, beta, thresh_low, thresh_high) tuple. Returns DataFrame."""
    # Monkey-patch the parameterized policy into POLICIES so simulate() can find it
    POLICIES["perhead_v1_param"] = make_parameterized_perhead_v1(
        alpha, beta, thresh_low, thresh_high)

    rows = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        attn_files = sorted([f for f in d.glob("*.attn.bin")
                             if not f.name.endswith(".v1.attn.bin")])
        attn_files = attn_files[:max_prompts_per_dir]
        for af in attn_files:
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None:
                continue
            for K in budgets:
                # Param policy
                kls_p, Ks_p, mass_p, _ = simulate(
                    attn_ph, n_kv_at, None, None, "perhead_v1_param", K)
                # TOVA baseline
                kls_t, Ks_t, mass_t, _ = simulate(
                    attn_ph, n_kv_at, None, None, "perhead_tova", K)
                rows.append({
                    "alpha": alpha, "beta": beta,
                    "thresh_low": thresh_low, "thresh_high": thresh_high,
                    "dir": d.name, "prompt_id": af.name[:-len(".attn.bin")],
                    "K_nominal": K,
                    "actual_K_param": float(np.mean(Ks_p)),
                    "actual_K_tova":  float(np.mean(Ks_t)),
                    "kl_param": float(np.mean(kls_p)),
                    "kl_tova":  float(np.mean(kls_t)),
                    "mass_param": float(np.mean(mass_p)),
                    "mass_tova":  float(np.mean(mass_t)),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["kl_improvement_pct"] = 100 * (df["kl_param"] - df["kl_tova"]) / df["kl_tova"]
        df["mass_improvement_pp"] = 100 * (df["mass_param"] - df["mass_tova"])
        df["win_vs_tova"] = df["kl_param"] < df["kl_tova"]
    return df


def run_sweep(grid_alpha: list, grid_beta: list,
              grid_thresh_low: list, grid_thresh_high: list,
              dirs: list, budgets: list, max_prompts: int = 2):
    all_frames = []
    configs = list(product(grid_alpha, grid_beta, grid_thresh_low, grid_thresh_high))
    print(f"[sweep] {len(configs)} configurations × "
          f"{len(dirs)} dirs × {max_prompts} prompts × {len(budgets)} budgets")
    t0 = time.time()
    for i, (a, b, tl, th) in enumerate(configs):
        if tl >= th:
            continue
        df = evaluate_config(a, b, tl, th, dirs, budgets, max_prompts)
        if not df.empty:
            all_frames.append(df)
        if (i + 1) % 5 == 0 or i == len(configs) - 1:
            elapsed = time.time() - t0
            print(f"  [{i+1:3d}/{len(configs)}] α={a:.2f} β={b:.2f} "
                  f"tl={tl:.2f} th={th:.2f}  elapsed={elapsed:.0f}s")
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def plot_heatmap(df: pd.DataFrame, out_dir: Path, mode: str):
    """For each (thresh_low, thresh_high), render an alpha × beta heatmap of
    mean KL improvement (% reduction vs TOVA). Lower (more negative) = better."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        print("[plot] no data, skipping heatmap")
        return

    # Aggregate: mean KL improvement per (alpha, beta, thresh_low, thresh_high)
    agg = (df.groupby(["alpha", "beta", "thresh_low", "thresh_high"])
             .agg(mean_kl_imp=("kl_improvement_pct", "mean"),
                  mean_mass_imp=("mass_improvement_pp", "mean"),
                  winrate=("win_vs_tova", "mean"))
             .reset_index())

    # One subplot per (thresh_low, thresh_high) combo
    pairs = sorted(set(zip(agg["thresh_low"], agg["thresh_high"])))
    n = len(pairs)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows),
                             dpi=130, squeeze=False)
    fig.suptitle(
        f"perhead_v1 hyperparameter sweep ({mode}) — KL improvement vs TOVA (%)\n"
        f"Cells: negative = perhead_v1 BEATS TOVA. Current paper setting α=1.3, β=0.6 highlighted.",
        fontsize=12, weight="bold", y=1.00)

    for ax_idx, (tl, th) in enumerate(pairs):
        sub = agg[(agg["thresh_low"] == tl) & (agg["thresh_high"] == th)]
        pivot = sub.pivot(index="beta", columns="alpha", values="mean_kl_imp")
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                       vmin=-30, vmax=10, origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r:.1f}" for r in pivot.index])
        ax.set_xlabel("α (max budget multiplier)")
        ax.set_ylabel("β (range of multiplier)")
        ax.set_title(f"thresh_low={tl}, thresh_high={th}", fontsize=10)
        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 15 else "black"
                    ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                            fontsize=9, color=color)
        # Mark the paper default (alpha=1.3, beta=0.6) if in grid
        if 1.3 in list(pivot.columns) and 0.6 in list(pivot.index):
            j_def = list(pivot.columns).index(1.3)
            i_def = list(pivot.index).index(0.6)
            ax.add_patch(plt.Rectangle((j_def - 0.5, i_def - 0.5), 1, 1,
                                       fill=False, edgecolor="#0033cc",
                                       linewidth=2.5))
        plt.colorbar(im, ax=ax, fraction=0.05, label="KL Δ (%)")
    # Hide extra subplots if any
    for k in range(len(pairs), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    out_path = out_dir / f"sweep_{mode}_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

    # Robustness band: cells where mean improvement <= -5%
    robust = agg[agg["mean_kl_imp"] <= -5].sort_values("mean_kl_imp")
    print(f"\n=== {len(robust)} configs achieve ≥5% KL reduction vs TOVA ===")
    if not robust.empty:
        print(robust.head(20).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick",
                    help="quick = 5×5 α×β at fixed thresh; full = 5×5×3×4")
    ap.add_argument("--out-dir", default="/home/mislam22/EndurKV_workspace/EndurKV/figures/sweep")
    ap.add_argument("--max-prompts", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    dirs = LONG_CTX_DIRS  # focus on long-context — the publication regime

    if args.mode == "quick":
        grid_alpha = [1.1, 1.2, 1.3, 1.4, 1.5]
        grid_beta  = [0.4, 0.5, 0.6, 0.7, 0.8]
        grid_tl    = [0.4]
        grid_th    = [0.8]
    else:  # full
        grid_alpha = [1.1, 1.2, 1.3, 1.4, 1.5]
        grid_beta  = [0.4, 0.5, 0.6, 0.7, 0.8]
        grid_tl    = [0.2, 0.3, 0.4]
        grid_th    = [0.6, 0.7, 0.8, 0.9]

    print(f"[sweep] mode={args.mode}: α∈{grid_alpha}, β∈{grid_beta}, "
          f"tl∈{grid_tl}, th∈{grid_th}")
    df = run_sweep(grid_alpha, grid_beta, grid_tl, grid_th, dirs, BUDGETS,
                   args.max_prompts)
    if df.empty:
        print("ERROR: no rows produced", file=sys.stderr)
        return 1

    csv_path = out_dir / f"sweep_{args.mode}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[sweep] wrote {len(df)} rows to {csv_path}")

    plot_heatmap(df, out_dir, args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
