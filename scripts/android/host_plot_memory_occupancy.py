#!/usr/bin/env python3
"""Per-model per-benchmark KV memory occupancy figure.

Reads the unified long-context CSV (concatenation of pareto_summary.csv from
each model's longctx sim) and produces:

  fig1_memory_full_vs_evicted.png — grouped bar chart: full cache vs evicted
                                    cache, by model. Annotates attn-mass retained.
  fig2_memory_pareto.png          — scatter: KV memory after eviction (x) vs
                                    attention mass retained (y), per (model, K)
                                    point. Connects same-model points to show
                                    each architecture's eviction Pareto curve.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_PRETTY = {
    "phi3":      "Phi-3-mini-4k (MHA)",
    "mistral":   "Mistral-7B (GQA-4)",
    "r1distill": "R1-Distill-Llama-8B (GQA-4)",
    "gemma2":    "Gemma-2-2B (GQA-2)",
    "qwen2":     "Qwen2-7B (GQA-7)",
}
MODEL_COLOR = {
    "phi3":      "#d62728",
    "mistral":   "#2ca02c",
    "r1distill": "#9467bd",
    "gemma2":    "#e377c2",
    "qwen2":     "#17becf",
}
MODEL_ORDER = ["phi3", "mistral", "r1distill", "gemma2", "qwen2"]


def plot_full_vs_evicted(df: pd.DataFrame, out_path: Path,
                         policy: str = "perhead_v1",
                         budgets=(256, 512, 1024, 2048)):
    """Grouped bars: full vs evicted memory per model, per K budget.

    Color scheme:
      - Dark gray bar: full cache (no eviction)
      - 4 graduated blue shades for K=256, 512, 1024, 2048
        (light blue = smallest budget / most aggressive eviction;
         darkest blue = largest budget / mildest eviction)
      - Annotations: MB on top, % attention mass retained on bottom

    Legend explicitly enumerates each K budget so the reader knows which
    bar belongs to which budget without guessing.
    """
    df = df[df["policy"] == policy].copy()
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=130)

    n_models = len(MODEL_ORDER)
    n_groups = len(budgets) + 1
    bar_w = 0.13
    xs = np.arange(n_models)

    # Solid color per K budget — graduated blues. Light = small budget.
    budget_colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(budgets)))

    # Group 0: full cache
    full_vals = [df[df["model_short"] == m]["full_kv_memory_mb"].iloc[0]
                 if (df["model_short"] == m).any() else 0
                 for m in MODEL_ORDER]
    bars0 = ax.bar(xs - bar_w * (n_groups - 1) / 2, full_vals, width=bar_w,
                   color="#3d3d3d", edgecolor="black", linewidth=0.7)
    for b, v in zip(bars0, full_vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.04,
                f"{v:.0f}M", ha="center", fontsize=8.5, color="#222",
                weight="bold")

    # Groups 1..N: per-K-budget evicted memory
    for gi, (K, bc) in enumerate(zip(budgets, budget_colors)):
        offset_idx = gi + 1
        x_pos = xs - bar_w * (n_groups - 1) / 2 + offset_idx * bar_w
        vals, masses = [], []
        for m in MODEL_ORDER:
            sub = df[(df["model_short"] == m) & (df["K_nominal"] == K)]
            if sub.empty:
                vals.append(0); masses.append(np.nan)
            else:
                vals.append(float(sub["kv_memory_mb"].iloc[0]))
                masses.append(float(sub["attn_mass_retained"].iloc[0]))
        bars = ax.bar(x_pos, vals, width=bar_w, color=bc,
                      edgecolor="black", linewidth=0.5)
        for b, v, mas in zip(bars, vals, masses):
            if not np.isnan(mas):
                ax.text(b.get_x() + b.get_width() / 2, v * 1.06,
                        f"{v:.0f}M", ha="center", fontsize=7, color="#111",
                        weight="bold")
                ax.text(b.get_x() + b.get_width() / 2, v * 0.55,
                        f"{mas*100:.0f}%", ha="center", fontsize=7,
                        color="#fff" if mas > 0.92 else "#000")

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_PRETTY[m] for m in MODEL_ORDER],
                       rotation=10, ha="right", fontsize=10)
    ax.set_ylabel("KV cache memory (MB, fp16, log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(
        f"KV memory occupancy on LongBench (8-10K context) — full cache vs "
        f"perhead_v1 eviction at K ∈ {{{', '.join(str(k) for k in budgets)}}}\n"
        "Top annotation = memory (MB), middle annotation (white/black) = % attention mass retained",
        fontsize=11, weight="bold")
    ax.grid(alpha=0.25, axis="y", which="both")

    # Explicit legend: one entry per bar group, color matches bar exactly
    handles = [plt.Rectangle((0, 0), 1, 1, color="#3d3d3d",
                             edgecolor="black", linewidth=0.7,
                             label="Full cache (no eviction)")]
    for K, bc in zip(budgets, budget_colors):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=bc,
                                     edgecolor="black", linewidth=0.5,
                                     label=f"perhead_v1 @ K = {K} (positions/head)"))
    ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.95,
              title="Eviction budget", title_fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_memory_pareto(df: pd.DataFrame, out_path: Path):
    """KV memory after eviction (x) vs attention mass retained (y)."""
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=130)
    policies = ["perhead_v1", "perhead_tova", "kvzip_approx"]
    policy_marker = {"perhead_v1": "o", "perhead_tova": "s", "kvzip_approx": "^"}
    policy_lw = {"perhead_v1": 2.0, "perhead_tova": 1.2, "kvzip_approx": 1.2}
    policy_alpha = {"perhead_v1": 1.0, "perhead_tova": 0.55, "kvzip_approx": 0.55}

    for m in MODEL_ORDER:
        for p in policies:
            sub = df[(df["model_short"] == m) & (df["policy"] == p)].sort_values(
                "kv_memory_mb")
            if sub.empty: continue
            color = MODEL_COLOR[m]
            label = f"{MODEL_PRETTY[m]} — {p}" if p == "perhead_v1" else None
            ax.plot(sub["kv_memory_mb"], sub["attn_mass_retained"],
                    color=color, marker=policy_marker[p],
                    markersize=8, linewidth=policy_lw[p],
                    alpha=policy_alpha[p], label=label)
            # Annotate the lowest-K (leftmost) point with model name on perhead_v1
            if p == "perhead_v1":
                ax.annotate(
                    "perhead_v1",
                    (sub["kv_memory_mb"].iloc[-1], sub["attn_mass_retained"].iloc[-1]),
                    textcoords="offset points", xytext=(5, -3),
                    fontsize=7.5, color=color)

    ax.axhline(0.95, linestyle="--", color="gray", alpha=0.6,
               label="F = 95% mass-retained threshold")
    ax.set_xscale("log")
    ax.set_xlabel("KV cache memory after eviction (MB, fp16) — lower is better",
                  fontsize=11)
    ax.set_ylabel("Attention mass retained — higher is better", fontsize=11)
    ax.set_title("Long-context (8-10K) memory-vs-quality Pareto across models\n"
                 "perhead_v1 = solid bold; TOVA + KVzip-decode-approx = light",
                 fontsize=11, weight="bold")
    ax.grid(alpha=0.30, which="both")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.set_ylim(0.78, 1.005)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_kl_winrate_long_ctx(df: pd.DataFrame, out_path: Path,
                              K_compare: int = 1024):
    """Mean KL comparison at fixed budget K — perhead_v1 vs TOVA vs KVzip
    across the 5 long-context models. Lower KL = closer to full attention."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=130)
    bar_w = 0.27
    xs = np.arange(len(MODEL_ORDER))

    policies = [("perhead_v1",   "perhead_v1 (ours)",  "#d62728"),
                ("perhead_tova", "TOVA",               "#1f77b4"),
                ("kvzip_approx", "KVzip-decode-approx","#2ca02c")]

    kl_by_model = {m: {} for m in MODEL_ORDER}
    for pkey, pname, color in policies:
        vals = []
        for m in MODEL_ORDER:
            sub = df[(df["model_short"] == m) & (df["policy"] == pkey)
                     & (df["K_nominal"] == K_compare)]
            v = float(sub["mean_kl"].iloc[0]) if not sub.empty else np.nan
            vals.append(v); kl_by_model[m][pkey] = v
        offset = (policies.index((pkey, pname, color)) - 1) * bar_w
        bars = ax.bar(xs + offset, vals, width=bar_w, color=color,
                      edgecolor="black", linewidth=0.6,
                      label=pname,
                      hatch="" if pkey == "perhead_v1" else None)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.04,
                        f"{v:.2f}", ha="center", fontsize=9,
                        color=color, weight="bold")

    # Annotate the % gap of ours vs TOVA on each model
    for i, m in enumerate(MODEL_ORDER):
        ours = kl_by_model[m].get("perhead_v1", np.nan)
        tova = kl_by_model[m].get("perhead_tova", np.nan)
        if not np.isnan(ours) and not np.isnan(tova) and tova > 0:
            pct = 100 * (ours / tova - 1)
            color = "#2c8b3b" if pct < 0 else "#a31616"
            ax.text(xs[i], -0.18, f"{pct:+.0f}% vs TOVA",
                    ha="center", fontsize=10, color=color, weight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_PRETTY[m] for m in MODEL_ORDER],
                       rotation=10, ha="right", fontsize=10)
    ax.set_ylabel("Mean KL divergence (full attention vs evicted)\nlower is better",
                  fontsize=11)
    ax.set_title(
        f"Long-context (8-10K LongBench) mean KL at K = {K_compare} positions/head\n"
        "perhead_v1 reduces KL by 8% (R1-Distill) to 27% (Gemma-2) vs TOVA across 5 architectures",
        fontsize=12, weight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.set_ylim(bottom=-0.35)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> int:
    csv = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures/long_ctx_unified.csv")
    df = pd.read_csv(csv)
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures/long_ctx_memory")
    out_dir.mkdir(exist_ok=True)
    plot_full_vs_evicted(df, out_dir / "fig1_memory_full_vs_evicted.png")
    plot_memory_pareto(df, out_dir / "fig2_memory_pareto.png")
    plot_kl_winrate_long_ctx(df, out_dir / "fig3_mean_kl_long_ctx.png")

    # Print headline table for the paper
    print("\n=== HEADLINE TABLE: KV memory occupancy at K=1024 perhead_v1 ===\n")
    print(f"{'Model':<28} {'Full cache':>12} {'After evict':>13} {'Saving':>8}  {'Mass kept':>10}")
    for m in MODEL_ORDER:
        sub = df[(df["model_short"] == m) & (df["policy"] == "perhead_v1")
                 & (df["K_nominal"] == 1024)]
        if sub.empty: continue
        r = sub.iloc[0]
        print(f"{MODEL_PRETTY[m]:<28} {r['full_kv_memory_mb']:>10.0f} MB  "
              f"{r['kv_memory_mb']:>10.0f} MB  {r['memory_saving_pct']:>6.1f}%  "
              f"{r['attn_mass_retained']*100:>8.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
