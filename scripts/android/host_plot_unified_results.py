#!/usr/bin/env python3
"""Slide figures from the unified all-models dataset (1008 rows, 7 models, 5 datasets).

Output:
  win_rate_per_model.png      — per-model win counts
  win_rate_per_dataset.png    — per-dataset win counts
  median_delta_tova_grid.png  — 7×6 heatmap of median Δ TOVA per (model, policy)
  pareto_cache_vs_kl_all.png  — Pareto plot across all 168 cells
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

OUT = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures/architecture_fig")
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv("/home/mislam22/EndurKV_workspace/EndurKV/figures/unified_all_models_results.csv")

COLORS = {
    "v1_linear":    "#2e7d32",
    "v6_logistic":  "#66bb6a",
    "v1_sigmoid":   "#a5d6a7",
    "tova":         "#1976d2",
    "pyramidkv":    "#7b1fa2",
    "adakv":        "#e65100",
}
PRETTY = {
    "v1_linear":   "EndurKV-Evict v1 (ours)",
    "v6_logistic": "v6 logistic (ours)",
    "v1_sigmoid":  "v1-sigmoid (ablation)",
    "tova":        "TOVA",
    "pyramidkv":   "PyramidKV",
    "adakv":       "AdaKV",
}
MODEL_PRETTY = {
    "gemma2":   "Gemma-2-2B",
    "llama1b":  "Llama-3.2-1B",
    "llama8b":  "Llama-3.1-8B",
    "mistral":  "Mistral-7B",
    "phi3":     "Phi-3-mini",
    "qwen2":    "Qwen2-7B",
    "r1distill":"R1-Distill-8B",
}
DATASET_PRETTY = {
    "short":    "Short (≤1K)",
    "long":     "Long (Llama, ~10K)",
    "longbench":"LongBench (~10K)",
    "niah":     "NIAH (~4K)",
    "reasoning":"Reasoning (CoT)",
}
NAVY = "#1a3661"
policies = ["v1_linear","v6_logistic","v1_sigmoid","pyramidkv","tova","adakv"]

# ===========================================================================
# Figure 1: Win-rate per MODEL (7 models)
# ===========================================================================
models = sorted(df.model.unique())
win_counts = {m: {p:0 for p in policies} for m in models}
for m in models:
    sub = df[df.model==m]
    for (ds, pid, K), grp in sub.groupby(['dataset','prompt_id','K_nominal']):
        winner = grp.loc[grp.kl_mean.idxmin(), 'policy']
        win_counts[m][winner] += 1

fig, ax = plt.subplots(figsize=(13, 6.5))
x = np.arange(len(models))
width = 0.13
for i, p in enumerate(policies):
    counts = [win_counts[m][p] for m in models]
    bars = ax.bar(x + i*width - 2.5*width, counts, width,
                  label=PRETTY[p], color=COLORS[p], edgecolor='white', linewidth=1)
    for j, c in enumerate(counts):
        if c > 0:
            ax.text(x[j] + i*width - 2.5*width, c + 0.4, str(c),
                    ha='center', fontsize=8.5, fontweight='bold', color=COLORS[p])

ax.set_xticks(x)
ax.set_xticklabels([MODEL_PRETTY[m] for m in models], fontsize=10, rotation=12, ha='right')
ax.set_ylabel("Per-cell winner count", fontsize=11)
# Per-model total cells:
totals = [sum(win_counts[m].values()) for m in models]
ax.set_ylim(0, max(totals) + 8)
ax.set_title("Per-cell winner count by model — v1_linear wins 144/168 (86%) overall",
             fontsize=13, color=NAVY, pad=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9.5, framealpha=0.95, ncol=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.set_axisbelow(True)
# Per-model total annotations
for i, m in enumerate(models):
    ax.text(x[i], -3.5, f"n = {totals[i]}", ha='center', fontsize=8, color='#666666')
plt.tight_layout()
plt.savefig(OUT / "win_rate_per_model.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(OUT / "win_rate_per_model.pdf", bbox_inches='tight', facecolor='white')
print("saved win_rate_per_model.png")
plt.close()

# ===========================================================================
# Figure 2: Win-rate per DATASET (5 datasets)
# ===========================================================================
datasets = ["short","long","longbench","niah","reasoning"]
win_counts_ds = {d: {p:0 for p in policies} for d in datasets}
for d_ in datasets:
    sub = df[df.dataset==d_]
    for (m, pid, K), grp in sub.groupby(['model','prompt_id','K_nominal']):
        winner = grp.loc[grp.kl_mean.idxmin(), 'policy']
        win_counts_ds[d_][winner] += 1

fig, ax = plt.subplots(figsize=(13, 6.5))
x = np.arange(len(datasets))
for i, p in enumerate(policies):
    counts = [win_counts_ds[d_][p] for d_ in datasets]
    ax.bar(x + i*width - 2.5*width, counts, width,
           label=PRETTY[p], color=COLORS[p], edgecolor='white', linewidth=1)
    for j, c in enumerate(counts):
        if c > 0:
            ax.text(x[j] + i*width - 2.5*width, c + 1.0, str(c),
                    ha='center', fontsize=8.5, fontweight='bold', color=COLORS[p])

ax.set_xticks(x)
ax.set_xticklabels([DATASET_PRETTY[d_] for d_ in datasets], fontsize=10)
ax.set_ylabel("Per-cell winner count", fontsize=11)
totals_ds = [sum(win_counts_ds[d_].values()) for d_ in datasets]
ax.set_ylim(0, max(totals_ds) + 12)
ax.set_title("Per-cell winner count by dataset / context regime",
             fontsize=13, color=NAVY, pad=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95, ncol=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.set_axisbelow(True)
for i, d_ in enumerate(datasets):
    ax.text(x[i], -4.0, f"n = {totals_ds[i]}", ha='center', fontsize=8, color='#666666')
plt.tight_layout()
plt.savefig(OUT / "win_rate_per_dataset.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(OUT / "win_rate_per_dataset.pdf", bbox_inches='tight', facecolor='white')
print("saved win_rate_per_dataset.png")
plt.close()

# ===========================================================================
# Figure 3: Median Δ TOVA heatmap (model × policy)
# ===========================================================================
mat = np.zeros((len(models), len(policies)))
for i, m in enumerate(models):
    for j, p in enumerate(policies):
        sub = df[(df.model==m) & (df.policy==p)]
        mat[i, j] = float(sub.pct_vs_tova.median()) if len(sub) else np.nan

fig, ax = plt.subplots(figsize=(11, 6))
vmax = 30; vmin = -30
im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
ax.set_xticks(range(len(policies)))
ax.set_xticklabels([PRETTY[p] for p in policies], rotation=15, ha='right', fontsize=10)
ax.set_yticks(range(len(models)))
ax.set_yticklabels([MODEL_PRETTY[m] for m in models], fontsize=10)
for i in range(len(models)):
    for j in range(len(policies)):
        v = mat[i, j]
        color = 'white' if abs(v) > 18 else 'black'
        ax.text(j, i, f"{v:+.1f}%", ha='center', va='center',
                fontsize=10, color=color, fontweight='bold')
plt.colorbar(im, ax=ax, label='Median Δ KL vs TOVA (%, negative=better)')
ax.set_title("Median Δ KL vs TOVA — per model × policy",
             fontsize=13, color=NAVY, pad=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / "median_delta_tova_grid.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(OUT / "median_delta_tova_grid.pdf", bbox_inches='tight', facecolor='white')
print("saved median_delta_tova_grid.png")
plt.close()

# ===========================================================================
# Figure 4: Pareto plot across ALL data
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))
for p in policies:
    sub = df[df.policy==p]
    mean_cache = float(sub.cache_ratio_vs_tova.mean())
    med_delta = float(sub.pct_vs_tova.median())
    marker = '*' if p == 'v1_linear' else ('s' if p == 'tova' else 'o')
    msize = 420 if p == 'v1_linear' else (240 if p == 'tova' else 200)
    ax.scatter(mean_cache, med_delta, s=msize, c=COLORS[p], marker=marker,
               edgecolor='black', linewidth=1.2, zorder=5, label=PRETTY[p])

# Labels
labels = {
    'v1_linear':   (1.045, -11.3, (0.04, -2.5)),
    'v6_logistic': (0.92, -0.75,  (0.01, 2.0)),
    'v1_sigmoid':  (0.88, 1.0,    (0.01, -2.5)),
    'pyramidkv':   (1.00, -1.3,   (0.01, 2.5)),
    'tova':        (1.00, 0,      (0.01, -2.5)),
    'adakv':       (1.00, 20.4,   (0.01, 1.5)),
}
for p, (x_, y_, (dx, dy)) in labels.items():
    ax.annotate(PRETTY[p], (x_ + dx, y_ + dy), fontsize=10,
                fontweight='bold' if p == 'v1_linear' else 'normal')

ax.axhline(0, color='#1976d2', linewidth=1.5, alpha=0.5, linestyle='--')
ax.axvline(1.0, color='#666666', linewidth=1.0, alpha=0.4, linestyle=':')
ax.text(1.001, 28, "TOVA cache budget", rotation=90, fontsize=8, color='#666666', va='top')
ax.set_xlabel("cache× (relative to TOVA) — lower-left = better", fontsize=11)
ax.set_ylabel("Median Δ KL vs TOVA (%) — lower = better", fontsize=11)
ax.set_xlim(0.85, 1.10)
ax.set_ylim(-15, 30)
ax.set_title("Pareto: cache cost vs quality — 168 cells across 7 models × 5 datasets",
             fontsize=13, color=NAVY, pad=12, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(linestyle=':', alpha=0.4)
ax.set_axisbelow(True)
ax.annotate("← Winner: 86% per-cell\n   wins, −11.3% median KL",
            xy=(1.045, -11.3), xytext=(0.88, -9),
            fontsize=10.5, fontweight='bold', color='#2e7d32',
            arrowprops=dict(arrowstyle='->', color='#2e7d32', linewidth=1.5))
plt.tight_layout()
plt.savefig(OUT / "pareto_cache_vs_kl_all.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(OUT / "pareto_cache_vs_kl_all.pdf", bbox_inches='tight', facecolor='white')
print("saved pareto_cache_vs_kl_all.png")
plt.close()

print("\nAll 4 unified-results figures rendered to:", OUT)
