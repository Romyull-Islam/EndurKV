#!/usr/bin/env python3
"""Final results figures for the slide deck:

  1. win_rate_per_regime.png    — bar chart of per-cell wins (v1_linear: 112/120)
  2. median_delta_tova.png      — horizontal grouped bars by regime × policy
  3. pareto_cache_vs_kl.png     — scatter (cache×, median Δ TOVA) per system
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

OUT = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures/architecture_fig")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("/home/mislam22/EndurKV_workspace/EndurKV/figures/llama_wide_results.csv")

# Color scheme — consistent with design-space figure
COLORS = {
    "v1_linear":    "#2e7d32",  # green — ours
    "v6_logistic":  "#66bb6a",  # light green
    "v1_sigmoid":   "#a5d6a7",  # paler green
    "tova":         "#1976d2",  # blue — baseline
    "pyramidkv":    "#7b1fa2",  # purple — strong prior
    "adakv":        "#e65100",  # orange — weak prior
}
PRETTY = {
    "v1_linear":    "EndurKV-Evict v1 (ours)",
    "v6_logistic":  "v6 logistic (ours)",
    "v1_sigmoid":   "v1-sigmoid (overfit)",
    "tova":         "TOVA",
    "pyramidkv":    "PyramidKV",
    "adakv":        "AdaKV",
}
REGIME_PRETTY = {
    "llama1b_short": "Llama-1B short",
    "llama1b_long":  "Llama-1B long",
    "llama8b_short": "Llama-8B short",
    "llama8b_long":  "Llama-8B long",
}
NAVY = "#1a3661"

# ===========================================================================
# Figure 1: Win-rate per regime
# ===========================================================================
regimes = ["llama1b_short","llama1b_long","llama8b_short","llama8b_long"]
policies = ["v1_linear","v6_logistic","v1_sigmoid","pyramidkv","tova","adakv"]

# Count per-cell winners (lowest kl_mean per (prompt, K))
wins_per_regime = {r: {p: 0 for p in policies} for r in regimes}
for regime in regimes:
    sub = df[df.regime==regime]
    for (pid, K), grp in sub.groupby(['prompt_id','K_nominal']):
        winner = grp.loc[grp.kl_mean.idxmin(), 'policy']
        wins_per_regime[regime][winner] += 1

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(regimes))
width = 0.13
for i, p in enumerate(policies):
    counts = [wins_per_regime[r][p] for r in regimes]
    ax.bar(x + i*width - 2.5*width, counts, width,
           label=PRETTY[p], color=COLORS[p], edgecolor='white', linewidth=1)
    # annotate
    for j, c in enumerate(counts):
        if c > 0:
            ax.text(x[j] + i*width - 2.5*width, c + 0.5, str(c),
                    ha='center', fontsize=9, fontweight='bold', color=COLORS[p])

ax.set_xticks(x)
ax.set_xticklabels([REGIME_PRETTY[r] for r in regimes], fontsize=11)
ax.set_ylabel("Per-cell winner count (out of 30)", fontsize=11)
ax.set_ylim(0, 35)
ax.set_title("EndurKV-Evict v1 wins 112 of 120 cells (93%) across 60 Llama prompts",
             fontsize=13, color=NAVY, pad=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.95, ncol=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.set_axisbelow(True)
# Add total at bottom
total_text = "Total cells: 30/regime × 4 regimes = 120.  v1_linear: 112 (93%) · TOVA: 6 (5%) · PyramidKV: 2 (2%)"
plt.figtext(0.5, 0.01, total_text, ha='center', fontsize=10, style='italic', color='#4a4a4a')
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT / "win_rate_per_regime.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(OUT / "win_rate_per_regime.pdf", bbox_inches='tight', facecolor='white')
print(f"saved win_rate_per_regime.png")
plt.close()

# ===========================================================================
# Figure 2: Median Δ TOVA per regime (horizontal grouped bars)
# ===========================================================================
fig, ax = plt.subplots(figsize=(12, 7))
y_positions = []
y_labels = []
y = 0
SHOWN = ["adakv","v1_sigmoid","pyramidkv","tova","v6_logistic","v1_linear"]   # low-to-best
SPACING = 7
GAP = 1.2

for regime in regimes:
    sub = df[df.regime==regime]
    block_top = y
    for p in SHOWN:
        sub_p = sub[sub.policy==p]
        med = float(sub_p.pct_vs_tova.median()) if len(sub_p) else np.nan
        color = COLORS[p]
        ax.barh(y, med, height=0.85, color=color, edgecolor='white', linewidth=0.5,
                label=PRETTY[p] if regime==regimes[0] else None)
        # annotate
        offset = -1.5 if med < 0 else 1.5
        ha = 'right' if med < 0 else 'left'
        ax.text(med + offset, y, f"{med:+.1f}%", ha=ha, va='center', fontsize=9, fontweight='bold')
        y_positions.append(y)
        y_labels.append(f"  {PRETTY[p]}")
        y += 1
    # regime separator
    ax.axhline(y - 0.5, color='#aaaaaa', linewidth=0.5, linestyle='--')
    ax.text(-50, block_top - 0.5, REGIME_PRETTY[regime], ha='left', va='top',
            fontsize=11, fontweight='bold', color=NAVY)
    y += GAP

ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Median Δ KL vs TOVA  (negative = better)", fontsize=11)
ax.set_xlim(-50, 50)
ax.axvline(0, color='#1976d2', linewidth=1.5, alpha=0.6)
ax.set_title("Median KL improvement vs TOVA — per regime",
             fontsize=13, color=NAVY, pad=12, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle=':', alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT / "median_delta_tova_per_regime.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(OUT / "median_delta_tova_per_regime.pdf", bbox_inches='tight', facecolor='white')
print(f"saved median_delta_tova_per_regime.png")
plt.close()

# ===========================================================================
# Figure 3: Pareto plot — (cache×, median Δ TOVA) per policy
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))
for p in policies:
    sub = df[df.policy==p]
    mean_cache = float(sub.cache_ratio_vs_tova.mean())
    med_delta = float(sub.pct_vs_tova.median())
    marker = '*' if p == 'v1_linear' else ('s' if p == 'tova' else 'o')
    msize = 380 if p == 'v1_linear' else (240 if p == 'tova' else 200)
    ax.scatter(mean_cache, med_delta, s=msize, c=COLORS[p], marker=marker,
               edgecolor='black', linewidth=1.2, zorder=5, label=PRETTY[p])
    # label
    dx, dy = 0.012, -1.5
    if p == 'v1_linear': dx, dy = 0.02, -2.5
    if p == 'tova': dx, dy = 0.014, 2.0
    ax.annotate(PRETTY[p], (mean_cache + dx, med_delta + dy), fontsize=10,
                fontweight='bold' if p == 'v1_linear' else 'normal')

# Pareto-front shading
ax.axhline(0, color='#1976d2', linewidth=1.5, alpha=0.5, linestyle='--')
ax.axvline(1.0, color='#666666', linewidth=1.0, alpha=0.4, linestyle=':')
ax.text(1.001, 22, "TOVA cache budget", rotation=90, fontsize=8, color='#666666', va='top')
ax.text(0.84, 0.5, "TOVA baseline (Δ=0)", fontsize=8, color='#1976d2', va='bottom')

ax.set_xlabel("cache× (relative to TOVA)  — lower-left = better", fontsize=11)
ax.set_ylabel("Median Δ KL vs TOVA (%)  — lower = better", fontsize=11)
ax.set_xlim(0.78, 1.06)
ax.set_ylim(-15, 25)
ax.set_title("Pareto: cache cost vs quality on Llama-1B + Llama-8B (720 cells)",
             fontsize=13, color=NAVY, pad=12, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(linestyle=':', alpha=0.4)
ax.set_axisbelow(True)

# Highlight v1_linear as winner
ax.annotate("← Winner: cache-neutral\n   AND lowest KL",
            xy=(1.02, -11), xytext=(0.84, -8),
            fontsize=10, fontweight='bold', color='#2e7d32',
            arrowprops=dict(arrowstyle='->', color='#2e7d32', linewidth=1.5))
plt.tight_layout()
plt.savefig(OUT / "pareto_cache_vs_kl.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(OUT / "pareto_cache_vs_kl.pdf", bbox_inches='tight', facecolor='white')
print(f"saved pareto_cache_vs_kl.png")
plt.close()

print("\nDone. Three new figures:")
for f in ["win_rate_per_regime", "median_delta_tova_per_regime", "pareto_cache_vs_kl"]:
    print(f"  {OUT}/{f}.png")
