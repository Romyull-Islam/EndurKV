#!/bin/bash
# Queued runner for the full novel-variant experiment:
#   1. Wait for v2 hyperparameter sweep to finish (PID-based wait).
#   2. Run cross-variant comparison (v1..v6 vs TOVA on 5 LongBench dirs).
#   3. Write a final consolidated summary.
#
# Run with: nohup bash this_script.sh > queue.log 2>&1 &

set -e
cd /home/mislam22/EndurKV_workspace

echo "[queue] $(date) — waiting for v2 sweep to finish ..."
# Wait for any python process running host_sweep_perhead_v2 to exit
while pgrep -f host_sweep_perhead_v2 > /dev/null 2>&1; do
    sleep 30
done
echo "[queue] $(date) — v2 sweep done, starting cross-variant comparison"

source .venv/bin/activate

# Cross-variant comparison
echo "[queue] [1/2] cross-variant comparison ..."
python EndurKV/scripts/android/host_compare_all_variants.py

# Re-run v2 with its BEST hyperparameters (from v2_sweep) and include in summary
echo "[queue] [2/2] consolidated final report ..."
python -c "
import pandas as pd
from pathlib import Path
fig = Path('/home/mislam22/EndurKV_workspace/EndurKV/figures')

print()
print('=' * 72)
print('CONSOLIDATED NOVEL-VARIANT REPORT')
print('=' * 72)
print()

# v2 sweep optimum
v2_csv = fig / 'v2_sweep/v2_sweep_results.csv'
if v2_csv.exists():
    df = pd.read_csv(v2_csv)
    agg = (df.groupby(['alpha_low','alpha_high','gamma','c0','lam'])
             .agg(mean_cache_ratio=('actual_K_v2', lambda s: float((s / df.loc[s.index,'K_nominal']).mean())),
                  mean_kl_v2=('kl_v2','mean'),
                  mean_kl_v1=('kl_v1','mean'),
                  mean_kl_tova=('kl_tova','mean'))
             .reset_index())
    agg['v2_vs_v1_pct'] = 100 * (agg['mean_kl_v2']/agg['mean_kl_v1'] - 1)
    agg['v2_vs_tova_pct'] = 100 * (agg['mean_kl_v2']/agg['mean_kl_tova'] - 1)
    neutral = agg[(agg.mean_cache_ratio > 0.95) & (agg.mean_cache_ratio < 1.05)]
    best_neutral = neutral.sort_values('v2_vs_v1_pct').head(3) if not neutral.empty else agg.sort_values('v2_vs_v1_pct').head(3)
    print('v2 best CACHE-NEUTRAL configs:')
    print(best_neutral.to_string(index=False))
    print()

# Cross-variant overall ranking
ranked = fig / 'all_variants_ranked.csv'
if ranked.exists():
    rd = pd.read_csv(ranked)
    overall = rd.groupby('variant')['cache_adjusted_kl_pct'].mean().sort_values().reset_index()
    print('Cross-variant overall (cache-adjusted) ranking:')
    print(overall.to_string(index=False))
"

echo "[queue] $(date) — DONE."
