#!/usr/bin/env python3
"""Cross-variant comparison: v1, v2, v3, v4, v5, v6 vs TOVA.

Runs every perhead variant on the 5 long-context LongBench captures and the
NIAH captures, at multiple K budgets. Computes:
  * mean KL per (model, K, variant)
  * cache ratio (actual_K / K_nominal) per variant
  * cache-matched KL improvement vs TOVA (so configs that use MORE cache
    don't get unfair credit)
  * per-cell rank
  * overall win-rate

Outputs:
  EndurKV/figures/all_variants_comparison.csv
  EndurKV/figures/all_variants_ranked.csv
  EndurKV/figures/all_variants_summary.txt
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import load_attn_perhead, simulate


VARIANTS = ["perhead_v1", "perhead_v2", "perhead_v3",
            "perhead_v4", "perhead_v5", "perhead_v6"]
BASELINE = "perhead_tova"

DIRS = [
    "/home/mislam22/EndurKV_workspace/logs/study_phone_phi3_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_mistral_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_qwen2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_gemma2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_r1distill_longbench",
]
BUDGETS = [256, 512, 1024, 2048]


def run_one(attn_ph, n_kv_at, policy, K):
    kls, Ks, mass, _ = simulate(attn_ph, n_kv_at, None, None, policy, K)
    return float(np.mean(kls)), float(np.mean(Ks)), float(np.mean(mass))


def main() -> int:
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures")
    out_dir.mkdir(exist_ok=True)

    rows = []
    t0 = time.time()
    for d_str in DIRS:
        d = Path(d_str)
        if not d.is_dir():
            print(f"[skip] {d}")
            continue
        model = d.name.replace("study_phone_", "").replace("_longbench", "")
        attn_files = sorted([f for f in d.glob("*.attn.bin")
                             if not f.name.endswith(".v1.attn.bin")])[:2]
        for af in attn_files:
            pid = af.name[:-len(".attn.bin")]
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None: continue
            for K in BUDGETS:
                # TOVA baseline first
                kl_t, Ka_t, mass_t = run_one(attn_ph, n_kv_at, BASELINE, K)
                for var in VARIANTS:
                    kl_v, Ka_v, mass_v = run_one(attn_ph, n_kv_at, var, K)
                    rows.append({
                        "model": model, "prompt_id": pid, "K_nominal": K,
                        "variant": var,
                        "actual_K_var": Ka_v, "actual_K_tova": Ka_t,
                        "cache_ratio": Ka_v / max(Ka_t, 1e-12),
                        "kl_variant": kl_v, "kl_tova": kl_t,
                        "mass_variant": mass_v, "mass_tova": mass_t,
                        "kl_pct_vs_tova": 100 * (kl_v - kl_t) / max(kl_t, 1e-12),
                        "mass_pp_vs_tova": 100 * (mass_v - mass_t),
                    })
        print(f"  [{time.time()-t0:5.0f}s] {model}: {len(attn_files)} prompts done")

    if not rows:
        print("NO DATA")
        return 1
    df = pd.DataFrame(rows)
    csv_path = out_dir / "all_variants_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[compare] wrote {len(df)} rows to {csv_path}")

    # Aggregate per (variant, K) — average across model/prompt
    agg = (df.groupby(["variant", "K_nominal"])
             .agg(n=("model", "count"),
                  mean_cache_ratio=("cache_ratio", "mean"),
                  mean_kl_var=("kl_variant", "mean"),
                  mean_kl_tova=("kl_tova", "mean"),
                  mean_mass_var=("mass_variant", "mean"),
                  raw_kl_pct=("kl_pct_vs_tova", "mean"),
                  raw_mass_pp=("mass_pp_vs_tova", "mean"))
             .reset_index())
    # Cache-matched score: penalize variants that use more cache than TOVA.
    # Score = raw_kl_pct + κ * max(0, (cache_ratio - 1) * 100)
    # where κ=1.0 means "1% extra cache buys you 1% KL improvement for free".
    KAPPA = 1.0
    agg["cache_adjusted_kl_pct"] = (
        agg["raw_kl_pct"] + KAPPA * np.maximum(0.0, (agg["mean_cache_ratio"] - 1.0) * 100))
    ranked = agg.sort_values(["K_nominal", "cache_adjusted_kl_pct"]).reset_index(drop=True)
    ranked_path = out_dir / "all_variants_ranked.csv"
    ranked.to_csv(ranked_path, index=False)
    print(f"[compare] wrote ranked aggregate to {ranked_path}")

    # Text summary
    lines = []
    lines.append("=" * 72)
    lines.append("CROSS-VARIANT COMPARISON: perhead_v1..v6 vs TOVA")
    lines.append("on 5 LongBench long-context (8-10K) captures, 4 budgets each")
    lines.append("=" * 72)
    lines.append("\nCache-adjusted KL improvement (lower = better; "
                 "ratio>1 penalized at κ=1.0):\n")
    for K in BUDGETS:
        sub = ranked[ranked["K_nominal"] == K]
        lines.append(f"\n  K_nominal = {K}:")
        for _, r in sub.iterrows():
            lines.append(f"    {r['variant']:15s}  cache×{r['mean_cache_ratio']:.2f}  "
                         f"raw_KL_Δ={r['raw_kl_pct']:+6.1f}%  "
                         f"cache_adj_Δ={r['cache_adjusted_kl_pct']:+6.1f}%  "
                         f"mass_Δ={r['raw_mass_pp']:+4.1f} pp")
    # Overall winner per K
    lines.append("\n" + "─" * 72)
    lines.append("\nOverall (avg across K) cache-adjusted ranking:\n")
    overall = (ranked.groupby("variant")["cache_adjusted_kl_pct"].mean()
               .sort_values().reset_index())
    for i, r in overall.iterrows():
        lines.append(f"  {i+1}. {r['variant']:15s}  mean cache-adj KL Δ = {r['cache_adjusted_kl_pct']:+6.2f}%")
    summary_path = out_dir / "all_variants_summary.txt"
    summary_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\n[compare] wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
