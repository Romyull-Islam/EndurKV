#!/usr/bin/env python3
"""Quick hyperparameter exploration for EndurKV-Evict v2 (Participation Resonance).

Goal: find a cache-neutral setting (avg actual_K ≈ K_nominal across heads), then
compare KL vs perhead_v1 and TOVA at the same actual_K.

Sweeps (cheap, ~20 configs total):
  α_low ∈ {0.80, 0.85}                # min multiplier (sharp heads)
  α_high ∈ {1.15, 1.20, 1.25, 1.30}   # max multiplier (diffuse heads)
  γ      ∈ {6.0, 10.0}                # logistic steepness
  c0     ∈ {0.45, 0.55}               # sigmoid center
  λ      ∈ {0.0, 0.3, 0.6}            # edge-aware mixing (0 = off)
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import (
    POLICIES, load_attn_perhead, simulate, policy_perhead_v2,
)


def make_v2(alpha_low, alpha_high, gamma, c0, lam):
    def policy(attn_ph_layer_step, K, **kw):
        return policy_perhead_v2(
            attn_ph_layer_step, K,
            alpha_low=alpha_low, alpha_high=alpha_high,
            gamma=gamma, c0=c0, lam=lam, **{k: v for k, v in kw.items() if k != "attn_cum_max"})
    return policy


DIRS = [
    "/home/mislam22/EndurKV_workspace/logs/study_phone_phi3_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_mistral_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_qwen2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_gemma2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_r1distill_longbench",
]
BUDGETS = [512, 1024]   # mid-range, where eviction matters most


def evaluate(alpha_low, alpha_high, gamma, c0, lam, dirs, budgets):
    POLICIES["perhead_v2_x"] = make_v2(alpha_low, alpha_high, gamma, c0, lam)
    rows = []
    for d in dirs:
        d = Path(d)
        attn_files = sorted([f for f in d.glob("*.attn.bin")
                             if not f.name.endswith(".v1.attn.bin")])
        for af in attn_files[:2]:
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None: continue
            for K in budgets:
                kls_v2, Ks_v2, mass_v2, _ = simulate(
                    attn_ph, n_kv_at, None, None, "perhead_v2_x", K)
                kls_v1, Ks_v1, _, _ = simulate(
                    attn_ph, n_kv_at, None, None, "perhead_v1", K)
                kls_t, _, _, _ = simulate(
                    attn_ph, n_kv_at, None, None, "perhead_tova", K)
                rows.append({
                    "alpha_low": alpha_low, "alpha_high": alpha_high,
                    "gamma": gamma, "c0": c0, "lam": lam,
                    "dir": d.name, "K_nominal": K,
                    "actual_K_v2": float(np.mean(Ks_v2)),
                    "actual_K_v1": float(np.mean(Ks_v1)),
                    "kl_v2": float(np.mean(kls_v2)),
                    "kl_v1": float(np.mean(kls_v1)),
                    "kl_tova": float(np.mean(kls_t)),
                    "mass_v2": float(np.mean(mass_v2)),
                })
    return pd.DataFrame(rows)


def main() -> int:
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures/v2_sweep")
    out_dir.mkdir(exist_ok=True, parents=True)

    grid = list(product(
        [0.80, 0.85],        # alpha_low
        [1.15, 1.20, 1.25],  # alpha_high
        [6.0, 10.0],         # gamma
        [0.45, 0.55],        # c0
        [0.0, 0.3],          # lam (off vs moderate edge-aware)
    ))
    print(f"[v2-sweep] {len(grid)} configs × {len(DIRS)} dirs × 2 prompts × {len(BUDGETS)} K")
    frames = []
    for i, (al, ah, g, c, lm) in enumerate(grid):
        df = evaluate(al, ah, g, c, lm, DIRS, BUDGETS)
        frames.append(df)
        if (i + 1) % 4 == 0 or i == len(grid) - 1:
            print(f"  [{i+1:3d}/{len(grid)}] α_lo={al} α_hi={ah} γ={g} c0={c} λ={lm}")
    big = pd.concat(frames, ignore_index=True)
    big["cache_ratio"] = big["actual_K_v2"] / big["K_nominal"]
    big["kl_v2_vs_v1_pct"] = 100 * (big["kl_v2"] - big["kl_v1"]) / big["kl_v1"]
    big["kl_v2_vs_tova_pct"] = 100 * (big["kl_v2"] - big["kl_tova"]) / big["kl_tova"]
    csv = out_dir / "v2_sweep_results.csv"
    big.to_csv(csv, index=False)
    print(f"\n[v2-sweep] wrote {len(big)} rows to {csv}")

    # Aggregate per config
    agg = (big.groupby(["alpha_low", "alpha_high", "gamma", "c0", "lam"])
             .agg(mean_cache_ratio=("cache_ratio", "mean"),
                  mean_kl_v2=("kl_v2", "mean"),
                  mean_kl_v1=("kl_v1", "mean"),
                  mean_kl_tova=("kl_tova", "mean"),
                  v2_vs_v1=("kl_v2_vs_v1_pct", "mean"),
                  v2_vs_tova=("kl_v2_vs_tova_pct", "mean"))
             .reset_index())
    # Focus on cache-neutral configs (ratio within 5% of 1.0)
    neutral = agg[(agg["mean_cache_ratio"] > 0.95) & (agg["mean_cache_ratio"] < 1.05)]
    print("\n=== Cache-neutral configs (0.95 ≤ ratio ≤ 1.05), sorted by v2_vs_v1 ===")
    print(neutral.sort_values("v2_vs_v1").to_string(index=False))
    print("\n=== Top 10 by v2_vs_v1 across ALL configs (any ratio) ===")
    print(agg.sort_values("v2_vs_v1").head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
