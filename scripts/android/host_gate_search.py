#!/usr/bin/env python3
"""Finer hyperparameter + gate-shape search for v1.

Goal: find a config with cache× ≤ 1.01 and Δ TOVA ≤ −10% (or as close as possible).

Test set:
  5 models × 1 long-context prompt × K ∈ {512, 1024} = 10 cells per config.

Configs tested:
  (A) 4-parameter linear-clipped sweep: (alpha, beta, thresh_low, thresh_high)
      Tighter thresh windows + softened endpoints to widen per-head spread
      while keeping average cache near 1.0.
  (B) Sigmoid gates: smoother saturation, parameterised by (alpha, beta,
      center c, steepness gamma).
  (C) Quadratic gates: steeper near saturation (more bimodal allocation).
  (D) Step gates: hard threshold — sharp heads get β, diffuse get α.

Output:
  EndurKV/figures/gate_search_results.csv
  EndurKV/figures/gate_search_ranking.csv
  console: cache-neutral champion
"""
import sys, time
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import load_attn_perhead, simulate, POLICIES


# ---------------------------------------------------------------------------
# Gate-shape policy factory
# ---------------------------------------------------------------------------

def make_linear_clipped(alpha, beta, tl, th):
    def policy(attn_ph_layer_step, K, **kw):
        nh, nk = attn_ph_layer_step.shape
        if nk <= K: return np.ones((nh, nk), dtype=bool)
        m = np.zeros((nh, nk), dtype=bool)
        denom = max(th - tl, 1e-6)
        for h in range(nh):
            a = attn_ph_layer_step[h]
            max_a = float(a.max())
            norm = max(0.0, min(1.0, (max_a - tl) / denom))
            mult = alpha - (alpha - beta) * norm
            K_t = max(1, min(nk, int(round(K * mult))))
            idx = np.argpartition(-a, K_t - 1)[:K_t]
            m[h, idx] = True
        return m
    return policy


def make_sigmoid(alpha, beta, c, gamma):
    def policy(attn_ph_layer_step, K, **kw):
        nh, nk = attn_ph_layer_step.shape
        if nk <= K: return np.ones((nh, nk), dtype=bool)
        m = np.zeros((nh, nk), dtype=bool)
        for h in range(nh):
            a = attn_ph_layer_step[h]
            max_a = float(a.max())
            # sigmoid: high when max_a small (diffuse), low when max_a large (sharp)
            sig = 1.0 / (1.0 + np.exp(gamma * (max_a - c)))
            mult = beta + (alpha - beta) * sig
            K_t = max(1, min(nk, int(round(K * mult))))
            idx = np.argpartition(-a, K_t - 1)[:K_t]
            m[h, idx] = True
        return m
    return policy


def make_quadratic(alpha, beta, tl, th):
    def policy(attn_ph_layer_step, K, **kw):
        nh, nk = attn_ph_layer_step.shape
        if nk <= K: return np.ones((nh, nk), dtype=bool)
        m = np.zeros((nh, nk), dtype=bool)
        denom = max(th - tl, 1e-6)
        for h in range(nh):
            a = attn_ph_layer_step[h]
            max_a = float(a.max())
            norm = max(0.0, min(1.0, (max_a - tl) / denom))
            # quadratic: more bimodal — slow at edges, fast in middle
            mult = alpha - (alpha - beta) * (norm * norm)
            K_t = max(1, min(nk, int(round(K * mult))))
            idx = np.argpartition(-a, K_t - 1)[:K_t]
            m[h, idx] = True
        return m
    return policy


def make_inv_quadratic(alpha, beta, tl, th):
    """Inverse quadratic: 1 - (1-norm)^2 = 2*norm - norm^2.
    Fast near sharp end, slow near diffuse end."""
    def policy(attn_ph_layer_step, K, **kw):
        nh, nk = attn_ph_layer_step.shape
        if nk <= K: return np.ones((nh, nk), dtype=bool)
        m = np.zeros((nh, nk), dtype=bool)
        denom = max(th - tl, 1e-6)
        for h in range(nh):
            a = attn_ph_layer_step[h]
            max_a = float(a.max())
            norm = max(0.0, min(1.0, (max_a - tl) / denom))
            shape = 1.0 - (1.0 - norm) ** 2
            mult = alpha - (alpha - beta) * shape
            K_t = max(1, min(nk, int(round(K * mult))))
            idx = np.argpartition(-a, K_t - 1)[:K_t]
            m[h, idx] = True
        return m
    return policy


def make_step(alpha, beta, c):
    def policy(attn_ph_layer_step, K, **kw):
        nh, nk = attn_ph_layer_step.shape
        if nk <= K: return np.ones((nh, nk), dtype=bool)
        m = np.zeros((nh, nk), dtype=bool)
        for h in range(nh):
            a = attn_ph_layer_step[h]
            max_a = float(a.max())
            mult = beta if max_a >= c else alpha
            K_t = max(1, min(nk, int(round(K * mult))))
            idx = np.argpartition(-a, K_t - 1)[:K_t]
            m[h, idx] = True
        return m
    return policy


# ---------------------------------------------------------------------------
# Configs to test
# ---------------------------------------------------------------------------

CONFIGS = []

# (A) Linear-clipped sweep — focus on cache-neutral region (α≈1.2, β≈0.8)
# Vary thresh_low / thresh_high to find the sensitive sweet spot
for alpha, beta in [(1.2, 0.8), (1.25, 0.75), (1.3, 0.7), (1.3, 0.8), (1.4, 0.6)]:
    for tl, th in [(0.3, 0.6), (0.3, 0.7), (0.3, 0.8), (0.4, 0.7), (0.4, 0.8),
                   (0.5, 0.8), (0.2, 0.6)]:
        CONFIGS.append({
            "tag": f"linclip_a{alpha}_b{beta}_tl{tl}_th{th}",
            "gate": "linear-clipped",
            "alpha": alpha, "beta": beta, "thresh_low": tl, "thresh_high": th,
            "policy_fn": make_linear_clipped(alpha, beta, tl, th),
        })

# (B) Sigmoid sweep — same (alpha, beta) range, different (center, steepness)
for alpha, beta in [(1.3, 0.7), (1.4, 0.6), (1.5, 0.5)]:
    for c, gamma in [(0.4, 8.0), (0.4, 12.0), (0.5, 8.0), (0.5, 12.0), (0.6, 8.0)]:
        CONFIGS.append({
            "tag": f"sig_a{alpha}_b{beta}_c{c}_g{gamma}",
            "gate": "sigmoid",
            "alpha": alpha, "beta": beta, "center": c, "gamma": gamma,
            "policy_fn": make_sigmoid(alpha, beta, c, gamma),
        })

# (C) Quadratic — more bimodal
for alpha, beta in [(1.3, 0.7), (1.4, 0.6), (1.5, 0.5)]:
    for tl, th in [(0.3, 0.7), (0.4, 0.8)]:
        CONFIGS.append({
            "tag": f"quad_a{alpha}_b{beta}_tl{tl}_th{th}",
            "gate": "quadratic",
            "alpha": alpha, "beta": beta, "thresh_low": tl, "thresh_high": th,
            "policy_fn": make_quadratic(alpha, beta, tl, th),
        })

# (D) Inverse quadratic
for alpha, beta in [(1.3, 0.7), (1.4, 0.6)]:
    for tl, th in [(0.3, 0.7), (0.4, 0.8)]:
        CONFIGS.append({
            "tag": f"invquad_a{alpha}_b{beta}_tl{tl}_th{th}",
            "gate": "inv-quadratic",
            "alpha": alpha, "beta": beta, "thresh_low": tl, "thresh_high": th,
            "policy_fn": make_inv_quadratic(alpha, beta, tl, th),
        })

# (E) Step — hardest split
for alpha, beta in [(1.5, 0.5), (1.4, 0.6), (1.3, 0.7)]:
    for c in [0.4, 0.5, 0.6]:
        CONFIGS.append({
            "tag": f"step_a{alpha}_b{beta}_c{c}",
            "gate": "step",
            "alpha": alpha, "beta": beta, "center": c,
            "policy_fn": make_step(alpha, beta, c),
        })

print(f"[gate-search] {len(CONFIGS)} configs queued")


DIRS = [
    "/home/mislam22/EndurKV_workspace/logs/study_phone_phi3_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_mistral_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_qwen2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_gemma2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_r1distill_longbench",
]
BUDGETS = [512, 1024]


def main():
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures")
    rows = []
    t0 = time.time()

    # Pre-load attn data per model
    attn_per_model = {}
    for d_str in DIRS:
        d = Path(d_str)
        if not d.is_dir(): continue
        model = d.name.replace("study_phone_", "").replace("_longbench", "")
        attn_files = sorted([f for f in d.glob("*.attn.bin")
                             if not f.name.endswith(".v1.attn.bin")])[:1]
        for af in attn_files:
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None: continue
            attn_per_model[model] = (attn_ph, n_kv_at)
            print(f"[gate-search] preloaded {model} ({af.stat().st_size/1e6:.0f} MB)", flush=True)

    # Also need TOVA baseline per model for reference
    print(f"[gate-search] running TOVA reference for all models, both K")
    tova_ref = {}  # (model, K) -> (mean_kl, mean_K)
    for model, (attn_ph, n_kv_at) in attn_per_model.items():
        for K in BUDGETS:
            kls, Ks, _, _ = simulate(attn_ph, n_kv_at, None, None, "perhead_tova", K)
            tova_ref[(model, K)] = (float(np.mean(kls)), float(np.mean(Ks)))
            print(f"  TOVA {model} K={K}: mean_kl={tova_ref[(model,K)][0]:.4f}", flush=True)

    # Sweep
    for cfg_i, cfg in enumerate(CONFIGS):
        POLICIES["__gs__"] = cfg["policy_fn"]
        for model, (attn_ph, n_kv_at) in attn_per_model.items():
            for K in BUDGETS:
                tova_kl, tova_K = tova_ref[(model, K)]
                kls, Ks, mass, _ = simulate(attn_ph, n_kv_at, None, None, "__gs__", K)
                rows.append({
                    "config_tag": cfg["tag"],
                    "gate": cfg["gate"],
                    "alpha": cfg.get("alpha"), "beta": cfg.get("beta"),
                    "thresh_low": cfg.get("thresh_low"),
                    "thresh_high": cfg.get("thresh_high"),
                    "center": cfg.get("center"), "gamma": cfg.get("gamma"),
                    "model": model, "K_nominal": K,
                    "actual_K": float(np.mean(Ks)),
                    "cache_ratio_vs_tova": float(np.mean(Ks)) / tova_K,
                    "kl_mean": float(np.mean(kls)),
                    "mass_pct": 100*float(np.mean(mass)),
                    "delta_tova_pct": 100*(float(np.mean(kls)) / tova_kl - 1),
                })
        print(f"  [{time.time()-t0:5.0f}s] {cfg_i+1}/{len(CONFIGS)} {cfg['tag']} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gate_search_results.csv", index=False)
    print(f"\n[gate-search] wrote {len(df)} rows")

    agg = (df.groupby(["config_tag","gate","alpha","beta"])
             .agg(mean_delta_tova=("delta_tova_pct","mean"),
                  mean_cache=("cache_ratio_vs_tova","mean"),
                  mean_mass=("mass_pct","mean"),
                  n_cells=("delta_tova_pct","count"))
             .reset_index())
    agg["cache_pp_over"] = (agg["mean_cache"] - 1.0)*100
    agg["cache_adj"] = agg["mean_delta_tova"] + agg["cache_pp_over"].clip(lower=0)
    agg = agg.sort_values("cache_adj")
    agg.to_csv(out_dir / "gate_search_ranking.csv", index=False)

    print("\n=== TOP 20 by cache-adjusted Δ TOVA (lower = better) ===")
    print(agg.head(20).to_string(index=False))
    print("\n=== Cache-neutral (cache ≤ 1.01) best Δ TOVA ===")
    neutral = agg[agg.mean_cache <= 1.01].sort_values("mean_delta_tova")
    print(neutral.head(10).to_string(index=False))
    print("\n=== Best 10+% gain at any cache ===")
    big_gain = agg[agg.mean_delta_tova <= -10.0].sort_values("cache_pp_over")
    print(big_gain.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
