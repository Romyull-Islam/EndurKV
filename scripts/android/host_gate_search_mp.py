#!/usr/bin/env python3
"""Multiprocess gate-shape + hyperparameter search.

16 parallel workers via multiprocessing.Pool. Workers inherit pre-loaded
attention data from the parent via Linux fork() copy-on-write — no actual
memory duplication when we only read.

Expected wall time on i9-14900K (32 threads): ~30 min for 69 configs.
"""
import sys, time
from itertools import product
from multiprocessing import Pool, set_start_method
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import load_attn_perhead, simulate, POLICIES


# ---------------------------------------------------------------------------
# Gate-shape policy factories (same as host_gate_search.py)
# ---------------------------------------------------------------------------

def lin_clip(a, max_a, alpha, beta, tl, th):
    denom = max(th - tl, 1e-6)
    norm = max(0.0, min(1.0, (max_a - tl) / denom))
    return alpha - (alpha - beta) * norm


def sigmoid(a, max_a, alpha, beta, c, gamma):
    sig = 1.0 / (1.0 + np.exp(gamma * (max_a - c)))
    return beta + (alpha - beta) * sig


def quadratic(a, max_a, alpha, beta, tl, th):
    denom = max(th - tl, 1e-6)
    norm = max(0.0, min(1.0, (max_a - tl) / denom))
    return alpha - (alpha - beta) * (norm * norm)


def inv_quadratic(a, max_a, alpha, beta, tl, th):
    denom = max(th - tl, 1e-6)
    norm = max(0.0, min(1.0, (max_a - tl) / denom))
    shape = 1.0 - (1.0 - norm) ** 2
    return alpha - (alpha - beta) * shape


def step(a, max_a, alpha, beta, c):
    return beta if max_a >= c else alpha


GATE_FN = {
    "linear-clipped": lin_clip,
    "sigmoid":        sigmoid,
    "quadratic":      quadratic,
    "inv-quadratic":  inv_quadratic,
    "step":           step,
}


def make_policy(gate, params):
    """Wrap a gate formula into a per-head top-K policy."""
    gate_fn = GATE_FN[gate]
    def policy(attn_ph_layer_step, K, **kw):
        nh, nk = attn_ph_layer_step.shape
        if nk <= K: return np.ones((nh, nk), dtype=bool)
        m = np.zeros((nh, nk), dtype=bool)
        for h in range(nh):
            a = attn_ph_layer_step[h]
            max_a = float(a.max())
            mult = gate_fn(a, max_a, **params)
            K_t = max(1, min(nk, int(round(K * mult))))
            idx = np.argpartition(-a, K_t - 1)[:K_t]
            m[h, idx] = True
        return m
    return policy


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

CONFIGS = []
# (A) Linear-clipped — 35 configs
for alpha, beta in [(1.2, 0.8), (1.25, 0.75), (1.3, 0.7), (1.3, 0.8), (1.4, 0.6)]:
    for tl, th in [(0.3, 0.6), (0.3, 0.7), (0.3, 0.8), (0.4, 0.7), (0.4, 0.8),
                   (0.5, 0.8), (0.2, 0.6)]:
        CONFIGS.append({
            "tag": f"linclip_a{alpha}_b{beta}_tl{tl}_th{th}",
            "gate": "linear-clipped",
            "params": dict(alpha=alpha, beta=beta, tl=tl, th=th),
        })
# (B) Sigmoid — 15
for alpha, beta in [(1.3, 0.7), (1.4, 0.6), (1.5, 0.5)]:
    for c, gamma in [(0.4, 8.0), (0.4, 12.0), (0.5, 8.0), (0.5, 12.0), (0.6, 8.0)]:
        CONFIGS.append({
            "tag": f"sig_a{alpha}_b{beta}_c{c}_g{gamma}",
            "gate": "sigmoid",
            "params": dict(alpha=alpha, beta=beta, c=c, gamma=gamma),
        })
# (C) Quadratic — 6
for alpha, beta in [(1.3, 0.7), (1.4, 0.6), (1.5, 0.5)]:
    for tl, th in [(0.3, 0.7), (0.4, 0.8)]:
        CONFIGS.append({
            "tag": f"quad_a{alpha}_b{beta}_tl{tl}_th{th}",
            "gate": "quadratic",
            "params": dict(alpha=alpha, beta=beta, tl=tl, th=th),
        })
# (D) Inverse quadratic — 4
for alpha, beta in [(1.3, 0.7), (1.4, 0.6)]:
    for tl, th in [(0.3, 0.7), (0.4, 0.8)]:
        CONFIGS.append({
            "tag": f"invquad_a{alpha}_b{beta}_tl{tl}_th{th}",
            "gate": "inv-quadratic",
            "params": dict(alpha=alpha, beta=beta, tl=tl, th=th),
        })
# (E) Step — 9
for alpha, beta in [(1.5, 0.5), (1.4, 0.6), (1.3, 0.7)]:
    for c in [0.4, 0.5, 0.6]:
        CONFIGS.append({
            "tag": f"step_a{alpha}_b{beta}_c{c}",
            "gate": "step",
            "params": dict(alpha=alpha, beta=beta, c=c),
        })

print(f"[gate-search-mp] {len(CONFIGS)} configs total")


DIRS = [
    "/home/mislam22/EndurKV_workspace/logs/study_phone_phi3_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_mistral_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_qwen2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_gemma2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_r1distill_longbench",
]
BUDGETS = [512, 1024]


# Global state populated in parent before fork — workers see this via COW
ATTN_PER_MODEL = {}
TOVA_REF = {}


def init_data():
    """Called in parent. Loads attn data and computes TOVA baselines."""
    global ATTN_PER_MODEL, TOVA_REF
    for d_str in DIRS:
        d = Path(d_str)
        if not d.is_dir(): continue
        model = d.name.replace("study_phone_", "").replace("_longbench", "")
        attn_files = sorted([f for f in d.glob("*.attn.bin")
                             if not f.name.endswith(".v1.attn.bin")])[:1]
        for af in attn_files:
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None: continue
            ATTN_PER_MODEL[model] = (attn_ph, n_kv_at)
            print(f"[gate-search-mp] preloaded {model} ({af.stat().st_size/1e6:.0f} MB)", flush=True)

    print(f"[gate-search-mp] computing TOVA references ...", flush=True)
    for model, (attn_ph, n_kv_at) in ATTN_PER_MODEL.items():
        for K in BUDGETS:
            kls, Ks, _, _ = simulate(attn_ph, n_kv_at, None, None, "perhead_tova", K)
            TOVA_REF[(model, K)] = (float(np.mean(kls)), float(np.mean(Ks)))
    print(f"[gate-search-mp] TOVA ref done", flush=True)


def worker(args):
    """Process one (cfg_idx, model, K) task. Returns dict for the row."""
    cfg_idx, model, K = args
    cfg = CONFIGS[cfg_idx]
    attn_ph, n_kv_at = ATTN_PER_MODEL[model]
    tova_kl, tova_K = TOVA_REF[(model, K)]
    fn = make_policy(cfg["gate"], cfg["params"])
    POLICIES["__gs__"] = fn
    kls, Ks, mass, _ = simulate(attn_ph, n_kv_at, None, None, "__gs__", K)
    return {
        "config_tag": cfg["tag"],
        "gate": cfg["gate"],
        **cfg["params"],
        "model": model, "K_nominal": K,
        "actual_K": float(np.mean(Ks)),
        "cache_ratio_vs_tova": float(np.mean(Ks)) / tova_K,
        "kl_mean": float(np.mean(kls)),
        "mass_pct": 100*float(np.mean(mass)),
        "delta_tova_pct": 100*(float(np.mean(kls)) / tova_kl - 1),
    }


def main():
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures")
    init_data()
    tasks = [(i, m, K) for i, _ in enumerate(CONFIGS)
                       for m in ATTN_PER_MODEL.keys()
                       for K in BUDGETS]
    print(f"[gate-search-mp] {len(tasks)} tasks, {len(CONFIGS)} configs × "
          f"{len(ATTN_PER_MODEL)} models × {len(BUDGETS)} K")

    t0 = time.time()
    rows = []
    with Pool(processes=16) as pool:
        for i, row in enumerate(pool.imap_unordered(worker, tasks, chunksize=4)):
            rows.append(row)
            if (i+1) % 25 == 0 or i+1 == len(tasks):
                print(f"  [{time.time()-t0:5.0f}s] {i+1}/{len(tasks)} tasks done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gate_search_mp_results.csv", index=False)
    print(f"\n[gate-search-mp] wrote {len(df)} rows in {time.time()-t0:.0f}s")

    # Aggregate
    agg = (df.groupby(["config_tag","gate"])
             .agg(mean_delta_tova=("delta_tova_pct","mean"),
                  mean_cache=("cache_ratio_vs_tova","mean"),
                  mean_mass=("mass_pct","mean"),
                  n_cells=("delta_tova_pct","count"))
             .reset_index())
    agg["cache_pp_over"] = (agg["mean_cache"] - 1.0)*100
    agg["cache_adj"] = agg["mean_delta_tova"] + agg["cache_pp_over"].clip(lower=0)
    agg = agg.sort_values("cache_adj")
    agg.to_csv(out_dir / "gate_search_mp_ranking.csv", index=False)

    print("\n=== TOP 15 by cache-adjusted Δ TOVA (lower = better) ===")
    print(agg.head(15)[['config_tag','gate','mean_delta_tova','mean_cache','cache_adj']].to_string(index=False))
    print("\n=== Cache-neutral (cache ≤ 1.01) sorted by raw Δ TOVA ===")
    neutral = agg[agg.mean_cache <= 1.01].sort_values("mean_delta_tova")
    print(neutral.head(10)[['config_tag','gate','mean_delta_tova','mean_cache','cache_adj']].to_string(index=False))
    print("\n=== Best at ≥10% gain — sorted by lowest cache overshoot ===")
    big_gain = agg[agg.mean_delta_tova <= -10.0].sort_values("cache_pp_over")
    print(big_gain.head(10)[['config_tag','gate','mean_delta_tova','mean_cache','cache_adj']].to_string(index=False))
    return 0


if __name__ == "__main__":
    set_start_method("fork")
    sys.exit(main())
