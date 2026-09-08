#!/usr/bin/env python3
"""Comprehensive evaluation: every dataset × every model × top 6 policies.

Captures covered:
  - LongBench (long context ~8K-10K): 5 paper models
  - NIAH (4K with needle): 5 paper models
  - Reasoning (GSM8K/math): R1-distill
  - Short context (~500-1K): Mistral, Llama-1B, Llama-8B
  - Long context for Llama family: 1B, 8B

Policies (6):
  1. v1 (original linear, α=1.3 β=0.6)
  2. v6 (logistic original, α=1.3 β=0.7)
  3. v1-sigmoid (NEW, α=1.5 β=0.5 c=0.4 γ=8.0)
  4. TOVA (per-head, fixed K)
  5. PyramidKV (per-layer budget × TOVA selection)
  6. AdaKV (entropy-proportional per-head budget × TOVA selection)

Output: comprehensive_all_contexts_results.csv + stratified ranking.
"""
import sys, time
from multiprocessing import Pool, set_start_method
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import load_attn_perhead


# ---------------------------------------------------------------------------
# Capture manifest: (dataset_family, dir, sample_prompt_ids, K_budgets)
# ---------------------------------------------------------------------------
LOGS = Path("/home/mislam22/EndurKV_workspace/logs")
LONG_K = [512, 1024]
MED_K = [256, 512]
SHORT_K = [64, 128]

CELLS = [
    # 5 paper models × LongBench (long context ~8K-10K)
    ("phi3", "longbench", LOGS/"study_phone_phi3_longbench", ["narrativeqa_pub_001"], LONG_K),
    ("mistral", "longbench", LOGS/"study_phone_mistral_longbench", ["narrativeqa_pub_001"], LONG_K),
    ("qwen2", "longbench", LOGS/"study_phone_qwen2_longbench", ["narrativeqa_pub_001"], LONG_K),
    ("gemma2", "longbench", LOGS/"study_phone_gemma2_longbench", ["narrativeqa_pub_001"], LONG_K),
    ("r1distill", "longbench", LOGS/"study_phone_r1distill_longbench", ["narrativeqa_pub_001"], LONG_K),
    # 5 paper models × NIAH (medium-long ~4K)
    ("phi3", "niah", LOGS/"study_phone_phi3_niah", ["niah_L4K_d00_n0"], MED_K),
    ("mistral", "niah", LOGS/"study_phone_mistral_niah", ["niah_L4K_d00_n0"], MED_K),
    ("qwen2", "niah", LOGS/"study_phone_qwen2_niah", ["niah_L4K_d00_n0"], MED_K),
    ("gemma2", "niah", LOGS/"study_phone_gemma2_niah", ["niah_L4K_d00_n0"], MED_K),
    ("r1distill", "niah", LOGS/"study_phone_r1distill_niah", ["niah_L4K_d00_n0"], MED_K),
    # R1-distill × reasoning
    ("r1distill", "reasoning", LOGS/"study_phone_r1distill_reasoning", ["gsm8k_pub_001"], MED_K),
    # Mistral × short context
    ("mistral", "short", LOGS/"study_phone_mistral7b_short", ["qasper_001"], SHORT_K),
    # Llama-1B × short, long
    ("llama1b", "short", LOGS/"study_phone_1b_short_perhead", ["cnn_dailymail_001"], SHORT_K),
    ("llama1b", "long", LOGS/"study_phone_1b_longctx_perhead", ["cnn_dailymail_lc_01"], MED_K),
    # Llama-8B × short, long
    ("llama8b", "short", LOGS/"study_phone_8b_short_perhead", ["hotpotqa_001"], SHORT_K),
    ("llama8b", "long", LOGS/"study_phone_8b_longctx_perhead_remaining", ["cnn_dailymail_lc_01"], MED_K),
]


# ---------------------------------------------------------------------------
# Policy implementations (self-contained, no POLICIES dict needed)
# ---------------------------------------------------------------------------

def policy_tova(attn_ph, K, **kw):
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        idx = np.argpartition(-attn_ph[h], K - 1)[:K]
        m[h, idx] = True
    return m


def policy_v1_linear(attn_ph, K, **kw):
    """Original v1: α=1.3, β=0.6, linear-clipped, thresh [0.4, 0.8]"""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        a = attn_ph[h]; max_a = float(a.max())
        norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
        mult = 1.3 - 0.6 * norm
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-a, K_t - 1)[:K_t]
        m[h, idx] = True
    return m


def policy_v6_logistic(attn_ph, K, gamma=10.0, c=0.4, alpha_low=0.7, alpha_high=1.3, **kw):
    """Original v6 logistic sigmoid gate."""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        a = attn_ph[h]; max_a = float(a.max())
        sig = 1.0 / (1.0 + np.exp(gamma * (max_a - c)))
        mult = alpha_low + (alpha_high - alpha_low) * sig
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-a, K_t - 1)[:K_t]
        m[h, idx] = True
    return m


def policy_v1_sigmoid(attn_ph, K, **kw):
    """NEW best: sigmoid α=1.5, β=0.5, c=0.4, γ=8.0"""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        a = attn_ph[h]; max_a = float(a.max())
        sig = 1.0 / (1.0 + np.exp(8.0 * (max_a - 0.4)))
        mult = 0.5 + 1.0 * sig
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-a, K_t - 1)[:K_t]
        m[h, idx] = True
    return m


def policy_pyramidkv(attn_ph, K, n_layers=32, layer_idx=0, ratio=0.5, **kw):
    """PyramidKV: linear K_max → K_min across depth, TOVA selection within layer."""
    if n_layers <= 1:
        K_eff = K
    else:
        frac = layer_idx / (n_layers - 1)
        K_max = K * (1 + ratio); K_min = K * (1 - ratio)
        K_eff = max(1, int(round(K_max - (K_max - K_min) * frac)))
    return policy_tova(attn_ph, K_eff)


def policy_adakv(attn_ph, K, **kw):
    """AdaKV: per-head budget proportional to sharpness, total = K*n_head."""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    sharp = np.zeros(nh); log_nk = max(1e-9, np.log(max(2, nk)))
    for h in range(nh):
        p = np.clip(attn_ph[h].astype(np.float32), 1e-12, 1.0)
        H = -float(np.sum(p * np.log(p)))
        sharp[h] = 1.0 - H / log_nk
    total = K * nh
    if sharp.sum() > 1e-9:
        budgets = sharp / sharp.sum() * total
    else:
        budgets = np.full(nh, K, dtype=np.float64)
    budgets = np.clip(budgets, 0.5 * K, 1.5 * K)
    budgets = budgets * (total / max(1.0, budgets.sum()))
    budgets = np.maximum(1, budgets.astype(np.int32))
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        K_h = min(int(budgets[h]), nk)
        idx = (np.argpartition(-attn_ph[h], K_h - 1)[:K_h]
               if K_h < nk else np.arange(nk))
        m[h, idx] = True
    return m


POLICIES = {
    "tova":          (policy_tova,         False),
    "v1_linear":     (policy_v1_linear,    False),
    "v6_logistic":   (policy_v6_logistic,  False),
    "v1_sigmoid":    (policy_v1_sigmoid,   False),
    "pyramidkv":     (policy_pyramidkv,    True),   # needs layer_idx + n_layers
    "adakv":         (policy_adakv,        False),
}


# ---------------------------------------------------------------------------
# Simulator with per-cell stats
# ---------------------------------------------------------------------------
def kl(p, q, eps=1e-12):
    pf = np.clip(p.astype(np.float32, copy=False), eps, 1.0)
    pe = np.clip(q.astype(np.float32, copy=False), eps, 1.0)
    return float(np.sum(pf * (np.log(pf) - np.log(pe))))


def simulate_policy(attn_ph, n_kv_at, K_nominal, policy_name):
    n_steps, n_layers, n_head, max_kv = attn_ph.shape
    fn, needs_layer_idx = POLICIES[policy_name]
    out_kl = np.zeros((n_steps, n_layers), dtype=np.float64)
    out_K  = np.zeros((n_steps, n_layers), dtype=np.float64)
    out_mass = np.zeros((n_steps, n_layers), dtype=np.float64)

    for s in range(n_steps):
        n_kv = int(n_kv_at[s])
        if n_kv == 0: continue
        for l in range(n_layers):
            ph = attn_ph[s, l, :, :n_kv]
            kw = {}
            if needs_layer_idx:
                kw["layer_idx"] = l
                kw["n_layers"] = n_layers
            mask = fn(ph, K_nominal, **kw)
            kls, masses = [], []
            kept_sum = 0
            for h in range(n_head):
                p_full = ph[h]
                kept = p_full * mask[h]
                tk = kept.sum(); tf = p_full.sum()
                masses.append(float(tk / tf) if tf > 0 else 1.0)
                kls.append(kl(p_full, kept / tk) if tk > 0 else 20.0)
                kept_sum += int(mask[h].sum())
            out_kl[s, l] = float(np.mean(kls))
            out_mass[s, l] = float(np.mean(masses))
            out_K[s, l] = kept_sum / n_head
    return out_kl.mean(axis=1), out_K.mean(axis=1), out_mass.mean(axis=1)


# ---------------------------------------------------------------------------
# Global data (preloaded; workers inherit via fork COW)
# ---------------------------------------------------------------------------
DATA = {}   # (model, dataset, prompt_id) -> (attn_ph, n_kv_at)


def init_data():
    print("[comp-eval] loading captures ...", flush=True)
    for model, dataset, d, prompts, K_budgets in CELLS:
        if not d.is_dir():
            print(f"  [skip] {model}/{dataset}: dir missing {d}")
            continue
        for pid in prompts:
            af = d / f"{pid}.attn.bin"
            if not af.exists():
                print(f"  [skip] {model}/{dataset}/{pid}: file missing")
                continue
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None:
                print(f"  [skip] {model}/{dataset}/{pid}: load failed")
                continue
            DATA[(model, dataset, pid)] = (attn_ph, n_kv_at)
            n_kv = int(n_kv_at[-1]) if len(n_kv_at) else 0
            n_steps = attn_ph.shape[0]
            n_layers = attn_ph.shape[1]
            n_head = attn_ph.shape[2]
            print(f"  loaded {model}/{dataset}/{pid}: n_steps={n_steps} layers={n_layers} "
                  f"heads={n_head} n_kv={n_kv}", flush=True)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def worker(task):
    model, dataset, pid, K_nominal, policy_name = task
    attn_ph, n_kv_at = DATA[(model, dataset, pid)]
    kls, Ks, mass = simulate_policy(attn_ph, n_kv_at, K_nominal, policy_name)
    return {
        "model": model, "dataset": dataset, "prompt_id": pid,
        "K_nominal": K_nominal, "policy": policy_name,
        "n_kv": int(n_kv_at[-1]) if len(n_kv_at) else 0,
        "n_steps": len(n_kv_at),
        "actual_K": float(np.mean(Ks)),
        "kl_mean": float(np.mean(kls)),
        "kl_min": float(np.min(kls)),
        "kl_max": float(np.max(kls)),
        "kl_std": float(np.std(kls)),
        "mass_pct": 100*float(np.mean(mass)),
    }


def main():
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures")
    init_data()
    tasks = []
    for model, dataset, d, prompts, K_budgets in CELLS:
        for pid in prompts:
            if (model, dataset, pid) not in DATA: continue
            for K in K_budgets:
                for pol in POLICIES:
                    tasks.append((model, dataset, pid, K, pol))
    print(f"\n[comp-eval] {len(tasks)} tasks queued "
          f"({len(DATA)} prompts × {len(POLICIES)} policies × ~2 K)")

    t0 = time.time()
    rows = []
    with Pool(processes=16) as pool:
        for i, row in enumerate(pool.imap_unordered(worker, tasks, chunksize=2)):
            rows.append(row)
            if (i+1) % 25 == 0 or i+1 == len(tasks):
                print(f"  [{time.time()-t0:5.0f}s] {i+1}/{len(tasks)} done", flush=True)
    df = pd.DataFrame(rows)

    # Compute pct_vs_v1 and pct_vs_tova per (model, dataset, prompt, K)
    pct_v1 = []; pct_tova = []; cache_ratio = []
    for _, r in df.iterrows():
        key = (r.model, r.dataset, r.prompt_id, r.K_nominal)
        v1 = df[(df.model==r.model) & (df.dataset==r.dataset) & (df.prompt_id==r.prompt_id) &
                (df.K_nominal==r.K_nominal) & (df.policy=="v1_linear")]
        tv = df[(df.model==r.model) & (df.dataset==r.dataset) & (df.prompt_id==r.prompt_id) &
                (df.K_nominal==r.K_nominal) & (df.policy=="tova")]
        pct_v1.append(100*(r.kl_mean/float(v1.kl_mean.iloc[0]) - 1) if len(v1) else np.nan)
        pct_tova.append(100*(r.kl_mean/float(tv.kl_mean.iloc[0]) - 1) if len(tv) else np.nan)
        cache_ratio.append(r.actual_K/float(tv.actual_K.iloc[0]) if len(tv) else np.nan)
    df["pct_vs_v1"] = pct_v1
    df["pct_vs_tova"] = pct_tova
    df["cache_ratio_vs_tova"] = cache_ratio

    df.to_csv(out_dir / "comprehensive_all_contexts.csv", index=False)
    print(f"\n[comp-eval] wrote {len(df)} rows in {time.time()-t0:.0f}s")

    # Stratified rankings by regime
    regime_map = {
        "longbench": "long", "niah": "NIAH",
        "reasoning": "reasoning", "short": "short", "long": "long_llama",
    }
    df["regime"] = df["dataset"].map(regime_map).fillna(df["dataset"])

    print("\n=== OVERALL ranking (all cells) ===")
    overall = (df.groupby("policy")
                 .agg(mean_kl=("kl_mean","mean"),
                      mean_cache=("cache_ratio_vs_tova","mean"),
                      mean_vs_tova=("pct_vs_tova","mean"),
                      mean_vs_v1=("pct_vs_v1","mean"),
                      mean_mass=("mass_pct","mean"),
                      n=("kl_mean","count"))
                 .reset_index().sort_values("mean_vs_tova"))
    print(overall.to_string(index=False))

    print("\n=== PER-REGIME ranking ===")
    for regime in df.regime.unique():
        sub = df[df.regime==regime]
        agg = (sub.groupby("policy")
                  .agg(mean_kl=("kl_mean","mean"),
                       mean_cache=("cache_ratio_vs_tova","mean"),
                       mean_vs_tova=("pct_vs_tova","mean"),
                       mean_mass=("mass_pct","mean"),
                       n=("kl_mean","count"))
                  .reset_index().sort_values("mean_vs_tova"))
        print(f"\n--- regime: {regime} (n_cells={len(sub)//6}) ---")
        print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    set_start_method("fork")
    sys.exit(main())
