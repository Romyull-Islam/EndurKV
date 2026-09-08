#!/usr/bin/env python3
"""Wider Llama-1B and Llama-8B sweep — 15 prompts per regime × 4 regimes = 60.

Confirms whether v1_linear's robustness holds across many prompts and benchmarks,
not just the 1 prompt each we used in the comprehensive eval.

Output:
  EndurKV/figures/llama_wide_results.csv
  EndurKV/figures/llama_wide_ranking.csv
"""
import sys, time
from multiprocessing import Pool, set_start_method
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import load_attn_perhead


# Prompt selection (from earlier categorize_prompts)
PROMPTS = {
    "llama1b_short": [
        ('cnn_dailymail_001'), ('gov_report_001'), ('hotpotqa_001'),
        ('lcc_001'), ('multifieldqa_en_001'), ('openbookqa_001'),
        ('piqa_001'), ('qasper_001'), ('samsum_001'),
        ('trec_001'), ('triviaqa_001'), ('xsum_001'),
        ('cnn_dailymail_002'), ('gov_report_002'), ('hotpotqa_002'),
    ],
    "llama1b_long": [
        ('cnn_dailymail_lc_01'), ('gov_report_lc_01'), ('hotpotqa_lc_01'),
        ('lcc_lc_01'), ('multi_news_lc_01'), ('multifieldqa_en_lc_01'),
        ('narrativeqa_lc_01'), ('qasper_lc_01'), ('qmsum_lc_01'),
        ('samsum_lc_01'), ('trec_lc_01'), ('triviaqa_lc_01'),
        ('xsum_lc_01'), ('cnn_dailymail_lc_02'), ('gov_report_lc_02'),
    ],
    "llama8b_short": [
        ('hotpotqa_001'), ('multifieldqa_en_001'), ('qasper_001'),
        ('samsum_001'), ('triviaqa_001'), ('hotpotqa_002'),
        ('multifieldqa_en_002'), ('qasper_002'), ('samsum_002'),
        ('triviaqa_002'), ('multifieldqa_en_003'), ('qasper_003'),
        ('samsum_003'), ('triviaqa_003'), ('multifieldqa_en_004'),
    ],
    "llama8b_long": [
        ('cnn_dailymail_lc_01'), ('gov_report_lc_04'), ('hotpotqa_lc_01'),
        ('lcc_lc_01'), ('multi_news_lc_04'), ('multifieldqa_en_lc_04'),
        ('narrativeqa_lc_04'), ('qasper_lc_04'), ('qmsum_lc_04'),
        ('samsum_lc_01'), ('trec_lc_01'), ('triviaqa_lc_01'),
        ('xsum_lc_01'), ('cnn_dailymail_lc_02'), ('hotpotqa_lc_02'),
    ],
}
DIRS_BY_TAG = {
    "llama1b_short": Path("/home/mislam22/EndurKV_workspace/logs/study_phone_1b_short_perhead"),
    "llama1b_long":  Path("/home/mislam22/EndurKV_workspace/logs/study_phone_1b_longctx_perhead"),
    "llama8b_short": Path("/home/mislam22/EndurKV_workspace/logs/study_phone_8b_short_perhead"),
    "llama8b_long":  Path("/home/mislam22/EndurKV_workspace/logs/study_phone_8b_longctx_perhead_remaining"),
}
K_BY_REGIME = {
    "llama1b_short": [64, 128],
    "llama1b_long":  [256, 512],
    "llama8b_short": [64, 128],
    "llama8b_long":  [256, 512],
}


# ---------------------------------------------------------------------------
# Policies (copied from comprehensive_eval to keep this script standalone)
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

def policy_v6_logistic(attn_ph, K, **kw):
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        a = attn_ph[h]; max_a = float(a.max())
        sig = 1.0 / (1.0 + np.exp(10.0 * (max_a - 0.4)))
        mult = 0.7 + (1.3 - 0.7) * sig
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-a, K_t - 1)[:K_t]
        m[h, idx] = True
    return m

def policy_v1_sigmoid(attn_ph, K, **kw):
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
    if n_layers <= 1:
        K_eff = K
    else:
        frac = layer_idx / (n_layers - 1)
        K_max = K * (1 + ratio); K_min = K * (1 - ratio)
        K_eff = max(1, int(round(K_max - (K_max - K_min) * frac)))
    return policy_tova(attn_ph, K_eff)

def policy_adakv(attn_ph, K, **kw):
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
    "tova":          (policy_tova, False),
    "v1_linear":     (policy_v1_linear, False),
    "v6_logistic":   (policy_v6_logistic, False),
    "v1_sigmoid":    (policy_v1_sigmoid, False),
    "pyramidkv":     (policy_pyramidkv, True),
    "adakv":         (policy_adakv, False),
}


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
# Main
# ---------------------------------------------------------------------------
DATA = {}

def init_data():
    print("[llama-wide] loading captures ...", flush=True)
    total_bytes = 0
    for tag, prompt_ids in PROMPTS.items():
        d = DIRS_BY_TAG[tag]
        for pid in prompt_ids:
            af = d / f"{pid}.attn.bin"
            if not af.exists():
                print(f"  [skip] {tag}/{pid}: file missing")
                continue
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None:
                print(f"  [skip] {tag}/{pid}: load failed"); continue
            DATA[(tag, pid)] = (attn_ph, n_kv_at)
            total_bytes += attn_ph.nbytes
    print(f"[llama-wide] loaded {len(DATA)} prompts, total RAM {total_bytes/1e9:.1f} GB", flush=True)


def worker(task):
    tag, pid, K_nominal, policy_name = task
    attn_ph, n_kv_at = DATA[(tag, pid)]
    kls, Ks, mass = simulate_policy(attn_ph, n_kv_at, K_nominal, policy_name)
    return {
        "regime": tag, "prompt_id": pid,
        "K_nominal": K_nominal, "policy": policy_name,
        "n_kv": int(n_kv_at[-1]) if len(n_kv_at) else 0,
        "n_steps": len(n_kv_at),
        "actual_K": float(np.mean(Ks)),
        "kl_mean": float(np.mean(kls)),
        "kl_min": float(np.min(kls)),
        "kl_max": float(np.max(kls)),
        "mass_pct": 100*float(np.mean(mass)),
    }


def main():
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures")
    init_data()
    tasks = []
    for tag, prompt_ids in PROMPTS.items():
        for pid in prompt_ids:
            if (tag, pid) not in DATA: continue
            for K in K_BY_REGIME[tag]:
                for pol in POLICIES:
                    tasks.append((tag, pid, K, pol))
    print(f"[llama-wide] {len(tasks)} tasks queued", flush=True)

    t0 = time.time()
    rows = []
    with Pool(processes=12) as pool:
        for i, row in enumerate(pool.imap_unordered(worker, tasks, chunksize=2)):
            rows.append(row)
            if (i+1) % 50 == 0 or i+1 == len(tasks):
                print(f"  [{time.time()-t0:5.0f}s] {i+1}/{len(tasks)} done", flush=True)
    df = pd.DataFrame(rows)

    # Compute pct_vs_v1 / pct_vs_tova / cache_ratio per (regime, prompt, K)
    pct_v1, pct_tova, cr = [], [], []
    for _, r in df.iterrows():
        v1 = df[(df.regime==r.regime) & (df.prompt_id==r.prompt_id) &
                (df.K_nominal==r.K_nominal) & (df.policy=="v1_linear")]
        tv = df[(df.regime==r.regime) & (df.prompt_id==r.prompt_id) &
                (df.K_nominal==r.K_nominal) & (df.policy=="tova")]
        pct_v1.append(100*(r.kl_mean/float(v1.kl_mean.iloc[0]) - 1) if len(v1) else np.nan)
        pct_tova.append(100*(r.kl_mean/float(tv.kl_mean.iloc[0]) - 1) if len(tv) else np.nan)
        cr.append(r.actual_K/float(tv.actual_K.iloc[0]) if len(tv) else np.nan)
    df["pct_vs_v1"] = pct_v1
    df["pct_vs_tova"] = pct_tova
    df["cache_ratio_vs_tova"] = cr

    df.to_csv(out_dir / "llama_wide_results.csv", index=False)
    print(f"\n[llama-wide] wrote {len(df)} rows in {time.time()-t0:.0f}s")

    print("\n=== PER-REGIME RANKING ===")
    for regime in ["llama1b_short","llama1b_long","llama8b_short","llama8b_long"]:
        sub = df[df.regime==regime]
        agg = (sub.groupby("policy")
                  .agg(mean_kl=("kl_mean","mean"),
                       mean_cache=("cache_ratio_vs_tova","mean"),
                       mean_vs_tova=("pct_vs_tova","mean"),
                       mean_mass=("mass_pct","mean"),
                       n=("kl_mean","count"))
                  .reset_index().sort_values("mean_vs_tova"))
        print(f"\n--- {regime} (n_cells = {len(sub)}) ---")
        print(agg.to_string(index=False))

    print("\n=== OVERALL ranking (60 prompts × 2 K × 6 policies = 720 cells) ===")
    overall = (df.groupby("policy")
                 .agg(mean_kl=("kl_mean","mean"),
                      mean_cache=("cache_ratio_vs_tova","mean"),
                      mean_vs_tova=("pct_vs_tova","mean"),
                      mean_mass=("mass_pct","mean"),
                      n=("kl_mean","count"))
                 .reset_index().sort_values("mean_vs_tova"))
    print(overall.to_string(index=False))
    overall.to_csv(out_dir / "llama_wide_ranking.csv", index=False)
    return 0


if __name__ == "__main__":
    set_start_method("fork")
    sys.exit(main())
