#!/usr/bin/env python3
"""v1-adaptive: layer-aware blend between v1 (sharp regime) and TOVA (diffuse regime).

Mechanism:
  1. Compute per-LAYER attention sharpness:  L_sharp = mean over heads of max_a[h]
  2. Blend factor:  b = clip( (L_sharp − tau_low) / (tau_high − tau_low) , 0, 1 )
       b = 1  → full v1 behavior  (high layer sharpness)
       b = 0  → TOVA behavior     (low layer sharpness, e.g., reasoning)
  3. Effective α/β interpolated between TOVA (1.0, 1.0) and v1 (1.3, 0.6):
       α_eff = 1.0 + 0.3·b
       β_eff = 1.0 − 0.4·b
  4. Per-head multiplier (within layer):
       μ[h] = α_eff − (α_eff − β_eff) · clip((max_a[h] − 0.4)/0.4, 0, 1)
       K_h  = round(K_nominal · μ[h])
  5. Selection: top-K_h by current attention (same as TOVA / v1)

Defaults (chosen by inspection of layer-sharpness distributions in our captures):
  tau_low  = 0.30   below → reasoning regime  (TOVA)
  tau_high = 0.50   above → retrieval regime   (full v1)

Then runs on all 168 cells from unified_all_models_results.csv and merges.
"""
import sys, time
from multiprocessing import Pool, set_start_method
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import load_attn_perhead


# ---------------------------------------------------------------------------
# v1-adaptive policy
# ---------------------------------------------------------------------------
TAU_LOW = 0.30
TAU_HIGH = 0.50
ALPHA_V1 = 1.3
BETA_V1 = 0.6


def policy_v1_adaptive(attn_ph, K, **kw):
    nh, nk = attn_ph.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    # 1. Layer-level sharpness = mean of per-head max_a
    max_a_per_head = attn_ph.max(axis=1)               # [nh]
    L_sharp = float(max_a_per_head.mean())
    # 2. Blend factor (0 = TOVA, 1 = full v1)
    denom = max(TAU_HIGH - TAU_LOW, 1e-6)
    b = max(0.0, min(1.0, (L_sharp - TAU_LOW) / denom))
    # 3. Effective α/β
    alpha_eff = 1.0 + (ALPHA_V1 - 1.0) * b      # 1.0 → 1.3
    beta_eff  = 1.0 - (1.0 - BETA_V1) * b       # 1.0 → 0.6
    # 4. Per-head budget
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        max_a = float(max_a_per_head[h])
        norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
        mult = alpha_eff - (alpha_eff - beta_eff) * norm
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-attn_ph[h], K_t - 1)[:K_t]
        m[h, idx] = True
    return m


# ---------------------------------------------------------------------------
# Reuse the simulator from the comprehensive eval
# ---------------------------------------------------------------------------
def kl(p, q, eps=1e-12):
    pf = np.clip(p.astype(np.float32, copy=False), eps, 1.0)
    pe = np.clip(q.astype(np.float32, copy=False), eps, 1.0)
    return float(np.sum(pf * (np.log(pf) - np.log(pe))))


def simulate_v1_adaptive(attn_ph, n_kv_at, K_nominal):
    n_steps, n_layers, n_head, max_kv = attn_ph.shape
    out_kl = np.zeros((n_steps, n_layers), dtype=np.float64)
    out_K  = np.zeros((n_steps, n_layers), dtype=np.float64)
    out_mass = np.zeros((n_steps, n_layers), dtype=np.float64)
    for s in range(n_steps):
        n_kv = int(n_kv_at[s])
        if n_kv == 0: continue
        for l in range(n_layers):
            ph = attn_ph[s, l, :, :n_kv]
            mask = policy_v1_adaptive(ph, K_nominal)
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
# All capture cells from the unified dataset
# ---------------------------------------------------------------------------
LOGS = Path("/home/mislam22/EndurKV_workspace/logs")
LONG_K = [512, 1024]
MED_K = [256, 512]
SHORT_K = [64, 128]
CELLS = [
    # 5 paper models × LongBench (2 prompts each)
    ("phi3", "longbench", LOGS/"study_phone_phi3_longbench",
     ["narrativeqa_pub_001","narrativeqa_pub_002"], LONG_K),
    ("mistral", "longbench", LOGS/"study_phone_mistral_longbench",
     ["narrativeqa_pub_001","narrativeqa_pub_002"], LONG_K),
    ("qwen2", "longbench", LOGS/"study_phone_qwen2_longbench",
     ["narrativeqa_pub_001","narrativeqa_pub_002"], LONG_K),
    ("gemma2", "longbench", LOGS/"study_phone_gemma2_longbench",
     ["narrativeqa_pub_001","narrativeqa_pub_002"], LONG_K),
    ("r1distill", "longbench", LOGS/"study_phone_r1distill_longbench",
     ["narrativeqa_pub_001","narrativeqa_pub_002"], LONG_K),
    # 5 paper models × NIAH
    ("phi3", "niah", LOGS/"study_phone_phi3_niah",
     ["niah_L4K_d00_n0","niah_L4K_d17_n0"], MED_K),
    ("mistral", "niah", LOGS/"study_phone_mistral_niah",
     ["niah_L4K_d00_n0","niah_L4K_d17_n0"], MED_K),
    ("qwen2", "niah", LOGS/"study_phone_qwen2_niah",
     ["niah_L4K_d00_n0","niah_L4K_d17_n0"], MED_K),
    ("gemma2", "niah", LOGS/"study_phone_gemma2_niah",
     ["niah_L4K_d00_n0","niah_L4K_d17_n0"], MED_K),
    ("r1distill", "niah", LOGS/"study_phone_r1distill_niah",
     ["niah_L4K_d00_n0","niah_L4K_d17_n0"], MED_K),
    # R1 reasoning
    ("r1distill", "reasoning", LOGS/"study_phone_r1distill_reasoning",
     ["gsm8k_pub_001","gsm8k_pub_002"], MED_K),
    # Mistral short
    ("mistral", "short", LOGS/"study_phone_mistral7b_short",
     ["qasper_001","qasper_002"], SHORT_K),
]

# Llama wide-sweep prompts (60 cells)
LLAMA_PROMPTS = {
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
LLAMA_DIRS = {
    "llama1b_short": LOGS/"study_phone_1b_short_perhead",
    "llama1b_long":  LOGS/"study_phone_1b_longctx_perhead",
    "llama8b_short": LOGS/"study_phone_8b_short_perhead",
    "llama8b_long":  LOGS/"study_phone_8b_longctx_perhead_remaining",
}
LLAMA_K = {
    "llama1b_short": SHORT_K, "llama1b_long": MED_K,
    "llama8b_short": SHORT_K, "llama8b_long": MED_K,
}

# Build the full task list — same shape as unified data
def all_cells():
    out = []
    for model, dataset, d, prompts, K_budgets in CELLS:
        for pid in prompts:
            af = d / f"{pid}.attn.bin"
            if af.exists():
                out.append((model, dataset, pid, af, K_budgets))
    for tag, prompts in LLAMA_PROMPTS.items():
        model = tag.split("_")[0]   # llama1b, llama8b
        dataset = tag.split("_")[1] # short, long
        d = LLAMA_DIRS[tag]
        K_budgets = LLAMA_K[tag]
        for pid in prompts:
            af = d / f"{pid}.attn.bin"
            if af.exists():
                out.append((model, dataset, pid, af, K_budgets))
    return out


DATA = {}

def init_data():
    print("[v1-adaptive] loading captures ...", flush=True)
    for model, dataset, pid, af, Ks in all_cells():
        attn_ph, n_kv_at = load_attn_perhead(af)
        if attn_ph is None: continue
        DATA[(model, dataset, pid)] = (attn_ph, n_kv_at, Ks)
    print(f"[v1-adaptive] loaded {len(DATA)} prompts", flush=True)


def worker(task):
    model, dataset, pid, K_nominal = task
    attn_ph, n_kv_at, _ = DATA[(model, dataset, pid)]
    kls, Ks, mass = simulate_v1_adaptive(attn_ph, n_kv_at, K_nominal)
    return {
        "model": model, "dataset": dataset, "prompt_id": pid,
        "K_nominal": K_nominal, "policy": "v1_adaptive",
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
    for (model, dataset, pid), (_,_,Ks) in DATA.items():
        for K in Ks:
            tasks.append((model, dataset, pid, K))
    print(f"[v1-adaptive] {len(tasks)} tasks queued (1 policy × all cells × 2 K)")

    t0 = time.time()
    rows = []
    with Pool(processes=12) as pool:
        for i, row in enumerate(pool.imap_unordered(worker, tasks, chunksize=2)):
            rows.append(row)
            if (i+1) % 30 == 0 or i+1 == len(tasks):
                print(f"  [{time.time()-t0:5.0f}s] {i+1}/{len(tasks)} done", flush=True)
    df_new = pd.DataFrame(rows)

    # Compute pct_vs_v1, pct_vs_tova, cache_ratio_vs_tova by merging with existing data
    unified = pd.read_csv(out_dir / "unified_all_models_results.csv")
    refs = unified[unified.policy.isin(['v1_linear','tova'])]
    pct_v1, pct_tova, cr = [], [], []
    for _, r in df_new.iterrows():
        v1 = refs[(refs.model==r.model) & (refs.dataset==r.dataset) &
                  (refs.prompt_id==r.prompt_id) & (refs.K_nominal==r.K_nominal) &
                  (refs.policy=='v1_linear')]
        tv = refs[(refs.model==r.model) & (refs.dataset==r.dataset) &
                  (refs.prompt_id==r.prompt_id) & (refs.K_nominal==r.K_nominal) &
                  (refs.policy=='tova')]
        pct_v1.append(100*(r.kl_mean/float(v1.kl_mean.iloc[0]) - 1) if len(v1) else np.nan)
        pct_tova.append(100*(r.kl_mean/float(tv.kl_mean.iloc[0]) - 1) if len(tv) else np.nan)
        cr.append(r.actual_K/float(tv.actual_K.iloc[0]) if len(tv) else np.nan)
    df_new["pct_vs_v1"] = pct_v1
    df_new["pct_vs_tova"] = pct_tova
    df_new["cache_ratio_vs_tova"] = cr
    df_new.to_csv(out_dir / "v1_adaptive_results.csv", index=False)
    print(f"\n[v1-adaptive] wrote {len(df_new)} rows in {time.time()-t0:.0f}s")

    # Merge into unified
    common = ['model','dataset','prompt_id','K_nominal','policy','n_kv',
              'actual_K','kl_mean','kl_min','kl_max','mass_pct',
              'pct_vs_v1','pct_vs_tova','cache_ratio_vs_tova']
    # df_new doesn't have kl_std — fill 0
    df_new['kl_std'] = 0.0
    if 'kl_std' in unified.columns:
        common2 = common + (['kl_std'] if 'kl_std' in unified.columns else [])
    else:
        common2 = common
    big = pd.concat([unified[unified.columns.intersection(common2).tolist()], df_new[df_new.columns.intersection(common2).tolist()]], ignore_index=True)
    big.to_csv(out_dir / "unified_with_v1_adaptive.csv", index=False)
    print(f"[v1-adaptive] merged unified table → unified_with_v1_adaptive.csv ({len(big)} rows)")
    return 0


if __name__ == "__main__":
    set_start_method("fork")
    sys.exit(main())
