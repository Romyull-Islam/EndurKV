#!/usr/bin/env python3
"""v1-adaptive memory-safe: load 1 prompt at a time, run, unload, repeat.
Single-threaded but reliable on the ~30 GB memory budget."""
import sys, time, gc
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import load_attn_perhead

TAU_LOW, TAU_HIGH = 0.30, 0.50
ALPHA_V1, BETA_V1 = 1.3, 0.6


def policy_v1_adaptive(attn_ph, K):
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    max_a_per_head = attn_ph.max(axis=1)
    L_sharp = float(max_a_per_head.mean())
    denom = max(TAU_HIGH - TAU_LOW, 1e-6)
    b = max(0.0, min(1.0, (L_sharp - TAU_LOW) / denom))
    alpha_eff = 1.0 + (ALPHA_V1 - 1.0) * b
    beta_eff  = 1.0 - (1.0 - BETA_V1) * b
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        max_a = float(max_a_per_head[h])
        norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
        mult = alpha_eff - (alpha_eff - beta_eff) * norm
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-attn_ph[h], K_t - 1)[:K_t]
        m[h, idx] = True
    return m


def kl(p, q, eps=1e-12):
    pf = np.clip(p.astype(np.float32, copy=False), eps, 1.0)
    pe = np.clip(q.astype(np.float32, copy=False), eps, 1.0)
    return float(np.sum(pf * (np.log(pf) - np.log(pe))))


def simulate_one_K(attn_ph, n_kv_at, K):
    n_steps, n_layers, n_head, max_kv = attn_ph.shape
    kls_all, Ks_all, mass_all = [], [], []
    for s in range(n_steps):
        n_kv = int(n_kv_at[s])
        if n_kv == 0: continue
        for l in range(n_layers):
            ph = attn_ph[s, l, :, :n_kv]
            mask = policy_v1_adaptive(ph, K)
            kls, masses = [], []
            kept_sum = 0
            for h in range(n_head):
                p_full = ph[h]
                kept = p_full * mask[h]
                tk = kept.sum(); tf = p_full.sum()
                masses.append(float(tk/tf) if tf > 0 else 1.0)
                kls.append(kl(p_full, kept/tk) if tk > 0 else 20.0)
                kept_sum += int(mask[h].sum())
            kls_all.append(float(np.mean(kls)))
            mass_all.append(float(np.mean(masses)))
            Ks_all.append(kept_sum / n_head)
    return np.array(kls_all), np.array(Ks_all), np.array(mass_all)


LOGS = Path("/home/mislam22/EndurKV_workspace/logs")
LONG_K = [512, 1024]; MED_K = [256, 512]; SHORT_K = [64, 128]
CELLS = [
    ("phi3","longbench",LOGS/"study_phone_phi3_longbench",["narrativeqa_pub_001","narrativeqa_pub_002"],LONG_K),
    ("mistral","longbench",LOGS/"study_phone_mistral_longbench",["narrativeqa_pub_001","narrativeqa_pub_002"],LONG_K),
    ("qwen2","longbench",LOGS/"study_phone_qwen2_longbench",["narrativeqa_pub_001","narrativeqa_pub_002"],LONG_K),
    ("gemma2","longbench",LOGS/"study_phone_gemma2_longbench",["narrativeqa_pub_001","narrativeqa_pub_002"],LONG_K),
    ("r1distill","longbench",LOGS/"study_phone_r1distill_longbench",["narrativeqa_pub_001","narrativeqa_pub_002"],LONG_K),
    ("phi3","niah",LOGS/"study_phone_phi3_niah",["niah_L4K_d00_n0","niah_L4K_d17_n0"],MED_K),
    ("mistral","niah",LOGS/"study_phone_mistral_niah",["niah_L4K_d00_n0","niah_L4K_d17_n0"],MED_K),
    ("qwen2","niah",LOGS/"study_phone_qwen2_niah",["niah_L4K_d00_n0","niah_L4K_d17_n0"],MED_K),
    ("gemma2","niah",LOGS/"study_phone_gemma2_niah",["niah_L4K_d00_n0","niah_L4K_d17_n0"],MED_K),
    ("r1distill","niah",LOGS/"study_phone_r1distill_niah",["niah_L4K_d00_n0","niah_L4K_d17_n0"],MED_K),
    ("r1distill","reasoning",LOGS/"study_phone_r1distill_reasoning",["gsm8k_pub_001","gsm8k_pub_002"],MED_K),
    ("mistral","short",LOGS/"study_phone_mistral7b_short",["qasper_001","qasper_002"],SHORT_K),
]
LLAMA_PROMPTS = {
    "llama1b_short": ["cnn_dailymail_001","gov_report_001","hotpotqa_001","lcc_001",
                      "multifieldqa_en_001","openbookqa_001","piqa_001","qasper_001",
                      "samsum_001","trec_001","triviaqa_001","xsum_001",
                      "cnn_dailymail_002","gov_report_002","hotpotqa_002"],
    "llama1b_long": ["cnn_dailymail_lc_01","gov_report_lc_01","hotpotqa_lc_01",
                     "lcc_lc_01","multi_news_lc_01","multifieldqa_en_lc_01",
                     "narrativeqa_lc_01","qasper_lc_01","qmsum_lc_01","samsum_lc_01",
                     "trec_lc_01","triviaqa_lc_01","xsum_lc_01",
                     "cnn_dailymail_lc_02","gov_report_lc_02"],
    "llama8b_short": ["hotpotqa_001","multifieldqa_en_001","qasper_001","samsum_001",
                      "triviaqa_001","hotpotqa_002","multifieldqa_en_002","qasper_002",
                      "samsum_002","triviaqa_002","multifieldqa_en_003","qasper_003",
                      "samsum_003","triviaqa_003","multifieldqa_en_004"],
    "llama8b_long": ["cnn_dailymail_lc_01","gov_report_lc_04","hotpotqa_lc_01",
                     "lcc_lc_01","multi_news_lc_04","multifieldqa_en_lc_04",
                     "narrativeqa_lc_04","qasper_lc_04","qmsum_lc_04","samsum_lc_01",
                     "trec_lc_01","triviaqa_lc_01","xsum_lc_01",
                     "cnn_dailymail_lc_02","hotpotqa_lc_02"],
}
LLAMA_DIRS = {
    "llama1b_short": LOGS/"study_phone_1b_short_perhead",
    "llama1b_long":  LOGS/"study_phone_1b_longctx_perhead",
    "llama8b_short": LOGS/"study_phone_8b_short_perhead",
    "llama8b_long":  LOGS/"study_phone_8b_longctx_perhead_remaining",
}
LLAMA_K = {"llama1b_short":SHORT_K,"llama1b_long":MED_K,
           "llama8b_short":SHORT_K,"llama8b_long":MED_K}


def all_cells():
    for m, d, dir_, prompts, Ks in CELLS:
        for pid in prompts:
            af = dir_/f"{pid}.attn.bin"
            if af.exists(): yield m, d, pid, af, Ks
    for tag, prompts in LLAMA_PROMPTS.items():
        m, d = tag.split("_")
        dir_ = LLAMA_DIRS[tag]; Ks = LLAMA_K[tag]
        for pid in prompts:
            af = dir_/f"{pid}.attn.bin"
            if af.exists(): yield m, d, pid, af, Ks


def main():
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures")
    rows = []; t0 = time.time()
    cells = list(all_cells())
    print(f"[v1-adaptive-v2] {len(cells)} prompts, ~{len(cells)*2} sims", flush=True)
    for i, (model, dataset, pid, af, Ks) in enumerate(cells):
        try:
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None: continue
            for K in Ks:
                kls, Ks_avg, mass = simulate_one_K(attn_ph, n_kv_at, K)
                rows.append({
                    "model": model, "dataset": dataset, "prompt_id": pid,
                    "K_nominal": K, "policy": "v1_adaptive",
                    "n_kv": int(n_kv_at[-1]) if len(n_kv_at) else 0,
                    "n_steps": len(n_kv_at),
                    "actual_K": float(np.mean(Ks_avg)),
                    "kl_mean": float(np.mean(kls)),
                    "kl_min": float(np.min(kls)),
                    "kl_max": float(np.max(kls)),
                    "mass_pct": 100*float(np.mean(mass)),
                })
            del attn_ph, n_kv_at; gc.collect()
            if (i+1) % 3 == 0 or i+1 == len(cells):
                print(f"  [{time.time()-t0:5.0f}s] {i+1}/{len(cells)} prompts", flush=True)
                # Incremental save (in case we crash again)
                pd.DataFrame(rows).to_csv(out_dir/"v1_adaptive_partial.csv", index=False)
        except Exception as e:
            print(f"  [skip] {model}/{dataset}/{pid}: {e}", flush=True)

    df_new = pd.DataFrame(rows)
    unified = pd.read_csv(out_dir/"unified_all_models_results.csv")
    refs = unified[unified.policy.isin(['v1_linear','tova'])]
    pct_v1, pct_tova, cr = [], [], []
    for _, r in df_new.iterrows():
        v1 = refs[(refs.model==r.model)&(refs.dataset==r.dataset)&
                  (refs.prompt_id==r.prompt_id)&(refs.K_nominal==r.K_nominal)&
                  (refs.policy=='v1_linear')]
        tv = refs[(refs.model==r.model)&(refs.dataset==r.dataset)&
                  (refs.prompt_id==r.prompt_id)&(refs.K_nominal==r.K_nominal)&
                  (refs.policy=='tova')]
        pct_v1.append(100*(r.kl_mean/float(v1.kl_mean.iloc[0])-1) if len(v1) else np.nan)
        pct_tova.append(100*(r.kl_mean/float(tv.kl_mean.iloc[0])-1) if len(tv) else np.nan)
        cr.append(r.actual_K/float(tv.actual_K.iloc[0]) if len(tv) else np.nan)
    df_new["pct_vs_v1"] = pct_v1
    df_new["pct_vs_tova"] = pct_tova
    df_new["cache_ratio_vs_tova"] = cr
    df_new.to_csv(out_dir/"v1_adaptive_results.csv", index=False)
    print(f"\n[v1-adaptive-v2] wrote {len(df_new)} rows in {time.time()-t0:.0f}s")

    common = ['model','dataset','prompt_id','K_nominal','policy','n_kv',
              'actual_K','kl_mean','kl_min','kl_max','mass_pct',
              'pct_vs_v1','pct_vs_tova','cache_ratio_vs_tova']
    big = pd.concat([unified[unified.columns.intersection(common).tolist()],
                     df_new[df_new.columns.intersection(common).tolist()]],
                    ignore_index=True)
    big.to_csv(out_dir/"unified_with_v1_adaptive.csv", index=False)
    print(f"merged → unified_with_v1_adaptive.csv ({len(big)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
