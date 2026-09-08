#!/usr/bin/env python3
"""Extend the comprehensive eval to 2 prompts per non-Llama (model, dataset) cell.
Then merge with the Llama-wide sweep for a single unified results table.

Coverage after this:
  - 5 paper models (phi3, mistral, qwen2, gemma2, r1distill) × 2 prompts each
    × LongBench, NIAH, + reasoning(r1distill only)
  - Mistral × short × 2 prompts
  - Llama-1B + Llama-8B from llama_wide_results.csv (60 prompts × 2 K)
"""
import sys, time
from multiprocessing import Pool, set_start_method
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import load_attn_perhead


# Reuse policies from llama_wide_sweep
sys.path.insert(0, str(Path(__file__).parent))
from host_llama_wide_sweep import (
    policy_tova, policy_v1_linear, policy_v6_logistic,
    policy_v1_sigmoid, policy_pyramidkv, policy_adakv,
    POLICIES, kl, simulate_policy,
)


# Capture cells (model, dataset) → directory + prompt list + K budgets
LOGS = Path("/home/mislam22/EndurKV_workspace/logs")
LONG_K = [512, 1024]
MED_K = [256, 512]
SHORT_K = [64, 128]

CELLS = [
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

    ("r1distill", "reasoning", LOGS/"study_phone_r1distill_reasoning",
     ["gsm8k_pub_001","gsm8k_pub_002"], MED_K),
    ("mistral", "short", LOGS/"study_phone_mistral7b_short",
     ["qasper_001","qasper_002"], SHORT_K),
]


DATA = {}

def init_data():
    print("[extend] loading captures ...", flush=True)
    for model, dataset, d, prompts, K_budgets in CELLS:
        for pid in prompts:
            af = d / f"{pid}.attn.bin"
            if not af.exists():
                print(f"  [skip] {model}/{dataset}/{pid}: missing"); continue
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None: continue
            DATA[(model, dataset, pid)] = (attn_ph, n_kv_at)
    print(f"[extend] loaded {len(DATA)} prompts", flush=True)


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
    print(f"[extend] {len(tasks)} tasks queued")
    t0 = time.time()
    rows = []
    with Pool(processes=12) as pool:
        for i, row in enumerate(pool.imap_unordered(worker, tasks, chunksize=2)):
            rows.append(row)
            if (i+1) % 50 == 0 or i+1 == len(tasks):
                print(f"  [{time.time()-t0:5.0f}s] {i+1}/{len(tasks)} done", flush=True)
    df = pd.DataFrame(rows)

    # pct_vs_v1 / pct_vs_tova / cache_ratio per (model, dataset, prompt, K)
    pct_v1, pct_tova, cr = [], [], []
    for _, r in df.iterrows():
        v1 = df[(df.model==r.model) & (df.dataset==r.dataset) & (df.prompt_id==r.prompt_id) &
                (df.K_nominal==r.K_nominal) & (df.policy=="v1_linear")]
        tv = df[(df.model==r.model) & (df.dataset==r.dataset) & (df.prompt_id==r.prompt_id) &
                (df.K_nominal==r.K_nominal) & (df.policy=="tova")]
        pct_v1.append(100*(r.kl_mean/float(v1.kl_mean.iloc[0]) - 1) if len(v1) else np.nan)
        pct_tova.append(100*(r.kl_mean/float(tv.kl_mean.iloc[0]) - 1) if len(tv) else np.nan)
        cr.append(r.actual_K/float(tv.actual_K.iloc[0]) if len(tv) else np.nan)
    df["pct_vs_v1"] = pct_v1
    df["pct_vs_tova"] = pct_tova
    df["cache_ratio_vs_tova"] = cr

    df.to_csv(out_dir / "nonllama_extended_results.csv", index=False)
    print(f"\n[extend] wrote {len(df)} rows in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    set_start_method("fork")
    sys.exit(main())
