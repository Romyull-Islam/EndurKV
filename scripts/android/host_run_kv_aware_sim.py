#!/usr/bin/env python3
"""Run R-KV / KeyDiff / LaProx on the host-captured K/V data.

For each model with a host K/V capture at /home/mislam22/EndurKV_workspace/logs/host_kv_<model>/,
simulate at K ∈ {512, 1024}. Reuse the existing K/V-aware simulator in
host_simulate_kv_baselines.py.

Outputs:
  EndurKV/figures/kv_aware_per_cell.csv     (per-cell)
  appended to final comparison
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import (
    load_attn_perhead, load_kv_sidecar, simulate
)

MODELS = ["mistral", "qwen2", "gemma2", "r1distill", "phi3"]
PROMPT_ID = "narrativeqa_pub_001"
BUDGETS = [512, 1024]
# K/V-aware policies + cross-check baselines (v1, tova)
POLICIES = ["perhead_v1", "perhead_tova", "rkv", "keydiff", "laprox"]


def main():
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures")
    rows = []
    t0 = time.time()
    for model in MODELS:
        cap_dir = Path(f"/home/mislam22/EndurKV_workspace/logs/host_kv_{model}")
        attn_path = cap_dir / f"{PROMPT_ID}.attn.bin"
        kv_path = cap_dir / f"{PROMPT_ID}.kv.bin"
        if not attn_path.exists() or not kv_path.exists():
            print(f"[skip] {model}: missing capture at {cap_dir}", flush=True)
            continue
        print(f"[{model}] loading attn ({attn_path.stat().st_size/1e6:.0f} MB) "
              f"and kv ({kv_path.stat().st_size/1e6:.0f} MB)", flush=True)
        attn_ph, n_kv_at = load_attn_perhead(attn_path)
        if attn_ph is None:
            print(f"[{model}] attn load failed"); continue
        kv_data = load_kv_sidecar(kv_path)
        if kv_data is None:
            print(f"[{model}] kv load failed"); continue
        # We don't have W_o, so LaProx will use ||V||_2 as proxy for projection.
        for K in BUDGETS:
            for pol in POLICIES:
                kls, Ks, mass, _ = simulate(
                    attn_ph, n_kv_at, kv_data, None, pol, K)
                rows.append({
                    "model": model,
                    "n_kv": int(n_kv_at[-1]) if len(n_kv_at) else 0,
                    "K_nominal": K, "variant": pol,
                    "actual_K": float(np.mean(Ks)),
                    "kl_mean": float(np.mean(kls)),
                    "kl_min": float(np.min(kls)),
                    "kl_max": float(np.max(kls)),
                    "kl_std": float(np.std(kls)),
                    "mass_pct": 100*float(np.mean(mass)),
                })
            print(f"  [{time.time()-t0:5.0f}s] {model} K={K} done", flush=True)
    df = pd.DataFrame(rows)
    # Compute pct_vs_v1 / pct_vs_tova per (model, K)
    for (m, K), grp in df.groupby(["model","K_nominal"]):
        v1 = grp[grp.variant=="perhead_v1"]
        tv = grp[grp.variant=="perhead_tova"]
        if v1.empty or tv.empty: continue
        v1_kl = float(v1.kl_mean.iloc[0])
        tv_kl = float(tv.kl_mean.iloc[0])
        tv_K = float(tv.actual_K.iloc[0])
        for i in grp.index:
            df.at[i, "pct_vs_v1"] = 100*(df.at[i,"kl_mean"]/v1_kl - 1)
            df.at[i, "pct_vs_tova"] = 100*(df.at[i,"kl_mean"]/tv_kl - 1)
            df.at[i, "cache_ratio_vs_tova"] = df.at[i,"actual_K"]/tv_K

    df.to_csv(out_dir / "kv_aware_per_cell.csv", index=False)
    print(f"\n[kv-aware] wrote {len(df)} rows")
    # Quick summary
    agg = (df.groupby("variant")
             .agg(mean_kl=("kl_mean","mean"),
                  mean_cache=("actual_K", lambda s: float((s/df.loc[s.index,"K_nominal"]).mean())),
                  mean_vs_v1=("pct_vs_v1","mean"),
                  mean_vs_tova=("pct_vs_tova","mean"))
             .reset_index().sort_values("mean_vs_tova"))
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
