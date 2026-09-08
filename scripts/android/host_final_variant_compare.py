#!/usr/bin/env python3
"""Final streamlined cross-variant comparison.

Runs every perhead variant (v1-v7) + TOVA on 1 prompt per long-context dir,
at K ∈ {512, 1024} — the budgets that matter for mobile. Total ~70 sim calls,
~10 min wall clock.

Outputs:
  EndurKV/figures/final_variant_results.csv  — per-(model, K, variant) results
  EndurKV/figures/final_evaluation_tables.md — clean markdown tables for the paper
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import load_attn_perhead, simulate

VARIANTS = ["perhead_v1", "perhead_v2", "perhead_v3", "perhead_v4",
            "perhead_v5", "perhead_v6", "perhead_v7", "perhead_tova"]
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
    out_dir.mkdir(exist_ok=True)
    rows = []
    t0 = time.time()
    for d_str in DIRS:
        d = Path(d_str)
        if not d.is_dir(): continue
        model = d.name.replace("study_phone_", "").replace("_longbench", "")
        attn_files = sorted([f for f in d.glob("*.attn.bin")
                             if not f.name.endswith(".v1.attn.bin")])[:1]
        for af in attn_files:
            pid = af.name[:-len(".attn.bin")]
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None: continue
            for K in BUDGETS:
                for pol in VARIANTS:
                    kls, Ks, mass, _ = simulate(attn_ph, n_kv_at, None, None, pol, K)
                    rows.append({"model": model, "prompt_id": pid,
                                 "n_kv": int(n_kv_at[-1]) if len(n_kv_at) else 0,
                                 "K_nominal": K, "variant": pol,
                                 "actual_K": float(np.mean(Ks)),
                                 "kl_mean": float(np.mean(kls)),
                                 "kl_min": float(np.min(kls)),
                                 "kl_max": float(np.max(kls)),
                                 "kl_std": float(np.std(kls)),
                                 "mass": float(np.mean(mass))})
        print(f"  [{time.time()-t0:5.0f}s] {model} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "final_variant_results.csv", index=False)
    print(f"\n[final] wrote {len(df)} rows to final_variant_results.csv")

    # Compute %vs_v1 and %vs_TOVA per (model, K)
    piv = df.pivot_table(index=["model","K_nominal"], columns="variant",
                         values=["kl_mean","actual_K","mass"]).reset_index()
    piv.columns = ["_".join(map(str,c)).rstrip("_") for c in piv.columns]
    print(f"\nPivot columns: {list(piv.columns)}")

    # Per-cell numbers
    cells = []
    for _, r in df.iterrows():
        v1_row = df[(df["model"]==r["model"]) & (df["K_nominal"]==r["K_nominal"]) & (df["variant"]=="perhead_v1")]
        tova_row = df[(df["model"]==r["model"]) & (df["K_nominal"]==r["K_nominal"]) & (df["variant"]=="perhead_tova")]
        v1_kl = float(v1_row["kl_mean"].iloc[0]) if not v1_row.empty else np.nan
        tova_kl = float(tova_row["kl_mean"].iloc[0]) if not tova_row.empty else np.nan
        v1_actK = float(v1_row["actual_K"].iloc[0]) if not v1_row.empty else np.nan
        tova_actK = float(tova_row["actual_K"].iloc[0]) if not tova_row.empty else np.nan
        cells.append({
            "model": r["model"], "n_kv": r["n_kv"], "K_nominal": r["K_nominal"],
            "variant": r["variant"], "actual_K": r["actual_K"],
            "cache_ratio_vs_tova": r["actual_K"]/tova_actK if tova_actK else np.nan,
            "kl_mean": r["kl_mean"], "kl_min": r["kl_min"],
            "kl_max": r["kl_max"], "kl_std": r["kl_std"],
            "mass_pct": 100*r["mass"],
            "pct_vs_v1": 100*(r["kl_mean"]/v1_kl - 1) if v1_kl else np.nan,
            "pct_vs_tova": 100*(r["kl_mean"]/tova_kl - 1) if tova_kl else np.nan,
        })
    cells_df = pd.DataFrame(cells)
    cells_df.to_csv(out_dir / "final_variant_per_cell.csv", index=False)
    print(f"[final] wrote per-cell to final_variant_per_cell.csv")

    # Overall ranking — mean across (model, K) of pct_vs_v1
    overall = (cells_df.groupby("variant")
                 .agg(mean_kl=("kl_mean","mean"),
                      mean_cache_ratio=("cache_ratio_vs_tova","mean"),
                      mean_mass=("mass_pct","mean"),
                      mean_pct_vs_v1=("pct_vs_v1","mean"),
                      mean_pct_vs_tova=("pct_vs_tova","mean"),
                      worst_pct_vs_v1=("pct_vs_v1","max"),
                      best_pct_vs_v1=("pct_vs_v1","min"))
                 .reset_index()
                 .sort_values("mean_pct_vs_v1"))
    print("\n=== OVERALL RANKING (lower mean_pct_vs_v1 = better) ===")
    print(overall.to_string(index=False))
    overall.to_csv(out_dir / "final_variant_ranking.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
