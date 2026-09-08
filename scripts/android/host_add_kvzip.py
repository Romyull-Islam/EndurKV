#!/usr/bin/env python3
"""Incremental: run kvzip_approx on the same cells as host_final_variant_compare.py
and merge into final_variant_per_cell.csv."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import load_attn_perhead, simulate

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
    cells = pd.read_csv(out_dir / "final_variant_per_cell.csv")
    # Need v1 + tova references for pct columns
    rows = []
    for d_str in DIRS:
        d = Path(d_str)
        if not d.is_dir(): continue
        model = d.name.replace("study_phone_", "").replace("_longbench", "")
        attn_files = sorted([f for f in d.glob("*.attn.bin")
                             if not f.name.endswith(".v1.attn.bin")])[:1]
        for af in attn_files:
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None: continue
            for K in BUDGETS:
                kls, Ks, mass, _ = simulate(attn_ph, n_kv_at, None, None, "kvzip_approx", K)
                v1 = cells[(cells["model"]==model) & (cells["K_nominal"]==K) & (cells["variant"]=="perhead_v1")]
                tv = cells[(cells["model"]==model) & (cells["K_nominal"]==K) & (cells["variant"]=="perhead_tova")]
                v1_kl = float(v1["kl_mean"].iloc[0])
                tv_kl = float(tv["kl_mean"].iloc[0])
                tv_actK = float(tv["actual_K"].iloc[0])
                rows.append({
                    "model": model,
                    "n_kv": int(n_kv_at[-1]) if len(n_kv_at) else 0,
                    "K_nominal": K, "variant": "kvzip_approx",
                    "actual_K": float(np.mean(Ks)),
                    "cache_ratio_vs_tova": float(np.mean(Ks))/tv_actK,
                    "kl_mean": float(np.mean(kls)),
                    "kl_min": float(np.min(kls)),
                    "kl_max": float(np.max(kls)),
                    "kl_std": float(np.std(kls)),
                    "mass_pct": 100*float(np.mean(mass)),
                    "pct_vs_v1": 100*(float(np.mean(kls))/v1_kl - 1),
                    "pct_vs_tova": 100*(float(np.mean(kls))/tv_kl - 1),
                })
        print(f"  {model} done", flush=True)
    new_df = pd.DataFrame(rows)
    merged = pd.concat([cells, new_df], ignore_index=True)
    merged.to_csv(out_dir / "final_variant_per_cell.csv", index=False)
    print(f"appended {len(new_df)} kvzip_approx rows; total now {len(merged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
