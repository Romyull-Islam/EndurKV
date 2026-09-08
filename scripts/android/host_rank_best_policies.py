"""Consolidated 'best policy across all sims' ranker.

Reads every `pareto_summary.csv` we can find under logs/ (both layer-averaged
sims and per-head sims) and produces a single headline table:

    | rank | policy | sim_kind | dataset | n_prompts | KL@K=64 | KL@K=128 | KL@K=256 | margin vs TOVA |

The point: one place to see which policy wins on which dataset, across the
whole experimental record. Drives the final 'best policy' answer.

Outputs to logs/_consolidated/:
  - best_per_dataset.csv     (best policy per dataset/budget)
  - global_ranking.csv       (across all sims)
  - tova_margins.csv         (each policy's mean margin over TOVA)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(os.environ.get("WORKSPACE", str(Path(__file__).resolve().parents[3])))
LOGS = WORKSPACE / "logs"


def tag_dir(dirname: str) -> tuple[str, str, str]:
    """Infer (model_size, context_type, sim_kind) from a study dir name."""
    name = dirname.lower()
    # Skip smoke / dev dirs
    if "smoke" in name:
        return ("smoke", "smoke", "smoke")
    # Model size
    if "_8b" in name:
        model = "8B"
    elif "_1b" in name:
        model = "1B"
    else:
        model = "?"
    # Context type
    if "longctx" in name or "long_ctx" in name:
        ctx = "long-ctx"
    elif "reasoning" in name:
        ctx = "reasoning"
    elif "eviction_v2" in name or name.endswith("_eviction_v2"):
        # study_phone_{1b,8b}_eviction_v2 = 80-prompt short-ctx eviction sweep
        ctx = "short-ctx"
    else:
        ctx = "short-ctx"
    # Sim kind
    if "perhead" in name or "per_head" in name:
        kind = "per-head"
    elif "math3" in name:
        kind = "layer-avg+math3"
    elif "math2" in name:
        kind = "layer-avg+math2"
    elif "math" in name:
        kind = "layer-avg+math"
    elif "offgrid" in name:
        kind = "layer-avg (offgrid)"
    else:
        kind = "layer-avg"
    return (model, ctx, kind)


def load_all() -> pd.DataFrame:
    """Auto-discover EVERY pareto_summary.csv under logs/ (skip smoke + _consolidated/_xarch)."""
    parts = []
    for p in LOGS.rglob("pareto_summary.csv"):
        if "_consolidated" in p.parts or "_xarch" in p.parts:
            continue
        parent = p.parent.name
        model, ctx, kind = tag_dir(parent)
        if model == "smoke":
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  [skip] {parent}: {e}"); continue
        # Filter degenerate per-head sims where every row is NaN (pre-fix)
        if "mean_kl" in df.columns and df["mean_kl"].isna().all():
            print(f"  [skip] {parent}: all-NaN (pre-fix sim)"); continue
        df["dataset"] = f"{model}-{ctx}"
        df["sim_kind"] = kind
        df["source_dir"] = parent
        df["model_size"] = model
        df["context_type"] = ctx
        parts.append(df)
    if not parts:
        print("No pareto_summary.csv files found yet."); return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    out_dir = LOGS / "_consolidated"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df = load_all()
    if all_df.empty:
        return 1
    print(f"Loaded {len(all_df)} rows from {all_df['source_dir'].nunique()} sims:")
    for d in all_df["source_dir"].unique():
        print(f"  - {d}")

    # 1. Best policy per (dataset, sim_kind, K_nominal)
    rows = []
    for (dataset, kind, K), g in all_df.groupby(["dataset", "sim_kind", "K_nominal"]):
        non_full = g[~g["policy"].isin(["full"])]
        if non_full.empty: continue
        winner = non_full.sort_values("mean_kl").iloc[0]
        # Find TOVA at the closest K_nominal for comparison
        tova_pool = non_full[non_full["policy"].isin(["tova", "perhead_tova"])]
        if not tova_pool.empty:
            tova_row = tova_pool.iloc[(tova_pool["K_nominal"] - K).abs().argsort()[:1]].iloc[0]
            tova_kl = float(tova_row["mean_kl"])
            margin = (float(winner["mean_kl"]) - tova_kl) / max(1e-9, tova_kl) * 100
        else:
            tova_kl = None; margin = None
        rows.append({
            "dataset": dataset, "sim_kind": kind, "K_nominal": int(K),
            "best_policy": winner["policy"],
            "best_KL": round(float(winner["mean_kl"]), 4),
            "best_avg_K": round(float(winner["avg_actual_K"]), 1),
            "tova_KL_at_same_K": round(tova_kl, 4) if tova_kl is not None else None,
            "winner_margin_pct_vs_tova": round(margin, 2) if margin is not None else None,
        })
    best_per = pd.DataFrame(rows).sort_values(["dataset", "K_nominal"])
    best_per.to_csv(out_dir / "best_per_dataset.csv", index=False)
    print(f"\n=== BEST POLICY per dataset x K ===")
    if not best_per.empty:
        print(best_per.to_string(index=False))

    # 2. Mean margin vs TOVA per policy (across ALL datasets, K_nominals)
    margin_rows = []
    for (policy, sim_kind), g in all_df.groupby(["policy", "sim_kind"]):
        if policy in ("full", "tova", "perhead_tova", "local"): continue
        margins = []
        for _, r in g.iterrows():
            tova_candidates = all_df[
                (all_df["sim_kind"] == sim_kind)
                & (all_df["dataset"] == r["dataset"])
                & (all_df["K_nominal"] == r["K_nominal"])
                & (all_df["policy"].isin(["tova", "perhead_tova"]))
            ]
            if tova_candidates.empty: continue
            tova_kl = float(tova_candidates["mean_kl"].iloc[0])
            margins.append((float(r["mean_kl"]) - tova_kl) / max(1e-9, tova_kl) * 100)
        if margins:
            margin_rows.append({
                "policy": policy, "sim_kind": sim_kind,
                "n_observations": len(margins),
                "mean_margin_vs_tova_pct": round(float(np.mean(margins)), 2),
                "median_margin_vs_tova_pct": round(float(np.median(margins)), 2),
                "best_margin_vs_tova_pct": round(float(min(margins)), 2),
                "worst_margin_vs_tova_pct": round(float(max(margins)), 2),
            })
    margins_df = pd.DataFrame(margin_rows).sort_values(
        ["sim_kind", "mean_margin_vs_tova_pct"]
    )
    margins_df.to_csv(out_dir / "tova_margins.csv", index=False)
    print(f"\n=== POLICY margin vs TOVA (negative = better) ===")
    if not margins_df.empty:
        print(margins_df.to_string(index=False))

    # 3. Global ranking (combined across all sims)
    glob_rows = []
    for policy, g in all_df.groupby("policy"):
        if policy == "full": continue
        glob_rows.append({
            "policy": policy,
            "n_rows": len(g),
            "datasets": g["dataset"].nunique(),
            "mean_kl_overall": round(float(g["mean_kl"].mean()), 4),
            "median_kl_overall": round(float(g["mean_kl"].median()), 4),
            "min_kl": round(float(g["mean_kl"].min()), 4),
        })
    glob = pd.DataFrame(glob_rows).sort_values("mean_kl_overall")
    glob.to_csv(out_dir / "global_ranking.csv", index=False)
    print(f"\n=== GLOBAL ranking (mean KL across all sims) ===")
    print(glob.to_string(index=False))

    print(f"\nAll outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
