"""Multi-axis robust-policy ranker.

Extends host_rank_best_policies.py with the metrics the dissertation needs to
defend "alltime best eviction algorithm":

  1. mean_kl_overall            — avg KL across every (dataset, K_nominal) cell
  2. worst_case_kl              — max KL across cells (catastrophic-failure guard)
  3. mean_kl_tightK             — avg KL filtered to the TIGHTEST K per dataset
                                  (minimum-memory regime)
  4. mean_margin_vs_tova_pct    — average margin over TOVA at matched K
  5. win_rate_pct               — fraction of cells where this policy is the
                                  rank-1 winner
  6. latency_class              — hand-graded per-step compute cost
                                  (cheap=O(K), midweight=O(K*H), heavy=O(K^2))
  7. coverage                   — n_observations / n_cells_total
                                  (penalises policies tested in fewer regimes)

Produces a single "robust_rank" by rank-product across the four primary axes:
mean_kl_overall, worst_case_kl, mean_kl_tightK, mean_margin_vs_tova_pct.

Outputs (logs/_consolidated/):
  - robust_ranking.csv           (full table, sorted by robust_rank)
  - robust_top10.md              (markdown summary for EXPERIMENTS.md)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(os.environ.get("WORKSPACE", str(Path(__file__).resolve().parents[3])))
LOGS = WORKSPACE / "logs"

# Hand-graded per-step compute cost. "cheap" = ~TOVA cost. "midweight" = adds
# accumulators / EWMA but still O(K*H). "heavy" = matrix ops O(K^2) or SVD.
LATENCY_CLASS = {
    # cheap layer-avg
    "tova": "cheap", "h2o": "cheap", "h2o_norec": "cheap",
    "streamingllm": "cheap", "snapkv": "cheap", "scissorhands": "cheap",
    "endurkv_tova_spread": "cheap", "endurkv_attn_spread": "cheap",
    "endurkv_v2": "cheap", "endurkv_v3": "cheap", "endurkv_v4": "cheap",
    "endurkv_v6": "cheap", "endurkv_v1_logistic": "cheap",
    "endurkv_v1_geometric": "cheap", "endurkv_v1_all": "cheap",
    "endurkv_differential": "cheap", "endurkv_v1_differential": "cheap",
    "lazyeviction": "cheap", "lwkd": "cheap", "ahakv": "cheap",
    "pyramidkv": "cheap", "cake": "cheap", "cake_entropy": "cheap",
    # midweight layer-avg (FFT / Wasserstein require sorting / fft per step)
    "endurkv_spectral": "midweight", "endurkv_v1_spectral": "midweight",
    "endurkv_v1_spec_nodc": "midweight", "endurkv_v1_spec_diff": "midweight",
    "endurkv_wasserstein": "midweight", "endurkv_v1_wasserstein": "midweight",
    "endurkv_v1_phase": "midweight",
    # heavy layer-avg (SVD / eigen / matrix ops)
    "endurkv_v1_svd": "heavy", "endurkv_v1_lowrank_err": "heavy",
    "endurkv_v1_eigencentrality": "heavy", "endurkv_v5": "heavy",
    # per-head
    "perhead_tova": "cheap", "perhead_v1": "cheap",
    "perhead_v1_tir": "cheap",
    "adakv": "cheap", "headkv": "cheap", "duoattention": "cheap",
    "endurkv_perhead": "midweight", "endurkv_disagree": "midweight",
    "endurkv_perhead_v2": "midweight", "perhead_volatility": "midweight",
    "perhead_svd": "heavy",
    # reference
    "full": "n/a", "local": "cheap",
}


def tag_dir(dirname: str) -> tuple[str, str, str]:
    name = dirname.lower()
    if "smoke" in name:
        return ("smoke", "smoke", "smoke")
    # Model detection — order matters (Llama-3.2 is 1B/8B; cross-arch models named explicitly).
    if "mistral" in name:
        model = "Mistral-7B"
    elif "qwen2" in name or "qwen_2" in name:
        model = "Qwen2-7B"
    elif "phi3" in name or "phi_3" in name or "phimini" in name or "phi-3" in name:
        model = "Phi-3-mini"
    elif "gemma2" in name or "gemma_2" in name or "gemma-2" in name:
        model = "Gemma-2-2B"
    elif "r1distill" in name or "r1_distill" in name or "deepseek" in name:
        model = "R1-Distill-8B"
    elif "_8b" in name or ("8b" in name and "_1b" not in name):
        model = "Llama-8B"
    elif "_1b" in name or "1b" in name:
        model = "Llama-1B"
    else:
        model = "?"

    # Context — explicit benchmarks override generic short/long detection.
    if "longbench" in name:
        ctx = "longbench"
    elif "niah" in name:
        ctx = "niah"
    elif "reasoning" in name:
        ctx = "reasoning"
    elif "longctx" in name or "long_ctx" in name or "long_perhead" in name \
         or "_long_" in name or name.endswith("_long") or "longlayer" in name:
        ctx = "long-ctx"
    elif "medctx" in name or "med_ctx" in name or "_med_" in name:
        ctx = "med-ctx"
    else:
        ctx = "short-ctx"

    # Probe kind. Explicit suffix wins. Order matters here:
    #   *_layeravg          → layer-avg (cross-arch ATTN sister dirs)
    #   *perhead* / *per_head* → per-head
    #   eval_<arch>_<bench>  (without _layeravg) → per-head (default cross-arch sim)
    #   anything else        → layer-avg
    if name.endswith("_layeravg") or "_layeravg_" in name:
        kind = "layer-avg"
    elif "perhead" in name or "per_head" in name:
        kind = "per-head"
    elif name.startswith("eval_") and (("longbench" in name) or ("niah" in name) or ("reasoning" in name)):
        kind = "per-head"
    else:
        kind = "layer-avg"
    return (model, ctx, kind)


def load_all() -> pd.DataFrame:
    parts = []
    for p in LOGS.rglob("pareto_summary.csv"):
        if "_consolidated" in p.parts or "_xarch" in p.parts:
            continue
        if "_smoke" in p.parts[-2].lower():
            continue
        parent = p.parent.name
        model, ctx, kind = tag_dir(parent)
        if model == "smoke":
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  [skip] {parent}: {e}"); continue
        if "mean_kl" not in df.columns or df["mean_kl"].isna().all():
            print(f"  [skip] {parent}: all-NaN (pre-fix sim)"); continue
        df["dataset"] = f"{model}-{ctx}"
        df["sim_kind"] = kind
        df["source_dir"] = parent
        df["cell"] = f"{model}-{ctx}-{kind}"
        parts.append(df)
    if not parts:
        print("No pareto_summary.csv files found.")
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    out_dir = LOGS / "_consolidated"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all()
    if df.empty:
        return 1

    # Prefer the freshest sim per cell — pick source_dir with the most rows
    # (so the comprehensive 38-policy sim wins over an older 22-policy sim).
    pref = (df.groupby(["cell", "source_dir"]).size().rename("n_rows").reset_index())
    pref = pref.sort_values(["cell", "n_rows"], ascending=[True, False])
    chosen = pref.drop_duplicates("cell", keep="first")[["cell", "source_dir"]]
    df = df.merge(chosen, on=["cell", "source_dir"], how="inner")

    cells = sorted(df["cell"].unique())
    n_cells = len(cells)
    print(f"Loaded {len(df)} rows across {n_cells} cells:")
    for c in cells:
        sd = df[df["cell"] == c]["source_dir"].iloc[0]
        n = len(df[df["cell"] == c])
        print(f"  {c:<35} from {sd} ({n} rows)")

    # Per-(cell, K_nominal) winner
    win_counter: dict[str, int] = {}
    total_cells_K = 0
    for (cell, K), g in df.groupby(["cell", "K_nominal"]):
        g = g[~g["policy"].isin(["full"])]
        if g.empty:
            continue
        total_cells_K += 1
        winner = g.sort_values("mean_kl").iloc[0]["policy"]
        win_counter[winner] = win_counter.get(winner, 0) + 1

    # Margin vs TOVA per (policy, cell, K) — TOVA at matched cell+K
    margin_rows = []
    for _, r in df.iterrows():
        if r["policy"] in ("full", "tova", "perhead_tova", "local"):
            continue
        # Find matching TOVA row in the same cell at matched K
        tova_name = "perhead_tova" if "per-head" in r["sim_kind"] else "tova"
        match = df[(df["cell"] == r["cell"])
                   & (df["K_nominal"] == r["K_nominal"])
                   & (df["policy"] == tova_name)]
        if match.empty:
            continue
        tova_kl = float(match["mean_kl"].iloc[0])
        margin = (float(r["mean_kl"]) - tova_kl) / max(1e-9, tova_kl) * 100
        margin_rows.append({"policy": r["policy"], "cell": r["cell"],
                            "K_nominal": int(r["K_nominal"]), "margin_pct": margin})
    margin_df = pd.DataFrame(margin_rows)

    # Build per-policy summary
    rows = []
    all_cells_for_policy = df.groupby("policy")["cell"].nunique()
    for policy, g in df.groupby("policy"):
        if policy == "full":
            continue
        # tightest K per dataset
        tight_kl = []
        for cell, g2 in g.groupby("cell"):
            min_K = g2["K_nominal"].min()
            tight_kl.extend(g2[g2["K_nominal"] == min_K]["mean_kl"].tolist())

        m = margin_df[margin_df["policy"] == policy]
        rows.append({
            "policy": policy,
            "n_observations": len(g),
            "n_cells_covered": int(all_cells_for_policy[policy]),
            "coverage_pct": round(all_cells_for_policy[policy] / n_cells * 100, 1),
            "mean_kl_overall": round(float(g["mean_kl"].mean()), 4),
            "median_kl_overall": round(float(g["mean_kl"].median()), 4),
            "worst_case_kl": round(float(g["mean_kl"].max()), 4),
            "mean_kl_tightK": round(float(np.mean(tight_kl)) if tight_kl else float("nan"), 4),
            "mean_margin_vs_tova_pct": round(float(m["margin_pct"].mean()), 2) if not m.empty else None,
            "worst_margin_vs_tova_pct": round(float(m["margin_pct"].max()), 2) if not m.empty else None,
            "win_count": win_counter.get(policy, 0),
            "win_rate_pct": round(win_counter.get(policy, 0) / max(1, total_cells_K) * 100, 1),
            "latency_class": LATENCY_CLASS.get(policy, "?"),
        })
    rank = pd.DataFrame(rows)

    # Robust rank = rank-product across the four primary axes (lower = better).
    # Penalise low coverage by adding a coverage-rank too.
    for col in ("mean_kl_overall", "worst_case_kl", "mean_kl_tightK"):
        rank[f"r_{col}"] = rank[col].rank(method="min", ascending=True)
    # margin: NaN means "couldn't compare to TOVA" — push to the bottom.
    rank["r_margin"] = rank["mean_margin_vs_tova_pct"].rank(method="min", ascending=True, na_option="bottom")
    rank["r_coverage"] = rank["coverage_pct"].rank(method="min", ascending=False)  # higher coverage = lower rank number

    rank["robust_rank_sum"] = (
        rank["r_mean_kl_overall"]
        + rank["r_worst_case_kl"]
        + rank["r_mean_kl_tightK"]
        + rank["r_margin"]
        + 0.5 * rank["r_coverage"]   # coverage weighted half
    )
    rank = rank.sort_values("robust_rank_sum").reset_index(drop=True)
    rank.insert(0, "robust_rank", range(1, len(rank) + 1))
    rank.to_csv(out_dir / "robust_ranking.csv", index=False)

    print("\n" + "=" * 100)
    print(f"ROBUST RANKING — across {n_cells} cells × {total_cells_K} (cell, K) pairs")
    print("=" * 100)
    cols_display = [
        "robust_rank", "policy", "latency_class",
        "mean_kl_overall", "worst_case_kl", "mean_kl_tightK",
        "mean_margin_vs_tova_pct", "win_rate_pct",
        "n_cells_covered",
    ]
    print(rank[cols_display].head(20).to_string(index=False))

    # Markdown top-10 for the experiments log
    md = ["# Robust eviction-policy ranking", ""]
    md.append(f"Aggregated across {n_cells} (model × context × probe-format) cells "
              f"and {total_cells_K} (cell, K) pairs.")
    md.append("Robust rank = sum of ranks on mean_KL, worst-case KL, "
              "mean_KL at tightest K, and margin-vs-TOVA, plus half-weight coverage rank.")
    md.append("")
    md.append("| Rank | Policy | Latency | mean_KL | worst_KL | tightK_KL | vs_TOVA | win% | cells |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in rank.head(10).iterrows():
        md.append(f"| {int(r['robust_rank'])} | `{r['policy']}` | {r['latency_class']} | "
                  f"{r['mean_kl_overall']:.3f} | {r['worst_case_kl']:.3f} | "
                  f"{r['mean_kl_tightK']:.3f} | {r['mean_margin_vs_tova_pct']}% | "
                  f"{r['win_rate_pct']}% | {int(r['n_cells_covered'])}/{n_cells} |")
    md.append("")
    md.append("## Per-cell winners")
    md.append("| Cell | K | Winner | KL |")
    md.append("|---|---|---|---|")
    for (cell, K), g in df.groupby(["cell", "K_nominal"]):
        g = g[~g["policy"].isin(["full"])]
        if g.empty:
            continue
        w = g.sort_values("mean_kl").iloc[0]
        md.append(f"| {cell} | {int(K)} | `{w['policy']}` | {float(w['mean_kl']):.3f} |")
    (out_dir / "robust_top10.md").write_text("\n".join(md))

    print(f"\nOutputs: {out_dir}/robust_ranking.csv, {out_dir}/robust_top10.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
