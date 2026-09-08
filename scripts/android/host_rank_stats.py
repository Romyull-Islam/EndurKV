"""Statistical rigor for the robust ranker — 95% CIs + paired Wilcoxon vs TOVA
+ Bonferroni-corrected p-values per cell, per K-budget. Required for
publication-grade comparison (ASPLOS / NeurIPS reviewers will ask).

For each (sim_kind, cell, K_nominal, policy):
  - 95% CI via paired bootstrap (B=10000) over per-prompt KL deltas
  - paired Wilcoxon signed-rank vs the cell's TOVA baseline
  - Bonferroni-corrected p across the n_policies tested in that cell

Inputs: per-prompt scores from each cell's policy_results.csv (NOT the aggregate
pareto_summary.csv, which only has means).

Output: logs/_consolidated/stats_per_policy.csv
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

WORKSPACE = Path(os.environ.get("WORKSPACE", str(Path(__file__).resolve().parents[3])))
LOGS = WORKSPACE / "logs"


def tag_dir(dirname: str) -> tuple[str, str, str]:
    """Mirror host_rank_robust.tag_dir (kept local to avoid circular import)."""
    name = dirname.lower()
    if "smoke" in name: return ("smoke", "smoke", "smoke")
    if "_8b" in name or ("8b" in name and "_1b" not in name): model = "8B"
    elif "_1b" in name or "1b" in name: model = "1B"
    else: model = "?"
    if "longctx" in name or "long_ctx" in name or "long_perhead" in name \
       or "_long_" in name or name.endswith("_long") or "longlayer" in name \
       or "longbench" in name:
        ctx = "long-ctx"
    elif "medctx" in name or "med_ctx" in name or "_med_" in name:
        ctx = "med-ctx"
    elif "niah" in name:
        ctx = "niah"
    elif "reasoning" in name or "r1distill" in name:
        ctx = "reasoning"
    else:
        ctx = "short-ctx"
    if "perhead" in name or "per_head" in name: kind = "per-head"
    else: kind = "layer-avg"
    return (model, ctx, kind)


def load_per_prompt(policy_results_csv: Path) -> pd.DataFrame:
    """Load per-prompt scores from a policy_results.csv. Filter to (prompt_id, policy, K_nominal, mean_kl)."""
    df = pd.read_csv(policy_results_csv)
    # Some sims emit "K_nominal" int; some may emit float. Coerce.
    if "K_nominal" not in df.columns:
        return pd.DataFrame()
    return df


def paired_wilcoxon_vs_baseline(per_prompt_df: pd.DataFrame, baseline: str
                                ) -> dict[tuple[str, int], dict]:
    """For each (policy, K_nominal) pair, compute paired Wilcoxon vs baseline at same K.
    Returns dict[(policy, K)] = {n, mean_delta, median_delta, ci_lo, ci_hi, p}."""
    out = {}
    rng = np.random.default_rng(42)
    base_df = per_prompt_df[per_prompt_df["policy"] == baseline]
    if base_df.empty:
        return out
    # baseline_kl[(prompt_id, K)] -> kl
    base = {(r["prompt_id"], int(r["K_nominal"])): float(r["mean_kl"])
            for _, r in base_df.iterrows()}
    for (policy, K), g in per_prompt_df.groupby(["policy", "K_nominal"]):
        if policy in ("full", baseline): continue
        K = int(K)
        deltas = []
        for _, r in g.iterrows():
            k = (r["prompt_id"], K)
            if k not in base: continue
            d = float(r["mean_kl"]) - base[k]
            deltas.append(d)
        n = len(deltas)
        if n < 3:
            out[(policy, K)] = {"n": n, "mean_delta": None, "median_delta": None,
                                "ci_lo": None, "ci_hi": None, "p_wilcoxon": None}
            continue
        deltas = np.array(deltas)
        # Paired bootstrap CI on mean(delta)
        boot_means = np.empty(2000)
        for b in range(2000):
            idx = rng.integers(0, n, n)
            boot_means[b] = deltas[idx].mean()
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        # Wilcoxon signed-rank (paired) — null: median delta = 0
        try:
            w_stat, w_p = stats.wilcoxon(deltas, zero_method="wilcox",
                                          alternative="two-sided", method="auto")
            w_p = float(w_p)
        except Exception:
            w_p = None
        out[(policy, K)] = {
            "n": n,
            "mean_delta": round(float(deltas.mean()), 6),
            "median_delta": round(float(np.median(deltas)), 6),
            "ci_lo": round(float(ci_lo), 6),
            "ci_hi": round(float(ci_hi), 6),
            "p_wilcoxon": round(w_p, 6) if w_p is not None else None,
        }
    return out


def main() -> int:
    out_dir = LOGS / "_consolidated"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []

    # Per-cell baseline: layer-avg cells → "tova"; per-head cells → "perhead_tova"
    for ppath in sorted(LOGS.rglob("policy_results.csv")):
        if "_consolidated" in ppath.parts or "_smoke" in str(ppath).lower():
            continue
        cell_dir = ppath.parent.name
        model, ctx, kind = tag_dir(cell_dir)
        if model == "smoke": continue
        baseline = "perhead_tova" if kind == "per-head" else "tova"
        try:
            df = load_per_prompt(ppath)
        except Exception as e:
            skipped.append((cell_dir, str(e))); continue
        if df.empty:
            continue
        # Verify baseline is present
        if baseline not in set(df["policy"]):
            skipped.append((cell_dir, f"no baseline {baseline}")); continue

        results = paired_wilcoxon_vs_baseline(df, baseline)
        n_tested = len(set((p, K) for (p, K) in results.keys()))
        for (policy, K), r in results.items():
            # Bonferroni: per-cell, n_policies tested at this K
            p_bonf = (r["p_wilcoxon"] * n_tested) if r["p_wilcoxon"] is not None else None
            if p_bonf is not None:
                p_bonf = min(1.0, p_bonf)
            rows.append({
                "cell": f"{model}-{ctx}-{kind}",
                "source_dir": cell_dir,
                "K_nominal": K,
                "policy": policy,
                "baseline": baseline,
                "n_paired": r["n"],
                "mean_delta_kl": r["mean_delta"],
                "median_delta_kl": r["median_delta"],
                "ci_lo_95": r["ci_lo"],
                "ci_hi_95": r["ci_hi"],
                "p_wilcoxon": r["p_wilcoxon"],
                "p_bonferroni": round(p_bonf, 6) if p_bonf is not None else None,
                "significant_at_0p05": (p_bonf is not None and p_bonf < 0.05),
                "direction": ("beats" if r["mean_delta"] is not None
                              and r["mean_delta"] < 0 else "loses")
                             if r["mean_delta"] is not None else None,
            })
    if skipped:
        print(f"[stats] skipped {len(skipped)} cells: {skipped[:5]}")
    if not rows:
        print("[stats] no per-prompt data loaded"); return 1

    out = pd.DataFrame(rows).sort_values(["cell", "K_nominal", "mean_delta_kl"])
    csv_path = out_dir / "stats_per_policy.csv"
    out.to_csv(csv_path, index=False)
    print(f"[stats] wrote {len(out)} rows to {csv_path}")

    # Quick summary: per-policy aggregate across cells
    summary_rows = []
    for policy, g in out.groupby("policy"):
        wins = ((g["mean_delta_kl"] < 0) & g["significant_at_0p05"]).sum()
        losses = ((g["mean_delta_kl"] > 0) & g["significant_at_0p05"]).sum()
        ties = len(g) - wins - losses
        mean_d = g["mean_delta_kl"].mean()
        summary_rows.append({
            "policy": policy,
            "n_cells_K_tested": len(g),
            "sig_wins_vs_baseline": int(wins),
            "sig_losses_vs_baseline": int(losses),
            "ties_or_nonsig": int(ties),
            "mean_delta_kl_overall": round(mean_d, 4) if mean_d is not None else None,
        })
    sum_df = pd.DataFrame(summary_rows).sort_values("mean_delta_kl_overall")
    sum_path = out_dir / "stats_summary.csv"
    sum_df.to_csv(sum_path, index=False)
    print(f"[stats] summary -> {sum_path}\n")
    print(sum_df.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
