"""Cross-arch results aggregator + comparison plotter.

After `run_xarch_chain.ps1` finishes (or in between models), this script:
  1. Auto-discovers all `study_phone_<modelname>_short` and `_longctx` dirs.
  2. For each that hasn't been simulated yet (no `pareto_summary.csv`), runs
     `host_simulate_eviction_policies_v2.py` (layer-avg) and optionally
     `host_simulate_eviction_perhead.py` (per-head, if ATNH captures present).
  3. Builds a unified comparison CSV across all models.
  4. Generates a grid figure: rows = study type (short / longctx / reasoning),
     columns = model, lines = top-5 policies on KL-vs-actual-K Pareto.

Outputs to `logs/_xarch/`:
  - all_models_pareto.csv
  - all_models_pareto_grid.png
  - per_model_ranking.csv       (each model's policy ranking at matched K)
  - cross_model_v1_margin.csv   (v1's margin over TOVA per model — the headline number)

Usage:
  python host_xarch_aggregate.py [--no-sim]   # skip simulation, just aggregate
  python host_xarch_aggregate.py --rerun-sim  # force re-run sim
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(os.environ.get("WORKSPACE", r"D:/Research/EndurKV_workspace"))
LOGS = WORKSPACE / "logs"
SIM_LAYER_AVG = WORKSPACE / "EndurKV/scripts/android/host_simulate_eviction_policies_v2.py"
SIM_PERHEAD   = WORKSPACE / "EndurKV/scripts/android/host_simulate_eviction_perhead.py"

# (study_dir_pattern, model_label, study_type)
# Match what run_xarch_chain.ps1 writes
STUDY_PATTERNS = [
    (re.compile(r"study_phone_(mistral7b|qwen2_7b|phi3mini|gemma2_2b|r1distill)_(short|longctx|reasoning)$"),
     None,  # inferred from match group
     None),
    # Also catch the original Llama 1B + 8B studies for the grid
    (re.compile(r"study_phone_(1b|8b)_(short|longctx|longctx_perhead)$"), None, None),
    (re.compile(r"study_phone_(1b|8b)_(eviction_v2)$"), None, None),
]

MODEL_LABEL_MAP = {
    "1b": "Llama-3.2-1B",
    "8b": "Llama-3.1-8B",
    "mistral7b": "Mistral-7B",
    "qwen2_7b": "Qwen2-7B",
    "phi3mini": "Phi-3-mini",
    "gemma2_2b": "Gemma-2-2B",
    "r1distill": "R1-Distill-8B",
}


def discover_studies(rerun_sim: bool = False) -> list[dict]:
    found = []
    for d in sorted(LOGS.iterdir()):
        if not d.is_dir() or not d.name.startswith("study_phone_"):
            continue
        m = None
        for pat, _, _ in STUDY_PATTERNS:
            m = pat.match(d.name)
            if m:
                break
        if not m:
            continue
        model_key, study_type = m.group(1), m.group(2)
        model_label = MODEL_LABEL_MAP.get(model_key, model_key)
        # Capture types: any *.attn.bin?
        bins = list(d.glob("*.attn.bin"))
        if not bins:
            continue
        # Look ahead — is the first file ATNH (per-head) or ATTN (layer-avg)?
        with open(bins[0], "rb") as f:
            magic = f.read(4)
        is_perhead = (magic == b"ATNH")
        # Is the sim already done? Per-head sim writes to a *_perhead sibling
        summary_dir = Path(str(d) + "_perhead") if is_perhead else d
        has_summary = (summary_dir / "pareto_summary.csv").exists()
        found.append({
            "dir": d, "model": model_label, "type": study_type,
            "n_prompts": len(bins), "perhead": is_perhead,
            "needs_sim": rerun_sim or not has_summary,
        })
    return found


def run_sim(study: dict) -> bool:
    sim_script = SIM_PERHEAD if study["perhead"] else SIM_LAYER_AVG
    label = f"{study['model']} {study['type']}"
    print(f"[sim] {label}  ({study['n_prompts']} prompts, "
          f"{'per-head' if study['perhead'] else 'layer-avg'})")
    args = [sys.executable, str(sim_script),
            "--log-dir", str(study["dir"]),
            "--budgets", "64,74,128,149,256,298"]
    if study["perhead"]:
        # perhead sim writes pareto_summary.csv to a *_perhead sibling dir
        args.extend(["--out-dir", str(study["dir"]) + "_perhead"])
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print(f"  [FAIL] returncode={r.returncode}")
            print(r.stderr[-1500:])
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT after 2h]")
        return False


def load_summary(study: dict) -> pd.DataFrame | None:
    base = Path(str(study["dir"]) + "_perhead") if study["perhead"] else study["dir"]
    p = base / "pareto_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["model"] = study["model"]
    df["study_type"] = study["type"]
    df["perhead"] = study["perhead"]
    df["source_dir"] = base.name
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sim",   action="store_true", help="skip running sims; aggregate only")
    ap.add_argument("--rerun-sim", action="store_true", help="re-run sims even if summary exists")
    args = ap.parse_args()

    out_dir = LOGS / "_xarch"
    out_dir.mkdir(parents=True, exist_ok=True)

    studies = discover_studies(rerun_sim=args.rerun_sim)
    if not studies:
        print(f"No qualifying study_phone_* dirs with *.attn.bin found in {LOGS}")
        return 1
    print(f"Discovered {len(studies)} studies:")
    for s in studies:
        print(f"  - {s['model']:<16}  {s['type']:<18}  "
              f"n={s['n_prompts']:<3}  "
              f"perhead={s['perhead']}  needs_sim={s['needs_sim']}")

    # Run sims for those missing summary
    if not args.no_sim:
        for s in studies:
            if s["needs_sim"]:
                run_sim(s)

    # Aggregate
    parts = []
    for s in studies:
        df = load_summary(s)
        if df is not None:
            parts.append(df)
    if not parts:
        print("No pareto_summary.csv to aggregate — run with sim enabled first.")
        return 1
    all_df = pd.concat(parts, ignore_index=True)
    all_df.to_csv(out_dir / "all_models_pareto.csv", index=False)
    print(f"\nwrote {out_dir / 'all_models_pareto.csv'} ({len(all_df)} rows)")

    # v1's margin over TOVA per model+study_type — the headline cross-arch finding
    headline_rows = []
    # Match v1's actual avg_K to TOVA's nearest available K
    v1_name = "endurkv_tova_spread"  # v1 in layer-avg sim
    perhead_v1_name = "perhead_v1"
    for (model, study_type), g in all_df.groupby(["model", "study_type"]):
        # Pick the v1 row at smallest K_nominal
        v1_candidates = g[g["policy"].isin([v1_name, perhead_v1_name])]
        if v1_candidates.empty:
            continue
        for _, v1_row in v1_candidates.iterrows():
            target_K = float(v1_row["avg_actual_K"])
            # Find TOVA / perhead_tova at a K_nominal that gives nearest avg_actual_K
            tova_pool = g[g["policy"].isin(["tova", "perhead_tova"])]
            if tova_pool.empty:
                continue
            tova_row = tova_pool.iloc[(tova_pool["avg_actual_K"] - target_K).abs().argsort()[:1]].iloc[0]
            tova_K  = float(tova_row["avg_actual_K"])
            tova_kl = float(tova_row["mean_kl"])
            v1_kl   = float(v1_row["mean_kl"])
            # Margin in % (negative = v1 better)
            margin_pct = (v1_kl - tova_kl) / max(1e-9, tova_kl) * 100.0
            headline_rows.append({
                "model": model, "study_type": study_type,
                "v1_policy": v1_row["policy"],
                "v1_K_nominal": int(v1_row["K_nominal"]),
                "v1_avg_K": round(target_K, 1),
                "v1_KL": round(v1_kl, 4),
                "tova_at_K_nominal": int(tova_row["K_nominal"]),
                "tova_avg_K": round(tova_K, 1),
                "tova_KL": round(tova_kl, 4),
                "v1_margin_pct_over_tova": round(margin_pct, 2),
            })
    if headline_rows:
        hd = pd.DataFrame(headline_rows)
        hd.to_csv(out_dir / "cross_model_v1_margin.csv", index=False)
        print(f"\n--- Headline: v1 vs TOVA per model x study_type ---")
        print(hd.to_string(index=False))

    # Plot grid
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skip plot")
        return 0

    models = sorted(all_df["model"].unique())
    types  = sorted(all_df["study_type"].unique())
    if not models or not types:
        return 0

    fig, axes = plt.subplots(len(types), len(models),
                             figsize=(3.5 * len(models), 3 * len(types)),
                             squeeze=False, sharex=False, sharey=False)
    # Highlight v1 family + TOVA + best 3 others
    POLICY_COLORS = {
        "endurkv_tova_spread": ("tab:red",   "-",  2.5,  "EndurKV-Evict v1"),
        "perhead_v1":          ("tab:red",   "-",  2.5,  "EndurKV-Evict v1 (per-head)"),
        "tova":                ("tab:blue",  "--", 1.5,  "TOVA"),
        "perhead_tova":        ("tab:blue",  "--", 1.5,  "TOVA (per-head)"),
        "h2o":                 ("gray",      ":",  1.0,  "H2O"),
        "snapkv":              ("gray",      "-.", 1.0,  "SnapKV"),
        "lazyeviction":        ("tab:green", "-",  1.5,  "LazyEviction"),
        "ahakv":               ("tab:orange","-",  1.5,  "AhaKV"),
        "adakv":               ("tab:purple","-",  1.5,  "AdaKV"),
        "endurkv_perhead_v2":  ("tab:pink",  "-",  1.5,  "EndurKV v2 (combined)"),
    }
    for r, t in enumerate(types):
        for c, mdl in enumerate(models):
            ax = axes[r, c]
            sub = all_df[(all_df["model"] == mdl) & (all_df["study_type"] == t)]
            if sub.empty:
                ax.set_axis_off(); continue
            for pol, g in sub.groupby("policy"):
                if pol not in POLICY_COLORS:
                    continue
                color, ls, lw, lbl = POLICY_COLORS[pol]
                g_sorted = g.sort_values("avg_actual_K")
                ax.plot(g_sorted["avg_actual_K"], g_sorted["mean_kl"],
                        color=color, linestyle=ls, linewidth=lw,
                        marker="o", markersize=3, label=lbl)
            ax.set_title(f"{mdl}\n{t}", fontsize=9)
            ax.set_xlabel("avg actual K (tokens)", fontsize=8)
            ax.set_ylabel("mean KL", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == len(models) - 1:
                ax.legend(loc="upper right", fontsize=6)

    fig.suptitle("Cross-architecture Pareto: KL fidelity vs cache size",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    out_png = out_dir / "all_models_pareto_grid.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
