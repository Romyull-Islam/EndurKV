#!/usr/bin/env python3
"""
Phase E.1+ — entropy vs. positional distribution of high-attention source tokens.

For each (prompt, step) pair, computes per-step features describing WHERE in
the source sequence the model's attention is concentrated, then correlates
those features with output entropy.

Features per step (averaging over layers; head-averaged inside each .attn.bin):
    centroid_norm       weighted mean source-token position, normalized to [0,1]
    spread_norm         weighted std of source-token position, normalized to [0,1]
    top1_pos_norm       position of the most-attended source token, normalized
    mass_first_4        fraction of attention on the first 4 source tokens
                        (the "attention sink" region)
    mass_last_16        fraction of attention on the most recent 16 positions
                        (the "recency window")
    mass_middle         1 - mass_first_4 - mass_last_16  (the bulk we'd prune)

Outputs:
    figures/pos_01_entropy_vs_centroid_norm.{pdf,png}
    figures/pos_02_entropy_vs_spread.{pdf,png}
    figures/pos_03_entropy_vs_mass_buckets.{pdf,png}      (3-panel: first/last/middle)
    figures/pos_04_heatmap_entropy_x_top1pos.{pdf,png}    (entropy bin × top1-pos bin density)
    figures/position_summary.json                          numerical results

Usage:
    LOG_SUBDIR=study_8b_longctx PROMPTS_PATH=data/prompts_longctx.jsonl \\
        python3 scripts/16_position_distribution.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

# Auto-reexec under venv.
_ROOT_FOR_VENV = Path(__file__).resolve().parents[1]
_VENV_DIR = _ROOT_FOR_VENV / ".venv"
_VENV_PY  = _VENV_DIR / "bin" / "python3"
if _VENV_PY.exists():
    try:
        _under_venv = Path(sys.prefix).resolve() == _VENV_DIR.resolve()
    except OSError:
        _under_venv = False
    if not _under_venv:
        os.execv(str(_VENV_PY), [str(_VENV_PY), __file__, *sys.argv[1:]])

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT     = Path(__file__).resolve().parents[1]
LOG_SUB  = os.environ.get("LOG_SUBDIR", "study")
SUFFIX   = "" if LOG_SUB == "study" else f"_{LOG_SUB.replace('study_', '')}"
STUDY_DIR = ROOT / "logs" / LOG_SUB
FIG_DIR   = ROOT / "figures"
SUMMARY   = FIG_DIR / f"position_summary{SUFFIX}.json"

K_FIRST = 4    # attention-sink window
K_LAST  = 16   # recency window


# --------------------------------------------------------------------------- #
def load_attn_bin(path: Path):
    with open(path, "rb") as f:
        if f.read(4) != b"ATTN":
            raise ValueError(f"bad magic {path}")
        n_steps, n_layers, n_head = struct.unpack("<III", f.read(12))
        per_step_layer: list[list[np.ndarray]] = []
        for _ in range(n_steps):
            layers = []
            for _ in range(n_layers):
                (n_kv,) = struct.unpack("<I", f.read(4))
                arr = np.frombuffer(f.read(n_kv * 4), dtype=np.float32).copy()
                layers.append(arr)
            per_step_layer.append(layers)
    return n_steps, n_layers, n_head, per_step_layer


def step_position_features(layers_for_step: list[np.ndarray]):
    """Compute positional features from one step's per-layer attention.
    Each per-layer array is already head-averaged (sums to 1 per layer).
    We further average over layers to get a single distribution over source
    positions for this step, then compute features.
    """
    if not layers_for_step or layers_for_step[0].size == 0:
        return None
    n_kv = layers_for_step[0].shape[0]
    valid = [a for a in layers_for_step if a.size == n_kv]
    if not valid:
        return None
    p = np.mean(np.stack(valid, axis=0), axis=0)  # mean over layers
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        return None
    p = p / s

    positions = np.arange(n_kv, dtype=np.float64)
    centroid  = float((p * positions).sum())
    var       = float((p * (positions - centroid) ** 2).sum())
    spread    = float(np.sqrt(max(0.0, var)))

    # normalize to [0,1] by sequence length
    denom = max(1, n_kv - 1)
    centroid_norm = centroid / denom
    spread_norm   = spread   / denom

    top1_idx      = int(np.argmax(p))
    top1_pos_norm = top1_idx / denom

    # Bucket masses
    k_first = min(K_FIRST, n_kv)
    k_last  = min(K_LAST,  n_kv)
    mass_first = float(p[:k_first].sum())
    # Avoid double-counting if n_kv is small
    if n_kv <= K_FIRST + K_LAST:
        mass_last = float(p[max(k_first, n_kv - k_last):].sum())
    else:
        mass_last = float(p[-k_last:].sum())
    mass_middle = max(0.0, 1.0 - mass_first - mass_last)

    return {
        "n_kv": int(n_kv),
        "centroid_norm": centroid_norm,
        "spread_norm":   spread_norm,
        "top1_pos_norm": top1_pos_norm,
        "mass_first_4":  mass_first,
        "mass_last_16":  mass_last,
        "mass_middle":   mass_middle,
    }


# --------------------------------------------------------------------------- #
def task_from_prompt_id(pid: str, lookup: dict) -> str:
    if pid in lookup:
        return lookup[pid]
    s = pid
    while s and s[-1].isdigit(): s = s[:-1]
    for sfx in ("_lc_", "_lc", "_"):
        if s.endswith(sfx):
            s = s[:-len(sfx)]; break
    return s or "unknown"


def main() -> int:
    FIG_DIR.mkdir(exist_ok=True)
    if not STUDY_DIR.exists():
        print(f"missing {STUDY_DIR}", file=sys.stderr); return 1

    # task lookup
    prompts_jsonl = Path(os.environ.get("PROMPTS_PATH",
                                        ROOT / "data" / "prompts.jsonl"))
    task_by_id: dict[str, str] = {}
    if prompts_jsonl.exists():
        with open(prompts_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    task_by_id[item["prompt_id"]] = item["task"]

    # load entropy CSVs and attention sidecars
    rows = []
    n_files_with_attn = 0
    for csv_path in sorted(STUDY_DIR.glob("*.csv")):
        try:
            df_csv = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
        except Exception:
            continue
        if "prompt_id" not in df_csv.columns or len(df_csv) == 0:
            continue
        prompt_id = df_csv["prompt_id"].iloc[0]
        task      = task_from_prompt_id(prompt_id, task_by_id)
        bin_path  = csv_path.with_suffix(".attn.bin")
        if not bin_path.exists():
            continue
        try:
            n_steps, n_layers, n_head, layers = load_attn_bin(bin_path)
        except Exception:
            continue
        n_files_with_attn += 1
        for i, csv_row in df_csv.sort_values("step_index").iterrows():
            step = int(csv_row["step_index"])
            if step >= n_steps: continue
            feat = step_position_features(layers[step])
            if feat is None: continue
            rows.append({
                "prompt_id":   prompt_id,
                "task":        task,
                "step_index":  step,
                "H_nats":      float(csv_row["H_nats"]),
                **feat,
            })
    if not rows:
        print("no usable rows", file=sys.stderr); return 1
    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} step rows from {n_files_with_attn} prompts, "
          f"{df['task'].nunique()} tasks  (LOG_SUBDIR={LOG_SUB})")

    summary = {
        "log_subdir": LOG_SUB,
        "n_rows": int(len(df)),
        "n_prompts": int(df["prompt_id"].nunique()),
        "n_tasks": int(df["task"].nunique()),
        "K_first": K_FIRST,
        "K_last":  K_LAST,
        "global": {},
        "per_task": {},
    }

    # --- global Spearman: entropy vs each feature ---
    feats = ["centroid_norm", "spread_norm", "top1_pos_norm",
             "mass_first_4", "mass_last_16", "mass_middle"]
    for f in feats:
        rho, p = stats.spearmanr(df["H_nats"], df[f])
        summary["global"][f] = {
            "spearman_rho": float(rho), "p_value": float(p),
            "mean": float(df[f].mean()), "median": float(df[f].median()),
        }

    # --- per-task Spearman ---
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task]
        if len(sub) < 8: continue
        per = {}
        for f in feats:
            rho, p = stats.spearmanr(sub["H_nats"], sub[f])
            per[f] = {"n": int(len(sub)), "spearman_rho": float(rho), "p_value": float(p)}
        summary["per_task"][task] = per

    # ====================================================================== #
    # Plot 01 — entropy vs centroid_norm                                      #
    # ====================================================================== #
    fig, ax = plt.subplots(figsize=(8, 5))
    for task, gsub in df.groupby("task"):
        ax.scatter(gsub["H_nats"], gsub["centroid_norm"], s=6, alpha=0.4, label=task)
    rho, p = stats.spearmanr(df["H_nats"], df["centroid_norm"])
    ax.set_xlabel("output entropy H (nats)")
    ax.set_ylabel("attention centroid (normalized position 0=start  1=end)")
    ax.set_title(f"Where does attention concentrate by output entropy?\n"
                 f"Spearman ρ = {rho:+.3f}    p = {p:.1e}    n = {len(df)}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"pos_01_entropy_vs_centroid{SUFFIX}.pdf")
    fig.savefig(FIG_DIR / f"pos_01_entropy_vs_centroid{SUFFIX}.png", dpi=200)
    plt.close(fig)

    # ====================================================================== #
    # Plot 02 — entropy vs spread                                             #
    # ====================================================================== #
    fig, ax = plt.subplots(figsize=(8, 5))
    for task, gsub in df.groupby("task"):
        ax.scatter(gsub["H_nats"], gsub["spread_norm"], s=6, alpha=0.4, label=task)
    rho, p = stats.spearmanr(df["H_nats"], df["spread_norm"])
    ax.set_xlabel("output entropy H (nats)")
    ax.set_ylabel("attention spread (normalized weighted std of position)")
    ax.set_title(f"Spatial spread of attention vs output entropy\n"
                 f"Spearman ρ = {rho:+.3f}    p = {p:.1e}    n = {len(df)}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"pos_02_entropy_vs_spread{SUFFIX}.pdf")
    fig.savefig(FIG_DIR / f"pos_02_entropy_vs_spread{SUFFIX}.png", dpi=200)
    plt.close(fig)

    # ====================================================================== #
    # Plot 03 — entropy quintile vs mass in [first / middle / last] buckets   #
    # ====================================================================== #
    df["H_quintile"] = pd.qcut(df["H_nats"], q=5,
                               labels=["Q1 (lowest H)", "Q2", "Q3", "Q4", "Q5 (highest H)"])
    bucket_means = df.groupby("H_quintile")[["mass_first_4", "mass_last_16", "mass_middle"]].mean()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bottoms = np.zeros(len(bucket_means))
    colors  = {"mass_first_4": "#1f77b4", "mass_last_16": "#2ca02c", "mass_middle": "#d62728"}
    labels  = {"mass_first_4": f"first {K_FIRST}  (sinks)",
               "mass_last_16": f"last {K_LAST}  (recency)",
               "mass_middle":  "middle  (prunable bulk)"}
    for col in ("mass_first_4", "mass_middle", "mass_last_16"):
        vals = bucket_means[col].values
        ax.bar(range(len(bucket_means)), vals, bottom=bottoms,
               color=colors[col], label=labels[col], edgecolor="black", linewidth=0.4)
        bottoms = bottoms + vals
    ax.set_xticks(range(len(bucket_means)))
    ax.set_xticklabels(bucket_means.index, rotation=15, ha="right")
    ax.set_ylabel("share of attention mass (mean across all decode steps)")
    ax.set_title("Attention positional buckets vs. output-entropy quintile\n"
                 "Higher entropy → more mass shifts into the middle (prunable region shrinks)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"pos_03_entropy_vs_mass_buckets{SUFFIX}.pdf")
    fig.savefig(FIG_DIR / f"pos_03_entropy_vs_mass_buckets{SUFFIX}.png", dpi=200)
    plt.close(fig)

    summary["mass_by_entropy_quintile"] = {
        str(q): {col: float(v) for col, v in bucket_means.loc[q].items()}
        for q in bucket_means.index
    }

    # ====================================================================== #
    # Plot 04 — 2D density: entropy bin × top1-position bin                   #
    # ====================================================================== #
    H_bins = np.linspace(0, max(0.5, df["H_nats"].quantile(0.99)), 21)
    P_bins = np.linspace(0, 1, 21)
    H_idx  = np.clip(np.digitize(df["H_nats"],     H_bins) - 1, 0, len(H_bins) - 2)
    P_idx  = np.clip(np.digitize(df["top1_pos_norm"], P_bins) - 1, 0, len(P_bins) - 2)
    grid = np.zeros((len(H_bins) - 1, len(P_bins) - 1), dtype=np.float64)
    for hi, pi in zip(H_idx, P_idx):
        grid[hi, pi] += 1
    # row-normalise so each entropy row sums to 1 — shows P(top1_pos | H bin)
    row_sums = grid.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    grid_n = grid / row_sums

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(grid_n, aspect="auto", origin="lower", cmap="viridis",
                   extent=[P_bins[0], P_bins[-1], H_bins[0], H_bins[-1]])
    ax.set_xlabel("top-1 attention position (normalized 0=start  1=end)")
    ax.set_ylabel("output entropy H (nats)")
    ax.set_title("P(top-1 attention position | output entropy)\n"
                 "row-normalized 2D density — bright = where the most-attended source token tends to sit")
    fig.colorbar(im, ax=ax, label="density (row-normalised)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"pos_04_heatmap_entropy_x_top1pos{SUFFIX}.pdf")
    fig.savefig(FIG_DIR / f"pos_04_heatmap_entropy_x_top1pos{SUFFIX}.png", dpi=200)
    plt.close(fig)

    # ====================================================================== #
    # Save numerical summary                                                  #
    # ====================================================================== #
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    # ====================================================================== #
    # Headline print                                                          #
    # ====================================================================== #
    print()
    print("=" * 78)
    print("HEADLINE — entropy  vs  positional-attention features  (Spearman ρ)")
    print("=" * 78)
    g = summary["global"]
    for f in feats:
        v = g[f]
        sig = "***" if v["p_value"] < 1e-3 else ("**" if v["p_value"] < 0.01 else ("*" if v["p_value"] < 0.05 else "n.s."))
        print(f"  {f:<18}  ρ={v['spearman_rho']:+.3f}  p={v['p_value']:.2e}  "
              f"mean={v['mean']:.3f}  median={v['median']:.3f}   {sig}")
    print()
    print("Mass shares by entropy quintile (mean over steps):")
    for q in bucket_means.index:
        row = bucket_means.loc[q]
        print(f"  {str(q):<22}  first{K_FIRST}={row['mass_first_4']:.3f}  "
              f"middle={row['mass_middle']:.3f}  "
              f"last{K_LAST}={row['mass_last_16']:.3f}")
    print("=" * 78)
    print(f"figures: {FIG_DIR}/pos_*{SUFFIX}.{{pdf,png}}")
    print(f"summary: {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
