#!/usr/bin/env python3
"""
Phase E.1 — offline analysis of the LongBench study.

Reads:
    logs/study_full.csv           per-step entropy + chosen-token stats
    logs/study/<id>.attn.bin      per-step per-layer per-source attention

Writes:
    figures/study_01_entropy_by_task.{pdf,png}
        violin plot, H_nats grouped by task. Kruskal-Wallis test across tasks.
    figures/study_02_entropy_over_time.{pdf,png}
        H_nats vs step_index for 2 prompts per task (8 total) — shows that
        entropy varies *within* a single decode, justifying per-step gating.
    figures/study_03_entropy_autocorrelation.{pdf,png}
        Auto-correlation of H_nats at lags 1–10, averaged within task. Tells
        us whether entropy at step t predicts entropy at step t+k — a
        precondition for using entropy as a control signal.
    figures/study_04_entropy_vs_attention_entropy.{pdf,png}
        Scatter: output entropy (H_nats) vs. attention entropy (entropy of
        the source-token attention distribution, averaged over layers and
        heads). If high output entropy correlates with diffuse attention,
        the model's "uncertainty" is consistent across both signals.
    figures/study_05_attention_concentration_vs_entropy.{pdf,png}
        Scatter: H_nats vs. top-1 attention probability and vs. recent-16
        attention mass. Tests whether the safety-gate intuition is visible
        analytically (sharp attention -> safe to prune; diffuse -> unsafe).
    figures/study_06_heavy_hitters_by_task.{pdf,png}
        H2O-style heavy-hitter plots per task, averaged over prompts in that
        task. Shows which source-token positions stay "hot" across decode
        steps.
    figures/study_summary.json
        all numerical stats (means, p-values, correlations) in one place.

Usage:
    python3 scripts/11_analyze.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
from collections import defaultdict
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
import seaborn as sns
from scipy import stats

ROOT       = Path(__file__).resolve().parents[1]
LOG_SUB    = os.environ.get("LOG_SUBDIR", "study")
SUFFIX     = "" if LOG_SUB == "study" else f"_{LOG_SUB.replace('study_', '')}"
STUDY_CSV  = ROOT / "logs" / (f"{LOG_SUB}_full.csv" if LOG_SUB != "study" else "study_full.csv")
STUDY_DIR  = ROOT / "logs" / LOG_SUB
FIG_DIR    = ROOT / "figures"
SUMMARY    = FIG_DIR / (f"study_summary{SUFFIX}.json")

def make_palette(task_names):
    """Build a {task_name: color} mapping that scales to any task set."""
    n = max(10, len(task_names))
    base = sns.color_palette("tab20", n_colors=n)
    return {t: base[i % n] for i, t in enumerate(sorted(task_names))}


PALETTE: dict = {}  # populated in main()


# --------------------------------------------------------------------------- #
# attention sidecar loader                                                    #
# --------------------------------------------------------------------------- #
def load_attn_bin(path: Path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"ATTN":
            raise ValueError(f"bad magic in {path}")
        n_steps, n_layers, n_head = struct.unpack("<III", f.read(12))
        per_step_layer: list[list[np.ndarray]] = []
        for _ in range(n_steps):
            layers: list[np.ndarray] = []
            for _ in range(n_layers):
                (n_kv,) = struct.unpack("<I", f.read(4))
                arr = np.frombuffer(f.read(n_kv * 4), dtype=np.float32).copy()
                layers.append(arr)
            per_step_layer.append(layers)
    return n_steps, n_layers, n_head, per_step_layer


def step_attention_summary(layers_for_step: list[np.ndarray]):
    """For one step, average per-source-token attention over layers, then
    return summary stats (entropy of attention dist, top1 prob, recent-16 mass)."""
    if not layers_for_step or layers_for_step[0].size == 0:
        return None
    n_kv = layers_for_step[0].shape[0]
    if any(arr.size != n_kv for arr in layers_for_step):
        return None
    avg = np.mean(np.stack(layers_for_step, axis=0), axis=0)  # mean over layers
    s = avg.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    p = avg / s
    H = float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
    top1 = float(p.max())
    recent_k = 16
    recent_mass = float(p[-recent_k:].sum()) if n_kv >= 1 else 0.0
    return {"attn_H": H, "attn_top1": top1, "attn_recent16": recent_mass, "n_kv": int(n_kv), "p": p}


# --------------------------------------------------------------------------- #
# main analysis                                                               #
# --------------------------------------------------------------------------- #
def main() -> int:
    FIG_DIR.mkdir(exist_ok=True)

    # Load per-prompt CSVs directly (more robust than relying on the global
    # study_full.csv which can be truncated if a single row in any per-prompt
    # CSV has a parse oddity).
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
    else:
        print(f"warn: {prompts_jsonl} not found — will derive task from prompt_id prefix", file=sys.stderr)

    def task_from_prompt_id(pid: str) -> str:
        """Fallback when prompts.jsonl doesn't list this prompt_id.
        Our prompt_id naming is `<task>_<NNN>` or `<task>_lc_<NN>`, so strip the trailing
        digits and any '_lc' suffix to recover the task name."""
        if pid in task_by_id:
            return task_by_id[pid]
        s = pid
        # strip trailing digits
        while s and s[-1].isdigit():
            s = s[:-1]
        # strip any trailing '_' or '_lc_' or '_lc'
        for suffix in ("_lc_", "_lc", "_"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                break
        return s or "unknown"

    parts = []
    n_files = 0
    n_skipped = 0
    for csv_path in sorted(STUDY_DIR.glob("*.csv")):
        try:
            d = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
        except Exception as e:
            print(f"  warn: skipping {csv_path.name} ({e})", file=sys.stderr)
            n_skipped += 1
            continue
        if "prompt_id" not in d.columns or len(d) == 0:
            continue
        d["task"] = d["prompt_id"].apply(task_from_prompt_id)
        parts.append(d)
        n_files += 1
    if not parts:
        print(f"no usable per-prompt CSVs found under {STUDY_DIR}", file=sys.stderr)
        return 1
    df = pd.concat(parts, ignore_index=True)
    print(f"loaded {len(df)} rows from {n_files} per-prompt CSVs (skipped {n_skipped})")
    print(f"tasks: {sorted(df['task'].unique())}")
    print(f"prompts: {df['prompt_id'].nunique()}")

    global PALETTE
    PALETTE = make_palette(df["task"].unique().tolist())

    # --- Per-step attention features (joined onto df) -------------------------
    print("loading attention sidecars...")
    attn_features = []
    per_task_avg_attn = defaultdict(list)  # task -> list of per-step prob vectors (variable length)
    for prompt_id, sub in df.groupby("prompt_id"):
        bin_path = STUDY_DIR / f"{prompt_id}.attn.bin"
        if not bin_path.exists():
            print(f"  warn: missing {bin_path}", file=sys.stderr)
            continue
        n_steps, n_layers, n_head, layers = load_attn_bin(bin_path)
        n_csv_steps = len(sub)
        n_use = min(n_steps, n_csv_steps)
        task = sub["task"].iloc[0]
        for step in range(n_use):
            row = step_attention_summary(layers[step])
            if row is None:
                continue
            attn_features.append({
                "prompt_id": prompt_id,
                "step_index": step,
                "attn_H": row["attn_H"],
                "attn_top1": row["attn_top1"],
                "attn_recent16": row["attn_recent16"],
                "n_kv": row["n_kv"],
            })
            per_task_avg_attn[task].append(row["p"])
    attn_df = pd.DataFrame(attn_features)
    df = df.merge(attn_df, on=["prompt_id", "step_index"], how="left")
    print(f"merged: {len(df)} rows; {df['attn_H'].notna().sum()} rows have attention data")

    summary: dict = {
        "n_rows": int(len(df)),
        "n_prompts": int(df["prompt_id"].nunique()),
        "n_tasks": int(df["task"].nunique()),
        "tasks": sorted(df["task"].unique().tolist()),
        "by_task": {},
    }
    for t, sub in df.groupby("task"):
        summary["by_task"][t] = {
            "n_rows": int(len(sub)),
            "H_nats_mean": float(sub["H_nats"].mean()),
            "H_nats_median": float(sub["H_nats"].median()),
            "H_nats_std": float(sub["H_nats"].std()),
            "attn_H_mean": float(sub["attn_H"].mean(skipna=True)) if sub["attn_H"].notna().any() else None,
            "attn_top1_mean": float(sub["attn_top1"].mean(skipna=True)) if sub["attn_top1"].notna().any() else None,
        }

    # ===================================================================== #
    # 01 — entropy distribution by task (violin) + Kruskal-Wallis            #
    # ===================================================================== #
    print("plot 01: entropy by task ...")
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * df["task"].nunique() + 2), 4.5))
    sns.violinplot(data=df, x="task", y="H_nats", hue="task", ax=ax,
                   palette=PALETTE, legend=False, inner="quartile", cut=0)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel("output entropy H (nats)")
    ax.set_xlabel("LongBench task")
    ax.set_title("Per-step output entropy by task")
    samples_by_task = [df.loc[df.task == t, "H_nats"].values for t in summary["tasks"]]
    if len(samples_by_task) >= 2 and all(len(s) > 0 for s in samples_by_task):
        kw_stat, kw_p = stats.kruskal(*samples_by_task)
        ax.text(0.02, 0.98, f"Kruskal–Wallis: H={kw_stat:.2f}, p={kw_p:.2e}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.5"))
        summary["kruskal_wallis_entropy_by_task"] = {"H": float(kw_stat), "p_value": float(kw_p)}
    fig.tight_layout()
    fig.savefig(FIG_DIR / "study_01_entropy_by_task.pdf")
    fig.savefig(FIG_DIR / "study_01_entropy_by_task.png", dpi=200)
    plt.close(fig)

    # ===================================================================== #
    # 02 — entropy over time (2 prompts per task)                            #
    # ===================================================================== #
    print("plot 02: entropy over time ...")
    n_tasks = len(summary["tasks"])
    n_cols = min(4, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.6 * n_rows), sharey=True)
    axes_flat = axes.flatten() if n_tasks > 1 else [axes]
    for ax, task in zip(axes_flat, summary["tasks"]):
        prompts = sorted(df.loc[df.task == task, "prompt_id"].unique())[:2]
        for pid in prompts:
            sub = df[df["prompt_id"] == pid].sort_values("step_index")
            ax.plot(sub["step_index"], sub["H_nats"], marker="o", markersize=2, linewidth=1, label=pid)
        ax.set_title(task, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("H", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
    for ax in axes_flat[n_tasks:]:
        ax.axis("off")
    fig.suptitle("Entropy trajectory within a decode (2 prompts per task)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "study_02_entropy_over_time.pdf")
    fig.savefig(FIG_DIR / "study_02_entropy_over_time.png", dpi=200)
    plt.close(fig)

    # ===================================================================== #
    # 03 — entropy auto-correlation by task                                  #
    # ===================================================================== #
    print("plot 03: entropy auto-correlation ...")
    lags = [1, 2, 5, 10]
    autocorr_by_task: dict[str, dict[int, float]] = {}
    for task in summary["tasks"]:
        per_lag: dict[int, list[float]] = {l: [] for l in lags}
        for pid, sub in df[df.task == task].groupby("prompt_id"):
            x = sub.sort_values("step_index")["H_nats"].values
            if len(x) < max(lags) + 2:
                continue
            for l in lags:
                a = x[:-l]; b = x[l:]
                if a.std() < 1e-9 or b.std() < 1e-9:
                    continue
                r = float(np.corrcoef(a, b)[0, 1])
                if np.isfinite(r):
                    per_lag[l].append(r)
        autocorr_by_task[task] = {l: float(np.mean(per_lag[l])) if per_lag[l] else float("nan") for l in lags}
    summary["entropy_autocorr_by_task_and_lag"] = autocorr_by_task

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(summary["tasks"]) + 2), 4.5))
    n_t = len(summary["tasks"])
    width = 0.8 / max(1, n_t)
    x = np.arange(len(lags))
    for i, task in enumerate(summary["tasks"]):
        vals = [autocorr_by_task[task].get(l, float("nan")) for l in lags]
        offset = (i - (n_t - 1) / 2.0) * width
        ax.bar(x + offset, vals, width, label=task, color=PALETTE[task])
    ax.set_xticks(x)
    ax.set_xticklabels([f"lag {l}" for l in lags])
    ax.set_ylabel("Pearson auto-correlation of H_nats")
    ax.set_title("Entropy auto-correlation across decode steps")
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "study_03_entropy_autocorrelation.pdf")
    fig.savefig(FIG_DIR / "study_03_entropy_autocorrelation.png", dpi=200)
    plt.close(fig)

    # ===================================================================== #
    # 04 — output entropy vs. attention entropy                              #
    # ===================================================================== #
    print("plot 04: entropy vs. attention entropy ...")
    sub = df.dropna(subset=["attn_H"])
    if len(sub) >= 5:
        pearson_r, pearson_p = stats.pearsonr(sub["H_nats"], sub["attn_H"])
        spearman_rho, spearman_p = stats.spearmanr(sub["H_nats"], sub["attn_H"])
        summary["entropy_vs_attention_entropy"] = {
            "n": int(len(sub)),
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho), "spearman_p": float(spearman_p),
        }
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for t, gsub in sub.groupby("task"):
            ax.scatter(gsub["H_nats"], gsub["attn_H"], s=8, alpha=0.5,
                       color=PALETTE[t], label=t)
        ax.set_xlabel("output entropy H_nats (nats)")
        ax.set_ylabel("attention entropy (nats, layer-avg)")
        ax.set_title("Output entropy vs. attention entropy")
        ax.text(0.02, 0.98,
                f"Pearson r={pearson_r:.3f} (p={pearson_p:.1e})\n"
                f"Spearman ρ={spearman_rho:.3f} (p={spearman_p:.1e})",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.5"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "study_04_entropy_vs_attention_entropy.pdf")
        fig.savefig(FIG_DIR / "study_04_entropy_vs_attention_entropy.png", dpi=200)
        plt.close(fig)

    # ===================================================================== #
    # 05 — attention concentration vs. output entropy                        #
    # ===================================================================== #
    print("plot 05: attention concentration vs. entropy ...")
    sub = df.dropna(subset=["attn_top1", "attn_recent16"])
    if len(sub) >= 5:
        r_top1, p_top1     = stats.spearmanr(sub["H_nats"], sub["attn_top1"])
        r_recent, p_recent = stats.spearmanr(sub["H_nats"], sub["attn_recent16"])
        summary["entropy_vs_attn_top1"]     = {"spearman_rho": float(r_top1),   "p_value": float(p_top1)}
        summary["entropy_vs_attn_recent16"] = {"spearman_rho": float(r_recent), "p_value": float(p_recent)}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for t, gsub in sub.groupby("task"):
            axes[0].scatter(gsub["H_nats"], gsub["attn_top1"], s=8, alpha=0.5, color=PALETTE[t], label=t)
            axes[1].scatter(gsub["H_nats"], gsub["attn_recent16"], s=8, alpha=0.5, color=PALETTE[t], label=t)
        axes[0].set_xlabel("output H_nats")
        axes[0].set_ylabel("max attention prob (top-1, layer-avg)")
        axes[0].set_title(f"Sharper attention vs. lower entropy   ρ={r_top1:.3f}")
        axes[1].set_xlabel("output H_nats")
        axes[1].set_ylabel("recent-16 attention mass")
        axes[1].set_title(f"Recent-window mass vs. entropy   ρ={r_recent:.3f}")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "study_05_attention_concentration_vs_entropy.pdf")
        fig.savefig(FIG_DIR / "study_05_attention_concentration_vs_entropy.png", dpi=200)
        plt.close(fig)

    # ===================================================================== #
    # 06 — H2O-style heavy-hitter aggregate per task                         #
    # ===================================================================== #
    print("plot 06: heavy hitters by task ...")
    n_tasks = len(summary["tasks"])
    n_cols = min(4, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.6 * n_rows))
    axes_iter = axes.flatten() if n_tasks > 1 else [axes]
    for ax, task in zip(axes_iter, summary["tasks"]):
        ps = per_task_avg_attn.get(task, [])
        if not ps:
            ax.set_title(f"{task} (no data)")
            continue
        max_kv = max(p.size for p in ps)
        agg = np.zeros(max_kv, dtype=np.float64)
        cnt = np.zeros(max_kv, dtype=np.float64)
        for p in ps:
            agg[: p.size] += p
            cnt[: p.size] += 1
        cnt[cnt == 0] = 1
        mean = agg / cnt
        ax.bar(np.arange(max_kv), mean, width=1.0, color=PALETTE[task], alpha=0.85)
        ax.set_title(f"{task}  (n={len(ps)})", fontsize=9)
        ax.set_xlabel("src pos", fontsize=8)
        ax.set_ylabel("mean attn", fontsize=8)
        ax.set_yscale("log")
    for ax in list(axes_iter)[n_tasks:]:
        ax.axis("off")
    fig.suptitle("H2O-style heavy-hitter pattern by task (log scale, mean attn prob per source position)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "study_06_heavy_hitters_by_task.pdf")
    fig.savefig(FIG_DIR / "study_06_heavy_hitters_by_task.png", dpi=200)
    plt.close(fig)

    # --- write summary json ---
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {SUMMARY}")

    # --- print a short text summary so the user sees the headline numbers ---
    print()
    print("=" * 72)
    print("HEADLINE NUMBERS")
    print("=" * 72)
    for t, s in summary["by_task"].items():
        print(f"  {t:<20}  n={s['n_rows']:>4}  "
              f"H mean={s['H_nats_mean']:.3f}  median={s['H_nats_median']:.3f}  std={s['H_nats_std']:.3f}  "
              f"attn_H mean={s['attn_H_mean']:.3f}" if s.get("attn_H_mean") is not None else "")
    if "kruskal_wallis_entropy_by_task" in summary:
        kw = summary["kruskal_wallis_entropy_by_task"]
        print(f"\n  Kruskal–Wallis across tasks: H={kw['H']:.2f}, p={kw['p_value']:.2e}")
    if "entropy_vs_attention_entropy" in summary:
        e = summary["entropy_vs_attention_entropy"]
        print(f"  Output H vs. attention H:   r={e['pearson_r']:.3f} (p={e['pearson_p']:.1e})  "
              f"ρ={e['spearman_rho']:.3f} (p={e['spearman_p']:.1e})  n={e['n']}")
    if "entropy_vs_attn_top1" in summary:
        e = summary["entropy_vs_attn_top1"]
        print(f"  H vs. attn top-1:           ρ={e['spearman_rho']:.3f} (p={e['p_value']:.1e})")
    if "entropy_vs_attn_recent16" in summary:
        e = summary["entropy_vs_attn_recent16"]
        print(f"  H vs. recent-16 attn mass:  ρ={e['spearman_rho']:.3f} (p={e['p_value']:.1e})")

    # Per-task correlations — these are usually stronger than the aggregate
    # because aggregate Spearman is diluted by between-task variation.
    print()
    print("PER-TASK Spearman correlation  H_nats  vs  attn_top1   (the safety-gate signal):")
    per_task_rho: dict[str, dict] = {}
    for task in summary["tasks"]:
        s = df[(df.task == task) & df["attn_top1"].notna()]
        if len(s) < 8:
            continue
        rho_t, p_t = stats.spearmanr(s["H_nats"], s["attn_top1"])
        per_task_rho[task] = {"n": int(len(s)), "spearman_rho": float(rho_t), "p_value": float(p_t)}
        marker = " ***" if p_t < 1e-3 else ("  **" if p_t < 0.01 else ("   *" if p_t < 0.05 else "    "))
        print(f"  {task:<22}  n={len(s):>4}  ρ={rho_t:+.3f}  p={p_t:.2e}{marker}")
    summary["entropy_vs_attn_top1_by_task"] = per_task_rho
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print("=" * 72)
    print(f"figures in {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
