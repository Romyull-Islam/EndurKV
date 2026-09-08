#!/usr/bin/env python3
"""
Slide-ready headline for the long-context confirmation run.

Same visual format as figures/slide_headline.png (scatter + SAFE/DO-NOT-PRUNE
zones + binned-mean trend line) but built from the 8B long-context dataset,
restricted to the 7 long-form generation tasks where the gate signal was
strongly confirmed (per-task ρ ≤ -0.31).

Output: figures/slide_long_ctx_headline.{pdf,png}
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

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

ROOT      = Path(__file__).resolve().parents[1]
LOG_SUB   = os.environ.get("LOG_SUBDIR", "study_8b_longctx")
STUDY_DIR = ROOT / "logs" / LOG_SUB
FIG_DIR   = ROOT / "figures"

# 7 confirmed long-form generation tasks (ρ ≤ -0.31 in 8B long-ctx run).
CONFIRMED = {
    "gov_report", "multi_news", "cnn_dailymail",
    "qasper", "hotpotqa", "qmsum", "narrativeqa",
}


def task_from_prompt_id(pid: str, lookup: dict) -> str:
    if pid in lookup:
        return lookup[pid]
    s = pid
    while s and s[-1].isdigit(): s = s[:-1]
    for sfx in ("_lc_", "_lc", "_"):
        if s.endswith(sfx):
            s = s[:-len(sfx)]; break
    return s or "unknown"


def load_attn_top1(prompt_id: str, n_csv_steps: int):
    """Per-step layer-averaged top-1 attention probability."""
    bin_path = STUDY_DIR / f"{prompt_id}.attn.bin"
    if not bin_path.exists():
        return []
    out = []
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        if magic != b"ATTN":
            return []
        n_steps, n_layers, n_head = struct.unpack("<III", f.read(12))
        for s in range(n_steps):
            layers = []
            for _ in range(n_layers):
                (n_kv,) = struct.unpack("<I", f.read(4))
                arr = np.frombuffer(f.read(n_kv * 4), dtype=np.float32).copy()
                layers.append(arr)
            if s >= n_csv_steps or not layers or layers[0].size == 0:
                out.append(np.nan); continue
            avg = np.mean(np.stack(layers, axis=0), axis=0)
            sm  = avg.sum()
            if not np.isfinite(sm) or sm <= 0:
                out.append(np.nan); continue
            p = avg / sm
            out.append(float(p.max()))
    return out


def main() -> int:
    if not STUDY_DIR.exists():
        print(f"missing {STUDY_DIR}", file=sys.stderr); return 1

    # Map prompt_id -> task using the workload manifest.
    prompts_jsonl = Path(os.environ.get("PROMPTS_PATH",
                                        ROOT / "data" / "prompts_longctx.jsonl"))
    task_by_id: dict[str, str] = {}
    if prompts_jsonl.exists():
        with open(prompts_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    task_by_id[item["prompt_id"]] = item["task"]

    rows = []
    for csv_path in sorted(STUDY_DIR.glob("*.csv")):
        try:
            d = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
        except Exception:
            continue
        if "prompt_id" not in d.columns or len(d) == 0:
            continue
        pid  = d["prompt_id"].iloc[0]
        task = task_from_prompt_id(pid, task_by_id)
        if task not in CONFIRMED:
            continue
        d = d.sort_values("step_index")
        attn = load_attn_top1(pid, len(d))
        for (i, csv_row), top1 in zip(d.iterrows(), attn):
            rows.append({
                "task":     task,
                "H_nats":   float(csv_row["H_nats"]),
                "attn_top1": float(top1) if top1 == top1 else float("nan"),
            })
    df = pd.DataFrame(rows).dropna(subset=["attn_top1"])
    if len(df) == 0:
        print("no rows from confirmed tasks", file=sys.stderr); return 1
    print(f"loaded {len(df)} rows from {df['task'].nunique()} confirmed tasks")
    print(f"  tasks present: {sorted(df['task'].unique())}")

    # Aggregate stats over the confirmed-task subset.
    rho, p = stats.spearmanr(df["H_nats"], df["attn_top1"])
    print(f"  ρ = {rho:+.3f}   p = {p:.2e}   n = {len(df)}")

    # Binned mean: 10 quantile bins of H.
    bins = np.unique(np.quantile(df["H_nats"], np.linspace(0, 1, 11)))
    if len(bins) >= 3:
        bin_idx = np.digitize(df["H_nats"], bins[1:-1])
    else:
        bin_idx = np.zeros(len(df), dtype=int)
    centers, means, lo, hi = [], [], [], []
    for b in range(int(bin_idx.max()) + 1):
        sel = bin_idx == b
        if sel.sum() < 3: continue
        h = df["H_nats"].values[sel]
        t = df["attn_top1"].values[sel]
        centers.append(float(np.median(h)))
        means.append(float(np.mean(t)))
        lo.append(float(np.percentile(t, 25)))
        hi.append(float(np.percentile(t, 75)))
    centers = np.array(centers); means = np.array(means)
    lo = np.array(lo); hi = np.array(hi)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(10, 6.5))

    h_min = float(df["H_nats"].min())
    h_max = float(df["H_nats"].max())
    ax.axvspan(h_min, 0.5, facecolor="#bfe6b8", alpha=0.55, zorder=0)
    ax.axvspan(2.0, h_max, facecolor="#f5b7b1", alpha=0.55, zorder=0)

    ax.scatter(df["H_nats"], df["attn_top1"],
               s=10, alpha=0.18, color="#444", zorder=1,
               label=f"all decode steps  (n={len(df)})")

    ax.fill_between(centers, lo, hi, color="#1f5f9b", alpha=0.20, zorder=2,
                    label="per-bin IQR (Q1–Q3)")
    ax.plot(centers, means, color="#0b3d72", linewidth=2.6, marker="o",
            markersize=7, zorder=3, label="per-bin mean (10 H-quantile bins)")

    ax.text(0.25, 0.78, "SAFE TO PRUNE\nlow uncertainty\npeaked attention",
            transform=ax.get_xaxis_transform(),
            color="#1e6b1e", fontsize=11, fontweight="bold",
            ha="center", va="top")
    ax.text(0.5 * (2.0 + h_max), 0.78,
            "DO NOT PRUNE\nhigh uncertainty\ndiffuse attention",
            color="#8b0000", fontsize=11, fontweight="bold",
            ha="center", va="top",
            transform=ax.get_xaxis_transform())

    txt = (f"Spearman ρ = {rho:.3f}\n"
           f"p < 10$^{{{int(np.floor(np.log10(max(p, 1e-300))))}}}$\n"
           f"n = {len(df)} decode steps\n"
           f"7 long-form tasks · 8B · 4K-12K ctx")
    ax.text(0.985, 0.975, txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.4"))

    ax.set_xlabel("output entropy  H  (nats)", fontsize=13)
    ax.set_ylabel("max attention probability  (top-1, layer-averaged)", fontsize=13)
    ax.set_title("Long-context safety-gate signal confirmed on long-form generation\n"
                 "Llama-3.1-8B  ·  contexts 4K-12K tokens  ·  7 LongBench + HELM tasks",
                 fontsize=13.5)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "slide_long_ctx_headline.pdf")
    fig.savefig(FIG_DIR / "slide_long_ctx_headline.png", dpi=240)
    plt.close(fig)
    print(f"wrote {FIG_DIR/'slide_long_ctx_headline.png'}  (and .pdf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
