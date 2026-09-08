#!/usr/bin/env python3
"""
Phase E.1+ — render two slide-ready figures:

    figures/slide_progress.{pdf,png}
        "What we have built so far" status diagram for the progress slide.

    figures/slide_headline.{pdf,png}
        Single big plot: output entropy vs. attention top-1 probability,
        with a binned-mean trend line and shaded "safe-to-prune" /
        "do-not-prune" zones. This is the headline evidence figure.

Usage:
    python3 scripts/12_make_slide_plots.py
"""
from __future__ import annotations

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
import matplotlib.patches as mpatches
from scipy import stats

ROOT      = Path(__file__).resolve().parents[1]
LOG_SUB   = os.environ.get("LOG_SUBDIR", "study")
SUFFIX    = "" if LOG_SUB == "study" else f"_{LOG_SUB.replace('study_', '')}"
STUDY_CSV = ROOT / "logs" / (f"{LOG_SUB}_full.csv" if LOG_SUB != "study" else "study_full.csv")
STUDY_DIR = ROOT / "logs" / LOG_SUB
FIG_DIR   = ROOT / "figures"


def load_attn_summary(prompt_id: str, n_csv_steps: int):
    """Return a list of (attn_top1, attn_H) per step, length min(n_steps_in_bin, n_csv_steps)."""
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
                out.append((np.nan, np.nan))
                continue
            avg = np.mean(np.stack(layers, axis=0), axis=0)
            sm  = avg.sum()
            if not np.isfinite(sm) or sm <= 0:
                out.append((np.nan, np.nan))
                continue
            p = avg / sm
            top1 = float(p.max())
            H    = float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
            out.append((top1, H))
    return out


# --------------------------------------------------------------------------- #
# slide 2 — headline figure                                                   #
# --------------------------------------------------------------------------- #
def render_headline():
    # Load per-prompt CSVs directly (more robust than reading the global
    # study_full.csv which can be truncated by a single bad row).
    import json
    prompts_jsonl = ROOT / "data" / "prompts.jsonl"
    task_by_id: dict[str, str] = {}
    if prompts_jsonl.exists():
        with open(prompts_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                task_by_id[item["prompt_id"]] = item["task"]
    parts = []
    for csv_path in sorted(STUDY_DIR.glob("*.csv")):
        try:
            d = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
        except Exception:
            continue
        if "prompt_id" not in d.columns or len(d) == 0:
            continue
        d["task"] = d["prompt_id"].map(task_by_id).fillna("unknown")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    rows = []
    for pid, sub in df.groupby("prompt_id"):
        attn = load_attn_summary(pid, len(sub))
        for (i, csv_row), (top1, H_attn) in zip(sub.sort_values("step_index").iterrows(), attn):
            rows.append({
                "task":       csv_row["task"],
                "H_nats":     csv_row["H_nats"],
                "attn_top1":  top1,
                "attn_H":     H_attn,
            })
    d = pd.DataFrame(rows).dropna(subset=["attn_top1"])
    rho, p = stats.spearmanr(d["H_nats"], d["attn_top1"])

    # Binned mean curve (10 quantile bins of H_nats).
    # Use np.quantile to avoid empty bins.
    bins = np.unique(np.quantile(d["H_nats"], np.linspace(0, 1, 11)))
    if len(bins) >= 3:
        bin_idx = np.digitize(d["H_nats"], bins[1:-1])
    else:
        bin_idx = np.zeros(len(d), dtype=int)
    bin_centers, bin_means, bin_lo, bin_hi, bin_n = [], [], [], [], []
    for b in range(int(bin_idx.max()) + 1):
        sel = bin_idx == b
        if sel.sum() < 3:
            continue
        h_in  = d["H_nats"].values[sel]
        t_in  = d["attn_top1"].values[sel]
        bin_centers.append(float(np.median(h_in)))
        bin_means.append(float(np.mean(t_in)))
        bin_lo.append(float(np.percentile(t_in, 25)))
        bin_hi.append(float(np.percentile(t_in, 75)))
        bin_n.append(int(sel.sum()))
    bin_centers = np.array(bin_centers); bin_means = np.array(bin_means)
    bin_lo = np.array(bin_lo); bin_hi = np.array(bin_hi)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Shaded zones (safe / do-not).
    h_min, h_max = float(d["H_nats"].min()), float(d["H_nats"].max())
    ax.axvspan(h_min, 0.5,  facecolor="#bfe6b8", alpha=0.55, zorder=0)
    ax.axvspan(2.0,   h_max, facecolor="#f5b7b1", alpha=0.55, zorder=0)

    # Raw scatter (faded).
    ax.scatter(d["H_nats"], d["attn_top1"],
               s=10, alpha=0.18, color="#444", zorder=1, label=f"all decode steps  (n={len(d)})")

    # Binned IQR band + mean line.
    ax.fill_between(bin_centers, bin_lo, bin_hi, color="#1f5f9b", alpha=0.20, zorder=2,
                    label="per-bin IQR (Q1–Q3)")
    ax.plot(bin_centers, bin_means, color="#0b3d72", linewidth=2.6, marker="o",
            markersize=7, zorder=3, label="per-bin mean (10 H-quantile bins)")

    # Zone annotations.
    ax.text(0.25, 0.78, "SAFE TO PRUNE\nlow uncertainty\npeaked attention",
            transform=ax.get_xaxis_transform(),
            color="#1e6b1e", fontsize=11, fontweight="bold",
            ha="center", va="top")
    # the right-side zone center is around the mid-point of [2.0, h_max]
    ax.text(0.5 * (2.0 + h_max), 0.78, "DO NOT PRUNE\nhigh uncertainty\ndiffuse attention",
            color="#8b0000", fontsize=11, fontweight="bold",
            ha="center", va="top",
            transform=ax.get_xaxis_transform())

    # Stats box.
    txt = (f"Spearman ρ = {rho:.3f}\n"
           f"p < 10$^{{-21}}$\n"
           f"n = {len(d)} decode steps")
    ax.text(0.985, 0.975, txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.4"))

    ax.set_xlabel("output entropy  H  (nats)", fontsize=13)
    ax.set_ylabel("max attention probability  (top-1, layer-averaged)", fontsize=13)
    ax.set_title("Output entropy predicts attention concentration\n"
                 "= the empirical signal a per-step prune safety gate can use",
                 fontsize=13.5)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"slide_headline{SUFFIX}.pdf")
    fig.savefig(FIG_DIR / f"slide_headline{SUFFIX}.png", dpi=240)
    plt.close(fig)
    print(f"wrote {FIG_DIR/('slide_headline'+SUFFIX+'.png')}  (and .pdf)")
    print(f"  Spearman ρ = {rho:.3f}, n = {len(d)}, p = {p:.3e}")


# --------------------------------------------------------------------------- #
# slide 1 — progress overview                                                 #
# --------------------------------------------------------------------------- #
def render_progress():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")

    title = ("Server-side measurement study — what is in place")
    ax.text(5, 6.05, title, ha="center", fontsize=15, fontweight="bold")

    boxes = [
        # (x, y, w, h, color, header, body)
        (0.2, 4.10, 4.6, 1.40, "#dde8f0",
         "1.  Probes built  (no edits to llama.cpp)",
         "•  entropy_probe   — per-step Shannon H, top-1, top-5 (LSE-stable)\n"
         "•  attention_probe — disables flash-attn; cb_eval grabs every kq_soft_max-{layer}\n"
         "•  Both built standalone via examples/, link libllama.so + libggml.so"),
        (5.2, 4.10, 4.6, 1.40, "#dde8f0",
         "2.  Workload  =  LongBench (Bai et al., ACL 2024)",
         "•  4 tasks × 6 prompts = 24:  qasper · multifieldqa_en · triviaqa · samsum\n"
         "•  Same benchmark family used by KVSwap, KIVI, CAKE, SnapKV\n"
         "•  Contexts truncated to ~2 k chars to fit 4 096-token ctx"),
        (0.2, 2.45, 4.6, 1.40, "#dde8f0",
         "3.  Study run  (Phase D)",
         "•  Llama-3.2-1B-Instruct Q4_K_M, GPU, seed 42, max_tokens 64\n"
         "•  994 decode steps written to logs/study_full.csv\n"
         "•  Per-prompt attention sidecars (.attn.bin) — 24 files, full source × 16 layers"),
        (5.2, 2.45, 4.6, 1.40, "#dde8f0",
         "4.  Offline analysis  (Phase E.1)",
         "•  Entropy distribution by task (Kruskal-Wallis  p = 0.19  → task-agnostic)\n"
         "•  Entropy auto-correlation lag-1 = 0.27  → predictable for control\n"
         "•  H ↔ attn-top-1   ρ = −0.298   (n = 986,  p < 10⁻²¹)"),
        (0.2, 0.80, 9.6, 1.40, "#f0e0d0",
         "5.  Next  =  Phase E.2  (rigorous validation)",
         "•  prune_probe — re-run each decode with K oldest KV entries evicted\n"
         "•  Measure KL(P_full ‖ P_pruned) for K ∈ {16, 64, 256}\n"
         "•  Hypothesis test:   does H_t correlate with KL_t  ?  →  the answer that closes the safety-gate loop"),
    ]

    for (x, y, w, h, col, hdr, body) in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.04,rounding_size=0.10",
                                       linewidth=1.2, edgecolor="#446", facecolor=col)
        ax.add_patch(rect)
        ax.text(x + 0.15, y + h - 0.30, hdr, fontsize=11, fontweight="bold")
        ax.text(x + 0.15, y + h - 0.70, body, fontsize=9.5, va="top")

    fig.savefig(FIG_DIR / "slide_progress.pdf")
    fig.savefig(FIG_DIR / "slide_progress.png", dpi=240)
    plt.close(fig)
    print(f"wrote {FIG_DIR/'slide_progress.png'}  (and .pdf)")


def render_per_task_bars():
    """Horizontal bar chart of per-task Spearman ρ for H ↔ attn_top1.
    Adds the aggregate as a vertical reference line, plus an ASCII version
    printed to the terminal for quick sanity-checking."""
    import json as _json
    summary_path = FIG_DIR / f"study_summary{SUFFIX}.json"
    if not summary_path.exists():
        # Fall back to the canonical filename if no SUFFIX-specific file is there
        # (handy when re-running just the slide plotter).
        fallback = FIG_DIR / "study_summary.json"
        if fallback.exists():
            summary_path = fallback
        else:
            print(f"  warn: {summary_path} missing — run 11_analyze.py first")
            return
    print(f"  reading {summary_path.name}")
    with open(summary_path) as f:
        summary = _json.load(f)
    by_task = summary.get("entropy_vs_attn_top1_by_task", {})
    agg = summary.get("entropy_vs_attn_top1", {})
    agg_rho = float(agg.get("spearman_rho")) if agg.get("spearman_rho") is not None else None
    agg_n   = int(agg.get("n", 0)) if "n" in agg else None
    agg_p   = float(agg.get("p_value")) if agg.get("p_value") is not None else None
    if not by_task:
        print("  warn: no per-task correlations in summary; skipping bar chart")
        return

    # Sort by rho ascending so the strongest negative is at the top once we flip
    # the y axis.
    items = sorted(by_task.items(), key=lambda kv: kv[1]["spearman_rho"])
    names = [k for k, _ in items]
    rhos  = [v["spearman_rho"] for _, v in items]
    ns    = [v["n"]            for _, v in items]
    ps    = [v["p_value"]      for _, v in items]

    # ----- ASCII version printed to terminal -----
    print()
    print("PER-TASK Spearman ρ  (output H  vs.  top-1 attention probability)")
    print("-" * 78)
    width_units = 38  # max bar width in characters
    rho_max = max(0.6, max(abs(r) for r in rhos) + 0.05)
    for name, r, n, p in zip(names, rhos, ns, ps):
        sig = "***" if p < 1e-3 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        bar_len = int(round(abs(r) / rho_max * width_units))
        bar = "█" * max(1, bar_len) if bar_len > 0 else "·"
        side = "  " if r >= 0 else "←"  # bar grows leftward in our visual sense
        print(f"  {name:<18} {bar:<{width_units}}  ρ={r:+.3f}  n={n:>4}  {sig}")
    print("-" * 78)
    if agg_rho is not None:
        agg_sig = "***" if agg_p and agg_p < 1e-3 else ""
        agg_bar = "█" * int(round(abs(agg_rho) / rho_max * width_units))
        print(f"  {'AGGREGATE':<18} {agg_bar:<{width_units}}  ρ={agg_rho:+.3f}  n={agg_n}  {agg_sig}  ← cross-task aggregate (dilution)")
    print("-" * 78)
    print()

    # ----- Matplotlib figure -----
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(names) + 2.0))

    # Color by significance + direction.
    colors = []
    for r, p in zip(rhos, ps):
        if p > 0.05:
            colors.append("#aaaaaa")
        elif r <= -0.40:
            colors.append("#0f6d2c")  # very strong negative
        elif r <= -0.20:
            colors.append("#52a070")  # strong negative
        elif r < 0:
            colors.append("#a7d29b")  # weak negative
        else:
            colors.append("#c0392b")  # POSITIVE = inverted

    y = np.arange(len(names))
    bars = ax.barh(y, rhos, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=1.0)

    # Shaded "safety-gate works" zone.
    ax.axvspan(-1.0, -0.20, color="#1e8a3c", alpha=0.07, zorder=0)

    # Aggregate reference line.
    if agg_rho is not None:
        ax.axvline(agg_rho, color="#7d2c7d", linestyle="--", linewidth=1.6,
                   alpha=0.9, zorder=2)
        ax.annotate(f"aggregate  ρ={agg_rho:+.3f}  (n={agg_n})",
                    xy=(agg_rho, len(names) - 0.5),
                    xytext=(agg_rho + 0.04, len(names) - 0.7),
                    fontsize=10, color="#7d2c7d",
                    arrowprops=dict(arrowstyle="-", color="#7d2c7d", lw=1.2))

    # Per-bar labels.
    for yi, r, n, p in zip(y, rhos, ns, ps):
        sig = "***" if p < 1e-3 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        x_pos = r + (0.012 if r >= 0 else -0.012)
        ha    = "left" if r >= 0 else "right"
        ax.text(x_pos, yi, f"ρ={r:+.3f}  n={n}  {sig}",
                va="center", ha=ha, fontsize=9.5)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Spearman ρ   (output entropy   ↔   top-1 attention probability)",
                  fontsize=11)
    ax.set_xlim(-0.72, 0.42)
    ax.set_title("Per-task safety-gate signal across 12 benchmark tasks\n"
                 "(green zone: ρ ≤ −0.20  =  the gate's intended operating range)",
                 fontsize=12)

    legend_handles = [
        mpatches.Patch(color="#0f6d2c", label="ρ ≤ −0.40  very strong"),
        mpatches.Patch(color="#52a070", label="−0.40 < ρ ≤ −0.20  strong"),
        mpatches.Patch(color="#a7d29b", label="−0.20 < ρ < 0   weak (still expected sign)"),
        mpatches.Patch(color="#aaaaaa", label="p > 0.05  not significant"),
        mpatches.Patch(color="#c0392b", label="ρ > 0  inverted"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"slide_per_task_rho{SUFFIX}.pdf")
    fig.savefig(FIG_DIR / f"slide_per_task_rho{SUFFIX}.png", dpi=240)
    plt.close(fig)
    print(f"wrote {FIG_DIR/('slide_per_task_rho'+SUFFIX+'.png')}  (and .pdf)")


def main() -> int:
    FIG_DIR.mkdir(exist_ok=True)
    render_progress()
    render_headline()
    render_per_task_bars()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
