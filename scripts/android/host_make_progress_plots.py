#!/usr/bin/env python3
"""
host_make_progress_plots.py — defense-quality research-progress plot panel.

Loads BOTH the 1B and 8B joined CSVs and emits seven plots into
logs/progress_figures/:

  01_rho_per_task_with_paper.png   — per-task rho overlaid with paper baseline
  02_rho_1b_vs_8b_paired.png       — paired-bar cross-regime contrast
  03_peak_temp_per_task.png        — peak active-zone temperature, 1B vs 8B
  04_pswpout_endurance.png         — Δpswpout per prompt, log-scale, 1B vs 8B
  05_latency_vs_temp.png           — per-step decode latency vs ambient temp
  06_sustained_latency_drift.png   — latency trend across the full run
  10_validation_panel.png          — one-page 4-subplot "phone validation" summary

Uses workspace JOINED CSVs (not raw per-prompt files), so this is fast.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", r"D:/Research/EndurKV_workspace"))

# Paper baselines from slides 22 / 23 of the proposal
PAPER_RHO = {
    # 1B short-ctx baseline (slide 23)
    "1B_short": {
        "gov_report":     -0.43,
        "cnn_dailymail":  -0.57,
        "hotpotqa":       -0.40,
        "triviaqa":       -0.28,
        "lcc":            -0.03,
        "trec":            0.19,
    },
    # 8B long-ctx baseline (slides 22/23) — these are NOT exactly comparable to
    # our 8B SHORT-ctx phone runs, but we show them as a reference.
    "8B_long":  {
        "gov_report":     -0.47,
        "multi_news":     -0.47,
        "cnn_dailymail":  -0.43,
        "qasper":         -0.34,  # approximate; paper plot shows ~-0.34
        "multifieldqa_en":-0.23,
        "narrativeqa":    -0.31,
        "qmsum":          -0.32,
        "hotpotqa":       -0.34,
        "samsum":         -0.06,
        "triviaqa":       -0.08,
        "xsum":           -0.20,  # approximate from slide
    },
}

LONG_FORM = {"gov_report", "cnn_dailymail", "xsum", "multi_news", "qasper",
             "multifieldqa_en", "hotpotqa", "samsum", "narrativeqa", "qmsum"}
OOS = {"lcc", "trec", "piqa", "openbookqa", "triviaqa"}

TRIP_PATTERNS = ("cpu-hw-trip", "_trip_", "bcl-lvl", "ibat-lvl", "vbat",
                 "pmh", "pmr", "pmih010", "wireless", "usb", "sdr0")


def is_trip(col: str) -> bool:
    low = col.lower()
    return any(p in low for p in TRIP_PATTERNS)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9})

    out_dir = WORKSPACE / "logs" / "progress_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_1b_path = WORKSPACE / "logs" / "study_phone_1b_joined.csv"
    df_8b_path = WORKSPACE / "logs" / "study_phone_8b_joined.csv"
    if not df_1b_path.exists() or not df_8b_path.exists():
        print(f"ERROR: need both {df_1b_path.name} and {df_8b_path.name}", file=sys.stderr)
        return 1

    def load(path):
        d = pd.read_csv(path)
        d = d.dropna(subset=["H_nats"])
        d["task"] = d["prompt_id"].str.rsplit("_", n=1).str[0]
        return d
    d1 = load(df_1b_path)
    d8 = load(df_8b_path)
    d1_attn = d1.dropna(subset=["attn_top1_layer_avg"])
    d8_attn = d8.dropna(subset=["attn_top1_layer_avg"])
    print(f"[load] 1B: {len(d1)} steps, {d1.prompt_id.nunique()} prompts, {d1.task.nunique()} tasks")
    print(f"[load] 8B: {len(d8)} steps, {d8.prompt_id.nunique()} prompts, {d8.task.nunique()} tasks")

    # ---- per-task rho table ------------------------------------------------
    def per_task_rho(df):
        rows = []
        for t, g in df.groupby("task"):
            if len(g) >= 10:
                r, p = spearmanr(g["H_nats"], g["attn_top1_layer_avg"])
                rows.append({"task": t, "rho": r, "p": p, "n": len(g)})
        return pd.DataFrame(rows).sort_values("rho")
    r1 = per_task_rho(d1_attn)
    r8 = per_task_rho(d8_attn)

    # ========================================================================
    # 01 — per-task rho with paper baseline overlay
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (title, r, paper_key) in zip(axes,
            [("1B Q4_K_M short-ctx on OnePlus 15", r1, "1B_short"),
             ("8B Q4_K_M short-ctx on OnePlus 15", r8, "1B_short")]):
        order = r.sort_values("rho")
        ys = np.arange(len(order))
        colors = ["#1b6e1b" if rho <= -0.40 else "#5fa55f" if rho <= -0.20
                  else "#cccccc" if rho < 0 else "#c33"
                  for rho in order["rho"]]
        bars = ax.barh(ys, order["rho"], color=colors)
        ax.set_yticks(ys)
        ax.set_yticklabels(order["task"])
        # paper baseline diamond markers (when available)
        for i, (_, row) in enumerate(order.iterrows()):
            pap = PAPER_RHO[paper_key].get(row["task"])
            if pap is not None:
                ax.scatter([pap], [i], marker="D", s=60,
                           edgecolor="black", facecolor="orange", zorder=5,
                           label="paper baseline" if i == 0 else "")
            ax.text(row["rho"] - 0.015 if row["rho"] < 0 else row["rho"] + 0.015,
                    i, f"{row['rho']:+.2f}  n={row['n']}",
                    va="center", ha="right" if row["rho"] < 0 else "left",
                    fontsize=8)
        ax.axvline(-0.20, color="green", ls="--", lw=1.2, label="fallback gate (ρ ≤ −0.20)")
        ax.axvline(-0.37, color="black", ls=":",  lw=1.0, label="paper headline ρ = −0.37")
        ax.axvspan(-0.7, -0.20, color="green", alpha=0.06)
        ax.set_xlim(min(-0.7, order["rho"].min() - 0.05),
                    max(0.35, order["rho"].max() + 0.05))
        ax.set_xlabel("Spearman ρ (H ↔ max attention, layer-averaged)")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="x")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Per-task ρ on OnePlus 15 vs. paper baseline (orange ◆ = paper)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "01_rho_per_task_with_paper.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 01_rho_per_task_with_paper.png")

    # ========================================================================
    # 02 — 1B vs 8B paired bars per task
    # ========================================================================
    common_tasks = sorted(set(r1["task"]) & set(r8["task"]),
                          key=lambda t: r1[r1.task == t]["rho"].iloc[0])
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(common_tasks) + 2.5))
    ys = np.arange(len(common_tasks))
    width = 0.4
    r1_vals = [float(r1[r1.task == t]["rho"]) for t in common_tasks]
    r8_vals = [float(r8[r8.task == t]["rho"]) for t in common_tasks]
    ax.barh(ys - width / 2, r1_vals, height=width, label="1B Q4_K_M (n=80)",
            color="#4a90e2")
    ax.barh(ys + width / 2, r8_vals, height=width, label="8B Q4_K_M (n≈79)",
            color="#e07c3a")
    for i, t in enumerate(common_tasks):
        ax.text(r1_vals[i] - 0.01 if r1_vals[i] < 0 else r1_vals[i] + 0.01,
                i - width / 2, f"{r1_vals[i]:+.2f}",
                va="center", ha="right" if r1_vals[i] < 0 else "left",
                fontsize=7, color="#1b3a66")
        ax.text(r8_vals[i] - 0.01 if r8_vals[i] < 0 else r8_vals[i] + 0.01,
                i + width / 2, f"{r8_vals[i]:+.2f}",
                va="center", ha="right" if r8_vals[i] < 0 else "left",
                fontsize=7, color="#7a3c0e")
    ax.set_yticks(ys); ax.set_yticklabels(common_tasks)
    ax.axvline(0, color="black", lw=0.5)
    ax.axvline(-0.20, color="green", ls="--", lw=1.2, label="fallback gate ρ ≤ −0.20")
    ax.axvspan(-0.7, -0.20, color="green", alpha=0.06)
    ax.set_xlim(min(-0.6, min(r1_vals + r8_vals) - 0.05),
                max(0.4, max(r1_vals + r8_vals) + 0.05))
    ax.set_xlabel("Spearman ρ (H ↔ max attention)")
    ax.set_title("Cross-regime contrast: 1B vs 8B short-ctx on OnePlus 15 (per task)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "02_rho_1b_vs_8b_paired.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 02_rho_1b_vs_8b_paired.png")

    # ========================================================================
    # 03 — peak active-zone temperature per task, paired 1B/8B
    # ========================================================================
    def peak_temp_per_prompt(d):
        zone_cols = [c for c in d.columns if c.endswith("_temp_mc") and not is_trip(c)]
        out = []
        for pid, g in d.groupby("prompt_id"):
            t = pid.rsplit("_", 1)[0]
            v = g[zone_cols].apply(pd.to_numeric, errors="coerce")
            if v.empty: continue
            out.append((t, float(v.max().max()) / 1000.0))
        return pd.DataFrame(out, columns=["task", "max_temp_C"])

    t1 = peak_temp_per_prompt(d1)
    t8 = peak_temp_per_prompt(d8)
    common = sorted(set(t1.task) & set(t8.task))
    fig, ax = plt.subplots(figsize=(13, 5))
    positions = np.arange(len(common))
    data1 = [t1[t1.task == t]["max_temp_C"].values for t in common]
    data8 = [t8[t8.task == t]["max_temp_C"].values for t in common]
    bp1 = ax.boxplot(data1, positions=positions - 0.2, widths=0.35,
                     patch_artist=True, showmeans=True,
                     boxprops=dict(facecolor="#cfe1f5", color="#1b3a66"))
    bp8 = ax.boxplot(data8, positions=positions + 0.2, widths=0.35,
                     patch_artist=True, showmeans=True,
                     boxprops=dict(facecolor="#fbd9bf", color="#7a3c0e"))
    ax.set_xticks(positions); ax.set_xticklabels(common, rotation=30, ha="right")
    ax.axhline(70.0, color="red", ls="--", lw=1.0, label="proposal trip 70°C")
    ax.set_ylabel("max active-zone temperature per prompt (°C)")
    ax.set_title("Peak temperature per task — 1B (blue) vs 8B (orange)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(handles=[bp1["boxes"][0], bp8["boxes"][0],
                       plt.Line2D([0], [0], color="red", ls="--")],
              labels=["1B Q4_K_M", "8B Q4_K_M", "70 °C trip"],
              loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "03_peak_temp_per_task.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 03_peak_temp_per_task.png")

    # ========================================================================
    # 04 — pswpout per prompt, log-scale, 1B vs 8B
    # ========================================================================
    def pswpout_per_prompt(d):
        out = []
        for pid, g in d.groupby("prompt_id"):
            t = pid.rsplit("_", 1)[0]
            if "vmstat_pswpout" not in g.columns: continue
            s = pd.to_numeric(g["vmstat_pswpout"], errors="coerce").dropna()
            if len(s) >= 2:
                delta = int(s.iloc[-1] - s.iloc[0])
                out.append((t, max(delta, 0) * 4))  # × 4 KB
        return pd.DataFrame(out, columns=["task", "kib"])
    p1 = pswpout_per_prompt(d1)
    p8 = pswpout_per_prompt(d8)
    common = sorted(set(p1.task) & set(p8.task))
    fig, ax = plt.subplots(figsize=(13, 5))
    positions = np.arange(len(common))
    data1 = [np.clip(p1[p1.task == t]["kib"].values, 0.5, None) for t in common]
    data8 = [np.clip(p8[p8.task == t]["kib"].values, 0.5, None) for t in common]
    bp1 = ax.boxplot(data1, positions=positions - 0.2, widths=0.35,
                     patch_artist=True, showfliers=True, showmeans=True,
                     boxprops=dict(facecolor="#cfe1f5"))
    bp8 = ax.boxplot(data8, positions=positions + 0.2, widths=0.35,
                     patch_artist=True, showfliers=True, showmeans=True,
                     boxprops=dict(facecolor="#fbd9bf"))
    ax.set_xticks(positions); ax.set_xticklabels(common, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Δpswpout per prompt (KiB written to UFS via swap, log)")
    ax.set_title("Endurance pressure per task — 1B vs 8B (log scale)")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(handles=[bp1["boxes"][0], bp8["boxes"][0]],
              labels=["1B Q4_K_M", "8B Q4_K_M"], loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "04_pswpout_endurance.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 04_pswpout_endurance.png")

    # ========================================================================
    # 05 — decode latency vs temperature (8B mostly)
    # ========================================================================
    def latency_vs_temp(d, label):
        d = d.copy().sort_values(["prompt_id", "step_index"])
        d["dt_ms"] = d.groupby("prompt_id")["wall_clock_us"].diff() / 1000.0
        d = d[d["dt_ms"].between(1, 500)]
        # use the hottest active CPU zone as the ambient temperature
        cpu_cols = [c for c in d.columns if c.endswith("_temp_mc")
                    and "cpu" in c.lower() and not is_trip(c)]
        if not cpu_cols:
            return d, label, None, None
        cpu = d[cpu_cols].apply(pd.to_numeric, errors="coerce") / 1000.0
        d["cpu_max_C"] = cpu.max(axis=1)
        return d, label, "cpu_max_C", "dt_ms"
    d1_lt, _, _, _ = latency_vs_temp(d1, "1B")
    d8_lt, _, x, y = latency_vs_temp(d8, "8B")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, d, name, color in [(axes[0], d1_lt, "1B Q4_K_M", "#4a90e2"),
                               (axes[1], d8_lt, "8B Q4_K_M", "#e07c3a")]:
        if x not in d.columns or len(d) < 30:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.scatter(d[x], d[y], s=4, color=color, alpha=0.25)
        # add per-quantile mean
        try:
            bins = pd.qcut(d[x], q=8, duplicates="drop")
            mu = d.groupby(bins, observed=True)[y].mean()
            centres = [iv.mid for iv in mu.index]
            ax.plot(centres, mu.values, "o-", color="black", lw=1.5,
                    label="per-bin mean")
        except Exception:
            pass
        ax.axvline(70.0, color="red", ls="--", lw=1.0, alpha=0.8, label="70 °C trip")
        ax.set_xlabel("max CPU active-zone temperature (°C) at decode step")
        ax.set_ylabel("per-step decode latency (ms)")
        ax.set_title(f"{name} — latency vs ambient temp")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "05_latency_vs_temp.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 05_latency_vs_temp.png")

    # ========================================================================
    # 06 — sustained-run latency drift (scatter over full run)
    # ========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    for ax, d, name, col in [(axes[0], d1, "1B Q4_K_M, 80 prompts, 4025 steps", "#4a90e2"),
                             (axes[1], d8, "8B Q4_K_M, 79 prompts, ~4000 steps", "#e07c3a")]:
        d = d.copy().sort_values("step_wall_s")
        d["dt_ms"] = d.groupby("prompt_id")["wall_clock_us"].diff() / 1000.0
        d = d[d["dt_ms"].between(1, 600)]
        d["t_rel_s"] = d["step_wall_s"] - d["step_wall_s"].iloc[0]
        ax.scatter(d["t_rel_s"], d["dt_ms"], s=2, color=col, alpha=0.4)
        # rolling median
        try:
            d2 = d.sort_values("t_rel_s")
            d2["roll"] = d2["dt_ms"].rolling(window=200, min_periods=20).median()
            ax.plot(d2["t_rel_s"], d2["roll"], color="black", lw=1.3,
                    label="rolling median (window=200)")
        except Exception:
            pass
        ax.set_xlabel("time since first decode step (s)")
        ax.set_ylabel("per-step decode latency (ms)")
        ax.set_title(f"Sustained-run latency drift — {name}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "06_sustained_latency_drift.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 06_sustained_latency_drift.png")

    # ========================================================================
    # 10 — one-page validation panel (paper-format)
    # ========================================================================
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # (a) per-task rho (8B)
    ax = fig.add_subplot(gs[0, 0])
    ord_ = r8.sort_values("rho")
    colors = ["#1b6e1b" if rho <= -0.40 else "#5fa55f" if rho <= -0.20
              else "#cccccc" if rho < 0 else "#c33" for rho in ord_["rho"]]
    ax.barh(ord_["task"], ord_["rho"], color=colors)
    ax.axvline(-0.20, color="green", ls="--", lw=1.2, label="fallback gate")
    ax.axvline(-0.37, color="black", ls=":",  lw=1.0, label="paper headline")
    ax.set_xlabel("Spearman ρ")
    ax.set_title("(a) 8B on-phone ρ vs. paper")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    # (b) peak temp box
    ax = fig.add_subplot(gs[0, 1])
    common = sorted(set(t1.task) & set(t8.task))
    pos = np.arange(len(common))
    bp1 = ax.boxplot([t1[t1.task == t]["max_temp_C"].values for t in common],
                     positions=pos - 0.2, widths=0.35, patch_artist=True,
                     boxprops=dict(facecolor="#cfe1f5"))
    bp8 = ax.boxplot([t8[t8.task == t]["max_temp_C"].values for t in common],
                     positions=pos + 0.2, widths=0.35, patch_artist=True,
                     boxprops=dict(facecolor="#fbd9bf"))
    ax.set_xticks(pos); ax.set_xticklabels(common, rotation=35, ha="right", fontsize=8)
    ax.axhline(70.0, color="red", ls="--", lw=1.0)
    ax.set_ylabel("peak active-zone temp (°C)")
    ax.set_title("(b) Thermal regime, 1B (blue) vs 8B (orange)")
    ax.grid(alpha=0.3, axis="y")

    # (c) pswpout endurance
    ax = fig.add_subplot(gs[1, 0])
    bp1 = ax.boxplot([np.clip(p1[p1.task == t]["kib"].values, 0.5, None) for t in common],
                     positions=pos - 0.2, widths=0.35, patch_artist=True, showfliers=True,
                     boxprops=dict(facecolor="#cfe1f5"))
    bp8 = ax.boxplot([np.clip(p8[p8.task == t]["kib"].values, 0.5, None) for t in common],
                     positions=pos + 0.2, widths=0.35, patch_artist=True, showfliers=True,
                     boxprops=dict(facecolor="#fbd9bf"))
    ax.set_xticks(pos); ax.set_xticklabels(common, rotation=35, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Δpswpout per prompt (KiB, log)")
    ax.set_title("(c) Endurance pressure — UFS swap bytes")
    ax.grid(alpha=0.3, axis="y", which="both")

    # (d) sustained latency 8B
    ax = fig.add_subplot(gs[1, 1])
    d8s = d8.copy().sort_values("step_wall_s")
    d8s["dt_ms"] = d8s.groupby("prompt_id")["wall_clock_us"].diff() / 1000.0
    d8s = d8s[d8s["dt_ms"].between(1, 600)]
    d8s["t_rel_s"] = d8s["step_wall_s"] - d8s["step_wall_s"].iloc[0]
    ax.scatter(d8s["t_rel_s"], d8s["dt_ms"], s=2, color="#e07c3a", alpha=0.4)
    d8s2 = d8s.sort_values("t_rel_s")
    d8s2["roll"] = d8s2["dt_ms"].rolling(window=200, min_periods=20).median()
    ax.plot(d8s2["t_rel_s"], d8s2["roll"], color="black", lw=1.3, label="rolling median")
    ax.set_xlabel("time since first decode step (s)")
    ax.set_ylabel("per-step decode latency (ms)")
    ax.set_title("(d) 8B sustained-run latency drift")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("EndurKV — OnePlus 15 Validation Panel  (Snapdragon 8 Elite Gen 5 · Android 16 · attention_probe)",
                 fontsize=13, weight="bold")
    fig.savefig(out_dir / "10_validation_panel.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 10_validation_panel.png")

    print(f"\nall figures in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
