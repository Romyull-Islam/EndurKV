#!/usr/bin/env python3
"""
Llama-3.2-1B Wave-3 narrativeqa PPL/DDR/tps bar figure.

Source: /home/mislam22/EndurKV_workspace/phone-logs/wave3_real_1780680903/
Cells:  vanilla / v1_K512 / v1_K2048 / v1_fa_K512

For each cell we read:
  - <cell>/iter*/meta.json       -> per-iter PPL, decode_tps
  - <cell>/iter*/steps.csv       -> per-step NLL (fallback error bar)
  - <cell>/sensors.csv           -> ddr_temp_mc peak for the cell

3-panel figure:
  Top:    PPL bars per cell, error bar from per-iter PPL stdev
          (if zero -> SEM of per-step NLL aggregated across iters)
  Middle: peak DDR temperature per cell
  Bottom: mean decode tok/s per cell (with stdev across iters)

Pareto annotations are overlaid on each panel:
  - v1 K=2048   wins PPL    (1.87)
  - v1_FA K=512 wins tps    (7.77)
  - v1 K=512    wins thermal (49.4 C)

Output: /home/mislam22/EndurKV_workspace/EndurKV/figures/eval_plots/llama1b_ppl_bars.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WAVE_DIR = Path("/home/mislam22/EndurKV_workspace/phone-logs/wave3_real_1780680903")
OUT_PATH = Path(
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/eval_plots/llama1b_ppl_bars.png"
)

CELLS = ["vanilla", "v1_K512", "v1_K2048", "v1_fa_K512"]
LABELS = {
    "vanilla": "vanilla",
    "v1_K512": "v1 K=512",
    "v1_K2048": "v1 K=2048",
    "v1_fa_K512": "v1_FA K=512",
}
COLORS = {
    "vanilla":     "#888888",
    "v1_K512":     "#4C9AFF",
    "v1_K2048":    "#1F77B4",
    "v1_fa_K512":  "#E07B00",
}


def collect_cell(cell_dir: Path) -> dict:
    """Aggregate per-iter PPL / tps and per-step NLL for one cell."""
    ppls: list[float] = []
    tps: list[float] = []
    nlls: list[float] = []
    iter_dirs = sorted([p for p in cell_dir.iterdir() if p.is_dir() and p.name.startswith("iter")])
    for it in iter_dirs:
        meta_p = it / "meta.json"
        if meta_p.exists():
            m = json.loads(meta_p.read_text())
            if m.get("perplexity") is not None:
                ppls.append(float(m["perplexity"]))
            if m.get("decode_tps") is not None:
                tps.append(float(m["decode_tps"]))
        steps_p = it / "steps.csv"
        if steps_p.exists():
            try:
                df = pd.read_csv(steps_p)
                if "nll" in df.columns:
                    nlls.extend(df["nll"].dropna().astype(float).tolist())
            except Exception:
                pass

    ppl_mean = float(np.mean(ppls)) if ppls else float("nan")
    ppl_std_iter = float(np.std(ppls, ddof=0)) if len(ppls) >= 2 else 0.0
    # Fallback error bar: standard error of mean NLL across decode steps,
    # propagated through PPL = exp(mean_nll). dPPL ~= PPL * SEM(nll)
    if nlls:
        nll_arr = np.array(nlls, dtype=float)
        sem_nll = float(nll_arr.std(ddof=1) / math.sqrt(len(nll_arr))) if len(nll_arr) > 1 else 0.0
        ppl_std_nll = ppl_mean * sem_nll
    else:
        ppl_std_nll = 0.0
    ppl_err = ppl_std_iter if ppl_std_iter > 1e-9 else ppl_std_nll

    tps_mean = float(np.mean(tps)) if tps else float("nan")
    tps_std = float(np.std(tps, ddof=0)) if len(tps) >= 2 else 0.0

    return dict(
        n_iters=len(iter_dirs),
        ppl_mean=ppl_mean,
        ppl_err=ppl_err,
        ppl_std_iter=ppl_std_iter,
        ppl_std_nll=ppl_std_nll,
        tps_mean=tps_mean,
        tps_std=tps_std,
    )


def peak_ddr_c(cell_dir: Path) -> float:
    """Read sensors.csv and return peak DDR temperature in degrees C."""
    sensors_p = cell_dir / "sensors.csv"
    if not sensors_p.exists():
        return float("nan")
    df = pd.read_csv(sensors_p, low_memory=False)
    col = "ddr_temp_mc"
    if col not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    # ignore obvious sentinels
    vals = vals[(vals > 0) & (vals < 200000)]
    if vals.empty:
        return float("nan")
    return float(vals.max()) / 1000.0


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for cell in CELLS:
        cdir = WAVE_DIR / cell
        agg = collect_cell(cdir)
        agg["cell"] = cell
        agg["peak_ddr_c"] = peak_ddr_c(cdir)
        rows.append(agg)
    df = pd.DataFrame(rows).set_index("cell").loc[CELLS]

    print("=== Wave-3 Llama-1B aggregate ===")
    print(df.to_string())

    x = np.arange(len(CELLS))
    bar_w = 0.62
    bar_colors = [COLORS[c] for c in CELLS]
    xlabels = [LABELS[c] for c in CELLS]

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.5), sharex=True)
    ax_ppl, ax_ddr, ax_tps = axes

    # ---------------- Top: PPL ----------------
    ppl_vals = df["ppl_mean"].values
    ppl_err = df["ppl_err"].values
    ax_ppl.bar(x, ppl_vals, width=bar_w, color=bar_colors, edgecolor="black", linewidth=0.7,
               yerr=ppl_err, capsize=4, error_kw=dict(ecolor="black", elinewidth=1.0))
    for xi, v in zip(x, ppl_vals):
        ax_ppl.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax_ppl.set_ylabel("Perplexity\n(narrativeqa_pub_001)")
    ax_ppl.set_title("Llama-3.2-1B Wave-3 narrativeqa: 4 policies measured",
                     fontsize=13, fontweight="bold")
    ax_ppl.grid(axis="y", linestyle=":", alpha=0.5)
    ymax_ppl = max(ppl_vals) * 1.18
    ax_ppl.set_ylim(0, ymax_ppl)

    # Pareto annotation: PPL winner = v1_K2048
    winner_ppl_idx = CELLS.index("v1_K2048")
    ax_ppl.annotate(
        "Pareto: best PPL (1.87)",
        xy=(winner_ppl_idx, ppl_vals[winner_ppl_idx]),
        xytext=(winner_ppl_idx + 0.55, ppl_vals[winner_ppl_idx] + 0.30),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.4),
        fontsize=10, color="darkgreen", fontweight="bold",
        ha="left", va="bottom",
    )

    # ---------------- Middle: peak DDR ----------------
    ddr_vals = df["peak_ddr_c"].values
    ax_ddr.bar(x, ddr_vals, width=bar_w, color=bar_colors, edgecolor="black", linewidth=0.7)
    for xi, v in zip(x, ddr_vals):
        ax_ddr.text(xi, v + 0.15, f"{v:.1f}C", ha="center", va="bottom", fontsize=10)
    ax_ddr.set_ylabel("Peak DDR temperature ( C )")
    ax_ddr.grid(axis="y", linestyle=":", alpha=0.5)
    ddr_lo = min(ddr_vals) - 1.5
    ddr_hi = max(ddr_vals) + 2.0
    ax_ddr.set_ylim(ddr_lo, ddr_hi)
    # 55 C throttle-risk reference line
    ax_ddr.axhline(55.0, color="red", linestyle="--", linewidth=0.9, alpha=0.6)
    ax_ddr.text(len(CELLS) - 0.5, 55.05, "DDR throttle-risk 55C",
                color="red", fontsize=8, ha="right", va="bottom", alpha=0.8)

    # Pareto annotation: thermal winner = v1_K512 (lowest peak DDR)
    winner_ddr_idx = CELLS.index("v1_K512")
    ax_ddr.annotate(
        "Pareto: coolest DDR (49.4C)",
        xy=(winner_ddr_idx, ddr_vals[winner_ddr_idx]),
        xytext=(winner_ddr_idx + 0.45,
                ddr_vals[winner_ddr_idx] - (ddr_hi - ddr_lo) * 0.20),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.4),
        fontsize=10, color="darkgreen", fontweight="bold",
        ha="left", va="top",
    )

    # ---------------- Bottom: mean decode tps ----------------
    tps_vals = df["tps_mean"].values
    tps_err = df["tps_std"].values
    ax_tps.bar(x, tps_vals, width=bar_w, color=bar_colors, edgecolor="black", linewidth=0.7,
               yerr=tps_err, capsize=4, error_kw=dict(ecolor="black", elinewidth=1.0))
    for xi, v in zip(x, tps_vals):
        ax_tps.text(xi, v + 0.10, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax_tps.set_ylabel("Mean decode tok/s\n(across iters)")
    ax_tps.grid(axis="y", linestyle=":", alpha=0.5)
    ax_tps.set_ylim(0, max(tps_vals) * 1.22)
    ax_tps.set_xticks(x)
    ax_tps.set_xticklabels(xlabels)

    # Pareto annotation: tps winner = v1_fa_K512
    winner_tps_idx = CELLS.index("v1_fa_K512")
    ax_tps.annotate(
        "Pareto: fastest decode (7.77 tok/s)",
        xy=(winner_tps_idx, tps_vals[winner_tps_idx]),
        xytext=(winner_tps_idx - 2.4, tps_vals[winner_tps_idx] + 0.7),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.4),
        fontsize=10, color="darkgreen", fontweight="bold",
        ha="left", va="bottom",
    )

    # Footer caption
    fig.text(
        0.5, 0.005,
        f"Source: {WAVE_DIR.name}  |  per-cell iters: " +
        ", ".join(f"{LABELS[c]}={int(df.loc[c,'n_iters'])}" for c in CELLS),
        ha="center", va="bottom", fontsize=8, color="dimgray",
    )

    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
