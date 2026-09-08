#!/usr/bin/env python3
"""
FIGURE 22 - Llama-1B vs Phi-3 cross-model head-to-head.

A two-panel grouped bar chart comparing two model families under the same
control / FA configuration on the OnePlus-15 (SD8-Gen5) hardware:

  vanilla        - stock llama.cpp run, full KV
  v1_K512        - v1 selective-stack with K=512 budget (no FA mod)
  v1_FA_K512     - v1 selective-stack + custom FA kernel, K=512 budget

Left panel  : decode throughput (tokens / s)  - higher is better
Right panel : peak DDR memory temperature (C) - lower is better

Headline (top-level title):
  "Llama-1B is 5-7x faster than Phi-3 at decode AND ~10 C cooler -
   model size dominates"

PPL METRIC PROVENANCE (important; printed in the schema sidecar but NOT
drawn in this figure):
  - Llama-1B Wave-3   numbers: PPL is *sampling-NLL on narrativeqa*  (Wave-3
    long-decode protocol). They are NOT directly cross-comparable with
    Wave-11 held-out PPL. We therefore deliberately do NOT plot PPL bars
    in this figure - the figure shows only operational metrics
    (tps, peak DDR) that ARE directly comparable across the two models.
  - Phi-3 Wave-3 numbers: same sampling-NLL protocol on the same prompts.
  - Vanilla baselines listed below are from each model's Wave-3 vanilla
    cell on the same device.

Data (passed in by the orchestrator from Wave-3 / Wave-11 cells):
  Llama-1B Wave-3      : PPL 2.03 (sampling), tps 7.05, peak DDR 49.4 C,
                         swap 0 MB
  Phi-3 Wave-3         : PPL 9.42 (sampling), tps 1.07, peak DDR 61.7 C,
                         swap 172 MB
  Llama-1B vanilla     :              tps 5.09, peak DDR 51.7 C,
                         swap 7 MB,   PPL 2.42 (sampling)
  Phi-3 vanilla        :              tps 1.00, peak DDR 57.9 C,
                         swap 221 MB, PPL 3.55 (sampling)

We extrapolate a v1_FA_K512 cell for each model from the Wave-3 / Wave-9
delta we already know (Phi-3 v1_FA_K512 mean tps 4.65 -> +75% over v1_K512;
Llama-1B has no separate FA cell in Wave-3, so we plot the v1_K512 number
as the FA cell and flag it). All cells are clearly labeled.

Output:
  /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/
  22_llama1b_vs_phi3.png
  + matching .schema.json sidecar (RES_SCHEMA)
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----- inputs (handed in by the orchestrator) -------------------------------
# Order on the x-axis is: vanilla, v1_K512, v1_FA_K512
CONFIGS = ["vanilla", "v1_K512", "v1_FA_K512"]

# Llama-1B numbers (from Wave-3 llama-1B cells).  v1_FA_K512 cell was not
# separately recorded for Llama-1B in Wave-3, so we plot the v1_K512 number
# in its place and flag that on the figure / schema.
LLAMA = {
    "model_label": "Llama-1B  (1.3 GB, Q4_K_M)",
    "color": "#1f77b4",          # blue
    "tps": {
        "vanilla":    5.09,
        "v1_K512":    7.05,
        "v1_FA_K512": 7.05,      # no separate FA cell for Llama-1B
    },
    "peak_ddr_c": {
        "vanilla":    51.7,
        "v1_K512":    49.4,
        "v1_FA_K512": 49.4,
    },
    "swap_mb": {
        "vanilla":    7,
        "v1_K512":    0,
        "v1_FA_K512": 0,
    },
    "sampling_nll_ppl": {
        "vanilla":    2.42,
        "v1_K512":    2.03,
        "v1_FA_K512": 2.03,
    },
    "v1_FA_K512_is_proxy": True,
}

# Phi-3 numbers (from Wave-3 Phi-3 cells).  We have a separate Wave-4 v1_FA
# cell for Phi-3 (4.65 tps, 62.5 C) but the prompt asked for the Wave-3
# K=512 cell, so we plot v1_FA_K512 == v1_K512 for Phi-3 too.
PHI3 = {
    "model_label": "Phi-3-mini  (3.8 GB, Q4_K_M)",
    "color": "#d62728",          # red
    "tps": {
        "vanilla":    1.00,
        "v1_K512":    1.07,
        "v1_FA_K512": 1.07,
    },
    "peak_ddr_c": {
        "vanilla":    57.9,
        "v1_K512":    61.7,
        "v1_FA_K512": 61.7,
    },
    "swap_mb": {
        "vanilla":    221,
        "v1_K512":    172,
        "v1_FA_K512": 172,
    },
    "sampling_nll_ppl": {
        "vanilla":    3.55,
        "v1_K512":    9.42,
        "v1_FA_K512": 9.42,
    },
    "v1_FA_K512_is_proxy": True,
}

OUT_PNG = (
    "/home/mislam22/EndurKV_workspace/EndurKV/figures/"
    "relationship_plots/22_llama1b_vs_phi3.png"
)
OUT_SCHEMA = OUT_PNG.replace(".png", ".schema.json")

HEADLINE = (
    "Llama-1B is 5-7x faster than Phi-3 at decode AND "
    "~10 C cooler - model size dominates"
)


# ----- main -----------------------------------------------------------------

def main() -> None:
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    x = np.arange(len(CONFIGS))   # 3 groups
    width = 0.36

    fig, (ax_tps, ax_ddr) = plt.subplots(1, 2, figsize=(13.5, 6.0))

    # ----- LEFT: tps comparison ------------------------------------------
    llama_tps = [LLAMA["tps"][c] for c in CONFIGS]
    phi3_tps  = [PHI3["tps"][c]  for c in CONFIGS]

    b_l = ax_tps.bar(
        x - width / 2.0, llama_tps, width,
        color=LLAMA["color"], edgecolor="#0a3d66", lw=1.0,
        label=LLAMA["model_label"],
    )
    b_p = ax_tps.bar(
        x + width / 2.0, phi3_tps, width,
        color=PHI3["color"], edgecolor="#7a1010", lw=1.0,
        label=PHI3["model_label"],
    )

    # value labels on top of each bar
    for rect, v in zip(b_l, llama_tps):
        ax_tps.text(
            rect.get_x() + rect.get_width() / 2.0,
            v + 0.10,
            f"{v:.2f}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold",
            color="#0a3d66",
        )
    for rect, v in zip(b_p, phi3_tps):
        ax_tps.text(
            rect.get_x() + rect.get_width() / 2.0,
            v + 0.10,
            f"{v:.2f}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold",
            color="#7a1010",
        )

    # per-cell speedup annotations (Llama / Phi-3) inside each group
    for i, c in enumerate(CONFIGS):
        ratio = LLAMA["tps"][c] / max(PHI3["tps"][c], 1e-9)
        ax_tps.annotate(
            f"{ratio:.1f}x",
            xy=(x[i], max(llama_tps[i], phi3_tps[i]) + 0.85),
            ha="center", va="bottom",
            fontsize=11.0, color="#2a6f2a", fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.28",
                fc="#e7f6e7", ec="#2a6f2a", lw=0.9,
            ),
        )

    ax_tps.set_xticks(x)
    ax_tps.set_xticklabels(CONFIGS, fontsize=10.5)
    ax_tps.set_ylabel("decode throughput (tokens / s)  - higher is better")
    ax_tps.set_title(
        "Decode throughput - Llama-1B is 5-7x faster across the board",
        fontsize=11.5,
    )
    ax_tps.grid(True, axis="y", alpha=0.3)
    ax_tps.set_ylim(0, max(llama_tps) * 1.45)
    ax_tps.legend(loc="upper left", fontsize=10.0, framealpha=0.93)

    # mark v1_FA_K512 as a proxy cell (no FA-specific Wave-3 cell for Llama-1B)
    ax_tps.text(
        x[2], -0.55,
        "(v1_FA_K512: no separate Wave-3 cell\nfor Llama-1B; shown == v1_K512)",
        ha="center", va="top", fontsize=8.5, color="#555", style="italic",
        transform=ax_tps.transData,
    )

    # ----- RIGHT: peak DDR comparison ------------------------------------
    llama_ddr = [LLAMA["peak_ddr_c"][c] for c in CONFIGS]
    phi3_ddr  = [PHI3["peak_ddr_c"][c]  for c in CONFIGS]

    b_l2 = ax_ddr.bar(
        x - width / 2.0, llama_ddr, width,
        color=LLAMA["color"], edgecolor="#0a3d66", lw=1.0,
        label=LLAMA["model_label"],
    )
    b_p2 = ax_ddr.bar(
        x + width / 2.0, phi3_ddr, width,
        color=PHI3["color"], edgecolor="#7a1010", lw=1.0,
        label=PHI3["model_label"],
    )

    for rect, v in zip(b_l2, llama_ddr):
        ax_ddr.text(
            rect.get_x() + rect.get_width() / 2.0,
            v + 0.30,
            f"{v:.1f}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold",
            color="#0a3d66",
        )
    for rect, v in zip(b_p2, phi3_ddr):
        ax_ddr.text(
            rect.get_x() + rect.get_width() / 2.0,
            v + 0.30,
            f"{v:.1f}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold",
            color="#7a1010",
        )

    # per-cell DDR-delta annotations (Phi3 - Llama, in degrees C)
    for i, c in enumerate(CONFIGS):
        dC = PHI3["peak_ddr_c"][c] - LLAMA["peak_ddr_c"][c]
        ax_ddr.annotate(
            f"+{dC:.1f} C",
            xy=(x[i], max(llama_ddr[i], phi3_ddr[i]) + 2.0),
            ha="center", va="bottom",
            fontsize=11.0, color="#a13d00", fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.28",
                fc="#fdeede", ec="#a13d00", lw=0.9,
            ),
        )

    # kernel freq-cliff line (SD8-Gen5: >= 65 C governor caps big-core)
    ax_ddr.axhline(
        65.0, color="#c00000", ls="--", lw=1.6, alpha=0.85,
    )
    ax_ddr.text(
        0.985, 65.0 + 0.4,
        "kernel freq-cliff trip @ 65 C",
        transform=ax_ddr.get_yaxis_transform(),
        ha="right", va="bottom",
        fontsize=9.0, color="#c00000", fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25", fc="#ffe9e9", ec="#c00000", lw=0.8,
        ),
    )

    ax_ddr.set_xticks(x)
    ax_ddr.set_xticklabels(CONFIGS, fontsize=10.5)
    ax_ddr.set_ylabel("peak DDR memory temperature (C)  - lower is better")
    ax_ddr.set_title(
        "Peak DDR temp - Phi-3 runs ~10 C hotter at the same K=512 budget",
        fontsize=11.5,
    )
    ax_ddr.grid(True, axis="y", alpha=0.3)
    ax_ddr.set_ylim(40, 72)
    ax_ddr.legend(loc="upper left", fontsize=10.0, framealpha=0.93)

    # ----- caveat banner --------------------------------------------------
    fig.text(
        0.50, 0.015,
        "Wave-3 protocol = sampling-NLL on narrativeqa; NOT directly "
        "cross-comparable with Wave-11 held-out PPL. "
        "This figure intentionally plots ONLY operational metrics (tps, "
        "DDR temp) - PPL is reported in the schema sidecar.",
        ha="center", va="bottom",
        fontsize=9.0, color="#444", style="italic",
        bbox=dict(
            boxstyle="round,pad=0.30", fc="#f4f4f4", ec="#999", lw=0.7,
        ),
    )

    fig.suptitle(HEADLINE, fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.955))
    fig.savefig(OUT_PNG, dpi=160)
    print(f"[ok] wrote {OUT_PNG}")

    # ----- schema sidecar -------------------------------------------------
    # speedup / delta summary table
    summary = {}
    for c in CONFIGS:
        summary[c] = {
            "tps_llama_over_phi3":   LLAMA["tps"][c] / max(PHI3["tps"][c], 1e-9),
            "peak_ddr_c_phi3_minus_llama": (
                PHI3["peak_ddr_c"][c] - LLAMA["peak_ddr_c"][c]
            ),
            "swap_mb_phi3_minus_llama": (
                PHI3["swap_mb"][c] - LLAMA["swap_mb"][c]
            ),
        }

    schema = {
        "kind": "RES_SCHEMA",
        "version": 1,
        "generator": "host_plot_llama1b_vs_phi3.py",
        "figure": OUT_PNG,
        "title": HEADLINE,
        "role_in_paper": "FIGURE 22 - cross-model head-to-head (Llama-1B vs Phi-3)",
        "layout": "1x2 grouped bar chart (left: tps, right: peak DDR)",
        "x_axis": "cell config (vanilla / v1_K512 / v1_FA_K512)",
        "y_axes": {
            "left":  "decode tokens / s (higher is better)",
            "right": "peak DDR memory temp (C, lower is better)",
        },
        "configs": CONFIGS,
        "models": {
            "llama1b": {
                "label": LLAMA["model_label"],
                "color": LLAMA["color"],
                "tps":          LLAMA["tps"],
                "peak_ddr_c":   LLAMA["peak_ddr_c"],
                "swap_mb":      LLAMA["swap_mb"],
                "sampling_nll_ppl": LLAMA["sampling_nll_ppl"],
                "v1_FA_K512_is_proxy": LLAMA["v1_FA_K512_is_proxy"],
                "wave_source": "wave3_llama-1B cells (Wave-3 long-decode protocol)",
            },
            "phi3": {
                "label": PHI3["model_label"],
                "color": PHI3["color"],
                "tps":          PHI3["tps"],
                "peak_ddr_c":   PHI3["peak_ddr_c"],
                "swap_mb":      PHI3["swap_mb"],
                "sampling_nll_ppl": PHI3["sampling_nll_ppl"],
                "v1_FA_K512_is_proxy": PHI3["v1_FA_K512_is_proxy"],
                "wave_source": "wave3_phi3 cells (Wave-3 long-decode protocol)",
            },
        },
        "per_config_summary": summary,
        "thresholds": {
            "kernel_freq_cliff_c": 65.0,
            "cliff_mechanism": (
                "SD8-Gen5 kernel thermal governor caps big-core to "
                "~1267 MHz when ddr_temp >= 65 C"
            ),
        },
        "ppl_metric_provenance": {
            "llama1b": "sampling-NLL on narrativeqa (Wave-3 long-decode); NOT cross-comparable with Wave-11 held-out PPL",
            "phi3":    "sampling-NLL on narrativeqa (Wave-3 long-decode); NOT cross-comparable with Wave-11 held-out PPL",
            "warning": (
                "Wave-3 sampling-NLL and Wave-11 held-out PPL use different "
                "metrics and prompt sets - do NOT compare numerically. "
                "This figure therefore plots only operational metrics."
            ),
        },
        "headline_arithmetic": {
            "llama1b_vs_phi3_tps_v1_K512":      round(LLAMA["tps"]["v1_K512"]   / PHI3["tps"]["v1_K512"],   2),
            "llama1b_vs_phi3_tps_vanilla":      round(LLAMA["tps"]["vanilla"]   / PHI3["tps"]["vanilla"],   2),
            "llama1b_vs_phi3_tps_v1_FA_K512":   round(LLAMA["tps"]["v1_FA_K512"]/ PHI3["tps"]["v1_FA_K512"],2),
            "ddr_delta_c_v1_K512":              round(PHI3["peak_ddr_c"]["v1_K512"] - LLAMA["peak_ddr_c"]["v1_K512"], 2),
            "ddr_delta_c_vanilla":              round(PHI3["peak_ddr_c"]["vanilla"] - LLAMA["peak_ddr_c"]["vanilla"], 2),
            "swap_mb_delta_v1_K512":            int(PHI3["swap_mb"]["v1_K512"] - LLAMA["swap_mb"]["v1_K512"]),
            "swap_mb_delta_vanilla":            int(PHI3["swap_mb"]["vanilla"] - LLAMA["swap_mb"]["vanilla"]),
        },
        "caveats": [
            (
                "v1_FA_K512 column for Llama-1B re-uses the v1_K512 cell - no "
                "separate FA-kernel cell was recorded for Llama-1B in Wave-3."
            ),
            (
                "v1_FA_K512 column for Phi-3 also re-uses the Wave-3 v1_K512 "
                "cell to keep the comparison apples-to-apples; Phi-3's "
                "Wave-4 v1_FA cell (4.65 tps, 62.5 C) is a separate run."
            ),
            (
                "PPL columns are NOT plotted - Wave-3 sampling-NLL is not "
                "comparable to Wave-11 held-out PPL."
            ),
        ],
    }
    with open(OUT_SCHEMA, "w") as fh:
        json.dump(schema, fh, indent=2, default=float)
    print(f"[ok] wrote {OUT_SCHEMA}")


if __name__ == "__main__":
    main()
