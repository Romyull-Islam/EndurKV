#!/usr/bin/env python3
"""Comparison figure of KV cache management methods, including mobile-inference ones.

Panel A: properties matrix (method x design dimension). Panel B: measured critical KV
footprint at 95% mass retention on the Llama-8B smoke prompt, for methods we implemented."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# KV-cache methods by category. signal is the eviction signal, mobile_focus means the paper
# targets mobile or edge, our_impl/our_data mean implemented in our simulator and run on our captures.
METHODS = [
    # Per-token KV eviction (direct comparison set)
    {"name": "perhead_v1 (ours)", "category": "Per-token eviction",
     "signal": "attention+spread", "mobile_focus": True,
     "our_impl": True, "our_data": True,
     "year": 2026, "venue": "EndurKV", "paper_ref": "this work"},
    {"name": "TOVA", "category": "Per-token eviction",
     "signal": "attention", "mobile_focus": False,
     "our_impl": True, "our_data": True,
     "year": 2024, "venue": "ACL", "paper_ref": "Oren+2024"},
    {"name": "SnapKV", "category": "Per-token eviction",
     "signal": "attention (observation window)", "mobile_focus": False,
     "our_impl": True, "our_data": True,
     "year": 2024, "venue": "NeurIPS", "paper_ref": "Li+2024"},
    {"name": "PyramidKV", "category": "Per-token eviction",
     "signal": "attention (layer-tapered budget)", "mobile_focus": False,
     "our_impl": True, "our_data": True,
     "year": 2024, "venue": "arXiv", "paper_ref": "Cai+2024"},
    {"name": "H2O", "category": "Per-token eviction",
     "signal": "attention (cumulative)", "mobile_focus": False,
     "our_impl": True, "our_data": True,
     "year": 2023, "venue": "NeurIPS", "paper_ref": "Zhang+2023"},
    {"name": "StreamingLLM", "category": "Per-token eviction",
     "signal": "position (sliding window+sinks)", "mobile_focus": False,
     "our_impl": True, "our_data": True,
     "year": 2024, "venue": "ICLR", "paper_ref": "Xiao+2024"},
    {"name": "KVzip (decode-approx)", "category": "Per-token eviction",
     "signal": "attention (cumulative-max)", "mobile_focus": False,
     "our_impl": True, "our_data": True,
     "year": 2025, "venue": "NeurIPS", "paper_ref": "Kim+2025"},
    {"name": "R-KV", "category": "Per-token eviction",
     "signal": "attention + K-redundancy", "mobile_focus": False,
     "our_impl": True, "our_data": True,
     "year": 2025, "venue": "NeurIPS", "paper_ref": "Cai+2025"},
    {"name": "KeyDiff", "category": "Per-token eviction",
     "signal": "K-vector cosine", "mobile_focus": False,
     "our_impl": True, "our_data": True,
     "year": 2025, "venue": "arXiv", "paper_ref": "Park+2025"},
    {"name": "LaProx", "category": "Per-token eviction",
     "signal": "attention × ||W_o·V||", "mobile_focus": False,
     "our_impl": True, "our_data": True,
     "year": 2026, "venue": "arXiv", "paper_ref": "Mai+2026"},
    # Trained / structural eviction
    {"name": "DuoAttention", "category": "Trained head classification",
     "signal": "trained (retrieval vs streaming heads)", "mobile_focus": False,
     "our_impl": True, "our_data": True,
     "year": 2025, "venue": "ICLR", "paper_ref": "Xiao+2025"},
    {"name": "PruLong", "category": "Trained mask",
     "signal": "trained (end-to-end Bernoulli mask)", "mobile_focus": False,
     "our_impl": False, "our_data": False,
     "year": 2025, "venue": "arXiv", "paper_ref": "Bhaskar+2025"},
    # Mobile system level, a different problem class, not directly comparable
    {"name": "KVSwap (disk offload)", "category": "Mobile system",
     "signal": "compressed K + disk reload", "mobile_focus": True,
     "our_impl": False, "our_data": False,
     "year": 2025, "venue": "arXiv", "paper_ref": "Zhang+2025"},
    {"name": "MobiLoRA", "category": "Mobile system",
     "signal": "LoRA-aware delta KV + app context", "mobile_focus": True,
     "our_impl": False, "our_data": False,
     "year": 2025, "venue": "ACL", "paper_ref": "Li+2025"},
    {"name": "PerCache (mobile RAG)", "category": "Mobile system",
     "signal": "hierarchical QA + QKV cache reuse", "mobile_focus": True,
     "our_impl": False, "our_data": False,
     "year": 2025, "venue": "arXiv", "paper_ref": "Liu+2025"},
]


# Measured critical KV footprint at 95% mass retention, Llama-8B smoke prompt.
MEASURED_CRITICAL_K = {
    "perhead_v1 (ours)": 61.5,
    "TOVA": 64.0,
    "KVzip (decode-approx)": 64.0,
    "LaProx": 128.0,
    "R-KV": 256.0,
    "KeyDiff": np.nan,   # never reaches 95%
}


def make_properties_matrix(out_path: Path):
    """Panel A: properties grid, method x design dimension, with a column saying
    why the non-eviction methods are not simulated."""
    fig, ax = plt.subplots(figsize=(15.5, len(METHODS) * 0.50 + 1.8), dpi=130)

    # Comparability reason per method (None = directly comparable)
    not_comparable_reason = {
        "PruLong": "requires end-to-end training of mask params (~12 GPU-hrs)",
        "KVSwap (disk offload)":
            "system-level: stores cache on NVMe/eMMC disk, not eviction",
        "MobiLoRA":
            "LoRA-adapter delta-KV reuse, not single-prompt eviction",
        "PerCache (mobile RAG)":
            "cross-query RAG cache hit rate, not within-prompt eviction",
    }

    rows = []
    for m in METHODS:
        comp_reason = not_comparable_reason.get(m["name"])
        if comp_reason is None:
            comp_text = "Yes — same problem class"
        else:
            comp_text = f"No — {comp_reason}"
        rows.append([
            m["name"],
            m["category"],
            m["signal"],
            f"{m['venue']} {m['year']}",
            "✓" if m["mobile_focus"] else "—",
            comp_text,
            "✓" if m["our_impl"] else "—",
            "✓" if m["our_data"] else "—",
        ])
    headers = ["Method", "Category", "Eviction signal", "Venue/year",
               "Mobile-focused", "Comparable to perhead_v1?",
               "Impl. in our sim", "Measured on our data"]
    cell_colors = []
    for m in METHODS:
        comp_reason = not_comparable_reason.get(m["name"])
        is_comparable = comp_reason is None
        row_color = []
        # Method column: ours in pale red
        row_color.append("#fde0e0" if "(ours)" in m["name"] else "#f5f5f5")
        # Category color by group
        cat_colors = {
            "Per-token eviction": "#e8f4ff",
            "Trained head classification": "#fff4e0",
            "Trained mask": "#fff4e0",
            "Mobile system": "#ffece5",
        }
        row_color.append(cat_colors.get(m["category"], "#f5f5f5"))
        row_color.append("white")
        row_color.append("white")
        row_color.append("#d8f0d8" if m["mobile_focus"] else "white")
        # Comparability column: pale green if yes, pale orange if no
        row_color.append("#d8f0d8" if is_comparable else "#ffe5d2")
        # Impl/Data: red only for comparable methods we have not run, gray otherwise.
        if is_comparable:
            row_color.append("#d8f0d8" if m["our_impl"] else "#fcd8d8")
            row_color.append("#d8f0d8" if m["our_data"] else "#fcd8d8")
        else:
            row_color.append("#eaeaea")  # neutral gray, not red
            row_color.append("#eaeaea")
        cell_colors.append(row_color)

    ax.set_axis_off()
    tbl = ax.table(cellText=rows, colLabels=headers,
                   cellColours=cell_colors,
                   colColours=["#bcbcbc"] * len(headers),
                   loc="center", cellLoc="left", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)
    for i in range(len(METHODS) + 1):
        tbl[i, 0].set_text_props(weight="bold")
    ax.set_title(
        "Top-15 KV cache management methods — only the first 11 are "
        "directly comparable to perhead_v1\n"
        "(bottom 4 solve a different problem class — see 'Comparable to perhead_v1?' column)",
        pad=14, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def make_performance_bar(out_path: Path):
    """Panel B: critical-KV-footprint bar chart for measured methods."""
    measured = [(name, k) for name, k in MEASURED_CRITICAL_K.items()
                if not np.isnan(k)]
    measured.sort(key=lambda x: x[1])
    measured.append(("KeyDiff (>full cache)", 800.0))  # render as out-of-axis red bar

    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    names = [m[0] for m in measured]
    vals = [m[1] for m in measured]
    colors = ["#d62728" if "(ours)" in n else
              "#888888" if "KeyDiff" in n else "#1f77b4"
              for n in names]
    bars = ax.barh(names, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Critical KV cache budget (positions per head) "
                  "to retain ≥95% attention mass — lower is better")
    ax.set_title("Llama-3.1-8B smoke (n_kv=512) — critical KV footprint @ F=95%",
                 pad=10, fontsize=11, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    # Annotate each bar
    for bar, v in zip(bars, vals):
        ax.text(v + 5, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}", va="center", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> int:
    out_dir = Path("/tmp/kv_method_comparison")
    out_dir.mkdir(exist_ok=True)
    make_properties_matrix(out_dir / "properties_matrix.png")
    make_performance_bar(out_dir / "critical_footprint_bar.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
