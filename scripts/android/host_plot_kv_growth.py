#!/usr/bin/env python3
"""Plot KV cache memory growth vs context length, per model.

Generates two figures:
  kv_growth_linear.png  — linear x-axis (4K..32K mobile-realistic regime)
  kv_growth_log.png     — log x-axis (512..128K full range, cloud included)

Phone-RAM reference lines (8 GB and 12 GB) overlaid so the reader sees where
each model crosses the practical mobile envelope.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model_arch_specs import MODEL_SPECS, kv_bytes_per_token


# Plot styling: order models by per-token memory so the legend is informative
PLOT_ORDER = [
    "phi3-mini-4k",      # 384 KB/token — MHA, worst
    "llama3.1-8b",       # 128 KB/token
    "mistral-7b",        # 128 KB/token (same)
    "r1distill-llama-8b",# 128 KB/token (same)
    "llama3.2-3b",       # 112 KB/token
    "gemma2-2b",         # 104 KB/token (global layers)
    "qwen2-7b",          # 56 KB/token
    "llama3.2-1b",       # 32 KB/token — smallest
]
COLORS = {
    "phi3-mini-4k":       "#d62728",
    "llama3.1-8b":        "#1f77b4",
    "mistral-7b":         "#2ca02c",
    "r1distill-llama-8b": "#9467bd",
    "llama3.2-3b":        "#8c564b",
    "gemma2-2b":          "#e377c2",
    "qwen2-7b":           "#17becf",
    "llama3.2-1b":        "#7f7f7f",
}


def make_plot(ctx_lengths, log_x: bool, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    for key in PLOT_ORDER:
        if key not in MODEL_SPECS: continue
        bpt = kv_bytes_per_token(key)
        mem_mb = np.array(ctx_lengths) * bpt / (1024 * 1024)
        spec = MODEL_SPECS[key]
        label = f"{spec['full_name']} ({spec['attn_kind']}, {bpt//1024} KB/tok)"
        ax.plot(ctx_lengths, mem_mb, marker="o", markersize=5,
                linewidth=1.8, color=COLORS[key], label=label)
    # Reference lines: phone RAM envelopes
    # Typical flagship phone: ~12 GB RAM; per-app budget ~4-6 GB after OS+model weights
    for ref_gb, ref_label, style in [
        (4.0, "4 GB ceiling (per-app on 8 GB phone)", "--"),
        (8.0, "8 GB ceiling (per-app on 12 GB phone)", ":"),
        (12.0, "12 GB ceiling (high-end limit)", "-."),
    ]:
        ax.axhline(ref_gb * 1024, linestyle=style, color="black", alpha=0.45,
                   linewidth=1.0, label=ref_label)
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("KV cache memory (MB, fp16)")
    ax.set_title("KV cache growth vs. context length — Llama / Mistral / Phi / Qwen / Gemma / R1-Distill")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> int:
    out_dir = Path("/tmp/kv_growth_figs")
    out_dir.mkdir(exist_ok=True)
    # Linear: 1K..32K (mobile realistic to cloud-medium)
    ctx_lin = [512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768]
    make_plot(ctx_lin, log_x=False, out_path=out_dir / "kv_growth_linear.png")
    # Log: full range 512..128K
    ctx_log = [512, 1024, 2048, 4096, 8192, 13312, 16384, 32768, 65536, 131072]
    make_plot(ctx_log, log_x=True, out_path=out_dir / "kv_growth_log.png")

    # Also print the numeric table for reference
    print("\nKV cache memory (MB) at standard context lengths:")
    header = f"{'Model':22} | " + " | ".join(f"{n:>7}" for n in [1024, 4096, 8192, 13312, 32768, 131072])
    print(header)
    print("-" * len(header))
    for key in PLOT_ORDER:
        spec = MODEL_SPECS[key]
        bpt = kv_bytes_per_token(key)
        row = f"{key:22} | " + " | ".join(
            f"{bpt*n/1024/1024:>6.1f}M" if bpt*n/1024/1024 < 1024
            else f"{bpt*n/1024/1024/1024:>6.2f}G"
            for n in [1024, 4096, 8192, 13312, 32768, 131072])
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
