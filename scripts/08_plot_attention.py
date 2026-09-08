#!/usr/bin/env python3
"""
Phase B' — load the entropy CSV and the attention sidecar, produce slide plots:
  figures/attn_demo_entropy.{pdf,png}   line plot of H_nats over decode steps
  figures/attn_demo_heatmap.{pdf,png}   per-decode-step attention heatmap
                                        (x=source token position, y=decode step,
                                         color=attention mass summed over layers and heads),
                                        H2O-style heavy-hitter visualization.

Usage:
    source .venv/bin/activate
    python3 scripts/08_plot_attention.py logs/attention/attn_demo
        ^ pass the prompt-id stem (without .csv / .attn.bin)

Files expected:
    <stem>.csv         entropy_probe-format CSV
    <stem>.attn.bin    binary sidecar written by attention_probe

Sidecar binary format (little-endian host order):
    magic [4] = b'ATTN'
    u32 n_steps, u32 n_layers, u32 n_head
    for each step:
        for each layer:
            u32 n_kv
            n_kv * float32 (per-source-token attention, head-averaged)
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_attention_bin(path: Path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"ATTN":
            raise ValueError(f"bad magic {magic!r} in {path}")
        n_steps, n_layers, n_head = struct.unpack("<III", f.read(12))
        # per_step_layer[step][layer] -> 1D float32 array of length n_kv
        per_step_layer: list[list[np.ndarray]] = []
        for s in range(n_steps):
            layers: list[np.ndarray] = []
            for l in range(n_layers):
                (n_kv,) = struct.unpack("<I", f.read(4))
                arr = np.frombuffer(f.read(n_kv * 4), dtype=np.float32).copy()
                layers.append(arr)
            per_step_layer.append(layers)
    return n_steps, n_layers, n_head, per_step_layer


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    stem = Path(sys.argv[1])
    csv_path = stem.with_suffix(".csv")
    bin_path = stem.with_suffix(".attn.bin")
    if not csv_path.exists():
        print(f"missing {csv_path}", file=sys.stderr); return 1
    if not bin_path.exists():
        print(f"missing {bin_path}", file=sys.stderr); return 1

    df = pd.read_csv(csv_path)
    n_steps, n_layers, n_head, attn = load_attention_bin(bin_path)
    print(f"csv rows: {len(df)}, attn n_steps: {n_steps}, n_layers: {n_layers}, n_head: {n_head}")
    if len(df) != n_steps:
        print(f"warning: csv ({len(df)}) and attn ({n_steps}) step counts differ; using min", file=sys.stderr)
    n_steps = min(len(df), n_steps)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    prompt_id = df["prompt_id"].iloc[0] if len(df) else stem.stem

    # ---------- entropy line plot --------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(df["step_index"][:n_steps], df["H_nats"][:n_steps],
            marker="o", linewidth=1.4, markersize=4)
    ax.set_xlabel("decode step")
    ax.set_ylabel("entropy H (nats)")
    ax.set_title(f"output entropy per decode step — {prompt_id}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prompt_id}_entropy.pdf")
    fig.savefig(out_dir / f"{prompt_id}_entropy.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out_dir / (prompt_id + '_entropy.pdf')}")

    # ---------- attention heatmap --------------------------------------------
    # For each step, sum attention over layers (head-averaged, layer-summed).
    # Pad to max n_kv so we get a rectangular [step, src] matrix.
    max_kv = max(int(layers[0].shape[0]) if layers and layers[0].size else 0
                 for layers in attn[:n_steps]) if n_steps else 0
    H = np.full((n_steps, max_kv), np.nan, dtype=np.float32)
    for s in range(n_steps):
        layers = attn[s]
        if not layers or layers[0].size == 0:
            continue
        n_kv = int(layers[0].shape[0])
        sum_layers = np.zeros(n_kv, dtype=np.float64)
        for arr in layers:
            if arr.size != n_kv:
                continue
            sum_layers += arr
        # report per-layer-mean for a probability-like scale in [0,1]
        sum_layers /= max(1, len(layers))
        H[s, :n_kv] = sum_layers

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(H, aspect="auto", cmap="magma", origin="lower",
                   vmin=0.0, vmax=np.nanpercentile(H, 99) if np.isfinite(H).any() else 1.0)
    ax.set_xlabel("source token position")
    ax.set_ylabel("decode step")
    ax.set_title(f"attention mass to source tokens (head- and layer-averaged) — {prompt_id}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("attention probability")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prompt_id}_attn_heatmap.pdf")
    fig.savefig(out_dir / f"{prompt_id}_attn_heatmap.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out_dir / (prompt_id + '_attn_heatmap.pdf')}")

    # ---------- accumulated heavy-hitter score (H2O-style) -------------------
    acc = np.nansum(H, axis=0)  # over decode steps
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(np.arange(acc.size), acc, color="#444", width=1.0)
    ax.set_xlabel("source token position")
    ax.set_ylabel("Σ attention mass over decode steps")
    ax.set_title(f"H2O-style heavy-hitter score — {prompt_id}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prompt_id}_heavy_hitters.pdf")
    fig.savefig(out_dir / f"{prompt_id}_heavy_hitters.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out_dir / (prompt_id + '_heavy_hitters.pdf')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
