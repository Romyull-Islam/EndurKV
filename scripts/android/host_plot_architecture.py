#!/usr/bin/env python3
"""Architectural diagram of the perhead_v1 (EndurKV-Evict) eviction algorithm.

Pure architecture-only view — no baseline comparison. The companion script
host_plot_architecture_vs_tova.py renders the side-by-side perhead_v1 vs TOVA
comparison.

Layout (top-to-bottom):
  Row 1: Title (one line)
  Row 2: 5 large numbered circles forming the pipeline ① → ⑤
  Row 3: Stage ② — input per-head attention heatmap
  Row 4: Stage ③ — spread gate curve with the FORMULA displayed
         Stage ④ — per-head budget K_h bars
  Row 5: Stage ① — KV cache BEFORE eviction
         Stage ⑤ — KV cache AFTER perhead_v1 eviction
  Row 6: Variable-definition box (K, K_h, mult_h, max_a_h, all formulas)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap


RNG = np.random.default_rng(11)
N_HEAD = 8
N_KV = 32
K_NOMINAL = 10


def build_attention():
    a = np.zeros((N_HEAD, N_KV), dtype=np.float32)
    cfg = [
        (4,  0.82, 0.005, False),
        (19, 0.72, 0.008, False),
        (11, 0.48, 0.020, True),
        (27, 0.42, 0.022, True),
        (None, 0.22, 0.040, True),
        (None, 0.20, 0.040, True),
        (None, 0.18, 0.040, True),
        (None, 0.16, 0.040, True),
    ]
    for h, (sp, target_max, noise, plateau) in enumerate(cfg):
        a[h] = RNG.random(N_KV) * noise
        if plateau:
            mid = (N_KV // 2 + h * 3) % N_KV
            for k in range(N_KV):
                a[h, k] += 0.08 * np.exp(-0.5 * ((k - mid) / 7.5) ** 2)
        if sp is not None:
            B = a[h].sum()
            S = target_max * B / (1.0 - target_max)
            a[h, sp] = S
            if target_max < 0.55:
                a[h, (sp + 5) % N_KV] = S * 0.6
        a[h] /= a[h].sum()
    return a


def perhead_v1(a, K):
    n_head, n_kv = a.shape
    mask = np.zeros_like(a, dtype=bool)
    K_h, mult_h, max_a_h, norm_h = [], [], [], []
    for h in range(n_head):
        ma = float(a[h].max())
        nrm = max(0.0, min(1.0, (ma - 0.4) / 0.4))
        mult = 1.3 - 0.6 * nrm
        kh = max(1, min(n_kv, int(round(K * mult))))
        idx = np.argpartition(-a[h], kh - 1)[:kh]
        mask[h, idx] = True
        K_h.append(kh); mult_h.append(mult); max_a_h.append(ma); norm_h.append(nrm)
    return mask, np.array(K_h), np.array(mult_h), np.array(max_a_h), np.array(norm_h)


def main() -> int:
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures/architecture_fig")
    out_dir.mkdir(exist_ok=True, parents=True)

    a = build_attention()
    mask_ours, K_h, mult_h, max_a_h, norm_h = perhead_v1(a, K_NOMINAL)

    fig = plt.figure(figsize=(16, 13.0), dpi=130, facecolor="white")
    gs = fig.add_gridspec(
        nrows=5, ncols=4,
        height_ratios=[0.10, 0.55, 0.95, 1.00, 1.00],
        width_ratios=[1.0, 0.10, 1.0, 0.10],
        hspace=0.55, wspace=0.10,
        left=0.04, right=0.985, top=0.965, bottom=0.04,
    )

    # ─────────────── Row 1: title ───────────────
    ax_title = fig.add_subplot(gs[0, :]); ax_title.axis("off")
    ax_title.text(0.5, 0.5,
                  "perhead_v1 (EndurKV-Evict) — eviction algorithm architecture",
                  ha="center", va="center", fontsize=18, weight="bold",
                  color="#0f2e57")

    # ─────────────── Row 2: numbered pipeline ───────────────
    ax_pipe = fig.add_subplot(gs[1, :])
    ax_pipe.set_xlim(0, 1); ax_pipe.set_ylim(0, 1); ax_pipe.axis("off")
    stages = [
        ("①", "Full KV cache\nin RAM",          "#fef0e8", "#c14a1d"),
        ("②", "FA-off attention\ncompute",       "#e7f1ff", "#1f5fa8"),
        ("③", "Spread gate\nper head",           "#fff4d6", "#a8740c"),
        ("④", "Adaptive top-K\nselection",       "#e3f6e3", "#2c8b3b"),
        ("⑤", "Compressed KV\ncache",            "#fcdbdb", "#a31616"),
    ]
    cx_positions = np.linspace(0.085, 0.915, 5)
    r = 0.10  # circle radius in axes coords (relative to ax_pipe span)
    for i, ((tag, txt, fc, ec), cx) in enumerate(zip(stages, cx_positions)):
        circ = Circle((cx, 0.55), r, facecolor=fc, edgecolor=ec,
                      linewidth=2.4, transform=ax_pipe.transAxes, zorder=3)
        ax_pipe.add_patch(circ)
        ax_pipe.text(cx, 0.55, tag, ha="center", va="center",
                     fontsize=24, weight="bold", color=ec,
                     transform=ax_pipe.transAxes, zorder=4)
        ax_pipe.text(cx, 0.10, txt, ha="center", va="center",
                     fontsize=10, weight="bold", color=ec,
                     transform=ax_pipe.transAxes)
        if i < 4:
            arr = FancyArrowPatch(
                (cx + r + 0.015, 0.55),
                (cx_positions[i + 1] - r - 0.015, 0.55),
                arrowstyle="-|>", mutation_scale=22, linewidth=2.2,
                color="#444444", transform=ax_pipe.transAxes, zorder=2,
            )
            ax_pipe.add_patch(arr)

    # ─────────────── Row 3: input attention heatmap ───────────────
    ax_attn = fig.add_subplot(gs[2, :])
    cmap_attn = LinearSegmentedColormap.from_list(
        "attn", ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd", "#08306b"])
    im = ax_attn.imshow(a, aspect="auto", cmap=cmap_attn, vmin=0, vmax=a.max())
    ax_attn.set_yticks(range(N_HEAD))
    ax_attn.set_yticklabels([f"H{i}" for i in range(N_HEAD)], fontsize=10)
    ax_attn.set_xticks(range(0, N_KV, 4))
    ax_attn.set_xlabel("KV position k (token index in the cache)", fontsize=11)
    ax_attn.set_ylabel("Attention head h", fontsize=11)
    ax_attn.set_title(
        "②  Per-head attention a[h, k] computed at the current decode step "
        "(FA-off softmax(Q·Kᵀ/√d))",
        fontsize=12, weight="bold", color="#1f5fa8", pad=10)
    for h in range(N_HEAD):
        kind = "SHARP" if max_a_h[h] > 0.6 else ("medium" if max_a_h[h] > 0.3 else "diffuse")
        kind_col = "#a31616" if kind == "SHARP" else (
            "#a8740c" if kind == "medium" else "#2c8b3b")
        ax_attn.text(N_KV + 0.4, h, f"max_a[H{h}]={max_a_h[h]:.2f}  {kind}",
                     fontsize=9, va="center", color=kind_col, weight="bold")
    cbar = fig.colorbar(im, ax=ax_attn, fraction=0.025, pad=0.10)
    cbar.set_label("attention weight a[h, k]", fontsize=9)

    # ─────────────── Row 4 left: spread gate curve with formula ───────────────
    ax_gate = fig.add_subplot(gs[3, 0])
    max_grid = np.linspace(0.0, 1.0, 200)
    norm_grid = np.clip((max_grid - 0.4) / 0.4, 0, 1)
    mult_grid = 1.3 - 0.6 * norm_grid
    ax_gate.plot(max_grid, mult_grid, color="#a8740c", linewidth=2.8, zorder=2)
    ax_gate.scatter(max_a_h, mult_h, s=120, zorder=5,
                    edgecolor="white", linewidth=1.8,
                    c=["#a31616" if mu < 0.95 else "#2c8b3b" if mu > 1.05 else "#7f7f7f"
                       for mu in mult_h])
    for h in range(N_HEAD):
        ax_gate.annotate(f"H{h}", (max_a_h[h], mult_h[h]),
                         textcoords="offset points", xytext=(8, 8),
                         fontsize=9, weight="bold")
    ax_gate.axhline(1.0, color="#999", linewidth=0.8, linestyle="--")
    ax_gate.axvspan(0.0, 0.4, alpha=0.10, color="#2c8b3b")
    ax_gate.axvspan(0.8, 1.0, alpha=0.10, color="#a31616")
    ax_gate.text(0.20, 1.27, "diffuse heads →\nmore cache (mult > 1)",
                 ha="center", fontsize=9.5, color="#2c8b3b", weight="bold")
    ax_gate.text(0.90, 0.78, "sharp heads →\nless cache (mult < 1)",
                 ha="center", fontsize=9.5, color="#a31616", weight="bold")
    ax_gate.set_xlabel("max attention  max_a[h] = max_k a[h, k]", fontsize=11)
    ax_gate.set_ylabel("budget multiplier  mult_h", fontsize=11)
    ax_gate.set_title(
        "③  Spread gate: mult_h = 1.3 − 0.6 · clip((max_a[h] − 0.4)/0.4, 0, 1)",
        fontsize=12, weight="bold", color="#a8740c", pad=10)
    ax_gate.set_xlim(-0.02, 1.02); ax_gate.set_ylim(0.55, 1.45)
    ax_gate.grid(alpha=0.25)

    # ─────────────── Row 4 right: per-head budget K_h ───────────────
    ax_kh = fig.add_subplot(gs[3, 2])
    xs = np.arange(N_HEAD)
    bar_colors = ["#a31616" if mult_h[h] < 0.95 else
                  "#2c8b3b" if mult_h[h] > 1.05 else "#7f7f7f"
                  for h in range(N_HEAD)]
    ax_kh.bar(xs, K_h, width=0.7, color=bar_colors, edgecolor="black")
    for h in range(N_HEAD):
        ax_kh.text(h, K_h[h] + 0.35, f"K_h={K_h[h]}",
                   ha="center", fontsize=9, color=bar_colors[h], weight="bold")
        ax_kh.text(h, -1.2,
                   f"mult={mult_h[h]:.2f}",
                   ha="center", fontsize=8, color=bar_colors[h])
    ax_kh.axhline(K_NOMINAL, color="#000", linewidth=1.2, linestyle=":",
                  label=f"K_nominal = {K_NOMINAL} (user-set budget)")
    ax_kh.set_xticks(xs)
    ax_kh.set_xticklabels([f"H{i}" for i in range(N_HEAD)], fontsize=10)
    ax_kh.set_ylabel("Positions kept per head", fontsize=11)
    ax_kh.set_title(
        f"④  Per-head budget K_h = round(K_nominal · mult_h)  "
        f"|  total kept = {int(K_h.sum())}",
        fontsize=12, weight="bold", color="#2c8b3b", pad=10)
    ax_kh.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax_kh.grid(alpha=0.25, axis="y")
    ax_kh.set_ylim(-2.5, max(K_h.max() + 4, K_NOMINAL + 4))

    # ─────────────── Row 5: memory layouts BEFORE / AFTER ───────────────
    # BEFORE eviction
    ax_before = fig.add_subplot(gs[4, 0])
    for h in range(N_HEAD):
        for k in range(N_KV):
            color = plt.cm.Blues(0.20 + 0.75 * a[h, k] / a[h].max())
            ax_before.add_patch(Rectangle((k - 0.5, h - 0.5), 1, 1,
                                          facecolor=color, edgecolor="white",
                                          linewidth=0.3))
    ax_before.set_yticks(range(N_HEAD))
    ax_before.set_yticklabels([f"H{i}" for i in range(N_HEAD)], fontsize=10)
    ax_before.set_xticks(range(0, N_KV, 4))
    ax_before.set_xlabel("KV position", fontsize=11)
    ax_before.set_title(
        f"①  KV cache BEFORE eviction\n"
        f"all {N_HEAD*N_KV} (head, position) cells stored",
        fontsize=11.5, weight="bold", color="#c14a1d", pad=8)
    ax_before.set_xlim(-0.5, N_KV - 0.5)
    ax_before.set_ylim(N_HEAD - 0.5, -0.5)

    # AFTER eviction
    ax_after = fig.add_subplot(gs[4, 2])
    for h in range(N_HEAD):
        for k in range(N_KV):
            if mask_ours[h, k]:
                color = plt.cm.Reds(0.30 + 0.65 * a[h, k] / a[h].max())
                ax_after.add_patch(Rectangle((k - 0.5, h - 0.5), 1, 1,
                                             facecolor=color, edgecolor="white",
                                             linewidth=0.3))
            else:
                ax_after.add_patch(Rectangle((k - 0.5, h - 0.5), 1, 1,
                                             facecolor="#f3f3f3", edgecolor="white",
                                             linewidth=0.3, hatch="///", alpha=0.55))
    ax_after.set_yticks(range(N_HEAD))
    ax_after.set_yticklabels(
        [f"H{i} (K_h={K_h[i]})" for i in range(N_HEAD)], fontsize=9)
    ax_after.set_xticks(range(0, N_KV, 4))
    ax_after.set_xlabel("KV position", fontsize=11)
    kept = int(mask_ours.sum()); total = mask_ours.size
    ax_after.set_title(
        f"⑤  KV cache AFTER perhead_v1 eviction\n"
        f"{kept}/{total} cells kept ({100*kept/total:.0f}% cache, adaptive per head)",
        fontsize=11.5, weight="bold", color="#a31616", pad=8)
    ax_after.set_xlim(-0.5, N_KV - 0.5)
    ax_after.set_ylim(N_HEAD - 0.5, -0.5)

    out_path = out_dir / "perhead_v1_architecture.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
