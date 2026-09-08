#!/usr/bin/env python3
# ============================================================================
# fig_backend_crossover.py -- the paper's thesis in one figure. (2026-08-16)
#
# REPLACES fig_main_results as the flagship results figure. That figure had four
# problems a reviewer would each catch:
#   1. STALE DATA. Its StreamingLLM bar (~6.6 tok/s CPU) is the handicapped
#      configuration -- FA-off, uncompacted, half its published budget. Faithfully
#      configured (start_size 4 + recent_size 2000, compacted, the way its own repo
#      runs it) StreamingLLM does 12.19 tok/s on the same prompts. The figure also
#      still carried a "muKV-swap" arm, the state-swap variant superseded by
#      in-place compaction.
#   2. RAINBOW PALETTE. Nine arbitrary hues, one per policy: color carried identity
#      only, and identity was already on the axis. Color now carries the MECHANISM
#      CLASS, which is the thing the paper argues about:
#         vermillion = muKV; blue = positional sequence-level (StreamingLLM);
#         gray = per-head evictors. Okabe-Ito values, CVD-safe.
#   3. NO STRUCTURE. Four disconnected panels; the FA-on/FA-off and seq/per-head
#      groupings that explain the numbers were invisible.
#   4. ONE BACKEND. It showed only CPU, silently implying the win generalizes. The
#      corrected finding -- the paper's actual thesis -- is that it does NOT:
#      eviction pays 3.2x on the CPU and ties on the GPU, and KV/W predicts which.
#      Two panels, same policies, same row order, is that argument made visual.
#
# DATA PROVENANCE (all cells from this workspace, none hand-typed from memory):
#   CPU panel: /tmp/lb_native, Llama-3.2-1B, LongBench prompts, every policy at its
#     own published budget, matched samples n=27, one campaign, one build. NO cool
#     gate (the campaign was built for F1) -- all policies interleaved under
#     identical conditions, so the ORDERING is trustworthy; absolute tok/s carries
#     thermal drift and the caption must say so.
#   GPU panel, solid bars: /tmp/sllm_faithful, pinned (taskset f0 nice -20),
#     cool-gated, n=3, SD <= 0.5%.
#   GPU panel, hatched bars: /tmp/phone_gpu_16k, the per-head evictors -- unpinned,
#     n=1, OLD device build, and their output tokens are invalid on this backend
#     (the cb_eval graph-split corruption, root-caused 2026-08-14). Timing only.
#     Drawn hatched and lighter so the protocol difference is visible in the mark
#     itself, not only in the caption.
# ============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MUKV = "#D55E00"   # vermillion  - muKV
SEQ  = "#0072B2"   # blue        - sequence-level positional (StreamingLLM)
PH   = "#8C8C8C"   # gray        - per-head evictors
VAN  = "#333333"   # near-black  - full-cache reference line

# (label, tok/s, cells retained, class-color, hatched=old-protocol)
CPU = [
    (r"$\mu$KV",        24.02, 1331, MUKV, False),
    ("StreamingLLM",    12.19, 2005, SEQ,  False),
    ("SnapKV",           9.24, 7605, PH,   False),
    ("Ada-KV",           9.21, 5109, PH,   False),
    ("H2O",              7.79, 7704, PH,   False),
    ("TOVA",             7.44, 5495, PH,   False),
]
CPU_VAN = 7.56          # 8541 cells
GPU = [
    (r"$\mu$KV",        29.05,  723, MUKV, False),
    ("StreamingLLM",    29.07, 2005, SEQ,  False),
    ("SnapKV",           4.76, 6592, PH,   True),
    ("Ada-KV",           2.86, 3519, PH,   True),
    ("H2O",              3.58, 6110, PH,   True),
    ("TOVA",             4.34, 4125, PH,   True),
]
GPU_VAN = 24.28         # 9741 cells

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)

for ax, data, van, title, xmax in (
        (axes[0], CPU, CPU_VAN,
         "Phone CPU — attention is compute-bound:\neviction pays", 30),
        (axes[1], GPU, GPU_VAN,
         "Phone GPU — weights dominate the read:\neviction ties", 34)):
    ys = range(len(data))[::-1]
    for y, (lab, v, cells, col, old) in zip(ys, data):
        ax.barh(y, v, height=0.62, color=col, alpha=0.45 if old else 1.0,
                hatch="///" if old else None, edgecolor=col, linewidth=0.8)
        ratio = v / van
        txt = "%.1f  (%.2f×)" % (v, ratio)
        if old:
            txt += r"$^\dagger$"
        ax.annotate(txt, (v, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8,
                    color="#555555" if old else "#111111")
        # retained cells: inside the bar when it is wide enough; on narrow bars
        # (the hatched GPU per-head arms) an in-bar label collided with the value
        # annotation, so it moves under the value at the bar end instead.
        if v > 0.22 * xmax:
            ax.annotate("%d cells" % cells, (0, y), xytext=(3, -1),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=6.5, color="white" if not old else "#666666")
        else:
            ax.annotate("%d cells" % cells, (v, y), xytext=(4, -9),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=6, color="#888888")
    ax.axvline(van, color=VAN, lw=1.1, ls="--", zorder=0)
    ax.set_ylim(-0.95, len(data) - 0.35)
    ax.text(van, -0.55, " full cache %.1f tok/s" % van,
            fontsize=7, color=VAN, ha="left", va="center")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([d[0] for d in data], fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("decode throughput (tok/s)", fontsize=9)
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

# legend: mechanism classes, not policies
h = [plt.Rectangle((0, 0), 1, 1, color=MUKV),
     plt.Rectangle((0, 0), 1, 1, color=SEQ),
     plt.Rectangle((0, 0), 1, 1, color=PH),
     plt.Rectangle((0, 0), 1, 1, facecolor="#CCCCCC", hatch="///", edgecolor=PH)]
fig.legend(h, [r"$\mu$KV (attention-driven, seq-level)",
               "positional, seq-level (own budget)",
               "per-head evictors",
               r"$^\dagger$old build, unpinned, timing-only"],
           fontsize=7.5, frameon=False, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Same policy, two backends: KV/W decides what eviction is worth "
             "(Llama-3.2-1B, ctx 16384, each policy at its own published budget)",
             fontsize=9.5, y=1.02)
fig.tight_layout(rect=[0, 0.05, 1, 1])
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_backend_crossover"
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf")
