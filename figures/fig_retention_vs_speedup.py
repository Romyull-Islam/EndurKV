#!/usr/bin/env python3
# ============================================================================
# fig_retention_vs_speedup.py -- retention does not predict speed. (REWRITTEN 2026-08-16)
#
# WHY THE REWRITE. The previous version's central annotation was "StreamingLLM keeps
# 8.0% of cells and runs 0.23x; muKV keeps 7.4% and runs 1.23x -- same retention,
# 5.3x apart". Both halves of that comparison came from a StreamingLLM we had
# handicapped three ways at once: FA forced off, cache never compacted, and K=1024
# instead of its published start_size 4 + recent_size 2000 = 2004. Its own
# implementation (mit-han-lab/streaming-llm, kv_cache.py) keeps FA usable and
# torch.cat-compacts on every eviction. Measured faithfully (pinned, cool-gated,
# n=3): 29.07 tok/s = 1.20x, at 20.6% retention. The annotation was measuring our
# own harness, and it is gone.
#
# WHAT THE FIGURE NOW SHOWS -- the claim that survived: retention still does not
# predict speed, but the mechanism split is what does. Sequence-level policies that
# stay on the FlashAttention path (muKV, faithful StreamingLLM) sit above 1x; the
# per-head evictors sit at 0.12-0.20x at ANY retention. And the two fast policies
# separate on a different axis entirely: muKV reaches the same 1.20x at 2.8x fewer
# cells and 14/14 NIAH against StreamingLLM's 7/14 -- stated in the caption, not as
# an in-plot arrow, because speed-vs-retention is no longer where they differ.
#
# ONE PANEL, NOT TWO. The old Phi-3 panel's StreamingLLM point was the same
# handicapped configuration; a faithful Phi-3 StreamingLLM has not been measured
# yet, so that panel cannot be drawn honestly and is dropped until it is.
#
# POINT PROVENANCE:
#   filled  : pinned, cool-gated, n=3 (/tmp/sllm_faithful), SD <= 0.5%
#   hollow +: per-head evictors from the old build, unpinned, n=1
#     (/tmp/phone_gpu_16k); additionally their OUTPUT is numerically invalid on
#     this backend (cb_eval graph-split corruption, root-caused 2026-08-14), so
#     these points are timing-only in every sense. Hollow + dagger says so in the
#     mark itself.
# ============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MUKV, SEQ, PH, VAN = "#D55E00", "#0072B2", "#8C8C8C", "#333333"

# (label, % cells kept, decode x vs vanilla, color, old-protocol?)
PTS = [
    ("vanilla",                 100.0, 1.00, VAN,  False),
    (r"$\mu$KV",                  7.4, 1.20, MUKV, False),
    ("StreamingLLM\n(own budget)", 20.6, 1.20, SEQ, False),
    ("SnapKV",                   67.7, 0.20, PH,   True),
    ("Ada-KV",                   36.1, 0.12, PH,   True),
    ("TOVA",                     42.3, 0.18, PH,   True),
    ("H2O",                      62.7, 0.15, PH,   True),
]

fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.axhline(1.0, color="#444444", lw=1.0, ls="--", zorder=1)
ax.text(0.03, 1.04, "break-even: evicting is no faster than not evicting",
        transform=ax.get_yaxis_transform(), ha="left", va="bottom",
        fontsize=7, color="#444444")

for lab, x, y, c, old in PTS:
    ax.scatter(x, y, s=80, facecolor="none" if old else c, edgecolor=c,
               linewidth=1.6, zorder=3)
    lab2 = lab + (r"$^\dagger$" if old else "")
    # vanilla sits exactly on the break-even line; its label goes BELOW the
    # point so it cannot collide with the line's caption text.
    below = (lab == "vanilla")
    dy = 0.88 if below else (1.13 if y < 1 else 1.07)
    ax.annotate(lab2, (x, y * dy), fontsize=7.5, color=c, ha="center",
                va="top" if below else "baseline", zorder=4)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(5, 160); ax.set_ylim(0.09, 2.2)
ax.set_xticks([5, 10, 25, 50, 100])
ax.set_xticklabels(["5%", "10%", "25%", "50%", "100%"])
ax.set_yticks([0.1, 0.25, 0.5, 1, 2])
ax.set_yticklabels(["0.1×", "0.25×", "0.5×", "1×", "2×"])
ax.set_xlabel("KV cache retained (% of prompt)")
ax.set_ylabel("decode speedup vs full cache")
ax.set_title("Phone GPU (Adreno 840), Llama-3.2-1B, ctx 16384:\n"
             "retention does not predict speed — the attention-access mechanism does",
             fontsize=9, loc="left")
ax.grid(alpha=0.25, lw=0.5, which="major")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

h = [plt.Line2D([], [], marker="o", ls="", mfc=MUKV, mec=MUKV,
                label="seq-level, FA on (pinned, n=3)"),
     plt.Line2D([], [], marker="o", ls="", mfc="none", mec=PH,
                label=r"per-head, FA off$^\dagger$ (old build, n=1, timing-only)")]
ax.legend(handles=h, fontsize=7, frameon=False, loc="center right")
fig.tight_layout()
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_retention_vs_speedup"
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf")
