#!/usr/bin/env python3
# ============================================================================
# fig_mechanism_v2.py -- what happens to the CACHE under each policy class.
# (2026-08-16, replaces fig_mechanism)
#
# WHY THE REDRAW. The old fig_mechanism was a 12-box portrait flowchart: the
# reader had to execute it like code, and its one decisive image -- the same-
# positions-dropped vs union-cannot-compact contrast -- sat at the bottom at
# small scale under a decorative heatmap. The canon mechanism figures this paper
# will be judged against (StreamingLLM Fig.1, PagedAttention, TinyMem at
# HotMobile'25) all do the opposite: they draw the DATA STRUCTURE itself, side
# by side across designs, with a verdict per design. This figure is that: one
# cell array, three policy classes, and what each one leaves the engine to work
# with. Every verdict line carries a MEASURED number from this paper, not a
# qualitative claim.
#
# THE THREE PANELS ARE THE PAPER'S ARGUMENT:
#   (a) per-head evictors: each head keeps its own top-K, the shared cell array
#       frees a cell only if EVERY head dropped it, the union leaves holes ->
#       cannot compact. Drawn with 4 head masks whose union fills ~2/3 of the
#       array -- the real measured figure (SnapKV retains 67.7%).
#   (b) StreamingLLM: position-blind sinks+recent window -- compacts fine, but
#       the needle in the discarded middle is unrecoverable (7/14 NIAH).
#   (c) muKV: attention-scored, sequence-level -- ONE keep-set shared by all
#       heads and layers, so eviction actually frees cells, the needle is kept
#       (14/14), and the survivors slide into a dense prefix.
# ============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

MUKV, SEQ, PH = "#D55E00", "#0072B2", "#8C8C8C"
GOOD, BAD, WARN = "#009E73", "#C1272D", "#B8860B"
EMPTY_EC = "#BBBBBB"

N = 24            # cells drawn per array
STAR = 7          # needle position (~30% depth -- the depth StreamingLLM loses)

rng = np.random.default_rng(11)

def draw_array(ax, x0, y0, keep, w=0.30, h=0.42, color=SEQ, star=None,
               star_kept=None, gap=0.06):
    """One cell array: filled = kept, outline = dropped. Returns x extent."""
    for i in range(len(keep)):
        x = x0 + i * (w + gap)
        if keep[i]:
            ax.add_patch(Rectangle((x, y0), w, h, facecolor=color,
                                   edgecolor=color, lw=0.6))
        else:
            ax.add_patch(Rectangle((x, y0), w, h, facecolor="white",
                                   edgecolor=EMPTY_EC, lw=0.6))
    if star is not None:
        xs = x0 + star * (w + gap) + w / 2
        kept = keep[star] if star_kept is None else star_kept
        ax.text(xs, y0 + h / 2, "★", ha="center", va="center", fontsize=9,
                color="white" if kept else BAD, zorder=5)
        if not kept:
            ax.plot([xs - 0.16, xs + 0.16], [y0 - 0.10, y0 + h + 0.10],
                    color=BAD, lw=1.4, zorder=6)
    return x0 + len(keep) * (w + gap) - gap

fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
for ax in axes:
    ax.set_xlim(0, 9.2); ax.set_ylim(0, 7.6)
    ax.axis("off")

# ── (a) per-head evictors ──────────────────────────────────────────────────
a = axes[0]
a.text(0.05, 7.25, "(a)  per-head evictors", fontsize=10, fontweight="bold")
a.text(0.05, 6.75, "SnapKV · Ada-KV · H2O · TOVA", fontsize=8, color="#555555")
# four heads, each keeps its own scattered K=6
union = np.zeros(N, dtype=bool)
for hidx in range(4):
    kept_idx = rng.choice(N, size=6, replace=False)
    keep = np.zeros(N, dtype=bool); keep[kept_idx] = True
    union |= keep
    y = 5.9 - hidx * 0.62
    a.text(0.05, y + 0.18, "head %d" % (hidx + 1), fontsize=6.5,
           color="#777777", ha="left")
    draw_array(a, 1.05, y, keep, w=0.26, h=0.38, color=PH, gap=0.052)
a.annotate("", xy=(4.6, 2.55), xytext=(4.6, 3.35),
           arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.2))
a.text(4.85, 2.95, "union: a cell frees only if\nEVERY head dropped it",
       fontsize=7, color="#444444", va="center")
draw_array(a, 1.05, 1.9, union, w=0.26, h=0.42, color=PH, gap=0.052)
a.text(1.05, 1.45, "live = %d/%d cells — holes everywhere" % (union.sum(), N),
       fontsize=7.5, color="#555555")
a.text(0.05, 0.55, "✗  cannot compact — SnapKV retains 67.7% (GPU;\n"
       "    91–100% at 32 heads) · FA-off decode 0.1–0.2×",
       fontsize=8, color=BAD)

# ── (b) StreamingLLM ───────────────────────────────────────────────────────
b = axes[1]
b.text(0.05, 7.25, "(b)  position-blind, seq-level", fontsize=10, fontweight="bold")
b.text(0.05, 6.75, "StreamingLLM (its own budget)", fontsize=8, color="#555555")
keep_sl = np.zeros(N, dtype=bool)
keep_sl[:2] = True          # sinks
keep_sl[-8:] = True         # recent window
b.text(0.05, 5.55, "keep by position only:", fontsize=7, color="#777777")
draw_array(b, 0.35, 4.9, keep_sl, color=SEQ, star=STAR, star_kept=False)
b.text(0.65, 4.45, "sinks", fontsize=6.5, color=SEQ)
b.text(7.4, 4.45, "recent window", fontsize=6.5, color=SEQ, ha="center")
b.text(3.6, 3.95, "middle deleted unseen", fontsize=6.5, color=BAD, ha="center")
b.annotate("", xy=(2.6, 2.85), xytext=(3.6, 3.95),
           arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.2))
b.text(3.75, 3.35, "compacts\n(contiguous by design)", fontsize=7, color="#444444")
draw_array(b, 0.35, 2.2, np.ones(10, dtype=bool), color=SEQ)
b.text(0.35, 1.75, "dense — but the needle is gone", fontsize=7.5, color="#555555")
b.text(0.05, 0.55, "△  same speed as $\\mu$KV (1.20×), but blind:\n"
       "    needle at ≤67% depth lost — NIAH 7/14", fontsize=8, color=WARN)

# ── (c) muKV ───────────────────────────────────────────────────────────────
c = axes[2]
c.text(0.05, 7.25, r"(c)  attention-scored, seq-level", fontsize=10, fontweight="bold")
c.text(0.05, 6.75, r"$\mu$KV — one keep-set for all heads & layers", fontsize=8,
       color="#555555")
# attention score sparkline above the array
score = rng.gamma(1.6, 1.0, N); score[STAR] = score.max() * 1.25
score[:2] += score.max() * 0.5
keep_mu = np.zeros(N, dtype=bool)
keep_mu[np.argsort(score)[-10:]] = True
xw, gap = 0.30, 0.06
for i in range(N):
    x = 0.35 + i * (xw + gap)
    c.add_patch(Rectangle((x, 5.05), xw, 0.85 * score[i] / score.max(),
                          facecolor=MUKV, alpha=0.45, edgecolor="none"))
c.text(0.05, 6.15, "prefill attention (kq_evict side node, ~+1% prefill):",
       fontsize=7, color="#777777")
draw_array(c, 0.35, 4.5, keep_mu, color=MUKV, star=STAR, star_kept=True)
c.annotate("", xy=(2.6, 2.85), xytext=(3.6, 3.65),
           arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.2))
c.text(3.75, 3.2, "in-place slide\n(peak memory never rises)",
       fontsize=7, color="#444444")
draw_array(c, 0.35, 2.2, np.ones(10, dtype=bool), color=MUKV, star=4,
           star_kept=True)
c.text(0.35, 1.75, "K dense cells — needle kept", fontsize=7.5, color="#555555")
c.text(0.05, 0.55, "✓  compacts to K=1024 → 723 live cells (7.4%)\n"
       "    NIAH 14/14 · decode 1.20× GPU / 4.80× CPU", fontsize=8, color=GOOD)

fig.suptitle("One shared cell array, three eviction classes: only a sequence-level, "
             "attention-scored keep-set both compacts and keeps the needle",
             fontsize=10, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_mechanism_v2"
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf")
