#!/usr/bin/env python3
# ============================================================================
# fig_stalls_vs_cache.py -- does a bigger KV cache stall the CPU? (2026-08-17)
#
# THE HYPOTHESIS THIS TESTS. The intuitive story is: more cache -> more memory
# pressure -> stalls -> the governor drops the clock -> throughput and energy
# suffer. If true, cache size would be a strong energy actuator and the whole
# battery-aware section writes itself.
#
# THE DATA REFUTES IT. Measured on the phone CPU (9.7K prompt + 4K decode, one
# build, vendor DVFS, cool start per cell):
#   * psi_io = 0.000 and psi_mem <= 0.009 for EVERY policy, with 6.6-7.2 GB free
#     -- there is no paging and no memory pressure at any cache size.
#   * The FA-off policies span an 8x range of live cells (777 -> 6112) and sit at
#     an INDISTINGUISHABLE clock (1084-1116 MHz), deep-throttle residency
#     (57.7-62.8% of the run at the 883 MHz floor) and stall level (psi_cpu
#     6.6-7.2). Cache size moves none of them.
#   * vanilla holds the LARGEST cache (9741 cells) and stalls the LEAST
#     (psi_cpu 0.985, 23.3% at the floor).
#
# WHAT ACTUALLY DRIVES IT is the attention path. FlashAttention-off forces the
# engine to materialise the attention tensor every decode step; that is a CPU
# compute burst, it trips the vendor's current/thermal governor down to the
# 883 MHz floor for ~60% of the run, and it costs 3.6x throughput -- at the SAME
# bytes-per-token. muKV and StreamingLLM read an identical 823 MB/token (721 vs
# 777 cells) and differ 3.6x in throughput purely on FA on/off.
#
# REVISED 2026-08-17 after two objections, both of which the data upheld:
#  (1) "muKV should stall LEAST by design." Correct, and panel (a) alone hides it:
#      a raw percentage-of-run penalises the policy that FINISHES SOONEST. muKV
#      generates the same 4096 tokens in 169 s against vanilla's 815 s, so an
#      equal fraction of a shorter run is far less absolute exposure. Normalised
#      by work done (panel b) muKV is the LOWEST of all nine: 13.5 s at the floor
#      per 1000 tokens, 3.4x under vanilla and 6.6x under the FA-off cluster.
#  (2) "the throttling may just be because a small cache makes the CPU work
#      harder." A real confound, and it is why panel (b) exists. Across all nine:
#      residency vs cache r = -0.12, vs tok/s r = -0.49, vs FA-off flag r = +0.94.
#      The attention path dominates. Within the FA-off subset alone there is a
#      weak positive cache trend (r = +0.52, n=6, not significant) -- reported
#      rather than flattened into "no relationship".
#  The watchdog is NOT the explanation: its log for this campaign ends
#  "last tier=0 driver=none threat=0/1000" -- it never engaged. What is measured
#  here is eviction efficiency plus the FA path, with the governor dormant.
# ============================================================================
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FAOFF, FAON, MUKV, INK, GRID = "#0072B2", "#666666", "#D55E00", "#1A1A1A", "#DDDDDD"

D = json.load(open("/tmp/stall_fig_data.json"))
# 2026-08-22: muKV is fa_on + in-place defrag, one configuration. The state-swap variant
# (FA-off prefill -> FA-on decode) is retired and is NOT reported as a muKV configuration,
# so it is dropped here rather than drawn as a second muKV point.
D = [r for r in D if r["pol"] != "mukv_swap"]
LABEL = {"h2o": "H2O", "snapkv": "SnapKV", "tova": "TOVA", "tova_canon": "TOVA-canon",
         "adakv": "Ada-KV", "streamingllm": "StreamingLLM", "vanilla": "vanilla",
         "mukv_faon": "$\\mu$KV", "mukv_swap": "$\\mu$KV (swap)"}

# muKV state-swap runs FA-OFF during PREFILL (it captures attention through the
# eval-callback, then restores the compacted cache into an FA-on context to
# decode). It is drawn hollow to say so: faon vs swap is the cleanest control in
# the whole set -- same policy, same keep-set, 721 vs 733 cells, differing ONLY in
# whether prefill leaves the FA path.
def style(r):
    if r["pol"] == "mukv_faon":  return MUKV, "o", 95, 1.0, MUKV
    return ((FAOFF, "s", 62, 1.0, "white") if r["fa"] == "off"
            else (FAON, "D", 62, 1.0, "white"))

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))

for r in D:                      # absolute exposure, normalised by work done
    wall = 4096 / r["tps"] if r["tps"] else 0
    r["norm"] = (wall * r["deep"] / 100) / 4.096      # seconds at 883 MHz per 1k tokens

PANELS = [
    ("deep", "% of run at the 883 MHz throttle floor",
     "(a)  raw fraction — penalises whoever finishes first", (0, 72)),
    ("norm", "seconds at 883 MHz per 1000 tokens",
     "(b)  normalised by work done",                          (0, 132)),
]

for ax, (key, ylab, title, ylim) in zip(axes, PANELS):
    faoff = [r for r in D if r["fa"] == "off"]
    lo, hi = min(r[key] for r in faoff), max(r[key] for r in faoff)
    ax.axhspan(lo, hi, color=FAOFF, alpha=0.10, zorder=0)
    if key == "deep":
        # bracket belongs on the raw panel only -- in (b) the FA-off band is much
        # wider, so calling it "one band" there would overstate the flatness.
        ax.annotate("", xy=(777, hi + (ylim[1] * .05)), xytext=(6112, hi + (ylim[1] * .05)),
                    arrowprops=dict(arrowstyle="<->", color=FAOFF, lw=1.2))
        ax.text(2180, hi + (ylim[1] * .08),
                "8$\\times$ cache range, one band\n(within it, weak trend r=+0.52)",
                ha="center", va="bottom", fontsize=7.8, color=FAOFF, linespacing=1.35)
    if key == "norm":
        mk = next(r for r in D if r["pol"] == "mukv_faon")
        ax.annotate("$\\mu$KV lowest of all\n3.4$\\times$ under vanilla",
                    (mk["cells"], mk["norm"]), textcoords="offset points",
                    xytext=(16, 26), fontsize=8, color=MUKV, linespacing=1.35,
                    arrowprops=dict(arrowstyle="->", color=MUKV, lw=1.1))

    for r in D:
        c, m, sz, a, ec = style(r)
        ax.scatter(r["cells"], r[key], s=sz, marker=m,
                   facecolor=c, alpha=a, edgecolor=ec,
                   linewidth=1.6 if c == "none" else .8, zorder=3)
    if key == "norm":   # the within-policy control, drawn as a link
        pass

    # Selective direct labels only -- never one per point. Offsets are in POINTS
    # via textcoords so they cannot collide with the axis on a log x-scale, which
    # is what a data-space offset did in the first render.
    labels = (("streamingllm", 12, -13, "left"), ("h2o", 9, 10, "left"),
              ("vanilla", -10, -13, "right")) + \
             ((("mukv_faon", 13, 1, "left"),) if key == "deep" else ())
    for pol, ox, oy, ha in labels:
        r = next((x for x in D if x["pol"] == pol), None)
        if not r: continue
        ax.annotate(LABEL[pol], (r["cells"], r[key]),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=8.5, color=INK, ha=ha, va="center")

    ax.set_xscale("log")
    ax.set_xlim(520, 15000)
    ax.set_ylim(*ylim)
    ax.set_xlabel("live KV cells retained  (log scale)", fontsize=9.5)
    ax.set_ylabel(ylab, fontsize=9.5)
    ax.set_title(title, fontsize=10.5, loc="left", fontweight="bold", pad=8)
    ax.set_xticks([1000, 2000, 5000, 10000])
    ax.set_xticklabels(["1k", "2k", "5k", "10k"])
    ax.grid(True, color=GRID, lw=.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#999999")
    ax.tick_params(labelsize=8.5, color="#999999")

handles = [Line2D([], [], marker="s", ls="", color=FAOFF, ms=7, label="FlashAttention OFF (score-reading)"),
           Line2D([], [], marker="D", ls="", color=FAON, ms=7, label="FlashAttention ON (vanilla)"),
           Line2D([], [], marker="o", ls="", color=MUKV, ms=8.5, label="$\\mu$KV  (FA-on prefill, side node)"),
           ]
axes[0].legend(handles=handles, fontsize=8.5, frameon=False, loc="lower left",
               handletextpad=.5, borderaxespad=.4)

fig.suptitle("Throttling tracks the attention path, not cache size (r=+0.94 vs FA-off, −0.12 vs cache).  "
             "psi$_{io}$=0.000, 6.6–7.2 GB free, watchdog dormant throughout.",
             fontsize=11, x=.007, ha="left", y=.985)
fig.tight_layout(rect=(0, 0, 1, .945))
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_stalls_vs_cache"
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf")
