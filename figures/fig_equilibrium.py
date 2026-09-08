#!/usr/bin/env python3
# ============================================================================
# fig_equilibrium.py -- where the balance points are. (2026-08-11)
#
# muKV has two knobs that both trade performance for power, and they are NOT
# interchangeable. This finds the balance point of each by plotting the actual
# operating points measured on the phone GPU, cooled identically before every cell.
#
# THE TWO LEVERS BEHAVE DIFFERENTLY BECAUSE THE TWO PHASES ARE BOUND DIFFERENTLY.
#   decode  is BANDWIDTH-bound: it waits on DRAM, so dropping the shader clock costs it
#           nothing until the clock stops being the bottleneck's bottleneck. Measured:
#           throughput is FLAT 1200 -> 902 MHz (30.03 -> 30.33 tok/s) while power falls 25%.
#   prefill is COMPUTE-bound: it scales with clock, 137.5 -> 170.8 s over the same range.
# So the clock lever is nearly free for decode and expensive for prefill. A CPU power-law
# fit on this device (P ~ f^1.10, voltage pinned at Vmin => E ~ f^0.10) predicts clock
# capping is a poor ENERGY lever -- true for compute-bound work, and decisively false for
# bandwidth-bound decode. That exception is the whole point of this figure.
#
# The cache lever cuts power monotonically (7.00 -> 5.98 W) but costs throughput below
# k-pct 20, because decode-time eviction work stays roughly constant (~2500 evictions per
# 4096 tokens) while K shrinks 4x, so the per-step overhead stops being amortised.
#
# WHY BOTH TRACES CANNOT SHARE AN ABSOLUTE Y-AXIS: they were measured in separate campaigns
# whose vanilla baselines differ (32.97 vs 30.03 tok/s) because the phone's thermal history
# differed. Each trace is therefore normalised to ITS OWN best point; only the SHAPE of each
# curve, and where its knee falls, is being compared.
# ============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLOCK = "#0072B2"
CACHE = "#D55E00"

# (label, decode tok/s, mJ/token, mean W, DDR peak C, wall s)
CLK = [("1200", 30.03, 333.6, 4.95, 74.1, 276.1),
       ("1050", 30.11, 302.8, 4.29, 63.7, 288.8),
       ("902",  30.33, 280.7, 3.73, 57.5, 308.0),
       ("826",  29.62, 274.6, 3.51, 55.6, 320.8),
       ("726",  28.30, 273.3, 3.13, 54.0, 357.3)]
# (label, decode tok/s, mJ/token, mean W, PPL, wall s)
CACHE_PTS = [("vanilla", 32.97, 436.9, 7.00, 23.442, 255.7),
             ("20%",     34.74, 404.2, 6.54, 22.732, 253.3),
             ("10%",     32.09, 412.6, 6.46, 22.997, 261.4),
             ("5%",      30.67, 392.5, 5.98, 23.252, 268.7)]

fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.1))

# ---- Panel 1: the equilibrium plot -- throughput retained vs energy saved -------------
a = ax[0]
for pts, col, lab, ref in ((CLK, CLOCK, "GPU clock cap (MHz)", CLK[0]),
                           (CACHE_PTS, CACHE, "cache budget (k-pct)", CACHE_PTS[0])):
    x = [100 * (1 - p[2] / ref[2]) for p in pts]        # energy saved vs that trace's ref
    y = [100 * p[1] / max(q[1] for q in pts) for p in pts]  # throughput vs trace's own best
    a.plot(x, y, "-o", color=col, lw=1.4, ms=6, label=lab)
    for p, xx, yy in zip(pts, x, y):
        a.annotate(p[0], (xx, yy), fontsize=7, color=col,
                   xytext=(0, 7), textcoords="offset points", ha="center")
a.axhline(100, color="#444444", lw=0.9, ls="--")
a.text(0.02, 100.3, "no throughput lost", transform=a.get_yaxis_transform(),
       fontsize=7, color="#444444")
a.scatter([100 * (1 - 280.7 / 333.6)], [100 * 30.33 / 30.33], s=190, facecolor="none",
          edgecolor=CLOCK, lw=2.0, zorder=5)
a.annotate("balance point: 902 MHz\n16% less energy, full throughput",
           xy=(15.9, 100), xytext=(9.0, 90.5), fontsize=7.5, color=CLOCK,
           arrowprops=dict(arrowstyle="->", color=CLOCK, lw=0.9))
a.set_ylim(87, 102.5)
a.set_xlabel("energy saved per token (%)")
a.set_ylabel("decode throughput\n(% of that lever's best)")
a.set_title("Equilibrium: what each lever costs to save energy", fontsize=9.5)
a.legend(fontsize=7.5, frameon=False, loc="lower left")

# ---- Panel 2: thermal headroom -------------------------------------------------------
b = ax[1]
b.plot([p[2] for p in CLK], [p[4] for p in CLK], "-o", color=CLOCK, lw=1.4, ms=6)
for p in CLK:
    b.annotate(p[0], (p[2], p[4]), fontsize=7, color=CLOCK,
               xytext=(0, 7), textcoords="offset points", ha="center")
b.axhline(60, color="#B00020", lw=0.9, ls=":")
b.text(0.98, 60, " DDR 60 °C", transform=b.get_yaxis_transform(), ha="right",
       va="bottom", fontsize=7, color="#B00020")
b.set_xlabel("energy per token (mJ)")
b.set_ylabel("peak DDR temperature (°C)")
b.set_title("Thermal headroom bought by the clock lever", fontsize=9.5)

# ---- Panel 3: the phase split -- why the levers differ --------------------------------
# Prefill and decode are plotted separately because they are bound by different resources,
# and that split is the whole explanation for panel 1: the clock lever looks free only
# because decode dominates tok/s while prefill quietly pays for it in wall time.
c = ax[2]
pre = [276.1 - 136.4, 288.8 - 136.0, 308.0 - 135.1, 320.8 - 138.3, 357.3 - 144.7]
dec = [136.4, 136.0, 135.1, 138.3, 144.7]
xs = [int(p[0]) for p in CLK]
c.plot(xs, [100 * v / pre[0] for v in pre], "-o", color="#009E73", lw=1.4, ms=6,
       label="prefill (compute-bound)")
c.plot(xs, [100 * v / dec[0] for v in dec], "-o", color=CLOCK, lw=1.4, ms=6,
       label="decode (bandwidth-bound)")
c.axhline(100, color="#444444", lw=0.9, ls="--")
c.set_xticks(xs); c.set_xticklabels([str(v) for v in xs])
c.set_ylim(92, 160)
c.invert_xaxis()
c.set_xlabel("GPU clock cap (MHz)")
c.set_ylabel("time relative to 1200 MHz (%)")
c.set_title("Why: only compute-bound work pays for a lower clock", fontsize=9.5)
c.legend(fontsize=7.5, frameon=False, loc="upper left")
c.annotate("decode flat to 902 MHz", xy=(902, 99.0), xytext=(1150, 108),
           fontsize=7.5, color=CLOCK, arrowprops=dict(arrowstyle="->", color=CLOCK, lw=0.8))

for z in ax:
    z.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        z.spines[sp].set_visible(False)
fig.tight_layout()
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_equilibrium"
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf")
