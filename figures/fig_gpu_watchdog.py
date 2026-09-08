#!/usr/bin/env python3
# ============================================================================
# fig_gpu_watchdog.py -- what the clock watchdog buys, and in which regime.
# (2026-08-24)
#
# WHY THIS FIGURE EXISTS. Our short-run table shows the watchdog doing almost
# nothing: on a single cool-started 12K+4K cell, muKV runs 27.58 tok/s without it
# and 27.37/27.38 with it -- inside the run-to-run SD. Read alone, that says the
# governor is pointless. It is not; it says the SHORT RUN IS THE WRONG TEST.
#
# The ladders are anchored at the MEASURED vendor deep-throttle trigger (battery
# 47 C for the real ladder, 36 C for the early one). A single cell from a cold
# gate peaks at battery 34.6 C and finishes in ~5 minutes, so it never approaches
# either anchor and the watchdog stays at tier 0 by construction. Sustained load
# does reach it -- and there the watchdog is worth +10.8% throughput (low ladder)
# or a third less time on the throttle floor (real ladder).
#
# So the figure contrasts the two regimes directly rather than reporting one
# number: panel (a) is the 8-generation soak, where the effect lives; panel (b)
# is the residency that explains it. The cost is stated too -- a higher sustained
# clock means MORE mean DDR heat, not less, which is why the paper claims throttle
# avoidance rather than cooling.
# ============================================================================
import os, json, re, csv, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VAN, NOWD, LOW, REAL = "#666666", "#0072B2", "#D55E00", "#009E73"
INK, GRID = "#1A1A1A", "#DDDDDD"

def J(f):
    try: return json.loads(re.sub(r':\s*-?nan\b', ': NaN', re.sub(r':\s*-?inf\b', ': Infinity', open(f).read())))
    except Exception: return {}

ARMS = [("vanilla", "vanilla", VAN, "o", "-"),
        ("mukv_nowd", "$\\mu$KV, watchdog off", NOWD, "s", "-"),
        ("mukv_wdlow", "$\\mu$KV + wd (early ladder)", LOW, "^", "-"),
        ("mukv_wdhigh", "$\\mu$KV + wd (vendor-anchored)", REAL, "D", "-")]

def gens(arm):
    out = []
    for i in range(1, 9):
        j = J(f"/tmp/gpu_sustained/{arm}/iter_{i}.json")
        if j.get("decode_ms") and j.get("n_decode_steps"):
            out.append(j["n_decode_steps"] * 1000.0 / j["decode_ms"])
    return out

def residency(arm):
    f = f"/tmp/gpusus_clk/{arm}.csv"
    if not os.path.exists(f): return None
    C = []
    for r in csv.DictReader(open(f)):
        try: C.append(float(r["gpu_clk"]))
        except Exception: pass
    return 100.0 * sum(1 for x in C if x < 500) / len(C) if C else None

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.2),
                             gridspec_kw={"width_ratios": [1.55, 1]})

# ---- (a) throughput across 8 consecutive generations -----------------------
for arm, lbl, c, mk, ls in ARMS:
    y = gens(arm)
    if not y: continue
    ax.plot(range(1, len(y) + 1), y, ls, color=c, marker=mk, ms=5,
            lw=1.8, label=lbl, zorder=3)
ax.set_xlabel("generation (12.2K prompt + 4K decode each)", fontsize=9.5)
ax.set_ylabel("decode throughput (tok/s)", fontsize=9.5)
ax.set_title("(a)  sustained soak — where the watchdog acts", fontsize=10.5,
             loc="left", fontweight="bold", pad=8)
ax.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color("#999999")
ax.tick_params(labelsize=8.5, color="#999999")
ax.legend(fontsize=8.2, frameon=False, loc="upper right", handletextpad=.6)

# the short-run result, drawn as a band so the contrast is visible not asserted
ax.axhspan(27.37, 27.58, color="#BBBBBB", alpha=.35, zorder=0)
ax.annotate("single cool-started cell:\n27.4–27.6 tok/s with the watchdog\non OR off — it never leaves tier 0",
            xy=(1.35, 27.48), xytext=(1.25, 24.0), fontsize=7.8, color="#555555",
            linespacing=1.4, ha="left",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0,
                            connectionstyle="arc3,rad=-0.2"))
ax.set_ylim(21.0, 39.6)

# ---- (b) time on the throttle floor ---------------------------------------
labels, vals, cols = [], [], []
for arm, lbl, c, _, _ in ARMS:
    r = residency(arm)
    if r is None: continue
    labels.append(lbl.replace("$\\mu$KV", "μKV").replace(" + wd", "\n+ wd").replace(", watchdog", "\nwatchdog"))
    vals.append(r); cols.append(c)
bars = bx.barh(range(len(vals)), vals, color=cols, height=.62, zorder=3)
bx.set_yticks(range(len(vals)))
bx.set_yticklabels(labels, fontsize=8.2)
bx.invert_yaxis()
for i, v in enumerate(vals):
    bx.text(v + .8, i, f"{v:.1f}%", va="center", fontsize=8.5,
            color=cols[i], fontweight="bold")
bx.set_xlabel("% of run at the 500 MHz throttle floor", fontsize=9.5)
bx.set_xlim(0, max(vals) * 1.28)
bx.set_title("(b)  why: throttle residency", fontsize=10.5, loc="left",
             fontweight="bold", pad=8)
bx.grid(True, axis="x", color=GRID, lw=.6); bx.set_axisbelow(True)
for sp in ("top", "right", "left"): bx.spines[sp].set_visible(False)
bx.spines["bottom"].set_color("#999999")
bx.tick_params(labelsize=8.5, color="#999999")

fig.suptitle("The watchdog is dormant on a short cell and worth +10.8% throughput on a sustained one — "
             "its ladders are anchored at the vendor's deep-throttle trigger.",
             fontsize=10.5, x=.006, ha="left", y=.985)
fig.tight_layout(rect=(0, 0, 1, .935))
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_gpu_watchdog"
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf")
