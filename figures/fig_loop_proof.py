#!/usr/bin/env python3
"""The two loops and the lever, provoked on the phone (from /tmp/loop_proof, run_loop_proof.sh).
(a) the lever bias per tier across the request sequence, with the plan each request ran;
(b) every request's metered time and energy against the budgets its loops check
    (1.0 = budget): a bar above 1.0 is a miss and the loop that owns it moved the lever.
"""
import os, re, glob, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, AQUA, RED, PURPLE, GREEN, ORANGE = "#2a78d6", "#1b9aa8", "#e34948", "#5b2a86", "#3aa41c", "#e08a1e"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "pdf.fonttype": 42, "ps.fonttype": 42})
HOST = sys.argv[1] if len(sys.argv) > 1 else "/tmp/loop_proof"
ORDER = ["ctrl_mains", "healthy", "mid_1", "mid_burn_1", "mid_burn_2", "mid_after_1", "mid_after_2",
         "low_1", "low_cap_1", "low_cap_2", "low_cap_3", "low_after_1", "low_after_2"]
DIST = {"mid_burn_1": "4 cores\nspinning", "mid_burn_2": "4 cores\nspinning", "low_cap_1": "726 cap\nat 12 s", "low_cap_2": "726 cap\nat 12 s", "low_cap_3": "726 cap\nat 12 s"}

def parse(path):
    s = open(path).read().strip()
    if not s: return None
    f = dict(re.findall(r"(\w+)=(\"[^\"]*\"|\S+)", s))
    b0, b1 = f["bias"].split("->")
    return dict(tier=f["tier"], lever=float(f["lever"]), b0=float(b0), b1=float(b1), plan=f["plan"], cap=int(f["nout_cap"]),
                E=float(f["meas_J"]), T=float(f["meas_s"]), PE=float(f["pred_J"]), PT=float(f["pred_s"]),
                tb=float(f["tbud"]), eb=float(f["ebud"]), act=s.split("; ")[-1])
rows = [(t, parse(f"{HOST}/{t}/sched_log.txt")) for t in ORDER if os.path.exists(f"{HOST}/{t}/sched_log.txt")]
rows = [(t, r) for t, r in rows if r]
if not rows:
    sys.exit("no cells yet")
short = {"gpu1200_k1024": "1200", "gpu1200d902_k1024": "1200/d902", "gpu1200d726_k1024": "1200/d726", "gpu902_k1024": "902", "gpu726_k1024": "726"}
tcol = {"mains": BLUE, "healthy": BLUE, "mid": ORANGE, "low": RED}


from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

fig, (a, b) = plt.subplots(2, 1, figsize=(7.0, 3.4), sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.0]))
fig.subplots_adjust(left=0.075, right=0.995, top=0.925, bottom=0.135, hspace=0.62)
xs = np.arange(len(rows))
for s in ("top", "right"):
    a.spines[s].set_visible(False); b.spines[s].set_visible(False)
a.set_xlim(-0.6, len(rows) - 0.4)

# (a) lever used (marker) and the nudge the loops applied afterwards (arrow); plan in one line above
for i, (x, (t, r)) in enumerate(zip(xs, rows)):
    c = tcol[r["tier"]]
    L_after = min(1, max(0, {"mains": 1.0, "healthy": 1.0, "mid": 0.5, "low": 0.0}[r["tier"]] + r["b1"]))
    a.plot([x], [r["lever"]], marker="o", ms=4.2, color=c, mec=SURF, mew=0.5, zorder=4, ls="none")
    if abs(L_after - r["lever"]) > 1e-9:
        a.annotate("", (x + 0.42, L_after), (x + 0.12, r["lever"]),
                   arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=6), zorder=4)
    else:
        a.plot([x + 0.12, x + 0.42], [r["lever"], r["lever"]], color=c, lw=0.9, zorder=4)
    plan = f"{short.get(r['plan'], r['plan'])}, {r['cap']}"
    a.text(x, 1.22 if i % 2 == 0 else 1.5, plan, ha="center", va="center", fontsize=6.5, color=c, zorder=5)
a.set_ylim(-0.12, 1.66); a.set_yticks([0, 0.5, 1.0]); a.set_ylabel("lever L")
a.set_title("(a) lever and plan per request", loc="left", fontsize=7.5, pad=3)
hs = [Line2D([], [], color=c, marker="o", ms=4, ls="-", lw=0.9, label=lab)
      for c, lab in ((BLUE, "mains / healthy"), (ORANGE, "mid (SoC 40)"), (RED, "low (SoC 15)"))]
a.legend(handles=hs, loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=3, fontsize=6.5, handlelength=1.6,
         columnspacing=1.2, handletextpad=0.5, borderaxespad=0)
# disturbances: short bracket + label just below panel (a)
DSPAN = {"4 cores spinning": [i for i, (t, _) in enumerate(rows) if t.startswith("mid_burn")],
         "726 MHz cap at 12 s": [i for i, (t, _) in enumerate(rows) if t.startswith("low_cap")]}
for lab, idx in DSPAN.items():
    if not idx: continue
    x0, x1 = min(idx) - 0.3, max(idx) + 0.3
    a.plot([x0, x0, x1, x1], [-0.30, -0.36, -0.36, -0.30], color=PURPLE, lw=0.7, clip_on=False, transform=a.transData)
    a.text((x0 + x1) / 2, -0.42, lab, ha="center", va="top", fontsize=6.5, color=PURPLE, clip_on=False)

# (b) meter against budgets
w = 0.34
labels = []
HALO = dict(facecolor=SURF, edgecolor="none", pad=0.6)
for x, (t, r) in zip(xs, rows):
    rt, re_ = r["T"] / r["tb"], r["E"] / r["eb"]
    b.bar(x - w / 2, rt, w, color=RED if r["T"] > r["tb"] else "#f2b8b7", edgecolor=SURF, zorder=3)
    b.bar(x + w / 2, re_, w, color=GREEN if r["E"] > r["eb"] else "#bfe3b4", edgecolor=SURF, zorder=3)
    labels.append(b.text(x - w / 2, rt + 0.02, f"{r['T']:.0f}s", ha="center", va="bottom", fontsize=6.5, color=INK, zorder=2.5, bbox=HALO))
    labels.append(b.text(x + w / 2, re_ + 0.02, f"{r['E']:.0f}J", ha="center", va="bottom", fontsize=6.5, color=INK, zorder=2.5, bbox=HALO))
b.axhline(1.0, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=2)
b.set_ylim(0, 1.75); b.set_yticks([0, 0.5, 1.0, 1.5]); b.set_ylabel("metered / budget")
b.set_title("(b) metered time and energy over budget (dark = miss)", loc="left", fontsize=7.5, pad=3)
b.set_xticks(xs); b.set_xticklabels([t.replace("_", "\n", 1).replace("_", " ") for t, _ in rows], fontsize=6.5)
b.tick_params(axis="x", length=0, pad=2)
hb = [Patch(color="#f2b8b7", label="time (performance loop)"), Patch(color="#bfe3b4", label="energy (energy loop)"),
      (Patch(color=RED), Patch(color=GREEN)), Line2D([], [], color=INK, lw=0.8, ls=(0, (3, 2)))]
b.legend(handles=hb, labels=["time (perf. loop)", "energy (energy loop)", "miss", "budget"],
         handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, loc="lower right", bbox_to_anchor=(1.0, 0.98),
         ncol=4, fontsize=6.5, handlelength=1.4, columnspacing=0.9, handletextpad=0.45, borderaxespad=0, handleheight=0.7)

# nudge value labels so no two touch and none touches another bar (display space); lift the higher-bottomed one
def unclash(ax, texts, obstacles=(), pad_px=1.8, tries=40):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    obs = [o.get_window_extent(rend) for o in obstacles]
    def hit(p, q):
        return p.x0 < q.x1 + pad_px and q.x0 < p.x1 + pad_px and p.y0 < q.y1 + pad_px and q.y0 < p.y1 + pad_px
    def lift(k, top_px):
        t = texts[k]; x, y = t.get_position()
        y_disp = ax.transData.transform((x, y))[1] + (top_px + pad_px - t.get_window_extent(rend).y0)
        t.set_position((x, ax.transData.inverted().transform((0, y_disp))[1]))
    for _ in range(tries):
        moved = False
        for i in range(len(texts)):
            bi = texts[i].get_window_extent(rend)
            for o in obs:
                if hit(bi, o) and bi.y0 < o.y1:      # label sits inside a neighbouring bar
                    lift(i, o.y1); bi = texts[i].get_window_extent(rend); moved = True
            for j in range(len(texts)):
                if j == i: continue
                bj = texts[j].get_window_extent(rend)
                if hit(bi, bj) and bi.y0 >= bj.y0:   # i is the higher one: lift i above j
                    lift(i, bj.y1); bi = texts[i].get_window_extent(rend); moved = True
        if not moved: break
unclash(b, labels, obstacles=b.patches)

def audit(fig, pad=0.5):
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); items = []
    for ax in fig.axes:
        items += [(t.get_text(), t.get_window_extent(rend)) for t in ax.texts if t.get_text().strip()]
        items += [(t.get_text(), t.get_window_extent(rend)) for t in (ax.title, ax._left_title, ax._right_title) if t.get_text().strip()]
        items += [(t.get_text(), t.get_window_extent(rend)) for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_text().strip() and t.get_visible()]
        if ax.get_legend(): items += [(t.get_text(), t.get_window_extent(rend)) for t in ax.get_legend().get_texts()]
    bad = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            p, q = items[i][1], items[j][1]
            if p.x0 < q.x1 + pad and q.x0 < p.x1 + pad and p.y0 < q.y1 + pad and q.y0 < p.y1 + pad:
                print(f"  OVERLAP: {items[i][0]!r} <-> {items[j][0]!r}"); bad += 1
    w, h = fig.get_size_inches()
    for name, bb in items:
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > w * fig.dpi or bb.y1 > h * fig.dpi: print(f"  CLIPPED: {name!r}"); bad += 1
    print("  audit:", "clean" if not bad else f"{bad} issue(s)")
audit(fig)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_loop_proof")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200); print("wrote", out)
for t, r in rows:
    print(f"  {t:12s} {r['tier']:8s} L={r['lever']:.2f} bias {r['b0']:+.2f}->{r['b1']:+.2f}  {r['plan']:18s} cap {r['cap']:5d}  T {r['T']:5.0f}/{r['tb']:.0f} s  E {r['E']:5.0f}/{r['eb']:.0f} J  {r['act']}")
