#!/usr/bin/env python3
"""muKV with the energy-aware layer, drawn as a systems diagram: inputs, the scheduler with its lever,
the SoC with the KV cache as memory blocks, the two phases and their clock caps, the thermal path, and
the measurement loop back to the lever. Numbers from the OnePlus 15, Llama-3.2-1B, 9737-token prompt."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Arc

INK, MUTED, SURF, PAPER = "#0b0b0b", "#6b6963", "#ffffff", "#f7f7f5"
PUR, BLUE, TEAL, RED, GREEN, ORANGE, GREY = "#5b2a86", "#2a78d6", "#1b9aa8", "#e34948", "#3aa41c", "#e08a1e", "#9a9891"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.2, 4.35)); fig.subplots_adjust(0, 0, 1, 1); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

def rbox(x, y, w, h, fc=SURF, ec=INK, lw=0.9, r=1.0, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))
def T(x, y, s, size=5.6, color=INK, ha="left", va="center", bold=False, z=6, ls_=1.2):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight="bold" if bold else "normal", zorder=z, linespacing=ls_)
def arrow(p, q, color, lw=1.1, style="-|>", ls="-", rad=0.0, z=5, ms=9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z))
def path(pts, color, lw=1.1, ls="-", z=5, head=True, ms=9):
    for i in range(len(pts) - 1):
        last = i == len(pts) - 2
        arrow(pts[i], pts[i + 1], color, lw=lw, ls=ls, z=z, ms=ms, style="-|>" if (head and last) else "-")

# ---------------- icons -----------------------------------------------------------------------
def battery(x, y, w, h, level, color):
    rbox(x, y, w, h, fc=SURF, ec=INK, lw=0.8, r=0.5); ax.add_patch(Rectangle((x + w, y + h * 0.3), 0.9, h * 0.4, fc=INK, ec=INK, zorder=3))
    ax.add_patch(Rectangle((x + 0.5, y + 0.5), (w - 1.0) * level, h - 1.0, fc=color, ec="none", zorder=3))
def thermometer(x, y, h, frac, color):
    ax.add_patch(FancyBboxPatch((x - 0.55, y + 1.2), 1.1, h, boxstyle="round,pad=0,rounding_size=0.55", fc=SURF, ec=INK, lw=0.8, zorder=3))
    ax.add_patch(Rectangle((x - 0.3, y + 1.4), 0.6, h * frac, fc=color, ec="none", zorder=4)); ax.add_patch(Circle((x, y + 0.9), 1.0, fc=color, ec=INK, lw=0.8, zorder=4))
def document(x, y, w, h):
    ax.add_patch(Polygon([(x, y), (x + w - 1.2, y), (x + w, y + 1.2), (x + w, y + h), (x, y + h)], closed=True, fc=SURF, ec=INK, lw=0.8, zorder=3))
    for i in range(4): ax.plot([x + 0.7, x + w - 0.9], [y + h - 1.0 - i * 0.9] * 2, color=GREY, lw=0.6, zorder=4)
def cells(x, y, n, cw, ch, colors, gap=0.12):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * (cw + gap), y), cw, ch, fc=colors[i], ec=SURF, lw=0.25, zorder=4))
def cylinder(x, y, w, h, fc, ec):
    from matplotlib.patches import Ellipse
    ax.add_patch(Rectangle((x, y + h * 0.12), w, h * 0.76, fc=fc, ec=ec, lw=0.8, zorder=3))
    ax.add_patch(Ellipse((x + w / 2, y + h * 0.12), w, h * 0.24, fc=fc, ec=ec, lw=0.8, zorder=3)); ax.add_patch(Ellipse((x + w / 2, y + h * 0.88), w, h * 0.24, fc=fc, ec=ec, lw=0.8, zorder=4))
    ax.plot([x, x], [y + h * 0.12, y + h * 0.88], color=ec, lw=0.8, zorder=4); ax.plot([x + w, x + w], [y + h * 0.12, y + h * 0.88], color=ec, lw=0.8, zorder=4)

# ================= 1. inputs ==================================================================
rbox(1.5, 30, 17, 28.5, fc=PAPER, ec=GREY, lw=0.7, r=1.2, z=1)
T(2.8, 57.2, "1  PHONE STATE AND REQUEST", 5.4, MUTED, bold=True)
battery(3.5, 50.5, 7.5, 3.6, 0.47, ORANGE); T(12.5, 52.3, "battery 47%\nnot charging", 5.2)
thermometer(5.3, 41.2, 5.5, 0.6, RED); T(8.5, 44.6, "DDR 58 C\nbattery 35 C", 5.2)
document(3.6, 32.2, 5.2, 6.2); T(10.0, 35.3, "prompt 9.7K tokens\nanswer length open", 5.2)

# ================= 2. scheduler ===============================================================
rbox(22, 30, 31.5, 28.5, fc="#f4effa", ec=PUR, lw=1.0, r=1.2, z=1)
T(23.3, 57.2, "2  muKV SCHEDULER  (0.3 s per request)", 5.4, PUR, bold=True)
# the lever: a slider with the three tier anchors and the loop bias
sx0, sx1, sy = 25.5, 48.5, 51.5
ax.plot([sx0, sx1], [sy, sy], color=PUR, lw=2.4, solid_capstyle="round", zorder=3)
for f, lab in ((0, "0  low\n<= 20%"), (0.5, "0.5  mid\n21 to 50%"), (1, "1  healthy\n> 50%, mains")):
    xx = sx0 + f * (sx1 - sx0); ax.plot([xx, xx], [sy - 0.9, sy + 0.9], color=PUR, lw=1.0, zorder=4); T(xx, sy - 1.6, lab, 4.6, PUR, ha="center", va="top")
kx = sx0 + 0.4 * (sx1 - sx0)
ax.add_patch(Circle((kx, sy), 1.15, fc=SURF, ec=PUR, lw=1.3, zorder=5)); T(kx, sy + 2.3, "lever L = 0.4  (tier 0.5, loop bias -0.1)", 4.9, PUR, ha="center", va="bottom", bold=True)
T(25.5, 45.4, "L sets the exchange rate, quality floor, time slack and output cap", 4.2, MUTED)
# the ladder: five plans in time order, the chosen one highlighted
tiles = [("1200", GREY), ("1200/d902", GREY), ("1200/d726", PUR), ("902", GREY), ("726", GREY)]
for i, (lab, c) in enumerate(tiles):
    x0 = 25.5 + i * 5.0
    ax.add_patch(Rectangle((x0, 39.5), 4.6, 4.0, fc="#e8ddf5" if c == PUR else SURF, ec=c, lw=1.5 if c == PUR else 0.9, zorder=3))
    T(x0 + 2.3, 41.5, lab, 3.9 if "/" in lab else 4.4, PUR if c == PUR else MUTED, ha="center", bold=(c == PUR))
    if i < 4: arrow((x0 + 4.7, 41.5), (x0 + 5.0, 41.5), MUTED, lw=0.6, ms=5)
T(25.5, 38.2, "ladder walk in time order: take a step only if the energy saved is at least\nL's exchange rate times the time given up; stop at the time budget or quality floor", 3.9, MUTED, va="top")
rbox(25.5, 31.2, 24.5, 4.4, fc=SURF, ec=PUR, lw=1.0, r=0.5, z=3)
T(26.3, 34.4, "plan   GPU, prefill 1200 MHz, decode 726", 4.6, PUR, bold=True)
T(26.3, 32.4, "K = 1024 kept   answer cap 1024 tokens", 4.6, PUR)

# ================= 3. the SoC =================================================================
rbox(55.5, 30, 43, 28.5, fc="#eef6f7", ec=TEAL, lw=1.0, r=1.2, z=1)
T(56.8, 57.2, "3  SNAPDRAGON 8 ELITE GEN 5", 5.4, TEAL, bold=True)
# GPU block with the phase bar
rbox(57, 45.5, 17.5, 9.5, fc=SURF, ec=BLUE, lw=0.9, r=0.6, z=2)
T(58, 53.7, "Adreno 840 GPU", 5.2, BLUE, bold=True)
ax.add_patch(Rectangle((58, 47.2), 9.5, 2.6, fc="#2a78d6", ec="none", zorder=3)); ax.add_patch(Rectangle((67.5, 47.2), 6.0, 1.6, fc="#8fb8e8", ec="none", zorder=3))
T(62.7, 48.5, "prefill 1200 MHz", 4.4, SURF, ha="center", bold=True); T(70.5, 48.0, "decode 726", 4.1, INK, ha="center")
T(58, 51.5, "compute-bound        bandwidth-bound", 4.2, MUTED)
T(58, 46.3, "cap written as root: max_pwrlevel + max_gpuclk", 4.0, MUTED)
# CPU block
rbox(76, 45.5, 21, 9.5, fc=SURF, ec=RED, lw=0.9, r=0.6, z=2)
T(77, 53.7, "Oryon CPU", 5.2, RED, bold=True)
T(77, 51.3, "clock: not an energy lever (flat 883 to 1632)", 4.2, MUTED)
T(77, 49.3, "watchdog ladder 1497 / 1382 / 1267 MHz", 4.4, RED)
T(77, 47.3, "one step before the vendor's 883", 4.2, MUTED)
# DRAM with the KV cache as memory blocks
rbox(57, 31.2, 35.5, 13.2, fc=SURF, ec=INK, lw=0.9, r=0.6, z=2)
T(58, 43.4, "LPDDR: the KV cache", 5.2, INK, bold=True)
T(58, 41.3, "after prefill: 9737 cells, every one scored by the kq_evict side node", 4.3, MUTED)
cells(58, 38.6, 52, 0.52, 1.8, [GREY] * 52)
arrow((66, 38.4), (66, 36.6), TEAL, lw=1.0, ms=7); T(67.2, 37.5, "compact in place, FA kernel stays on", 4.2, TEAL)
cols = [INK] * 3 + [TEAL] * 18 + [ORANGE] * 4
cells(58, 33.8, 25, 0.52, 1.8, cols)
T(58 + 25 * 0.62 + 1.0, 34.7, "1024 kept: 4 sink, top-K by score, recent", 4.2, INK)
T(58, 32.2, "22.6 MiB live on the GPU; decode reads only this", 4.2, MUTED)

# ================= 4. thermal (right, below SoC) ==============================================
rbox(55.5, 15.5, 43, 12, fc="#fdf0f0", ec=RED, lw=0.9, r=1.2, z=1)
T(56.8, 26.2, "4  HEAT HAS ITS OWN PATH", 5.4, RED, bold=True)
thermometer(59.5, 17.2, 5.0, 0.8, RED)
T(62.5, 22.4, "watchdog (CPU): steps the clock down at the battery and skin ladders,\nbefore the vendor limiter", 4.5, INK, va="center")
T(62.5, 18.3, "vendor limiter (last resort): GPU to 726 MHz at DDR ~64 C on the cable,\nwithin a minute on battery; our caps only ever sit below it", 4.5, INK, va="center")

# ================= 5. meter and loops (left, below) ============================================
rbox(1.5, 3, 50.5, 24.5, fc="#eef7ea", ec=GREEN, lw=0.9, r=1.2, z=1)
T(2.8, 26.2, "5  METER AND THE TWO LOOPS  (after every request)", 5.4, GREEN, bold=True)
# meter icon: a small trace
xs = np.linspace(3.5, 14.5, 80); ys = 9.5 + 2.2 * (np.abs(np.sin(xs * 1.7)) ** 3) + 0.4 * np.sin(xs * 9)
ax.plot(xs, ys, color=GREEN, lw=0.9, zorder=4); ax.plot([3.5, 14.5], [9.3, 9.3], color=GREY, lw=0.5, zorder=3)
T(3.5, 14.2, "meter", 5.0, GREEN, bold=True); T(3.5, 21.5, "USB rail + coulomb counter,\nclocks, temperatures;\nstart-anchored split into\nprefill and decode energy", 4.5, INK, va="top")
# cost table cylinder
cylinder(19.5, 7.5, 9, 7.5, "#ffffff", GREEN); T(24, 11.2, "cost table", 4.7, GREEN, ha="center", bold=True); T(24, 9.4, "EMA, clip 10%", 4.1, MUTED, ha="center")
T(25.8, 20.5, "predicts E and T per plan,\nlearns from the meter", 4.5, INK, ha="left", va="top")
# two loops as circular arrows around the lever's mirror
for cx, lab, sub, col in ((36.5, "performance loop", "time over budget\n-> lever +0.1", BLUE), (46.0, "energy loop", "energy over prediction\n-> lever -0.1", GREEN)):
    ax.add_patch(Arc((cx, 12.5), 5.2, 5.2, angle=0, theta1=30, theta2=330, color=col, lw=1.3, zorder=4))
    ang = np.deg2rad(30); ax.add_patch(Polygon([(cx + 2.6 * np.cos(ang), 12.5 + 2.6 * np.sin(ang)), (cx + 2.6 * np.cos(ang) - 1.1, 12.5 + 2.6 * np.sin(ang) + 0.4), (cx + 2.6 * np.cos(ang) - 0.2, 12.5 + 2.6 * np.sin(ang) + 1.2)], closed=True, fc=col, ec="none", zorder=5))
    T(cx, 17.4, lab, 4.9, col, ha="center", bold=True); T(cx, 8.6, sub, 4.3, INK, ha="center", va="top")
T(38.5, 5.4, "both budgets met: the bias decays toward the tier default", 4.2, MUTED, ha="center")

# ================= arrows between blocks =========================================================
path([(18.5, 44), (22, 44)], PUR, lw=1.2)                                    # inputs -> scheduler
path([(53.5, 44), (55.5, 44)], PUR, lw=1.2)                                    # plan -> SoC
path([(74.5, 50.2), (76, 50.2)], GREY, lw=0.8, head=False)
path([(95, 45.5), (95, 30), (95, 27.5)], RED, lw=1.0, ls=(0, (2, 2)))  # heat -> thermal path
path([(62, 30), (62, 28.6), (10, 28.6), (10, 27.5)], GREEN, lw=1.1, ls=(0, (3, 2)))  # meter reads the run
T(53.5, 29.2, "rail, coulombs, clocks, temperatures", 4.2, GREEN, ha="left", va="bottom")
path([(16.5, 11.5), (19.5, 11.5)], GREEN, lw=1.0)                            # meter -> table
path([(28.5, 11.5), (33.5, 12.5)], GREEN, lw=1.0)                            # table -> loops (E, T vs prediction)
path([(41.3, 12.5), (43.3, 12.5)], GREEN, lw=0.8, head=False); path([(48.7, 12.5), (51, 12.5)], PUR, lw=1.1, head=False)
path([(51, 12.5), (51, 27.5), (51, 30)], PUR, lw=1.1)                         # loops -> lever
T(50.3, 22.5, "bias per tier", 4.4, PUR, ha="right")
path([(24, 15), (24, 27.5), (24, 30)], GREEN, lw=1.0, ls=(0, (3, 2)))        # table -> scheduler
T(23.2, 24.5, "predictions", 4.4, GREEN, ha="right")
# legend
ax.plot([57, 60], [8.5, 8.5], color=PUR, lw=1.2); T(60.8, 8.5, "control, per request", 4.4, MUTED)
ax.plot([57, 60], [6.5, 6.5], color=GREEN, lw=1.1, ls=(0, (3, 2))); T(60.8, 6.5, "measurement and learning", 4.4, MUTED)
ax.plot([57, 60], [4.5, 4.5], color=RED, lw=1.0, ls=(0, (2, 2))); T(60.8, 4.5, "heat", 4.4, MUTED)
ax.add_patch(Rectangle((76, 8.0), 1.2, 1.2, fc=INK)); ax.add_patch(Rectangle((77.5, 8.0), 1.2, 1.2, fc=TEAL)); ax.add_patch(Rectangle((79, 8.0), 1.2, 1.2, fc=ORANGE)); ax.add_patch(Rectangle((80.5, 8.0), 1.2, 1.2, fc=GREY))
T(82.2, 8.6, "KV cells: sink, kept, recent, evicted", 4.4, MUTED)
T(57, 2.6, "Measured: 1336 J with the decode cap against 1428 J uncapped on the cable, +1.5% time;\non the real battery 885, 748 and 506 J per request across the three tiers.", 4.1, MUTED, va="center")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_system_architecture")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); print("wrote", out)
