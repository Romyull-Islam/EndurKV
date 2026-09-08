#!/usr/bin/env python3
"""muKV architecture, v8: the inference pipeline on top, and a control plane below with three
columns, thermal, energy and learning. Fewer words than v7; every box two lines at most. Badges
1 to 8 number the muKV mechanisms in execution order. Arrow styles: data solid, control dashed,
heat dotted, measurement dash-dot. Colors (Okabe-Ito): orange = new muKV mechanism, gray =
unmodified llama.cpp / hardware, green = muKV outcome, blue = KV cache."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

GRAY='#9a9a9a'; GRAYF='#f2f2f2'; INK='#1a1a1a'
ORANGE='#D55E00'; ORANGEF='#fdeee6'; BLUE='#0072B2'; BLUEF='#eaf3fa'
GREEN='#009E73'; GREENF='#e8f6f1'; HEAT='#777777'; MEAS='#009E73'
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x, y, w, h, title, sub='', ec=GRAY, fc='white', fs=6.4, fsub=4.9, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.4', fc=fc, ec=ec, lw=lw, zorder=3))
    if sub:
        ax.text(x+w/2, y+h/2+2.6, title, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold', zorder=4)
        ax.text(x+w/2, y+h/2-2.4, sub, ha='center', va='center', fontsize=fsub, color=INK, zorder=4, linespacing=1.25)
    else:
        ax.text(x+w/2, y+h/2, title, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold', zorder=4)
def arr(x1, y1, x2, y2, c=INK, lw=1.2, ls='-', ms=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c, ls=ls, zorder=2, shrinkA=0.5, shrinkB=0.5))
def seg(pts, c, lw=1.2, ls='-'):
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]): ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2, solid_capstyle='round')
def step(x, y, n):
    ax.annotate(str(n), (x, y), ha='center', va='center', fontsize=6.0, color='white', fontweight='bold', zorder=7, bbox=dict(boxstyle='circle,pad=0.28', fc=ORANGE, ec='none'))
def lab(x, y, t, c=INK, fs=5.0, ha='center', va='center', rot=0, bold=False):
    ax.text(x, y, t, fontsize=fs, ha=ha, va=va, color=c, rotation=rot, fontweight='bold' if bold else 'normal', zorder=6, linespacing=1.2)
def head(x, y, t, c):
    ax.text(x, y, t, fontsize=5.6, fontweight='bold', color=c, ha='left', va='center', zorder=2)

# ================= device boundary, two bands =================
ax.add_patch(FancyBboxPatch((0.8, 4.5), 98.4, 94.5, boxstyle='round,pad=0.1,rounding_size=1.5', fc='white', ec=INK, lw=1.2, zorder=0))
ax.text(3.0, 97.0, 'OnePlus 15', fontsize=7.5, fontweight='bold', color=INK, ha='left', va='center', zorder=1)
ax.add_patch(FancyBboxPatch((2.5, 57), 95, 38, boxstyle='round,pad=0.1,rounding_size=1.2', fc='#fcfcfc', ec='#bbbbbb', lw=0.9, zorder=1))
ax.text(4.0, 92.6, 'inference: llama.cpp on the CPU or the Adreno GPU', fontsize=6.0, style='italic', color='#555555', ha='left', va='center', zorder=2)
ax.add_patch(FancyBboxPatch((2.5, 7), 95, 46, boxstyle='round,pad=0.1,rounding_size=1.2', fc='#fcfcfc', ec='#bbbbbb', lw=0.9, zorder=1))
ax.text(4.0, 50.6, 'control plane', fontsize=6.0, style='italic', color='#555555', ha='left', va='center', zorder=2)

# ================= band A: the pipeline =================
y0, h0 = 76, 13
box(4,  y0, 9,  h0, 'Prompt', 'N tokens', ec=GRAY, fc=GRAYF)
box(17, y0, 14, h0, 'Prefill', 'FA on, full graph', ec=GRAY, fc=GRAYF)
box(37, y0, 14, h0, 'Score', 'kq_evict side node\nscores every token', ec=ORANGE, fc=ORANGEF)
box(57, y0, 14, h0, 'Evict + compact', 'keep K, slide in place\nFA kernel stays on', ec=ORANGE, fc=ORANGEF)
box(77, y0, 19, h0, 'Decode', 'over the K-cell cache;\nkeep set frozen, window slides', ec=GREEN, fc=GREENF)
step(37, y0 + h0, 1); step(57, y0 + h0, 2); step(77, y0 + h0, 3)
arr(13.2, y0 + 6.5, 16.8, y0 + 6.5); arr(31.2, y0 + 6.5, 36.8, y0 + 6.5, c=ORANGE); arr(51.2, y0 + 6.5, 56.8, y0 + 6.5, c=ORANGE); arr(71.2, y0 + 6.5, 76.8, y0 + 6.5, c=ORANGE)
lab(34, y0 + 9.2, 'K, Q', ORANGE, 4.8); lab(54, y0 + 9.2, 'keep set', ORANGE, 4.8)
# KV strips
ys = 62
for i in range(18): ax.add_patch(Rectangle((8 + i*2.0, ys), 1.85, 4.6, fc=BLUEF, ec=BLUE, lw=0.5, zorder=3))
lab(26, ys - 2.4, 'full KV cache, N cells', fs=5.0)
arr(24, y0 - 0.4, 24, ys + 5.2, c=BLUE); lab(25.2, 70.5, 'write all N', BLUE, 4.8, ha='left')
arr(45, ys + 2.3, 56, ys + 2.3, c=BLUE, lw=2.0, ms=15); lab(50.5, ys + 5.4, 'N → K', fs=5.2, bold=True)
xk = 57.5
for n, fc, name in ((2, '#c7dcef', 'sink'), (4, '#9ec6e8', 'kept'), (3, '#dcebf7', 'recent')):
    for i in range(n): ax.add_patch(Rectangle((xk + i*1.55, ys), 1.45, 4.6, fc=fc, ec=BLUE, lw=0.5, zorder=3))
    lab(xk + n*1.55/2, ys - 2.0, name, fs=4.6); xk += n*1.55
arr(64, y0 - 0.4, 64, ys + 5.2, c=ORANGE, ls=(0, (4, 2))); lab(65.2, 70.5, 'slide', ORANGE, 4.8, ha='left')
seg([(71.8, ys + 2.3), (86, ys + 2.3)], BLUE); arr(86, ys + 2.3, 86, y0 - 0.4, c=BLUE); lab(79, ys + 0.2, 'read K', BLUE, 4.8)

# ================= band B: control plane, three columns =================
# ---- column 1: thermal
head(5, 47.5, 'THERMAL', HEAT)
box(4, 36, 27, 9, 'Sensors', 'skin, battery, DDR, CPU at 2 Hz', ec=GRAY, fc=GRAYF)
box(4, 22.5, 27, 9, 'Watchdog: CPU + GPU clock', 'CPU 1497 → 1382 → 1267 at battery, skin\nGPU 1050 → 967 → 902 at DDR 60 / 62 / 63.5', ec=ORANGE, fc=ORANGEF, fsub=4.6)
box(4, 9, 27, 9, 'Vendor limiter (last resort)', 'GPU to 726 MHz late, CPU to 883;\nthe ladders step first, by max()', ec=GRAY, fc=GRAYF, fsub=4.6)
step(4, 31.5, 4)
arr(17.5, 36, 17.5, 31.9, c=INK); arr(17.5, 22.5, 17.5, 18.4, c=GRAY, ls=(0, (4, 2))); lab(18.7, 20.4, 'if it is late', GRAY, 4.6, ha='left')
arr(20, 57.4, 20, 45.4, c=HEAT, ls=(0, (1, 2))); lab(21.2, 51.5, 'heat', HEAT, 4.8, ha='left')
# ---- column 2: energy
head(35, 47.5, 'ENERGY  (once per request, 0.3 s)', ORANGE)
box(34, 36, 30, 9, 'Battery, charging, request', 'level and status; prompt N; length asked?', ec=GRAY, fc=GRAYF)
box(34, 22.5, 30, 9, 'Lever L = tier + loop bias', '1 above 50% or mains; 0.5 at 21 to 50; 0 at 20', ec=ORANGE, fc=ORANGEF)
box(34, 9, 30, 9, 'Plan', 'GPU clock caps (prefill, decode), K, answer cap', ec=GREEN, fc=GREENF)
step(34, 31.5, 5); step(34, 18, 6)
arr(49, 36, 49, 31.9, c=INK); arr(49, 22.5, 49, 18.4, c=ORANGE); lab(50.2, 20.4, 'ladder walk on the cost table', ORANGE, 4.6, ha='left')
seg([(64, 13.5), (66, 13.5), (66, 55)], ORANGE, 1.2, (0, (4, 2))); arr(66, 55, 66, 57.4, c=ORANGE, ls=(0, (4, 2)))
lab(67.4, 45.5, 'plan: clock caps, K, answer cap', ORANGE, 4.6, ha='left', va='center', rot=90)
# ---- column 3: learning
head(72, 47.5, 'LEARNING', MEAS)
box(71, 36, 25, 9, 'Meter', 'USB rail + coulomb counter,\nsplit into prefill and decode', ec=ORANGE, fc=ORANGEF)
box(71, 22.5, 25, 9, 'Two loops on model error', 'time over budget: L +0.1\nenergy over prediction: L −0.1', ec=ORANGE, fc=ORANGEF)
box(71, 9, 25, 9, 'Cost table learns', 'EMA per plan, clipped 10%;\nbandit check: same plans (13/15)', ec=ORANGE, fc=ORANGEF)
step(71, 45, 7); step(71, 31.5, 8)
arr(83.5, 36, 83.5, 31.9, c=INK); arr(83.5, 22.5, 83.5, 18.4, c=INK)
seg([(93, 57.4), (93, 45.4)], MEAS, 1.1, (0, (5, 2, 1, 2))); ax.plot([93], [45.4], marker='v', ms=3.5, color=MEAS, zorder=6); lab(94.2, 51.5, 'measure', MEAS, 4.8, ha='left')
seg([(71, 27), (68.5, 27), (68.5, 27)], ORANGE, 1.1, (0, (4, 2))); arr(68.5, 27, 64.2, 27, c=ORANGE, ls=(0, (4, 2))); lab(66.6, 29.0, 'bias', ORANGE, 4.6)
arr(71, 13.5, 64.2, 13.5, c=MEAS, ls=(0, (5, 2, 1, 2))); lab(67.6, 15.5, 'costs', MEAS, 4.6)

# ================= legend =================
ly = 2.2
for x, ec, fc, t in ((3, ORANGE, ORANGEF, 'new μKV mechanism'), (20, GRAY, GRAYF, 'unmodified llama.cpp / hardware'), (44, GREEN, GREENF, 'μKV outcome'), (56.5, BLUE, BLUEF, 'KV cache')):
    ax.add_patch(FancyBboxPatch((x, ly - 0.9), 3.2, 1.8, boxstyle='round,pad=0.1,rounding_size=0.5', fc=fc, ec=ec, lw=0.9)); lab(x + 4.0, ly, t, fs=4.6, ha='left')
for x, c, ls, t in ((66, INK, '-', 'data'), (73.5, ORANGE, (0, (4, 2)), 'control'), (82.5, HEAT, (0, (1, 2)), 'heat'), (89.5, MEAS, (0, (5, 2, 1, 2)), 'measure')):
    ax.plot([x, x + 2.6], [ly, ly], color=c, lw=1.2, ls=ls); lab(x + 3.2, ly, t, fs=4.6, ha='left')
fig.savefig('fig_architecture_v8.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig('fig_architecture_v8.png', bbox_inches='tight', pad_inches=0.02); print('wrote v8')
