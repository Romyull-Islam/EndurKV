#!/usr/bin/env python3
"""muKV architecture, v7 = v6 plus the energy-aware layer as a third band. Conventions kept from v6:
left-to-right dataflow inside the OnePlus 15 boundary; the KV cache drawn as a cell strip (N -> K);
every arrow labeled with its payload; arrow styles: data solid, control dashed, heat dotted,
measurement dash-dot. Orange badges number the muKV mechanisms in order of execution (1-8).
Colors (Okabe-Ito): orange = new muKV mechanism, gray = unmodified llama.cpp / hardware,
green = muKV outcome, blue = KV-cache data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

GRAY='#9a9a9a'; GRAYF='#f2f2f2'; INK='#1a1a1a'
ORANGE='#D55E00'; ORANGEF='#fdeee6'; BLUE='#0072B2'; BLUEF='#eaf3fa'
GREEN='#009E73'; GREENF='#e8f6f1'; HEAT='#777777'; MEAS='#009E73'
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(7.2, 5.3), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x, y, w, h, title, sub='', ec=GRAY, fc='white', fs=6.6, fsub=5.2, lw=1.1, tdy=2.4, sdy=-2.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.4', fc=fc, ec=ec, lw=lw, zorder=3))
    dy = tdy if sub else 0
    ax.text(x+w/2, y+h/2+dy, title, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold', zorder=4)
    if sub: ax.text(x+w/2, y+h/2+sdy, sub, ha='center', va='center', fontsize=fsub, color=INK, zorder=4, linespacing=1.25)
def chip(x, y, w, h, text, fs=4.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=0.7', fc=ORANGEF, ec=ORANGE, lw=0.8, zorder=5))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, color=ORANGE, zorder=6)
def arr(x1, y1, x2, y2, c=INK, lw=1.2, ls='-', ms=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c, ls=ls, zorder=2, shrinkA=0.5, shrinkB=0.5))
def seg(pts, c, lw=1.2, ls='-'):
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2, solid_capstyle='round')
def step(x, y, n):
    ax.annotate(str(n), (x, y), ha='center', va='center', fontsize=6.0, color='white', fontweight='bold', zorder=7, bbox=dict(boxstyle='circle,pad=0.28', fc=ORANGE, ec='none'))
def lab(x, y, t, c=INK, fs=5.0, ha='center', va='center', rot=0, bold=False):
    ax.text(x, y, t, fontsize=fs, ha=ha, va=va, color=c, rotation=rot, fontweight='bold' if bold else 'normal', zorder=6, linespacing=1.2)

# ================= device boundary + three bands =================
ax.add_patch(FancyBboxPatch((0.8, 4.5), 98.4, 94.5, boxstyle='round,pad=0.1,rounding_size=1.5', fc='white', ec=INK, lw=1.2, zorder=0))
ax.text(3.0, 97.0, 'OnePlus 15', fontsize=7.5, fontweight='bold', color=INK, ha='left', va='center', zorder=1)
for (y, h, t) in ((48.5, 46.5, 'llama.cpp inference (CPU / Adreno GPU)'), (28.5, 17.5, 'SoC sensors + DVFS'), (6.0, 20.0, 'energy-aware scheduler, once per request (0.3 s)')):
    ax.add_patch(FancyBboxPatch((2.5, y), 95, h, boxstyle='round,pad=0.1,rounding_size=1.2', fc='#fcfcfc', ec='#bbbbbb', lw=0.9, zorder=1))
    ax.text(4.0, y + h - 2.4, t, fontsize=6.0, style='italic', color='#555555', ha='left', va='center', zorder=2)

# ================= band A: pipeline row =================
y0, h0 = 72, 18
box(4,  y0, 8,  h0, 'Prompt', 'N tokens', ec=GRAY, fc=GRAYF)
box(17, y0, 14, h0, 'FA-on prefill', 'llama.cpp graph', ec=GRAY, fc=GRAYF, tdy=4.6, sdy=1.4)
chip(18.2, y0 + 1.3, 11.6, 3.6, 'clock cap from plan', 4.4)
box(40, y0, 15, h0, 'Score + select', 'budget gate ·\naggregate · α-gate', ec=ORANGE, fc=ORANGEF, fsub=5.0, sdy=-3.4)
box(62, y0, 12, h0, 'Evict + compact', 'seq-level ·\nin-place slide', ec=ORANGE, fc=ORANGEF, fsub=5.0, tdy=5.0, sdy=1.3)
chip(63.0, y0 + 1.3, 10.0, 3.6, 'K from plan', 4.4)
ax.add_patch(FancyBboxPatch((79, y0), 17, h0, boxstyle='round,pad=0.4,rounding_size=1.4', fc=GREENF, ec=GREEN, lw=1.1, zorder=3))
lab(87.5, y0 + 14.6, 'FA-on decode', fs=6.6, bold=True); lab(87.5, y0 + 11.4, 'bounded cache', fs=5.2)
chip(80.2, y0 + 5.6, 14.6, 3.4, 'keep-set frozen · window slides', 3.9)
chip(80.2, y0 + 1.3, 14.6, 3.4, 'decode cap, max tokens from plan', 3.9)
step(40, y0 + h0, 1); step(62, y0 + h0, 2); step(79, y0 + h0, 3)

arr(12.2, y0 + 9, 16.8, y0 + 9, c=INK); lab(14.5, y0 + 11.6, 'tokens')
arr(31.2, y0 + 12.5, 39.8, y0 + 12.5, c=ORANGE, lw=1.1); ax.plot([31.2], [y0 + 12.5], marker='o', ms=3.4, color=ORANGE, zorder=6)
lab(35.4, y0 + 15.0, 'kq_evict', ORANGE, 5.2, bold=True); lab(35.5, y0 + 10.0, 'K,Q (last chunk)', ORANGE, 4.7)
arr(55.2, y0 + 9, 61.8, y0 + 9, c=ORANGE, lw=1.1); lab(58.4, y0 + 11.5, 'keep set', ORANGE, 4.8); lab(58.4, y0 + 6.6, '(K indices)', ORANGE, 4.8)

# ================= band A: KV cache strips =================
ys = 55.5
for i in range(18): ax.add_patch(Rectangle((8 + i*2.0, ys), 1.85, 5, fc=BLUEF, ec=BLUE, lw=0.5, zorder=3))
lab(26, ys - 3.0, 'full KV cache · N cells', fs=5.2)
arr(24, y0 - 0.4, 24, ys + 5.7, c=BLUE, lw=1.2); lab(25.3, 65.0, 'write full KV (all N)', BLUE, 5.0, ha='left')
arr(45, ys + 2.5, 60.7, ys + 2.5, c=BLUE, lw=2.0, ms=15); lab(53, ys + 5.8, 'N → K', fs=5.3, bold=True); lab(53, ys - 1.4, 'evict + compact', fs=5.0)
segs = [(2, '#c7dcef', 'sink'), (4, '#9ec6e8', 'anchor'), (3, '#dcebf7', 'recent')]
xk = 62
for n, fc, name in segs:
    for i in range(n): ax.add_patch(Rectangle((xk + i*1.55, ys), 1.45, 5, fc=fc, ec=BLUE, lw=0.5, zorder=3))
    lab(xk + n*1.55/2, ys - 2.1, name, fs=4.8); xk += n*1.55
lab(69, ys - 5.0, 'K cells · contiguous', fs=5.2)
arr(68, y0 - 0.4, 68, ys + 5.7, c=ORANGE, lw=1.1, ls=(0, (4, 2))); lab(69.3, 65.0, 'slide in place (4.8 ms)', ORANGE, 5.0, ha='left')
seg([(76.4, ys + 2.5), (86, ys + 2.5)], BLUE); arr(86, ys + 2.5, 86, y0 - 0.4, c=BLUE, lw=1.2); lab(81.2, ys + 0.1, 'read K cells', BLUE, 5.0)

# ================= band B: sensors -> watchdog -> clock =================
yb, hb = 31.5, 11
box(6,  yb, 20, hb, 'Sensors', 'skin · battery · DDR · CPU\nbattery level, charging', ec=GRAY, fc=GRAYF, fsub=4.8, sdy=-3.0)
box(34, yb, 22, hb, 'Watchdog (CPU)', 'battery and skin ladders,\none step before the vendor', ec=ORANGE, fc=ORANGEF, fsub=4.8, sdy=-3.0)
box(72, yb, 22, hb, 'CPU / GPU clock', 'DVFS caps: CPU from the watchdog,\nGPU from the plan, vendor by max()', ec=GRAY, fc=GRAYF, fsub=4.6, sdy=-3.0)
step(34, yb + hb, 4)
arr(26.2, yb + 5.5, 33.8, yb + 5.5, c=INK); lab(30, yb + 8.4, r'$T$ @ 2 Hz', fs=5.0)
arr(56.2, yb + 5.5, 71.8, yb + 5.5, c=ORANGE, lw=1.2, ls=(0, (4, 2))); lab(64, yb + 8.4, 'step down early,', ORANGE, 4.8); lab(64, yb + 2.8, 'debounced restore', ORANGE, 4.8)
arr(90, yb + hb + 0.4, 90, y0 - 0.4, c=GRAY, lw=1.1, ls=(0, (4, 2))); lab(91.3, 58, r'$f \leq$ cap', '#555555', 5.0, ha='left', rot=90)
arr(24, 48.6, 24, yb + hb + 0.4, c=HEAT, lw=1.2, ls=(0, (1, 2))); lab(25.5, 46.0, 'heat', HEAT, 5.0, ha='left')

# ================= band C: the energy-aware scheduler =================
yc, hc = 9.5, 11
box(3.5, yc, 16.5, hc, 'Battery + request', 'level, charging;\nprompt N; length asked?', ec=GRAY, fc=GRAYF, fsub=4.6, sdy=-3.0)
box(22, yc, 14, hc, 'Tier → lever L', '1 · 0.5 · 0 by battery,\nplus the loops\' bias', ec=ORANGE, fc=ORANGEF, fsub=4.6, sdy=-3.0)
box(39, yc, 19, hc, 'Ladder walk', 'cost table (measured, learned):\nE, T per plan; exchange rate,\ntime budget, quality floor', ec=ORANGE, fc=ORANGEF, fsub=4.3, tdy=3.4, sdy=-1.8)
box(61, yc, 11, hc, 'Plan', 'GPU/CPU · clock caps\nK · output cap', ec=GREEN, fc=GREENF, fsub=4.5, sdy=-3.0)
box(75, yc, 21, hc, 'Meter → two loops', 'rail + coulombs, per phase\ntime over budget: L +0.1\nenergy over prediction: L −0.1', ec=ORANGE, fc=ORANGEF, fsub=4.4, tdy=3.4, sdy=-1.8)
step(22, yc + hc, 5); step(39, yc + hc, 6); step(75, yc + hc, 7)
arr(20.2, yc + 5.5, 21.8, yc + 5.5, c=INK); arr(36.2, yc + 5.5, 38.8, yc + 5.5, c=INK); arr(58.2, yc + 5.5, 60.8, yc + 5.5, c=INK)
# plan -> DVFS caps (control), plan -> engine args (chips above)
seg([(66.5, yc + hc + 0.4), (66.5, 26.3), (83, 26.3)], ORANGE, 1.2, (0, (4, 2))); arr(83, 26.3, 83, yb - 0.4, c=ORANGE, lw=1.2, ls=(0, (4, 2)))
lab(84.2, 27.4, 'prefill cap, decode cap', ORANGE, 4.6, ha='left', va='bottom')
lab(4.5, 26.3, 'K and max tokens reach the engine as arguments (the orange chips in the boxes above)', ORANGE, 4.4, ha='left')
# measurement: the sampler watches the whole run
seg([(95, 48.6), (95, yc + hc + 0.4)], MEAS, 1.1, (0, (5, 2, 1, 2))); ax.plot([95], [yc + hc + 0.4], marker='v', ms=3.5, color=MEAS, zorder=6)
lab(96.3, 36, 'rail, coulombs, clocks', MEAS, 4.8, ha='left', rot=90)
# loops -> lever bias and table update
seg([(85.5, yc - 0.4), (85.5, 7.0), (29, 7.0), (29, yc - 0.4)], ORANGE, 1.1, (0, (4, 2))); ax.plot([29], [yc - 0.4], marker='^', ms=3.5, color=ORANGE, zorder=6)
lab(57, 7.25, 'bias per tier · table update (EMA, clipped to 10% per request)', ORANGE, 4.3, va='bottom')

# ================= legend =================
ly = 2.2
for x, ec, fc, t in ((3, ORANGE, ORANGEF, 'new μKV mechanism'), (19, GRAY, GRAYF, 'unmodified llama.cpp / hardware'), (40, GREEN, GREENF, 'μKV outcome'), (53, BLUE, BLUEF, 'KV cache')):
    ax.add_patch(FancyBboxPatch((x, ly - 0.9), 3.2, 1.8, boxstyle='round,pad=0.1,rounding_size=0.5', fc=fc, ec=ec, lw=0.9)); lab(x + 4.0, ly, t, fs=4.8, ha='left')
for x, c, ls, t in ((64, INK, '-', 'data'), (71.5, ORANGE, (0, (4, 2)), 'control'), (81, HEAT, (0, (1, 2)), 'heat'), (88.5, MEAS, (0, (5, 2, 1, 2)), 'measure')):
    ax.plot([x, x + 2.6], [ly, ly], color=c, lw=1.2, ls=ls); lab(x + 3.2, ly, t, fs=4.8, ha='left')

fig.savefig('fig_architecture_v7.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig('fig_architecture_v7.png', bbox_inches='tight', pad_inches=0.02); print('wrote fig_architecture_v7')
