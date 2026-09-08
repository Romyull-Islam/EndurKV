#!/usr/bin/env python3
"""The energy-aware control plane as its own figure, in the language of the architecture figure:
six numbered blocks in a loop (sense, decide, act, run, measure, learn) plus the thermal guard.
One label per arrow. Data solid, control dashed, measurement dash-dot, heat dotted."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

GRAY='#9a9a9a'; GRAYF='#f2f2f2'; INK='#1a1a1a'
ORANGE='#D55E00'; ORANGEF='#fdeee6'; BLUE='#0072B2'; BLUEF='#eaf3fa'
GREEN='#009E73'; GREENF='#e8f6f1'; HEAT='#777777'; MEAS='#009E73'
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def block(x, y, w, h, n, title, lines, ec, fc, fs=6.6, fl=5.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.6', fc=fc, ec=ec, lw=1.2, zorder=3))
    ax.text(x + w/2, y + h - 4.2, title, ha='center', va='center', fontsize=fs, fontweight='bold', color=INK, zorder=4)
    ax.plot([x + 2, x + w - 2], [y + h - 8.2, y + h - 8.2], color=ec, lw=0.6, zorder=4)
    ax.text(x + 2.2, y + h - 10.8, "\n".join(lines), ha='left', va='top', fontsize=fl, color=INK, zorder=4, linespacing=1.32)
    if n: ax.annotate(str(n), (x, y + h), ha='center', va='center', fontsize=6.2, color='white', fontweight='bold', zorder=7, bbox=dict(boxstyle='circle,pad=0.3', fc=ec, ec='none'))
def arr(p, q, c, ls='-', lw=1.3, ms=10):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c, ls=ls, zorder=2, shrinkA=0.5, shrinkB=0.5))
def seg(pts, c, ls='-', lw=1.3):
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]): ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2, solid_capstyle='round')
def lab(x, y, t, c=INK, fs=5.0, ha='center', va='center', rot=0):
    ax.text(x, y, t, fontsize=fs, ha=ha, va=va, color=c, rotation=rot, zorder=6, linespacing=1.2, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))

ax.add_patch(FancyBboxPatch((0.8, 2), 99, 97, boxstyle='round,pad=0.1,rounding_size=1.5', fc='white', ec=INK, lw=1.2, zorder=0))
ax.text(3.0, 96.3, 'energy-aware control plane, once per request (0.3 s)', fontsize=7.2, fontweight='bold', color=INK, ha='left', va='center', zorder=1)

# ---- the loop: five blocks across the top, learn and thermal below ----
Y, H, W = 52, 38, 17
block(3,  Y, W, H, 1, 'Sense', ['battery level, charging', 'DDR and battery temp.', 'prompt tokens', 'length asked?'], GRAY, GRAYF)
block(23, Y, W, H, 2, 'Decide', ['tier from the battery', 'lever L = tier + bias', 'ladder walk over plans', 'on the cost table'], ORANGE, ORANGEF)
block(43, Y, W, H, 3, 'Act', ['GPU clock caps:', '  prefill, decode', 'cache budget K', 'answer cap'], ORANGE, ORANGEF)
block(63, Y, W, H, 4, 'Run', ['prefill, score, evict,', 'compact, decode', '(the pipeline figure)', 'GPU or CPU'], GRAY, GRAYF)
block(83, Y, 14, H, 5, 'Measure', ['USB rail +', 'coulomb counter', 'per phase; clocks,', 'temperatures'], ORANGE, ORANGEF)
block(3, 8, 49, 30, 6, 'Learn', ['cost table: EMA per plan, clipped to 10% per request', 'performance loop: time over budget → lever +0.1', 'energy loop: energy over prediction → lever −0.1', 'both budgets met → the bias decays to the tier default'], ORANGE, ORANGEF, fl=5.0)
block(62, 8, 35, 30, None, 'Thermal guard', ['CPU: clock ladder at battery and skin temperatures', 'GPU: cap ladder at DDR 60 / 62 / 63.5 C, measured:', '  peak 75 C not 90, 7% less energy, same time', 'vendor limiter last (GPU 726 MHz); ours sit below'], HEAT, GRAYF, fl=4.9)

# ---- arrows along the top row ----
arr((20.2, Y + 19), (22.8, Y + 19), INK); arr((40.2, Y + 19), (42.8, Y + 19), INK); arr((60.2, Y + 19), (62.8, Y + 19), ORANGE, ls=(0, (4, 2)))
arr((80.2, Y + 19), (82.8, Y + 19), MEAS, ls=(0, (5, 2, 1, 2)))
lab(21.5, Y + 23.5, 'state', fs=4.8); lab(41.5, Y + 23.5, 'plan', fs=4.8); lab(61.5, Y + 23.5, 'caps, args', ORANGE, 4.8); lab(81.5, Y + 23.5, 'trace', MEAS, 4.8)
# measure -> learn, around the right edge and along the bottom
seg([(90, Y - 0.4), (90, 41), (98.6, 41)], MEAS, ls=(0, (5, 2, 1, 2)))
seg([(98.6, 41), (98.6, 4.8), (27, 4.8)], MEAS, ls=(0, (5, 2, 1, 2))); arr((27, 4.8), (27, 7.6), MEAS, ls=(0, (5, 2, 1, 2)))
lab(62, 6.3, 'energy and time per phase, against the prediction', MEAS, 4.6)
# learn -> decide
arr((31.5, 38.4), (31.5, Y - 0.4), ORANGE, ls=(0, (4, 2))); lab(31.5, 45, 'bias per tier, table costs', ORANGE, 4.8)
# heat down from run; cap up into run
arr((75, Y - 0.4), (75, 38.4), HEAT, ls=(0, (1, 2))); lab(78.5, 45, 'heat', HEAT, 4.8)
arr((68, 38.4), (68, Y - 0.4), HEAT, ls=(0, (4, 2)), lw=1.1); lab(63.5, 45, 'clock cap', HEAT, 4.8)

# ---- legend, in the title row ----
ly = 96.3
for x, c, ls, t in ((60, INK, '-', 'data'), (67, ORANGE, (0, (4, 2)), 'control'), (76, MEAS, (0, (5, 2, 1, 2)), 'measurement'), (89, HEAT, (0, (1, 2)), 'heat')):
    ax.plot([x, x + 2.6], [ly, ly], color=c, lw=1.2, ls=ls); ax.text(x + 3.2, ly, t, fontsize=4.8, ha='left', va='center')
fig.savefig('fig_control_plane.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig('fig_control_plane.png', bbox_inches='tight', pad_inches=0.02); print('wrote control plane')
