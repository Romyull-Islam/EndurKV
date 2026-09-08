#!/usr/bin/env python3
"""Energy-aware control plane, v2: a ring of five blocks (sense, decide, act, run, measure) with
learn and the thermal guard below. Boxes sized to their text; every arrow label sits in free space."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

GRAY='#8f8f8f'; GRAYF='#f3f3f3'; INK='#1a1a1a'; SUB='#333333'
ORANGE='#D55E00'; ORANGEF='#fdeee6'; GREEN='#009E73'; GREENF='#e6f5f0'; HEAT='#7a7a7a'; MEAS='#009E73'
CTRL=(0, (4, 2)); MEA=(0, (5, 2, 1, 2)); HOT=(0, (1, 2))
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(7.2, 3.15), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def block(x, y, w, h, n, title, lines, ec, fc, fs=6.6, fl=5.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.3', fc=fc, ec=ec, lw=1.1, zorder=3))
    ax.text(x + w/2, y + h - 4.6, title, ha='center', va='center', fontsize=fs, fontweight='bold', color=INK, zorder=4)
    ax.plot([x + 2, x + w - 2], [y + h - 8.6, y + h - 8.6], color=ec, lw=0.6, zorder=4)
    ax.text(x + w/2, y + h - 11.0, "\n".join(lines), ha='center', va='top', fontsize=fl, color=SUB, zorder=4, linespacing=1.35)
    if n: ax.annotate(str(n), (x, y + h), ha='center', va='center', fontsize=5.8, color='white', fontweight='bold', zorder=7, bbox=dict(boxstyle='circle,pad=0.25', fc=ec, ec='none'))
def arr(p, q, c, ls='-', lw=1.1, ms=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c, ls=ls, zorder=2, shrinkA=0, shrinkB=0))
def seg(pts, c, ls='-', lw=1.1):
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]): ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2, solid_capstyle='round')
def lab(x, y, t, c=INK, fs=4.8, ha='center', va='center'):
    ax.text(x, y, t, fontsize=fs, ha=ha, va=va, color=c, zorder=6, linespacing=1.2)

ax.add_patch(FancyBboxPatch((0.6, 1.5), 98.8, 97.5, boxstyle='round,pad=0.1,rounding_size=1.4', fc='white', ec=INK, lw=1.1, zorder=0))
ax.text(2.6, 95.0, 'ENERGY-AWARE CONTROL PLANE   once per request, 0.3 s', fontsize=6.2, fontweight='bold', color='#555555', ha='left', va='center', zorder=1)

Y, H = 50, 36
cols = [(3.0, 17.0), (23.5, 17.0), (44.0, 17.0), (64.5, 17.0), (85.0, 12.5)]
block(cols[0][0], Y, cols[0][1], H, 1, 'Sense', ['battery level, charging', 'prompt N, answer length', 'DDR, battery temperature'], GRAY, GRAYF)
block(cols[1][0], Y, cols[1][1], H, 2, 'Decide', ['tier from the battery', 'lever L = tier + bias', 'walk the cost table'], ORANGE, ORANGEF)
block(cols[2][0], Y, cols[2][1], H, 3, 'Act', ['GPU clock caps', '(prefill, decode)', 'cache budget K', 'answer cap'], ORANGE, ORANGEF)
block(cols[3][0], Y, cols[3][1], H, 4, 'Run', ['prefill, score, select,', 'compact, decode', '(Figure 1)'], GRAY, GRAYF)
block(cols[4][0], Y, cols[4][1], H, 5, 'Measure', ['USB rail +', 'coulomb counter', 'per phase'], ORANGE, ORANGEF)

YL, HL = 8, 30
block(3.0, YL, 48.0, HL, 6, 'Learn', ['cost table: EMA per plan, clipped to 10% per request',
      'time miss: L + 0.1      energy miss: L - 0.1', 'both met: the bias decays'], ORANGE, ORANGEF)
block(58.0, YL, 39.5, HL, None, 'Thermal guard', ['CPU ladder at battery 47 to 49.5 C and skin 50 to 52.5 C',
      'GPU ladder at DDR 60, 62, 63.5 C', 'vendor limiter last; kernel takes the lower cap'], HEAT, GRAYF)

# ring arrows along the top row, labels above the boxes
ym = Y + H/2
for i, (t, c, ls) in enumerate([('state', INK, '-'), ('plan', INK, '-'), ('caps', ORANGE, CTRL), ('trace', MEAS, MEA)]):
    x1 = cols[i][0] + cols[i][1] + 0.1; x2 = cols[i+1][0] - 0.1
    arr((x1, ym), (x2, ym), c, ls=ls); lab((x1 + x2)/2, Y - 2.9, t, c)
# measure -> learn: down the right edge, along the bottom, up into learn
xm = cols[4][0] + cols[4][1]/2
seg([(xm, Y - 0.1), (xm, 43.5), (98.2, 43.5), (98.2, 4.5), (27.0, 4.5)], MEAS, ls=MEA); arr((27.0, 4.5), (27.0, YL - 0.1), MEAS, ls=MEA)
lab(62.0, 6.3, 'energy and time per phase', MEAS, fs=4.6)
# learn -> decide
xl = cols[1][0] + cols[1][1]/2
arr((xl, YL + HL + 0.1), (xl, Y - 0.1), ORANGE, ls=CTRL); lab(xl + 1.2, 44.0, 'bias, table costs', ORANGE, ha='left')
# run <-> thermal guard
xr = cols[3][0] + cols[3][1]/2
arr((xr + 7.0, Y - 0.1), (xr + 7.0, YL + HL + 0.1), HEAT, ls=HOT, lw=1.2); lab(xr + 8.0, 44.0, 'heat', HEAT, ha='left')
arr((xr + 2.0, YL + HL + 0.1), (xr + 2.0, Y - 0.1), HEAT, ls=CTRL); lab(xr + 1.0, 44.0, 'clock cap', HEAT, ha='right')
# legend in the title row
x = 58.0
for c, ls, t in ((INK, '-', 'data'), (ORANGE, CTRL, 'control'), (MEAS, MEA, 'measurement'), (HEAT, HOT, 'heat')):
    ax.plot([x, x + 3.0], [95.0, 95.0], color=c, lw=1.2, ls=ls); ax.text(x + 3.8, 95.0, t, fontsize=5.0, ha='left', va='center'); x += 3.8 + len(t) * 0.64 * 5.0 / 72 / 7.2 * 100 + 2.5
fig.savefig('fig_control_plane_v2.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig('fig_control_plane_v2.png', bbox_inches='tight', pad_inches=0.02, dpi=220); print('ok')
