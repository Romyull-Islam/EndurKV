#!/usr/bin/env python3
"""muKV architecture for the 6-page HotMobile paper, in the v9 style.
Top band: the inference pass with the KV memory strip. Bottom band: the thermal control chain only,
because the short paper carries no scheduler. Laid out in one row so the type can be set large
enough to read at print size (target: nothing below about 6.8 pt at 0.96 textwidth)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

GRAY='#8f8f8f'; GRAYF='#f3f3f3'; INK='#1a1a1a'; SUB='#333333'
ORANGE='#D55E00'; ORANGEF='#fdeee6'; BLUE='#0072B2'; BLUEF='#e6f1f9'
GREEN='#009E73'; GREENF='#e6f5f0'; HEAT='#7a7a7a'; MEAS='#009E73'
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
FW, FH = 7.2, 3.32
fig, ax = plt.subplots(figsize=(FW, FH), dpi=200)
ax.set_position([0, 0, 1, 1])   # axes fill the figure, so 1 x-unit = FW/100 in and 1 y-unit = FH/100 in
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x, y, w, h, title, sub='', ec=GRAY, fc='white', fs=9.0, fsub=7.6, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec=ec, lw=lw, zorder=3))
    if sub:
        ax.text(x+w/2, y+h/2+3.6, title, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold', zorder=4)
        ax.text(x+w/2, y+h/2-3.8, sub, ha='center', va='center', fontsize=fsub, color=SUB, zorder=4, linespacing=1.3)
    else:
        ax.text(x+w/2, y+h/2, title, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold', zorder=4)
def arr(x1, y1, x2, y2, c=INK, lw=1.1, ls='-', ms=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c, ls=ls, zorder=2, shrinkA=0, shrinkB=0))
def path(pts, c, lw=1.1, ls='-', ms=9):
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]): ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2, solid_capstyle='round')
    arr(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], c=c, lw=lw, ls=ls, ms=ms)
def step(x, y, n):
    ax.annotate(str(n), (x, y), ha='center', va='center', fontsize=7.3, color='white', fontweight='bold', zorder=7,
                bbox=dict(boxstyle='circle,pad=0.22', fc=ORANGE, ec='none'))
def lab(x, y, t, c=INK, fs=7.3, ha='center', va='center', bold=False, bg=False):
    ax.text(x, y, t, fontsize=fs, ha=ha, va=va, color=c, fontweight='bold' if bold else 'normal', zorder=6, linespacing=1.2,
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none') if bg else None)

# ---------------- device frame and the two bands ----------------
ax.add_patch(FancyBboxPatch((0.6, 3.0), 98.8, 94.0, boxstyle='round,pad=0.1,rounding_size=1.4', fc='white', ec=INK, lw=1.1, zorder=0))
ax.text(2.6, 95.4, 'OnePlus 15', fontsize=8.4, fontweight='bold', color=INK, ha='left', va='center', zorder=1)
ax.add_patch(FancyBboxPatch((2.2, 48.0), 95.6, 44.0, boxstyle='round,pad=0.1,rounding_size=1.0', fc='#fcfcfc', ec='#c4c4c4', lw=0.8, zorder=1))
ax.text(3.6, 90.0, 'INFERENCE  (llama.cpp, CPU or Adreno GPU, one request)', fontsize=7.6, fontweight='bold', color='#555555', ha='left', va='center', zorder=2)
ax.add_patch(FancyBboxPatch((2.2, 13.0), 95.6, 31.0, boxstyle='round,pad=0.1,rounding_size=1.0', fc='#fcfcfc', ec='#c4c4c4', lw=0.8, zorder=1))
ax.text(3.6, 41.6, 'THERMAL CONTROL  (continuous, 2 Hz, reduce only)', fontsize=7.6, fontweight='bold', color='#555555', ha='left', va='center', zorder=2)

# ---------------- band A: the pass ----------------
y0, h0 = 66.5, 19.0
bx = [(4.5, 11.0, 'Prompt', 'N tokens', GRAY, GRAYF), (18.1, 13.0, 'Prefill', 'FA on,\nunmodified', GRAY, GRAYF),
      (33.7, 14.0, 'Score', 'kq_evict\nside node', ORANGE, ORANGEF), (50.3, 13.0, 'Select', 'one keep\nset of K', ORANGE, ORANGEF),
      (65.9, 14.0, 'Compact', 'slide\nin place', ORANGE, ORANGEF), (82.5, 13.0, 'Decode', 'FA on,\nK cells', GREEN, GREENF)]
for x, w, t, s, ec, fc in bx: box(x, y0, w, h0, t, s, ec=ec, fc=fc)
for i, (x, w, *_) in enumerate(bx[:-1]):
    arr(x + w + 0.2, y0 + h0/2, bx[i+1][0] - 0.2, y0 + h0/2, c=ORANGE if i >= 1 else INK)
for n, x in zip((1, 2, 3), (33.7, 50.3, 65.9)): step(x, y0 + h0 + 0.3, n)

# KV memory strip, drawn to the measured K=1024 split: 4 sinks, 721 anchors, 299 recent
ys, hs = 52.5, 6.4
ax.text(4.5, ys + hs/2, 'KV cache', fontsize=7.6, fontweight='bold', color=BLUE, ha='left', va='center', zorder=4)
for i in range(16): ax.add_patch(Rectangle((18.1 + i*1.8, ys), 1.65, hs, fc=BLUEF, ec=BLUE, lw=0.5, zorder=3))
lab(32.5, ys - 3.6, 'N cells after prefill', BLUE, 7.3)
arr(24.5, y0 - 0.2, 24.5, ys + hs + 0.3, c=BLUE); lab(25.5, y0 - 5.4, 'write N', BLUE, 7.3, ha='left')
arr(48.5, ys + hs/2, 63.5, ys + hs/2, c=BLUE, lw=1.8, ms=13); lab(56.0, ys + hs + 3.4, 'N to K, in place', BLUE, 7.3, bold=True)
for x, w, fc in [(64.5, 0.6, '#9dc3e0'), (65.1, 13.6, '#5ea3d0'), (78.7, 5.6, '#c9dff0')]:
    ax.add_patch(Rectangle((x, ys), w, hs, fc=fc, ec=BLUE, lw=0.6, zorder=3))
lab(74.7, ys - 3.6, '4 sink, 721 anchor, 299 recent', BLUE, 7.3)
arr(71.5, y0 - 0.2, 71.5, ys + hs + 0.3, c=ORANGE, ls='--'); lab(72.5, y0 - 5.4, 'slide', ORANGE, 7.3, ha='left')
ax.plot([84.3, 89.0], [ys + hs/2, ys + hs/2], color=BLUE, lw=1.1, zorder=2)
arr(89.0, ys + hs/2, 89.0, y0 - 0.2, c=BLUE); lab(89.9, y0 - 5.4, 'read K', BLUE, 7.3, ha='left')

# ---------------- band B: the thermal chain, one row ----------------
yb, hb = 20.0, 16.0
box(5.0, yb, 24.0, hb, 'Sensors', 'skin, battery and DDR\nsampled at 2 Hz', ec=GRAY, fc=GRAYF)
box(33.0, yb, 36.0, hb, 'Watchdog', 'CPU 1497 to 1017 MHz, both clusters\nGPU 1050 to 902 MHz, on DDR', ec=ORANGE, fc=ORANGEF)
step(33.0, yb + hb + 0.3, 4)
box(73.0, yb, 22.0, hb, 'Vendor limiter', 'kernel takes\nthe lower cap', ec=GRAY, fc=GRAYF)
arr(29.2, yb + hb/2, 32.8, yb + hb/2, c=INK)
arr(69.2, yb + hb/2, 72.8, yb + hb/2, c=GRAY, ls='--'); lab(71.0, yb - 2.9, 'if the ladders miss', GRAY, 7.3)

# ---------------- links between the bands ----------------
path([(24.0, 47.9), (24.0, yb + hb + 0.4)], HEAT, ls=':', lw=1.2); lab(25.0, 45.9, 'heat', HEAT, 7.3, ha='left')
path([(4.8, yb + hb/2), (3.5, yb + hb/2), (3.5, 48.1)], HEAT, ls='--'); lab(4.4, 45.9, 'clock caps', HEAT, 7.3, ha='left')

# ---------------- legend, one row ----------------
def tw(t): return len(t) * 0.80 * 7.3 / 72 / FW * 100
x = 4.0; ly = 7.5
for ec, fc, t in [(ORANGE, ORANGEF, 'μKV mechanism'), (GRAY, GRAYF, 'unmodified')]:
    ax.add_patch(FancyBboxPatch((x, ly - 1.6), 3.4, 3.2, boxstyle='round,pad=0.1,rounding_size=0.6', fc=fc, ec=ec, lw=1.0))
    ax.text(x + 4.6, ly, t, fontsize=7.3, va='center'); x += 4.6 + tw(t) + 3.0
for c, ls, t in [(INK, '-', 'data'), (ORANGE, '--', 'control'), (HEAT, ':', 'heat')]:
    ax.plot([x, x + 4.0], [ly, ly], color=c, ls=ls, lw=1.2); ax.text(x + 5.0, ly, t, fontsize=7.3, va='center'); x += 5.0 + tw(t) + 3.0

fig.savefig('fig_architecture_6pg_v9.pdf', bbox_inches='tight', pad_inches=0.02)
print('ok')
