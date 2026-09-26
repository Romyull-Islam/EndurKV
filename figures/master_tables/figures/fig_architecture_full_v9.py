#!/usr/bin/env python3
"""muKV architecture in the v9 layout, complete: all nine components, used by BOTH papers.
Top band: the inference pass and the KV memory strip. Bottom band: the control plane in three
columns (thermal, energy, learning). Badges 1 to 9 give execution order.
Two fixes over the original v9: the axes fill the figure (v9 omits set_position, so its x-units are
0.056 in, not the 0.072 that FW/100 implies, and every box is ~29% narrower than its text needs),
and the type is set at 7.3 pt minimum so nothing falls below about 7 pt at print."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

GRAY='#8f8f8f'; GRAYF='#f3f3f3'; INK='#1a1a1a'; SUB='#333333'
ORANGE='#D55E00'; ORANGEF='#fdeee6'; BLUE='#0072B2'; BLUEF='#e6f1f9'
GREEN='#009E73'; GREENF='#e6f5f0'; HEAT='#7a7a7a'; MEAS='#009E73'
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
FW, FH = 7.2, 3.6
fig, ax = plt.subplots(figsize=(FW, FH), dpi=200)
ax.set_position([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x, y, w, h, title, sub='', ec=GRAY, fc='white', fs=9.0, fsub=7.4, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec=ec, lw=lw, zorder=3))
    if sub:
        n = sub.count('\n') + 1
        ax.text(x+w/2, y+h/2+(2.4 if n == 1 else 3.1), title, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold', zorder=4)
        ax.text(x+w/2, y+h/2-(2.4 if n == 1 else 2.9), sub, ha='center', va='center', fontsize=fsub, color=SUB, zorder=4, linespacing=1.22)
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
def lab(x, y, t, c=INK, fs=7.3, ha='center', bold=False, bg=False):
    ax.text(x, y, t, fontsize=fs, ha=ha, va='center', color=c, fontweight='bold' if bold else 'normal', zorder=6, linespacing=1.2,
            bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none') if bg else None)
def head(x, y, t, c):
    ax.text(x, y, t, fontsize=7.6, fontweight='bold', color=c, ha='left', va='center', zorder=2)

# ---------------- device frame and the two bands ----------------
ax.add_patch(FancyBboxPatch((0.6, 0.3), 98.8, 94.9, boxstyle='round,pad=0.1,rounding_size=1.4', fc='white', ec=INK, lw=1.1, zorder=0))
ax.text(2.6, 92.6, 'OnePlus 15', fontsize=8.4, fontweight='bold', color=INK, ha='left', va='center', zorder=1)
ax.add_patch(FancyBboxPatch((2.2, 58.6), 95.6, 31.6, boxstyle='round,pad=0.1,rounding_size=1.0', fc='#fcfcfc', ec='#c4c4c4', lw=0.8, zorder=1))
head(3.6, 87.8, 'INFERENCE  (llama.cpp, CPU or Adreno GPU, one request)', '#555555')
ax.add_patch(FancyBboxPatch((2.2, 6.9), 95.6, 45.9, boxstyle='round,pad=0.1,rounding_size=1.0', fc='#fcfcfc', ec='#c4c4c4', lw=0.8, zorder=1))

# ---------------- band A: the pass ----------------
y0, h0 = 70.8, 14.6
bx = [(4.5, 11.0, 'Prompt', 'N tokens', GRAY, GRAYF), (18.1, 13.0, 'Prefill', 'FA on,\nunmodified', GRAY, GRAYF),
      (33.7, 14.0, 'Score', 'kq_evict\nside node', ORANGE, ORANGEF), (50.3, 13.0, 'Select', 'one keep\nset of K', ORANGE, ORANGEF),
      (65.9, 14.0, 'Compact', 'slide\nin place', ORANGE, ORANGEF), (82.5, 13.0, 'Decode', 'FA on,\nK cells', GREEN, GREENF)]
for x, w, t, s, ec, fc in bx: box(x, y0, w, h0, t, s, ec=ec, fc=fc)
for i, (x, w, *_) in enumerate(bx[:-1]):
    arr(x + w + 0.2, y0 + h0/2, bx[i+1][0] - 0.2, y0 + h0/2, c=ORANGE if i >= 1 else INK)
for n, x in zip((1, 2, 3), (33.7, 50.3, 65.9)): step(x, y0 + h0 + 0.3, n)

# KV memory strip, drawn to the measured K=1024 split: 4 sinks, 721 anchors, 299 recent
ys, hs = 62.4, 5.6
ax.text(4.5, ys + hs/2, 'KV cache', fontsize=7.6, fontweight='bold', color=BLUE, ha='left', va='center', zorder=4)
for i in range(16): ax.add_patch(Rectangle((18.1 + i*1.8, ys), 1.65, hs, fc=BLUEF, ec=BLUE, lw=0.5, zorder=3))
lab(32.5, 60.2, 'N cells after prefill', BLUE)
arr(24.5, y0 - 0.2, 24.5, ys + hs + 0.3, c=BLUE); lab(25.5, 69.4, 'write N', BLUE, ha='left')
arr(48.5, ys + hs/2, 63.5, ys + hs/2, c=BLUE, lw=1.8, ms=13); lab(52.0, 69.4, 'N to K, in place', BLUE, bold=True)
for x, w, fc in [(64.5, 0.6, '#9dc3e0'), (65.1, 13.6, '#5ea3d0'), (78.7, 5.6, '#c9dff0')]:
    ax.add_patch(Rectangle((x, ys), w, hs, fc=fc, ec=BLUE, lw=0.6, zorder=3))
lab(74.7, 60.2, '4 sink, 721 anchor, 299 recent', BLUE)
arr(71.5, y0 - 0.2, 71.5, ys + hs + 0.3, c=ORANGE, ls='--'); lab(72.5, 69.4, 'slide', ORANGE, ha='left')
ax.plot([84.3, 89.0], [ys + hs/2, ys + hs/2], color=BLUE, lw=1.1, zorder=2)
arr(89.0, ys + hs/2, 89.0, y0 - 0.2, c=BLUE); lab(89.9, 69.4, 'read K', BLUE, ha='left')

# ---------------- band B: three control columns ----------------
cA, cB, cC = (5.0, 27.0), (37.0, 28.0), (70.0, 27.0)
hb = 11.8
r1, r2, r3 = 35.4, 21.3, 7.2
head(cA[0], 51.0, 'THERMAL   continuous, 2 Hz', HEAT)
head(cB[0], 51.0, 'ENERGY   once per request, 0.3 s', ORANGE)
head(cC[0] + 3.4, 51.0, 'LEARN   after the request', MEAS)

box(cA[0], r1, cA[1], hb, 'Sensors', 'skin, battery, DDR, CPU', ec=GRAY, fc=GRAYF)
box(cA[0], r2, cA[1], hb, 'Watchdog', 'CPU 1497 to 1017 MHz\nGPU 1050 to 902 MHz', ec=ORANGE, fc=ORANGEF); step(cA[0], r2 + hb + 0.3, 4)
box(cA[0], r3, cA[1], hb, 'Vendor limiter', 'if a ladder misses,\nthe kernel takes the lower cap', ec=GRAY, fc=GRAYF)
arr(cA[0] + cA[1]/2, r1 - 0.2, cA[0] + cA[1]/2, r2 + hb + 0.3, c=INK)
arr(cA[0] + cA[1]/2, r2 - 0.2, cA[0] + cA[1]/2, r3 + hb + 0.3, c=ORANGE, ls='--')

box(cB[0], r1, cB[1], hb, 'Battery state', 'level, charging, prompt N', ec=GRAY, fc=GRAYF)
box(cB[0], r2, cB[1], hb, 'Tier to lever L', 'mains or above 50%: 1\n21 to 50%: 0.5;  20% or below: 0', ec=ORANGE, fc=ORANGEF); step(cB[0], r2 + hb + 0.3, 5)
box(cB[0], r3, cB[1], hb, 'Plan', 'from L and time slack\nGPU clocks, K, answer cap', ec=GREEN, fc=GREENF); step(cB[0], r3 + hb + 0.3, 6)
arr(cB[0] + cB[1]/2, r1 - 0.2, cB[0] + cB[1]/2, r2 + hb + 0.3, c=INK)
arr(cB[0] + cB[1]/2, r2 - 0.2, cB[0] + cB[1]/2, r3 + hb + 0.3, c=ORANGE, ls='--')

box(cC[0], r1, cC[1], hb, 'Meter', 'USB rail + coulomb\ncounter, per phase', ec=ORANGE, fc=ORANGEF); step(cC[0], r1 + hb + 0.3, 7)
box(cC[0], r2, cC[1], hb, 'Two loops', 'time miss: bias + 0.1\nenergy miss: bias - 0.1', ec=ORANGE, fc=ORANGEF); step(cC[0], r2 + hb + 0.3, 8)
box(cC[0], r3, cC[1], hb, 'Cost table', 'EMA per plan,\nclipped to 10%', ec=ORANGE, fc=ORANGEF); step(cC[0], r3 + hb + 0.3, 9)
arr(cC[0] + cC[1]/2, r1 - 0.2, cC[0] + cC[1]/2, r2 + hb + 0.3, c=INK)
arr(cC[0] + cC[1]/2, r2 - 0.2, cC[0] + cC[1]/2, r3 + hb + 0.3, c=INK)
arr(cC[0] - 0.2, r2 + hb/2, cB[0] + cB[1] + 0.3, r2 + hb/2, c=ORANGE, ls='--'); lab((cC[0] + cB[0] + cB[1])/2, r2 + hb/2 + 2.6, 'bias', ORANGE)
arr(cC[0] - 0.2, r3 + hb/2, cB[0] + cB[1] + 0.3, r3 + hb/2, c=MEAS, ls='-.'); lab((cC[0] + cB[0] + cB[1])/2, r3 + hb/2 + 2.6, 'costs', MEAS)

# ---------------- links between the bands ----------------
path([(30.8, 58.5), (30.8, r1 + hb + 0.4)], HEAT, ls=':', lw=1.2); lab(31.6, 55.6, 'heat', HEAT, ha='left')
path([(cA[0] - 0.2, r2 + hb/2), (3.4, r2 + hb/2), (3.4, 58.4)], ORANGE, ls='--'); lab(4.3, 55.6, 'clock caps', ORANGE, ha='left')
path([(cB[0] - 0.2, r3 + hb/2), (35.4, r3 + hb/2), (35.4, 58.4)], ORANGE, ls='--'); lab(36.3, 55.6, 'plan: clocks, K, answer cap', ORANGE, ha='left')
path([(95.2, 58.5), (95.2, r1 + hb + 0.4)], MEAS, ls='-.', lw=1.2); lab(94.3, 55.6, 'energy, time', MEAS, ha='right')

# ---------------- legend, two rows so it stays inside the frame ----------------
def tw(t): return len(t) * 0.80 * 7.3 / 72 / FW * 100
x = 4.0; ly = 4.4
for ec, fc, t in [(ORANGE, ORANGEF, 'μKV mechanism'), (GRAY, GRAYF, 'unmodified'), (GREEN, GREENF, 'outcome')]:
    ax.add_patch(FancyBboxPatch((x, ly - 1.5), 3.4, 3.0, boxstyle='round,pad=0.1,rounding_size=0.6', fc=fc, ec=ec, lw=1.0))
    ax.text(x + 4.6, ly, t, fontsize=7.3, va='center'); x += 4.6 + tw(t) + 3.0
x = 4.0; ly = 1.7
for c, ls, t in [(INK, '-', 'data'), (ORANGE, '--', 'control'), (HEAT, ':', 'heat'), (MEAS, '-.', 'measure')]:
    ax.plot([x, x + 4.0], [ly, ly], color=c, ls=ls, lw=1.2); ax.text(x + 5.0, ly, t, fontsize=7.3, va='center'); x += 5.0 + tw(t) + 3.0

fig.savefig('fig_architecture_full_v9.pdf', bbox_inches='tight', pad_inches=0.02)
print('ok; legend row 2 ends at', round(x,1), 'of 100')
