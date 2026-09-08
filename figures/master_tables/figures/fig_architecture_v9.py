#!/usr/bin/env python3
"""muKV architecture, v9: box-first, minimal text. Top band: the inference pass with the KV memory
strip. Bottom band: three control columns (thermal, energy, learning). Upward control arrows run in
dedicated channels so no line crosses a box. Every box: a bold title and at most one short line.
Badges 1 to 9 give execution order. Colors (Okabe-Ito): orange = muKV mechanism, gray = unmodified
llama.cpp or hardware, green = outcome, blue = KV cache. Arrows: data solid, control dashed,
heat dotted, measurement dash-dot."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

GRAY='#8f8f8f'; GRAYF='#f3f3f3'; INK='#1a1a1a'; SUB='#333333'
ORANGE='#D55E00'; ORANGEF='#fdeee6'; BLUE='#0072B2'; BLUEF='#e6f1f9'
GREEN='#009E73'; GREENF='#e6f5f0'; HEAT='#7a7a7a'; MEAS='#009E73'
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(7.2, 4.55), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x, y, w, h, title, sub='', ec=GRAY, fc='white', fs=6.6, fsub=5.1, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec=ec, lw=lw, zorder=3))
    if sub:
        ax.text(x+w/2, y+h/2+2.3, title, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold', zorder=4)
        ax.text(x+w/2, y+h/2-2.3, sub, ha='center', va='center', fontsize=fsub, color=SUB, zorder=4, linespacing=1.25)
    else:
        ax.text(x+w/2, y+h/2, title, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold', zorder=4)
def arr(x1, y1, x2, y2, c=INK, lw=1.1, ls='-', ms=8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c, ls=ls, zorder=2, shrinkA=0, shrinkB=0))
def path(pts, c, lw=1.1, ls='-', ms=8):
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]): ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2, solid_capstyle='round')
    arr(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], c=c, lw=lw, ls=ls, ms=ms)
def step(x, y, n):
    ax.annotate(str(n), (x, y), ha='center', va='center', fontsize=5.8, color='white', fontweight='bold', zorder=7, bbox=dict(boxstyle='circle,pad=0.25', fc=ORANGE, ec='none'))
def lab(x, y, t, c=INK, fs=5.0, ha='center', va='center', rot=0, bold=False, bg=True):
    ax.text(x, y, t, fontsize=fs, ha=ha, va=va, color=c, rotation=rot, fontweight='bold' if bold else 'normal', zorder=6, linespacing=1.2,
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none') if bg else None)
def head(x, y, t, c):
    ax.text(x, y, t, fontsize=5.6, fontweight='bold', color=c, ha='left', va='center', zorder=2)

# ---------------- device and bands ----------------
ax.add_patch(FancyBboxPatch((0.6, 5.0), 98.8, 94.0, boxstyle='round,pad=0.1,rounding_size=1.4', fc='white', ec=INK, lw=1.1, zorder=0))
ax.text(2.6, 96.9, 'OnePlus 15', fontsize=7.2, fontweight='bold', color=INK, ha='left', va='center', zorder=1)
ax.add_patch(FancyBboxPatch((2.2, 59.5), 95.6, 35.5, boxstyle='round,pad=0.1,rounding_size=1.0', fc='#fcfcfc', ec='#c4c4c4', lw=0.8, zorder=1))
ax.text(3.6, 92.9, 'INFERENCE  (llama.cpp, CPU or Adreno GPU, one request)', fontsize=5.6, fontweight='bold', color='#555555', ha='left', va='center', zorder=2)
ax.add_patch(FancyBboxPatch((2.2, 7.5), 95.6, 45.5, boxstyle='round,pad=0.1,rounding_size=1.0', fc='#fcfcfc', ec='#c4c4c4', lw=0.8, zorder=1))
ax.text(3.6, 51.4, 'CONTROL PLANE', fontsize=5.6, fontweight='bold', color='#555555', ha='left', va='center', zorder=2)

# ---------------- band A: the pass ----------------
y0, h0 = 78.5, 11.5
bx = [(4.5, 7.0, 'Prompt', 'N tokens', GRAY, GRAYF), (16.5, 12.0, 'Prefill', 'FA on, unmodified', GRAY, GRAYF),
      (33.5, 12.5, 'Score', 'kq_evict side node', ORANGE, ORANGEF), (51.0, 12.0, 'Select', 'one keep set of K', ORANGE, ORANGEF),
      (68.0, 12.0, 'Compact', 'slide in place', ORANGE, ORANGEF), (85.0, 12.0, 'Decode', 'FA on, K cells', GREEN, GREENF)]
for x, w, t, s, ec, fc in bx: box(x, y0, w, h0, t, s, ec=ec, fc=fc)
for i, (x, w, *_ ) in enumerate(bx[:-1]):
    nx = bx[i+1][0]; c = ORANGE if i >= 1 else INK
    arr(x + w + 0.1, y0 + h0/2, nx - 0.1, y0 + h0/2, c=c)
for n, x in zip((1, 2, 3), (33.5, 51.0, 68.0)): step(x, y0 + h0 + 0.2, n)
pass  # arrow labels dropped: the box titles name what flows

# KV memory strip
ys, hs = 63.5, 4.4
ax.text(4.5, ys + hs/2, 'KV cache', fontsize=5.6, fontweight='bold', color=BLUE, ha='left', va='center', zorder=4)
for i in range(16): ax.add_patch(Rectangle((16.5 + i*1.8, ys), 1.65, hs, fc=BLUEF, ec=BLUE, lw=0.5, zorder=3))
lab(30.9, ys - 2.2, 'N cells after prefill', BLUE, 4.8, bg=False)
arr(22.5, y0 - 0.1, 22.5, ys + hs + 0.2, c=BLUE); lab(23.4, y0 - 4.6, 'write N', BLUE, 4.6, ha='left', bg=False)
arr(46.4, ys + hs/2, 65.5, ys + hs/2, c=BLUE, lw=1.8, ms=13); lab(56.0, ys + hs + 2.0, 'N to K, in place', BLUE, 5.0, bold=True, bg=False)
# proportional to the measured K=1024 split on Llama-3.2-1B: 4 sinks, 721 anchors, 299 recent (Table 1 cell)
segs = [(66.3, 0.5, '#9dc3e0', 'sink 4'), (66.8, 10.9, '#5ea3d0', 'anchor 721'), (77.7, 4.6, '#c9dff0', 'recent 299')]
for x, w, fc, t in segs:
    ax.add_patch(Rectangle((x, ys), w, hs, fc=fc, ec=BLUE, lw=0.6, zorder=3))
    if w > 2: lab(x + w/2, ys - 2.2, t, BLUE, 4.6, bg=False)
    else: lab(x + 0.2, ys + hs + 1.7, t, BLUE, 4.6, ha='left', bg=False)
arr(74.0, y0 - 0.1, 74.0, ys + hs + 0.2, c=ORANGE, ls='--'); lab(74.9, y0 - 4.6, 'slide', ORANGE, 4.6, ha='left', bg=False)
ax.plot([81.9, 91.0], [ys + hs/2, ys + hs/2], color=BLUE, lw=1.1, zorder=2); arr(91.0, ys + hs/2, 91.0, y0 - 0.1, c=BLUE); lab(91.9, y0 - 4.6, 'read K', BLUE, 4.6, ha='left', bg=False)

# ---------------- band B: three control columns ----------------
cA, cB, cC = (6.0, 24.0), (37.0, 26.0), (70.0, 26.0)   # (x, w)
hb = 9.2
yr1, yr2, yr3 = 35.3, 22.3, 9.3                        # three rows
head(cA[0], 48.1, 'THERMAL   continuous, 2 Hz', HEAT); head(cB[0], 48.1, 'ENERGY   once per request, 0.3 s', ORANGE); head(cC[0], 48.1, 'LEARN   after each request', MEAS)

box(cA[0], yr1, cA[1], hb, 'Sensors', 'skin, battery, DDR, CPU', ec=GRAY, fc=GRAYF)
box(cA[0], yr2, cA[1], hb, 'Watchdog', 'CPU ladder 1497 to 1017 MHz\nGPU ladder 1050 to 902 MHz', ec=ORANGE, fc=ORANGEF, fsub=5.0); step(cA[0], yr2 + hb + 0.2, 4)
box(cA[0], yr3, cA[1], hb, 'Vendor limiter', 'kernel takes the lower cap', ec=GRAY, fc=GRAYF)
arr(cA[0] + cA[1]/2, yr1 - 0.1, cA[0] + cA[1]/2, yr2 + hb + 0.3, c=INK)
arr(cA[0] + cA[1]/2, yr2 - 0.1, cA[0] + cA[1]/2, yr3 + hb + 0.3, c=GRAY); lab(cA[0] + cA[1]/2 + 1.0, (yr2 + yr3 + hb)/2, 'if the ladders miss', GRAY, 4.6, ha='left', bg=False)

box(cB[0], yr1, cB[1], hb, 'Battery state', 'level, charging, prompt N', ec=GRAY, fc=GRAYF)
box(cB[0], yr2, cB[1], hb, 'Tier to lever L', 'mains or above 50%: 1\n21 to 50%: 0.5;  20% or below: 0', ec=ORANGE, fc=ORANGEF, fsub=4.8); step(cB[0], yr2 + hb + 0.2, 5)
box(cB[0], yr3, cB[1], hb, 'Plan', 'walk the cost table\nGPU clocks, K, answer cap', ec=GREEN, fc=GREENF); step(cB[0], yr3 + hb + 0.2, 6)
arr(cB[0] + cB[1]/2, yr1 - 0.1, cB[0] + cB[1]/2, yr2 + hb + 0.3, c=INK)
arr(cB[0] + cB[1]/2, yr2 - 0.1, cB[0] + cB[1]/2, yr3 + hb + 0.3, c=ORANGE, ls='--'); lab(cB[0] + cB[1]/2 + 1.0, (yr2 + yr3 + hb)/2, 'rates, slack, cap', ORANGE, 4.6, ha='left', bg=False)

box(cC[0], yr1, cC[1], hb, 'Meter', 'USB rail + coulomb counter\nper phase', ec=ORANGE, fc=ORANGEF); step(cC[0], yr1 + hb + 0.2, 7)
box(cC[0], yr2, cC[1], hb, 'Two loops', 'time miss: L + 0.1\nenergy miss: L - 0.1', ec=ORANGE, fc=ORANGEF, fsub=5.0); step(cC[0], yr2 + hb + 0.2, 8)
box(cC[0], yr3, cC[1], hb, 'Cost table', 'EMA per plan, clipped to 10%', ec=ORANGE, fc=ORANGEF); step(cC[0], yr3 + hb + 0.2, 9)
arr(cC[0] + cC[1]/2, yr1 - 0.1, cC[0] + cC[1]/2, yr2 + hb + 0.3, c=INK)
arr(cC[0] + cC[1]/2, yr2 - 0.1, cC[0] + cC[1]/2, yr3 + hb + 0.3, c=INK)
# cross arrows C -> B
arr(cC[0] - 0.1, yr2 + hb/2, cB[0] + cB[1] + 0.2, yr2 + hb/2, c=ORANGE, ls='--'); lab((cC[0] + cB[0] + cB[1])/2, yr2 + hb/2 + 2.0, 'bias', ORANGE, 4.8, bg=False)
arr(cC[0] - 0.1, yr3 + hb/2, cB[0] + cB[1] + 0.2, yr3 + hb/2, c=MEAS, ls='-.'); lab((cC[0] + cB[0] + cB[1])/2, yr3 + hb/2 + 2.0, 'costs', MEAS, 4.8, bg=False)

# ---------------- vertical links between bands ----------------
# heat: pipeline -> sensors (dotted)
path([(29.0, 59.4), (29.0, yr1 + hb + 0.3)], HEAT, ls=':', lw=1.2); lab(29.8, 54.6, 'heat', HEAT, 4.8, ha='left', bg=False)
# clock cap: watchdog -> pipeline, left channel
path([(cA[0] - 0.1, yr2 + hb/2), (3.6, yr2 + hb/2), (3.6, 59.6)], HEAT, ls='--'); lab(4.6, 56.5, 'clock cap', HEAT, 4.8, ha='left', bg=False)
# plan: -> pipeline, channel left of column B
path([(cB[0] - 0.1, yr3 + hb/2), (33.6, yr3 + hb/2), (33.6, 59.6)], ORANGE, ls='--'); lab(34.5, 56.5, 'plan: clocks, K, answer cap', ORANGE, 4.8, ha='left', bg=False)
# measure: decode -> meter (dash-dot)
path([(94.0, 59.4), (94.0, yr1 + hb + 0.3)], MEAS, ls='-.', lw=1.2); lab(93.1, 56.5, 'energy, time', MEAS, 4.8, ha='right', bg=False)

# ---------------- legend ----------------
ly = 2.6
def tw(t): return len(t) * 0.64 * 5.0 / 72 / 7.2 * 100  # approx text width in axis units
x = 3.0
for ec, fc, t in [(ORANGE, ORANGEF, 'μKV mechanism'), (GRAY, GRAYF, 'unmodified llama.cpp / hardware'), (GREEN, GREENF, 'outcome'), (BLUE, BLUEF, 'KV cache')]:
    ax.add_patch(FancyBboxPatch((x, ly - 1.3), 3.2, 2.6, boxstyle='round,pad=0.1,rounding_size=0.6', fc=fc, ec=ec, lw=1.0)); ax.text(x + 4.0, ly, t, fontsize=5.0, va='center'); x += 4.0 + tw(t) + 2.0
for c, ls, t in [(INK, '-', 'data'), (ORANGE, '--', 'control'), (HEAT, ':', 'heat'), (MEAS, '-.', 'measure')]:
    ax.plot([x, x + 3.6], [ly, ly], color=c, ls=ls, lw=1.2); ax.text(x + 4.4, ly, t, fontsize=5.0, va='center'); x += 4.4 + tw(t) + 2.0
fig.savefig('fig_architecture_v9.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig('fig_architecture_v9.png', bbox_inches='tight', pad_inches=0.02, dpi=220)
print('ok')
