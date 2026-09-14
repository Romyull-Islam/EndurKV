#!/usr/bin/env python3
"""Slide figure: what happens to a request after it runs.

Every value is from scripts/android/ukv_sched.sh: tier defaults (lines 118-121),
nudge +/-0.1 clamped +/-0.3 and decay 0.05 (lines 26-27, 278-283), table EMA
alpha 0.3 clipped to 10% per request (lines 55, 261).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ORANGE, ORANGEF = '#D55E00', '#fdeee6'
GREEN, GREENF = '#009E73', '#e6f5f0'
GRAY, GRAYF, INK, SUB = '#8f8f8f', '#f3f3f3', '#1a1a1a', '#333333'
plt.rcParams.update({'font.family': 'DejaVu Sans'})

FW, FH = 11.0, 3.6
fig, ax = plt.subplots(figsize=(FW, FH), dpi=200)
ax.set_position([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x, y, w, h, title, sub, ec, fc, fs=11.5, fsub=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.5',
                                fc=fc, ec=ec, lw=1.5, zorder=3))
    ax.text(x + w/2, y + h/2 + 4.6, title, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=INK, zorder=4)
    ax.text(x + w/2, y + h/2 - 4.2, sub, ha='center', va='center', fontsize=fsub,
            color=SUB, zorder=4, linespacing=1.35)

def arrow(x1, y1, x2, y2, c, ls='-', lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16,
                                 lw=lw, color=c, ls=ls, zorder=2, shrinkA=0, shrinkB=0))

def lab(x, y, t, c, fs=9.5, ha='center', bold=False):
    ax.text(x, y, t, fontsize=fs, ha=ha, va='center', color=c,
            fontweight='bold' if bold else 'normal', zorder=6, linespacing=1.3)

Y, HB = 56.0, 30.0
box(2.0, Y, 20.0, HB, 'Plan', 'walk the cost table\nunder the lever L', GREEN, GREENF)
box(27.0, Y, 18.0, HB, 'Run', 'prefill and decode\nat that plan', GRAY, GRAYF)
box(50.0, Y, 20.0, HB, 'Meter', 'USB rail + coulomb\ncounter, per phase', ORANGE, ORANGEF)
box(75.0, Y, 23.0, HB, 'Compare', 'measured against the\nbudget the plan assumed', ORANGE, ORANGEF)
for x1, x2 in ((22.0, 27.0), (45.0, 50.0), (70.0, 75.0)):
    arrow(x1, Y + HB/2, x2, Y + HB/2, INK)

YL, HL = 10.0, 30.0
box(27.0, YL, 30.0, HL, 'Lever bias', 'time over budget by 5%: +0.1\nenergy over by 5%: −0.1\nboth met: decays 0.05', ORANGE, ORANGEF)
box(63.0, YL, 30.0, HL, 'Cost table', 'EMA α = 0.3 on the plan just run,\nclipped to 10% per request\none table per model', ORANGE, ORANGEF)

# compare feeds both loops
ax.plot([86.5, 86.5], [Y - 0.4, YL + HL + 6.0], color=ORANGE, ls='--', lw=1.6, zorder=2)
ax.plot([42.0, 86.5], [YL + HL + 6.0, YL + HL + 6.0], color=ORANGE, ls='--', lw=1.6, zorder=2)
arrow(42.0, YL + HL + 6.0, 42.0, YL + HL + 0.4, ORANGE, ls='--')
arrow(78.0, YL + HL + 6.0, 78.0, YL + HL + 0.4, ORANGE, ls='--')
lab(64.0, YL + HL + 8.4, 'the error, not the measurement', ORANGE)

# both loops return to the plan
ax.plot([27.0 - 0.4, 12.0], [YL + HL/2, YL + HL/2], color=GREEN, ls='-.', lw=1.6, zorder=2)
arrow(12.0, YL + HL/2, 12.0, Y - 0.4, GREEN, ls='-.')
lab(13.2, 30.0, 'next request starts here', GREEN, ha='left')
ax.plot([63.0 - 0.4, 59.5], [YL + HL/2, YL + HL/2], color=GREEN, ls='-.', lw=1.6, zorder=2)
ax.plot([59.5, 59.5], [YL + HL/2, YL - 4.0], color=GREEN, ls='-.', lw=1.6, zorder=2)
ax.plot([59.5, 12.0], [YL - 4.0, YL - 4.0], color=GREEN, ls='-.', lw=1.6, zorder=2)
ax.plot([12.0, 12.0], [YL - 4.0, YL + HL/2], color=GREEN, ls='-.', lw=1.6, zorder=2)

lab(50.0, 96.0, 'L is the tier default plus the bias, clipped to [0, 1].  '
                'The bias is clamped to ±0.3.  No policy is learned online.',
    INK, fs=10.0)

fig.savefig('parts/11_loop.png', dpi=220, bbox_inches='tight', pad_inches=0.04, facecolor='white')
print('wrote parts/11_loop.png')
