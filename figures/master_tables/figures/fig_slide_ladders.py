#!/usr/bin/env python3
"""Slide figure: the two watchdog ladders against the vendor's single cliff.

CPU steps and thresholds: scripts/android/preempt_throttle_watchdog_v2.sh (lines 363-364).
GPU steps and thresholds: scripts/android/gpu_watchdog.sh (header, lines 8-9).
Vendor triggers are the measured ones quoted in the paper: 883 MHz at battery 50 C,
726 MHz at DDR 64 C.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORANGE, RED, GRAY, INK = '#D55E00', '#C0392B', '#8f8f8f', '#1a1a1a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

def stair(ax, xs, ys, x_end, color, lw=2.4, label=None):
    px, py = [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        nx = xs[i + 1] if i + 1 < len(xs) else x_end
        px += [x, nx]; py += [y, y]
    ax.plot(px, py, color=color, lw=lw, solid_joinstyle='miter', label=label, zorder=4)
    for i in range(1, len(xs)):
        ax.plot([xs[i], xs[i]], [ys[i - 1], ys[i]], color=color, lw=lw, zorder=4)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5), dpi=200)

# ---- CPU: battery ladder, five steps, floor above the vendor cliff ----
ax = axes[0]
xs = [47.0, 48.0, 48.5, 49.0, 49.5]; ys = [1497, 1382, 1267, 1132, 1017]
ax.plot([45.5, 47.0], [1632, 1632], color=ORANGE, lw=2.4, zorder=4)
ax.plot([47.0, 47.0], [1632, 1497], color=ORANGE, lw=2.4, zorder=4)
stair(ax, xs, ys, 51.2, ORANGE, label='μKV watchdog')
ax.plot([45.5, 50.0], [1632, 1632], color=RED, lw=2.0, ls='--', zorder=3, label='vendor engine')
ax.plot([50.0, 50.0], [1632, 883], color=RED, lw=2.0, ls='--', zorder=3)
ax.plot([50.0, 51.2], [883, 883], color=RED, lw=2.0, ls='--', zorder=3)
ax.axhline(883, color=RED, lw=0.8, ls=':', zorder=1)
ax.text(51.1, 897, 'vendor floor 883 MHz', fontsize=8, color=RED, va='bottom', ha='right')
ax.text(50.05, 1560, 'one cliff, at 50 °C', fontsize=8.5, color=RED, va='top')
ax.text(46.0, 1150, 'five steps, all\nabove the cliff', fontsize=8.5, color=ORANGE)
ax.set_xlim(45.5, 51.2); ax.set_ylim(820, 1700)
ax.set_xlabel('battery temperature (°C)'); ax.set_ylabel('CPU cluster cap (MHz)')
ax.set_title('CPU: battery ladder, skin is the same staircase 3 °C higher', fontsize=9.5, pad=7)
ax.set_yticks([883, 1017, 1132, 1267, 1382, 1497, 1632])
ax.legend(loc='lower left', frameon=False, fontsize=8.5)

# ---- GPU: DDR ladder, three steps, floor above the vendor trip ----
ax = axes[1]
xs = [60.0, 62.0, 63.5]; ys = [1050, 967, 902]
ax.plot([58.5, 60.0], [1200, 1200], color=ORANGE, lw=2.4, zorder=4)
ax.plot([60.0, 60.0], [1200, 1050], color=ORANGE, lw=2.4, zorder=4)
stair(ax, xs, ys, 66.0, ORANGE, label='μKV watchdog')
ax.plot([58.5, 64.0], [1200, 1200], color=RED, lw=2.0, ls='--', zorder=3, label='vendor engine')
ax.plot([64.0, 64.0], [1200, 726], color=RED, lw=2.0, ls='--', zorder=3)
ax.plot([64.0, 66.0], [726, 726], color=RED, lw=2.0, ls='--', zorder=3)
ax.axhline(726, color=RED, lw=0.8, ls=':', zorder=1)
ax.text(65.9, 736, 'vendor trip 726 MHz', fontsize=8, color=RED, va='bottom', ha='right')
ax.text(64.05, 1160, 'one cliff, at 64 °C', fontsize=8.5, color=RED, va='top')
ax.annotate('', xy=(62.0, 1010), xytext=(63.5, 1010),
            arrowprops=dict(arrowstyle='<->', color=GRAY, lw=1.0))
ax.text(62.75, 1062, '1.5 °C hysteresis\non the way back up', fontsize=8, color=GRAY, ha='center')
ax.set_xlim(58.5, 66.0); ax.set_ylim(680, 1260)
ax.set_xlabel('DDR temperature (°C)'); ax.set_ylabel('Adreno cap (MHz)')
ax.set_title('GPU: DDR ladder', fontsize=9.5, pad=7)
ax.set_yticks([726, 902, 967, 1050, 1200])
ax.legend(loc='lower left', frameon=False, fontsize=8.5)

for ax in axes:
    ax.grid(axis='y', color='#e6e6e6', lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color('#bbbbbb'); ax.spines['bottom'].set_color('#bbbbbb')
    ax.tick_params(colors='#555555', labelsize=8)

fig.tight_layout(pad=0.6)
fig.savefig('parts/10_ladders.png', dpi=220, bbox_inches='tight', pad_inches=0.03, facecolor='white')
print('wrote parts/10_ladders.png')
