#!/usr/bin/env python3
"""The same policies, the same phone, the two backends.

Job: show that a published policy's sign flips between CPU and GPU. Form is a
paired dot plot (dumbbell) because every entity has exactly two values and the
reader's question is about the gap between them, not the absolute level.
x is a ratio spanning 40x, so it is logarithmic; the 1.0 rule is the anchor.

CPU: /tmp/nat_cpu, vanilla 5.02 tok/s. GPU: vanilla 24.34 tok/s (pooled median, n=4); per-head GPU values are
medians of two campaigns.
Llama-3.2-1B, 9737-token prompt, 4096 generated, ctx 16384.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CPU, GPU, REF, INK, MUTED = '#0072B2', '#D55E00', '#8f8f8f', '#1a1a1a', '#5a5a5a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.6})

#  policy      cpu tok/s  gpu tok/s
R = [('SnapKV',    6.78,  4.35),
     ('Ada-KV',    6.70,  3.42),
     ('TOVA',      5.99,  4.57),
     ('H2O',       5.84,  3.59)]
CPU_BASE, GPU_BASE = 5.02, 24.34

fig, ax = plt.subplots(figsize=(3.33, 2.1), dpi=200)
ax.axvspan(0.08, 1.0, color='#f6f2ef', zorder=0)
ax.axvline(1.0, color=REF, lw=0.9, ls='--', zorder=1)

for i, (lab, c, g) in enumerate(R):
    y = len(R) - 1 - i
    xc, xg = c / CPU_BASE, g / GPU_BASE
    ax.plot([xg, xc], [y, y], color='#d6d6d6', lw=1.6, zorder=2, solid_capstyle='round')
    ax.scatter([xg], [y], s=34, marker='X', color=GPU, edgecolor='white', lw=0.8, zorder=4)
    ax.scatter([xc], [y], s=34, marker='o', color=CPU, edgecolor='white', lw=0.8, zorder=4)

ax.set_yticks(range(len(R))); ax.set_yticklabels([r[0] for r in R][::-1], fontsize=7.4)
ax.set_xscale('log')
ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 5])
ax.set_xticklabels(['0.1', '0.25', '0.5', '1', '2', '5'])
ax.set_xlim(0.08, 6.5); ax.set_ylim(-0.6, len(R) - 0.2)
ax.set_xlabel('decode speed, $\\times$ no eviction on the same backend')
ax.text(0.95, len(R) - 0.35, 'slower than keeping\nevery token', fontsize=6.8, color=MUTED,
        ha='right', va='top', linespacing=1.25)

ax.grid(axis='x', color='#ececec', lw=0.5); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#c0c0c0')
ax.tick_params(colors=MUTED, labelsize=7.0, length=2)

h = [plt.Line2D([], [], marker='o', ls='', color=CPU, markeredgecolor='white', label='CPU'),
     plt.Line2D([], [], marker='X', ls='', color=GPU, markeredgecolor='white', label='Adreno GPU')]
ax.legend(handles=h, loc='lower right', frameon=False, fontsize=7.0,
          handletextpad=0.35, borderpad=0.15, labelspacing=0.25)
fig.tight_layout(pad=0.35)
fig.savefig('fig_backend_inversion_b.pdf'); fig.savefig('fig_backend_inversion_b.png', dpi=200)
print('wrote fig_backend_inversion_b')
