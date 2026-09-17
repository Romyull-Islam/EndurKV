#!/usr/bin/env python3
"""Realized cache against decode speed on the phone GPU.

The job of this chart: show that how much cache a policy frees does not predict
how fast it decodes, but whether it kept the fused kernel does. Form is a scatter
because the claim is a relationship between two measures across entities. Kernel
state is the only categorical split, encoded by hue AND marker shape so identity
never rests on colour alone.

Every point is measured, Llama-3.2-1B, ctx 16384, 9737-token prompt, 4096
generated. Multipliers are against the pooled no-eviction median (24.68 tok/s, n=6);
per-head points are medians of three runs.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ON, OFF, REF, INK, MUTED = '#0072B2', '#D55E00', '#8f8f8f', '#1a1a1a', '#5a5a5a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.6})

#  label                       cells%  mult  kernel  (dx, dy) label nudge   ha
P = [('no eviction',            100.0, 1.00, 'on',  (-2.5,  0.11), 'right'),
     ('SnapKV',                  61.8, 0.18, 'off', ( 3.0,  0.05), 'left'),
     ('H2O',                     62.7, 0.15, 'off', (-3.0, -0.05), 'right'),
     ('TOVA',                    42.3, 0.19, 'off', ( 2.5,  0.02), 'left'),
     ('Ada-KV',                  36.1, 0.16, 'off', (-2.5,  0.00), 'right'),
     ('StreamingLLM, kernel off', 20.6, 0.26, 'off', ( 2.5,  0.02), 'left'),
     ('StreamingLLM, its published budget', 20.6, 1.18, 'on', ( 2.5, 0.03), 'left'),
     ('$\\mu$KV',                 7.4, 1.18, 'on',  ( 0.0,  0.12), 'center'),
     ('$\\mu$KV, no compaction',  7.4, 1.04, 'on',  ( 2.5,  0.00), 'left')]

fig, ax = plt.subplots(figsize=(7.0, 2.75), dpi=200)
ax.axhline(1.0, color=REF, lw=0.9, ls='--', zorder=1)
ax.text(101, 1.02, 'as fast as keeping every token', fontsize=7.0, color=MUTED,
        ha='right', va='bottom')
ax.axhspan(0, 1.0, color='#f6f2ef', zorder=0)

for lab, x, y, k, (dx, dy), ha in P:
    on = k == 'on'
    ax.scatter([x], [y], s=54 if on else 46, marker='o' if on else 'X',
               facecolor=ON if on else OFF, edgecolor='white', linewidth=0.9, zorder=4)
    ax.annotate(lab, (x + dx, y + dy), fontsize=7.0, color=INK, ha=ha, va='center', zorder=5)

ax.set_xlabel('cache actually live after prefill (% of the full cache)')
ax.set_ylabel('decode speed\n($\\times$ no eviction)')
ax.set_xlim(0, 104); ax.set_ylim(0, 1.42)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
ax.grid(axis='y', color='#e8e8e8', lw=0.5); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.spines['left'].set_color('#c0c0c0'); ax.spines['bottom'].set_color('#c0c0c0')
ax.tick_params(colors=MUTED, labelsize=7.2)

h = [plt.Line2D([], [], marker='o', ls='', color=ON, markeredgecolor='white', label='fused kernel kept'),
     plt.Line2D([], [], marker='X', ls='', color=OFF, markeredgecolor='white', label='fused kernel lost')]
ax.legend(handles=h, loc='lower right', bbox_to_anchor=(0.995, 0.02), frameon=False, fontsize=7.2,
          handletextpad=0.4, borderpad=0.2)

fig.tight_layout(pad=0.4)
fig.savefig('fig_gap_scatter.pdf'); fig.savefig('fig_gap_scatter.png', dpi=200)
print('wrote fig_gap_scatter')
