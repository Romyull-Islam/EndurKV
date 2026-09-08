#!/usr/bin/env python3
"""Compact demand-aware-K motivation: break-even budget scales with output length,
not the fixed prompt. Wide short strip to save column height."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False})
out_len = np.array([64, 256, 512, 1024, 2048])
break_even = np.array([256, 512, 1024, 2048, 4096])          # smallest K that still retrieves
demand_k = 2 * (36 + out_len)                                 # demand-aware: 2*(prompt+output)
prompt_floor = np.full_like(out_len, 36)                      # prompt-length budget (output-blind)

fig, ax = plt.subplots(figsize=(3.35, 1.55), dpi=200)
ax.plot(out_len, break_even, '-o', color='#0072B2', lw=1.3, ms=4, label='break-even $K$ (needed)')
ax.plot(out_len, demand_k, '--s', color='#009E73', lw=1.3, ms=4, label='demand-aware $K$')
ax.plot(out_len, prompt_floor, ':', color='#D55E00', lw=1.4, label='prompt-length $K$ (fails)')
ax.plot(out_len, prompt_floor, 'x', color='#D55E00', ms=5)
ax.set_yscale('log', base=2)
ax.set_yticks([32, 256, 2048]); ax.set_yticklabels(['32', '256', '2048'], fontsize=6)
ax.set_xlabel('output length (tokens)', fontsize=6.8)
ax.set_ylabel('KV budget $K$', fontsize=6.8)
ax.legend(fontsize=5.4, frameon=False, loc='upper left', handlelength=1.6, borderaxespad=0.1)
ax.tick_params(labelsize=6)
fig.tight_layout(pad=0.3)
out='/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_kadaptive'
fig.savefig(out+'.png', bbox_inches='tight', dpi=300)
fig.savefig(out+'.pdf', bbox_inches='tight')
print('saved', out)
