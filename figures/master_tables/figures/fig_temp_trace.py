#!/usr/bin/env python3
"""Battery + shell temperature over time, muKV vs vanilla, Llama-1B and Bonsai-8B.
2x2: rows = shell / battery, cols = Llama / Bonsai. Dashed line = vendor clamp
trigger (shell ~40C). Natural DVFS, new build."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os
import numpy as np

OR='#D55E00'; GY='#7a7a7a'
plt.rcParams.update({'font.size':7,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})

def series(path, col):
    if not os.path.exists(path): return None, None
    rows=list(csv.DictReader(open(path,encoding='utf-8',errors='replace')))
    if len(rows)<5: return None, None
    t0=float(rows[0]['monotonic_s']); T=[]; V=[]
    for r in rows:
        try:
            T.append((float(r['monotonic_s'])-t0)/60); V.append(float(r[col])/1000)
        except: pass
    return T, V

CELLS={'Llama-1B':{'muKV':'/tmp/nat_cpu/mukv_faon','vanilla':'/tmp/nat_cpu/vanilla'},
       'Bonsai-8B':{'muKV':'/tmp/nat_bonsai/mukv_faon','vanilla':'/tmp/nat_bonsai/vanilla'}}
COL={'shell':'shell_front_temp_mc','battery':'battery_temp_mc'}
YL={'shell':'Shell (°C)','battery':'Battery (°C)'}
YLIM={'shell':(33,50),'battery':(30,45)}

fig,axs=plt.subplots(2,2,figsize=(6.6,3.6),dpi=200,sharex='col')
for cj,(model,cells) in enumerate(CELLS.items()):
    for ri,metric in enumerate(['shell','battery']):
        ax=axs[ri][cj]
        for pol,c in [('vanilla',GY),('muKV',OR)]:
            T,V=series(os.path.join(cells[pol],'sensors.csv'),COL[metric])
            if T:
                # light smoothing
                V=np.convolve(V,np.ones(5)/5,mode='same')
                ax.plot(T,V,color=c,lw=1.1,label=pol)
                if pol=='muKV':
                    ax.annotate(f'done {T[-1]:.0f}m',xy=(T[-1],V[-1]),fontsize=5.5,color=OR,
                                xytext=(2,-1),textcoords='offset points')
        if metric=='shell':
            ax.axhline(40,color='#c00000',ls='--',lw=0.8)
            if cj==0: ax.text(0.5,40.3,'vendor clamp trigger ~40°C',fontsize=5.4,color='#c00000')
            ax.set_title(model,fontsize=8,fontweight='bold')
        ax.set_ylabel(YL[metric],fontsize=7); ax.set_ylim(*YLIM[metric])
        if ri==1: ax.set_xlabel('Time (min)',fontsize=7)
        if ri==0 and cj==0: ax.legend(fontsize=6,frameon=False,loc='lower right')
fig.tight_layout(w_pad=1.3,h_pad=0.8)
out='/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_temp_trace'
fig.savefig(out+'.png',bbox_inches='tight',dpi=300); fig.savefig(out+'.pdf',bbox_inches='tight')
print('saved',out,'| vanilla-Bonsai present:',os.path.exists('/tmp/nat_bonsai/vanilla/sensors.csv'))
