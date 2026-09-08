#!/usr/bin/env python3
"""Watchdog glide demo: as Bonsai muKV heats the phone from cold, the fixed
watchdog steps the big-core cap down gradually (1632->1497->1382->1267) as the
battery crosses 35/35.5/36. Two stacked panels sharing time:
  top    = watchdog cap (scaling_max) + actual clock
  bottom = battery + skin temperature with the ladder thresholds marked
Reads /tmp/wd_demo/wd_trace.csv (epoch,scaling_max,cur_freq,battery_mc,skin_mc)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os
import numpy as np

F='/tmp/wd_demo/wd_trace.csv'
if not os.path.exists(F):
    print('no wd_trace.csv yet'); raise SystemExit

t=[]; cap=[]; cur=[]; bat=[]; skin=[]
for r in csv.reader(open(F)):
    if len(r)<5: continue
    try:
        t.append(float(r[0])); cap.append(float(r[1])/1000); cur.append(float(r[2])/1000)
        bat.append(float(r[3])/1000); skin.append(float(r[4])/1000)
    except: pass
t0=t[0]; t=[(x-t0)/60 for x in t]

OR='#D55E00'; GY='#7a7a7a'; BL='#0072B2'; RD='#c00000'
plt.rcParams.update({'font.size':7,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
fig,(a1,a2)=plt.subplots(2,1,figsize=(3.5,3.0),dpi=200,sharex=True,gridspec_kw={'hspace':0.12})

# top: watchdog cap (step) + actual clock
a1.step(t,cap,where='post',color=OR,lw=1.3,label='watchdog cap')
a1.plot(t,np.convolve(cur,np.ones(6)/6,mode='same'),color=GY,lw=0.7,alpha=0.8,label='actual clock')
for f in (1632,1497,1382,1267):
    a1.axhline(f,color='#ccc',ls=':',lw=0.5)
a1.set_ylabel('CPU clock (MHz)',fontsize=7); a1.set_ylim(1150,1700)
a1.legend(fontsize=6,frameon=False,loc='upper right',ncols=2)
a1.set_yticks([1267,1382,1497,1632])

# bottom: battery + skin with ladder thresholds
a2.plot(t,bat,color=OR,lw=1.2,label='battery')
a2.plot(t,skin,color=BL,lw=1.2,label='skin')
for temp,lab in [(35,'35'),(35.5,''),(36,'36')]:
    a2.axhline(temp,color=OR,ls='--',lw=0.5,alpha=0.6)
a2.text(0.1,36.1,'battery ladder 35/35.5/36',fontsize=5.4,color=OR)
a2.set_ylabel('Temperature (°C)',fontsize=7); a2.set_xlabel('Time (min)',fontsize=7)
a2.legend(fontsize=6,frameon=False,loc='lower right')
fig.tight_layout()
out='/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_watchdog_glide'
fig.savefig(out+'.png',bbox_inches='tight',dpi=300); fig.savefig(out+'.pdf',bbox_inches='tight')
print('saved',out,'| samples:',len(t),'| cap range:',min(cap),'-',max(cap))
