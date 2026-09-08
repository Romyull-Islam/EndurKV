#!/usr/bin/env python3
"""Controlled watchdog experiment: same cold-start Bonsai muKV workload WITH the
(both-cluster) watchdog vs WITHOUT it. Handles both logger formats:
  7-field: epoch,cpu0_cur,cpu6_cur,cpu0_max,cpu6_max,bat,skin  (new, both clusters)
  5-field: epoch,cpu6_max,cpu6_cur,bat,skin                    (old control)
Panels vs time: perf-core clock, prime-core clock, battery, skin."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os
import numpy as np

def load(f):
    if not os.path.exists(f): return None
    rows=[r for r in csv.reader(open(f)) if len(r) in (5,7)]
    if not rows: return None
    d=dict(t=[],perf=[],prime=[],bat=[],sk=[])
    for r in rows:
        try:
            if len(r)==7:
                d['t'].append(float(r[0]));d['perf'].append(float(r[1])/1000);d['prime'].append(float(r[2])/1000)
                d['bat'].append(float(r[5])/1000);d['sk'].append(float(r[6])/1000)
            else:  # 5-field: no perf clock
                d['t'].append(float(r[0]));d['perf'].append(float('nan'));d['prime'].append(float(r[2])/1000)
                d['bat'].append(float(r[3])/1000);d['sk'].append(float(r[4])/1000)
        except: pass
    if not d['t']: return None
    t0=d['t'][0]; d['t']=[(x-t0)/60 for x in d['t']]; return d

WD=load('/tmp/wd_demo/wd_trace.csv'); NO=load('/tmp/wd_control/wd_trace.csv')
if not WD or not NO:
    print('need both:', 'WD ok' if WD else 'WD MISSING', '|', 'NO ok' if NO else 'NO MISSING'); raise SystemExit
# ALIGN at battery=35C crossing (removes different-cold-start confound): re-zero
# each run's time to when its battery first reaches 35.0C, keep samples from there.
def align35(d):
    i35=next((i for i,b in enumerate(d['bat']) if b>=35.0), 0)
    t0=d['t'][i35]
    out={key:[] for key in d}
    for i in range(len(d['t'])):
        if d['t'][i]>=t0-0.5:
            for key in d: out[key].append(d[key][i])
    out['t']=[x-t0 for x in out['t']]
    return out
WD=align35(WD); NO=align35(NO)
OR='#D55E00'; GY='#7a7a7a'
plt.rcParams.update({'font.size':7,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
def sm(v,n=6):
    v=np.array(v); return np.convolve(np.nan_to_num(v,nan=np.nanmean(v) if np.isfinite(v).any() else 0),np.ones(n)/n,mode='same')
fig,axs=plt.subplots(4,1,figsize=(3.5,4.8),dpi=200,sharex=True,gridspec_kw={'hspace':0.15})
a0,a1,a2,a3=axs
if np.isfinite(NO['perf']).any(): a0.plot(NO['t'],sm(NO['perf']),color=GY,lw=0.9,label='no watchdog')
if np.isfinite(WD['perf']).any(): a0.plot(WD['t'],sm(WD['perf']),color=OR,lw=0.9,label='watchdog')
a0.set_ylabel('Perf cores\n(cpu0-5) MHz',fontsize=6.5); a0.legend(fontsize=6,frameon=False,loc='upper right',ncols=2); a0.set_ylim(1100,2150)
a1.plot(NO['t'],sm(NO['prime']),color=GY,lw=0.9); a1.plot(WD['t'],sm(WD['prime']),color=OR,lw=0.9)
a1.set_ylabel('Prime cores\n(cpu6-7) MHz',fontsize=6.5); a1.set_ylim(1150,2550)
a2.plot(NO['t'],NO['bat'],color=GY,lw=1.1); a2.plot(WD['t'],WD['bat'],color=OR,lw=1.1)
a2.set_ylabel('Battery (°C)',fontsize=7); [a2.axhline(y,color='#ddd',ls=':',lw=0.4) for y in (35,35.5,36)]
a3.plot(NO['t'],NO['sk'],color=GY,lw=1.1); a3.plot(WD['t'],WD['sk'],color=OR,lw=1.1)
a3.set_ylabel('Skin (°C)',fontsize=7); a3.set_xlabel('Time since battery reached 35°C (min)',fontsize=6.8)
for ax in (a0,a1,a2,a3): ax.axvline(0,color='#bbb',ls='-',lw=0.4)
fig.tight_layout()
out='/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_wd_control'
fig.savefig(out+'.png',bbox_inches='tight',dpi=300); fig.savefig(out+'.pdf',bbox_inches='tight')
# aligned rise-rate over battery 35-38C
def rr(d,key,lo=35,hi=38):
    p=[(d['bat'][i],d[key][i],d['t'][i]) for i in range(len(d['t'])) if lo<=d['bat'][i]<=hi and np.isfinite(d[key][i])]
    if len(p)<3: return None
    return (p[-1][1]-p[0][1])/((p[-1][2]-p[0][2]) or 1)
print('=== aligned over battery 35-38C ===')
for lab,d in [('no-wd',NO),('watchdog',WD)]:
    print(f"  {lab:<9} skin rise={rr(d,'sk')}  battery rise={rr(d,'bat')}")
print('saved',out)
