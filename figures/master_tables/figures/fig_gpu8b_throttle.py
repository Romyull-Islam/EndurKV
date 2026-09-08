#!/usr/bin/env python3
# Bonsai-8B vanilla on Adreno GPU (natural, no watchdog): GPU clock + skin/battery/SoC.
# Prefill vs decode separated (prefill shaded). Key point: the clock collapse to ~160 MHz
# in DECODE is GPU IDLE between per-token steps (decode is memory-bound, GPU underutilized),
# NOT a thermal throttle -- the phone stays cool throughout (SoC ~40, skin ~33, battery ~30).
# Data source: DATA env or default crash-1 partial trace (run aborted via Vulkan DeviceLost).
import csv,os,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA=os.environ.get("DATA","/tmp/gpu8b_full_crash1/sensors.csv")
PREFILL_MIN=os.environ.get("PREFILL_MIN")  # exact boundary from meta.json prefill_ms, if known
r=list(csv.DictReader(open(DATA)))
def col(n): return np.array([float(x[n]) if x.get(n,'').strip() not in ('','nan','-nan','inf') else np.nan for x in r])
t=(col("wall_clock_s")-col("wall_clock_s")[0])/60
g=col("gpu_clk_hz"); g=g/1e6 if np.nanmax(g)>1e5 else g
bat=col("battery_temp_mc")/1000; sk=col("shell_front_temp_mc")/1000; soc=col("sys-therm-2_temp_mc")/1000
def sm(y,w=9):
    y=np.where(np.isnan(y),np.nanmean(y),y); p=w//2
    return np.convolve(np.pad(y,p,mode="edge"),np.ones(w)/w,mode="valid")[:len(y)]
# prefill/decode boundary: prefill = sustained high clock; decode = frequent idle dips (<600 MHz)
if PREFILL_MIN:
    pe=float(PREFILL_MIN)
else:
    low=(g<600).astype(float); win=25
    frac=np.convolve(low,np.ones(win)/win,mode="same")
    pe=t[-1]
    for i in range(win,len(frac)):
        if frac[i]>0.15 and np.mean(frac[i:i+40]>0.10)>0.6: pe=t[i]; break
plt.rcParams.update({"font.size":8,"axes.linewidth":0.6})
fig,ax=plt.subplots(4,1,figsize=(3.6,5.0),sharex=True)
for a in ax:  # shade prefill region in every panel
    a.axvspan(0,pe,color="#000000",alpha=0.06,lw=0)
ax[0].plot(t,g,color="#0072B2",lw=0.7); ax[0].set_ylabel("GPU clock\n(MHz)"); ax[0].set_ylim(0,1300)
ax[0].text(pe/2,1230,"prefill",fontsize=6,color="#555",ha="center",style="italic")
ax[0].text(pe+(t[-1]-pe)/2,1230,"decode",fontsize=6,color="#555",ha="center",style="italic")
ax[0].annotate("in decode the clock idles to ~160 MHz\nbetween tokens (GPU underutilized) —\nnot a thermal throttle",
               (0.12,430),fontsize=5.2,color="#333",ha="left",va="center")
ax[1].plot(t,sm(sk),color="#D55E00",lw=1.0); ax[1].set_ylabel("skin\n(°C)")
ax[2].plot(t,sm(bat),color="#009E73",lw=1.0); ax[2].set_ylabel("battery\n(°C)")
ax[3].plot(t,sm(soc),color="#8e44ad",lw=1.0); ax[3].set_ylabel("SoC\n(°C)"); ax[3].set_xlabel("elapsed time (min)")
ax[3].annotate("cool throughout: no thermal throttle",(0.5,0.08),xycoords="axes fraction",
               fontsize=5.6,color="#555",ha="center")
for a in ax: a.grid(True,lw=0.3,alpha=0.4); a.tick_params(labelsize=7)
fig.suptitle("Bonsai-8B vanilla on Adreno GPU (natural): prefill vs decode",fontsize=7.4,y=0.998)
fig.align_ylabels(ax); fig.tight_layout(h_pad=0.4,rect=(0,0,1,0.985))
fig.savefig("fig_gpu8b_throttle.png",dpi=200,bbox_inches="tight")
print(f"wrote fig_gpu8b_throttle.png  (prefill|decode boundary = {pe:.2f} min; trace {t[-1]:.1f} min)")
print(f"  peaks: gpu_med={np.nanmedian(g):.0f}MHz skin={np.nanmax(sk):.1f} bat={np.nanmax(bat):.1f} soc={np.nanmax(soc):.1f}")
