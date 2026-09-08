#!/usr/bin/env python3
# fig_throttle_v2 -- (a) the honest throttle metric; (b) the 6-gen CPU soak traces.
# Replaces fig_stalls_vs_cache: its panel (a) led with the misleading raw fraction,
# and the Discussion's soak/watchdog story had no figure. (2026-08-28)
import json,re,os,csv,glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE,ORANGE,GRAY,GREEN="#0072B2","#D55E00","#666666","#009E73"
def J(f):
    try: return json.loads(re.sub(r':\s*-?nan\b',': NaN',re.sub(r':\s*-?inf\b',': Infinity',open(f).read())))
    except: return {}
fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,2.45),gridspec_kw={"width_ratios":[1,1.35]})

# ---- (a) throttle-seconds per 1000 tokens vs live cells (the normalised metric) ----
PTS=[("vanilla","/tmp/nat_cpu/vanilla",GRAY,"D"),("muKV","/tmp/nat_cpu/mukv_faon",ORANGE,"o"),
     ("StreamingLLM","/tmp/nat_cpu/streamingllm",BLUE,"s"),("H2O","/tmp/nat_cpu/h2o",BLUE,"s"),
     ("TOVA","/tmp/nat_cpu/tova",BLUE,"s"),("Ada-KV","/tmp/nat_cpu/adakv",BLUE,"s"),
     ("SnapKV","/tmp/nat_cpu/snapkv",BLUE,"s")]
def floorsec(p,j):
    f=p+"/sensors.csv"
    if not os.path.exists(f): return None
    T=[];F=[]
    for r in csv.DictReader(open(f)):
        try: T.append(float(r["monotonic_s"])); F.append(float(r["cpu6_freq_hz"])/1e6)
        except: pass
    if not T: return None
    tot=j["total_ms"]/1000; pre=j["prefill_ms"]/1000; t0=T[-1]-tot+pre
    sec=sum((b-a) for a,b,fr in zip(T,T[1:],F) if fr<0.9 and a>=t0)
    return sec/((j.get("n_decode_steps") or 4096)/1000.0)
lab_dy={"vanilla":(-12,-16),"muKV":(11,-3),"StreamingLLM":(-16,-15),"H2O":(-10,11)}
band=[]
for name,p,c,mk in PTS:
    j=J(p+"/gen.json")
    if not j.get("decode_tps"): continue
    L,H,D=j["n_layers"],j["n_kv_heads"],j["head_dim"]
    cells=int(round(j["retained_kv_bytes"]/(2.0*L*H*D*2)))
    y=floorsec(p,j)
    if y is None: continue
    if c==BLUE: band.append(y)
    print("  A-value %-14s cells=%-6d y=%.1f"%(name,cells,y))
    ax.scatter(cells,y,c=c,marker=mk,s=64,zorder=3)
    if name in lab_dy:
        dx,dy=lab_dy[name]
        t="$\\mu$KV  (3.4$\\times$ under vanilla)" if name=="muKV" else name
        ax.annotate(t,(cells,y),textcoords="offset points",xytext=(dx,dy),fontsize=8.2,
                    color=c if c!=BLUE else "#33547a")
if band: ax.axhspan(min(band)-3,max(band)+3,color=BLUE,alpha=.10,zorder=0)
ax.set_xscale("log"); ax.set_xlabel("live KV cells retained (log)",fontsize=8.5)
ax.set_ylabel("seconds at 883 MHz floor\nper 1000 tokens",fontsize=8.5)
ax.set_title("(a)  throttle exposure / work (CPU)",fontsize=9.8,loc="left",fontweight="bold")
ax.grid(True,color="#e3e3e3",lw=.6); ax.set_axisbelow(True)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.set_ylim(0,128)
pass  # band described in the body text, not inside the plot
pass

# ---- (b) 6-gen CPU soak: DDR temperature, three arms ----
# All four arms: the recommended CPU anchor (cliff) tracks wd-off almost exactly,
# so the governor is free there; the early anchor is the other end of the dial and
# is visibly cooler but longer. Plotting only the early arm made the watchdog look
# like a slowdown, which is true of that anchor alone and not of the shipped one.
ARMS=[("vanilla","vanilla",GRAY,"-"),
      ("mukv_wdoff","$\\mu$KV",ORANGE,"-"),
      ("mukv_wdon","$\\mu$KV + wd (cliff, shipped)",BLUE,"-"),
      ("mukv_wdearly","$\\mu$KV + wd (early)",GREEN,"--")]
for arm,lbl,c,ls in ARMS:
    f="/tmp/cpu_soak/%s/sensors.csv"%arm
    if not os.path.exists(f): continue
    T=[];D=[]
    for r in csv.DictReader(open(f)):
        for k in r:
            if "ddr" in k.lower() and "temp" in k.lower():
                try:
                    v=float(r[k]); D.append(v/1000 if v>200 else v); T.append(float(r["monotonic_s"]))
                except: pass
    if not T: continue
    t0=T[0]; X=[(t-t0)/60 for t in T]
    # light smoothing
    import statistics as st
    Y=[st.mean(D[max(0,i-4):i+5]) for i in range(len(D))]
    bx.plot(X,Y,ls,color=c,lw=1.7,label=lbl,zorder=3)
bx.axhline(65,color="#b22222",lw=1.2,ls=(0,(4,2)),zorder=2)
bx.text(1.5,65.7,"65$^\\circ$C DDR kernel cliff",fontsize=8,color="#b22222")
bx.set_xlabel("minutes of back-to-back 16K generations",fontsize=8.5)
bx.set_ylabel("DDR temp ($^\\circ$C)",fontsize=8.5)
bx.set_title("(b)  sustained soak (CPU)",fontsize=9.8,loc="left",fontweight="bold")
bx.grid(True,color="#e3e3e3",lw=.6); bx.set_axisbelow(True)
for sp in ("top","right"): bx.spines[sp].set_visible(False)

bx.annotate("vanilla: 105 min, peak 68.3$^\\circ$C",xy=(74,36.5),fontsize=6.6,color=GRAY)
bx.annotate("early anchor: $6.9^\\circ$C cooler,\n$37\\%$ longer",xy=(62,44.5),fontsize=6.6,color="#1a7a58")
_h,_l = bx.get_legend_handles_labels()
fig.legend(_h,_l,fontsize=7,frameon=False,loc="lower center",ncol=4,
           bbox_to_anchor=(0.72,-0.015),handlelength=1.6,columnspacing=1.3,handletextpad=0.5)
fig.tight_layout(rect=(0,0.10,1,1))
out="/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_throttle_v2"
for ext in ("pdf","png"): fig.savefig(out+"."+ext,dpi=190,bbox_inches="tight")
print("saved",out)
