#!/usr/bin/env python3
"""muKV paper deck: every table and figure of the paper, one idea per slide. Numbers taken from PAPER_MUKV_10PG.tex."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
F="/home/mislam22/EndurKV_workspace/EndurKV/figures"; D="/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad/deck"; T="/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad/tikz"
OUT="/home/mislam22/EndurKV_workspace/EndurKV/muKV_paper_talk.pptx"
INK, MUTED, BODY, RUST, BLUE, GREEN, LINE, PANEL, RUSTF, BLUEF, GREENF = "0C1116","6B7785","38424E","C4581A","2B6CA3","1A7A58","D9DEE4","F4F6F8","FBEDE3","E6F0F8","E4F3EC"
prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5); blank=prs.slide_layouts[6]
def rgb(h): return RGBColor.from_string(h)
def text(s,x,y,w,h,t,size=18,color=BODY,bold=False,align=PP_ALIGN.LEFT,anchor=MSO_ANCHOR.TOP,line=1.15):
    tb=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=tb.text_frame; tf.word_wrap=True; tf.vertical_anchor=anchor
    tf.margin_left=tf.margin_right=Inches(0.04); tf.margin_top=tf.margin_bottom=Inches(0.02)
    for i,l in enumerate(t if isinstance(t,list) else [t]):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.alignment=align; p.line_spacing=line
        r=p.add_run(); r.text=l; r.font.size=Pt(size); r.font.bold=bold; r.font.name="Arial"; r.font.color.rgb=rgb(color)
    return tb
def bullets(s,x,y,w,h,items,size=17,gap=10):
    tb=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=tb.text_frame; tf.word_wrap=True; tf.margin_left=tf.margin_right=Inches(0.04)
    for i,it in enumerate(items):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.space_after=Pt(gap); p.line_spacing=1.12
        r=p.add_run(); r.text="•  "+it; r.font.size=Pt(size); r.font.name="Arial"; r.font.color.rgb=rgb(BODY)
def rule(s,x,y,w,color=LINE):
    ln=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Emu(9525)); ln.fill.solid(); ln.fill.fore_color.rgb=rgb(color); ln.line.fill.background(); ln.shadow.inherit=False
def box(s,x,y,w,h,title,sub="",fill=PANEL,line=LINE,tcolor=INK,tsize=17,ssize=13,anchor=MSO_ANCHOR.MIDDLE):
    sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h)); sh.adjustments[0]=0.06
    sh.fill.solid(); sh.fill.fore_color.rgb=rgb(fill); sh.line.color.rgb=rgb(line); sh.line.width=Pt(1.25); sh.shadow.inherit=False
    tf=sh.text_frame; tf.word_wrap=True; tf.vertical_anchor=anchor; tf.margin_left=tf.margin_right=Inches(0.1)
    p=tf.paragraphs[0]; p.alignment=PP_ALIGN.CENTER; r=p.add_run(); r.text=title; r.font.size=Pt(tsize); r.font.bold=True; r.font.name="Arial"; r.font.color.rgb=rgb(tcolor)
    if sub:
        p2=tf.add_paragraph(); p2.alignment=PP_ALIGN.CENTER; r2=p2.add_run(); r2.text=sub; r2.font.size=Pt(ssize); r2.font.name="Arial"; r2.font.color.rgb=rgb(BODY)
    return sh
def arrow(s,x1,y1,x2,y2,color=INK,w=1.5):
    c=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2)); c.line.color.rgb=rgb(color); c.line.width=Pt(w)
    ln=c.line._get_or_add_ln(); from pptx.oxml.ns import qn; import lxml.etree as et
    tail=et.SubElement(ln,qn('a:tailEnd')); tail.set('type','triangle'); tail.set('w','med'); tail.set('len','med'); return c
def bignum(s,x,y,w,num,label,color=BLUE,size=40):
    text(s,x,y,w,0.9,num,size,color,True,align=PP_ALIGN.CENTER); text(s,x,y+0.85,w,0.8,label,13,BODY,align=PP_ALIGN.CENTER)
def header(s,kicker,title,n,accent=BLUE):
    text(s,0.6,0.35,9,0.3,kicker,11,accent,True); text(s,0.6,0.62,12.1,0.8,title,28,INK,True); rule(s,0.6,1.42,12.1); text(s,12.2,7.0,0.6,0.3,str(n),10,MUTED,align=PP_ALIGN.RIGHT)
def foot(s,t): text(s,0.6,6.98,11.4,0.4,t,9.5,MUTED)
def img(s,path,x,y,w=None,h=None):
    if w: return s.shapes.add_picture(path,Inches(x),Inches(y),width=Inches(w))
    return s.shapes.add_picture(path,Inches(x),Inches(y),height=Inches(h))
def table(s,x,y,w,h,rows,colw=None,size=13,hl=None):
    sh=s.shapes.add_table(len(rows),len(rows[0]),Inches(x),Inches(y),Inches(w),Inches(h)); tb=sh.table
    if colw:
        for i,cw in enumerate(colw): tb.columns[i].width=Inches(cw)
    for i,row in enumerate(rows):
        for j,val in enumerate(row):
            c=tb.cell(i,j); c.text=""; tf=c.text_frame; tf.word_wrap=True; c.margin_left=c.margin_right=Inches(0.08); c.margin_top=c.margin_bottom=Inches(0.05)
            r=tf.paragraphs[0].add_run(); r.text=str(val); r.font.size=Pt(size); r.font.name="Arial"; b=(i==0) or (hl and i in hl); r.font.bold=b; r.font.color.rgb=rgb(INK if b else BODY)
            c.fill.solid(); c.fill.fore_color.rgb=rgb("EEF1F4" if i==0 else ("F7F9FB" if i%2==0 else "FFFFFF"))
    return tb

FIG="/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures"
MECH=T+"/mech300-1.png"
n=0
def slide():
    global n; n+=1; return prs.slides.add_slide(blank), n

# 1 title
s,k=slide(); bg=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb=rgb("0C1116"); bg.line.fill.background()
text(s,0.8,1.2,11,0.4,"HOTMOBILE 2027",12,"9FB3C8",True)
text(s,0.8,1.9,11.6,2.2,["\u03bcKV: Energy- and Thermal-Aware","KV-Cache Eviction for LLM Inference on a Phone"],34,"FFFFFF",True,line=1.12)
text(s,0.8,4.5,11.5,1.4,["The KV cache, not compute, is the wall. The policies that bound it were built for server GPUs,","and three properties of a real on-device engine break them."],17,"C9D3DD")
text(s,0.8,6.1,11,0.8,["Md Romyull Islam  \u00b7  Kennesaw State University","OnePlus 15, Snapdragon 8 Elite Gen 5, Adreno 840  \u00b7  every number measured on the device"],13,"9FB3C8")

# 2 the wall
s,k=slide(); header(s,"THE PROBLEM","The phone is bound by the cache, not the matrix multiply",k)
for i,(t,sub) in enumerate([("Cache size","Phi-3 at 2K holds 1.2 GiB of f16 KV"),("DRAM traffic","every step reads the whole cache, about 750 MB per attention pass"),("Heat","the vendor cuts the prime cores to 883 MHz when the battery reaches 50 \u00b0C"),("Energy","a 4096-token request costs 1.4 kJ on the GPU at full clock")]):
    box(s,0.6+i*3.1,1.9,2.9,1.7,t,sub,fill=BLUEF,line=BLUE,tsize=17,ssize=13)
    if i<3: arrow(s,3.5+i*3.1,2.75,3.7+i*3.1,2.75,BLUE)
text(s,0.6,4.0,12.1,1.6,["The four are one chain. Cache size sets traffic, traffic sets power, power sets temperature, temperature sets the throttle, and the throttle sets throughput.","Across 372 on-device cells, live cache correlates with DDR temperature at r = 0.80 and with throughput at r = \u2212 0.77."],17,BODY)
text(s,0.6,5.9,12.1,0.8,"So a mobile cache policy has to be judged on quality, throughput, heat and energy at once, on real hardware.",18,INK,True)

# 3 three breaks
s,k=slide(); header(s,"WHAT BREAKS","Nine policies, and three engine properties that break them",k,RUST)
for i,(t,sub) in enumerate([("1  Budgets do not materialize","one cell array serves every head and layer, so a cell is freed only when all of them drop it"),("2  Scores cost the fast path","reading post-softmax attention forces the FlashAttention-off path"),("3  Compaction doubles memory","the state API restores survivors into a second full context")]):
    box(s,0.6+i*4.1,1.9,3.9,2.0,t,sub,fill=RUSTF,line=RUST,tsize=17,ssize=14)
text(s,0.6,4.3,12.1,2.2,["None of the three is visible on a server with per-head paged storage.","All three decide what a policy can actually do on a phone, and each is measured on the next three slides."],18,BODY,line=1.4)

# 4 break 1
s,k=slide(); header(s,"BREAK 1","A nominal budget is not a freed cell",k,RUST)
table(s,0.6,1.8,7.6,2.9,[["at a matched budget of 1024","live cells","share of the prompt"],["vanilla, Llama-3.2-1B CPU","9737","100%"],["\u03bcKV","721","7.4%"],["SnapKV","5929","61%"],["H2O","6110","63%"],["TOVA","4091","42%"],["Ada-KV","3484","36%"]],colw=[3.4,2.1,2.1],size=14,hl=[2])
bullets(s,8.4,1.8,4.5,4.6,["A cell survives if any head or any layer keeps it, so the per-head unions cover most of the prompt.","At their own published budgets the gap widens: on 110 LongBench prompts SnapKV at 2048 per head keeps 94% of the cache and H2O at 20% of the prompt keeps 92%.","SnapKV reports an 8.2 times smaller cache, H2O 5 to 10 times. On this engine they deliver 1.6 to 2.8.","\u03bcKV picks one keep set for every head and layer, which is what lets the cache compact."],15)
foot(s,"Live cells measured with the engine's own state size call, which serializes live cells only.")

# 5 break 2
s,k=slide(); header(s,"BREAK 2","Reading attention scores costs more than eviction saves",k,RUST)
table(s,0.6,1.8,6.4,2.9,[["Llama-3.2-1B, Adreno 840","tok/s","vs full cache"],["vanilla (full cache)","24.18","1.00"],["\u03bcKV","29.77","1.23"],["SnapKV","4.76","0.20"],["TOVA","4.34","0.18"],["H2O","3.58","0.15"],["Ada-KV","2.86","0.12"]],colw=[3.2,1.6,1.6],size=14,hl=[2])
bullets(s,7.2,1.8,5.7,4.6,["FlashAttention never writes the attention tensor, so a policy that reads scores cannot use it.","Every score-reading evictor decodes slower than not evicting at all, 0.12 to 0.30 times the full cache.","Retention does not predict speed. StreamingLLM keeps 777 cells, fewer than \u03bcKV's 723, and still runs at 0.23 because it stays on the slow path in this engine.","\u03bcKV scores inside the FlashAttention graph, so it keeps the signal and the fast path."],15)
foot(s,"12K-token prompt plus 4096 generated tokens, f16 KV for every policy, cold start.")

# 6 break 3
s,k=slide(); header(s,"BREAK 3","The state API needs a second full context",k,RUST)
box(s,0.6,1.9,3.7,1.9,"Phi-3-mini at 16K","two 6.4 GB caches plus 2.4 GB of weights",fill=RUSTF,line=RUST,tsize=17,ssize=14)
arrow(s,4.4,2.85,5.1,2.85,RUST)
box(s,5.2,1.9,3.7,1.9,"Android kills the process","reproduced twice, with six co-resident apps killed alongside",fill=RUSTF,line=RUST,tsize=17,ssize=14)
arrow(s,9.0,2.85,9.7,2.85,MUTED)
box(s,9.8,1.9,3.0,1.9,"and on a 24 GB desktop GPU","so it is the mechanism, not phone memory",tsize=16,ssize=13)
box(s,0.6,4.2,12.2,1.3,"\u03bcKV slides survivors into a dense prefix inside the tensors prefill already allocated","peak memory does not rise, and the same configuration runs; the two modes produce identical keep sets, with perplexity agreeing to 0.15%",fill=GREENF,line=GREEN,tsize=18,ssize=14)
text(s,0.6,5.8,12.1,0.8,"Measured: in-place compaction runs Phi-3 at 16K and reaches 2.64 times the full cache. The round trip runs only at ctx 8192.",17,BODY)

# 7 architecture
s,k=slide(); header(s,"\u03bcKV","The pass on top, the control plane below",k)
img(s,T+"/archw300-1.png",0.95,1.55,w=11.4)
text(s,0.6,6.6,12.1,0.4,"Steps 1 to 3 are the only changes to the inference pass. Steps 4 to 9 are the thermal, energy and learning columns.",12,BODY)
foot(s,"Figure 1 of the paper.")

# 8 mechanism
s,k=slide(); header(s,"SELECTION","One keep set for every head and layer",k)
img(s,MECH,4.15,1.5,h=5.0)
bullets(s,0.6,1.7,3.4,5.0,["The side node computes the real softmax scores for the last 16 queries of the last prefill chunk.","One score per position: the mean over layers and heads, max-pooled over 7 positions.","The split \u03b1 is the share of attention mass outside the recent window, floored at 0.70.","K = 1024 becomes 4 sinks, 721 anchors, 299 recent."],15)
bullets(s,9.3,1.7,3.5,5.0,["Per-head keep sets are a union, so 13 of 16 position bins stay live and nothing compacts.","Heatmap and grids are measured, not drawn: Llama-3.2-1B, layer 8."],15)
foot(s,"Figure 2 of the paper.")

# 9 CPU results
s,k=slide(); header(s,"RESULT  \u00b7  SUSTAINED DECODE ON THE CPU","9737-token prompt, 4096 generated, cold start, charging off",k,GREEN)
table(s,0.6,1.75,7.9,3.5,[["Llama-3.2-1B, K = 1024","tok/s","wall (s)","live cells","DDR \u00b0C","mWh"],
 ["vanilla (full cache)","5.0","1055","9737","59.8","1033"],["\u03bcKV","24.2","406","721","54.4","391"],
 ["SnapKV","6.8","902","5929","56.3","818"],["Ada-KV","6.7","875","3484","57.1","1002"],
 ["StreamingLLM","6.6","877","777","56.7","1052"],["H2O","5.8","1006","6110","54.8","1094"],["TOVA","6.0","947","4091","54.4","1069"]],
 colw=[2.8,1.0,1.1,1.3,0.9,0.8],size=13,hl=[2])
for i,(num,lab) in enumerate([("4.8\u00d7","the full cache's throughput"),("\u221262%","energy for the same work"),("7.4%","of the cells kept"),("\u22125 \u00b0C","peak DDR")]):
    bignum(s,8.8,1.8+i*1.25,4.0,num,lab,GREEN,26)
text(s,0.6,5.5,7.9,1.2,["Both policies run the same vendor-clamped 1.6 GHz, so the heat comes from memory traffic, not the clock.","Only \u03bcKV both compacts and keeps decode on the fused kernel."],14,BODY)
foot(s,"Table 1 of the paper. Three more models are in the paper: Bonsai-8B 4.5x, Phi-3-mini 4.5x, gemma-2-2b 2.4x, all against their own full cache.")

# 10 what compaction buys
s,k=slide(); header(s,"WHY","Every policy builds the full prompt cache. Only \u03bcKV gives it back",k,GREEN)
img(s,FIG+"/fig_cache_trajectory.png",0.7,1.7,w=7.4)
bullets(s,8.4,1.9,4.4,4.4,["At the end of prefill \u03bcKV compacts to 721 cells and stays bounded near 2K through decode.","SnapKV can only drop to 5.9K, because a cell needs every head to release it.","Vanilla grows without bound.","This one picture is the paper's argument: the budget a policy asks for is not the cache it frees."],16)
foot(s,"Figure 4 of the paper, Llama-3.2-1B on the phone CPU.")

# 11 GPU
s,k=slide(); header(s,"RESULT  \u00b7  MOBILE GPU","The same workload on the Adreno 840",k,GREEN)
table(s,0.6,1.75,5.9,2.6,[["Llama-3.2-1B","tok/s","dec \u00d7","cells"],["vanilla","24.18","1.00","9741"],["\u03bcKV","29.77","1.23","723"],["\u03bcKV, no compaction","25.61","1.06","723"],["best baseline (StreamingLLM)","5.55","0.23","777"]],colw=[2.6,1.1,1.1,1.1],size=13,hl=[2])
table(s,6.8,1.75,6.0,2.6,[["Phi-3-mini","tok/s","dec \u00d7","cells"],["vanilla","5.05","1.00","11157"],["\u03bcKV, in place","13.36","2.64","878"],["StreamingLLM, own budget","10.60","2.10","2000"],["SnapKV, gate promoted","5.80","1.15","10862"]],colw=[2.7,1.1,1.1,1.1],size=13,hl=[2])
bullets(s,0.6,4.6,12.2,2.1,["In-place compaction runs Phi-3 at 16K, where the state-API round trip is killed. That is the configuration the paper's third break describes.","Phi-3 needs a driver gate: the FlashAttention-off capture is numerically broken at head dimension 96, and uncorrected runs decode at full speed while emitting 18 to 44% corrupted tokens. The gate promotes SnapKV and Ada-KV to the in-graph side node and refuses H2O and TOVA."],15)

# 12 quality
s,k=slide(); header(s,"RESULT  \u00b7  QUALITY, BOTH ON THE PHONE","Retrieval and long-context, at each policy's own budget",k,GREEN)
table(s,0.6,1.75,12.2,3.3,[["Policy","needle hits, Phi-3 / Llama / Gemma / Bonsai","cells attended","LongBench F1","cache held"],
 ["vanilla (full cache)","14 / 13 / 11 / 14","4858","36.9","100%"],
 ["\u03bcKV","14 / 13 / 11 / 14","774","37.0","14%"],
 ["SnapKV","14 / 12 / 11 / 14","4237","38.0","94%"],
 ["Ada-KV","14 / 12 / 11 / 14","3361","37.8","71%"],
 ["TOVA","14 / 12 / 11 / 14","3670","37.4","77%"],
 ["H2O","14 / 8 / 10 / 13","4212","36.2","92%"],
 ["StreamingLLM","3 / 3 / 2 / 3","890","35.0","33%"]],colw=[2.6,4.0,1.9,1.9,1.8],size=13,hl=[2])
text(s,0.6,5.3,12.2,1.4,["\u03bcKV matches the full cache on both benchmarks while holding a seventh of the cache. The others buy about a point of F1 by holding 71 to 94% of it.","StreamingLLM fails retrieval everywhere: its window evicts exactly the middle of the context, which is where a buried fact lives."],16,BODY)
foot(s,"392 needle cells and 103 LongBench prompts, all on the phone. A four-model desktop sweep gives the same ordering.")

# 13 thermal
s,k=slide(); header(s,"RESULT  \u00b7  SUSTAINED HEAT","Two levers against the vendor's cliff",k,GREEN)
img(s,FIG+"/fig_bonsai_thermal.png",1.1,1.55,h=5.0)
bullets(s,4.6,1.8,8.2,4.8,["The full cache on Bonsai-8B runs 85.6 minutes and drives the battery to 50.2 \u00b0C. At 50 \u00b0C the vendor deep-throttles both clusters: 17 dips, 923 s at 883 MHz.",
 "\u03bcKV finishes in 33.2 minutes at a 44.2 \u00b0C peak, before the trigger. That is the cache lever.",
 "Given the watchdog as an ablation, the full cache's battery rise flattens from 0.48 to 0.05 \u00b0C per minute and settles at 49.1 \u00b0C. The prime cores never reach 883 MHz. That is the clock lever.",
 "The watchdog belongs to \u03bcKV and is never given to a baseline.",
 "On the GPU the ladder gives 1289 against 1390 J and a DDR peak of 75 against 90 \u00b0C, for 1% more time.",
 "Negative result: a cache budget tied to temperature does not cap peak heat. Cache controls throughput and energy; the clock caps heat."],14)

# 14 energy aware
s,k=slide(); header(s,"RESULT  \u00b7  ENERGY-AWARE CONTROL","One lever from the battery, two loops on model error",k,GREEN)
img(s,T+"/cp300-1.png",0.6,1.6,w=7.9)
table(s,8.7,1.8,4.2,2.0,[["tier","J","s","vs healthy"],["healthy, cap 4096","885","280","\u2014"],["mid, cap 1024","748","210","\u221215%"],["low, cap 512","506","177","\u221243%"]],colw=[1.5,0.9,0.9,0.9],size=13)
bullets(s,8.7,4.1,4.2,2.6,["123 requests, 91% to 11%, charging off, nothing forced.","It switched on the first request past each threshold, 50% and 19%.","A table seeded on the cable over-predicted by 36%; the clipped learning fixed it in three requests."],14)
foot(s,"Figure 3 and Table 6 of the paper. A decision costs 0.3 s on the phone.")

# 15 discharge picture
s,k=slide(); header(s,"RESULT  \u00b7  THE REAL DISCHARGE","It switches on the real battery, and it saves",k,GREEN)
img(s,FIG.replace("master_tables/figures","")+"fig_discharge_timeline.png",0.6,1.6,w=8.2)
bullets(s,9.0,1.9,3.8,4.6,["On the battery the phone is a different machine. The same healthy request costs 885 J instead of 1274 on the cable.","The OEM caps prefill at 726 MHz within a minute of every request, so the answer cap and the prefill cap carry the saving.","Mean prediction error over the run: 8.9% on the battery, 3.7% on the cable."],15)

# 16 RL
s,k=slide(); header(s,"THE RL QUESTION","Would a learned policy beat the rule? Three experiments said no",k,RUST)
for i,(t,sub) in enumerate([("1  Simulator","bandit over 72 values and a Q-learner over 576, 60 episodes and 6 seeds"),("2  Real cells, offline","a bandit fitted on logged requests, bootstrap for the best arm"),("3  On the phone","24 real requests, \u03b5-greedy over three GPU clocks")]):
    box(s,0.6+i*4.1,1.9,3.9,1.9,t,sub,fill=RUSTF,line=RUST,tsize=17,ssize=13)
bignum(s,0.9,4.2,3.5,"13 / 15","free pulls chose the rule's arm",GREEN,30)
bignum(s,4.8,4.2,3.5,"15.7 kJ","spent to learn it",MUTED,30)
bignum(s,8.7,4.2,3.5,"149 min","of phone time",MUTED,30)
text(s,0.6,6.0,12.2,0.8,"Energy and time per plan do not depend on the state of charge; only the weighting does, and that is a design choice. So the table learns and the rule decides.",17,BODY)

# 17 negatives
s,k=slide(); header(s,"REPORTED, NOT HIDDEN","The negative results are load-bearing",k,RUST)
bullets(s,0.6,1.9,12.2,4.6,["Cache size is not a temperature actuator. Halving K under co-load left peak DDR unchanged, and on a cool phone a smaller cache runs hotter at peak.",
 "The per-head budget gate we built is idle. Its candidate pool holds 96.5 to 99.8% of the prompt, and bypassing it leaves the keep set identical, so it is removed.",
 "A learned policy does not beat the rule, and we report what learning cost.",
 "Eviction still costs verbatim recall of the spans it drops, even though prediction and retrieval are preserved.",
 "The 1-bit kernels return non-finite logits on this GPU, so Bonsai's phone numbers are CPU numbers, and every backend now needs an output-validity check."],16)

# 18 summary
s,k=slide(); header(s,"SUMMARY","\u03bcKV in four lines",k)
bullets(s,0.6,1.9,12.2,4.6,["Three engine properties break server eviction policies on a phone: shared cells, the fast path, and the second context.",
 "\u03bcKV answers all three: scores from inside the FlashAttention graph, one keep set for every head and layer, compaction in place.",
 "4.8 times the full cache's decode throughput on the CPU and 1.23 on the GPU, at 7% of the cells and 62% less energy, with full-cache retrieval and LongBench.",
 "Around it, a clock watchdog keeps the vendor throttle unreachable and a per-request scheduler spends the battery by its level: 15% and 43% less per request at the lower tiers."],17)
prs.save(OUT); print("saved",OUT,n)
