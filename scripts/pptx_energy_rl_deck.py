#!/usr/bin/env python3
"""Energy-aware muKV and the RL question, v2: one idea per slide, few words, big type."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
F="/home/mislam22/EndurKV_workspace/EndurKV/figures"; D="/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad/deck"; T="/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad/tikz"
OUT="/home/mislam22/EndurKV_workspace/EndurKV/muKV_energy_rl_talk.pptx"
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
n=0
def slide():
    global n; n+=1; return prs.slides.add_slide(blank), n

# 1 title
s,k=slide(); bg=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb=rgb("0C1116"); bg.line.fill.background()
text(s,0.8,1.3,11,0.4,"μKV  ·  ENERGY-AWARE CONTROL, AND THE REINFORCEMENT-LEARNING QUESTION",12,"9FB3C8",True)
text(s,0.8,2.0,11.5,2.4,["One lever, two loops.","How a phone spends its battery on an LLM request, and what a learned policy adds."],36,"FFFFFF",True,line=1.1)
text(s,0.8,4.9,11,1.0,["OnePlus 15 · Llama-3.2-1B · every number metered on the device","Md Romyull Islam · Kennesaw State University · September 2026"],15,"C9D3DD")

# 2 the question: four knobs
s,k=slide(); header(s,"THE QUESTION","A request arrives. The phone has four knobs",k)
box(s,0.6,1.9,2.4,1.3,"Request","9737-token prompt, answer of open length",fill=BLUEF,line=BLUE)
arrow(s,3.0,2.55,3.7,2.55)
knobs=[("GPU clock","1200 → 902 MHz saves 19% energy, costs 19% time",GREEN,GREENF,"a lever"),("Answer length","0.17 to 0.21 J per token; the largest lever when the length is open",GREEN,GREENF,"a lever"),("Cache K","below 1024 saves 3 to 4%, quality falls to 0.63",MUTED,PANEL,"held at 1024"),("CPU clock","energy per token flat within 7%",MUTED,PANEL,"thermal only")]
for i,(t,sub,c,f,verdict) in enumerate(knobs):
    x=3.8+i*2.35; box(s,x,1.9,2.2,1.3,t,sub,fill=f,line=c,ssize=11); text(s,x,3.3,2.2,0.4,verdict,13,c,True,align=PP_ALIGN.CENTER)
text(s,0.6,4.2,12.1,1.2,["Two knobs carry the saving. Two do not. The phone's own governor reads none of them for an LLM request.","No shipped phone changes an LLM's clock, context or answer length by battery state."],17,BODY)
foot(s,"Cooled cells, charging off, OnePlus 15 GPU. Details on the next two slides.")

# 3 GPU clock and answer length, measured
s,k=slide(); header(s,"THE TWO LEVERS, MEASURED","GPU clock trades energy for time; answer length is linear",k)
img(s,D+"/crop_energy_a.png",0.6,1.7,w=5.9); img(s,D+"/crop_energy_c.png",6.8,1.7,w=5.9)
text(s,0.6,6.0,5.9,0.9,"1200 to 902 MHz: 1428 to 1154 J for 221 to 263 s. The last step to 726 buys 3% for 21% more time, so it is never taken.",14,BODY)
text(s,6.8,6.0,5.9,0.9,"Every output token costs 0.17 to 0.21 J on top of a fixed prefill. Capping the answer at 1024 or 512 tokens is the biggest saving available.",14,BODY)
foot(s,"fig_energy_curves.py, panels (a) and (c). n=3 at 4096 tokens, n=5 to 11 at 1024.")

# 4 GPU or CPU
s,k=slide(); header(s,"GPU OR CPU?","The GPU whenever it works; the CPU is the fallback",k)
table(s,0.6,1.8,6.9,2.3,[["plan, 9737-token prompt + 1024 tokens","energy","time"],["GPU 1200 MHz","782 J","142 s"],["GPU 902 MHz","589 J","181 s"],["GPU 726 MHz","571 J","219 s"],["CPU, K = 1024","822 J","304 s"]],colw=[3.6,1.7,1.6],size=14,hl=[4])
text(s,0.6,4.3,6.9,0.8,"CPU plans lose on both axes, so the ladder walk never reaches them while the GPU works.",15,BODY)
box(s,7.8,1.8,5.0,1.0,"1  No Vulkan library, or the caller forced CPU",tsize=14,anchor=MSO_ANCHOR.MIDDLE)
box(s,7.8,3.0,5.0,1.5,"2  The GPU runs it but the output is garbage","PrismML's 1-bit Bonsai-8B: non-finite logits on Adreno, decoded at full speed",fill=RUSTF,line=RUST,tsize=14,ssize=12)
text(s,7.8,4.7,5.0,0.5,"Every run is graded. A model that fails is marked, and the request re-runs on the CPU.",14,BODY)
box(s,7.8,5.4,5.0,1.1,"On the CPU one lever is left","the clock saves nothing, K cannot drop below 1024, so the answer cap tiers alone",fill=GREENF,line=GREEN,tsize=14,ssize=12)
table(s,0.6,5.3,6.9,1.2,[["Bonsai-8B on the CPU","cap 4096","cap 1024","cap 512"],["measured per-token costs","7853 J","5310 J, -32%","4886 J, -38%"]],colw=[2.7,1.4,1.4,1.4],size=13)
foot(s,"GPU and CPU plan costs predicted from the measured cost table. Bonsai figures from its own per-token costs, /tmp/nat_bonsai. Backend rules in ukv_sched.sh v2.5.")

# 4b the whole system: the final architecture figure
s,k=slide(); header(s,"THE WHOLE SYSTEM","Where the scheduler sits: below an unchanged inference pass",k)
img(s,T+"/archw300-1.png",0.95,1.55,w=11.4)
text(s,0.6,6.6,12.1,0.4,"Top: the pass; steps 1 to 3 are the only changes. Bottom: the thermal, energy and learning columns, 4 to 9. Dashed control, dotted heat, dash-dot measurement.",11.5,BODY)
foot(s,"fig_architecture_tikz, the same figure as Figure 1 of the paper.")

# 5 the scheduler picture
s,k=slide(); header(s,"THE SCHEDULER","Once per request, in 0.3 s",k)
img(s,T+"/cp300-1.png",0.6,1.7,w=8.6)
box(s,9.6,1.8,3.2,1.2,"Sense","battery, charging, prompt, length asked",ssize=12)
box(s,9.6,3.1,3.2,1.2,"Decide","tier → lever → walk the cost table",fill=RUSTF,line=RUST,ssize=12)
box(s,9.6,4.4,3.2,1.2,"Act","GPU clocks, cache K, answer cap",fill=GREENF,line=GREEN,ssize=12)
box(s,9.6,5.7,3.2,1.0,"Measure, learn","meter → table and lever",ssize=12)
foot(s,"ukv_sched.sh v2.3 on the phone. The thermal guard acts on the clock beside it.")

# 6 the lever
s,k=slide(); header(s,"THE LEVER","The battery tier sets one number; it sets everything else",k)
for i,(tier,L,f,c) in enumerate([("mains or above 50%","L = 1",GREENF,GREEN),("21 to 50%","L = 0.5",BLUEF,BLUE),("20% or below","L = 0",RUSTF,RUST)]):
    box(s,0.6+i*4.1,1.8,3.9,1.1,tier,L,fill=f,line=c,tsize=16,ssize=20)
table(s,0.6,3.2,12.1,2.6,[["what L sets","L = 1","L = 0.5","L = 0"],["exchange rate: joules a step must save per second it costs","1.5","0.8","0.5"],["quality floor","1.00","0.90","0.75"],["time slack over the fastest plan","5%","21.5%","40%"],["answer cap when the length is open","4096","1024","512"]],colw=[6.1,2.0,2.0,2.0],size=15)
text(s,0.6,6.0,12.1,0.8,"Full battery: a step down must pay well and may cost little time. Empty battery: almost any saving is taken.",16,BODY)
foot(s,"ukv_sched.sh: LAM=pl(0.5,0.8,1.5), QF=pl(0.75,0.90,1.00), TSLACK=0.03+0.37(1−L) floored at 0.05, CAPL by L.")

# 7 the walk
s,k=slide(); header(s,"THE WALK","Predict, step down while it pays, run, compare",k)
steps=[("1  Predict","every plan's energy and time from the table: per prompt token and per output token"),("2  Step down","from GPU 1200 while a step saves ≥ λ J per second it costs, quality stays above the floor, and time fits the budget"),("3  Run and meter","USB rail + coulomb counter, split at prefill / decode"),("4  Compare","metered against predicted; correct the lever and the table")]
for i,(t,sub) in enumerate(steps):
    box(s,0.6+i*3.1,1.9,2.9,2.4,t,sub,fill=(RUSTF if i==1 else PANEL),line=(RUST if i==1 else LINE),tsize=17,ssize=13)
    if i<3: arrow(s,3.5+i*3.1,3.1,3.7+i*3.1,3.1)
text(s,0.6,4.8,12.1,1.6,["The plans are the split clock (prefill 1200, decode 902), then 902, then 726, with K fixed at 1024 and the answer cap from the tier.","The split plan is the two-configuration schedule that is optimal under a deadline (Kim, Imes, Hoffmann 2015). It saves 6% for 1.5% of time."],16,BODY)

# 8 two loops
s,k=slide(); header(s,"TWO LOOPS","They act on the model's error, not on the raw meter",k)
box(s,0.6,1.9,3.8,1.6,"Time over budget by > 5%","performance loop: lever + 0.1",fill=RUSTF,line=RUST,tsize=16,ssize=14)
box(s,0.6,3.7,3.8,1.6,"Energy over prediction by > 5%","energy loop: lever − 0.1",fill=GREENF,line=GREEN,tsize=16,ssize=14)
box(s,0.6,5.5,3.8,1.1,"Both met","bias decays by 0.05 toward the tier",tsize=15,ssize=13)
arrow(s,4.4,2.7,5.2,2.7,RUST); arrow(s,4.4,4.5,5.2,4.5,GREEN)
box(s,5.3,2.6,2.6,2.4,"bias","clamped to ± 0.3, kept per tier",fill=BLUEF,line=BLUE,tsize=20,ssize=13)
arrow(s,7.9,3.8,8.7,3.8,BLUE)
box(s,8.8,2.6,4.0,2.4,"next request's lever","L = tier + bias; the walk trades at the new rate",fill=BLUEF,line=BLUE,tsize=16,ssize=13)
text(s,5.3,5.3,7.5,1.3,["A disturbance moves the lever; a quiet request moves it back. Knobs only one loop feels belong to that loop: decode clock and answer cap to energy, CPU clock to the watchdog. Only the prefill clock is shared."],14,BODY)
foot(s,"The JouleGuard coupling (Hoffmann, SOSP 2015). The cost table learns separately: EMA α 0.3, clipped to 10% per request.")

# 9 loops provoked
s,k=slide(); header(s,"THE LOOPS, PROVOKED","Thirteen cooled requests with real disturbances",k)
img(s,D+"/crop_loop_b.png",0.6,1.7,w=8.2)
bullets(s,9.0,1.8,3.9,5.0,["Four idle cores spinning: energy 30% over budget. The energy loop moved the lever down twice; two clean requests decayed it back.","An external 726 MHz cap 12 s into the request, what the vendor limiter does: time 10 to 12% over. The performance loop moved the lever up to its clamp, and back once the cap was gone.","Dark bars are misses. Every miss was answered by the loop that owns it."],14)
foot(s,"/tmp/loop_proof, 2026-09-05. Time budget and predicted energy, both + 5%.")

# 10 discharge
s,k=slide(); header(s,"THE REAL DISCHARGE","123 requests from 91% to 11%, nothing forced",k)
img(s,D+"/crop_discharge_a.png",0.6,1.7,w=8.0)
bignum(s,8.8,1.8,1.4,"885 J","healthy, cap 4096",BLUE,28); bignum(s,10.2,1.8,1.4,"748 J","mid, cap 1024",BLUE,28); bignum(s,11.5,1.8,1.4,"506 J","low, cap 512",BLUE,28)
bignum(s,8.9,3.6,1.9,"−15%","mid vs healthy",GREEN); bignum(s,10.9,3.6,1.9,"−43%","low vs healthy",GREEN)
text(s,8.9,5.4,4.0,1.4,["Switched at 50% and 19%, on the first request past each threshold. Predictions within 2% after three requests on the battery."],14,BODY)
foot(s,"/tmp/discharge_final, 2026-09-05/06. On the battery the OEM caps prefill at 726 MHz within a minute, so the caps carry the saving.")

# 11 the RL question
s,k=slide(); header(s,"THE RL QUESTION","Would a learned policy beat the rule? Three experiments",k,RUST)
for i,(t,sub) in enumerate([("1  Simulator","bandit and Q-learner on a phone model fitted from measurements"),("2  Real cells, offline","bandit fitted on logged requests"),("3  On the phone, online","bandit trained by 24 real requests")]):
    box(s,0.6+i*4.1,1.9,3.9,1.9,t,sub,fill=RUSTF,line=RUST,tsize=18,ssize=14)
    if i<2: arrow(s,4.5+i*4.1,2.85,4.7+i*4.1,2.85,RUST)
text(s,0.6,4.3,12.1,2.0,["Each step used less modeling and more measurement.","The answer never changed: one fixed plan per battery tier, and it is the plan the rule already picks."],20,INK,True,line=1.3)

# 12 experiment 1
s,k=slide(); header(s,"EXPERIMENT 1  ·  SIMULATOR","Two agents, and how big each one is",k,RUST)
box(s,0.6,1.8,5.9,3.4,"Contextual bandit","state: battery bucket (4) × charging (2) = 8 contexts\narms: 9 (cache 342 / 688 / 1379 cells × clock 883 / 1267 / 1632 MHz)\ntable: 72 values · ε = 0.15, sample-mean update",fill=RUSTF,line=RUST,tsize=18,ssize=14)
box(s,6.8,1.8,5.9,3.4,"Tabular Q-learning","state: battery (4) × charging (2) × DDR temperature (4) × throttled (2) = 64\narms: the same 9\ntable: 576 values · ε 0.15, α 0.15, γ 0.95",fill=PANEL,line=LINE,tsize=18,ssize=14)
text(s,0.6,5.4,12.1,1.5,["Reward: battery-weighted quality, throughput and energy, minus 0.25 when throttled. Training: 60 episodes × 6 seeds, each a discharge from 15%.","Result: the bandit converged to the best fixed arm at every battery level. The Q-learner never beat it. The simulator's thermal cliff never binds, so this is the weakest of the three."],14,BODY)
foot(s,"energy_rl/agents.py, sim.py. Simulator fitted from the GPU cache axis, the CPU clock axis and the CPU soak (τ 221 s).")

# 13 experiment 2
s,k=slide(); header(s,"EXPERIMENT 2  ·  REAL CELLS, OFFLINE","Every logged request is one sample: tier, plan, metered reward",k,RUST)
text(s,0.6,1.7,12.1,0.8,"reward = w_q · quality − w_t · time / T_ref − w_e · energy / E_ref, with the tier's weights: healthy (0.55, 0.40, 0.05), mid (0.35, 0.20, 0.45), low (0.20, 0.10, 0.70)",15,BODY)
table(s,0.6,2.7,7.2,2.4,[["GPU arm","healthy","mid","low"],["1200 MHz","best, P 1.00","",""],["902 MHz","","best, P 1.00","P 0.05"],["726 MHz","","","best, P 0.95"],["the rule picks","1200","902","902"]],colw=[2.4,1.6,1.6,1.6],size=15)
bullets(s,8.2,2.7,4.7,3.5,["Agrees with the rule at healthy and mid.","At low it takes 726 MHz: 3% less energy for 21% more time. The rule's stopping rule refuses that step.","That is the weights against the marginal criterion, a product choice, not data against the rule."],14)
foot(s,"energy_rl/bandit_real.py; P(best) from 2000 bootstrap resamples of the real cells. CPU arms: K = 1024 best at every tier.")

# 14 experiment 3 setup
s,k=slide(); header(s,"EXPERIMENT 3  ·  ON THE PHONE","One real request per pull",k,RUST)
for i,(num,lab) in enumerate([("3","arms: GPU 1200, 902, 726 MHz"),("3","contexts: battery tiers, cycled"),("0.25","ε, greedy otherwise"),("9 + 15","warm-start pulls + free pulls")]):
    bignum(s,0.6+i*3.1,1.9,2.9,num,lab,RUST)
text(s,0.6,4.1,12.1,0.6,"Reward from the meter: w_q − w_t · T/T_ref − w_e · E/E_ref. Each pull: cool the phone, pin the CPU, run 9737 prompt + 1024 output tokens, meter it.",15,BODY)
for i,(num,lab) in enumerate([("24","requests"),("15.7 kJ","spent learning"),("70 min","of inference"),("149 min","of phone time with cooling")]):
    bignum(s,0.6+i*3.1,4.9,2.9,num,lab,MUTED)
foot(s,"run_bandit_online.py, /tmp/bandit_online, 2026-09-03. The seed drew no exploration step after the warm start.")

# 15 experiment 3 result: learned values as a clear grid, greedy arm highlighted
s,k=slide(); header(s,"EXPERIMENT 3  ·  RESULT","What it learned: one value per battery tier and arm",k,RUST)
text(s,0.6,1.7,7.6,0.5,"mean reward per pull (higher is better); pulls in brackets; the greedy arm is shaded",13,MUTED)
vals=[["","1200 MHz","902 MHz","726 MHz","greedy arm","the rule"],
      ["healthy","+0.108  (6)","+0.004  (1)","−0.099  (1)","1200","1200"],
      ["mid","−0.305  (1)","−0.237  (6)","−0.290  (1)","902","902"],
      ["low","−0.605  (1)","−0.456  (4)","−0.458  (3)","902 (726 within 0.002)","902"]]
tb=table(s,0.6,2.2,7.6,2.6,vals,colw=[1.1,1.3,1.3,1.3,1.5,1.1],size=14)
best={1:1,2:2,3:2}
for r,c in best.items():
    cell=tb.cell(r,c); cell.fill.solid(); cell.fill.fore_color.rgb=rgb(GREENF)
    for pph in cell.text_frame.paragraphs:
        for run in pph.runs: run.font.bold=True; run.font.color.rgb=rgb(GREEN)
text(s,0.6,5.0,7.6,1.4,["Read across a row: the tier's best arm. Every row ends where the rule already sits.","The 726 arm at low is a tie with 902: 20 J less for 37 s more, which the rule's exchange rate refuses."],14,BODY)
bignum(s,8.6,1.9,2.1,"13 / 15","free pulls chose the rule's arm",GREEN,34); bignum(s,10.8,1.9,2.1,"15.7 kJ","spent to learn it",MUTED,34)
table(s,8.6,4.0,4.3,1.7,[["arm","J / request","s / request"],["1200 MHz","802","139"],["902 MHz","586","181"],["726 MHz","566","218"]],colw=[1.5,1.4,1.4],size=13)
text(s,8.6,5.9,4.3,0.9,"Same plans as the rule, so the same energy per request.",14,BODY)
foot(s,"/tmp/bandit_online/log.txt: values after pull 23; pooled cost per arm over the 24 pulls, 9737-token prompt, 1024 output tokens.")

# 16 RL vs rule
s,k=slide(); header(s,"RL VERSUS THE RULE","Same plans. Different costs, different failure modes",k,RUST)
table(s,0.6,1.8,12.1,4.3,[["","rule: table + lever + loops","learned bandit"],["reaching the plan","one measured campaign","24 pulls, 15.7 kJ, 149 min"],["a new prompt shape","predicted from per-token costs","new warm start"],["cable to battery","table re-learned in 3 requests (36% → 2%)","new pulls per context"],["a disturbance","loops correct the lever, then decay","absorbed as the arm's cost"],["temperature","watchdog ladders at 2 Hz","not in the state; adding it did not help"]],colw=[2.6,4.9,4.6],size=15)
foot(s,"The measurements removed the case for learning a policy. The case for learning the table stayed.")

# 17 why decision logic ships
s,k=slide(); header(s,"WHY THE SHIPPED CONTROLLER IS DECISION LOGIC","Learn the costs, not the policy",k)
bullets(s,0.6,1.9,12.1,4.5,["Energy and time per plan do not depend on the state of charge. Only the weighting does, and the weighting is a design choice.","The whole landscape is three thresholds and one stopping rule. Every agent rediscovered it.","What drifts is the table, so the table learns: clipped EMA from every request, no bad plans explored on the user's battery.","Learning a policy would earn its place only for something the meter cannot see, such as a user's tolerance for waiting."],18,12)

# 18 summary
s,k=slide(); header(s,"SUMMARY","Energy-aware μKV in four lines",k)
bullets(s,0.6,1.9,12.1,4.5,["Two knobs carry the saving: the GPU clock and the answer length. Cache and CPU clock do not.","One lever from the battery tier, two loops on model error, one learned cost table.","On a real discharge the lower tiers cost 15% and 43% less per request, and the GPU is used whenever it exists.","Three RL experiments, from a simulator to 24 real requests on the phone, all converged to the rule's plans at a cost the rule does not pay."],18,12)
prs.save(OUT); print("saved",OUT,n)
