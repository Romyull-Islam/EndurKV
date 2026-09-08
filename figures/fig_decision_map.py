#!/usr/bin/env python3
"""Decision map: what the scheduler chooses as a function of battery state, for a few request
shapes and phone conditions. Evaluates the on-phone policy (energy_rl/sched_policy.py, the same
table, weights, stopping rule and length rule) at every state of charge from 0 to 100 and on
mains. Nothing is simulated: this is the policy's own decision for each state.
"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, "/home/mislam22/EndurKV_workspace/EndurKV/energy_rl")
from sched_policy import load_table, weights, output_cap, choose

INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
PLAN_COL = {"gpu1200_k1024": "#0d366b", "gpu1200d902_k1024": "#1c5cab", "gpu1200d726_k1024": "#3987e5",
            "gpu902_k1024": "#6da7ec", "gpu726_k1024": "#b7d3f6",
            "cpu_k1024": "#1baf7a", "cpu_k512": "#7fd4b3", "cpu_k256": "#c9eddd"}
PLAN_LAB = {"gpu1200_k1024": "GPU 1200 MHz, K 1024", "gpu1200d902_k1024": "GPU 1200, decode cap 902", "gpu1200d726_k1024": "GPU 1200, decode cap 726",
            "gpu902_k1024": "GPU cap 902 MHz", "gpu726_k1024": "GPU cap 726 MHz",
            "cpu_k1024": "CPU, K 1024", "cpu_k512": "CPU, K 512", "cpu_k256": "CPU, K 256"}
CAP_COL = {4096: "#f2f1ec", 1024: "#dcdad2", 512: "#c3c2b7", "caller": "#ffffff"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "pdf.fonttype": 42, "ps.fonttype": 42,
})

rows = load_table()
scenarios = [
    ("long prompt, open answer", 9737, None, False, True, False),
    ("long prompt, caller fixes 64 tokens", 9737, 64, False, True, False),
    ("short prompt (512), open answer", 512, None, False, True, False),
    ("long prompt, open answer, hot phone\n(battery >= 40 C or DDR >= 60 C)", 9737, None, True, True, False),
    ("long prompt, GPU unavailable", 9737, None, False, False, False),
]
def decide(soc, mains, n_prompt, max_tokens, hot, gpu_ok):
    tier, w, lam, qf, ts = weights(soc, mains)
    cap, rule = output_cap(tier, max_tokens=max_tokens)
    best, _ = choose([dict(r) for r in rows], n_prompt, cap, w, lam, qf, hot=hot, gpu_ok=gpu_ok, tslack=ts)
    return best["plan"], (cap if max_tokens is None else "caller"), tier

fig, ax = plt.subplots(figsize=(7.2, 4.0))
fig.subplots_adjust(left=0.31, right=0.98, top=0.90, bottom=0.30)
H, GAP = 0.55, 0.22
for i, (name, np_, mt, hot, gpu_ok, _) in enumerate(scenarios):
    y = -(i * (H + GAP + 0.30))
    # plan band across SoC 0..100
    start = 0; prev = None
    segs = []
    for soc in range(0, 101):
        plan, cap, tier = decide(soc, False, np_, mt, hot, gpu_ok)
        if prev is None: prev = (plan, cap); start = soc
        elif (plan, cap) != prev:
            segs.append((start, soc, prev)); prev = (plan, cap); start = soc
    segs.append((start, 101, prev))
    for s0, s1, (plan, cap) in segs:
        ax.broken_barh([(s0, s1 - s0)], (y, H), facecolors=PLAN_COL[plan], edgecolors=SURF, linewidth=0.5)
        ax.broken_barh([(s0, s1 - s0)], (y - 0.20, 0.18), facecolors=CAP_COL[cap], edgecolors=AXIS, linewidth=0.4)
        mid = (s0 + s1) / 2
        ax.text(mid, y - 0.11, ("output cap " + str(cap)) if cap != "caller" else "length set by caller", ha="center", va="center", fontsize=5.6, color=INK)
    # mains column at x = 104..112
    plan, cap, tier = decide(100, True, np_, mt, hot, gpu_ok)
    ax.broken_barh([(104, 8)], (y, H), facecolors=PLAN_COL[plan], edgecolors=SURF, linewidth=0.5)
    ax.broken_barh([(104, 8)], (y - 0.20, 0.18), facecolors=CAP_COL[cap], edgecolors=AXIS, linewidth=0.4)
    ax.text(-2, y + H / 2, name, ha="right", va="center", fontsize=7, color=INK)
for x_ in (20.5, 50.5):
    ax.axvline(x_, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=5)
ax.text(10, 0.62, "low tier", ha="center", fontsize=6.5, color=MUTED); ax.text(35.5, 0.62, "mid tier", ha="center", fontsize=6.5, color=MUTED)
ax.text(75.5, 0.62, "healthy tier", ha="center", fontsize=6.5, color=MUTED); ax.text(108, 0.62, "mains", ha="center", fontsize=6.5, color=MUTED)
ax.set_xlim(-0.5, 113); ax.set_ylim(-(len(scenarios) * (H + GAP + 0.30)) + 0.1, 0.9)
ax.set_xticks([0, 20, 50, 80, 100, 108]); ax.set_xticklabels(["0%", "20%", "50%", "80%", "100%", "mains"])
ax.set_yticks([]); ax.set_xlabel("battery state of charge the scheduler reads")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.grid(False)
ax.set_title("What the scheduler decides, by battery state and request", loc="left")
handles = [Patch(facecolor=PLAN_COL[k], label=PLAN_LAB[k]) for k in ("gpu1200_k1024", "gpu1200d902_k1024", "gpu1200d726_k1024", "gpu902_k1024", "gpu726_k1024", "cpu_k1024", "cpu_k512", "cpu_k256")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, handlelength=1.4, columnspacing=1.0)
fig.text(0.30, 0.02, "Upper band: backend, clock (prefill / decode), cache. Lower band: output length rule. The CPU clock is never a decision: it saves no energy on this phone. "
         "The lever (1 at mains and above 50%, 0.5 at mid, 0 when low) sets the exchange rate, the quality floor and the time budget; a step down must save at least λ times its time cost, keep quality above the floor, and fit the time budget. Split plans are predictions until measured.",
         fontsize=6.2, color=MUTED, ha="left", va="bottom", wrap=True)
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/fig_decision_map"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
print("wrote", out)
for name, np_, mt, hot, gpu_ok, _ in scenarios:
    print(f"  {name.split(chr(10))[0]:44s}", " | ".join(f"{s}%: {decide(s, False, np_, mt, hot, gpu_ok)[0]}/{decide(s, False, np_, mt, hot, gpu_ok)[1]}" for s in (10, 40, 80)), "| mains:", decide(100, True, np_, mt, hot, gpu_ok)[0])
