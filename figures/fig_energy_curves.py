#!/usr/bin/env python3
"""Energy against each control, all measured on the OnePlus 15 with Llama-3.2-1B, 9737-token prompt.
(a) GPU clock, (b) CPU clock, (c) decoded tokens, (d) cache K. Points are cell means (n in caption);
lines in (c) are the scheduler's cost table. Sources: /tmp/ea_proof_v2, /tmp/sched_proof,
/tmp/split_proof, /tmp/bandit_online, /tmp/ea_n3, the CPU clock campaign, the six-generation soak.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, AQUA, RED, PURPLE, GREEN, ORANGE = "#2a78d6", "#1b9aa8", "#e34948", "#5b2a86", "#3aa41c", "#e08a1e"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.3, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "pdf.fonttype": 42, "ps.fonttype": 42})

fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(7.2, 5.4))
fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.145, wspace=0.3, hspace=0.5)
for ax in (a, b, c, d):
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# ---- (a) energy against GPU clock (whole request, 4096 and 1024 output tokens) ---------------------
clk = [726, 902, 1200]
E4096 = [1105, 1154, 1428]; T4096 = [303, 263, 221]           # ea_proof_v2, n=3 each
E1024 = [566, 586, 802];    T1024 = [218, 181, 139]           # bandit_online, n=5 to 11
a.plot(clk, E4096, color=BLUE, marker="o", ms=4.5, lw=1.2, label="4096 output tokens")
a.plot(clk, E1024, color=AQUA, marker="o", ms=4.5, lw=1.2, label="1024 output tokens")
for x, e, t in zip(clk, E4096, T4096): a.annotate(f"{e} J\n{t} s", (x, e), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=5.6, color=BLUE)
for x, e, t in zip(clk, E1024, T1024): a.annotate(f"{e} J\n{t} s", (x, e), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=5.6, color=AQUA)
# split plans: prefill at 1200, decode capped (4096 tokens)
a.scatter([902, 726], [1336, 1328], marker="D", s=22, color=RED, edgecolor=SURF, zorder=4, label="prefill 1200, decode capped")
a.annotate("decode 902: 1336 J, 224 s", (902, 1336), xytext=(8, 4), textcoords="offset points", fontsize=5.6, color=RED, va="center")
a.annotate("decode 726: 1328 J, 229 s", (726, 1328), xytext=(0, 9), textcoords="offset points", fontsize=5.6, color=RED, ha="left", va="bottom")
a.set_xticks(clk); a.set_xlim(650, 1290); a.set_ylim(400, 1700)
a.set_xlabel("GPU clock cap (MHz), Adreno 840"); a.set_ylabel("energy per request (J)")
a.set_title("(a) energy against GPU clock cap", loc="left", fontsize=7.6)
a.legend(loc="upper right", fontsize=5.8)

# ---- (b) energy against CPU clock (decode, prime core) ---------------------------------------------
cmhz = [883, 1018, 1267, 1498, 1632]; W = [2.289, 2.478, 2.983, 3.528, 3.970]; tps = [7.36, 8.31, 10.27, 11.82, 13.44]
jpt = [w / t for w, t in zip(W, tps)]
b.plot(cmhz, jpt, color=RED, marker="o", ms=4.5, lw=1.2)
for x, j, t in zip(cmhz, jpt, tps): b.annotate(f"{j*1000:.0f} mJ\n{t:.1f} tok/s", (x, j), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=5.6, color=RED)
b.set_ylim(0.20, 0.40); b.set_xticks(cmhz); b.set_xlim(820, 1700)
b.set_xlabel("CPU clock (MHz), prime core"); b.set_ylabel("energy per decoded token (J)")
b.set_title("(b) energy against CPU clock: flat within 7%", loc="left", fontsize=7.6)
b.text(0.03, 0.06, "power rises linearly with clock (voltage pinned),\nso a slower clock costs the same joules for longer", transform=b.transAxes, fontsize=5.8, color=MUTED, va="bottom")

# ---- (c) energy against decoded tokens, per clock: table lines and measured cells ------------------
NP = 9737
table = {1200: (58.2, 210.3, BLUE), 902: (40.6, 189.6, ORANGE), 726: (40.4, 173.3, RED)}
n = np.linspace(0, 4400, 50)
for mhz, (ep, ed, col) in table.items():
    c.plot(n, (NP * ep + n * ed) / 1000, color=col, lw=1.0, ls=(0, (3, 2)), label=f"table, {mhz} MHz")
meas = {1200: [(1024, 802), (4096, 1332), (4096, 1428)], 902: [(512, 500), (1024, 577), (1024, 586), (4096, 1152), (4096, 1154)], 726: [(1024, 566), (4096, 1105)]}
for mhz, pts in meas.items():
    c.scatter([p[0] for p in pts], [p[1] for p in pts], s=20, color=table[mhz][2], edgecolor=SURF, linewidth=0.5, zorder=4)
for cap, lab in ((512, "low cap"), (1024, "mid cap"), (4096, "healthy cap")):
    c.axvline(cap, color=MUTED, lw=0.5, ls=(0, (1, 2))); c.text(cap, 1560, f"{lab}\n{cap}", ha="center", va="top", fontsize=5.6, color=MUTED)
c.text(50, 400, "prefill alone: 567 J at 1200 MHz, 395 J at 902", fontsize=5.6, color=MUTED)
c.set_xlim(0, 4400); c.set_ylim(300, 1600)
c.set_xlabel("decoded tokens (output length)"); c.set_ylabel("energy per request (J)")
c.set_title("(c) energy against output length: linear in tokens", loc="left", fontsize=7.6)
c.legend(loc="lower right", fontsize=5.8)

# ---- (d) energy against cache K, relative to K = 1024 -----------------------------------------------
# CPU: full cache against K=1024 from the six-generation soak (per token, 4096-token generations);
# K tiers from the controller proof (per request, 1024 tokens). GPU: /tmp/ea_n3 (4096 tokens, native DVFS).
cpuK = [256, 512, 1024]; cpuE = [783 / 802, 789 / 802, 1.0]; cpu_full = 1296 / 490
gpuK = [487, 974, 1947]; gpuE = [1414 / 1393, 1.0, 1493 / 1393]
d.plot(cpuK + [10000], cpuE + [cpu_full], color=AQUA, marker="o", ms=4.5, lw=1.2, label="CPU (Oryon), full cache at right")
d.plot(gpuK, gpuE, color=BLUE, marker="s", ms=4.5, lw=1.2, label="GPU (Adreno 840), native DVFS")
d.set_xscale("log"); d.set_xticks([256, 512, 1024, 2048, 10000]); d.set_xticklabels(["256", "512", "1024", "2048", "full\ncache"])
d.axhline(1.0, color=AXIS, lw=0.6)
d.annotate(f"{cpu_full:.2f}x\n(1296 vs 490 mJ/token)", (10000, cpu_full), xytext=(-6, -4), textcoords="offset points", ha="right", va="top", fontsize=5.6, color=AQUA)
d.annotate("0.98x, quality 0.58", (256, cpuE[0]), xytext=(-4, -12), textcoords="offset points", ha="left", fontsize=5.6, color=AQUA)
d.annotate("0.98x, quality 0.63", (512, cpuE[1]), xytext=(4, -24), textcoords="offset points", ha="left", fontsize=5.6, color=AQUA)
d.annotate("1.07x", (1947, gpuE[2]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=5.6, color=BLUE)
d.annotate("1.02x", (487, gpuE[0]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=5.6, color=BLUE)
d.set_ylim(0.7, 2.9); d.set_xlim(200, 14000)
d.set_xlabel("cache budget K (tokens kept)"); d.set_ylabel("energy relative to K = 1024")
d.set_title("(d) energy against cache K, relative to K = 1024", loc="left", fontsize=7.6)
d.legend(loc="upper left", fontsize=5.8)

fig.text(0.08, 0.012, "OnePlus 15, Llama-3.2-1B, 9737-token prompt. (a) whole-request cells, n=3 at 4096 tokens and n=5 to 11 at 1024, cooled, charging off.\n"
         "(b) decode-phase rail power over tok/s, CPU clock campaign. (c) cost-table lines against cells from the controller, scheduler and bandit proofs.\n"
         "(d) CPU: six-generation soak (full cache) and controller proof (K tiers, 1024 tokens); GPU: cache axis, n=3 per K, clock not pinned, 4096 tokens.",
         fontsize=5.6, color=MUTED, va="bottom")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_energy_curves")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); print("wrote", out)
