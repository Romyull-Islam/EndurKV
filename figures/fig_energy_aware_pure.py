#!/usr/bin/env python3
"""Pure measured results for the energy-aware question, OnePlus 15, Llama-3.2-1B-Instruct Q4_K_M.

(a) GPU: cache budget K vs energy per decoded token, two accountings, three campaigns.
(b) GPU: K vs decode throughput, every cell, so the dispatch bimodality is visible.
(c) GPU: clock cap vs energy per token at K=1024 (the lever that pays on the GPU).
(d) Where a request's energy goes: prefill vs decode, per backend.
(e) What the v1 controller did per battery state (identical command line).
(f) Answer quality per tier, LongBench F1, n=15 per point (Llama-1B).

Inputs: /tmp/phase_split.json, /tmp/tier_quality.json, /tmp/ea_proof, /tmp/clock_sweep.
Energy = USB rail integral + battery pack coulomb delta (clock_cell_report.py method).
"""
import json, os, sys, glob, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "/home/mislam22/EndurKV_workspace/EndurKV/scripts")
from ea_proof_summary import load as load_cell

BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ps = json.load(open("/tmp/phase_split.json"))
tq = json.load(open("/tmp/tier_quality.json"))

camps = {"energy_tiers": ("o", "tier sweep, n=1"), "ea_n3": ("s", "n=3 campaign"), "ea_proof": ("^", "controller proof, n=1")}
gpu = {c: [] for c in camps}
cpu = []
vanilla = None
for tag, v in ps.items():
    camp, cell = tag.split("/")
    if cell.startswith("cpu"):
        cpu.append(v); continue
    if cell == "gen_vanilla":
        vanilla = v; continue
    gpu[camp].append(v)

fig, axs = plt.subplots(3, 2, figsize=(7.2, 8.4))
fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.085, wspace=0.30, hspace=0.55)
(a, b), (c, d), (e, f) = axs

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.5)

KT = ([487, 974, 1947, 9741], ["487", "974", "1947", "9741"])

# (a) K vs energy per token
for camp, cells in gpu.items():
    mk, _ = camps[camp]
    for v in cells:
        a.plot(v["K"], v["canon"], marker=mk, ms=5, mfc=BLUE, mec=SURF, mew=0.6, ls="none", zorder=3)
        a.plot(v["K"], v["dec_mJ"], marker=mk, ms=5, mfc=AQUA, mec=SURF, mew=0.6, ls="none", zorder=3)
for key, col in (("canon", BLUE), ("dec_mJ", AQUA)):
    byK = {}
    for v in gpu["ea_n3"]:
        byK.setdefault(v["K"], []).append(v[key])
    Ks = sorted(byK); a.plot(Ks, [st.mean(byK[k]) for k in Ks], color=col, lw=1.2, zorder=2)
if vanilla:
    a.plot(9741, vanilla["canon"], marker="D", ms=5, mfc=BLUE, mec=SURF, mew=0.6, ls="none", zorder=3)
    a.plot(9741, vanilla["dec_mJ"], marker="D", ms=5, mfc=AQUA, mec=SURF, mew=0.6, ls="none", zorder=3)
    a.annotate("full cache\n(vanilla)", (9741, vanilla["canon"]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5, color=MUTED)
a.set_xscale("log", base=2); a.set_xticks(KT[0]); a.set_xticklabels(KT[1])
a.set_xlabel("cache budget K (cells)"); a.set_ylabel("energy per decoded token (mJ)"); a.set_ylim(0, 480)
a.text(0.02, 0.97, "whole run (prefill + decode) / decoded tokens", transform=a.transAxes, color=BLUE, fontsize=7, va="top")
a.text(0.02, 0.89, "decode phase only", transform=a.transAxes, color=AQUA, fontsize=7, va="top")
a.set_title("(a) GPU: cache budget vs energy per token", loc="left"); style(a)

# (b) K vs throughput, every cell
for camp, cells in gpu.items():
    mk, _ = camps[camp]
    for v in cells:
        b.plot(v["K"], v["tps"], marker=mk, ms=5, mfc=BLUE, mec=SURF, mew=0.6, ls="none", zorder=3)
if vanilla:
    b.plot(9741, vanilla["tps"], marker="D", ms=5, mfc=BLUE, mec=SURF, mew=0.6, ls="none", zorder=3)
b.axhspan(29.0, 31.0, color=GRID, alpha=0.6, lw=0, zorder=1); b.axhspan(37.5, 40.0, color=GRID, alpha=0.6, lw=0, zorder=1)
b.text(9741, 30.0, "slow mode", fontsize=6.5, color=MUTED, ha="right", va="center")
b.text(9741, 38.75, "fast mode", fontsize=6.5, color=MUTED, ha="right", va="center")
b.set_xscale("log", base=2); b.set_xticks(KT[0]); b.set_xticklabels(KT[1])
b.set_xlabel("cache budget K (cells)"); b.set_ylabel("decode throughput (tok/s)"); b.set_ylim(26, 42)
b.set_title("(b) GPU: cache budget vs throughput, every cell", loc="left")
handles = [Line2D([], [], marker=camps[k][0], ls="none", mfc=MUTED, mec=SURF, ms=5, label=camps[k][1]) for k in camps]
handles.append(Line2D([], [], marker="D", ls="none", mfc=MUTED, mec=SURF, ms=5, label="vanilla, full cache"))
b.legend(handles=handles, loc="lower left", ncol=2, handletextpad=0.3, columnspacing=1.0); style(b)

# (c) GPU clock cap vs energy per token at K=1024
clk = []
for dd in sorted(glob.glob("/tmp/clock_sweep/clk_*"), key=lambda p: int(p.split("_")[-1])):
    x = load_cell(dd)
    mm = json.load(open(dd + "/meta.json"))
    clk.append((int(dd.split("_")[-1]), x["mJ"], x["tps"], mm["prefill_ms"] / 1000))
mhz = [q[0] for q in clk]; mj = [q[1] for q in clk]
c.plot(mhz, mj, color=BLUE, lw=1.2, marker="o", ms=5, mfc=BLUE, mec=SURF, mew=0.6, zorder=3)
for i, (m_, j_, t_, p_) in enumerate(clk):
    c.annotate(f"{t_:.1f} tok/s", (m_, j_), xytext=(0, 7 if i % 2 == 0 else 16), textcoords="offset points", ha="center", fontsize=6.5, color=MUTED)
    c.annotate(f"{p_:.0f} s", (m_, j_), xytext=(0, -12 if i % 2 == 0 else -21), textcoords="offset points", ha="center", fontsize=6.5, color=RED)
c.set_xticks(mhz); c.set_xlabel("GPU clock cap (MHz)"); c.set_ylabel("energy per decoded token (mJ)")
c.set_ylim(200, 380); c.set_xlim(680, 1250)
c.text(0.02, 0.97, f"K=1024, n=1 per point. 1200 to 902 MHz: {mj[-1]:.0f} to {mj[2]:.0f} mJ ({100*(1-mj[2]/mj[-1]):.0f}% less),\n"
                   f"decode unchanged, prefill {100*(clk[2][3]/clk[-1][3]-1):.0f}% longer (the trade).",
       transform=c.transAxes, fontsize=6.5, color=MUTED, va="top")
c.text(0.02, 0.80, "grey: decode tok/s   red: prefill time", transform=c.transAxes, fontsize=6.5, color=MUTED, va="top")
c.set_title("(c) GPU: clock cap vs energy per token", loc="left"); style(c)

# (d) request energy split, prefill vs decode
def mean_split(cells):
    pre = st.mean(v["preJ"] for v in cells); dec = st.mean(v["decJ"] for v in cells); return pre, dec
gpu_all = [v for cells in gpu.values() for v in cells]
gp, gd = mean_split(gpu_all); cp, cdc = mean_split(cpu)
xs = [0, 1]
d.bar(xs, [gp, cp], width=0.55, color=MUTED, edgecolor=SURF, linewidth=0.5, zorder=3, label="prefill")
d.bar(xs, [gd, cdc], width=0.55, bottom=[gp, cp], color=[BLUE, AQUA], edgecolor=SURF, linewidth=0.5, zorder=3)
for x, pre, dec, nt in ((0, gp, gd, 4096), (1, cp, cdc, 1024)):
    d.text(x, pre / 2, f"prefill\n{pre/(pre+dec):.0%}", ha="center", va="center", fontsize=6.5, color=SURF)
    d.text(x, pre + dec / 2, f"decode, {nt} tokens\n{dec/(pre+dec):.0%}", ha="center", va="center", fontsize=6, color=SURF)
    d.text(x, pre + dec + 25, f"{(pre+dec):.0f} J", ha="center", va="bottom", fontsize=6.5, color=INK)
d.set_xticks(xs); d.set_xticklabels(["GPU\n(9737-token prompt)", "CPU\n(9737-token prompt)"])
d.set_ylabel("energy per request (J)"); d.set_ylim(0, 1700)
d.set_title("(d) where a request's energy goes", loc="left"); style(d); d.grid(axis="x", visible=False)

# (e) what the v1 controller did per battery state
arms = [("gpu_healthy", "GPU\nhealthy"), ("gpu_mid", "GPU\nmid"), ("gpu_low", "GPU\nlow"),
        ("cpu_healthy", "CPU\nhealthy"), ("cpu_low", "CPU\nlow")]
vals = []
for tag, _ in arms:
    cell_dir = f"/tmp/ea_proof/{tag}"
    vals.append(load_cell(cell_dir) if os.path.exists(cell_dir + "/meta.json") else None)
xs = list(range(len(arms)))
cols = [BLUE if t.startswith("gpu") else AQUA for t, _ in arms]
e.bar(xs, [v["mJ"] if v else 0 for v in vals], width=0.62, color=cols, edgecolor=SURF, linewidth=0.5, zorder=3)
for x, v, (tag, _) in zip(xs, vals, arms):
    if v:
        e.text(x, v["mJ"] + 12, f"K={v['K']}\n{v['tps']:.1f} tok/s", ha="center", va="bottom", fontsize=6.5, color=INK)
ref = {"gpu": vals[0]["mJ"], "cpu": vals[3]["mJ"]}
for x, v, (tag, _) in zip(xs, vals, arms):
    if v and not tag.endswith("healthy"):
        e.text(x, 40, f"{(v['mJ'] / ref[tag[:3]] - 1):+.0%}", ha="center", fontsize=6.5, color=SURF, fontweight="bold")
e.set_xticks(xs); e.set_xticklabels([l for _, l in arms])
e.set_ylabel("energy per decoded token (mJ)"); e.set_ylim(0, 1400)
e.set_title("(e) v1 controller, same command line, battery state varied", loc="left")
e.text(0.02, 0.97, "CPU never adapted (K=487 at every state).", transform=e.transAxes,
       ha="left", va="top", fontsize=6.5, color=MUTED)
style(e); e.grid(axis="x", visible=False)

# (f) quality per tier
tiers = [("kpct5", 236), ("kpct10", 473), ("kpct20", 945)]
for task, col, off in (("hotpotqa", BLUE, 3), ("qasper", AQUA, -9)):
    xs_, ys_ = [], []
    for arm, K in tiers:
        r = tq.get(f"{arm}|{task}")
        if r:
            xs_.append(K); ys_.append(r["f1"])
    f.plot(xs_, ys_, color=col, lw=1.2, marker="o", ms=5, mfc=col, mec=SURF, mew=0.6, zorder=3)
    f.text(xs_[-1] * 1.06, ys_[-1], task, color=col, fontsize=7, va="center")
    for x, y in zip(xs_, ys_):
        f.annotate(f"{y:.1f}", (x, y), xytext=(0, off), textcoords="offset points", ha="center", fontsize=6.5, color=col)
full = tq.get("llama1b__vanilla|hotpotqa")
if full:
    f.axhline(full["f1"], color=BLUE, lw=0.8, ls=(0, (3, 2)), zorder=2)
    f.text(236, full["f1"] + 0.8, f"hotpotqa, full cache {full['f1']:.1f}", color=BLUE, fontsize=6.5, va="bottom")
f.set_xscale("log", base=2); f.set_xticks([236, 473, 945]); f.set_xticklabels(["236", "473", "945"]); f.set_xlim(200, 1300)
f.set_xlabel("cache budget K (cells)"); f.set_ylabel("LongBench F1 (n=15 per point)"); f.set_ylim(0, 45)
f.set_title("(f) answer quality per tier, Llama-1B", loc="left"); style(f)

fig.text(0.085, 0.012, "OnePlus 15, Llama-3.2-1B-Instruct Q4_K_M. Energy = USB rail + battery pack. No cell in (a), (b), (e) pinned the GPU clock.",
         fontsize=7, color=MUTED, ha="left", va="bottom")
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/fig_energy_aware_pure"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
print("wrote", out + ".pdf/.png")
print("(c) clock:", [(m_, f"{j_:.0f} mJ", f"{t_:.1f} tps") for m_, j_, t_, _ in clk])
print(f"(d) split: GPU prefill {gp:.0f} J decode {gd:.0f} J ({gp/(gp+gd):.0%} prefill); CPU prefill {cp:.0f} J decode {cdc:.0f} J ({cp/(cp+cdc):.0%})")
