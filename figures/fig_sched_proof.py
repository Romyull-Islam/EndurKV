#!/usr/bin/env python3
"""The scheduler, measured: the same prompt through ukv_sched.sh under forced battery states.
The control arm is the scheduler told it is on mains (full-performance plan, 4096 tokens).
(a) total energy per request, predicted by the scheduler's table vs measured on the phone.
(b) total time per request, prefill + decode.
(c) energy per output token, which removes the output-cap effect and leaves the clock effect.
From /tmp/sched_proof (n=2 per arm, CPU caps pinned, cooled per cell).
"""
import os, sys, glob, json, re, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/mislam22/EndurKV_workspace/EndurKV/scripts")
from ea_proof_summary import load as load_cell

BLUE, AQUA, RED, MUTEDB = "#2a78d6", "#1baf7a", "#e34948", "#9ec5f4"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

arms_order = [("control", "control\n4096"), ("sched_healthy", "80%\n4096"), ("sched_mid", "40%\n1024 cap"), ("sched_low", "15%\n512 cap"), ("sched_low_nocap", "15%\n4096 fix")]
pred = {}
for line in open("/tmp/sched_proof/ukv_sched.log"):
    m = re.search(r"tag=(\S+) .* pred_J=(\d+)", line)
    if m: pred[m.group(1)] = float(m.group(2))
cells = {}
for d in sorted(glob.glob("/tmp/sched_proof/*_r[0-9]")):
    if not os.path.exists(d + "/meta.json"): continue
    x = load_cell(d); m = json.load(open(d + "/meta.json"))
    tag = os.path.basename(d); arm = tag.rsplit("_r", 1)[0]
    cells.setdefault(arm, []).append(dict(E=x["prefill_J"] + x["decode_J"], pre=x["prefill_s"], dec=x["decode_s"], no=m["n_decode_steps"],
                                          np=m["n_prompt_tokens"], pred=pred.get(tag, float("nan")), gpu=x["gpu_mhz"], tps=x["tps"]))
def ms(v):
    v = [a for a in v if a == a]; return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0, len(v)) if v else (float("nan"), 0, 0)

fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 3.6))
fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.31, wspace=0.42)
for ax in (a, b, c):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.5); ax.grid(axis="x", visible=False)
xs = list(range(len(arms_order))); labels = [l for _, l in arms_order]
E = [ms([k["E"] for k in cells.get(a_, [])]) for a_, _ in arms_order]
P = [ms([k["pred"] for k in cells.get(a_, [])]) for a_, _ in arms_order]
ref = E[0][0]
a.bar([x - 0.2 for x in xs], [p[0] for p in P], width=0.38, color=MUTEDB, edgecolor=SURF, linewidth=0.5, zorder=3, label="predicted by the table")
a.bar([x + 0.2 for x in xs], [e[0] for e in E], width=0.38, color=BLUE, edgecolor=SURF, linewidth=0.5, yerr=[e[1] for e in E], capsize=2, ecolor=INK, zorder=3, label="measured (n=2)")
for x, e in zip(xs, E):
    a.text(x + 0.2, e[0] + 30, f"{e[0]:.0f} J" + (f"\n{e[0]/ref-1:+.0%}" if x else ""), ha="center", va="bottom", fontsize=6.3, color=INK)
a.set_xticks(xs); a.set_xticklabels(labels, fontsize=5.8); a.set_ylabel("total energy per request (J)"); a.set_ylim(0, 2150)
a.set_title("(a) energy: prediction vs meter", loc="left")
a.text(0.02, 0.97, "GPU chosen\nin every arm", transform=a.transAxes, fontsize=6.0, color=MUTED, va="top"); a.legend(loc="upper center", bbox_to_anchor=(0.68, 1.0), handlelength=1.2)
pre = [ms([k["pre"] for k in cells.get(a_, [])])[0] for a_, _ in arms_order]; dec = [ms([k["dec"] for k in cells.get(a_, [])])[0] for a_, _ in arms_order]
b.bar(xs, pre, width=0.6, color=MUTED, edgecolor=SURF, linewidth=0.5, zorder=3)
b.bar(xs, dec, width=0.6, bottom=pre, color=BLUE, edgecolor=SURF, linewidth=0.5, zorder=3)
for x, p_, d_ in zip(xs, pre, dec):
    b.text(x, p_ + d_ + 4, f"{p_+d_:.0f} s" + (f"\n{(p_+d_)/(pre[0]+dec[0])-1:+.0%}" if x else ""), ha="center", va="bottom", fontsize=6.3, color=INK)
b.set_xticks(xs); b.set_xticklabels(labels, fontsize=5.8); b.set_ylabel("total time per request (s)"); b.set_ylim(0, 340)
b.set_title("(b) time: prefill (grey) + decode", loc="left")
mj = [ms([k["E"] * 1000 / k["no"] for k in cells.get(a_, [])]) for a_, _ in arms_order]
c.bar(xs, [m_[0] for m_ in mj], width=0.6, color=AQUA, edgecolor=SURF, linewidth=0.5, yerr=[m_[1] for m_ in mj], capsize=2, ecolor=INK, zorder=3)
for x, m_ in zip(xs, mj):
    c.text(x, m_[0] + 15, f"{m_[0]:.0f}", ha="center", va="bottom", fontsize=6.3, color=INK)
c.set_xticks(xs); c.set_xticklabels(labels, fontsize=5.8); c.set_ylabel("energy per output token, whole run (mJ)"); c.set_ylim(0, 1350)
c.set_title("(c) energy per output token", loc="left")
c.text(0.02, 0.97, "prefill is a fixed cost:\nfewer output tokens\nraise this number.\nCompare equal-token\narms only: control, 80%,\n15% 4096 fix. The cap\nalone: -13%.", transform=c.transAxes, fontsize=6.0, color=MUTED, va="top")
fig.text(0.075, 0.10, "arms, as battery state told to the scheduler and the plan it chose:  control = mains, GPU 1200 MHz, 4096 tokens.   80% = GPU 1200 MHz, 4096 tokens.\n"
                      "40% = GPU cap 902 MHz, output cap 1024.   15% = GPU cap 902 MHz, output cap 512.   15% 4096 fix = told 15%, caller fixed the length: clock cap only.\n"
                      "The CPU is chosen only when the GPU is unavailable (see the decision map).",
         fontsize=6.3, color=INK, ha="left", va="bottom")
fig.text(0.075, 0.02, "OnePlus 15, Llama-3.2-1B, 9737-token prompt. Only the battery state was forced. n=2 per arm, CPU caps pinned, cooled per cell.",
         fontsize=6.3, color=MUTED, ha="left", va="bottom")
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/fig_sched_proof"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
print("wrote", out)
for (a_, _), e, p_, m_ in zip(arms_order, E, P, mj):
    k = cells.get(a_, [])
    print(f"  {a_:16s} n={e[2]} measured {e[0]:6.0f}±{e[1]:.0f} J  predicted {p_[0]:6.0f} J  ({e[0]/p_[0]-1:+.0%})  time {ms([x['pre']+x['dec'] for x in k])[0]:5.0f} s  {m_[0]:5.0f} mJ/output token  gpu {ms([x['gpu'] for x in k])[0]:.0f} MHz")
