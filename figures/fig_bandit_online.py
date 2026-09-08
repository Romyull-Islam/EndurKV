#!/usr/bin/env python3
"""The online bandit on the phone: 24 real pulls, GPU clock arms by battery tier, reward from the
meter. (a) the pull sequence: which arm was pulled in which context and the reward it earned;
(b) per-context mean reward per arm at the end (the learned value table), with the offline
table's ranking for comparison; (c) measured energy and time per arm per context."""
import json, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BLUE, MID, LIGHT, RED = "#0d366b", "#2a78d6", "#9ec5f4", "#e34948"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
ARMC = {1200: BLUE, 902: MID, 726: LIGHT}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "pdf.fonttype": 42})
s = json.load(open("/tmp/bandit_online/state.json")); h = [x for x in s["history"] if not x.get("failed")]
tiers = ["healthy", "mid", "low"]; ty = {t: i for i, t in enumerate(tiers)}
fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 3.0)); fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.30, wspace=0.42)
for ax in (a, b, c):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.5, width=0.5)
# (a) sequence
for x in h:
    a.scatter(x["i"], x["r"], s=22, color=ARMC[x["mhz"]], edgecolor=SURF, linewidth=0.4, zorder=3, marker="o" if x["why"] != "warm start" else "s")
    a.annotate(x["tier"][0], (x["i"], x["r"]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=5.5, color=MUTED)
a.axvline(8.5, color=MUTED, lw=0.6, ls=(0, (3, 2))); a.text(8.7, 0.21, "warm start ends", fontsize=6, color=MUTED, va="top"); a.set_ylim(-0.82, 0.24)
a.set_ylabel("reward (tier weights, metered E and T)"); a.set_title("(a) the 24 pulls: arm, context, reward", loc="left")
a.legend(handles=[Line2D([], [], marker="o", ls="none", color=ARMC[m], label=f"{m} MHz") for m in (1200, 902, 726)] +
         [Line2D([], [], marker="s", ls="none", color=MUTED, label="warm start pull")], loc="lower right", ncol=2, handlelength=1.0, columnspacing=0.8, handletextpad=0.3, fontsize=6)
a.set_xlabel("pull (letter = context: h healthy, m mid, l low)")
# (b) learned values per context
w = 0.26
for j, m in enumerate((1200, 902, 726)):
    vals = []; ns = []
    for t in tiers:
        cc = [x["r"] for x in h if x["tier"] == t and x["mhz"] == m]
        vals.append(st.mean(cc) if cc else float("nan")); ns.append(len(cc))
    xs = [ty[t] + (j - 1) * w for t in tiers]
    b.bar(xs, vals, width=w, color=ARMC[m], edgecolor=SURF, linewidth=0.5, zorder=3, label=f"{m} MHz")
    for x_, v, n in zip(xs, vals, ns):
        b.text(x_, v + (0.012 if v >= 0 else -0.012), f"n={n}", ha="center", va="bottom" if v >= 0 else "top", fontsize=5.6, color=MUTED)
b.set_xticks(range(3)); b.set_xticklabels(tiers); b.set_ylabel("mean reward per arm"); b.grid(axis="x", visible=False)
b.set_title("(b) learned values per context", loc="left"); b.legend(loc="lower left", handlelength=1.2); b.set_ylim(-0.72, 0.40)
b.text(0.98, 0.97, "greedy arm: healthy 1200,\nmid 902, low 902 (726 within 0.002):\nthe rule's choices", transform=b.transAxes, fontsize=6.2, color=INK, ha="right", va="top")
# (c) measured E and T per arm (all contexts share the request, so pool)
for j, m in enumerate((1200, 902, 726)):
    cc = [x for x in h if x["mhz"] == m]
    E = [x["E"] for x in cc]; T = [x["T"] for x in cc]
    c.errorbar(st.mean(T), st.mean(E), xerr=st.pstdev(T) if len(T) > 1 else 0, yerr=st.pstdev(E) if len(E) > 1 else 0, color=ARMC[m], marker="o", ms=6, mec=SURF, capsize=2, ls="none", zorder=3)
    off = {1200: (7, 0, "left", "center"), 902: (-8, 8, "right", "bottom"), 726: (-8, -8, "right", "top")}[m]
    c.annotate(f"{m} MHz, n={len(cc)}\n{st.mean(E):.0f} J, {st.mean(T):.0f} s", (st.mean(T), st.mean(E)), xytext=off[:2], textcoords="offset points", fontsize=6.2, color=ARMC[m], ha=off[2], va=off[3])
c.set_xlim(110, 275); c.set_ylim(480, 880); c.set_xlabel("total time per request (s)"); c.set_ylabel("total energy per request (J)")
c.set_title("(c) cost per arm, pooled", loc="left")
fig.text(0.08, 0.02, "OnePlus 15, Llama-3.2-1B, 9737-token prompt, 1024 output tokens, GPU, K=1024. One real request per pull, cooled and pinned. "
         "The seed happened to draw no exploration step after the warm start, so the non-greedy arms rest on one pull per context.", fontsize=6.2, color=MUTED, ha="left", va="bottom", wrap=True)
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/fig_bandit_online"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); print("wrote", out)
