#!/usr/bin/env python3
"""Column figure with three rows: (a) energy to performance and (b) performance to energy, each
for the GPU clock ladder and the cache budget, and (c) the discharge lever trace and the guarded
bandit's per-request energy prediction error. Data: energy_perf_data.json, /tmp/discharge3,
campaigns/guarded_bandit_20260921/requests/*/sched_log.txt.
"""
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "energy_perf_data.json")))
INK, MUTED, GRID, AXIS = "#1a1a1a", "#6b6b6b", "#e6e6e6", "#b8b8b8"
DOWN, UP, MU = "#0072B2", "#D55E00", "#009E73"
PERF, ENER = "#D55E00", "#0072B2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.0, "axes.labelsize": 6.0,
                     "xtick.labelsize": 5.6, "ytick.labelsize": 5.6, "pdf.fonttype": 42})

fig, axs = plt.subplots(3, 2, figsize=(3.33, 2.76), gridspec_kw=dict(width_ratios=[1.0, 1.0], height_ratios=[1, 1, 1.05]))
fig.subplots_adjust(left=0.115, right=0.985, top=0.94, bottom=0.08, wspace=0.48, hspace=1.0)
(a1, a2), (b1, b2), (c1, c2) = axs


def row_title(ax, text):
    ax.text(-0.19, 1.27, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4, color=INK, weight="bold")


def style(ax):
    ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=1.5, pad=1.5)


# data
lad = sorted(D["ladder_fine"]["points"], key=lambda p: p["mhz"])
top = next(p for p in lad if p["mhz"] == 1200); low = next(p for p in lad if p["mhz"] == 902)
pm = D["accuracy"]["per_model"]
NAME = {"Llama-3.2-1B": "Llama", "Phi-3-mini": "Phi-3", "gemma-2-2b": "gemma", "Bonsai-8B": "Bonsai"}

# (a) energy -> performance
row_title(a1, "(a) energy → performance: spend less energy")
xs = [100 * (1 - p["energy_J"] / top["energy_J"]) for p in lad]
ys = [100 * (p["time_s"] / top["time_s"] - 1) for p in lad]
order = sorted(range(len(lad)), key=lambda i: lad[i]["mhz"], reverse=True)
a1.plot([xs[i] for i in order if lad[i]["mhz"] >= 902], [ys[i] for i in order if lad[i]["mhz"] >= 902], color=DOWN, lw=0.9, zorder=2)
for i, p in enumerate(lad):
    dom = p["mhz"] == D["ladder_fine"]["dominated"]
    a1.scatter(xs[i], ys[i], s=11, color="white" if dom else DOWN, edgecolor=DOWN, lw=0.7, zorder=4)
    dx, dy = {1200: (1.5, 2.5), 1050: (1.5, -3.0), 967: (-1.2, 4.5), 902: (-1.2, 4.5), 826: (1.5, -1.0)}[p["mhz"]]
    a1.text(xs[i] + dx, ys[i] + dy, f"{p['mhz']}", fontsize=5.2, color=MUTED if dom else INK, ha="left" if dx > 0 else "right", va="center")
    if p["mhz"] == 967:
        a1.scatter(xs[i], ys[i], s=48, facecolor="none", edgecolor=INK, lw=0.7, zorder=5)
a1.set_xlim(-2, 34); a1.set_ylim(-3, 45)
a1.set_xlabel("energy saved (%)", labelpad=1); a1.set_ylabel("time added (%)", labelpad=1)
a1.set_title("GPU clock, from 1200 MHz", loc="left", fontsize=5.8, color=MUTED, pad=2)

for p in pm:
    x, y = 100 - p["energy_pct"], max(0.0, 100 - p["acc_kept"])
    a2.scatter(x, y, s=11, color=DOWN, edgecolor="white", lw=0.4, zorder=4)
    ha, dx, dy = ("right", -1.2, 0.0) if NAME[p["model"]] == "Phi-3" else ("left", 1.2, 0.0)
    if NAME[p["model"]] == "Llama": dy = 0.9
    a2.text(x + dx, y + dy, NAME[p["model"]], fontsize=5.2, color=INK, ha=ha, va="center")
a2.set_xlim(30, 76); a2.set_ylim(-1.0, 9)
a2.set_xlabel("energy saved (%)", labelpad=1); a2.set_ylabel("accuracy lost (%)", labelpad=1)
a2.set_title("cache, full → $K$ = 1024", loc="left", fontsize=5.8, color=MUTED, pad=2)

# (b) performance -> energy
row_title(b1, "(b) performance → energy: buy it back")
xs = [100 * (low["time_s"] / p["time_s"] - 1) for p in lad]
ys = [100 * (p["energy_J"] / low["energy_J"] - 1) for p in lad]
keep = [i for i in range(len(lad)) if lad[i]["mhz"] >= 902]
b1.plot([xs[i] for i in keep], [ys[i] for i in keep], color=UP, lw=0.9, zorder=2)
for i in keep:
    p = lad[i]
    b1.scatter(xs[i], ys[i], s=11, color=UP, edgecolor="white", lw=0.4, zorder=4)
    dx, dy = {1200: (-1.5, 0.0), 1050: (-1.5, 1.0), 967: (-1.5, 3.0), 902: (1.8, -3.0)}[p["mhz"]]
    b1.text(xs[i] + dx, ys[i] + dy, f"{p['mhz']}", fontsize=5.2, color=INK, ha="right" if dx < 0 else "left", va="center")
b1.set_xlim(-2, 34); b1.set_ylim(-3, 45)
b1.set_xlabel("throughput gained (%)", labelpad=1); b1.set_ylabel("energy added (%)", labelpad=1)
b1.set_title("GPU clock, from 902 MHz", loc="left", fontsize=5.8, color=MUTED, pad=2)

for p in pm:
    x, y = max(0.0, 100 - p["acc_kept"]), p["buy_back_energy_pct"]
    b2.scatter(x, y, s=11, color=UP, edgecolor="white", lw=0.4, zorder=4)
    dx, dy, ha = 0.35, 0.0, "left"
    if NAME[p["model"]] == "Llama": dy = 14
    if NAME[p["model"]] == "Phi-3": dy = -14
    b2.text(x + dx, y + dy, NAME[p["model"]], fontsize=5.2, color=INK, ha=ha, va="center")
b2.set_xlim(-0.6, 9); b2.set_ylim(0, 225)
b2.set_xlabel("accuracy bought back (%)", labelpad=1); b2.set_ylabel("energy added (%)", labelpad=1)
b2.set_title("cache, $K$ = 1024 → full", loc="left", fontsize=5.8, color=MUTED, pad=2)

# (c) the learner -> the two loops
row_title(c1, "(c) the learner → the two loops")
rows = [r for r in csv.DictReader(open("/tmp/discharge3/discharge3/timeline.csv")) if r["usb_powered"] == "false"]
n = [int(r["n"]) for r in rows]; L = [float(r["lever"]) + float(r["bias_before"] or 0) for r in rows]
soc = [int(r["soc"]) for r in rows]; tier = [r["tier"] for r in rows]
BAND = {"healthy": "#eef3fb", "mid": "#fdf3e4", "low": "#fce9ea"}; DEF = {"healthy": 1.0, "mid": 0.5, "low": 0.0}
i0 = 0
for i in range(1, len(rows) + 1):
    if i == len(rows) or tier[i] != tier[i0]:
        c1.axvspan(n[i0] - 0.5, n[i - 1] + 0.5, color=BAND[tier[i0]], zorder=0, lw=0)
        c1.plot([n[i0] - 0.5, n[i - 1] + 0.5], [DEF[tier[i0]]] * 2, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        xl = n[i - 1] + 0.3 if tier[i0] == "low" else n[i0] + 0.5
        c1.text(xl, 2.36, f"{tier[i0]}\n{soc[i0]} to {soc[i - 1]}%", ha="right" if tier[i0] == "low" else "left", va="top", fontsize=4.6, color=MUTED, linespacing=1.1)
        i0 = i
c1.step(n, L, where="mid", color=INK, lw=0.9, zorder=3)
np_, ne_ = 0, 0
for r, x, y in zip(rows, n, L):
    if r["loop"].startswith("performance"):
        np_ += 1; c1.scatter(x, y, marker="^", s=13, color=PERF, edgecolor="white", linewidth=0.4, zorder=5)
    elif r["loop"].startswith("energy"):
        ne_ += 1; c1.scatter(x, y, marker="v", s=13, color=ENER, edgecolor="white", linewidth=0.4, zorder=5)
h = [plt.Line2D([], [], color=INK, lw=0.9, label="lever + bias"),
     plt.Line2D([], [], marker="^", ls="", color=PERF, ms=3, label="time over +0.1"),
     plt.Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 2)), label="tier default"),
     plt.Line2D([], [], marker="v", ls="", color=ENER, ms=3, label="energy over −0.1")]
c1.legend(handles=h, loc="lower left", fontsize=4.6, frameon=False, ncol=2, handlelength=0.9, columnspacing=0.5,
          borderpad=0.1, labelspacing=0.15, handletextpad=0.3, bbox_to_anchor=(-0.04, -0.04))
c1.set_ylim(-0.68, 2.40); c1.set_yticks([0, 0.5, 1.0]); c1.set_xlim(n[0] - 0.5, n[-1] + 0.5)
c1.set_xlabel("request on battery", labelpad=1); c1.set_ylabel("lever $L$ + bias", labelpad=1)
c1.set_title("discharge, cycle 3", loc="left", fontsize=5.8, color=MUTED, pad=2)

G = D["guarded_bandit"]
err = {}
for d in glob.glob(os.path.join(HERE, "..", "..", "..", "..", "campaigns", "guarded_bandit_20260921", "requests", "*", "sched_log.txt")):
    kv = dict(re.findall(r"(\w+)=(\S+)", open(d).read()))
    m = re.match(r"gb(\d+)_", os.path.basename(os.path.dirname(d)))
    if m and "meas_J" in kv:
        err[int(m.group(1))] = 100 * abs(float(kv["meas_J"]) - float(kv["pred_J"])) / float(kv["pred_J"])
assert len(err) == G["n"] == 30, len(err)
MODE = {"explore": "#9a9a9a", "go": MU, "fallback": "#CC79A7"}
c2.axhspan(0, 5, color="#e3efe8", zorder=0, lw=0)
fired = 0
for r in G["per_request"]:
    i = r["i"]; c2.bar(i, err[i], color=MODE[r["kind"]], width=0.8, zorder=2)
    if r["loop"] != "hold":
        fired += 1; c2.scatter(i, err[i] + 0.9, marker="^", s=13, color=PERF, edgecolor="white", linewidth=0.4, zorder=5)
hb = [plt.Rectangle((0, 0), 1, 1, color=MODE[k], label=k) for k in ("explore", "go", "fallback")]
hb.append(plt.Line2D([], [], marker="^", ls="", color=PERF, ms=3, label="a loop fired"))
c2.legend(handles=hb, loc="upper right", fontsize=4.9, frameon=False, ncol=2, handlelength=1.0,
          columnspacing=0.6, borderpad=0.1, labelspacing=0.2, handletextpad=0.4, bbox_to_anchor=(1.03, 1.04))
c2.text(-0.8, 5.1, "5%", ha="left", va="bottom", fontsize=5.0, color=MUTED)
c2.set_xlim(-1, 30); c2.set_ylim(0, 12.6); c2.set_yticks([0, 4, 8, 12]); c2.set_xticks([0, 10, 20, 29]); c2.set_xticklabels(["1", "11", "21", "30"])
c2.set_xlabel("request, guarded learner", labelpad=1); c2.set_ylabel("|energy error| (%)", labelpad=1)
c2.set_title("on the cable, learner on", loc="left", fontsize=5.8, color=MUTED, pad=2)

for ax in axs.flat:
    style(ax)
c1.grid(axis="x", visible=False); c2.grid(axis="x", visible=False)

out = os.path.join(HERE, "fig_three_relations_col")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("wrote", out, "| perf events", np_, "energy events", ne_, "| bandit fired", fired)
