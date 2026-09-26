#!/usr/bin/env python3
"""The three relations the supervisor asked for, in one figure. (2026-09-24)

  (a) energy -> performance: spend less energy and see what performance costs. Two levers:
      the GPU clock (1200 -> 902 MHz: -28% energy, +28% time, same answer) and the cache
      (full -> K=1024: -40 to -66% energy, 1.7 to 2.6x faster, 0 to 7.5% accuracy lost).
  (b) performance -> energy: demand more performance and see what energy costs. The same two
      levers read upward: 902 -> 1200 MHz buys +28% throughput for +38% energy; buying back the
      last 0 to 7.5% of accuracy with the full cache costs +67 to +196% energy. The asymmetry is
      the point: what is cheap to give up is dear to buy back.
  (c) the learner -> the two loops: as the cost table learns, prediction error shrinks and the
      loops have less to do. Left: the real discharge, |time error| per request with loop
      actions marked; the cable-seeded table starts 39% wrong and is under 5% five requests
      later. Right: the guarded bandit's 30 requests, |energy error| per request coloured by its
      mode; the loops fired on 3 of 30, one in each mode.
Data: energy_perf_data.json (ladder_fine, accuracy, discharge, guarded_bandit) and the guarded
bandit's own scheduler logs (pred_J / meas_J per request).
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "energy_perf_data.json")))
INK, MUTED, GRID, AXIS = "#1a1a1a", "#5f6b73", "#e6e9eb", "#b8c0c6"
Q1C, Q2C, MU = "#0072B2", "#D55E00", "#009E73"     # blue: spend less; vermilion: buy more; green: muKV
MODE = {"explore": "#9a9a9a", "go": MU, "fallback": "#CC79A7"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.labelsize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42})

fig = plt.figure(figsize=(14.0, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.6], wspace=0.40, left=0.045, right=0.99, top=0.82, bottom=0.15)
axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
gsc = gs[2].subgridspec(1, 2, wspace=0.36, width_ratios=[1.4, 1])
axC1, axC2 = fig.add_subplot(gsc[0]), fig.add_subplot(gsc[1])

# ---------------------------------------------------------------- the two levers, read both ways
lad = {p["mhz"]: p for p in D["ladder_fine"]["points"]}
E1200, E902, T1200, T902 = lad[1200]["energy_J"], lad[902]["energy_J"], lad[1200]["time_s"], lad[902]["time_s"]
down_e, down_t = 100 * (1 - E902 / E1200), 100 * (T902 / T1200 - 1)      # clock read downward
up_thr, up_e = 100 * (T902 / T1200 - 1), 100 * (E1200 / E902 - 1)        # clock read upward (throughput = 1/T)
pm = D["accuracy"]["per_model"]
cache_e = [100 - p["energy_pct"] for p in pm]
cache_acc = [max(0.0, 100 - p["acc_kept"]) for p in pm]
cache_fast = [100 / p["time_pct"] for p in pm]
buy = [p["buy_back_energy_pct"] for p in pm]


def lever_panel(ax, title, sub, rows, xlabel, xmax, color):
    """One horizontal bar per lever: the cause is the bar, the effect is written at its end."""
    ys = [1, 0]
    for y, (lab, x, xtext, effect) in zip(ys, rows):
        ax.barh(y, x, color=color, height=0.5)
        ax.text(x + xmax * 0.015, y + 0.02, xtext, va="bottom", ha="left", fontsize=8.2, color=color, weight="bold")
        ax.text(x + xmax * 0.015, y - 0.03, effect, va="top", ha="left", fontsize=8.2, color=INK, linespacing=1.25)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylim(-0.6, 1.6); ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=24)
    ax.text(0, 1.035, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color=MUTED, style="italic")
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)


lever_panel(axA, "(a) energy → performance", "spend less energy: cheap to give up",
            [("GPU clock\n1200 → 902 MHz", down_e, f"−{down_e:.0f}% energy", f"+{down_t:.0f}% time, same answer"),
             ("cache\nfull → K = 1024", max(cache_e), f"−{min(cache_e):.0f} to −{max(cache_e):.0f}% energy",
              f"{min(cache_fast):.1f} to {max(cache_fast):.1f}× faster\n{min(cache_acc):.0f} to {max(cache_acc):.1f}% accuracy lost\n(four models)")],
            "energy saved (%)", 120, Q1C)
lever_panel(axB, "(b) performance → energy", "buy more performance: dear to buy back",
            [("GPU clock\n902 → 1200 MHz", up_e, f"+{up_e:.0f}% energy", f"for +{up_thr:.0f}% throughput"),
             ("cache\nK = 1024 → full", max(buy), f"+{min(buy)} to +{max(buy)}% energy",
              f"for the last {min(cache_acc):.0f} to {max(cache_acc):.1f}%\nof accuracy\n(four models)")],
            "energy spent (%)", 330, Q2C)

# ---------------------------------------------------------------- (c) the learner and the two loops
pr = D["discharge"]["per_request"]; tiers = D["discharge"]["tiers"]
xs = [r["i"] for r in pr]; te = [abs(r["time_err"]) for r in pr]
bounds = {}
for r in pr:
    bounds.setdefault(r["tier"], [r["i"], r["i"]]); bounds[r["tier"]][1] = r["i"]
TB = D["discharge"]["tiers"]
for t, (c, lab) in {"healthy": ("#e9eef6", "healthy"), "mid": ("#fbf3e3", "mid"), "low": ("#fbe8e6", "low")}.items():
    lo, hi = bounds[t]; axC1.axvspan(lo - 0.5, hi + 0.5, color=c, zorder=0)
    axC1.text((lo + hi) / 2, 79, f"{lab}\nfired {TB[t]['loops_fired']}/{TB[t]['n']}\nmean {TB[t]['mean_abs_time_err']:g}%",
              ha="center", va="top", fontsize=7.2, color=MUTED, linespacing=1.25)
axC1.axhspan(0, 5, color="#dfeee6", zorder=1)
axC1.plot(xs, te, color=INK, lw=1.1, marker="o", ms=2.4, zorder=3)
fired = [(r["i"], abs(r["time_err"])) for r in pr if r["loop"] != "hold"]
axC1.scatter([f[0] for f in fired], [f[1] for f in fired], s=28, marker="^", color=Q2C, zorder=4)
axC1.text(xs[-1] + 0.5, 6.5, "5% tolerance", ha="right", va="bottom", fontsize=7.0, color=MUTED)
axC1.set_xlim(0, xs[-1] + 1); axC1.set_ylim(0, 80); axC1.set_yticks([0, 20, 40, 60])
axC1.set_xlabel("request on battery"); axC1.set_ylabel("|time prediction error| (%)")
axC1.set_title("(c) the learner \u2192 the two loops   (\u25b2 a loop fired)", loc="left", fontsize=10.5, color=INK, pad=24)
axC1.text(0, 1.035, "real discharge, no bandit, cable-seeded table", transform=axC1.transAxes,
          ha="left", va="bottom", fontsize=8.6, color=MUTED, style="italic")

G = D["guarded_bandit"]["per_request"]
err = {}
for d in glob.glob(os.path.join(HERE, "..", "..", "..", "..", "campaigns", "guarded_bandit_20260921", "requests", "*", "sched_log.txt")):
    kv = dict(re.findall(r"(\w+)=(\S+)", open(d).read()))
    m = re.match(r"gb(\d+)_", os.path.basename(os.path.dirname(d)))
    if m and "meas_J" in kv:
        err[int(m.group(1))] = 100 * abs(float(kv["meas_J"]) - float(kv["pred_J"])) / float(kv["pred_J"])
axC2.axhspan(0, 5, color="#dfeee6", zorder=1)
for r in G:
    i = r["i"]
    axC2.bar(i, err.get(i, 0), color=MODE[r["kind"]], width=0.8, zorder=2)
    if r["loop"] != "hold":
        axC2.scatter([i], [err.get(i, 0) + 0.5], s=28, marker="^", color=Q2C, zorder=4)
h = [plt.Rectangle((0, 0), 1, 1, color=MODE[k], label=k) for k in ("explore", "go", "fallback")]
h.append(plt.Line2D([], [], marker="^", ls="", color=Q2C, label="a loop fired"))
axC2.legend(handles=h, loc="upper right", frameon=False, fontsize=7.8, ncol=2, columnspacing=0.8, handlelength=1.0, borderaxespad=0.2)
counts = D["guarded_bandit"]["counts"]; lf = D["guarded_bandit"]["loops_fired"]
axC2.set_ylim(0, 12); axC2.set_xlim(-1, 30)
axC2.set_xlabel("request (three tiers cycled)"); axC2.set_ylabel("|energy prediction error| (%)")
axC2.text(0.02, 0.77, f"loops fired {sum(v[0] for v in lf.values())} of 30,\none per mode", transform=axC2.transAxes, fontsize=7.8, color=INK, ha="left", va="top", linespacing=1.25)
axC2.text(0, 1.035, "guarded bandit, on the cable",
          transform=axC2.transAxes, ha="left", va="bottom", fontsize=8.6, color=MUTED, style="italic")

for ax in (axA, axB, axC1, axC2):
    ax.tick_params(colors=MUTED, length=2)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
for ax in (axC1, axC2):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

out = os.path.join(HERE, "fig_three_relations")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.03); fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("wrote", out, "| clock down", round(down_e, 1), round(down_t, 1), "| clock up", round(up_thr, 1), round(up_e, 1),
      "| cache", [round(x) for x in cache_e], [round(x, 1) for x in cache_acc], buy, "| bandit errors", len(err))
