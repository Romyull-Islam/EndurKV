#!/usr/bin/env python3
"""The two loops: (a) acting on their own through a real battery discharge, (b) under the guarded learner.

Job: show the feedback loops working without any provoked disturbance: the tier sets the
lever, and the performance and energy loops move its bias when the metered request misses
its time or energy budget. x is the request number on battery, y the lever the scheduler
used (tier default plus bias). Tier defaults are drawn as steps; every loop action is a
marker (up: performance loop over time budget; down: energy loop over energy budget).

Data: /tmp/discharge_final/discharge/timeline.csv (phone_discharge_loop.sh, 2026-09-05/06),
123 requests from 91 to 12%, the scheduler's own log of lever, bias and loop decision.
Requests on the cable are left out; the plot starts at the first request on battery.
(b) The guarded bandit's 30 cooled requests on the cable (campaigns/guarded_bandit_20260921): the
absolute energy prediction error of each request from its own sched_log (pred_J, meas_J), coloured by
the learner's mode for that request; a marker where a loop fired. This is the "RL -> two loops"
relation: the loops fired on 3 of 30, one in each mode. (2026-09-24)
"""
import csv, glob, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [r for r in csv.DictReader(open("/tmp/discharge_final/discharge/timeline.csv")) if r["usb_powered"] == "false"]
INK, MUTED, GRID, AXIS = "#1a1a1a", "#6b6b6b", "#ececec", "#b8b8b8"
PERF, ENER, LEV = "#D55E00", "#0072B2", "#1a1a1a"
BAND = {"healthy": "#eef3fb", "mid": "#fdf3e4", "low": "#fce9ea"}
DEF = {"healthy": 1.0, "mid": 0.5, "low": 0.0}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.6, "axes.labelsize": 6.6,
                     "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "pdf.fonttype": 42})

n = [int(r["n"]) for r in rows]; L = [float(r["lever"]) for r in rows]
soc = [int(r["soc"]) for r in rows]; tier = [r["tier"] for r in rows]
fig, (ax, b) = plt.subplots(1, 2, figsize=(3.33, 1.30), gridspec_kw=dict(width_ratios=[1.45, 1.0]))
# ---- (a) the lever through the discharge: tier bands, tier default and every loop action
i0 = 0
for i in range(1, len(rows) + 1):
    if i == len(rows) or tier[i] != tier[i0]:
        ax.axvspan(n[i0] - 0.5, n[i - 1] + 0.5, color=BAND[tier[i0]], zorder=0, lw=0)
        ax.plot([n[i0] - 0.5, n[i - 1] + 0.5], [DEF[tier[i0]]] * 2, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text((n[i0] + n[i - 1]) / 2, 1.3, f"{tier[i0]}, {soc[i0]} to {soc[i - 1]}%", ha="center", va="top", fontsize=5.6, color=MUTED)
        i0 = i
ax.step(n, L, where="mid", color=LEV, lw=1.0, zorder=3)
for r, x, y in zip(rows, n, L):
    loop = r["loop"]
    if loop.startswith("performance"):
        ax.scatter(x, y, marker="^", s=16, color=PERF, edgecolor="white", linewidth=0.4, zorder=5)
    elif loop.startswith("energy"):
        ax.scatter(x, y, marker="v", s=16, color=ENER, edgecolor="white", linewidth=0.4, zorder=5)
h = [plt.Line2D([], [], color=LEV, lw=1.0, label="lever used"),
     plt.Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 2)), label="tier default"),
     plt.Line2D([], [], marker="^", ls="", color=PERF, label="time over budget: +0.1"),
     plt.Line2D([], [], marker="v", ls="", color=ENER, label="energy over budget: -0.1")]
ax.legend(handles=h, loc="lower left", fontsize=5.3, frameon=True, framealpha=0.9, edgecolor="none",
          ncol=1, handlelength=1.3, borderpad=0.25, labelspacing=0.25, bbox_to_anchor=(0.0, 0.0))
ax.set_ylim(-0.42, 1.34); ax.set_yticks([0, 0.5, 1.0]); ax.set_xlim(n[0] - 0.5, n[-1] + 0.5)
ax.set_xlabel("request on battery", labelpad=1); ax.set_ylabel("lever $L$", labelpad=1)
ax.text(0.01, 0.985, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=6.4, weight="bold", color=INK)

# ---- (b) the loops under the guarded learner: energy prediction error per request, by the learner's mode
D = json.load(open(os.path.join(HERE, "..", "energy_perf_data.json")))["guarded_bandit"]
err = {}
for d in glob.glob(os.path.join(HERE, "..", "..", "..", "..", "campaigns", "guarded_bandit_20260921", "requests", "*", "sched_log.txt")):
    kv = dict(re.findall(r"(\w+)=(\S+)", open(d).read()))
    m = re.match(r"gb(\d+)_", os.path.basename(os.path.dirname(d)))
    if m and "meas_J" in kv:
        err[int(m.group(1))] = 100 * abs(float(kv["meas_J"]) - float(kv["pred_J"])) / float(kv["pred_J"])
assert len(err) == D["n"] == 30, len(err)
MODE = {"explore": "#9a9a9a", "go": "#009E73", "fallback": "#CC79A7"}
b.axhspan(0, 5, color="#e3efe8", zorder=0, lw=0)
fired = 0
for r in D["per_request"]:
    i = r["i"]; b.bar(i, err[i], color=MODE[r["kind"]], width=0.8, zorder=2)
    if r["loop"] != "hold":
        fired += 1; b.scatter(i, err[i] + 0.9, marker="^", s=16, color=PERF, edgecolor="white", linewidth=0.4, zorder=5)
hb = [plt.Rectangle((0, 0), 1, 1, color=MODE[k], label=k) for k in ("explore", "go", "fallback")]
hb.append(plt.Line2D([], [], marker="^", ls="", color=PERF, label="a loop fired"))
b.legend(handles=hb, loc="upper right", fontsize=5.3, frameon=False, ncol=2, handlelength=1.0,
         columnspacing=0.6, borderpad=0.2, labelspacing=0.25, handletextpad=0.4, bbox_to_anchor=(1.02, 1.03))
b.text(0.03, 0.72, f"fired {fired} of 30,\none per mode", transform=b.transAxes, ha="left", va="top", fontsize=5.6, color=INK, linespacing=1.2)
b.text(29.5, 5.15, "5%", ha="right", va="bottom", fontsize=5.4, color=MUTED)
b.set_xlim(-1, 30); b.set_ylim(0, 12.6); b.set_yticks([0, 4, 8, 12]); b.set_xticks([0, 10, 20, 29]); b.set_xticklabels(["1", "11", "21", "30"])
b.set_xlabel("request, guarded learner", labelpad=1); b.set_ylabel("|energy error| (%)", labelpad=1)
b.text(0.01, 0.985, "(b)", transform=b.transAxes, ha="left", va="top", fontsize=6.4, weight="bold", color=INK)

for a in (ax, b):
    a.grid(axis="y", color=GRID, lw=0.5); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(AXIS)
    a.tick_params(colors=MUTED, length=2)
fig.subplots_adjust(left=0.095, right=0.99, top=0.97, bottom=0.280, wspace=0.34)
out = os.path.join(HERE, "fig_discharge_loops")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
perf = sum(r["loop"].startswith("performance") for r in rows); ener = sum(r["loop"].startswith("energy") for r in rows)
print("wrote", out, "battery requests", len(rows), "perf events", perf, "energy events", ener, "| bandit fired", fired, "max err", round(max(err.values()), 1))
