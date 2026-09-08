#!/usr/bin/env python3
# Reframed watchdog preemption: vanilla Bonsai-8B WITH the battery-50C-anchored watchdog
# vs WITHOUT it (baseline). Both cold start. Five panels: prime clock, perf clock, battery
# (50C throttle line), DDR temp (the memory-traffic heat), skin. Shows the watchdog converting
# the vendor's 883 MHz deep-throttle into a controlled glide that holds battery below 50C, while
# DDR (memory heat) stays ~the same in both runs -- the clock is not the heat lever.
import csv, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [("no watchdog", "/tmp/nat_bonsai/vanilla", "#777777"),
        ("+ watchdog",  "/tmp/wd_vanilla",         "#0072B2")]

def load(path):
    raw = open(f"{path}/sensors.csv", "rb").read().decode("utf-8", "replace")
    sr = list(csv.DictReader(raw.splitlines()))
    def col(n):
        out = []
        for r in sr:
            try: out.append(float(r.get(n, "")))
            except: out.append(np.nan)
        return np.array(out)
    t = col("monotonic_s"); t = (t - t[0]) / 60.0
    return dict(t=t, prime=col("cpu6_freq_hz")/1e3, perf=col("cpu0_freq_hz")/1e3,
                bat=col("battery_temp_mc")/1000.0, skin=col("shell_front_temp_mc")/1000.0,
                ddr=col("ddr_temp_mc")/1000.0)

def smooth(y, w):
    y = np.asarray(y, float)
    if np.isnan(y).any(): y = np.where(np.isnan(y), np.nanmean(y), y)
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w)/w, mode="valid")[:len(y)]

data = {lbl: load(p) for lbl, p, _ in RUNS}

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "legend.frameon": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(5, 1, figsize=(3.4, 6.2), sharex=True)

for lbl, path, color in RUNS:
    d = data[lbl]
    lw = 1.0 if "+" in lbl else 0.9
    ax[0].plot(d["t"], smooth(d["prime"], 25), color=color, lw=lw, label=lbl)
    ax[1].plot(d["t"], smooth(d["perf"], 25),  color=color, lw=lw, label=lbl)
    ax[2].plot(d["t"], smooth(d["bat"], 7),    color=color, lw=1.1, label=lbl)
    ax[3].plot(d["t"], smooth(d["ddr"], 7),    color=color, lw=1.1, label=lbl)
    ax[4].plot(d["t"], smooth(d["skin"], 7),   color=color, lw=1.1, label=lbl)

# prime clock -- mark the 883 deep-throttle floor
ax[0].axhline(883, color="#c0392b", ls=":", lw=0.7)
ax[0].text(2, 905, "883 MHz vendor deep-throttle", fontsize=5.5, color="#c0392b")
ax[0].set_ylabel("prime clk\n(MHz)"); ax[0].set_ylim(780, 1780)
ax[0].legend(loc="upper center", fontsize=7, ncol=2, handlelength=1.3, bbox_to_anchor=(0.5, 1.02))

ax[1].set_ylabel("perf clk\n(MHz)"); ax[1].set_ylim(780, 2050)

# battery -- mark the 50C throttle threshold
ax[2].axhline(50, color="#c0392b", ls=":", lw=0.7)
ax[2].text(2, 50.15, "50 °C throttle trigger", fontsize=5.5, color="#c0392b")
ax[2].set_ylabel("battery\n(°C)")
ax[2].annotate("50.2 (throttles)", (data["no watchdog"]["t"][-1], 50.2), fontsize=5.5,
               color="#777777", ha="right", va="bottom")
ax[2].annotate("49.1 (held)", (data["+ watchdog"]["t"][-1], 49.1), fontsize=5.5,
               color="#0072B2", ha="right", va="top")

# DDR -- the memory-traffic heat; ~identical in both (clock is not the heat lever)
ax[3].set_ylabel("DDR\n(°C)")
ax[3].text(2, ax[3].get_ylim()[0] + 0.5, "memory-traffic heat: ~same with/without watchdog",
           fontsize=5.5, color="#555", transform=ax[3].get_yaxis_transform() if False else ax[3].transData)

ax[4].set_ylabel("skin\n(°C)"); ax[4].set_xlabel("elapsed time (min)")

for a in ax:
    a.grid(True, lw=0.3, alpha=0.4); a.tick_params(labelsize=7)
fig.align_ylabels(ax)
fig.suptitle("Vanilla Bonsai-8B: watchdog prevents the 883 MHz deep-throttle",
             fontsize=8, y=0.996)
fig.tight_layout(h_pad=0.35, rect=(0, 0, 1, 0.988))
fig.savefig("figures/fig_watchdog_preempt.png", dpi=200, bbox_inches="tight")
fig.savefig("figures/fig_watchdog_preempt.pdf", bbox_inches="tight")
print("wrote figures/fig_watchdog_preempt.{png,pdf}")
for lbl, path, _ in RUNS:
    d = data[lbl]
    print(f"  {lbl}: peakBAT={np.nanmax(d['bat']):.1f} peakDDR={np.nanmax(d['ddr']):.1f} peakSKIN={np.nanmax(d['skin']):.1f} deep%={100*np.mean(d['prime']<=1000):.0f}")
