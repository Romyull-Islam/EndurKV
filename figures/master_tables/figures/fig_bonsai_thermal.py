#!/usr/bin/env python3
# Fig -- "Two levers against the sustained throttle" (Prism/Bonsai-8B, cold start, CPU-only).
# Three runs: vanilla (the problem), muKV (the CACHE lever), vanilla+watchdog (the CLOCK lever).
# Four panels sharing elapsed time: prime clock, battery, skin, DDR. Clean/standard: threshold
# lines only; the narrative lives in the text (Sec. eval-thermal). Legend sits in the 2nd panel.
import csv, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [("vanilla",          "/tmp/nat_bonsai/vanilla",   "#888888"),
        (r"$\mu$KV",         "/tmp/nat_bonsai/mukv_faon",  "#0072B2"),
        ("vanilla+watchdog", "/tmp/wd_vanilla",            "#D55E00")]

def load(path):
    sr = list(csv.DictReader(open(f"{path}/sensors.csv", "rb").read().decode("utf-8", "replace").splitlines()))
    def col(n):
        out = []
        for r in sr:
            try: out.append(float(r.get(n, "")))
            except: out.append(np.nan)
        return np.array(out)
    t = col("monotonic_s"); t = (t - t[0]) / 60.0
    skcol = "shell_front_temp_mc" if "shell_front_temp_mc" in sr[0] else "shell_temp_mc"
    g = json.loads(re.sub(r"\binf|\bnan|-nan", "null", open(f"{path}/gen.json").read()))
    return dict(t=t, prime=col("cpu6_freq_hz")/1e3, bat=col("battery_temp_mc")/1000.0,
                skin=col(skcol)/1000.0, ddr=col("ddr_temp_mc")/1000.0,
                wall=(g["prefill_ms"]+g["decode_ms"])/60000.0)

def smooth(y, w):
    y = np.asarray(y, float)
    if np.isnan(y).any(): y = np.where(np.isnan(y), np.nanmean(y), y)
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w)/w, mode="valid")[:len(y)]

data = {lbl: load(p) for lbl, p, _ in RUNS}
xmax = max(d["wall"] for d in data.values()) * 1.03

plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "legend.frameon": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(4, 1, figsize=(3.4, 5.1), sharex=True)

for lbl, path, color in RUNS:
    d = data[lbl]; lw = 1.15 if "mu" in lbl or "KV" in lbl else 1.0
    ax[0].plot(d["t"], smooth(d["prime"], 27), color=color, lw=lw, label=lbl)
    ax[1].plot(d["t"], smooth(d["bat"], 7),    color=color, lw=lw, label=lbl)
    ax[2].plot(d["t"], smooth(d["skin"], 7),   color=color, lw=lw)
    ax[3].plot(d["t"], smooth(d["ddr"], 7),    color=color, lw=lw)

# panel 0: prime clock -- throttle floor
ax[0].axhline(883, color="#c0392b", ls=":", lw=0.7)
ax[0].text(xmax*0.05, 905, "883 MHz throttle", fontsize=6.5, color="#c0392b", ha="left", va="bottom")
ax[0].set_ylabel("prime clock\n(MHz)"); ax[0].set_ylim(800, 1700)

# panel 1: battery -- 50C trigger + legend here (2nd subfigure)
ax[1].axhline(50, color="#c0392b", ls=":", lw=0.7)
ax[1].text(xmax*0.02, 50.3, "50 °C throttle trigger", fontsize=6.5, color="#c0392b", ha="left", va="bottom")
ax[1].set_ylabel("battery\n(°C)"); ax[1].set_ylim(32, 53)
ax[1].legend(loc="lower right", fontsize=6.5, handlelength=1.3, labelspacing=0.28, borderpad=0.2)

# panel 2: skin -- 50C watchdog skin rung
ax[2].axhline(50, color="#8e44ad", ls=":", lw=0.7)
ax[2].text(xmax*0.02, 50.4, "50 °C skin rung", fontsize=6.5, color="#8e44ad", ha="left", va="bottom")
ax[2].set_ylabel("skin\n(°C)"); ax[2].set_ylim(32, 55)

# panel 3: DDR
ax[3].set_ylabel("DDR\n(°C)"); ax[3].set_xlabel("elapsed time (min)"); ax[3].set_ylim(44, 76)

for a in ax:
    a.set_xlim(0, xmax); a.grid(True, lw=0.3, alpha=0.4); a.tick_params(labelsize=7)
fig.align_ylabels(ax)
fig.tight_layout(h_pad=0.4)
fig.savefig("figures/fig_bonsai_thermal.png", dpi=200, bbox_inches="tight")
fig.savefig("figures/fig_bonsai_thermal.pdf", bbox_inches="tight")
print("wrote figures/fig_bonsai_thermal.{png,pdf}")
for lbl, p, _ in RUNS:
    d = data[lbl]
    print(f"  {lbl:18s} wall={d['wall']:.1f} bat={np.nanmax(d['bat']):.1f} "
          f"skin={np.nanmax(d['skin']):.1f} ddr={np.nanmax(d['ddr']):.1f}")
