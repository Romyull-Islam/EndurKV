#!/usr/bin/env python3
# Figure 6 -- "Where the heat comes from." Llama-3.2-1B, muKV vs vanilla, cold start, CPU-only.
# Three panels sharing elapsed time: (1) CPU big-core clock, (2) CPU temperature, (3) DDR temperature.
# Story: both run the SAME clock (muKV's win is not frequency); during the compute-bound PREFILL the
# CPU is the hot component while DDR is cool; during the memory-bound DECODE the heat source shifts to
# the DRAM (I/O), whose temperature climbs with the per-token cache reads; muKV, holding a far smaller
# cache, moves less I/O, so its DDR runs cooler AND it finishes 2.6x sooner.
import csv, io, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [("vanilla", "/tmp/nat_cpu/vanilla",   "#D55E00"),
        (r"$\mu$KV", "/tmp/nat_cpu/mukv_faon", "#0072B2")]

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
    cc = [k for k in sr[0] if (k.startswith("cpu-") or k.startswith("cpullc-"))
          and k.endswith("_temp_mc") and "trip" not in k]
    cpu = np.nanmax(np.vstack([col(k) for k in cc]), axis=0) / 1000.0
    g = json.loads(re.sub(r"\binf|\bnan|-nan", "null", open(f"{path}/gen.json").read()))
    return dict(t=t, clk=col("cpu6_freq_hz")/1e3, cpu=cpu, ddr=col("ddr_temp_mc")/1000.0,
                pe=g["prefill_ms"]/60000.0, wall=(g["prefill_ms"]+g["decode_ms"])/60000.0,
                kv_mb=g.get("retained_kv_bytes", 0)/1048576.0)

def smooth(y, w):
    y = np.asarray(y, float)
    if np.isnan(y).any(): y = np.where(np.isnan(y), np.nanmean(y), y)
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w)/w, mode="valid")[:len(y)]

data = {lbl: load(p) for lbl, p, _ in RUNS}
MU, VAN = r"$\mu$KV", "vanilla"
def ddr_end(d):
    return d["ddr"][int(np.argmin(np.abs(d["t"] - d["wall"])))]
pe = data[VAN]["pe"]                        # shared prefill boundary (muKV FA-on prefill == vanilla)
xmax = max(d["wall"] for d in data.values()) * 1.04

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "legend.frameon": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(3, 1, figsize=(3.4, 4.3), sharex=True)

# shade the prefill (compute-bound) region in every panel
for a in ax:
    a.axvspan(0, pe, color="#000000", alpha=0.05, lw=0)

for lbl, path, color in RUNS:
    d = data[lbl]
    ax[0].plot(d["t"], smooth(d["clk"], 33), color=color, lw=1.0, label=lbl)
    ax[1].plot(d["t"], smooth(d["cpu"], 5),  color=color, lw=1.1, label=lbl)
    ax[2].plot(d["t"], smooth(d["ddr"], 5),  color=color, lw=1.1, label=lbl)

# panel 0: clock -- same for both (win is not frequency)
ax[0].set_ylabel("CPU clock\n(MHz)"); ax[0].set_ylim(820, 1760)
ax[0].legend(loc="upper right", fontsize=7, ncol=2, handlelength=1.3)
ax[0].text(pe/2, 1690, "prefill", fontsize=6, color="#555", ha="center", style="italic")
ax[0].text(pe+1.6, 1690, "decode", fontsize=6, color="#555", ha="center", style="italic")
ax[0].annotate("both self-limit to the same ${\\sim}$1.6 GHz\n(the win is not frequency)", (xmax*0.60, 970),
               fontsize=5.4, color="#333", ha="center", va="center")

# panel 1: CPU temp -- compute heat, driven in prefill; similar for both
ax[1].set_ylabel("CPU temp\n(°C)"); ax[1].set_ylim(29, 70)
ax[1].annotate("prefill = compute\n$\\rightarrow$ CPU is the hot part", (0.4, 41),
               fontsize=5.4, color="#333", ha="left", va="center")

# panel 2: DDR temp -- I/O heat, climbs in decode; muKV cooler + shorter
ax[2].set_ylabel("DDR temp\n(°C)"); ax[2].set_xlabel("elapsed time (min)")
ax[2].annotate("decode = memory I/O $\\rightarrow$ DDR heat", (pe+0.2, 0.06),
               xycoords=("data", "axes fraction"), fontsize=5.6, color="#333", ha="left")
for lbl, path, color in RUNS:
    d = data[lbl]
    i = int(np.argmin(np.abs(d["t"] - d["wall"])))
    ax[2].plot(d["wall"], d["ddr"][i], "o", color=color, ms=4, zorder=5)
ax[2].annotate(f"$\\mu$KV done {data[MU]['wall']:.1f} min\n(cache {data[MU]['kv_mb']:.0f} MiB)",
               (data[MU]["wall"], ddr_end(data[MU])),
               fontsize=5.4, color="#0072B2", ha="left", va="top", xytext=(3, -2), textcoords="offset points")
ax[2].annotate(f"vanilla {data[VAN]['wall']:.0f} min\n(cache {data[VAN]['kv_mb']:.0f} MiB)",
               (data[VAN]["wall"], ddr_end(data[VAN])),
               fontsize=5.4, color="#D55E00", ha="right", va="bottom", xytext=(-2, 2), textcoords="offset points")

for a in ax:
    a.set_xlim(0, xmax); a.grid(True, lw=0.3, alpha=0.4); a.tick_params(labelsize=7)
fig.align_ylabels(ax)
fig.tight_layout(h_pad=0.4)
fig.savefig("figures/fig_thermal_trace.png", dpi=200, bbox_inches="tight")
fig.savefig("figures/fig_thermal_trace.pdf", bbox_inches="tight")
print("wrote figures/fig_thermal_trace.{png,pdf}")
for lbl, p, _ in RUNS:
    d = data[lbl]
    print(f"  {lbl}: prefill={d['pe']:.1f} wall={d['wall']:.1f}min cache={d['kv_mb']:.0f}MiB "
          f"DDR peak={np.nanmax(d['ddr']):.1f} CPU peak={np.nanmax(d['cpu']):.1f}")
