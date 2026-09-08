#!/usr/bin/env python3
"""Balance points: for each control, energy on top and its price below, on one x axis, with
the operating point the data supports marked. OnePlus 15, Llama-3.2-1B-Instruct Q4_K_M.

Columns: GPU clock cap | GPU cache K | CPU cache K | CPU clock cap
Row 1: energy (per request for clocks, per decoded token for caches).
Row 2: the price: time to first token (clocks) or LongBench F1 (caches).
Re-run after the proof finishes: it reads /tmp/ea_proof_v2 and adds every valid cell.
"""
import csv, json, os, re, sys, glob, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/mislam22/EndurKV_workspace/EndurKV/scripts")
from ea_proof_summary import load as load_cell

BLUE, AQUA, RED, YELLOW = "#2a78d6", "#1baf7a", "#e34948", "#eda100"
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

def ms(v):
    v = [x for x in v if x == x]
    return (st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0), len(v)) if v else (float("nan"), 0.0, 0)

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.5)

def balance(ax, x, text, y=0.04):
    ax.axvline(x, color=RED, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.text(x, ax.get_ylim()[0] + y * (ax.get_ylim()[1] - ax.get_ylim()[0]), " " + text, color=RED, fontsize=6.3, va="bottom", ha="left")

# ---------- data ----------
# GPU clock: sweep (CPU at 883 MHz) and proof (CPU at 1497 MHz)
sweep = {}
for d in sorted(glob.glob("/tmp/clock_sweep/clk_*")):
    x = load_cell(d); m = json.load(open(d + "/meta.json"))
    sweep[int(d.split("_")[-1])] = dict(E=x["mJ"] * m["n_decode_steps"] / 1000, pre=x["prefill_s"], wall=m["total_ms"] / 1000)
proof = {}
for d in sorted(glob.glob("/tmp/ea_proof_v2/*_r[0-9]")):
    b = os.path.basename(d)
    if not os.path.exists(d + "/meta.json"):
        continue
    x = load_cell(d); m = json.load(open(d + "/meta.json"))
    arm = b.rsplit("_r", 1)[0]
    proof.setdefault(arm, []).append(dict(E=x["mJ"] * m["n_decode_steps"] / 1000, pre=x["prefill_s"], wall=m["total_ms"] / 1000, mJ=x["mJ"], K=x["K"],
                                          dec_mJ=x["dec_mJ"], gpu_mhz=x["gpu_mhz"]))
gpu_mhz_of = {"gpu_healthy": 1200, "gpu_mid": 902, "gpu_low": 726}
# GPU cache K: n=3 campaign whole-run mJ/token, plus vanilla
gk = {}
for d in glob.glob("/tmp/ea_n3/r*"):
    x = load_cell(d); gk.setdefault(x["K"], []).append(x["mJ"])
van_gpu = load_cell("/tmp/energy_tiers/gen_vanilla")["mJ"]
# CPU clock sweep (muKV arms, 128 tokens)
csw = {}
for d in glob.glob("/tmp/phone_freq_sweep/v2_*"):
    if not os.path.isdir(d) or not os.path.exists(d + "/meta.json"):
        continue
    x = load_cell(d); m = json.load(open(d + "/meta.json"))
    csw[int(d.split("_")[-1]) // 1000] = dict(E=x["mJ"] * m["n_decode_steps"] / 1000, pre=x["prefill_s"], wall=m["total_ms"] / 1000)
# CPU cache: soak arms (whole-arm energy per token)
def soak(path):
    its = sorted(glob.glob(path + "/iter_*.json"))
    msj = [json.loads(re.sub(r":\s*-?nan\b", ": NaN", open(f).read())) for f in its]
    rs = list(csv.DictReader(open(path + "/sensors.csv", "rb").read().decode("utf-8", "replace").splitlines()))
    usb = 0.0; prev = None
    for r in rs:
        try:
            t = float(r["monotonic_s"]); v = float(r["usb_voltage_uv"]) / 1e6; i = abs(float(r["usb_current_ua"])) / 1e6
        except Exception:
            continue
        if prev is not None:
            dt = min(t - prev, 5.0)
            if dt > 0:
                usb += v * i * dt
        prev = t
    q = [float(r["bat_charge_uah"]) for r in rs if r.get("bat_charge_uah", "").strip().lstrip("-").isdigit()]
    vv = [float(r["bat_voltage_now_uv"]) / 1e6 for r in rs if r.get("bat_voltage_now_uv", "").strip().isdigit()]
    bat = max(0.0, (q[0] - q[-1]) / 1e6) * (sum(vv) / len(vv) if vv else 4.35) * 3600 if len(q) > 1 else 0.0
    toks = sum(m["n_decode_steps"] for m in msj)
    return (usb + bat) * 1000 / toks, st.mean(m["decode_tps"] for m in msj)
cpu_full_mJ, cpu_full_tps = soak("/tmp/cpu_soak/vanilla")
cpu_1024_mJ, cpu_1024_tps = soak("/tmp/cpu_soak/mukv_wdoff")
# quality per K (Llama-1B, n=15), model-level so it applies to both backends
tq = json.load(open("/tmp/tier_quality.json"))
qK = [236, 473, 945]
hot = [tq[f"kpct{p}|hotpotqa"]["f1"] for p in (5, 10, 20)]
qas = [tq[f"kpct{p}|qasper"]["f1"] for p in (5, 10, 20)]
hot_full = tq["llama1b__vanilla|hotpotqa"]["f1"]

# ---------- figure ----------
fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.9))
fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.13, wspace=0.42, hspace=0.42)
(a1, b1, c1, d1), (a2, b2, c2, d2) = axs

# --- column 1: GPU clock cap ---
xs = sorted(sweep)
a1.plot(xs, [sweep[x]["E"] for x in xs], color=MUTED, lw=1.0, marker="o", ms=4, mfc=MUTED, mec=SURF, label="sweep, CPU 883 MHz")
px = [gpu_mhz_of[a] for a in gpu_mhz_of if a in proof]; pe = [ms([c["E"] for c in proof[a]]) for a in gpu_mhz_of if a in proof]
a1.errorbar(px, [e[0] for e in pe], yerr=[e[1] for e in pe], color=BLUE, lw=1.2, marker="s", ms=4.5, mfc=BLUE, mec=SURF, capsize=2, label="proof, CPU 1497 MHz")
for x_, e in zip(px, pe):
    a1.annotate(f"{e[0]:.0f}", (x_, e[0]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.3, color=BLUE)
a1.set_ylim(900, 1750); a1.set_ylabel("energy per request (J)"); a1.set_title("GPU clock cap", loc="left")
a1.legend(loc="lower left", handlelength=1.2)
balance(a1, 902, "balance\n902 MHz", 0.80)
style(a1)
a2.plot(xs, [sweep[x]["wall"] for x in xs], color=MUTED, lw=1.0, marker="o", ms=4, mfc=MUTED, mec=SURF)
pp = [ms([c["wall"] for c in proof[a]]) for a in gpu_mhz_of if a in proof]
a2.errorbar(px, [p[0] for p in pp], yerr=[p[1] for p in pp], color=BLUE, lw=1.2, marker="s", ms=4.5, mfc=BLUE, mec=SURF, capsize=2)
for x_, p in zip(px, pp):
    a2.annotate(f"{p[0]:.0f} s", (x_, p[0]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.3, color=BLUE)
a2.set_ylim(180, 430); a2.set_ylabel("total time per request (s)"); a2.set_xlabel("GPU clock cap (MHz)")
a2.text(0.98, 0.96, "prefill + decode, 4096 tokens\nbelow 902: +2% saving,\n+16% time", transform=a2.transAxes, fontsize=6.3, color=MUTED, ha="right", va="top")
balance(a2, 902, "", 0.5)
style(a2)
for ax in (a1, a2):
    ax.set_xticks([726, 902, 1050, 1200]); ax.set_xlim(680, 1250)

# --- column 2: GPU cache K ---
ks = sorted(gk); ge = [ms(gk[k]) for k in ks]
b1.errorbar(ks, [e[0] for e in ge], yerr=[e[1] for e in ge], color=BLUE, lw=1.2, marker="s", ms=4.5, mfc=BLUE, mec=SURF, capsize=2, label="μKV (n=3)")
b1.plot([9741], [van_gpu], marker="D", ms=4.5, mfc=BLUE, mec=SURF, ls="none", label="vanilla, full cache")
b1.set_ylim(0, 480); b1.set_ylabel("whole-run energy per output token (mJ)"); b1.set_title("GPU cache budget K", loc="left")
b1.legend(loc="lower left", handlelength=1.5)
b1.text(0.98, 0.96, "flat: no energy to win", transform=b1.transAxes, fontsize=6.3, color=MUTED, ha="right", va="top")
style(b1)
for y_, col, lab in ((hot, BLUE, "hotpotqa"), (qas, AQUA, "qasper")):
    b2.plot(qK, y_, color=col, lw=1.2, marker="o", ms=4, mfc=col, mec=SURF)
    b2.text(qK[-1] * 1.12, y_[-1], lab, color=col, fontsize=6.5, va="center")
b2.plot([9741], [hot_full], marker="D", ms=4.5, mfc=BLUE, mec=SURF, ls="none")
b2.axhspan(0, 0.9 * qas[-1], color=RED, alpha=0.06, lw=0)
b2.axhline(0.9 * qas[-1], color=AQUA, lw=0.7, ls=(0, (3, 2)))
b2.text(240, 0.9 * qas[-1] - 0.6, "90% floor (a choice)", color=AQUA, fontsize=5.8, va="top")
b2.set_ylim(0, 45); b2.set_ylabel("LongBench F1 (n=15)"); b2.set_xlabel("cache budget K (cells)")
b2.text(0.98, 0.30, "below 945: qasper\nloses 37% for nothing", transform=b2.transAxes, fontsize=6.3, color=MUTED, ha="right", va="top")
style(b2)
for ax in (b1, b2):
    ax.set_xscale("log", base=2); ax.set_xticks([256, 1024, 4096, 9741]); ax.set_xticklabels(["256", "1024", "4096", "full"]); ax.set_xlim(200, 14000)
balance(b1, 1024, "hold 1024", 0.34); balance(b2, 1024, "", 0.5)

# --- column 3: CPU cache K ---
c1.plot([9741, 1024], [cpu_full_mJ, cpu_1024_mJ], color=AQUA, lw=1.2, marker="D", ms=4.5, mfc=AQUA, mec=SURF, label="soak, 6 generations each")
c1.annotate(f"{cpu_full_mJ:.0f}", (9741, cpu_full_mJ), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.3, color=AQUA)
c1.annotate(f"{cpu_1024_mJ:.0f}", (1024, cpu_1024_mJ), xytext=(-5, -10), textcoords="offset points", ha="right", fontsize=6.3, color=AQUA)
cpu_arms = [(a, proof[a]) for a in ("cpu_healthy", "cpu_mid", "cpu_low") if a in proof]
if cpu_arms:
    cx = [cells[0]["K"] for _, cells in cpu_arms]; ce = [ms([c["mJ"] for c in cells]) for _, cells in cpu_arms]
    c1.errorbar(cx, [e[0] for e in ce], yerr=[e[1] for e in ce], color=AQUA, lw=1.0, ls=(0, (2, 2)), marker="s", ms=4.5, mfc=SURF, mec=AQUA, capsize=2, label="proof ladder (1024 tokens)")
    for x_, e in zip(cx, ce):
        c1.annotate(f"{e[0]:.0f}", (x_, e[0]), xytext=(0, -10), textcoords="offset points", ha="center", fontsize=6.3, color=AQUA)
else:
    c1.text(0.03, 0.48, "1024 / 512 / 256:\nproof running", transform=c1.transAxes, fontsize=6.3, color=MUTED)
c1.set_ylim(0, 1500); c1.set_ylabel("whole-run energy per output token (mJ)"); c1.set_title("CPU cache budget K", loc="left")
c1.legend(loc="lower right", handlelength=1.2)
style(c1)
for y_, col, lab in ((hot, BLUE, "hotpotqa"), (qas, AQUA, "qasper")):
    c2.plot(qK, y_, color=col, lw=1.2, marker="o", ms=4, mfc=col, mec=SURF)
    c2.text(qK[-1] * 1.12, y_[-1], lab, color=col, fontsize=6.5, va="center")
c2.plot([9741], [hot_full], marker="D", ms=4.5, mfc=BLUE, mec=SURF, ls="none")
c2.axhspan(0, 0.9 * qas[-1], color=RED, alpha=0.06, lw=0)
c2.axhline(0.9 * qas[-1], color=AQUA, lw=0.7, ls=(0, (3, 2)))
c2.set_ylim(0, 45); c2.set_ylabel("LongBench F1 (n=15)"); c2.set_xlabel("cache budget K (cells)")
c2.text(0.98, 0.30, "1024: -62% energy,\n-24% hotpotqa\nbelow: qasper -37%", transform=c2.transAxes, fontsize=6.3, color=MUTED, ha="right", va="top")
style(c2)
for ax in (c1, c2):
    ax.set_xscale("log", base=2); ax.set_xticks([256, 1024, 4096, 9741]); ax.set_xticklabels(["256", "1024", "4096", "full"]); ax.set_xlim(200, 14000)
balance(c1, 1024, "balance\n1024", 0.72); balance(c2, 1024, "", 0.5)

# --- column 4: CPU clock cap ---
xs = sorted(csw)
d1.plot(xs, [csw[x]["E"] for x in xs], color=AQUA, lw=1.2, marker="o", ms=4.5, mfc=AQUA, mec=SURF)
d1.set_ylim(0, 700); d1.set_ylabel("energy per request (J)"); d1.set_title("CPU clock cap", loc="left")
d1.text(0.98, 0.96, "flat: no balance point,\nreducing buys nothing", transform=d1.transAxes, fontsize=6.3, color=MUTED, ha="right", va="top")
style(d1)
d2.plot(xs, [csw[x]["wall"] for x in xs], color=AQUA, lw=1.2, marker="o", ms=4.5, mfc=AQUA, mec=SURF)
d2.set_ylim(80, 260); d2.set_ylabel("total time per request (s)"); d2.set_xlabel("CPU clock cap (MHz)")
d2.text(0.98, 0.96, "prefill + decode, 128 tokens\nonly the time grows", transform=d2.transAxes, fontsize=6.3, color=MUTED, ha="right", va="top")
style(d2)
for ax in (d1, d2):
    ax.set_xticks([883, 1267, 1632]); ax.set_xlim(830, 1700)

fig.text(0.07, 0.015, "OnePlus 15, Llama-3.2-1B, 9737-token prompt. Clock columns at K=1024 (GPU 4096 tokens, CPU 128).\nEnergy = USB rail + battery pack. Quality: LongBench F1, n=15 per point, same model on both backends.",
         fontsize=6.5, color=MUTED, ha="left", va="bottom")
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/fig_balance_points"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
print("wrote", out)
print("proof arms used:", {a: len(v) for a, v in proof.items()})
