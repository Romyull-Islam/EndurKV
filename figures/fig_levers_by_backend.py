#!/usr/bin/env python3
"""The energy levers, one per backend, and what each trades. OnePlus 15, Llama-3.2-1B Q4_K_M.

(a) GPU clock cap at K=1024: energy per request split into prefill and decode, with prefill
    time on each bar. Energy falls in prefill only; the trade is time to first token.
(b) CPU clock cap 883 to 1632 MHz at K=1024: energy per request flat, prefill time falls.
    On the CPU in this range the clock is a time lever, not an energy lever.
(c) CPU cache: vanilla (full cache) vs muKV K=1024, six generations of 4096 tokens each,
    whole-arm energy per decoded token and throughput. The cache is the CPU lever.
Energy = USB rail integral + battery pack coulomb delta. Phase split anchored on trace end.
"""
import csv, json, os, re, sys, glob, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/mislam22/EndurKV_workspace/EndurKV/scripts")
from ea_proof_summary import load as load_cell

BLUE, AQUA, RED = "#2a78d6", "#1baf7a", "#e34948"
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

fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 3.1))
fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.30, wspace=0.42)

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.5); ax.grid(axis="x", visible=False)

def stacked(ax, cells, labels, col, title, note):
    xs = list(range(len(cells)))
    pre = [x["prefill_J"] for x in cells]; dec = [x["decode_J"] for x in cells]
    ax.bar(xs, pre, width=0.62, color=MUTED, edgecolor=SURF, linewidth=0.5, zorder=3)
    ax.bar(xs, dec, width=0.62, bottom=pre, color=col, edgecolor=SURF, linewidth=0.5, zorder=3)
    top = max(p + d for p, d in zip(pre, dec))
    for x, p, d, cell in zip(xs, pre, dec, cells):
        ax.text(x, p + d + top * 0.02, f"{cell['prefill_s']:.0f} s", ha="center", va="bottom", fontsize=6.5, color=RED)
        ax.text(x, p / 2, f"{p:.0f}", ha="center", va="center", fontsize=6, color=SURF)
        ax.text(x, p + d / 2, f"{d:.0f}", ha="center", va="center", fontsize=6, color=SURF)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylim(0, top * 1.62); ax.set_ylabel("energy per request (J)")
    ax.set_title(title, loc="left")
    ax.text(0.02, 0.985, note, transform=ax.transAxes, fontsize=6.3, color=MUTED, va="top")
    style(ax)

# (a) GPU clock sweep
gcells = []
for d in sorted(glob.glob("/tmp/clock_sweep/clk_*"), key=lambda p: int(p.split("_")[-1])):
    gcells.append((int(d.split("_")[-1]), load_cell(d)))
stacked(a, [x for _, x in gcells], [f"{m}" for m, _ in gcells], BLUE,
        "(a) GPU clock cap, K=1024",
        "grey prefill, blue decode (4096 tokens)\nred: prefill time. 1200 to 902 MHz:\nprefill energy -32%, decode -4%, time +24%")
a.set_xlabel("GPU clock cap (MHz)")

# (b) CPU clock sweep, muKV arms
ccells = []
for d in sorted([p for p in glob.glob("/tmp/phone_freq_sweep/v2_*") if os.path.isdir(p) and os.path.exists(p + "/meta.json")],
                key=lambda p: int(p.split("_")[-1])):
    ccells.append((int(d.split("_")[-1]) // 1000, load_cell(d)))
stacked(b, [x for _, x in ccells], [f"{m}" for m, _ in ccells], AQUA,
        "(b) CPU clock cap, K=1024",
        "grey prefill, green decode (128 tokens)\nred: prefill time. 883 to 1632 MHz:\nenergy flat, time -45%: not an energy lever")
b.set_xlabel("CPU clock cap (MHz)")

# (c) CPU cache: soak arms vanilla vs muKV wdoff (six generations each)
def arm(path):
    its = sorted(glob.glob(path + "/iter_*.json"))
    ms = [json.loads(re.sub(r":\s*-?nan\b", ": NaN", open(f).read())) for f in its]
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
    toks = sum(m["n_decode_steps"] for m in ms)
    return dict(mJ=(usb + bat) * 1000 / toks, tps=st.mean(m["decode_tps"] for m in ms), n=len(ms), K=ms[0]["k_nominal"])
van = arm("/tmp/cpu_soak/vanilla"); muk = arm("/tmp/cpu_soak/mukv_wdoff")
xs = [0, 1]
c.bar(xs, [van["mJ"], muk["mJ"]], width=0.55, color=[MUTED, AQUA], edgecolor=SURF, linewidth=0.5, zorder=3)
for x, r, lab in ((0, van, "full cache"), (1, muk, "K=1024")):
    c.text(x, r["mJ"] + 25, f"{r['mJ']:.0f} mJ\n{r['tps']:.1f} tok/s", ha="center", va="bottom", fontsize=6.5, color=INK)
c.set_xticks(xs); c.set_xticklabels(["vanilla\nfull cache", "muKV\nK=1024"])
c.set_ylim(0, van["mJ"] * 1.62); c.set_ylabel("energy per decoded token (mJ)")
c.set_title("(c) CPU cache, six generations each", loc="left")
c.text(0.02, 0.985, f"whole-arm energy / tokens\n{100*(1-muk['mJ']/van['mJ']):.0f}% less energy, {muk['tps']/van['tps']:.1f}x faster\nprice: hotpotqa 39.3 to 29.8 F1",
       transform=c.transAxes, fontsize=6.3, color=MUTED, va="top")
style(c)

fig.subplots_adjust(bottom=0.24)
fig.text(0.075, 0.03, "OnePlus 15, Llama-3.2-1B-Instruct Q4_K_M. Energy = USB rail + battery pack. (a), (b): n=1 per bar. (c): n=6 generations per arm.",
         fontsize=6.8, color=MUTED, ha="left", va="bottom")
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/fig_levers_by_backend"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
print("wrote", out)
print("(a)", [(m, round(x["prefill_J"]), round(x["decode_J"]), round(x["prefill_s"])) for m, x in gcells])
print("(b)", [(m, round(x["prefill_J"]), round(x["decode_J"]), round(x["prefill_s"])) for m, x in ccells])
print("(c)", van, muk)
