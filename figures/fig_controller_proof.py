#!/usr/bin/env python3
"""The energy-aware controller at work: identical command line, only the battery state the
controller sees changes. From /tmp/ea_proof_v2 (pinned protocol, n=3 per arm).

One row per backend, because the arm designs differ (GPU 4096 output tokens, CPU 1024) and
totals must not be compared across rows. Per row: (a) total energy per request by tier,
(b) total time split into prefill and decode, (c) energy against time with the constant
energy-delay curve through the healthy point. Re-run as cells land.
"""
import json, os, sys, glob, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/mislam22/EndurKV_workspace/EndurKV/scripts")
from ea_proof_summary import load as load_cell

BLUE, AQUA, RED = "#2a78d6", "#1baf7a", "#e34948"
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

arms = {}
for d in sorted(glob.glob("/tmp/ea_proof_v2/*_r[0-9]")):
    if not os.path.exists(d + "/meta.json"):
        continue
    x = load_cell(d); m = json.load(open(d + "/meta.json"))
    arms.setdefault(os.path.basename(d).rsplit("_r", 1)[0], []).append(
        dict(E=x["prefill_J"] + x["decode_J"], pre=x["prefill_s"], dec=x["decode_s"], wall=m["total_ms"] / 1000,
             tps=x["tps"], K=x["K"], gpu=x["gpu_mhz"], np=m["n_prompt_tokens"], no=m["n_decode_steps"]))
tiers = ["healthy", "mid", "low"]
action = {"gpu": {"healthy": "1200 MHz", "mid": "902 MHz cap", "low": "726 MHz cap"},
          "cpu": {"healthy": "K=1024", "mid": "K=512", "low": "K=256"}}

def stats(backend):
    out = []
    for t in tiers:
        c = arms.get(f"{backend}_{t}", [])
        out.append(dict(t=t, n=len(c), E=ms([k["E"] for k in c]), pre=ms([k["pre"] for k in c]), dec=ms([k["dec"] for k in c]),
                        wall=ms([k["wall"] for k in c]), toks=(c[0]["np"] + c[0]["no"]) if c else 0, no=c[0]["no"] if c else 0))
    return out

fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.2))
fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.12, wspace=0.40, hspace=0.62)
fig.text(0.075, 0.955, "GPU row: the battery tier caps the GPU clock; K is held at 1024.", fontsize=8, color=BLUE, weight="bold")
fig.text(0.075, 0.485, "CPU row: the battery tier sets K; the clock is untouched. Not comparable with the GPU row.", fontsize=7.4, color=AQUA, weight="bold")
for ax in axs.flat:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.5)

for row, (backend, col, label) in enumerate((("gpu", BLUE, "GPU, 4096 output tokens: total energy"),
                                             ("cpu", AQUA, "CPU, 1024 output tokens: total energy"))):
    a, b, c = axs[row]
    S = stats(backend)
    xs = list(range(3))
    # (a) total energy
    ys = [s["E"][0] if s["n"] else 0 for s in S]; es = [s["E"][1] for s in S]; ref = ys[0] if S[0]["n"] else float("nan")
    a.bar(xs, ys, width=0.55, color=col, edgecolor=SURF, linewidth=0.5, yerr=es, capsize=2, ecolor=INK, zorder=3)
    top = max(ys) if any(ys) else 1
    for j, (x_, y_, s) in enumerate(zip(xs, ys, S)):
        if not s["n"]:
            a.text(x_, top * 0.02, "pending", ha="center", va="bottom", fontsize=6, color=MUTED, rotation=90); continue
        a.text(x_, y_ + top * 0.02, f"{y_:.0f} J" + (f"\n{y_/ref-1:+.0%}" if j and ref == ref else ""), ha="center", va="bottom", fontsize=6.3, color=INK)
    a.set_xticks(xs); a.set_xticklabels([f"{t}\n{action[backend][t]}" for t in tiers], fontsize=6.3)
    a.set_ylabel("total energy per request (J)"); a.set_ylim(0, top * 1.35); a.grid(axis="x", visible=False)
    a.set_title(f"({'ad'[row]}) {label}", loc="left", fontsize=7.6)
    # (b) total time, stacked
    pre = [s["pre"][0] if s["n"] else 0 for s in S]; dec = [s["dec"][0] if s["n"] else 0 for s in S]
    b.bar(xs, pre, width=0.55, color=MUTED, edgecolor=SURF, linewidth=0.5, zorder=3)
    b.bar(xs, dec, width=0.55, bottom=pre, color=col, edgecolor=SURF, linewidth=0.5, zorder=3)
    tref = pre[0] + dec[0] if S[0]["n"] else float("nan"); ttop = max(p + d for p, d in zip(pre, dec)) if any(pre) else 1
    for j, (x_, p, d_, s) in enumerate(zip(xs, pre, dec, S)):
        if s["n"]:
            b.text(x_, p + d_ + ttop * 0.02, f"{p+d_:.0f} s" + (f"\n{(p+d_)/tref-1:+.0%}" if j and tref == tref else ""), ha="center", va="bottom", fontsize=6.3, color=INK)
    b.set_xticks(xs); b.set_xticklabels([f"{t}\n{action[backend][t]}" for t in tiers], fontsize=6.3)
    b.set_ylabel("total time per request (s)"); b.set_ylim(0, ttop * 1.35); b.grid(axis="x", visible=False)
    b.set_title(f"({'be'[row]}) total time: prefill (grey) + decode", loc="left", fontsize=7.6)
    # (c) energy against time with the constant E x T curve through the healthy point
    pts = [(s["wall"][0], s["E"][0], s["wall"][1], s["E"][1], s["t"], s["n"]) for s in S if s["n"]]
    if pts:
        c.errorbar([p[0] for p in pts], [p[1] for p in pts], xerr=[p[2] for p in pts], yerr=[p[3] for p in pts],
                   color=col, ls="-", lw=1.0, marker="o", ms=5, mfc=col, mec=SURF, capsize=2, zorder=3)
        h = pts[0]; edp = h[0] * h[1]
        tmin = min(p[0] for p in pts) * 0.72; tmax = max(p[0] for p in pts) * 1.25
        emin = min(p[1] for p in pts) * 0.72; emax = max(p[1] for p in pts) * 1.2
        T = np.linspace(tmin, tmax, 100); c.plot(T, edp / T, color=col, lw=0.8, ls=(0, (3, 2)), zorder=2)
        spread = (max(p[1] for p in pts) - min(p[1] for p in pts)) / h[1]
        offs = {"healthy": (-5, 6, "right", "bottom"), "mid": (7, 6, "left", "bottom"), "low": (7, -5, "left", "top")}
        for tw, te, _, _, t, n in pts:
            dx, dy, ha, va = offs[t]
            if spread < 0.05 and t != "healthy":
                continue
            c.annotate(t if spread >= 0.05 else "all three tiers\nwithin 3%", (tw, te), xytext=(dx, dy), textcoords="offset points", fontsize=6.3, color=col, ha=ha, va=va)
        lines = ["E x T against healthy:", "  ".join(f"{p[4]} {(p[0]*p[1])/edp-1:+.0%}" for p in pts[1:])]
        if backend == "gpu":
            # the split-clock plans (prefill 1200, decode capped), measured in /tmp/split_proof
            for tag, lab in (("gpu1200d902", "decode 902"), ("gpu1200d726", "decode 726")):
                sc = []
                for d in sorted(glob.glob(f"/tmp/split_proof/{tag}_r[0-9]")):
                    if os.path.exists(d + "/meta.json"):
                        x = load_cell(d); m = json.load(open(d + "/meta.json"))
                        sc.append((m["total_ms"] / 1000, x["prefill_J"] + x["decode_J"]))
                if sc:
                    tw = st.mean(v[0] for v in sc); te = st.mean(v[1] for v in sc)
                    c.errorbar(tw, te, xerr=st.pstdev([v[0] for v in sc]), yerr=st.pstdev([v[1] for v in sc]), color=RED, marker="D", ms=4.5, mec=SURF, capsize=2, ls="none", zorder=4)
                    c.annotate(lab, (tw, te), xytext=(-8, 3 if tag.endswith("902") else -9), textcoords="offset points", fontsize=6.0, color=RED, ha="right", va="bottom" if tag.endswith("902") else "top")
                    lines.append(f"1200 / {lab}: {(tw*te)/edp-1:+.0%}")
        c.text(0.02, 0.04, "\n".join(lines), transform=c.transAxes, fontsize=5.8, color=col, va="bottom", ha="left")
        c.set_xlim(tmin, tmax); c.set_ylim(emin, emax)
    c.set_xlabel("total time per request (s)"); c.set_ylabel("total energy per request (J)")
    c.set_title(f"({'cf'[row]}) energy against time", loc="left", fontsize=7.6)
    c.text(0.98, 0.97, "dashed: constant E x T\nthrough healthy", transform=c.transAxes, fontsize=5.8, color=MUTED, va="top", ha="right")

fig.text(0.075, 0.012, "OnePlus 15, Llama-3.2-1B, 9737-token prompt. Same command line every arm; only the battery state the controller reads differs. n=3 per arm.\n"
         "Tiers: healthy above 50% or mains, mid 21 to 50%, low 20% or less. CPU caps pinned, cooled per cell. Energy = USB rail + battery pack.",
         fontsize=6.3, color=MUTED, ha="left", va="bottom")
out = "/home/mislam22/EndurKV_workspace/EndurKV/figures/fig_controller_proof"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
print("wrote", out, {k: len(v) for k, v in arms.items()})
for backend in ("gpu", "cpu"):
    for s in stats(backend):
        if s["n"]:
            print(f"  {backend} {s['t']:8s} n={s['n']} E={s['E'][0]:.0f} J  wall={s['wall'][0]:.0f} s  ExT={s['E'][0]*s['wall'][0]/1000:.0f} kJs  all-in tok/s={s['toks']/s['wall'][0]:.1f}")
