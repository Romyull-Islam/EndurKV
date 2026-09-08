#!/usr/bin/env python3
"""The scheduler on a real discharge of the OnePlus 15 (phone_discharge_loop.sh, 2026-09-05/06).
Every request's energy is rebuilt from its raw sensor trace with the start-anchored split (the
scheduler's own on-battery readings before request 100 were wrong: teardown tail and gauge lag).
x axis: the battery level the scheduler read before the request. Sources: /tmp/discharge_final.
"""
import os, sys, csv, json, glob, statistics as st
import numpy as np
import matplotlib, matplotlib.collections
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, AQUA, RED, PURPLE, GREEN, ORANGE = "#2a78d6", "#1b9aa8", "#e34948", "#5b2a86", "#3aa41c", "#e08a1e"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.2, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "pdf.fonttype": 42, "ps.fonttype": 42})
D = sys.argv[1] if len(sys.argv) > 1 else "/tmp/discharge_final/discharge"

def split(f, pms, tms, lag=30):
    rows = list(csv.DictReader(open(f))); T = [float(r["monotonic_s"]) for r in rows]
    P = [abs(float(r["usb_current_ua"])) / 1e6 * float(r["usb_voltage_uv"]) / 1e6 for r in rows]
    Q = [float(r["bat_charge_uah"]) for r in rows]; V = [float(r["bat_voltage_now_uv"]) / 1e6 for r in rows]
    G = [float(r.get("gpu_clk_hz") or 0) for r in rows]
    t0 = T[1] + 3
    for i in range(1, len(T)):
        if G[i] >= 9e8 and T[i] - T[0] < 90: t0 = T[i]; break
    tpf, tend = t0 + pms, t0 + tms; tq = tend + lag; ep = ed = 0; qa = qb = qc = None; vm = sum(V) / len(V)
    for i in range(1, len(T)):
        dt = min(5, T[i] - T[i - 1])
        if t0 < T[i] <= tpf: ep += P[i] * dt
        elif tpf < T[i] <= tend: ed += P[i] * dt
        if T[i] >= t0 and qa is None: qa = Q[i]
        if T[i] <= tpf: qb = Q[i]
        if T[i] <= tq: qc = Q[i]
    if qa is not None and qb is not None: ep += max(0, qa - qb) / 1e6 * vm * 3600
    if qb is not None and qc is not None: ed += max(0, qb - qc) / 1e6 * vm * 3600
    return ep, ed

R = []
for r in csv.DictReader(open(f"{D}/timeline.csv")):
    d = f"{D}/dis_{int(r['n']):03d}"
    if not (os.path.exists(d + "/meta.json") and os.path.exists(d + "/sensors.csv")): continue
    m = json.load(open(d + "/meta.json"))
    ep, ed = split(d + "/sensors.csv", m["prefill_ms"] / 1000, m["total_ms"] / 1000)
    R.append(dict(n=int(r["n"]), soc=float(r["soc"]), usb=r["usb_powered"] == "true", tier=r["tier"], plan=r["plan"], cap=int(r["nout_cap"]),
                  ns=m["n_decode_steps"], np=m["n_prompt_tokens"], E=ep + ed, T=m["total_ms"] / 1000, pred=float(r["pred_J"]), lever=float(r["lever"]), loop=r["loop"]))
R.sort(key=lambda x: x["n"])
bat = [x for x in R if not x["usb"]]; usb = [x for x in R if x["usb"]]
pcol = {"gpu1200_k1024": INK, "gpu1200d902_k1024": BLUE, "gpu1200d726_k1024": AQUA, "gpu902_k1024": ORANGE, "gpu726_k1024": RED}
plab = {"gpu1200_k1024": "1200 MHz", "gpu1200d902_k1024": "1200, decode 902", "gpu1200d726_k1024": "1200, decode 726", "gpu902_k1024": "902 MHz", "gpu726_k1024": "726 MHz"}
tband = {"healthy": "#eef3fb", "mid": "#fff3e3", "low": "#fde9e9"}


# per-tier means on battery (unchanged formulas)
stats = {}
for t, lo, hi in (("healthy", 50.5, 100), ("mid", 20.5, 50.5), ("low", 0, 20.5)):
    g = [x for x in bat if lo < x["soc"] <= hi]
    if not g: continue
    stats[t] = dict(n=len(g), E=st.mean(x["E"] for x in g), T=st.mean(x["T"] for x in g), cap=g[0]["cap"],
                    soc=(max(x["soc"] for x in g), min(x["soc"] for x in g)))
ug = usb
XL, XR = 93, 6                      # reversed x axis: 93 down to 6
BANDS = {"healthy": (XL, 50.5), "mid": (50.5, 20.5), "low": (20.5, XR)}

def pts_disp(ax, rend):
    """scatter points of ax in display px with their marker radius (px)"""
    out = []
    for c in ax.collections:
        if not isinstance(c, matplotlib.collections.PathCollection) or not len(c.get_offsets()): continue
        r = (np.sqrt(np.max(c.get_sizes())) / 2) * (ax.figure.dpi / 72) + 1.0
        for x, y in ax.transData.transform(c.get_offsets()): out.append((x, y, r))
    return out

def place(ax, txt, cands, avoid=(), pad=2.0, tpad=4.0):
    """move txt through candidate (x, y, va) triples until its box clears every point of ax and every text in avoid"""
    fig = ax.figure; fig.canvas.draw(); rend = fig.canvas.get_renderer(); pts = pts_disp(ax, rend)
    boxes = [t.get_window_extent(rend) for t in avoid]
    for x, y, va in cands:
        txt.set_position((x, y)); txt.set_va(va); bb = txt.get_window_extent(rend)
        if any(bb.x0 - r - pad < px < bb.x1 + r + pad and bb.y0 - r - pad < py < bb.y1 + r + pad for px, py, r in pts): continue
        if any(bb.x0 < o.x1 + tpad and o.x0 < bb.x1 + tpad and bb.y0 < o.y1 + tpad and o.y0 < bb.y1 + tpad for o in boxes): continue
        return True
    txt.set_position(cands[0][:2]); txt.set_va(cands[0][2]); return False

def audit(fig, pad=0.5):
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); items = []
    for ax in fig.axes:
        items += [(t.get_text(), t.get_window_extent(rend)) for t in ax.texts if t.get_text().strip()]
        items += [(t.get_text(), t.get_window_extent(rend)) for t in (ax.title, ax._left_title) if t.get_text().strip()]
        xl, yl = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        items += [(tk.label1.get_text(), tk.label1.get_window_extent(rend)) for tk in ax.xaxis.get_major_ticks() if tk.label1.get_visible() and xl[0] <= tk.get_loc() <= xl[1] and tk.label1.get_text().strip()]
        items += [(tk.label1.get_text(), tk.label1.get_window_extent(rend)) for tk in ax.yaxis.get_major_ticks() if tk.label1.get_visible() and yl[0] <= tk.get_loc() <= yl[1] and tk.label1.get_text().strip()]
        if ax.get_legend(): items += [(t.get_text(), t.get_window_extent(rend)) for t in ax.get_legend().get_texts()]
        for t in ax.texts + ([] if not ax.get_legend() else ax.get_legend().get_texts()):
            bb = t.get_window_extent(rend)
            if any(bb.x0 - r < px < bb.x1 + r and bb.y0 - r < py < bb.y1 + r for px, py, r in pts_disp(ax, rend)):
                print(f"  POINT under text: {t.get_text()!r}")
    bad = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            p, q = items[i][1], items[j][1]
            if p.x0 < q.x1 + pad and q.x0 < p.x1 + pad and p.y0 < q.y1 + pad and q.y0 < p.y1 + pad:
                print(f"  OVERLAP: {items[i][0]!r} <-> {items[j][0]!r}"); bad += 1
    w, h = fig.get_size_inches()
    for name, bb in items:
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > w * fig.dpi or bb.y1 > h * fig.dpi: print(f"  CLIPPED: {name!r}"); bad += 1
    print("  audit:", "clean" if not bad else f"{bad} issue(s)")

def render(name, size, fs, tick, layout, three_line):
    plt.rcParams.update({"font.size": fs, "axes.labelsize": fs, "legend.fontsize": tick, "xtick.labelsize": tick, "ytick.labelsize": tick})
    fig, (a, b) = plt.subplots(2, 1, figsize=size, sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1.0]))
    fig.subplots_adjust(**layout)
    for ax in (a, b):
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.axvspan(100, 50.5, color=tband["healthy"], zorder=0, lw=0); ax.axvspan(50.5, 20.5, color=tband["mid"], zorder=0, lw=0); ax.axvspan(20.5, 0, color=tband["low"], zorder=0, lw=0)
        ax.axvspan(100, min(x["soc"] for x in usb) - 0.5, facecolor="none", edgecolor=MUTED, hatch="////", lw=0, zorder=0, alpha=0.35)
        ax.axvline(50.5, color=MUTED, lw=0.6, ls=(0, (3, 2))); ax.axvline(20.5, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.tick_params(length=2, pad=2)
    a.set_title("(a) energy per request vs battery level", loc="left", fontsize=fs, pad=3)
    b.set_title("(b) request time and output cap", loc="left", fontsize=fs, pad=3)
    ms = 14 if size[0] > 5 else 9
    for p in dict.fromkeys(x["plan"] for x in R):
        xs = [x["soc"] for x in R if x["plan"] == p]; ys = [x["E"] for x in R if x["plan"] == p]
        a.scatter(xs, ys, s=ms, color=pcol.get(p, MUTED), edgecolor=SURF, linewidth=0.4, zorder=4, label=plab.get(p, p))
        b.scatter(xs, [x["T"] for x in R if x["plan"] == p], s=ms * 0.85, color=pcol.get(p, MUTED), edgecolor=SURF, linewidth=0.4, zorder=4)
    a.set_ylim(0, 1750); a.set_xlim(XL, XR); b.set_ylim(0, 400)
    a.set_ylabel("energy per request (J)"); b.set_ylabel("request time (s)"); b.set_xlabel("battery level read before the request (%)")
    a.set_yticks([0, 500, 1000, 1500]); b.set_yticks([0, 100, 200, 300])
    ptsE = [(x["soc"], x["E"]) for x in R]; ptsT = [(x["soc"], x["T"]) for x in R]
    for t, (bx0, bx1) in BANDS.items():
        if t not in stats: continue
        g = [x for x in bat if BANDS[t][1] < x["soc"] <= (100 if t == "healthy" else bx0)]
        lo, hi = min(x["soc"] for x in g), max(x["soc"] for x in g)
        a.hlines(stats[t]["E"], lo - 0.4, hi + 0.4, color=INK, lw=1.0, zorder=5)
        b.hlines(stats[t]["T"], lo - 0.4, hi + 0.4, color=INK, lw=1.0, zorder=5)
        xc = ((min(x["soc"] for x in usb) - 0.5) + bx1) / 2 if t == "healthy" else (bx0 + bx1) / 2   # healthy: centre over the on-battery part
        if three_line: lab = f"{t}, {stats[t]['cap']}\n{stats[t]['E']:.0f} J\nn={stats[t]['n']}"
        else: lab = f"{t}, {stats[t]['cap']} tok\n{stats[t]['E']:.0f} J, n={stats[t]['n']}"
        a.text(xc, 1720, lab, ha="center", va="top", fontsize=tick, color=INK, linespacing=1.15, zorder=6)
        captxt = b.text(xc, 388, f"cap {stats[t]['cap']}", ha="center", va="top", fontsize=tick, color=MUTED, zorder=6)
        # time mean label: above the line where no points sit
        tt = b.text(0, 0, f"{stats[t]['T']:.0f} s", ha="center", va="bottom", fontsize=tick, color=INK, zorder=6)
        fr = (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.12, 0.88)
        cands = []
        for off in (7, 20, 35, 50):
            cands += [(lo + (hi - lo) * f, stats[t]["T"] + off, "bottom") for f in fr]
            cands += [(lo + (hi - lo) * f, stats[t]["T"] - off, "top") for f in fr]
        ok = place(b, tt, cands, avoid=[captxt])
        if not ok: print(f"  could not place time label for {t}")
    b.text(XL - 0.8, 14, f"on the cable, n={len(ug)}" if size[0] > 5 else f"cable, n={len(ug)}", ha="left", va="bottom", fontsize=tick, color=MUTED, zorder=6)
    a.legend(loc="lower left", ncol=2 if size[0] > 5 else 1, fontsize=tick, handletextpad=0.3, columnspacing=0.9, borderaxespad=0.3, markerscale=1.0, labelspacing=0.3)
    audit(fig)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200); print("wrote", out, len(R), "requests")
    plt.close(fig)

render("fig_discharge_timeline", (7.0, 3.0), 7.5, 6.5,
       dict(left=0.075, right=0.99, top=0.925, bottom=0.145, hspace=0.42), False)
render("fig_discharge_timeline_col", (3.4, 3.2), 7.0, 6.5,
       dict(left=0.15, right=0.985, top=0.94, bottom=0.13, hspace=0.32), True)
print("on battery:", {t: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()} for t, s in stats.items()})
h = stats.get("healthy"); 
for t in ("mid", "low"):
    if t in stats and h: print(f"  {t} vs healthy on battery: energy {stats[t]['E']/h['E']-1:+.0%}, time {stats[t]['T']/h['T']-1:+.0%}")
fixed = [x for x in bat if x["n"] >= 100]
print(f"  prediction error after the fix (n={len(fixed)}): mean {st.mean(abs(x['E']/x['pred']-1) for x in fixed):.1%}, within 10%: {sum(abs(x['E']/x['pred']-1)<0.1 for x in fixed)}/{len(fixed)}")
sw = [(x["n"], x["soc"], x["tier"], x["plan"], x["cap"]) for i, x in enumerate(R) if i and x["tier"] != R[i-1]["tier"]]
print("  tier switches:", sw)
print("  per-token on battery (mJ, prompt+output):", {t: round(st.mean(x["E"]/(x["np"]+x["ns"])*1000 for x in bat if x["tier"]==t), 1) for t in ("healthy","mid","low")})
