#!/usr/bin/env python3
# ============================================================================
# fig_gpu_sustained_thermal.py -- the GPU counterpart to fig_bonsai_thermal.
# (2026-08-10)
#
# The CPU figure shows the watchdog converting vanilla's prime-clock sawtooth into a
# plateau under sustained Bonsai-8B load. This is the same measurement on the Adreno 840
# GPU: 8 back-to-back 4096-token generations per arm, no cool gate between iterations, so
# the device reaches thermal equilibrium. Every arm is GPU (Vulkan0, 17/17 layers) and
# every arm starts from the project cool gate (DDR<=35 C, battery<=33 C); measured start
# temperatures agree across arms to 0.7 C, which is what makes them comparable.
#
# TRACES ARE TIME-TRUNCATED, AND THAT IS A REPAIR, NOT A CHOICE. The per-arm clock
# samplers were not torn down correctly during the run (the teardown pattern never
# matched), so all four kept sampling to the end of the campaign and each raw trace
# contains its own arm PLUS every later arm superimposed -- vanilla's file spans 287 min
# instead of 47. Because all samplers watch the SAME device, however, each trace is
# CORRECT during its own arm's window and only wrong afterwards. Truncating each at the
# moment the next arm begins cooling recovers exactly the right data. Any analysis of the
# untruncated files is void, which is how a "lower mean clock but higher throughput"
# contradiction appeared before this was caught.
#
# SKIN COMES FROM sensors.csv, NOT THE CLOCK SAMPLER. The clock sampler took skin as the
# max over hardcoded thermal zones 55/56/57/62, and zone IDs shift across reboots on this
# device -- it was reading a junction sensor at 95 C. sensors.csv carries named channels
# (shell_front/frame/back), so skin here is the max of those three by NAME. This is the
# same class of bug that once made the watchdog itself inert.
# ============================================================================
import csv, os, re, json, glob, datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLK = "/tmp/gpusus_clk"
SENS = "/tmp/gpusus_sens"
OUT = "/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures"

def ts(hhmm):
    d = dt.datetime(2026, 8, 9, *map(int, hhmm.split(":")))
    return d.timestamp()

# (tag, label, colour, run start, run end == next arm's cool-down start)
ARMS = [
    ("vanilla",     "vanilla",                       "#888888", ts("17:58"), ts("18:55")),
    ("mukv_nowd",   r"$\mu$KV",                      "#0072B2", ts("19:19"), ts("20:11")),
    ("mukv_wdlow",  r"$\mu$KV + watchdog",           "#D55E00", ts("20:38"), ts("21:25")),
    ("mukv_wdhigh", r"$\mu$KV + watchdog (never fired)", "#009E73", ts("21:50"), ts("22:45")),
]

def load_clk(tag, t0, t1):
    out = []
    f = os.path.join(CLK, tag + ".csv")
    if not os.path.exists(f):
        return out
    for r in csv.DictReader(open(f)):
        try:
            t = int(r["t_s"])
        except (KeyError, ValueError, TypeError):
            continue
        if t0 <= t <= t1:
            try:
                out.append(((t - t0) / 60.0, int(r["gpu_clk"]), int(r["gpu_max"])))
            except (ValueError, TypeError):
                pass
    return out

def load_sens(tag, t0, t1):
    """battery / skin / DDR from NAMED columns; skin = max(shell_front, frame, back)."""
    out = []
    f = os.path.join(SENS, tag + ".csv")
    if not os.path.exists(f):
        return out
    for r in csv.DictReader(open(f)):
        try:
            t = float(r.get("wall_clock_s") or 0)
        except (TypeError, ValueError):
            continue
        if not (t0 <= t <= t1):
            continue
        def g(c):
            try:
                return float(r[c]) / 1000.0
            except (KeyError, TypeError, ValueError):
                return None
        skin = [v for v in (g("shell_front_temp_mc"), g("shell_frame_temp_mc"), g("shell_back_temp_mc")) if v]
        b, d = g("battery_temp_mc"), g("ddr_temp_mc")
        if skin and b and d:
            out.append(((t - t0) / 60.0, b, max(skin), d))
    return out

# ---- per-iteration phase boundaries ------------------------------------------------
# WHY THIS PANEL. Each iteration is a 12220-token prompt followed by 4096 generated
# tokens, and PREFILL IS ~65% OF THE ITERATION -- so "when did prefill finish" is not a
# detail, it decides which phase the watchdog's higher clock is actually buying. Start
# times come from the run log; the model-load gap (wall between iterations minus the
# meta's total_ms) is attributed BEFORE prefill, which is where llama.cpp does it, so a
# boundary here is accurate to roughly the load time and is drawn as a band, not a line.
LOG = "/tmp/gpusus.log"
def iter_starts():
    out = {}
    if not os.path.exists(LOG):
        return out
    for line in open(LOG, errors="replace"):
        m = re.search(r'\[(\d\d):(\d\d):(\d\d)\]\s+(\S+) iter (\d+)/', line)
        if m:
            hh, mm, ss, tag, i = m.groups()
            out.setdefault(tag, {})[int(i)] = dt.datetime(2026, 8, 9, int(hh), int(mm), int(ss)).timestamp()
    return out

def phases(tag, t0, t1):
    """[(prefill_start_min, prefill_end_min, decode_end_min), ...] within the arm."""
    st_ = iter_starts().get(tag, {})
    out = []
    for f in sorted(glob.glob("/tmp/gpu_sustained/%s/iter_*.json" % tag)):
        i = int(re.search(r'iter_(\d+)', f).group(1))
        if i not in st_:
            continue
        j = json.loads(re.sub(r':\s*-?inf\b', ': Infinity', re.sub(r':\s*-?nan\b', ': NaN', open(f).read())))
        nxt = st_.get(i + 1)
        wall = (nxt - st_[i]) if nxt else (j["total_ms"] / 1000 + 20)
        load = max(0.0, wall - j["total_ms"] / 1000)          # model load precedes prefill
        a = (st_[i] - t0) / 60.0 + load / 60.0
        b = a + j["prefill_ms"] / 1000 / 60.0
        c = b + j["decode_ms"] / 1000 / 60.0
        out.append((a, b, c))
    return out

fig, ax = plt.subplots(5, 1, figsize=(6.4, 9.4), sharex=True,
                       gridspec_kw={"height_ratios": [1.1, 2, 2, 2, 2]})
axp = ax[0]; ax = ax[1:]
for k, (tag, lab, col, t0, t1) in enumerate(ARMS):
    y = len(ARMS) - 1 - k
    for (a_, b_, c_) in phases(tag, t0, t1):
        axp.barh(y, b_ - a_, left=a_, height=0.62, color=col, alpha=0.95)      # prefill
        axp.barh(y, c_ - b_, left=b_, height=0.62, color=col, alpha=0.30)      # decode
axp.set_yticks(range(len(ARMS)))
axp.set_yticklabels([l for _, l, _, _, _ in ARMS][::-1], fontsize=7)
axp.set_ylabel("phase", fontsize=9)
axp.set_title("Adreno 840, sustained: 8 x [12220-token prompt + 4096 generated], no gate between iterations\n"
              "solid = prefill   faded = decode", fontsize=7.5, loc="left", pad=4)
for sp in ("top", "right", "left"):
    axp.spines[sp].set_visible(False)
axp.tick_params(axis="y", length=0)
axp.grid(axis="x", alpha=0.25, lw=0.5)

for tag, lab, col, t0, t1 in ARMS:
    c = load_clk(tag, t0, t1)
    if c:
        ax[0].plot([x[0] for x in c], [x[2] for x in c], color=col, lw=1.2, label=lab)
    s = load_sens(tag, t0, t1)
    if s:
        ax[1].plot([x[0] for x in s], [x[1] for x in s], color=col, lw=1.2, label=lab)
        ax[2].plot([x[0] for x in s], [x[2] for x in s], color=col, lw=1.2)
        ax[3].plot([x[0] for x in s], [x[3] for x in s], color=col, lw=1.2)

ax[0].set_ylabel("GPU clock cap\n(MHz)")
ax[1].set_ylabel("battery\n(°C)")
ax[2].set_ylabel("skin\n(°C)")
ax[3].set_ylabel("DDR\n(°C)")
ax[3].set_xlabel("elapsed time within arm (min)")

# the v5LOW ladder's first two rungs -- the thresholds this workload actually reaches
ax[1].axhline(36.0, ls=":", lw=0.9, color="#B00020")
ax[1].text(0.99, 36.0, " v5LOW bat 36 °C", color="#B00020", fontsize=7,
           ha="right", va="bottom", transform=ax[1].get_yaxis_transform())
ax[2].axhline(39.5, ls=":", lw=0.9, color="#B00020")
ax[2].text(0.99, 39.5, " v5LOW skin 39.5 °C", color="#B00020", fontsize=7,
           ha="right", va="bottom", transform=ax[2].get_yaxis_transform())

for a in ax:
    a.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
ax[0].set_ylim(380, 1320)
ax[0].legend(fontsize=7, frameon=False, ncol=2, loc="lower center",
             bbox_to_anchor=(0.5, -0.04), handlelength=1.6, columnspacing=1.2)
ax[1].set_ylim(32, 43.5)
ax[2].set_ylim(33, 46.5)
ax[3].set_ylim(35, 84)
# annotate the one thing the figure exists to show
ax[0].annotate("watchdog holds 826-1050 MHz\nwhile every other arm sits at 726",
               xy=(11, 826), xytext=(24, 1080), fontsize=7, color="#D55E00",
               arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8))
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "fig_gpu_sustained_thermal." + ext), dpi=200, bbox_inches="tight")
print("wrote", os.path.join(OUT, "fig_gpu_sustained_thermal.pdf"))
for tag, lab, col, t0, t1 in ARMS:
    print("  %-12s clk pts=%5d  sens pts=%5d  window=%.0f min"
          % (tag, len(load_clk(tag, t0, t1)), len(load_sens(tag, t0, t1)), (t1 - t0) / 60))
