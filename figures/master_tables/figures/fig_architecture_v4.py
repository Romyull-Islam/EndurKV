#!/usr/bin/env python3
"""muKV architecture, v5. Conventions: left-to-right dataflow inside an explicit
OnePlus 15 device boundary with two hardware bands (llama.cpp inference, SoC
sensors + DVFS). The KV cache is drawn as a cell-grid artifact (N-cell strip ->
K-cell strip) so the shrink is visible. Every arrow labeled with its payload.
Arrow styles: data solid, control dashed, heat dotted. Orange badges 1-4 mark
muKV mechanisms only. Colors (Okabe-Ito): orange = new muKV mechanism, gray =
unmodified llama.cpp/hardware, green = muKV outcome, blue = KV-cache data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

GRAY='#9a9a9a'; GRAYF='#f2f2f2'; INK='#1a1a1a'
ORANGE='#D55E00'; ORANGEF='#fdeee6'; BLUE='#0072B2'; BLUEF='#eaf3fa'
GREEN='#009E73'; GREENF='#e8f6f1'; HEAT='#777777'
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x, y, w, h, title, sub='', ec=GRAY, fc='white', fs=6.8, fsub=5.4, lw=1.1,
        tdy=2.6, sdy=-3.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.4',
                                fc=fc, ec=ec, lw=lw, zorder=3))
    dy = tdy if sub else 0
    ax.text(x+w/2, y+h/2+dy, title, ha='center', va='center', fontsize=fs,
            color=INK, fontweight='bold', zorder=4)
    if sub: ax.text(x+w/2, y+h/2+sdy, sub, ha='center', va='center', fontsize=fsub,
                    color=INK, zorder=4)

def arr(x1, y1, x2, y2, c=INK, lw=1.2, ls='-', ms=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms,
                                 lw=lw, color=c, ls=ls, zorder=2, shrinkA=0.5, shrinkB=0.5))

def step(x, y, n):
    ax.annotate(str(n), (x, y), ha='center', va='center', fontsize=6.2, color='white',
                fontweight='bold', zorder=7,
                bbox=dict(boxstyle='circle,pad=0.30', fc=ORANGE, ec='none'))

# ================= device boundary + hardware bands =================
ax.add_patch(FancyBboxPatch((0.8, 5.5), 98.4, 93, boxstyle='round,pad=0.1,rounding_size=1.5',
                            fc='white', ec=INK, lw=1.2, zorder=0))
ax.text(3.0, 96.2, 'OnePlus 15', fontsize=7.5, fontweight='bold', color=INK, ha='left',
        va='center', zorder=1)

ax.add_patch(FancyBboxPatch((2.5, 41), 95, 53, boxstyle='round,pad=0.1,rounding_size=1.2',
                            fc='#fcfcfc', ec='#bbbbbb', lw=0.9, zorder=1))
ax.text(4.0, 91.3, 'llama.cpp inference (CPU / Adreno GPU)', fontsize=6.2, style='italic',
        color='#555555', ha='left', va='center', zorder=2)

ax.add_patch(FancyBboxPatch((2.5, 8), 95, 28, boxstyle='round,pad=0.1,rounding_size=1.2',
                            fc='#fcfcfc', ec='#bbbbbb', lw=0.9, zorder=1))
ax.text(4.0, 33.2, 'SoC sensors + DVFS', fontsize=6.2, style='italic', color='#555555',
        ha='left', va='center', zorder=2)

# ================= top band: pipeline row =================
y0, h0 = 71, 16
box(4,  y0, 8,  h0, 'Prompt', 'N tokens', ec=GRAY, fc=GRAYF)
box(17, y0, 14, h0, 'FA-on prefill', 'llama.cpp graph', ec=GRAY, fc=GRAYF)
box(40, y0, 15, h0, 'Score + select', 'budget gate ·\naggregate · α-gate',
    ec=ORANGE, fc=ORANGEF, fsub=5.2, sdy=-3.8)
box(62, y0, 12, h0, 'Evict + defrag', 'seq-level', ec=ORANGE, fc=ORANGEF)
# decode box drawn manually so the per-step-evict chip fits inside
ax.add_patch(FancyBboxPatch((79, y0), 17, h0, boxstyle='round,pad=0.4,rounding_size=1.4',
                            fc=GREENF, ec=GREEN, lw=1.1, zorder=3))
ax.text(87.5, 83.8, 'FA-on decode', ha='center', va='center', fontsize=6.8,
        fontweight='bold', color=INK, zorder=4)
ax.text(87.5, 79.9, 'bounded cache', ha='center', va='center', fontsize=5.4, color=INK, zorder=4)
ax.add_patch(FancyBboxPatch((81.5, 72.8), 12, 4, boxstyle='round,pad=0.25,rounding_size=0.8',
                            fc=ORANGEF, ec=ORANGE, lw=0.9, zorder=5))
ax.text(87.5, 74.8, 'per-step evict', ha='center', va='center', fontsize=5.2, color=INK, zorder=6)

# pipeline arrows with payload labels
arr(12.2, 79, 16.8, 79, c=INK)                       # data
ax.text(14.5, 81.6, 'tokens', fontsize=5.0, ha='center', va='center', color=INK)

# kq_evict tap: dot on prefill border, label on the arrow (data, solid)
arr(31.2, 83, 39.8, 83, c=ORANGE, lw=1.1)
ax.plot([31.2], [83], marker='o', ms=3.6, color=ORANGE, zorder=6)
ax.text(35.2, 85.4, 'kq_evict', fontsize=5.4, ha='center', va='center', color=ORANGE,
        fontweight='bold')
ax.text(35.5, 80.8, 'K,Q', fontsize=5.0, ha='center', va='center', color=ORANGE)
ax.text(35.5, 78.3, '(last chunk)', fontsize=5.0, ha='center', va='center', color=ORANGE)

arr(55.2, 79, 61.8, 79, c=ORANGE, lw=1.1)            # data
ax.text(58.4, 81.6, 'keep set', fontsize=5.0, ha='center', va='center', color=ORANGE)
ax.text(58.4, 76.9, '(K indices)', fontsize=5.0, ha='center', va='center', color=ORANGE)

# ================= top band: KV cache strips (the artifact) =================
# N-cell strip after prefill
for i in range(18):
    ax.add_patch(Rectangle((8 + i*2.0, 50.5), 1.85, 5, fc=BLUEF, ec=BLUE, lw=0.5, zorder=3))
ax.text(26, 47.4, 'full KV cache · N cells', fontsize=5.3, ha='center', va='center', color=INK)
arr(24, 70.6, 24, 56.2, c=BLUE, lw=1.2)              # data: prefill writes cache
ax.text(25.3, 63.5, 'write full KV (all N)', fontsize=5.2, ha='left', va='center', color=BLUE)

# shrink arrow N -> K
arr(45, 53, 60.7, 53, c=BLUE, lw=2.0, ms=15)
ax.text(53, 56.2, 'N → K', fontsize=5.4, ha='center', va='center', color=INK, fontweight='bold')
ax.text(53, 49.4, 'evict + compact', fontsize=5.2, ha='center', va='center', color=INK)

# K-cell strip after evict+defrag: sink | anchor | recent
segs = [(2, '#c7dcef', 'sink'), (4, '#9ec6e8', 'anchor'), (3, '#dcebf7', 'recent')]
xk = 62
for n, fc, name in segs:
    for i in range(n):
        ax.add_patch(Rectangle((xk + i*1.55, 50.5), 1.45, 5, fc=fc, ec=BLUE, lw=0.5, zorder=3))
    ax.text(xk + n*1.55/2, 48.4, name, fontsize=5.0, ha='center', va='center', color=INK)
    xk += n*1.55
ax.text(69, 45.4, 'K cells · contiguous', fontsize=5.3, ha='center', va='center', color=INK)
arr(68, 70.6, 68, 56.2, c=ORANGE, lw=1.1, ls=(0, (4, 2)))   # control: apply keep set
ax.text(69.3, 63.5, 'serialize → restore', fontsize=5.2, ha='left', va='center', color=ORANGE)

# decode reads the bounded strip (elbow)
ax.plot([76.4, 86], [53, 53], color=BLUE, lw=1.2, zorder=2)
arr(86, 53, 86, 70.6, c=BLUE, lw=1.2)
ax.text(81.2, 50.6, 'read K cells', fontsize=5.2, ha='center', va='center', color=BLUE)

# ================= bottom band: sensors -> watchdog -> clock loop =================
yb, hb = 14, 15
box(8,  yb, 20, hb, 'Sensors', 'skin · battery · DDR · CPU', ec=GRAY, fc=GRAYF, fsub=5.2)
box(42, yb, 20, hb, 'Watchdog', 'lower cap 1 step,\nnever raise', ec=ORANGE, fc=ORANGEF,
    fsub=5.2, sdy=-3.8)
box(74, yb, 20, hb, 'CPU / GPU clock', 'DVFS cap', ec=GRAY, fc=GRAYF)

arr(28.2, 21.5, 41.8, 21.5, c=INK)                   # data: telemetry
ax.text(35, 25.9, r'$T_{\rm skin}\ T_{\rm batt}\ T_{\rm DDR}\ T_{\rm CPU}$',
        fontsize=5.2, ha='center', va='center', color=INK)
ax.text(35, 23.4, '@ 2 Hz', fontsize=5.2, ha='center', va='center', color=INK)

arr(62.2, 21.5, 73.8, 21.5, c=ORANGE, lw=1.2, ls=(0, (4, 2)))   # control
ax.text(68, 24.3, 'cap −1 step', fontsize=5.2, ha='center', va='center', color=ORANGE)

# clock governs the compute band (control, dashed)
arr(90, 29.4, 90, 70.6, c=GRAY, lw=1.1, ls=(0, (4, 2)))
ax.text(91.3, 46, r'$f \leq$ cap', fontsize=5.2, ha='left', va='center', color='#555555',
        rotation=90)

# heat: silicon (compute band) -> sensors (dotted), closing the loop
arr(18, 41, 18, 29.4, c=HEAT, lw=1.4, ls=(0, (1, 1.6)))
ax.text(19.3, 35.2, 'heat', fontsize=5.2, ha='left', va='center', color=HEAT)

# ================= step badges: muKV mechanisms only =================
step(40, y0+h0, 1)      # score + select
step(62, y0+h0, 2)      # evict + defrag
step(81.5, 76.9, 3)     # per-step decode evict (on the orange chip, not the result box)
step(42, yb+hb, 4)      # watchdog

# ================= legend: colors + arrow styles =================
def sw(x, fc, ec):
    ax.add_patch(FancyBboxPatch((x, 1.0), 2.4, 2.2, boxstyle='round,pad=0.15',
                                fc=fc, ec=ec, lw=0.9))
sw(2, ORANGEF, ORANGE); ax.text(5.2, 2.1, 'new μKV mechanism', fontsize=5.4, va='center')
sw(18.5, GRAYF, GRAY);  ax.text(21.7, 2.1, 'unmodified llama.cpp / hardware', fontsize=5.4, va='center')
sw(42, GREENF, GREEN);  ax.text(45.2, 2.1, 'μKV outcome', fontsize=5.4, va='center')
sw(54.5, BLUEF, BLUE);  ax.text(57.7, 2.1, 'KV cache (data)', fontsize=5.4, va='center')
ax.plot([69.5, 73], [2.1, 2.1], color=INK, lw=1.2)
ax.text(74, 2.1, 'data', fontsize=5.4, va='center')
ax.plot([79.5, 83], [2.1, 2.1], color=INK, lw=1.2, ls=(0, (4, 2)))
ax.text(84, 2.1, 'control', fontsize=5.4, va='center')
ax.plot([90.5, 94], [2.1, 2.1], color=HEAT, lw=1.4, ls=(0, (1, 1.6)))
ax.text(95, 2.1, 'heat', fontsize=5.4, va='center')

fig.tight_layout()
out = '/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_architecture_v4'
fig.savefig(out + '.png', bbox_inches='tight', dpi=300)
fig.savefig(out + '.pdf', bbox_inches='tight')
print('saved', out)
