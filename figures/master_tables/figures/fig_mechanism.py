#!/usr/bin/env python3
"""Mechanism figure v5: two-column fork with labeled outcome grids.
Colors: orange = new muKV mechanism, gray = unmodified llama.cpp, tan = prior-art
per-head evictors, green = muKV outcome, red = baseline failure, blue = data
structure. Scoring visualized as a 16xN heatmap (last-16 queries) reduced into
per-layer scores. Outcome grids share row geometry: left rows = layers (same
positions kept in every layer -> compactable), right rows = heads (per-head
keep sets -> union stays live, cannot compact). Evicted cells are light
dashed-outline squares; verdicts in dark ink with check/cross symbols."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

GRAY='#9a9a9a'; GRAYF='#f2f2f2'; INK='#1a1a1a'
OR='#D55E00'; ORF='#fdeee6'; BL='#0072B2'; BLF='#eaf3fa'
GR='#009E73'; GRF='#e8f6f1'; RD='#c00000'; RDF='#fdeaea'
TAN='#8c7a6b'; TANF='#f3ede4'
GRN_CELL='#2ba884'; RED_CELL='#d9534f'
plt.rcParams.update({'font.size': 7, 'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, ax = plt.subplots(figsize=(3.4, 5.3), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x, y, w, h, t, sub='', ec=GRAY, fc='white', fs=6.0, fsub=5.0, lw=1.1,
        tdy=1.1, sdy=-1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.35,rounding_size=1.0',
                                fc=fc, ec=ec, lw=lw, zorder=3))
    dy = tdy if sub else 0
    ax.text(x+w/2, y+h/2+dy, t, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold', zorder=4)
    if sub: ax.text(x+w/2, y+h/2+sdy, sub, ha='center', va='center', fontsize=fsub,
                    color=INK, zorder=4)

def varr(x, y1, y2, c=INK, lw=1.2, ls='-'):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle='-|>', mutation_scale=7,
                                 lw=lw, color=c, ls=ls, zorder=2, shrinkA=0.3, shrinkB=0.3))

def harr(x1, x2, y, c=INK, lw=1.2, ls='-', ms=7):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle='-|>', mutation_scale=ms,
                                 lw=lw, color=c, ls=ls, zorder=2, shrinkA=0.3, shrinkB=0.3))

def elbow(x1, y1, ym, x2, y2, c=INK, lw=1.2, ls='-'):
    ax.plot([x1, x1], [y1, ym], color=c, lw=lw, ls=ls, zorder=2)
    ax.plot([x1, x2], [ym, ym], color=c, lw=lw, ls=ls, zorder=2)
    ax.add_patch(FancyArrowPatch((x2, ym), (x2, y2), arrowstyle='-|>', mutation_scale=7,
                                 lw=lw, color=c, ls=ls, zorder=2, shrinkA=0, shrinkB=0.3))

# ---------------- transformer band (unmodified) ----------------
ax.text(2, 99.3, 'transformer layer (unmodified)', fontsize=5.4, color='#555555', va='center')
box(2, 90.5, 13, 7, 'Q K V', ec=GRAY, fc=GRAYF, fs=6.0)
box(20, 90.5, 36, 7, 'flash attention', 'scores never stored', ec=GRAY, fc=GRAYF)
box(61, 90.5, 36, 7, 'layer output', ec=GRAY, fc=GRAYF)
harr(15.4, 19.6, 94)
harr(56.4, 60.6, 94)

# ---------------- kq_evict side node + attention heatmap ----------------
box(2, 80, 52, 7.5, 'kq_evict side node',
    r'softmax$(K\,Q^{\top}_{\rm last\,16})$ · last chunk', ec=OR, fc=ORF, fs=6.2, sdy=-1.6)
ax.plot([38], [90.4], marker='o', ms=3.2, color=OR, zorder=6)   # tap dot on FA border
varr(38, 90.4, 88.0, c=OR, lw=1.2)

# heatmap: measured attention of 16 consecutive queries over the prompt positions
# (Llama-3.2-1B Q4_K_M, layer 8, head mean; attention_probe capture of a 3049-token
# gov_report prompt, 2026-09-07; positions binned to 64 columns by max so sparse anchors show).
import os
A = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'attn_llama1b_layer8_16q_bins64.npy'))
A = np.log10(np.clip(A / A.max(), 1e-3, 1.0)) / 3 + 1   # log scale over three decades, for display
ax.imshow(A, extent=[62, 96, 79, 87], aspect='auto', cmap='Oranges', vmin=0, vmax=1.0,
          origin='upper', interpolation='nearest', zorder=3)
ax.add_patch(Rectangle((62, 79), 34, 8, fc='none', ec=OR, lw=0.8, zorder=4))
ax.text(79, 88.5, r'$A\in\mathbb{R}^{16\times N}$, measured (log scale)', fontsize=5.0,
        ha='center', va='center', color=INK)
harr(54.6, 61.6, 83.3, c=OR, lw=1.1)                            # kq_evict emits A
varr(79, 78.7, 75.8, c=OR, lw=1.1)                              # column-reduce
ax.text(81.6, 76.8, 'reduce over\nqueries', fontsize=5.0, ha='left', va='center', color=INK)

# ---------------- shared per-layer scores (data) ----------------
box(24, 69, 58, 6.4, 'per-layer scores',
    r'$s\in\mathbb{R}^{L\times N}$  (from $A\in\mathbb{R}^{16\times N}$)',
    ec=BL, fc=BLF, fs=6.0, sdy=-1.5)

# ---------------- fork + symmetric column headers ----------------
LXc, RXc = 27, 76
ax.text(LXc, 63.2, r'(a) $\mu$KV (ours)', fontsize=6.4, ha='center', va='center',
        fontweight='bold', color=INK)
ax.text(RXc, 63.2, '(b) per-head evictors', fontsize=6.4, ha='center', va='center',
        fontweight='bold', color=INK)
elbow(34, 69, 66.5, 12, 60.9, c=OR, lw=1.3)
elbow(70, 69, 66.5, 58, 55.2, c=TAN, lw=1.1, ls=(0, (4, 2)))

# ---------------- left column: muKV ----------------
Lx, LW_ = 4, 46
box(Lx, 55.2, LW_, 5.7, 'per-head budget gate',
    r'$K_h$: ramp on $\max a_h$ $\to$ candidate pool', ec=OR, fc=ORF)
box(Lx, 47.2, LW_, 5.7, 'aggregate layers + heads',
    'one shared score per position', ec=OR, fc=ORF)
box(Lx, 39.2, LW_, 5.7, r'$\alpha$-gate',
    r'split $K$ $\to$ sink + anchor + recent', ec=OR, fc=ORF)
varr(LXc, 55.2, 53.2, c=OR)
varr(LXc, 47.2, 45.2, c=OR)
varr(LXc, 39.2, 37.4, c=OR)

# keep-set strip (data) with labeled partition
ax.add_patch(FancyBboxPatch((Lx, 28), LW_, 9, boxstyle='round,pad=0.35,rounding_size=1.0',
                            fc=BLF, ec=BL, lw=1.1, zorder=3))
ax.text(LXc, 35.2, r'joint keep set: $K$ positions', ha='center', va='center',
        fontsize=6.0, fontweight='bold', zorder=4)
# strip proportional to the measured K=1024 split on Llama-3.2-1B (Table 1 cell): 4 sinks, 721 anchors, 299 recent
ax.add_patch(Rectangle((8, 30.8), 0.6, 2.6, fc='#c7dcef', ec=BL, lw=0.5, zorder=4))
ax.add_patch(Rectangle((8.6, 30.8), 26.5, 2.6, fc='#9ec6e8', ec=BL, lw=0.5, zorder=4))
ax.add_patch(Rectangle((35.1, 30.8), 10.9, 2.6, fc='#dcebf7', ec=BL, lw=0.5, zorder=4))
for xd in (8.6, 35.1):
    ax.plot([xd, xd], [30.1, 30.8], color=BL, lw=0.6, zorder=4)
ax.text(8.3, 29.3, 'sink 4', fontsize=5.0, ha='left', va='center', color=INK, zorder=5)
ax.text(24.5, 29.3, 'anchor 721', fontsize=5.0, ha='center', va='center',
        color=INK, zorder=5)
ax.text(46.0, 29.3, 'recent 299', fontsize=5.0, ha='right', va='center', color=INK, zorder=5)
varr(LXc, 28, 26.4, c=BL)

# ---------------- right column: prior-art per-head evictors (tan) ----------------
Rx, RW = 54, 44
box(Rx, 47, RW, 8, r'per-head top-$K$',
    'each head keeps its own\n$K$ positions', ec=TAN, fc=TANF, tdy=1.7, sdy=-1.6)
varr(RXc, 47, 26.4, c=TAN, ls=(0, (4, 2)), lw=1.1)

# ---------------- outcome grids (row-aligned) ----------------
# keep sets from the measured capture (see data/mech_keepsets.json): 16 position bins of the
# 3049-token prompt, last 16 queries, Llama-3.2-1B layer 8. Joint = top-6 bins of the mean over
# layers and heads (applied to every layer); per-head = each head's own top-6 bins; union = over all 32 heads.
_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_JOINT = np.load(os.path.join(_D, 'mech_joint_keep_16bins.npy'))
_PERHEAD = np.load(os.path.join(_D, 'mech_perhead_keep_layer8_16bins.npy'))
KEEP = [i for i in range(16) if _JOINT[i]]
NB = 16
GY = [20.2, 18.2, 16.2, 14.2]          # shared row baselines (grid rows)
CW, CH, PITCH = 2.05, 1.7, 2.25

def cell(x, y, kept, kc, ec_ev):
    if kept:
        ax.add_patch(Rectangle((x, y), CW, CH, fc=kc, ec='white', lw=0.3, zorder=4))
    else:
        ax.add_patch(Rectangle((x, y), CW, CH, fc='white', ec=ec_ev, lw=0.4,
                               ls=(0, (1.2, 1.2)), zorder=4))

# left container: muKV outcome (green)
ax.add_patch(FancyBboxPatch((Lx, 4.7), LW_, 21.5, boxstyle='round,pad=0.35,rounding_size=1.0',
                            fc=GRF, ec=GR, lw=1.1, zorder=3))
ax.text(LXc, 24.2, 'seq-level evict + compact', ha='center', va='center', fontsize=6.0,
        fontweight='bold', zorder=4)
for r, gy in enumerate(GY):
    for c in range(NB):
        cell(9.5 + c*PITCH, gy, c in KEEP, GRN_CELL, '#9fb8ae')
ax.text(7.8, 17.9, 'layers', fontsize=5.0, rotation=90, ha='center', va='center', zorder=4)
ax.text(27.5, 12.7, 'token position', fontsize=5.0, ha='center', va='center', zorder=4)
# compaction: scattered kept cells -> compact -> contiguous strip
for c in range(NB):
    x = 6 + c*1.22
    if c in KEEP:
        ax.add_patch(Rectangle((x, 7.7), 1.1, CH, fc=GRN_CELL, ec='white', lw=0.3, zorder=4))
    else:
        ax.add_patch(Rectangle((x, 7.7), 1.1, CH, fc='white', ec='#9fb8ae', lw=0.4,
                               ls=(0, (1.2, 1.2)), zorder=4))
harr(26.3, 31.6, 8.55, c=GR, lw=1.2, ms=6)
ax.text(28.9, 10.6, 'compact', fontsize=5.0, ha='center', va='center', zorder=4)
for i in range(len(KEEP)):
    ax.add_patch(Rectangle((32.6 + i*1.62, 7.7), 1.45, CH, fc=GRN_CELL, ec='white',
                           lw=0.3, zorder=4))
ax.text(27, 5.9, 'same positions dropped in every layer', fontsize=5.0, ha='center',
        va='center', color=INK, zorder=4)

# right container: baseline failure (red)
ax.add_patch(FancyBboxPatch((Rx, 4.7), RW, 21.5, boxstyle='round,pad=0.35,rounding_size=1.0',
                            fc=RDF, ec=RD, lw=1.1, zorder=3))
ax.text(RXc, 24.2, 'union across heads', ha='center', va='center', fontsize=6.0,
        fontweight='bold', zorder=4)
SHOW_HEADS = [0, 4, 16, 24]
PICKS = [[c for c in range(NB) if _PERHEAD[h, c]] for h in SHOW_HEADS]
for r, gy in enumerate(GY):
    for c in range(NB):
        cell(58.5 + c*PITCH, gy, c in PICKS[r], RED_CELL, '#c9a9a9')
ax.text(96.6, 17.9, 'heads', fontsize=5.0, rotation=270, ha='center', va='center', zorder=4)
ax.text(76.5, 12.7, 'token position', fontsize=5.0, ha='center', va='center', zorder=4)
# live set = union of all heads (only col 8 ever dies) -> long, gappy, not compactable
UNION = [c for c in range(NB) if _PERHEAD[:, c].any()]   # union over all 32 heads
for c in range(NB):
    cell(58.5 + c*PITCH, 7.7, c in UNION, RED_CELL, '#c9a9a9')
ax.text(76.5, 10.6, f'union of 32 heads: {len(UNION)} of {NB} live', fontsize=5.0, ha='center', va='center', zorder=4)
ax.text(76, 5.9, r'any head keeps it $\to$ it survives', fontsize=5.0, ha='center',
        va='center', color=INK, zorder=4)

# ---------------- verdicts (dark ink, grayscale-safe symbols) ----------------
ax.text(7, 3.1, r'live $= K$, contiguous', fontsize=5.0, ha='left', va='center',
        color=INK, fontweight='bold')      # measured width 29.8 -> ends x 36.8
ax.text(39.3, 3.1, '✓', fontsize=7, ha='center', va='center', color=GR, fontweight='bold')
ax.text(45.8, 3.1, r'live $\gg K$ (union), cannot compact', fontsize=5.0, ha='left',
        va='center', color=INK, fontweight='bold')  # measured width 49.3 -> ends x 95.1
ax.text(97.4, 3.1, '✗', fontsize=7, ha='center', va='center', color=RD, fontweight='bold')

# ---------------- legend row ----------------
def sw(x, fc, ec):
    ax.add_patch(FancyBboxPatch((x, 0.2), 2.0, 1.2, boxstyle='round,pad=0.12',
                                fc=fc, ec=ec, lw=0.8))
sw(4, ORF, OR);      ax.text(6.8, 0.8, 'new (μKV)', fontsize=5.2, va='center')
sw(21, GRAYF, GRAY); ax.text(23.8, 0.8, 'unmodified', fontsize=5.2, va='center')
sw(39.5, TANF, TAN); ax.text(42.3, 0.8, 'prior art', fontsize=5.2, va='center')
sw(56.5, GRF, GR);   ax.text(59.3, 0.8, 'outcome', fontsize=5.2, va='center')
sw(71, RDF, RD);     ax.text(73.8, 0.8, 'failure', fontsize=5.2, va='center')

fig.tight_layout()
out='/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures/fig_mechanism'
fig.savefig(out+'.png', bbox_inches='tight', dpi=300)
fig.savefig(out+'.pdf', bbox_inches='tight')
print('saved', out)
