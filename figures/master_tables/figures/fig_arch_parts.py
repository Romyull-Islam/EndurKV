#!/usr/bin/env python3
"""Break the paper's Figure 1 into talk-sized parts.

Runs fig_architecture_full_v9.py verbatim (minus its savefig) so every part is
pixel-identical to the published figure, then dims everything outside one region
and outlines that region. One PNG per part, for the presentation deck.
"""
import os, re, matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'fig_architecture_full_v9.py')).read()
SRC = re.sub(r"\nfig\.savefig\(.*?\n", "\n", SRC)
SRC = re.sub(r"\nprint\('ok;.*?\n", "\n", SRC)

# name, title, focus box (x0, y0, x1, y1) in the figure's 0..100 space
PARTS = [
    ('00_whole',    'The whole system',            None),
    ('01_prefill',  'Prefill runs unmodified',     (3.4, 69.8, 32.0, 86.2)),
    ('02_score',    'Score inside the FA graph',   (32.9, 69.8, 48.6, 87.3)),
    ('03_select',   'One keep set, all heads',     (49.4, 69.8, 64.2, 87.3)),
    ('04_compact',  'Compact in place',            (64.0, 59.4, 85.2, 87.3)),
    ('05_decode',   'Decode on K cells',           (81.8, 69.8, 96.5, 86.2)),
    ('06_kv',       'What the cache keeps',        (3.4, 58.9, 96.8, 68.8)),
    ('07_thermal',  'Thermal: clock ladders',      (3.2, 6.9, 33.2, 52.6)),
    ('08_energy',   'Energy: tier, lever, plan',   (35.2, 6.9, 66.2, 52.6)),
    ('09_learn',    'Learning: meter and loops',   (68.2, 6.9, 97.4, 52.6)),
]
DIM, EDGE = 0.80, '#D55E00'
FW0, FH0 = 7.2, 3.64          # the source figure's own size, so a crop keeps its proportions
UX, UY = 100.0 / FW0, 100.0 / FH0
MARGIN = 1.5

def build():
    g = {'__name__': '__parts__'}
    exec(compile(SRC, 'fig_architecture_full_v9.py', 'exec'), g)
    return g['fig'], g['ax']

def save(fig, name, tight=True):
    # zooms are small in inches and get blown up on a slide, so they need the extra dpi
    kw = dict(bbox_inches='tight', pad_inches=0.02) if tight else {}
    fig.savefig(os.path.join(HERE, 'parts', name + '.png'), dpi=220 if tight else 900,
                facecolor='white', **kw)
    print('wrote parts/%s.png' % name)

for name, _title, box in PARTS:
    # locator: the whole figure, everything but this part dimmed
    fig, ax = build()
    if box:
        x0, y0, x1, y1 = box
        for rx, ry, rw, rh in ((0, y1, 100, 100 - y1), (0, 0, 100, y0),
                               (0, y0, x0, y1 - y0), (x1, y0, 100 - x1, y1 - y0)):
            ax.add_patch(Rectangle((rx, ry), rw, rh, fc='white', ec='none', alpha=DIM, zorder=20))
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle='round,pad=0.2,rounding_size=1.2',
                                    fc='none', ec=EDGE, lw=2.2, zorder=21))
    save(fig, name + ('_loc' if box else ''))
    if not box:
        continue
    # zoom: the same drawing cropped to the part, so its type is magnified on a slide.
    # the surroundings stay dimmed, so a neighbour clipped by the crop reads as context.
    fig, ax = build()
    for rx, ry, rw, rh in ((0, y1, 100, 100 - y1), (0, 0, 100, y0),
                           (0, y0, x0, y1 - y0), (x1, y0, 100 - x1, y1 - y0)):
        ax.add_patch(Rectangle((rx, ry), rw, rh, fc='white', ec='none', alpha=0.93, zorder=20))
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle='round,pad=0.2,rounding_size=1.2',
                                fc='none', ec=EDGE, lw=1.4, zorder=21))
    cx0, cy0 = max(0.0, x0 - MARGIN), max(0.0, y0 - MARGIN)
    cx1, cy1 = min(100.0, x1 + MARGIN), min(100.0, y1 + MARGIN)
    ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1)
    fig.set_size_inches((cx1 - cx0) / UX, (cy1 - cy0) / UY)
    ax.set_position([0, 0, 1, 1])
    save(fig, name, tight=False)
