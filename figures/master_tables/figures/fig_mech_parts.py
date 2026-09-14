#!/usr/bin/env python3
"""Break fig_mechanism_tikz into talk-sized stages.

Compiles the same TikZ source once per stage with a white veil over everything
outside that stage and an outline around it, so each part is the published
figure with one thing lit. Coordinates are the figure's own TikZ cm.
"""
import os, subprocess, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'fig_mechanism_tikz.tex')).read()
CANVAS = (-1.2, -1.3, 9.4, 12.3)          # x0, y0, x1, y1, comfortably outside every node

# name, focus box in TikZ cm
STAGES = [
    ('m1_layer',     (-0.35, 10.20, 8.40, 11.95)),
    ('m2_sidenode',  (-0.35,  8.72, 8.40, 10.15)),
    ('m3_scores',    ( 1.55,  7.50, 6.95,  8.52)),
    ('m4_reduce',    (-0.15,  3.76, 4.25,  7.38)),
    ('m5_keepset',   (-0.15, -0.75, 4.25,  3.74)),
    ('m6_perhead',   ( 4.40, -0.75, 8.72,  5.85)),
]
OP = 0.90

def overlay(fx):
    x0, y0, x1, y1 = fx; cx0, cy0, cx1, cy1 = CANVAS
    bands = [(cx0, y1, cx1, cy1), (cx0, cy0, cx1, y0), (cx0, y0, x0, y1), (x1, y0, cx1, y1)]
    out = ['\\begin{scope}[opacity=%.2f]' % OP]
    out += ['  \\fill[white] (%.2f,%.2f) rectangle (%.2f,%.2f);' % b for b in bands]
    out += ['\\end{scope}',
            '\\draw[or,line width=1.1pt,rounded corners=3pt] (%.2f,%.2f) rectangle (%.2f,%.2f);' % (x0, y0, x1, y1)]
    return '\n'.join(out) + '\n'

os.makedirs(os.path.join(HERE, 'parts'), exist_ok=True)
MARGIN = 0.28

def build(name, fx, clip):
    tex = SRC.replace('\\end{tikzpicture}', overlay(fx) + '\\end{tikzpicture}', 1)
    if clip:
        x0, y0, x1, y1 = fx
        tex = tex.replace('\\begin{tikzpicture}[x=1cm,y=1cm]',
                          '\\begin{tikzpicture}[x=1cm,y=1cm]\n\\clip (%.2f,%.2f) rectangle (%.2f,%.2f);'
                          % (x0 - MARGIN, y0 - MARGIN, x1 + MARGIN, y1 + MARGIN), 1)
    return tex

JOBS = [(n + '_loc', fx, False) for n, fx in STAGES] + [(n, fx, True) for n, fx in STAGES]
for name, fx, clip in JOBS:
    tex = build(name, fx, clip)
    with tempfile.TemporaryDirectory() as td:
        for dep in ('heat.pdf', 'mech_grids.tex'):
            shutil.copy(os.path.join(HERE, dep), td)
        tp = os.path.join(td, name + '.tex')
        open(tp, 'w').write(tex)
        r = subprocess.run(['tectonic', '-X', 'compile', tp, '--outdir', td],
                           capture_output=True, text=True)
        pdf = os.path.join(td, name + '.pdf')
        if not os.path.exists(pdf):
            print('FAILED', name, r.stderr[-400:]); continue
        subprocess.run(['pdftoppm', '-r', '300', '-png', '-singlefile', pdf,
                        os.path.join(HERE, 'parts', name)], check=True)
        print('wrote parts/%s.png' % name)
