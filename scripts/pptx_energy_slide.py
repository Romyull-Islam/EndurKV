#!/usr/bin/env python3
"""Add (or refresh) the energy-aware control result slide in muKV_talk.pptx, in the deck's own
style (white slide, Consolas kicker and labels, Arial headline and body, white tables, stat
cards on the right). Reads the proof cells from /tmp/ea_proof_v2 so re-running updates the
numbers. The slide is placed after 'RESULT · THERMAL' and page numbers are renumbered.
"""
import sys, os, glob, json, copy, statistics as st
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

sys.path.insert(0, "/home/mislam22/EndurKV_workspace/EndurKV/scripts")
from ea_proof_summary import load as load_cell

DECK = "/home/mislam22/EndurKV_workspace/EndurKV/muKV_talk.pptx"
KICKER = "RESULT · ENERGY-AWARE CONTROL"
INK, MUTED, BODY, RUST, BLUE, GREEN = "0C1116", "6B7785", "38424E", "C4581A", "2B6CA3", "1A7A58"

# ---- data ----
arms = {}
for d in sorted(glob.glob("/tmp/ea_proof_v2/*_r[0-9]")) + sorted(glob.glob("/tmp/split_proof/*_r[0-9]")):
    if not os.path.exists(d + "/meta.json"):
        continue
    x = load_cell(d); m = json.load(open(d + "/meta.json"))
    x["np"] = m["n_prompt_tokens"]; x["no"] = m["n_decode_steps"]; x["wall"] = m["total_ms"] / 1000
    arms.setdefault(os.path.basename(d).rsplit("_r", 1)[0], []).append(x)

def mean(v):
    v = [a for a in v if a == a]
    return st.mean(v) if v else float("nan")

def row(arm, action, tier):
    c = arms.get(arm)
    if not c:
        return [tier, action, "pending", "", "", "", "", ""]
    n = len(c)
    return [tier + (f" (n={n})" if n > 1 else ""), action,
            f"{mean([x['prefill_J']*1000/x['np'] for x in c]):.0f}", f"{mean([x['prefill_s'] for x in c]):.0f}",
            f"{mean([x['dec_mJ'] for x in c]):.0f}", f"{mean([x['tps'] for x in c]):.1f}",
            f"{mean([x['prefill_J']+x['decode_J'] for x in c]):.0f}", f"{mean([x['wall'] for x in c]):.0f}"]

HDR = ["lever picks at", "prefill / decode MHz", "prefill mJ/tok", "prefill s", "decode mJ/tok", "decode tok/s", "total J", "total s"]
GPU = [row("gpu_healthy", "1200 / 1200", "full perf."), row("gpu1200d902", "1200 / decode 902", "healthy, mains"),
       row("gpu1200d726", "1200 / decode 726", "mid"), row("gpu_mid", "902 / 902", "low"), row("gpu_low", "726 / 726", "never")]
CPU = [row("cpu_healthy", "K 1024 (chosen)", "every tier"), row("cpu_mid", "K 512", "never"), row("cpu_low", "K 256", "never")]

def pct(a, b):
    try:
        return (float(b) / float(a) - 1) * 100
    except Exception:
        return float("nan")
g_mid = pct(GPU[0][6], GPU[1][6]); g_low = pct(GPU[0][6], GPU[2][6]); c_mid = pct(CPU[0][6], CPU[1][6])
g_mid_t = pct(GPU[0][7], GPU[1][7]); g_low_t = pct(GPU[0][7], GPU[2][7])
def ext(r):
    try: return float(r[6]) * float(r[7])
    except Exception: return float("nan")
edp_mid = (ext(GPU[1]) / ext(GPU[0]) - 1) * 100; edp_low = (ext(GPU[2]) / ext(GPU[0]) - 1) * 100

# ---- deck ----
prs = Presentation(DECK)
# remove a previous version of this slide, if any
for s in list(prs.slides):
    if any(sh.has_text_frame and sh.text_frame.text.strip() == KICKER for sh in s.shapes):
        rId = [r for r in prs.slides._sldIdLst if prs.slides.part.related_part(r.rId) is s.part][0]
        prs.part.drop_rel(rId.rId); prs.slides._sldIdLst.remove(rId)
blank = [l for l in prs.slide_layouts if l.name == "Blank"][0]
s = prs.slides.add_slide(blank)

def box(x, y, w, h, fill=None):
    sh = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.line.fill.background()
    if fill: sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor.from_string(fill)
    else: sh.fill.background()
    return sh

def text(x, y, w, h, t, size, color, bold=False, font="Arial", align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.02); tf.margin_top = tf.margin_bottom = Inches(0.01)
    lines = t if isinstance(t, list) else [t]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = ln; r.font.size = Pt(size); r.font.bold = bold; r.font.name = font
        r.font.color.rgb = RGBColor.from_string(color)
    return tb

def table(x, y, w, rows, widths, hdr_size=8.5, body_size=9.5, row_h=0.27, hdr=None):
    hdr = hdr or HDR
    shp = s.shapes.add_table(len(rows) + 1, len(HDR), Inches(x), Inches(y), Inches(w), Inches(row_h * (len(rows) + 1)))
    tb = shp.table
    for j, wd in enumerate(widths): tb.columns[j].width = Inches(wd)
    for i in range(len(rows) + 1): tb.rows[i].height = Inches(row_h if i else row_h + 0.08)
    for j, h in enumerate(hdr):
        c = tb.cell(0, j); c.text = ""
        c.fill.solid(); c.fill.fore_color.rgb = RGBColor.from_string("FFFFFF")
        r = c.text_frame.paragraphs[0].add_run(); r.text = h; r.font.size = Pt(hdr_size); r.font.bold = True; r.font.name = "Consolas"
        r.font.color.rgb = RGBColor.from_string(MUTED)
        c.margin_left = c.margin_right = Inches(0.04); c.margin_top = c.margin_bottom = Inches(0.02)
    for i, rw in enumerate(rows, 1):
        for j, v in enumerate(rw):
            c = tb.cell(i, j); c.text = ""
            c.fill.solid(); c.fill.fore_color.rgb = RGBColor.from_string("FFFFFF")
            r = c.text_frame.paragraphs[0].add_run(); r.text = v; r.font.size = Pt(body_size); r.font.name = "Arial"
            r.font.bold = (j == 1); r.font.color.rgb = RGBColor.from_string(INK if j != 1 else BLUE)
            c.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT if j < 2 else PP_ALIGN.RIGHT
            c.margin_left = c.margin_right = Inches(0.04); c.margin_top = c.margin_bottom = Inches(0.02)
    return shp

box(0, 0, 13.33, 7.5, "FFFFFF")
text(0.62, 6.92, 7.0, 0.30, "RESULTS · ENERGY-AWARE · LLAMA-3.2-1B · ONEPLUS 15 · 9737-TOKEN PROMPT · N=1 UNLESS STATED", 9, MUTED, bold=True, font="Consolas")
text(0.62, 0.55, 11.0, 0.40, KICKER, 11, RUST, bold=True, font="Consolas")
text(0.62, 1.00, 11.6, 0.75, "Same command line; only the battery state changes the plan", 26, INK, bold=True)
text(0.62, 1.78, 7.6, 0.28, "GPU · clock caps by prefill / decode phase · K held at 1024 · 4096 output tokens", 10.5, BLUE, bold=True, font="Consolas")
W = [1.30, 1.40, 0.90, 0.72, 0.90, 0.62, 0.80, 0.72]
table(0.62, 2.08, sum(W), GPU, W)
text(0.62, 3.95, 7.6, 0.28, "CPU · the tier sets the cache budget K · clock untouched · 1024 output tokens", 10.5, BLUE, bold=True, font="Consolas")
table(0.62, 4.25, sum(W), CPU, W, hdr=["lever picks at", "cache budget", "prefill mJ/tok", "prefill s", "decode mJ/tok", "decode tok/s", "total J", "total s"])
text(0.62, 5.50, 7.4, 1.3, [
    "A cap on decode alone is nearly free: 6% less energy for 1.5% more time and a DDR peak 19 C lower, the best energy x time of all plans. Capping prefill too saves 19% for 19% more time; below 902 the wait grows faster than the saving.",
    "On the CPU three quarters of a request is prefill, which the cache cannot touch: the tiers land within 3%, so 1024 is the CPU balance point. The cache is the CPU lever against the full cache (62% less energy), not between tiers."], 10, BODY)

def card(y, big, cap, color):
    box(8.30, y, 4.42, 1.24, "FFFFFF"); box(8.30, y + 0.06, 0.04, 1.12, color)
    text(8.52, y + 0.10, 4.07, 0.50, big, 26, color, bold=True, font="Consolas")
    text(8.52, y + 0.64, 4.07, 0.55, cap, 10, MUTED)
card(2.00, f"{pct(GPU[0][6], GPU[1][6]):+.0f}% energy", f"GPU, decode capped at 902 MHz, prefill untouched: {pct(GPU[0][7], GPU[1][7]):+.1f}% total time, energy x time {(ext(GPU[1])/ext(GPU[0])-1)*100:+.0f}%. The lever's choice at full charge.", RUST)
card(3.45, f"{pct(GPU[0][6], GPU[3][6]):+.0f}% energy", f"GPU, 902 MHz on both phases: {pct(GPU[0][7], GPU[3][7]):+.0f}% total time, energy x time {(ext(GPU[3])/ext(GPU[0])-1)*100:+.0f}%. The lever's choice at low charge; 726 is never taken.", BLUE)
card(4.90, f"{c_mid:+.0f}% energy", f"CPU, K 512 instead of 1024: 37% of qasper F1 for {-c_mid:.0f}% of the request. The lever holds K=1024; the cache pays against the full cache, not between tiers.", GREEN)

# ---- second slide: the picture ----
table_slide = s
s = prs.slides.add_slide(blank)
box(0, 0, 13.33, 7.5, "FFFFFF")
text(0.62, 6.92, 7.0, 0.30, "RESULTS · ENERGY-AWARE · LLAMA-3.2-1B · ONEPLUS 15 · 9737-TOKEN PROMPT", 9, MUTED, bold=True, font="Consolas")
text(0.62, 0.55, 11.0, 0.40, "RESULT · ENERGY-AWARE CONTROL, THE PICTURE", 11, RUST, bold=True, font="Consolas")
text(0.62, 1.00, 11.6, 0.75, "Energy falls with the tier; the price is time, not decode speed", 26, INK, bold=True)
text(0.62, 1.50, 11.6, 0.28, "Same command line every arm, n=3. GPU row: the tier caps the clock. CPU row: the tier sets K. Rows differ in output length.", 10.5, BLUE, bold=True, font="Consolas")
pic = s.shapes.add_picture("/home/mislam22/EndurKV_workspace/EndurKV/figures/fig_controller_proof.png", Inches(3.15), Inches(1.80), width=Inches(7.0))

# ---- third slide: the scheduler, measured ----
pic_slide = s
s = prs.slides.add_slide(blank)
box(0, 0, 13.33, 7.5, "FFFFFF")
text(0.62, 6.92, 7.0, 0.30, "RESULTS · ENERGY-AWARE · SCHEDULER PROOF · N=2 PER ARM", 9, MUTED, bold=True, font="Consolas")
text(0.62, 0.55, 11.0, 0.40, "RESULT · THE SCHEDULER, MEASURED", 11, RUST, bold=True, font="Consolas")
text(0.62, 1.00, 11.6, 0.75, "The scheduler chose every plan; the phone measured what it cost", 26, INK, bold=True)
text(0.62, 1.78, 11.6, 0.28, "Same prompt through ukv_sched.sh under forced battery states; control = told mains. Predictions from its table land within 1 to 7% of the meter.", 10.5, BLUE, bold=True, font="Consolas")
s.shapes.add_picture("/home/mislam22/EndurKV_workspace/EndurKV/figures/fig_sched_proof.png", Inches(2.55), Inches(2.10), width=Inches(8.2))
text(0.62, 2.10 + 8.2 * 3.6 / 7.2 + 0.05, 12.1, 0.5, "At 40% and 15% the request costs 57% and 62% less than the control, mostly through the output cap; with the output fixed by the caller (15%, 4096 fix) the clock cap alone saves 13% at equal tokens. At 80% and on mains the scheduler runs the full-performance plan.", 10, BODY)

# ---- fourth slide: the decision map ----
s3 = s
s = prs.slides.add_slide(blank)
box(0, 0, 13.33, 7.5, "FFFFFF")
text(0.62, 6.92, 7.0, 0.30, "RESULTS · ENERGY-AWARE · DECISION MAP", 9, MUTED, bold=True, font="Consolas")
text(0.62, 0.55, 11.0, 0.40, "RESULT · WHAT IT DECIDES, BY BATTERY STATE", 11, RUST, bold=True, font="Consolas")
text(0.62, 1.00, 11.6, 0.75, "When the GPU clock, when the cache, when the output cap", 26, INK, bold=True)
text(0.62, 1.78, 11.6, 0.28, "The policy at every state of charge, five request situations. CPU only when the GPU is unavailable; the CPU clock is never a decision.", 10.5, BLUE, bold=True, font="Consolas")
s.shapes.add_picture("/home/mislam22/EndurKV_workspace/EndurKV/figures/fig_decision_map.png", Inches(2.4), Inches(2.1), width=Inches(8.5))

# move the new slides right after 'RESULT · THERMAL', then renumber page-number boxes
ids = prs.slides._sldIdLst; new4 = ids[-1]; new3 = ids[-2]; new2 = ids[-3]; new = ids[-4]
titles = []
for sl in prs.slides:
    k = [sh.text_frame.text.strip() for sh in sl.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
    titles.append(k)
pos = next(i for i, k in enumerate(titles) if any(t == "RESULT · THERMAL" for t in k))
for it in (new, new2, new3, new4): ids.remove(it)
for k, it in enumerate((new, new2, new3, new4)): ids.insert(pos + 1 + k, it)
for i, sl in enumerate(prs.slides, 1):
    for sh in sl.shapes:
        if sh.has_text_frame and abs(sh.left / 914400 - 12.18) < 0.05 and abs(sh.top / 914400 - 6.92) < 0.05:
            r = sh.text_frame.paragraphs[0].runs
            if r: r[0].text = str(i)
            else: sh.text_frame.paragraphs[0].add_run().text = str(i)
for k, sl in ((pos + 2, table_slide), (pos + 3, pic_slide), (pos + 4, s3), (pos + 5, s)):
    if not any(abs(sh.left / 914400 - 12.18) < 0.05 and abs(sh.top / 914400 - 6.92) < 0.05 for sh in sl.shapes):
        tb = sl.shapes.add_textbox(Inches(12.18), Inches(6.92), Inches(0.60), Inches(0.30))
        r = tb.text_frame.paragraphs[0].add_run(); r.text = str(k); r.font.size = Pt(9); r.font.name = "Consolas"; r.font.color.rgb = RGBColor.from_string(MUTED)
        tb.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
prs.save(DECK)
print(f"saved {DECK}: {len(prs.slides)} slides; energy slide at position {pos + 2}")
print("GPU rows:", GPU); print("CPU rows:", CPU)
