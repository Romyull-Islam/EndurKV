#!/usr/bin/env python3
# ============================================================================
# make_gpu_table_full.py -- phone-GPU WikiText table with EVERY column. (2026-08-10)
#
# The compact version of this table showed only prefill / tok-s / cells, which hides the
# things a reader needs to judge the result: that muKV's prefill is slightly WORSE (the
# scoring cost), that its energy win comes from finishing sooner rather than drawing less
# power, and that peak RSS goes UP even though the cache goes down. All of it is here.
#
# TWO PERPLEXITIES, AND THEY MEASURE DIFFERENT THINGS.
#   PPL_dis  -- eval slice VERIFIED disjoint from the prompt (0/119 200-char and 0/399
#               60-char windows against the device's own prompt_12k.txt). This is
#               prediction, and it is the number that says whether eviction hurt the model.
#   PPL_rec  -- eval slice that is a PREFIX OF THE PROMPT (100% overlap, by design). This
#               scores verbatim recall of retained text, so an evictor is SUPPOSED to lose
#               here. Reporting it alone would be misleading; reporting it alongside
#               PPL_dis is what separates "forgot the text" from "got worse at language".
# Both are labelled. A 64K campaign was invalidated by exactly this distinction going
# unnoticed, so the disjointness is re-asserted here rather than assumed.
#
# ENERGY is integrated from the USB rail in each cell's sensors.csv:
#   E = SUM(V * I * dt), dt capped at 5 s so a stalled sampler cannot invent joules.
# mJ/token divides by n_decode_steps, which is 4096 for every timed cell, so it is
# comparable across rows.
#
# f16 K/V for every policy: quantized KV is numerically broken on this Adreno build, and
# a per-head evictor needs FA-off, which llama.cpp forbids with a quantized V.
# ============================================================================
import csv, json, os, re, sys, math, statistics as st

BASE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phone_gpu_16k"
GEOM = {"llama1b": (16, 8, 64), "phi3": (32, 32, 96)}
MODELS = [("llama1b", "Llama-3.2-1B"), ("phi3", "Phi-3-mini")]
POL = [("vanilla", "vanilla (full cache)"), ("mukv_dfg", "muKV + round-trip"),
       ("mukv_ip", "muKV + in-place +wd"),
       ("mukv_nodfg", "muKV, no compaction"), ("snapkv", "SnapKV"), ("adakv", "Ada-KV"),
       ("tova", "TOVA"), ("h2o", "H2O"), ("streamingllm", "StreamingLLM")]


def load(p):
    if not os.path.exists(p):
        return None
    s = open(p).read()
    s = re.sub(r':\s*-?nan\b', ': NaN', s)
    s = re.sub(r':\s*-?inf\b', ': Infinity', s)
    try:
        return json.loads(s)
    except Exception:
        return None


def fmt_of(d):
    e = os.path.join(d, "err")
    if not os.path.exists(e):
        return None
    for l in open(e, errors="replace"):
        if "KV cache types" in l:
            m = re.search(r'K=(\S+) V=(\S+)', l)
            if m:
                return m.group(1) + "/" + m.group(2)
    return None


def energy(d):
    f = os.path.join(d, "sensors.csv")
    if not os.path.exists(f):
        return None
    e, prev = 0.0, None
    try:
        rows = csv.DictReader(open(f, "rb").read().decode("utf-8", "replace").splitlines())
    except Exception:
        return None
    for r in rows:
        try:
            t = float(r["monotonic_s"])
            v = float(r["usb_voltage_uv"]) / 1e6
            i = abs(float(r["usb_current_ua"])) / 1e6
        except (KeyError, TypeError, ValueError):
            continue
        if prev is not None:
            dt = min(t - prev, 5.0)
            if dt > 0:
                e += v * i * dt
        prev = t
    return e or None


def ppl_of(mt, suf, kind):
    """kind: 'dis' (prediction, verified-disjoint slice) or 'rec' (recall, prompt prefix)."""
    j = load(os.path.join(BASE, "%s_%s_ppl_%s" % (mt, suf, kind), "meta.json"))
    if not j:
        return None
    p = j.get("perplexity")
    if not p and j.get("mean_nll"):
        try:
            p = math.exp(j["mean_nll"])
        except OverflowError:
            return None
    return p


def timed(mt, suf):
    """timed cells only: f16 and a full 4096-token generation."""
    out = []
    for tag in [suf, suf + "_r2", suf + "_r3"]:
        d = os.path.join(BASE, "%s_%s" % (mt, tag))
        j = load(os.path.join(d, "meta.json"))
        if not j or fmt_of(d) != "f16/f16":
            continue
        if (j.get("n_decode_steps") or 0) > 1000:
            out.append((j, d))
    return out


hdr = ("%-22s %2s %8s %8s %8s %8s %8s %7s %7s %7s %8s %8s %8s %8s" %
       ("policy", "n", "PPL_dis", "PPL_rec", "prefill", "decode", "wall",
        "tok/s", "dec x", "cells", "kept%", "evicted", "peakRSS", "mJ/tok"))
for mt, disp in MODELS:
    v = timed(mt, "vanilla")
    if not v:
        continue
    L_, H, D = GEOM[mt]
    bpc = L_ * H * D * 2 * 2.0
    vdec = st.median([j["decode_ms"] for j, _ in v])
    vcells = v[0][0]["retained_kv_bytes"] / bpc
    print("\n=== %s — phone GPU (Adreno 840, Vulkan), ctx 16384, f16 K/V, "
          "9737-token prompt + 4096 generated ===" % disp)
    print(hdr)
    print("-" * len(hdr))
    for suf, lab in POL:
        g = timed(mt, suf)
        if not g:
            note = ("round-trip needs a 2nd 6144 MiB cache — killed"
                    if (mt, suf) == ("phi3", "mukv_dfg") else "not measured")
            print("%-22s %2s %s" % (lab, "-", note))
            continue
        es = [energy(d) for _, d in g]
        es = [e for e in es if e]
        steps = st.median([j.get("n_decode_steps") or 4096 for j, _ in g])
        cells = g[0][0]["retained_kv_bytes"] / bpc
        pd_, pr_ = ppl_of(mt, suf, "dis"), ppl_of(mt, suf, "rec")
        # FIXED 2026-08-10: columns were each taking their OWN median, so a row could
        # report decode=509 s next to 9.59 tok/s (4096/509 = 8.05). With n=2 and a 2.3x
        # spread the two statistics came from DIFFERENT cells. Pick the MEDIAN CELL by
        # decode time and report that cell's numbers, so every column describes one run.
        g = sorted(g, key=lambda x: x[0]["decode_ms"])
        med = g[len(g) // 2][0]
        tpss = [j.get("decode_tps") or 0 for j, _ in g]
        spread = (max(tpss) - min(tpss)) / max(min(tpss), 1e-9) if len(g) > 1 else 0.0
        flag = "  <-- UNSTABLE %.0f%% spread" % (100 * spread) if spread > 0.25 else ""
        print("%-22s %2d %8s %8s %7.0fs %7.0fs %7.0fs %7.2f %6.2fx %7.0f %6.1f%% %8d %7.0fM %8.1f%s" % (
            lab, len(g),
            ("%.3f" % pd_) if pd_ else "-", ("%.3f" % pr_) if pr_ else "-",
            med["prefill_ms"] / 1000, med["decode_ms"] / 1000, med["total_ms"] / 1000,
            med.get("decode_tps") or 0, vdec / med["decode_ms"],
            cells, 100.0 * cells / vcells,
            g[0][0].get("evicted_prefill", 0),
            max(j.get("peak_rss_kb", 0) for j, _ in g) / 1024,
            (st.median(es) * 1000 / steps) if es else 0, flag))
print()
print("PPL_dis = prediction, eval slice verified disjoint from the prompt (0/119 windows).")
print("PPL_rec = verbatim recall, eval slice IS a prompt prefix (100% overlap by design);")
print("          an evictor is expected to lose here — it is the cost, not a quality defect.")
