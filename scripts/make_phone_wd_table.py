#!/usr/bin/env python3
# ============================================================================
# make_phone_wd_table.py -- the complete phone-GPU watchdog x compaction table.
# (2026-08-08)
#
# EVERY TIMING COLUMN, because the summary form hid what the run actually cost.
# prefill and decode are separated (not folded into wall) since muKV's scoring adds
# to PREFILL while its saving is entirely in DECODE -- a wall-clock-only view nets
# those against each other and understates both. Energy is reported as total joules
# AND as mJ/token, because the two answer different questions: joules is what the
# battery pays for the whole request, mJ/token is what it pays per unit of output.
#
# mJ/token IS ONLY COMPARABLE WHEN THE STEP COUNT MATCHES. Every gen cell generates
# exactly 4096 tokens, so mJ/token is comparable across the gen block. The ppl cells
# are teacher-forced over a different number of steps per arm (4165 vs 5474), so their
# mJ/token is NOT comparable and this table prints total joules for them instead of a
# per-token figure that would invite a false comparison.
#
# PEAK TEMPS ARE IN THE TABLE because the watchdog arms are a NULL and the temperatures
# are the evidence for why: both daemons started correctly and neither ever stepped the
# clock, since a cold-start 4096-token generation never reaches their trip points
# (v5LOW battery 36.0 C / skin 39.5 C; v5HIGH 47.0 / 50.0). Printing wd_steps beside the
# peaks is what separates "the controller did not help" from "the controller never ran".
#
# ENERGY is integrated from the USB rail (see phone_cell_report.py): battery current_now
# reads 0 with USB online on this device, so a gauge-based number would read zero.
# ============================================================================
import csv, json, math, os, re, sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phone_wd_matrix"
BPC = 16 * 8 * 64 * 2 * 2.0          # Llama-3.2-1B, f16 K/V, bytes per KV cell

GEN = [("gen_vanilla",      "vanilla (full cache)", "---",  "---"),
       ("gen_mukv_rt_off",  r"$\mu$KV",             "off",  "round-trip"),
       ("gen_mukv_ip_off",  r"$\mu$KV",             "off",  "in-place"),
       ("gen_mukv_rt_low",  r"$\mu$KV",             "v5LOW",  "round-trip"),
       ("gen_mukv_ip_low",  r"$\mu$KV",             "v5LOW",  "in-place"),
       ("gen_mukv_rt_high", r"$\mu$KV",             "v5HIGH", "round-trip"),
       ("gen_mukv_ip_high", r"$\mu$KV",             "v5HIGH", "in-place")]
PPL = [("ppl_vanilla",      "vanilla (full cache)", "---",  "---"),
       ("ppl_mukv_rt_off",  r"$\mu$KV",             "off",  "round-trip"),
       ("ppl_mukv_ip_off",  r"$\mu$KV",             "off",  "in-place")]


def load(p):
    s = open(p).read()
    s = re.sub(r':\s*-?nan\b', ': NaN', s)
    s = re.sub(r':\s*-?inf\b', ': Infinity', s)
    return json.loads(s)


def energy_j(d):
    """USB-rail integral; dt capped at 5 s so a stalled sampler cannot invent joules."""
    f = os.path.join(d, "sensors.csv")
    if not os.path.exists(f):
        return None
    e, prev = 0.0, None
    for r in csv.DictReader(open(f)):
        try:
            t = float(r["monotonic_s"]); v = float(r["usb_voltage_uv"]) / 1e6
            i = abs(float(r["usb_current_ua"])) / 1e6
        except (KeyError, ValueError, TypeError):
            continue
        if prev is not None:
            dt = min(t - prev, 5.0)
            if dt > 0:
                e += v * i * dt
        prev = t
    return e or None


def peaks(d):
    f = os.path.join(d, "sensors.csv")
    if not os.path.exists(f):
        return None, None
    bat = skin = 0.0
    for r in csv.DictReader(open(f)):
        for k, v in r.items():
            kl = k.lower()
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if "battery" in kl:
                bat = max(bat, x)
            elif kl.startswith("shell") or "skin" in kl:
                skin = max(skin, x)
    return (bat / 1000 or None), (skin / 1000 or None)


def wd_steps(d):
    f = os.path.join(d, "watchdog.log")
    if not os.path.exists(f):
        return None
    return sum(1 for l in open(f, errors="replace") if re.search(r'clk=\d+', l))


def row(tag):
    d = os.path.join(BASE, tag)
    f = os.path.join(d, "meta.json")
    if not os.path.exists(f):
        return None
    j = load(f)
    ppl = j.get("perplexity")
    if not ppl and j.get("mean_nll"):
        try:
            ppl = math.exp(j["mean_nll"])
        except OverflowError:
            ppl = None
    b, s = peaks(d)
    return dict(pre=j["prefill_ms"] / 1000, dec=j["decode_ms"] / 1000, wall=j["total_ms"] / 1000,
                tps=j.get("decode_tps") or 0, steps=j.get("n_decode_steps") or 0,
                cells=j.get("retained_kv_bytes", 0) / BPC, rss=j.get("peak_rss_kb", 0) / 1024,
                e=energy_j(d), ppl=ppl, bat=b, skin=s, wd=wd_steps(d))


# ------------------------------- text ---------------------------------------
g = {t: row(t) for t, *_ in GEN}
van = g.get("gen_vanilla")
print("=== GENERATION: 12K prompt + 4096 tokens, ctx 16384, f16, cool gate before each cell ===")
h = ("%-22s %-7s %-11s %8s %8s %8s %7s %7s %7s %7s %8s %9s %6s %6s %4s" %
     ("policy", "wdog", "compaction", "prefill", "decode", "wall", "tok/s", "dec x",
      "wall x", "cells", "peakRSS", "energy", "mJ/tok", "bat C", "wd"))
print(h); print("-" * len(h))
for tag, lab, wd, cm in GEN:
    r = g.get(tag)
    if not r:
        continue
    print("%-22s %-7s %-11s %7.1fs %7.1fs %7.1fs %7.2f %6.2fx %6.2fx %7.0f %7.0fMiB %8s %6s %6s %4s"
          % (lab.replace(r"$\mu$KV", "muKV"), wd, cm, r["pre"], r["dec"], r["wall"], r["tps"],
             van["dec"] / r["dec"] if van else 0, van["wall"] / r["wall"] if van else 0,
             r["cells"], r["rss"],
             ("%.0fJ" % r["e"]) if r["e"] else "n/a",
             ("%.1f" % (r["e"] * 1000 / r["steps"])) if (r["e"] and r["steps"]) else "n/a",
             ("%.1f" % r["bat"]) if r["bat"] else "-",
             ("%d" % r["wd"]) if r["wd"] is not None else "-"))

p = {t: row(t) for t, *_ in PPL}
vp = p.get("ppl_vanilla")
print()
print("=== QUALITY: teacher-forced PPL on a slice verified DISJOINT from the prompt (0/119 windows) ===")
h2 = ("%-22s %-11s %8s %8s %8s %7s %8s %9s %9s" %
      ("policy", "compaction", "prefill", "wall", "cells", "kept", "peakRSS", "energy", "PPL"))
print(h2); print("-" * len(h2))
for tag, lab, wd, cm in PPL:
    r = p.get(tag)
    if not r:
        continue
    print("%-22s %-11s %7.1fs %7.1fs %8.0f %6.1f%% %7.0fMiB %8s %9s"
          % (lab.replace(r"$\mu$KV", "muKV"), cm, r["pre"], r["wall"], r["cells"],
             100.0 * r["cells"] / vp["cells"] if vp else 0, r["rss"],
             ("%.0fJ" % r["e"]) if r["e"] else "n/a",
             ("%.4f" % r["ppl"]) if r["ppl"] else "-"))
print("  (mJ/token omitted here: the arms are teacher-forced over different step counts,")
print("   so a per-token energy figure would not be comparable across these rows.)")

# watchdog verdict, stated from the data rather than assumed
print()
for tag, lab, wd, cm in GEN:
    r = g.get(tag)
    if r and wd not in ("off", "---") and r["wd"] == 0:
        trip = {"v5LOW": "36.0 C battery / 39.5 C skin", "v5HIGH": "47.0 C battery / 50.0 C skin"}[wd]
        print("NOTE %-7s started but never stepped the clock: peak battery %.1f C, skin %.1f C vs trip %s"
              % (wd, r["bat"] or 0, r["skin"] or 0, trip))
