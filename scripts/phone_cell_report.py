#!/usr/bin/env python3
# ============================================================================
# phone_cell_report.py -- one-line summary of a phone cell, including ENERGY.
# (2026-08-08)
#
# ENERGY IS INTEGRATED FROM THE USB RAIL, not from a battery gauge. With USB online the
# battery's current_now reads 0 on this device, so a gauge-based estimate reads zero
# while the SoC is clearly drawing power. sample_sensors.sh logs usb_voltage_uv and
# usb_current_ua, and this integrates
#     E = SUM( V * I * dt )   over the samples spanning the run
# with dt taken from monotonic_s. Each dt is CAPPED AT 5 s: if the sampler stalls (adb
# hiccup, scheduler starvation) the gap would otherwise be multiplied by the instantaneous
# power and invent joules that were never drawn.
#
# |current| because the sign convention flips with charge direction; charging is disabled
# during the cell, but the cool-down before it is not, and a stray sample of the opposite
# sign would otherwise subtract energy.
# ============================================================================
import csv, json, math, os, re, sys

d = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(d)


def load(p):
    s = open(p).read()
    s = re.sub(r':\s*-?nan\b', ': NaN', s)
    s = re.sub(r':\s*-?inf\b', ': Infinity', s)
    return json.loads(s)


j = load(os.path.join(d, "meta.json"))

energy_j = None
sens = os.path.join(d, "sensors.csv")
if os.path.exists(sens):
    try:
        rows = list(csv.DictReader(open(sens)))
        e, prev = 0.0, None
        for r in rows:
            try:
                t = float(r["monotonic_s"])
                v = float(r["usb_voltage_uv"]) / 1e6
                i = abs(float(r["usb_current_ua"])) / 1e6
            except (KeyError, ValueError, TypeError):
                continue
            if prev is not None:
                dt = min(t - prev, 5.0)          # see header: cap stalled gaps
                if dt > 0:
                    e += v * i * dt
            prev = t
        energy_j = e if e > 0 else None
    except Exception:
        energy_j = None

steps = j.get("n_decode_steps") or 0
mj_tok = (energy_j * 1000.0 / steps) if (energy_j and steps) else None
cells = j.get("retained_kv_bytes", 0) / (16 * 8 * 64 * 2 * 2.0)   # f16

ppl = j.get("perplexity")
if not ppl and j.get("mean_nll"):
    try:
        ppl = math.exp(j["mean_nll"])
    except OverflowError:
        ppl = None

# how many times the watchdog actually stepped the clock down, and how low it went
wd_steps, wd_min = 0, None
wlog = os.path.join(d, "watchdog.log")
if os.path.exists(wlog):
    for line in open(wlog, errors="replace"):
        m = re.search(r'clk=(\d+)', line)
        if m:
            wd_steps += 1
            c = int(m.group(1))
            wd_min = c if wd_min is None else min(wd_min, c)

print("  [%-18s] prefill=%6.1fs decode=%6.1fs wall=%6.1fs tps=%6.2f cells=%5.0f "
      "peakRSS=%5.0fMiB E=%s mJ/tok=%s PPL=%s compact=%s wd_steps=%d wd_min=%s"
      % (label, j["prefill_ms"] / 1000, j["decode_ms"] / 1000, j["total_ms"] / 1000,
         j.get("decode_tps") or 0, cells, j.get("peak_rss_kb", 0) / 1024,
         ("%.1fJ" % energy_j) if energy_j else "n/a",
         ("%.2f" % mj_tok) if mj_tok else "n/a",
         ("%.4f" % ppl) if ppl else "-",
         j.get("compaction_mode", j.get("compaction_applied")),
         wd_steps, ("%dMHz" % wd_min) if wd_min else "-"))
