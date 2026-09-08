#!/usr/bin/env python3
# Energy = USB rail + battery pack. Rail-only undercounts by 4-36% because the pack
# supplements the rail whenever SoC draw exceeds what USB delivers, which is invisible
# unless charge_counter is read. dt capped at 5 s so a stalled sampler cannot invent joules.
import csv, json, math, os, re, sys
d, label = sys.argv[1], sys.argv[2]
j = json.loads(re.sub(r':\s*-?inf\b', ': Infinity', re.sub(r':\s*-?nan\b', ': NaN', open(os.path.join(d, "meta.json")).read())))
usb = bat = 0.0
f = os.path.join(d, "sensors.csv")
if os.path.exists(f):
    rows = list(csv.DictReader(open(f, 'rb').read().decode('utf-8', 'replace').splitlines()))
    prev = None
    for r in rows:
        try:
            t = float(r["monotonic_s"]); v = float(r["usb_voltage_uv"]) / 1e6; i = abs(float(r["usb_current_ua"])) / 1e6
        except Exception:
            continue
        if prev is not None:
            dt = min(t - prev, 5.0)
            if dt > 0: usb += v * i * dt
        prev = t
    q = [float(r['bat_charge_uah']) for r in rows if r.get('bat_charge_uah', '').strip().lstrip('-').isdigit()]
    vv = [float(r['bat_voltage_now_uv']) / 1e6 for r in rows if r.get('bat_voltage_now_uv', '').strip().isdigit()]
    if len(q) > 1:
        bat = max(0.0, (q[0] - q[-1]) / 1e6) * (sum(vv) / len(vv) if vv else 4.35) * 3600
tot = usb + bat
steps = j.get("n_decode_steps") or 1
ppl = j.get("perplexity") or (math.exp(j["mean_nll"]) if j.get("mean_nll") and j["mean_nll"] == j["mean_nll"] else None)
g = os.path.join(d, "gen.txt")
bang = (100 * open(g, errors="replace").read().count('!') / max(1, len(open(g, errors="replace").read()))) if os.path.exists(g) else None
print("  [%-14s] K=%-5s pre=%6.1fs dec=%6.1fs wall=%7.1fs tps=%6.2f cells=%5.0f RSS=%4.0fM "
      "E=%6.0fJ (usb %.0f + bat %.0f) W=%.2f mJ/tok=%6.1f PPL=%s bangs=%s"
      % (label, j.get("k_nominal"), j["prefill_ms"]/1000, j["decode_ms"]/1000, j["total_ms"]/1000,
         j.get("decode_tps") or 0, j["retained_kv_bytes"]/(16*8*64*2*2.0),
         j.get("peak_rss_kb", 0)/1024, tot, usb, bat,
         tot/max(j["total_ms"]/1000, 1e-9), tot*1000/steps,
         ("%.3f" % ppl) if ppl else "-", ("%.2f%%" % bang) if bang is not None else "n/a"))
