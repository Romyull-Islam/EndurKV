#!/usr/bin/env python3
# Per-clock-cap report: throughput, PEAK temperatures, mean power, energy per token.
# Energy = USB rail + battery pack; rail alone undercounts because the pack supplements it.
import csv, json, os, re, sys
d, clk = sys.argv[1], sys.argv[2]
j = json.loads(re.sub(r':\s*-?inf\b', ': Infinity', re.sub(r':\s*-?nan\b', ': NaN', open(os.path.join(d, "meta.json")).read())))
usb = bat = 0.0; pk = {}
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
    for key, cols in (("skin", ("shell_front_temp_mc", "shell_frame_temp_mc", "shell_back_temp_mc")),
                      ("ddr", ("ddr_temp_mc",)), ("batt", ("battery_temp_mc",))):
        best = 0.0
        for r in rows:
            for c in cols:
                try: best = max(best, float(r[c]) / 1000.0)
                except Exception: pass
        pk[key] = best
    q = [float(r['bat_charge_uah']) for r in rows if r.get('bat_charge_uah', '').strip().lstrip('-').isdigit()]
    vv = [float(r['bat_voltage_now_uv']) / 1e6 for r in rows if r.get('bat_voltage_now_uv', '').strip().isdigit()]
    if len(q) > 1:
        bat = max(0.0, (q[0] - q[-1]) / 1e6) * (sum(vv) / len(vv) if vv else 4.35) * 3600
tot = usb + bat; wall = j["total_ms"] / 1000
import math as _m
ppl = j.get("perplexity")
if ppl is None or ppl != ppl:
    n = j.get("mean_nll")
    ppl = _m.exp(n) if (n is not None and n == n) else None
g = os.path.join(d, "gen.txt")
bang = None
if os.path.exists(g):
    _t = open(g, errors="replace").read()
    bang = 100 * _t.count('!') / max(1, len(_t))
print("  [%-13s] K=%-5s pre=%6.1fs dec=%6.1fs wall=%6.1fs tps=%6.2f cells=%5.0f RSS=%4.0fM | "
      "skin=%.1fC ddr=%.1fC batt=%.1fC | E=%5.0fJ W=%.2f mJ/tok=%6.1f | PPL=%s bangs=%s"
      % (clk, j.get("k_nominal"), j["prefill_ms"]/1000, j["decode_ms"]/1000, wall,
         j.get("decode_tps") or 0, j["retained_kv_bytes"]/(16*8*64*2*2.0),
         j.get("peak_rss_kb", 0)/1024,
         pk.get("skin", 0), pk.get("ddr", 0), pk.get("batt", 0),
         tot, tot/max(wall, 1e-9), tot*1000/max(j.get("n_decode_steps") or 1, 1),
         ("%.3f" % ppl) if ppl else "-", ("%.2f%%" % bang) if bang is not None else "n/a"))
