#!/usr/bin/env python3
"""Summarize the phone-resident CPU StreamingLLM cells (cpu_sllm_published.sh).

Energy = USB rail + battery pack, exactly as scripts/energy_cell_report.py computes it: the rail is
sum(|V * I| * dt) over the sensor samples (dt capped at 5 s), and the pack is the drop in
bat_charge_uah times the mean bat_voltage_now_uv. Rail-only undercounts whenever the pack
supplements the cable, which it does with charging disabled. DDR peak from ddr_temp_mc.
Prints per-model decode and energy ratios of StreamingLLM against the same-build full cache.
"""
import csv, json, os, re, sys

RES = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qres_cpu"

def load(p):
    s = open(p).read()
    s = re.sub(r":\s*-?nan\b", ": NaN", s); s = re.sub(r":\s*-?inf\b", ": Infinity", s)
    return json.loads(s)

def energy(tag):
    p = os.path.join(RES, f"{tag}.sensors.csv")
    rows = list(csv.DictReader(open(p)))
    e, prev, ddr = 0.0, None, 0.0
    for r in rows:
        try:
            t = float(r["monotonic_s"]); v = float(r["usb_voltage_uv"]) / 1e6; i = abs(float(r["usb_current_ua"])) / 1e6
        except (ValueError, KeyError, TypeError):
            continue
        if prev is not None:
            e += v * i * min(t - prev, 5.0)
        prev = t
        try: ddr = max(ddr, float(r["ddr_temp_mc"]) / 1000)
        except (ValueError, KeyError, TypeError): pass
    q = [float(r["bat_charge_uah"]) for r in rows if r.get("bat_charge_uah", "").strip().lstrip("-").isdigit()]
    vv = [float(r["bat_voltage_now_uv"]) / 1e6 for r in rows if r.get("bat_voltage_now_uv", "").strip().isdigit()]
    bat = max(0.0, (q[0] - q[-1]) / 1e6) * (sum(vv) / len(vv) if vv else 4.35) * 3600 if len(q) > 1 else 0.0
    print(f"    [{tag}] usb {e:7.0f} J + pack {bat:6.0f} J")
    return e + bat, ddr, len(rows)

out = {}
for model in ("llama", "bonsai"):
    cells = {}
    for arm, tag in (("full", f"{model}_vanilla_cur"), ("sllm", f"{model}_sllm_pub")):
        j = os.path.join(RES, f"{tag}.json")
        if not os.path.exists(j):
            continue
        m = load(j); e, ddr, n = energy(tag)
        err = open(os.path.join(RES, f"{tag}.err"), errors="ignore").read()
        kv = re.search(r"retained_kv=([0-9.]+) MiB", err)
        cells[arm] = dict(tps=m["decode_tps"], prefill_s=m["prefill_ms"] / 1000, total_s=m["total_ms"] / 1000,
                          J=e, ddr=ddr, samples=n, retained_mib=float(kv.group(1)) if kv else None)
        print(f"{tag:22s} tok/s {m['decode_tps']:6.3f}  total {m['total_ms']/1000:7.0f} s  E {e/1000:6.2f} kJ  DDR peak {ddr:4.1f} C  retained {cells[arm]['retained_mib']} MiB")
    if "full" in cells and "sllm" in cells:
        f, s = cells["full"], cells["sllm"]
        out[model] = dict(dec=s["tps"] / f["tps"], E=s["J"] / f["J"], full_tps=f["tps"], full_kJ=f["J"] / 1000)
        print(f"  {model}: StreamingLLM dec x {out[model]['dec']:.2f}, energy x {out[model]['E']:.2f} (same build)")
json.dump(out, open(os.path.join(RES, "cpu_pub_ratios.json"), "w"), indent=1)
