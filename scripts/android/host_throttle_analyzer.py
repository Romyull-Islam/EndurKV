#!/usr/bin/env python3
"""
host_throttle_analyzer.py — detect thermal-throttle events from sensors.csv.

For each cell, finds:
  1. THROTTLE EVENTS: when DVFS frequency drops below the cell's achieved max
     for >1 sample, indicating active mitigation
  2. CAUSING ZONE: which thermal zone was above its passive trip point at that
     moment (likely cause)
  3. DURATION + DEPTH: how long throttled + how much frequency reduced
  4. RECOVERY: when frequency returned to max

Snapdragon 8 Elite Gen 5 trip points (from /sys/class/thermal/thermal_zone*/trip_point_*):
  cpullc-*: passive at 95°C (CPU LITTLE)  → big cores: ~95°C limit
  nsphmx-2: passive at 105°C (NPU shared) → DDR: 105°C limit
  Other zones: passive 112-135°C

Usage:
  python host_throttle_analyzer.py <wave3_cell_dir>
  python host_throttle_analyzer.py /home/mislam22/EndurKV_workspace/phone-logs/wave3_real_1780680903/v1_K512
"""
import sys, csv
from pathlib import Path

# Trip points discovered via /sys/class/thermal/*/trip_point_*_temp on the phone (Celsius)
# Conservative passive points only (we don't care about hot/critical for the user-experienced throttle story)
TRIP_C = {
    "cpullc":  95.0,    # big-core cluster passive
    "cpu-":    95.0,
    "ddr":    105.0,    # DDR memory passive (very high, rarely triggers in our workload)
    "shell":   55.0,    # skin temperature comfort trip
    "battery": 45.0,    # battery health trip
    "nsphmx": 105.0,
}

# Sustained "throttle" definition: frequency dropped by at least DROP_PCT
# below this cell's MAX observed freq, for at least DUR_SAMPLES consecutive samples.
DROP_PCT = 0.10        # 10% reduction qualifies as a throttle
DUR_SAMPLES = 3        # ≥3 samples (≥0.6s at 5 Hz) — filters single-sample noise

def find_col(cols, *needles):
    for c in cols:
        lc = c.lower()
        if all(n in lc for n in needles): return c
    return None

def load_sensors(path):
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        idx = {
            "t":     find_col(cols, "monotonic_s") or (cols[1] if len(cols)>1 else None),
            "ddr":   find_col(cols, "ddr_temp_mc"),
            "skin":  find_col(cols, "shell_front"),
            "cpu6f": find_col(cols, "cpu6_freq_hz"),
            "cpu7f": find_col(cols, "cpu7_freq_hz"),
            "cpu0f": find_col(cols, "cpu0_freq_hz"),
            "cpu0cs": find_col(cols, "cpu0_cool_state"),
            "cpu6cs": find_col(cols, "cpu6_cool_state"),
            "cpu7cs": find_col(cols, "cpu7_cool_state"),
            "cpu0t": find_col(cols, "cpullc-0-0_temp_mc"),
            "cpu6t": find_col(cols, "cpullc-1-0_temp_mc"),
            "cpu7t": find_col(cols, "cpullc-1-1_temp_mc"),
        }
        t0 = None
        for r in rd:
            try: t = float(r[idx["t"]])
            except Exception: continue
            if t0 is None: t0 = t
            def g(k, s=1.0):
                c = idx[k]
                if not c: return None
                try: return float(r[c]) * s
                except Exception: return None
            rows.append({
                "t":   t - t0,
                "ddr_c":   g("ddr", 1/1000.0),
                "skin_c":  g("skin", 1/1000.0),
                # NOTE: the column is named *_freq_hz but the raw value is kHz on Snapdragon.
                "cpu0f_mhz": g("cpu0f", 1/1000.0),
                "cpu6f_mhz": g("cpu6f", 1/1000.0),
                "cpu7f_mhz": g("cpu7f", 1/1000.0),
                "cpu0cs":    g("cpu0cs"),
                "cpu6cs":    g("cpu6cs"),
                "cpu7cs":    g("cpu7cs"),
                "cpu0t_c":   g("cpu0t", 1/1000.0),
                "cpu6t_c":   g("cpu6t", 1/1000.0),
                "cpu7t_c":   g("cpu7t", 1/1000.0),
            })
    return rows

def analyze(rows, cell_name=""):
    if not rows:
        return {"events":[], "summary":{"reason":"empty sensor data"}}
    # Establish per-CPU max frequency (filtering out 0/idle samples)
    max_f = {}
    for k in ("cpu0f_mhz","cpu6f_mhz","cpu7f_mhz"):
        vals = [r[k] for r in rows if r[k] is not None and r[k] > 100]  # >100 MHz: actually running
        max_f[k] = max(vals) if vals else None

    events = []
    state = {k: False for k in ("cpu0f_mhz","cpu6f_mhz","cpu7f_mhz")}  # currently throttled?
    start_t = {k: None for k in state}
    start_idx = {k: None for k in state}

    for k in ("cpu0f_mhz","cpu6f_mhz","cpu7f_mhz"):
        if max_f[k] is None: continue
        thresh = max_f[k] * (1.0 - DROP_PCT)
        # Map k to cool_state column
        cs_key = k.replace("f_mhz", "cs")
        # Find segments where freq is BELOW thresh AND > IDLE (>100 MHz) for ≥ DUR_SAMPLES samples
        # AND cool_state > 0 (actual cooling-device intervention).
        below_streak = 0
        seg_start_i = None
        for i, r in enumerate(rows):
            f = r[k]; cs = r.get(cs_key)
            if f is None: continue
            # Throttle = (freq < threshold) AND (freq > 100 = not idle) AND (cool_state > 0)
            is_throttled = (f < thresh) and (f > 100) and (cs is not None and cs > 0)
            if is_throttled:
                if below_streak == 0:
                    seg_start_i = i
                below_streak += 1
            else:
                if below_streak >= DUR_SAMPLES:
                    # close segment
                    seg_end_i = i - 1
                    sub = rows[seg_start_i:seg_end_i+1]
                    seg_t_start = rows[seg_start_i]["t"]
                    seg_t_end   = rows[seg_end_i]["t"]
                    min_f = min(x[k] for x in sub if x[k] is not None)
                    # Find the "cause" — which zone was above passive trip at seg_start?
                    cause_temp = {
                        "ddr": sub[0]["ddr_c"], "skin": sub[0]["skin_c"],
                        "cpu0": sub[0]["cpu0t_c"], "cpu6": sub[0]["cpu6t_c"], "cpu7": sub[0]["cpu7t_c"],
                    }
                    likely_cause = "unknown"
                    likely_temp  = None
                    likely_trip  = None
                    for zname, ztemp in cause_temp.items():
                        if ztemp is None: continue
                        for tk, tval in TRIP_C.items():
                            if tk in zname:
                                # margin from trip
                                m = ztemp - (tval - 10)   # within 10°C of trip = "approaching"
                                if m > 0 and (likely_temp is None or ztemp > likely_temp):
                                    likely_cause = f"{zname} (trip {tval}°C)"
                                    likely_temp = ztemp
                                    likely_trip = tval
                                break
                    events.append({
                        "core":      k.replace("f_mhz",""),
                        "t_start":   seg_t_start,
                        "t_end":     seg_t_end,
                        "dur_s":     seg_t_end - seg_t_start,
                        "min_mhz":   min_f,
                        "max_mhz":   max_f[k],
                        "depth_pct": (1 - min_f/max_f[k])*100,
                        "cause":     likely_cause,
                        "ddr_c":     sub[0]["ddr_c"],
                        "cpu_c":     sub[0]["cpu0t_c"] or sub[0]["cpu6t_c"] or sub[0]["cpu7t_c"],
                    })
                below_streak = 0
                seg_start_i = None
        # tail segment
        if below_streak >= DUR_SAMPLES:
            sub = rows[seg_start_i:]
            min_f = min(x[k] for x in sub if x[k] is not None)
            seg_t_start = rows[seg_start_i]["t"]
            seg_t_end   = rows[-1]["t"]
            events.append({
                "core":      k.replace("f_mhz",""),
                "t_start":   seg_t_start,
                "t_end":     seg_t_end,
                "dur_s":     seg_t_end - seg_t_start,
                "min_mhz":   min_f,
                "max_mhz":   max_f[k],
                "depth_pct": (1 - min_f/max_f[k])*100,
                "cause":     "tail (cell ended mid-throttle)",
                "ddr_c":     sub[0]["ddr_c"],
                "cpu_c":     sub[0]["cpu0t_c"] or sub[0]["cpu6t_c"] or sub[0]["cpu7t_c"],
            })
    return {"events": events, "max_f": max_f}

def main():
    if len(sys.argv) < 2:
        print("usage: host_throttle_analyzer.py <cell_dir>"); sys.exit(1)
    cell = Path(sys.argv[1])
    sensors = cell / "sensors.csv"
    if not sensors.exists():
        print(f"no sensors.csv in {cell}"); sys.exit(1)
    rows = load_sensors(sensors)
    name = cell.name
    res = analyze(rows, name)
    print(f"## Throttle analysis: {name}\n")
    print(f"Samples: {len(rows)}   Duration: {rows[-1]['t']:.0f} s")
    def fmt(v): return f"{v:.0f}" if v is not None else "?"
    print(f"Max freq observed: cpu0={fmt(res['max_f'].get('cpu0f_mhz'))} cpu6={fmt(res['max_f'].get('cpu6f_mhz'))} cpu7={fmt(res['max_f'].get('cpu7f_mhz'))} MHz")
    # Thermal margins from trip points
    ddrs = [r["ddr_c"] for r in rows if r["ddr_c"] is not None]
    cpus = [r["cpu0t_c"] for r in rows if r["cpu0t_c"] is not None]
    if ddrs:
        peak_ddr = max(ddrs)
        print(f"Peak DDR: {peak_ddr:.1f}°C   (margin to 105°C passive trip: {105-peak_ddr:.1f}°C)")
    if cpus:
        peak_cpu = max(cpus)
        print(f"Peak CPU LITTLE: {peak_cpu:.1f}°C   (margin to 95°C passive trip: {95-peak_cpu:.1f}°C)")
    # Cool_state evidence
    cs_max = {}
    for k in ("cpu0cs","cpu6cs","cpu7cs"):
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        cs_max[k] = max(vals) if vals else None
    cs_summary = ", ".join(f"{k}={cs_max[k]}" for k in cs_max if cs_max[k] is not None)
    print(f"Cool-state max: {cs_summary}    (any value > 0 = cooling-device actively limiting that core)")
    print()
    if not res["events"]:
        print("**NO throttle events detected.** Criteria: freq drop ≥10% below cell-max for ≥0.6s AND cool_state > 0.")
        # Explain why
        if cs_max and all(v == 0 for v in cs_max.values() if v is not None):
            print("Cool-state stayed at 0 across all cores throughout the cell — no thermal cooling device fired.")
            print(f"Closest approach to a trip point: DDR={peak_ddr:.1f}°C vs 105°C trip, CPU={peak_cpu:.1f}°C vs 95°C trip.")
        return
    print(f"### Detected {len(res['events'])} throttle event(s)")
    print()
    print("| Core | Start (s) | End (s) | Duration | Min MHz | Depth | Likely cause | DDR°C | CPU°C |")
    print("|---|---|---|---|---|---|---|---|---|")
    for e in res["events"]:
        print(f"| {e['core']} | {e['t_start']:.0f} | {e['t_end']:.0f} | {e['dur_s']:.0f}s | {e['min_mhz']:.0f} | -{e['depth_pct']:.0f}% | {e['cause']} | {e['ddr_c'] or '?':.1f} | {e['cpu_c'] or '?':.1f} |")

if __name__ == "__main__":
    main()
