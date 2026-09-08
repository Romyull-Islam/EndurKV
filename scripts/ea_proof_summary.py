#!/usr/bin/env python3
"""Summarise an energy-aware proof campaign (run_energy_aware_proof_v2.sh).

Energy method is the same as clock_cell_report.py: USB rail integral plus battery
pack coulomb delta times mean pack voltage. Energy per token divides whole-run
energy by decode steps. Arms are grouped across replicates (tag_rN) and reported
as mean and standard deviation. Prints the healthy-to-low delta per backend,
which is the claim the campaign exists to test.
"""
import csv, json, os, re, sys, glob, statistics as st


def cell_energy_J(d):
    f = os.path.join(d, "sensors.csv")
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f, "rb").read().decode("utf-8", "replace").splitlines()))
    usb = 0.0
    prev = None
    for r in rows:
        try:
            t = float(r["monotonic_s"]); v = float(r["usb_voltage_uv"]) / 1e6; i = abs(float(r["usb_current_ua"])) / 1e6
        except Exception:
            continue
        if prev is not None:
            dt = min(t - prev, 5.0)
            if dt > 0:
                usb += v * i * dt
        prev = t
    q = [float(r["bat_charge_uah"]) for r in rows if r.get("bat_charge_uah", "").strip().lstrip("-").isdigit()]
    vv = [float(r["bat_voltage_now_uv"]) / 1e6 for r in rows if r.get("bat_voltage_now_uv", "").strip().isdigit()]
    bat = 0.0
    if len(q) > 1:
        bat = max(0.0, (q[0] - q[-1]) / 1e6) * (sum(vv) / len(vv) if vv else 4.35) * 3600
    return usb + bat


def peak(d, cols):
    f = os.path.join(d, "sensors.csv")
    if not os.path.exists(f):
        return float("nan")
    best = 0.0
    for r in csv.DictReader(open(f, "rb").read().decode("utf-8", "replace").splitlines()):
        for c in cols:
            try:
                best = max(best, float(r[c]) / 1000.0)
            except Exception:
                pass
    return best


def dvfs(d):
    """Mean GPU clock (MHz), share of samples under a thermal cap, mean DDR clock (MHz),
    mean prime-core clock (MHz). Needs sample_sensors.sh v6 columns; NaN otherwise."""
    f = os.path.join(d, "sensors.csv")
    out = dict(gpu_mhz=float("nan"), tpl_pct=float("nan"), ddr_mhz=float("nan"), cpu6_mhz=float("nan"))
    if not os.path.exists(f):
        return out
    rows = csv.DictReader(open(f, "rb").read().decode("utf-8", "replace").splitlines())
    g, t, dd, c6 = [], [], [], []
    for r in rows:
        for key, lst, div in (("gpu_clk_hz", g, 1e6), ("ddr_freq_khz", dd, 1e3), ("cpu6_freq_hz", c6, 1e3)):
            try:
                v = float(r.get(key) or "")
                if v > 0:
                    lst.append(v / div)
            except ValueError:
                pass
        try:
            t.append(1.0 if float(r.get("gpu_thermal_pwrlevel") or "") > 0 else 0.0)
        except ValueError:
            pass
    if g: out["gpu_mhz"] = st.mean(g)
    if t: out["tpl_pct"] = 100.0 * st.mean(t)
    if dd: out["ddr_mhz"] = st.mean(dd)
    if c6: out["cpu6_mhz"] = st.mean(c6)
    return out


def phase_split(d, m):
    """Energy of the prefill and decode phases (rail + pack). The run window is detected from
    the rail power trace (first sustained rise above idle) and split at prefill_ms. Returns
    (prefill_J, decode_J) or (nan, nan)."""
    f = os.path.join(d, "sensors.csv")
    nan = float("nan")
    if not os.path.exists(f):
        return nan, nan
    rows = list(csv.DictReader(open(f, "rb").read().decode("utf-8", "replace").splitlines()))
    T, P, Q, V = [], [], [], []
    for r in rows:
        try:
            t = float(r["monotonic_s"]); v = float(r["usb_voltage_uv"]) / 1e6; i = abs(float(r["usb_current_ua"])) / 1e6
        except Exception:
            continue
        T.append(t); P.append(v * i)
        q = r.get("bat_charge_uah", "").strip(); Q.append(float(q) if q.lstrip("-").isdigit() else None)
        vb = r.get("bat_voltage_now_uv", "").strip(); V.append(float(vb) / 1e6 if vb.isdigit() else None)
    if len(P) < 10:
        return nan, nan
    # Anchor on the END of the trace: the campaign scripts kill the sampler the moment the
    # bench returns, so the run ends within about a second of the last sample. Start
    # detection from a power rise is unreliable when sampling begins after the run starts.
    tend = T[-1]; t0 = tend - m["total_ms"] / 1000; tpf = t0 + m["prefill_ms"] / 1000
    if t0 < T[0] - 5.0:      # the trace does not cover the run; do not guess
        return nan, nan

    def window(ta, tb):
        e = 0.0
        for k in range(1, len(T)):
            if T[k] <= ta or T[k - 1] >= tb:
                continue
            dt = min(T[k] - T[k - 1], 5.0)
            if dt > 0:
                e += P[k] * dt
        qa = next((Q[k] for k in range(len(T)) if T[k] >= ta and Q[k] is not None), None)
        qb = next((Q[k] for k in range(len(T) - 1, -1, -1) if T[k] <= tb and Q[k] is not None), None)
        vv = [V[k] for k in range(len(T)) if ta <= T[k] <= tb and V[k] is not None]
        eb = 0.0
        if qa is not None and qb is not None:
            eb = max(0.0, (qa - qb) / 1e6) * (sum(vv) / len(vv) if vv else 4.35) * 3600
        return e + eb
    return window(t0, tpf), window(tpf, tend)


def load(d):
    m = json.loads(re.sub(r":\s*-?inf\b", ": Infinity", re.sub(r":\s*-?nan\b", ": NaN", open(os.path.join(d, "meta.json")).read())))
    E = cell_energy_J(d)
    wall = m["total_ms"] / 1000
    steps = max(m.get("n_decode_steps") or 1, 1)
    r = dict(K=m.get("k_nominal"), level=m.get("ea_level"), tps=m.get("decode_tps") or 0.0,
             W=(E / max(wall, 1e-9)) if E is not None else float("nan"),
             mJ=(E * 1000 / steps) if E is not None else float("nan"),
             ddr=peak(d, ("ddr_temp_mc",)), batt=peak(d, ("battery_temp_mc",)),
             prefill_s=m["prefill_ms"] / 1000, decode_s=m["decode_ms"] / 1000)
    pj, dj = phase_split(d, m)
    r["prefill_J"] = pj; r["decode_J"] = dj
    r["dec_mJ"] = (dj * 1000 / steps) if dj == dj else float("nan")
    all_tok = (m.get("n_prompt_tokens") or 0) + steps
    r["all_tps"] = all_tok / max(wall, 1e-9)                       # prompt + output tokens per second of wall time
    r["mJ_all"] = (E * 1000 / all_tok) if (E is not None and all_tok) else float("nan")   # whole-run energy per processed token
    r.update(dvfs(d))
    st_file = os.path.join(d, "start_temps.txt")
    r["start"] = open(st_file).read().strip() if os.path.exists(st_file) else ""
    return r


def main(host):
    arms = {}
    for d in sorted(glob.glob(os.path.join(host, "*"))):
        b = os.path.basename(d)
        mm = re.match(r"^(gpu|cpu)_(healthy|mid|low)_r(\d+)$", b)
        if not mm or not os.path.exists(os.path.join(d, "meta.json")):
            continue
        arms.setdefault(f"{mm.group(1)}_{mm.group(2)}", []).append(load(d))

    def ms(v):
        v = [x for x in v if x == x]
        if not v:
            return "     -"
        return f"{st.mean(v):6.1f}" + (f"±{st.pstdev(v):<4.1f}" if len(v) > 1 else "     ")

    print(f"\n{'arm':12s}{'n':>2} {'K':>5} {'lvl':>3} {'tok/s':>12} {'W':>11} {'mJ/token':>12} {'peak DDR':>10}"
          f"{'GPU MHz':>10} {'cap%':>6} {'DDR MHz':>10} {'cpu6 MHz':>10}")
    order = [f"{b}_{t}" for b in ("gpu", "cpu") for t in ("healthy", "mid", "low")]
    for a in order:
        c = arms.get(a)
        if not c:
            print(f"{a:12s} 0   (no cells)")
            continue
        Ks = sorted({x['K'] for x in c}); lv = sorted({x['level'] for x in c})
        print(f"{a:12s}{len(c):>2} {'/'.join(map(str, Ks)):>5} {'/'.join(map(str, lv)):>3} "
              f"{ms([x['tps'] for x in c]):>12} {ms([x['W'] for x in c]):>11} {ms([x['mJ'] for x in c]):>12} {ms([x['ddr'] for x in c]):>10}"
              f"{ms([x['gpu_mhz'] for x in c]):>10} {ms([x['tpl_pct'] for x in c]):>6} {ms([x['ddr_mhz'] for x in c]):>10} {ms([x['cpu6_mhz'] for x in c]):>10}")
    print(f"\n  the trade, per phase:   {'arm':12s} {'prefill s':>12} {'decode s':>12} {'prefill J':>12} {'decode J':>12} {'decode mJ/tok':>14}")
    for a in order:
        c = arms.get(a)
        if c:
            print(f"                          {a:12s} {ms([x['prefill_s'] for x in c]):>12} {ms([x['decode_s'] for x in c]):>12} "
                  f"{ms([x['prefill_J'] for x in c]):>12} {ms([x['decode_J'] for x in c]):>12} {ms([x['dec_mJ'] for x in c]):>14}")
    print(f"\n  whole request:          {'arm':12s} {'total s':>12} {'total J':>12} {'E x T (kJ s)':>14} {'all-in tok/s':>13} {'mJ/all tokens':>14}")
    for a in order:
        c = arms.get(a)
        if c:
            wall = [x["prefill_s"] + x["decode_s"] for x in c]
            E = [x["prefill_J"] + x["decode_J"] for x in c]
            ext = [w_ * e_ / 1000 for w_, e_ in zip(wall, E)]
            print(f"                          {a:12s} {ms(wall):>12} {ms(E):>12} {ms(ext):>14} {ms([x['all_tps'] for x in c]):>13} {ms([x['mJ_all'] for x in c]):>14}")
    starts = sorted({x['start'] for c in arms.values() for x in c if x['start']})
    if starts:
        print("\n  start conditions per cell (temperature gate + caps in force):")
        for a in order:
            for x in arms.get(a, []):
                if x['start']:
                    print(f"    {a:12s} {x['start']}")
    print()
    for b in ("gpu", "cpu"):
        h, l = arms.get(f"{b}_healthy"), arms.get(f"{b}_low")
        if h and l:
            mh = st.mean([x["mJ"] for x in h if x["mJ"] == x["mJ"]] or [float("nan")])
            ml = st.mean([x["mJ"] for x in l if x["mJ"] == x["mJ"]] or [float("nan")])
            th = st.mean([x["tps"] for x in h]); tl = st.mean([x["tps"] for x in l])
            gh = st.mean([x["gpu_mhz"] for x in h if x["gpu_mhz"] == x["gpu_mhz"]] or [float("nan")])
            gl = st.mean([x["gpu_mhz"] for x in l if x["gpu_mhz"] == x["gpu_mhz"]] or [float("nan")])
            print(f"  {b.upper()} healthy -> low : K {h[0]['K']} -> {l[0]['K']}, GPU {gh:.0f} -> {gl:.0f} MHz (measured mean),  "
                  f"mJ/token {mh:.0f} -> {ml:.0f} ({ml/mh-1:+.0%}),  tok/s {th:.1f} -> {tl:.1f} ({tl/th-1:+.0%})")
    bad = glob.glob(os.path.join(host, "*.badlevel_*"))
    if bad:
        print(f"\n  discarded for level mismatch: {len(bad)}  ({', '.join(os.path.basename(x) for x in bad)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ea_proof_v2")
