#!/usr/bin/env python3
"""Host-side evaluator of the on-phone scheduler policy (scripts/android/ukv_sched.sh).

Same table, same weights, same utility. No phone dynamics are simulated: this only shows
which plan the scheduler picks for a given phone state and request shape, and what the
measured cost table predicts for it. Use it to read the policy; use the phone to test it.
"""
import sys, os

TABLE = os.path.join(os.path.dirname(__file__), "..", "scripts", "android", "ukv_sched_table.txt")

def load_table(path=TABLE):
    rows = []
    for line in open(path):
        if line.startswith("#") or len(line.split()) < 9:
            continue
        f = line.split()
        rows.append(dict(plan=f[0], backend=f[1], gpu_mhz=int(f[2]), K=int(f[3]),
                         e_pre=float(f[4]), e_dec=float(f[5]), t_pre=float(f[6]), t_dec=float(f[7]), q=float(f[8])))
    return rows

def lever_default(soc, mains):
    if mains: return "mains", 1.0
    if soc > 50: return "healthy", 1.0
    if soc > 20: return "mid", 0.5
    return "low", 0.0

def params(L):
    """The lever L in [0, 1] (1 = performance) sets every parameter the two loops trade at,
    piecewise linear through the tier anchors L = 0, 0.5, 1 (same mapping as ukv_sched.sh)."""
    pl = lambda a, b, c: a + (b - a) * (L / 0.5) if L < 0.5 else b + (c - b) * ((L - 0.5) / 0.5)
    w = (pl(0.20, 0.35, 0.55), pl(0.10, 0.20, 0.40), pl(0.70, 0.45, 0.05))
    return dict(w=w, lam=pl(0.5, 0.8, 1.5), qf=pl(0.75, 0.90, 1.00), tslack=0.03 + 0.37 * (1 - L), esave=0.25 * (1 - L),
                cap=4096 if L >= 0.75 else 1024 if L >= 0.25 else 512)

def weights(soc, mains, lever=None):
    """tier, (w_q, w_t, w_e), lam, q_floor, tslack for the lever position (default from the tier)."""
    tier, L0 = lever_default(soc, mains)
    p = params(L0 if lever is None else lever)
    return tier, p["w"], p["lam"], p["qf"], p["tslack"]

def output_cap(tier, max_tokens=None, size=None, prompt_states_length=False, lever=None):
    if max_tokens: return max_tokens, "caller --max-tokens"
    if size: return {"short": 256, "medium": 1024, "long": 4096}[size], f"caller --size {size}"
    if prompt_states_length: return 4096, "prompt states a length; not capped"
    L = lever_default(0, tier == "mains")[1] if lever is None else lever
    if lever is None:
        L = {"mains": 1.0, "healthy": 1.0, "mid": 0.5, "low": 0.0}[tier]
    return params(L)["cap"], f"lever cap for L={L:.2f}"

def choose(rows, n_prompt, n_out, w, lam, qf, hot=False, gpu_ok=True, tslack=0.03):
    """1. backend by weighted utility; 2. ladder walk inside it: exchange rate (energy loop),
    time budget (performance loop), quality floor. Split-clock plans (d<MHz> in the name) are
    ordinary rows; a hot phone only excludes plans that prefill at 1200 without a decode cap."""
    cands = [r for r in rows if (gpu_ok or r["backend"] != "gpu")
             and not (hot and r["backend"] == "gpu" and r["gpu_mhz"] >= 1200 and "d" not in r["plan"].split("_")[0][3:])]
    for r in cands:
        r["E"] = n_prompt * r["e_pre"] + n_out * r["e_dec"]
        r["T"] = n_prompt * r["t_pre"] + n_out * r["t_dec"]
    eb = min(r["E"] for r in cands); tb = min(r["T"] for r in cands)
    wq, wt, we = w
    for r in cands:
        r["U"] = wq * r["q"] - wt * r["T"] / tb - we * r["E"] / eb
    bk = max(cands, key=lambda r: r["U"])["backend"]
    ladder = sorted([r for r in cands if r["backend"] == bk], key=lambda r: (-r["q"], r["T"]))
    cur = ladder[0]; tbud = cur["T"] * (1 + tslack); walk = [f"start {cur['plan']}"]
    for k in ladder[1:]:
        # skip plans that save nothing or trade poorly against the current one (a later, larger
        # step may still pay); stop only on the monotone limits: time budget and quality floor
        if k["q"] < qf:
            walk.append(f"{k['plan']}: stop (quality)"); break
        if k["T"] > tbud:
            walk.append(f"{k['plan']}: stop (over time budget)"); break
        dE = (cur["E"] - k["E"]) / cur["E"]; dT = (k["T"] - cur["T"]) / cur["T"]
        ok = k["E"] < cur["E"] and (dT <= 0 or dE >= lam * dT)
        why = "no saving" if k["E"] >= cur["E"] else "poor exchange"
        walk.append(f"{k['plan']} (dE {dE:+.0%}, dT {dT:+.0%}): {'take' if ok else 'skip (' + why + ')'}")
        if ok:
            cur = k
    cur["walk"] = " -> ".join(walk)
    return cur, cands

def main(table=TABLE):
    rows = load_table(table)
    states = [("mains", 100, True, False), ("healthy 80%", 80, False, False), ("mid 40%", 40, False, False),
              ("low 15%", 15, False, False), ("healthy 80%, hot", 80, False, True)]
    shapes = [("long prompt, short answer", 9737, 64), ("long prompt, open answer", 9737, None), ("short prompt, open answer", 512, None)]
    print(f"{'state':20s} {'request':28s} {'cap':>5} {'plan':18s} {'E (J)':>7} {'T (s)':>6} {'q':>5}")
    for sname, soc, mains, hot in states:
        tier, w, lam, qf, ts = weights(soc, mains)
        for rname, np_, no in shapes:
            cap, rule = output_cap(tier, max_tokens=no)
            best, _ = choose([dict(r) for r in rows], np_, cap, w, lam, qf, hot=hot, tslack=ts)
            print(f"{sname:20s} {rname:28s} {cap:>5} {best['plan']:18s} {best['E']/1000:7.0f} {best['T']/1000:6.1f} {best['q']:5.2f}")
    print("\nlong prompt, open answer: the ladder walk at each tier (where the scheduler stops, and why):")
    for sname, soc, mains, hot in states[:4]:
        tier, w, lam, qf, ts = weights(soc, mains); cap, _ = output_cap(tier)
        best, cands = choose([dict(r) for r in rows], 9737, cap, w, lam, qf, tslack=ts)
        print(f"  {sname:12s} lam={lam:<4.2f} qf={qf:.2f} slack={ts:.2f}: {best['walk']}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else TABLE)
    raise SystemExit

if __name__ == "__main__":
    main()
