#!/usr/bin/env python3
"""Measure every GPU plan in PLANS in one session, alternating order each round. Energy drifts
about 5% between sessions, so rows the scheduler compares must come from the same session.
Plans run through the scheduler with --plan and --no-feedback, cooled, then write_table() updates the phone table.
Usage: ANDROID_SERIAL=... run_plan_block.py [--rounds 3]
"""
import argparse, json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_guarded_bandit as gb

PLANS = ["gpu1200_k1024", "gpu1050_k1024", "gpu967_k1024", "gpu902_k1024",
         "gpu1200d902_k1024", "gpu1200d726_k1024"]
SOC = 80                       # one lever for every plan, --plan overrides the walk anyway
NP, ND = 9737, 1024            # the prompt and answer this table is written for
SEED_ROWS = {                  # from campaigns/clock_fine_20260921, so --plan accepts them on round 0
    "gpu1050_k1024": "gpu1050_k1024    gpu     1050    1024  54.5   189.4  13.7   21.4   1.00",
    "gpu967_k1024":  "gpu967_k1024     gpu     967     1024  46.0   210.3  14.8   24.4   1.00",
}
TABLE = "/data/local/tmp/endurkv/tables/Llama-3.2-1B-Instruct-Q4_K_M.txt"


def seed_rows(out):
    have = gb.adb(f"cat {TABLE}", su=True)
    for plan, row in SEED_ROWS.items():
        if f"\n{plan} " in have or have.startswith(plan + " "):
            continue
        gb.adb(f"cp {TABLE} {TABLE}.bak_planblock; printf '%s\\n' '{row}' >> {TABLE}", su=True)
        gb.log(f"seeded {plan} into the phone's table (fine-ladder values, overwritten at the end)", out)
        have = gb.adb(f"cat {TABLE}", su=True)


def phases(tag, out):
    """Phase split (pre_J, dec_J, prefill_ms, decode_ms) from the saved scheduler log line."""
    f = os.path.join(out, "requests", tag, "sched_log.txt")
    if not os.path.exists(f):
        return None
    kv = dict(re.findall(r"(\w+)=(\S+)", open(f).read()))
    try:
        return dict(prefill_J=float(kv["pre_J"]), decode_J=float(kv["dec_J"]),
                    prefill_ms=float(kv["prefill_ms"]), decode_ms=float(kv["decode_ms"]))
    except KeyError:
        return None


def run_one(plan, tag, out):
    """One cooled request on this plan. Returns (metered result, phase split) or (None, None)."""
    for attempt in range(3):
        if not gb.settle(out):
            gb.log(f"    {tag}: phone would not cool", out)
            return None, None
        gb.pin_cpu()
        sub = f"{tag}_a{attempt}"
        res = gb.run_request(sub, SOC, plan, "", out)
        if res is None:
            gb.log(f"    {tag}: no scheduler line; retrying", out)
            continue
        if res["plan_ran"] != plan:
            gb.log(f"    {tag}: ran {res['plan_ran']}, not {plan}; retrying", out)
            continue
        if not gb.meter_ok(res):
            gb.log(f"    {tag}: meter cov={res['meter_cov']} E={res['E']:.0f} vs pred {res['pred_E']:.0f}; retrying", out)
            continue
        ph = phases(sub, out)
        if ph is None:
            gb.log(f"    {tag}: no phase split in the scheduler line; retrying", out)
            continue
        return res, ph
    return None, None


def write_table(state, out):
    """Rewrite measured rows with per-token energy (mJ) and time (ms).
    Split plans prefill at 1200 MHz, so they take the 1200 row's prefill columns."""
    rows, flat1200 = {}, None
    for plan in PLANS:
        R = [r for r in state["runs"] if r["plan"] == plan]
        if len(R) < 2:
            gb.log(f"write_table: {plan} has {len(R)} good runs; leaving its row alone", out)
            continue
        e_pre = st.mean(r["prefill_J"] for r in R) * 1000 / NP
        e_dec = st.mean(r["decode_J"] for r in R) * 1000 / ND
        t_pre = st.mean(r["prefill_ms"] for r in R) / NP
        t_dec = st.mean(r["decode_ms"] for r in R) / ND
        rows[plan] = [e_pre, e_dec, t_pre, t_dec]
        if plan == "gpu1200_k1024":
            flat1200 = rows[plan]
    for plan, v in rows.items():
        if "d" in plan.split("_")[0][3:] and flat1200:      # a split plan: prefill is the 1200 row's
            v[0], v[2] = flat1200[0], flat1200[2]
    open(os.path.join(out, "new_rows.txt"), "w").write(
        "\n".join(f"{p} {v[0]:.1f} {v[1]:.1f} {v[2]:.1f} {v[3]:.1f}" for p, v in rows.items()) + "\n")
    for plan, v in rows.items():
        mhz = plan[3:].split("d")[0].split("_")[0]
        line = f"{plan:17s}gpu     {mhz:7s} 1024  {v[0]:.1f}   {v[1]:.1f}  {v[2]:.1f}   {v[3]:.1f}   1.00"
        gb.adb(f"sed -i '/^{plan} /d' {TABLE}; printf '%s\\n' '{line}' >> {TABLE}", su=True)
        gb.log(f"table row {plan}: e_pre {v[0]:.1f} e_dec {v[1]:.1f} t_pre {v[2]:.1f} t_dec {v[3]:.1f}", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="/home/mislam22/EndurKV_workspace/campaigns/plan_block_20260922")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "requests"), exist_ok=True)
    sp = os.path.join(a.out, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {"runs": []}
    done = {(r["plan"], r["round"]) for r in state["runs"]}
    # Freeze the lever and the table during measurement. run_request builds every
    # command from gb.SCHED, so the flag reaches each launch.
    if "--no-feedback" not in gb.SCHED:
        gb.SCHED = gb.SCHED + " --no-feedback"
    gb.log(f"plan block: {len(PLANS)} plans x {a.rounds} rounds, one session, soc {SOC}, no feedback", a.out)
    gb.adb(f"cp {TABLE} {TABLE}.bak_before_planblock", su=True)
    seed_rows(a.out)
    try:
        for rd in range(a.rounds):
            order = PLANS if rd % 2 == 0 else PLANS[::-1]
            for plan in order:
                if (plan, rd) in done:
                    continue
                tag = f"pb_{plan}_r{rd}"
                gb.log(f"{tag}", a.out)
                res, phase = run_one(plan, tag, a.out)
                if res is None:
                    continue
                state["runs"].append(dict(plan=plan, round=rd, **res, **phase))
                json.dump(state, open(sp, "w"), indent=1)
                gb.log(f"    E={res['E']:.0f} J T={res['T']:.0f} s "
                       f"(prefill {phase['prefill_J']:.0f} J / {phase['prefill_ms'] / 1000:.0f} s, "
                       f"decode {phase['decode_J']:.0f} J / {phase['decode_ms'] / 1000:.0f} s)", a.out)
        write_table(state, a.out)
    finally:
        gb.restore(a.out)
    for plan in PLANS:
        R = [r for r in state["runs"] if r["plan"] == plan]
        if R:
            gb.log(f"{plan:20s} n={len(R)} E {st.mean(r['E'] for r in R):6.0f} J  T {st.mean(r['T'] for r in R):6.1f} s", a.out)
    open(os.path.join(a.out, "DONE"), "w").write("done\n")


if __name__ == "__main__":
    main()
