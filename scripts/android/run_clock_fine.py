#!/usr/bin/env python3
"""A finer GPU clock ladder: 1050, 967 and 826 MHz beside the 1200 / 902 anchors. (2026-09-21)

WHY. The scheduler and both bandits use three clocks (1200, 902, 726) of the Adreno 840's eighteen.
The mid tier cannot take 1200 -> 902 (+27% time on a 1024-token request, over its 21% slack), so it
settles for 1200 with a decode cap and saves about 6%. An earlier one-run sweep (clock_sweep, CPU at
883 MHz, 4096 output tokens) put 1050 on the steep side of the knee and showed 826 as fast as 902 and
no more costly than 726. If 1050 or 967 fits mid's slack, a finer ladder saves energy the coarse one
cannot reach; if 826 dominates 726, the bottom rung should change.

It reuses run_bandit_online.py's per-request runner unchanged (cool gate, CPU pinned at 1785.6 /
1497.6 MHz, 2 Hz sampler, USB rail + battery pack, K = 1024, 9737-token prompt, 1024 output tokens),
so the new points sit on the 24-request ladder's scale. 1200 and 902 are re-measured each round as
anchors against drift. Three rounds, interleaved.
Usage: ANDROID_SERIAL=... run_clock_fine.py [--rounds 3]
"""
import argparse, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_bandit_online as rbo

CLOCKS = [1200, 1050, 967, 902, 826]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="/home/mislam22/EndurKV_workspace/campaigns/clock_fine_20260921")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "pulls"), exist_ok=True)
    sp = os.path.join(a.out, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {"runs": []}
    done = {(r["mhz"], r["round"]) for r in state["runs"]}
    rbo.log(f"fine clock ladder: {CLOCKS} MHz, {a.rounds} rounds, K 1024, serial {rbo.SERIAL}", a.out)
    try:
        for rd in range(a.rounds):
            order = CLOCKS if rd % 2 == 0 else CLOCKS[::-1]     # alternate direction against drift
            for mhz in order:
                if (mhz, rd) in done:
                    continue
                tag = f"c{mhz}_r{rd}"
                rbo.log(f"{tag}: {mhz} MHz, round {rd}", a.out)
                res = rbo.run_pull(tag, mhz, a.out)
                if res is None:
                    continue
                E, T, extra = res
                state["runs"].append(dict(mhz=mhz, round=rd, E=E, T=T, **extra))
                json.dump(state, open(sp, "w"), indent=1)
                rbo.log(f"    E={E:.0f} J T={T:.0f} s decode {extra['tps']:.1f} tok/s gpu {extra['gpu_mhz']:.0f} MHz DDR {extra['ddr_peak']}", a.out)
    finally:
        rbo.restore()
    for mhz in CLOCKS:
        R = [r for r in state["runs"] if r["mhz"] == mhz]
        if R:
            rbo.log(f"{mhz:5d} MHz n={len(R)} E {st.mean(r['E'] for r in R):6.0f} J  T {st.mean(r['T'] for r in R):6.1f} s  "
                    f"decode {st.mean(r['tps'] for r in R):5.1f} tok/s", a.out)
    open(os.path.join(a.out, "DONE"), "w").write("done\n")


if __name__ == "__main__":
    main()
