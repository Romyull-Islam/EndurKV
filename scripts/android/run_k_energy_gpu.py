#!/usr/bin/env python3
"""Energy and time per request on the phone GPU at cache budget K = 512, 1024, 2048, 4096, in
interleaved rounds. Uses run_bandit_online.py's runner unchanged (9737-token prompt, 1024 output
tokens, Llama-3.2-1B, muKV, GPU 1200 MHz, cool gate), so only K changes.
Usage: ANDROID_SERIAL=... run_k_energy_gpu.py [--rounds 3]"""
import argparse, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_bandit_online as rbo

BASE_MU = rbo.MU.replace("--k-nominal 1024", "").strip()
KS = [512, 1024, 2048, 4096]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="/home/mislam22/EndurKV_workspace/campaigns/k_energy_gpu_20260921")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "pulls"), exist_ok=True)
    sp = os.path.join(a.out, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {"runs": []}
    done = {(r["K"], r["round"]) for r in state["runs"]}
    rbo.log(f"K energy sweep: K {KS}, {a.rounds} rounds, 1200 MHz, serial {rbo.SERIAL}", a.out)
    try:
        for rd in range(a.rounds):
            for K in KS:
                if (K, rd) in done:
                    continue
                rbo.MU = f"{BASE_MU} --k-nominal {K}"
                tag = f"k{K}_r{rd}"
                rbo.log(f"{tag}: K={K}, round {rd}", a.out)
                res = rbo.run_pull(tag, 1200, a.out)
                if res is None:
                    continue
                E, T, extra = res
                state["runs"].append(dict(K=K, round=rd, E=E, T=T, **extra))
                json.dump(state, open(sp, "w"), indent=1)
                rbo.log(f"    E={E:.0f} J T={T:.0f} s decode {extra['tps']:.1f} tok/s DDR {extra['ddr_peak']}", a.out)
    finally:
        rbo.restore()
    for K in KS:
        R = [r for r in state["runs"] if r["K"] == K]
        if R:
            rbo.log(f"K={K:5d} n={len(R)} E {st.mean(r['E'] for r in R):6.0f} J  T {st.mean(r['T'] for r in R):6.1f} s  "
                    f"decode {st.mean(r['tps'] for r in R):5.1f} tok/s", a.out)
    open(os.path.join(a.out, "DONE"), "w").write("done\n")


if __name__ == "__main__":
    main()
