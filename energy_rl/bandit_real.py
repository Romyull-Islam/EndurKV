#!/usr/bin/env python3
"""Offline contextual bandit fitted on REAL cells (no simulator).

Each logged request is one (context, arm, reward) sample:
  context = battery tier (healthy / mid / low), which sets the reward weights;
  arm     = the plan that ran (GPU clock cap with K=1024; or CPU cache K);
  reward  = w_q * q  -  w_t * T / T_ref  -  w_e * E / E_ref, from the METERED E and T of that cell.
The bandit's value table is the per-(context, arm) mean reward; its policy is the greedy arm.
Uncertainty from n = 3 to 7 cells per arm is shown by bootstrap: the probability each arm is
best under resampling of the real cells. Request shape is held fixed per table (GPU: 9737-token
prompt, 4096 output; CPU: 9737 prompt, 1024 output) because a bandit cannot compare arms across
different requests.
"""
import sys, os, glob, json, random, statistics as st
sys.path.insert(0, "/home/mislam22/EndurKV_workspace/EndurKV/scripts")
from ea_proof_summary import load

TIERS = {"healthy": (0.55, 0.40, 0.05), "mid": (0.35, 0.20, 0.45), "low": (0.20, 0.10, 0.70)}
QUAL = {"K1024": 1.00, "K512": 0.63, "K256": 0.58}   # worst task (qasper) relative to K=1024

def cells(paths):
    out = []
    for p in paths:
        for d in glob.glob(p):
            if os.path.exists(d + "/meta.json"):
                x = load(d); m = json.load(open(d + "/meta.json"))
                out.append(dict(E=x["prefill_J"] + x["decode_J"], T=m["total_ms"] / 1000, src=os.path.basename(d)))
    return out

GPU = {  # arm -> real cells, all 9737 prompt / 4096 output tokens
    "1200 MHz": cells(["/tmp/ea_proof_v2/gpu_healthy_r[0-9]", "/tmp/sched_proof/control_r[0-9]", "/tmp/sched_proof/sched_healthy_r[0-9]"]),
    "902 MHz":  cells(["/tmp/ea_proof_v2/gpu_mid_r[0-9]", "/tmp/sched_proof/sched_low_nocap_r[0-9]"]),
    "726 MHz":  cells(["/tmp/ea_proof_v2/gpu_low_r[0-9]"]),
}
CPU = {  # arm -> real cells, 9737 prompt / 1024 output tokens, CPU-only build
    "K1024": cells(["/tmp/ea_proof_v2/cpu_healthy_r[0-9]"]),
    "K512":  cells(["/tmp/ea_proof_v2/cpu_mid_r[0-9]"]),
    "K256":  cells(["/tmp/ea_proof_v2/cpu_low_r[0-9]"]),
}

def reward(c, w, q, Eref, Tref):
    wq, wt, we = w
    return wq * q - wt * c["T"] / Tref - we * c["E"] / Eref

def fit(table, name, qual, rng=random.Random(0), B=2000):
    ref_arm = list(table)[0]
    Eref = st.mean(c["E"] for c in table[ref_arm]); Tref = st.mean(c["T"] for c in table[ref_arm])
    print(f"\n== {name}: {sum(len(v) for v in table.values())} real cells; reference (full-performance arm) E={Eref:.0f} J, T={Tref:.0f} s")
    print(f"  {'arm':10s}{'n':>3}{'E (J)':>14}{'T (s)':>12}", "".join(f"{'r_'+t:>11}" for t in TIERS))
    for arm, cs in table.items():
        print(f"  {arm:10s}{len(cs):>3}{st.mean(c['E'] for c in cs):8.0f}±{(st.pstdev([c['E'] for c in cs]) if len(cs)>1 else 0):<5.0f}{st.mean(c['T'] for c in cs):7.0f}±{(st.pstdev([c['T'] for c in cs]) if len(cs)>1 else 0):<4.0f}",
              "".join(f"{st.mean(reward(c, TIERS[t], qual.get(arm, 1.0), Eref, Tref) for c in cs):11.3f}" for t in TIERS))
    policy = {}
    for t, w in TIERS.items():
        means = {arm: st.mean(reward(c, w, qual.get(arm, 1.0), Eref, Tref) for c in cs) for arm, cs in table.items()}
        best = max(means, key=means.get)
        wins = {arm: 0 for arm in table}
        for _ in range(B):
            samp = {arm: st.mean(reward(rng.choice(cs), w, qual.get(arm, 1.0), Eref, Tref) for _ in cs) for arm, cs in table.items()}
            wins[max(samp, key=samp.get)] += 1
        policy[t] = dict(best=best, p_best={a: wins[a] / B for a in table}, means=means)
        print(f"  tier {t:8s}: greedy arm = {best:9s}  P(best) under bootstrap: " + ", ".join(f"{a} {wins[a]/B:.2f}" for a in table))
    return policy

pol_gpu = fit(GPU, "GPU, K held at 1024, 4096 output tokens", {})
pol_cpu = fit(CPU, "CPU (GPU unavailable), 1024 output tokens, quality = worst task", QUAL)
json.dump({"gpu": pol_gpu, "cpu": pol_cpu}, open("/home/mislam22/EndurKV_workspace/EndurKV/energy_rl/bandit_real.json", "w"), indent=1)
print("\nrule's choice for comparison: GPU healthy 1200, mid 902, low 902; CPU: K1024 at every tier")
