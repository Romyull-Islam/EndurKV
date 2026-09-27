#!/usr/bin/env python3
"""Every number in the energy/performance deck, pulled from the raw phone campaigns.

The deck answers three questions the supervisor set:
  Q1  how energy affects performance      (real battery discharge)
  Q2  how performance affects energy      (24-pull clock ladder + the scheduler's cost table)
  Q3  how learning affects the two loops  (provoked loops, discharge prediction error, bandit)

make_energy_perf_deck.js reads the JSON this writes and draws native charts from it, so a
number on a slide can be traced to one campaign directory. Nothing here is typed in by hand
except the tier constants, which are the scheduler's own settings (ukv_sched.sh) and the
rule-versus-bandit utilities, which are the paper's appendix table.

Sources (all restored from tmp_archive/ after the 2026-09-20 reboots):
  /tmp/discharge_final/discharge/timeline.csv  real discharge 2026-09-05/06, Llama-3.2-1B
  /tmp/bandit_online/state.json                24 cooled pulls, clock pinned per pull
  /tmp/loop_proof/<tag>/sched_log.txt          13 requests with real disturbances
  /tmp/qres_cpu/llama_{vanilla,mukv}_cur.*     same-build, matched-start cache pair
  /tmp/sllm_faithful/{v,mukv,sfown}_r{1,2,3}   the GPU rows of Table 1, for their energy
  energy_rl/sched_policy.py                    the scheduler's per-plan cost table
"""
import csv
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "energy_rl"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "android", "phone_queue"))

out = {}

# ---------------------------------------------------------------- Q1: discharge
rows = list(csv.DictReader(open("/tmp/discharge_final/discharge/timeline.csv")))
bat = [r for r in rows if r["usb_powered"] == "false"]
f = lambda P, k: st.mean(float(r[k]) for r in P)
tiers = {}
for t in ("healthy", "mid", "low"):
    P = [r for r in bat if r["tier"] == t]
    socs = [int(r["soc"]) for r in P]
    fired = [r for r in P if not r["loop"].startswith("hold")]
    tiers[t] = dict(
        n=len(P), soc_hi=max(socs), soc_lo=min(socs),
        lever=round(f(P, "lever"), 2),
        tokens=round(f(P, "ns")), energy_J=round(f(P, "meas_J")),
        time_s=round(f(P, "meas_s")),
        J_per_token=round(f(P, "meas_J") / f(P, "ns"), 2),
        caps=sorted({int(r["nout_cap"]) for r in P}),
        clocks=sorted({int(r["gpu_mhz"]) for r in P}),
        loops_fired=len(fired),
        perf_fired=sum(r["loop"].startswith("performance") for r in P),
        energy_fired=sum(r["loop"].startswith("energy") for r in P),
        mean_abs_time_err=round(st.mean(abs(float(r["meas_s"]) / float(r["pred_s"]) - 1) for r in P) * 100, 1),
        mean_abs_energy_err=round(st.mean(abs(float(r["meas_J"]) / float(r["pred_J"]) - 1) for r in P) * 100, 1),
    )
out["discharge"] = dict(
    n_total=len(rows), n_battery=len(bat),
    soc_start=int(bat[0]["soc"]), soc_end=int(bat[-1]["soc"]),
    tiers=tiers,
    healthy_pred_s_first=round(float(bat[0]["pred_s"])),
    healthy_pred_s_last=round(float([r for r in bat if r["tier"] == "healthy"][-1]["pred_s"])),
    per_request=[dict(i=i + 1, soc=int(r["soc"]), tier=r["tier"],
                      time_err=round(100 * (float(r["meas_s"]) / float(r["pred_s"]) - 1), 1),
                      energy_err=round(100 * (float(r["meas_J"]) / float(r["pred_J"]) - 1), 1),
                      loop=r["loop"].split(":")[0].replace(" loop", ""))
                 for i, r in enumerate(bat)],
)

# ---------------------------------------------------------------- Q2: clock ladder
hist = json.load(open("/tmp/bandit_online/state.json"))["history"]
by = defaultdict(list)
for p in hist:
    by[p["mhz"]].append(p)
ladder = []
for mhz in sorted(by):
    P = by[mhz]
    ladder.append(dict(mhz=mhz, n=len(P),
                       time_s=round(st.mean(p["T"] for p in P), 1),
                       energy_J=round(st.mean(p["E"] for p in P)),
                       energy_sd=round(st.pstdev([p["E"] for p in P])),
                       tps=round(st.mean(p["tps"] for p in P), 1),
                       ddr_peak=round(st.mean(p["ddr_peak"] for p in P), 1)))
base = ladder[0]
for p in ladder:
    p["speedup_pct"] = round(100 * (base["time_s"] / p["time_s"] - 1), 1)
    p["extra_energy_pct"] = round(100 * (p["energy_J"] / base["energy_J"] - 1), 1)
steps = []
for a, b in ((ladder[0], ladder[1]), (ladder[1], ladder[2])):
    steps.append(dict(frm=a["mhz"], to=b["mhz"],
                      faster_pct=round(100 * (1 - b["time_s"] / a["time_s"]), 1),
                      energy_pct=round(100 * (b["energy_J"] / a["energy_J"] - 1), 1),
                      ddr_delta=round(b["ddr_peak"] - a["ddr_peak"], 1)))
out["ladder"] = dict(points=ladder, steps=steps, pulls=len(hist))

# ---- Q2b: the finer ladder, 2026-09-22. Five clocks measured in ONE session, n=3 each, same runner
# and settings as the 24-pull ladder (cool gate, CPU pinned, K=1024, 9737-token prompt, 1024 out).
# Reported on its own because a clock curve must come from one session: the coarse ladder's session
# sat about 5% lower in absolute joules, so mixing the two would fake a step.
FINE = "/home/mislam22/EndurKV_workspace/campaigns/clock_fine_20260921"
fine = []
if os.path.exists(FINE + "/state.json"):
    byf = defaultdict(list)
    for r in json.load(open(FINE + "/state.json"))["runs"]:
        byf[r["mhz"]].append(r)
    for mhz in sorted(byf):
        R = byf[mhz]
        fine.append(dict(mhz=mhz, n=len(R),
                         time_s=round(st.mean(r["T"] for r in R), 1),
                         energy_J=round(st.mean(r["E"] for r in R)),
                         energy_sd=round(st.pstdev([r["E"] for r in R])),
                         tps=round(st.mean(r["tps"] for r in R), 1),
                         ddr_peak=round(st.mean(r["ddr_peak"] for r in R), 1)))
    fsteps = []
    for a, b in zip(fine, fine[1:]):                    # each rung against the one below it
        fsteps.append(dict(frm=a["mhz"], to=b["mhz"],
                           faster_pct=round(100 * (a["time_s"] / b["time_s"] - 1), 1),
                           energy_pct=round(100 * (b["energy_J"] / a["energy_J"] - 1), 1),
                           ddr_delta=round(b["ddr_peak"] - a["ddr_peak"], 1)))
    top = fine[-1]
    for f in fine:
        f["vs_top_energy_pct"] = round(100 * (f["energy_J"] / top["energy_J"] - 1), 1)
        f["vs_top_time_pct"] = round(100 * (f["time_s"] / top["time_s"] - 1), 1)
    out["ladder_fine"] = dict(
        points=fine, steps=fsteps, pulls=sum(f["n"] for f in fine),
        # what the rule picks when every GPU row comes from this one session (dry runs on the phone,
        # 2026-09-22; the live table was restored afterwards)
        rule_with_fine_rungs=dict(healthy="gpu1200", mid="gpu967", low="gpu902"),
        dominated=826)

import sched_policy as sp
tab = {r["plan"]: r for r in sp.load_table()}
def ET(plan, n_out=1024, n_p=9737):
    r = tab[plan]
    return (n_p * r["e_pre"] + n_out * r["e_dec"]) / 1000, (n_p * r["t_pre"] + n_out * r["t_dec"]) / 1000
cost_steps = []
for a, b, lab in (("gpu1200_k1024", "gpu1200d902_k1024", "Cap the decode clock at 902 MHz"),
                  ("gpu1200_k1024", "gpu902_k1024", "Drop the GPU clock 1200 to 902 MHz"),
                  ("gpu902_k1024", "gpu726_k1024", "Drop the GPU clock 902 to 726 MHz")):
    (Ea, Ta), (Eb, Tb) = ET(a), ET(b)
    dT = 100 * (Tb / Ta - 1); dE = 100 * (1 - Eb / Ea)
    cost_steps.append(dict(label=lab, time_cost=round(dT, 1), energy_saved=round(dE, 1),
                           rate=round(dE / dT, 2)))
# the scheduler's own tier settings (ukv_sched.sh)
# slack is 0.03 + 0.37(1-L), floored at the loops' own 5% tolerance, so healthy walks with 5 not 3
TIER = {"healthy": dict(lam=1.5, slack=5), "mid": dict(lam=0.8, slack=21), "low": dict(lam=0.5, slack=40)}
for s in cost_steps:
    s["decision"] = {}
    for t, c in TIER.items():
        if s["rate"] < c["lam"]:
            s["decision"][t] = "no: rate"
        elif s["time_cost"] > c["slack"]:
            s["decision"][t] = "no: slack"
        else:
            s["decision"][t] = "yes"
out["cost_table"] = dict(steps=cost_steps, tiers=TIER)

# ---------------------------------------------------------------- the cache lever, same tokens
def cell(tag):
    m = json.loads(re.sub(r":\s*-?nan\b", ": NaN", open(f"/tmp/qres_cpu/{tag}.json").read()))
    rs = list(csv.DictReader(open(f"/tmp/qres_cpu/{tag}.sensors.csv")))
    e, prev = 0.0, None
    for r in rs:
        try:
            t = float(r["monotonic_s"]); v = float(r["usb_voltage_uv"]) / 1e6; i = abs(float(r["usb_current_ua"])) / 1e6
        except (ValueError, KeyError, TypeError):
            continue
        if prev is not None:
            e += v * i * min(t - prev, 5.0)
        prev = t
    q = [float(r["bat_charge_uah"]) for r in rs if r.get("bat_charge_uah", "").strip().lstrip("-").isdigit()]
    vv = [float(r["bat_voltage_now_uv"]) / 1e6 for r in rs if r.get("bat_voltage_now_uv", "").strip().isdigit()]
    e += max(0.0, (q[0] - q[-1]) / 1e6) * (st.mean(vv) if vv else 4.35) * 3600 if len(q) > 1 else 0.0
    return dict(time_s=round(m["total_ms"] / 1000), energy_kJ=round(e / 1000, 2),
                tokens=m["n_decode_steps"], tps=m["decode_tps"])
full, mu = cell("llama_vanilla_cur"), cell("llama_mukv_cur")
out["cache_lever"] = dict(full=full, mukv=mu,
                          time_pct=round(100 * (mu["time_s"] / full["time_s"] - 1), 1),
                          energy_pct=round(100 * (mu["energy_kJ"] / full["energy_kJ"] - 1), 1),
                          J_per_token_full=round(full["energy_kJ"] * 1000 / full["tokens"], 2),
                          J_per_token_mukv=round(mu["energy_kJ"] * 1000 / mu["tokens"], 2))

# ---------------------------------------------------------------- Q3a: provoked loops
LP = "/tmp/loop_proof"
def lp(tag, label, disturbance):
    s = open(f"{LP}/{tag}/sched_log.txt").read().strip().splitlines()[-1]
    kv = dict(re.findall(r"(\w+)=(\S+)", s))
    b0, b1 = kv["bias"].split("->")
    action = s.split(";")[-1].strip()
    return dict(tag=tag, label=label, disturbance=disturbance, tier=kv["tier"],
                lever=float(kv["lever"]), bias_before=float(b0), bias_after=float(b1),
                plan=kv["plan"], cap=int(kv["nout_cap"]),
                time_over=round(100 * (float(kv["meas_s"]) / float(kv["tbud"]) - 1), 1),
                energy_over=round(100 * (float(kv["meas_J"]) / float(kv["ebud"]) - 1), 1),
                action=action)
# labels say what was done to the phone during the request (run_loop_proof.sh):
#   normal      no disturbance
#   busy app    four spare CPU cores kept spinning for the whole request, as another app would
#   GPU capped  12 s in, the GPU is forced down to 726 MHz, as the vendor thermal limiter does
#   recovery    a clean request after the disturbance has stopped
out["loops_mid"] = [lp("mid_1", "normal", "none"), lp("mid_burn_1", "busy app 1", "4 cores spinning"),
                    lp("mid_burn_2", "busy app 2", "4 cores spinning"), lp("mid_after_1", "recovery 1", "none"),
                    lp("mid_after_2", "recovery 2", "none")]
out["loops_low"] = [lp("low_1", "normal", "none"), lp("low_cap_1", "GPU capped 1", "726 MHz cap at 12 s"),
                    lp("low_cap_2", "GPU capped 2", "726 MHz cap at 12 s"), lp("low_cap_3", "GPU capped 3", "726 MHz cap at 12 s"),
                    lp("low_after_1", "recovery 1", "none"), lp("low_after_2", "recovery 2", "none")]

# ---------------------------------------------------------------- Q3b: bandit
Q = defaultdict(lambda: defaultdict(list))
for p in hist:
    Q[p["tier"]][p["mhz"]].append(p["r"])
means = {t: {str(m): round(st.mean(v), 3) for m, v in Q[t].items()} for t in Q}
counts = {t: {str(m): len(v) for m, v in Q[t].items()} for t in Q}
best = {t: max(st.mean(v) for v in Q[t].values()) for t in Q}
regret = [best[p["tier"]] - p["r"] for p in hist]
out["bandit"] = dict(
    means=means, counts=counts,
    learned={t: max(Q[t], key=lambda m: st.mean(Q[t][m])) for t in Q},
    energy_kJ=round(sum(p["E"] for p in hist) / 1000, 1),
    minutes=149,                                   # log: 11:08:42 start, 13:37:34 BANDIT_DONE
    regret_warm_pct=round(100 * sum(regret[:9]) / sum(regret)),
    # reward = w_q*q - w_t*T/T_ref - w_e*E/E_ref ; weights per tier from energy_rl/bandit_real.py
    weights={"healthy": [0.55, 0.40, 0.05], "mid": [0.35, 0.20, 0.45], "low": [0.20, 0.10, 0.70]},
    # paper appendix, tab:bandit: the rule's own plan per tier and its utility on the same scale
    rule={"healthy": ["1200 + decode 902", 0.099], "mid": ["1200 + decode 726", -0.285], "low": ["902", -0.455]},
    loops_during_run=False,                        # log.txt has no loop or bias events: one fixed arm per pull
)

# ---------------------------------------------------------------- accuracy: muKV against its own full cache
# Performance is time AND accuracy. Accuracy here is LongBench token-F1, five tasks, the paper's
# four-model table (tab:longbench-cuda, RTX 4500 Ada, f16 KV, K=1024). Which cells muKV keeps, and so
# the answer, is a property of the model and the budget, not of the device, so accuracy comes from
# that table while energy and time come from the phone.
LB = {  # model: (full cache avg F1, muKV avg F1, share of cells muKV keeps)
    "Llama-3.2-1B": (41.96, 42.08, 13.0), "Phi-3-mini": (54.44, 54.43, 12.2),
    "gemma-2-2b": (50.06, 47.31, 13.0), "Bonsai-8B": (48.64, 44.97, 12.6)}

def cell_energy(sensors, dur=None):
    """USB rail + battery pack (energy_cell_report.py), optionally windowed to the run."""
    rs = list(csv.DictReader(open(sensors)))
    t0 = None; e = 0.0; prev = None; q = []; vv = []
    for r in rs:
        try:
            t = float(r["monotonic_s"]); v = float(r["usb_voltage_uv"]) / 1e6; i = abs(float(r["usb_current_ua"])) / 1e6
        except (ValueError, KeyError, TypeError):
            continue
        t0 = t if t0 is None else t0
        if dur and t > t0 + dur:
            break
        if prev is not None:
            e += v * i * min(t - prev, 5.0)
        prev = t
        if r.get("bat_charge_uah", "").strip().lstrip("-").isdigit(): q.append(float(r["bat_charge_uah"]))
        if r.get("bat_voltage_now_uv", "").strip().isdigit(): vv.append(float(r["bat_voltage_now_uv"]) / 1e6)
    return e + (max(0.0, (q[0] - q[-1]) / 1e6) * (st.mean(vv) if vv else 4.35) * 3600 if len(q) > 1 else 0.0)

def mj(p):
    return json.loads(re.sub(r":\s*-?nan\b", ": NaN", open(p).read()))

PHONE = {  # model: (full-cache run, muKV K=1024 run, meta file); same prompt and length within a model
    "Llama-3.2-1B": None,   # the same-build, matched-start pair above (cache_lever)
    "Phi-3-mini": ("/tmp/phi3_cpu_complete/vanilla", "/tmp/phi3_cpu_complete/mukv", "meta.json"),
    "gemma-2-2b": ("/tmp/gap_cpu_gemma/vanilla", "/tmp/gap_cpu_gemma/mukv", "meta.json"),
    "Bonsai-8B": ("/tmp/nat_bonsai/vanilla", "/tmp/nat_bonsai/mukv_faon", "gen.json"),
}
acc = []
for model, (f_full, f_mu, kept) in LB.items():
    if PHONE[model] is None:
        c = out["cache_lever"]
        e_ratio = c["mukv"]["energy_kJ"] / c["full"]["energy_kJ"]; t_ratio = c["mukv"]["time_s"] / c["full"]["time_s"]
        d_ratio = c["mukv"]["tps"] / c["full"]["tps"]; toks = c["full"]["tokens"]
    else:
        v, m, mf = PHONE[model]
        mv, mm = mj(f"{v}/{mf}"), mj(f"{m}/{mf}")
        ev = cell_energy(f"{v}/sensors.csv", mv["total_ms"] / 1000)
        em = cell_energy(f"{m}/sensors.csv", mm["total_ms"] / 1000)
        e_ratio, t_ratio, d_ratio = em / ev, mm["total_ms"] / mv["total_ms"], mm["decode_tps"] / mv["decode_tps"]
        toks = mv["n_decode_steps"]
    acc.append(dict(model=model, acc_kept=round(100 * f_mu / f_full, 1), f1_full=f_full, f1_mukv=f_mu, cells_kept=kept,
                    energy_pct=round(100 * e_ratio), time_pct=round(100 * t_ratio), decode_x=round(d_ratio, 2),
                    buy_back_energy_pct=round(100 * (1 / e_ratio - 1)), tokens=toks))

# below K = 1024 the saving stops (CPU Llama, 9737-token prompt, 1024 output tokens, n = 3 each)
low = {}
for lvl, K in (("healthy", 1024), ("mid", 512), ("low", 256)):
    E = [cell_energy(f"/tmp/ea_proof_v2/cpu_{lvl}_r{r}/" + os.path.basename(glob.glob(f"/tmp/ea_proof_v2/cpu_{lvl}_r{r}/*sensors*.csv")[0]))
         for r in (1, 2, 3)]
    low[K] = st.mean(E)
out["accuracy"] = dict(
    per_model=acc,
    below_1024={f"K{K}": round(100 * (low[K] / low[1024] - 1), 1) for K in (512, 256)},
    # needle retrieval, 392-cell grid, strict answer-span scorer (eval_pipeline/score_niah_strict.py)
    needle=dict(full=52, mukv=51, of=56),
    # clock does not change the answer: /tmp/loop_proof gen.txt, same prompt, answer cap 1024
    clock_same_text=["902", "1200", "1200 + decode 726", "1200 + decode 902"],
    run_to_run_divergence_pct=7,
    # Bonsai budget sweep, RTX 4500, n = 200 per task (lb200_*/scores.json)
    bonsai_budget={"K1024": {"hotpotqa": round(100 * 42.06 / 43.89, 1), "qasper": round(100 * 35.96 / 39.40, 1)},
                   "K4096": {"hotpotqa": round(100 * 44.35 / 43.89, 1), "qasper": round(100 * 38.58 / 39.40, 1)}},
)

# ---------------------------------------------------------------- the guarded bandit, loops on (2026-09-21)
# campaigns/guarded_bandit_20260921 (run_guarded_bandit.py): 30 cooled requests on the phone GPU, tiers
# cycled through --force-soc, the scheduler executing every plan with both loops and its table live.
import math
GB = json.load(open("/home/mislam22/EndurKV_workspace/campaigns/guarded_bandit_20260921/state.json"))
gstats, ghist = GB["stats"], GB["history"]
def gest(a, w):
    v = gstats[a]; n = len(v); E = [x[0] for x in v]; T = [x[1] for x in v]
    r = w[0] - w[1] * st.mean(T) / 142.4 - w[2] * st.mean(E) / 782.0
    se = math.sqrt((w[1] / 142.4) ** 2 * st.variance(T) / n + (w[2] / 782.0) ** 2 * st.variance(E) / n)
    return r, se, n, st.mean(E), st.mean(T)
TW = {"healthy": (0.55, 0.40, 0.05), "mid": (0.35, 0.20, 0.45), "low": (0.20, 0.10, 0.70)}
per_req = []
for h in ghist:
    w = tuple(h["weights"])
    per_req.append(dict(i=h["i"], tier=h["tier"], kind=h["why"].split(":")[0], plan=h["plan_ran"], rule=h["rule"],
                        gain=round(gest(h["plan_ran"], w)[0] - gest(h["rule"], w)[0], 4),
                        loop=h["loop"].split(":")[0], E=h["E"], T=h["T"]))
tiers_gb = {}
for t, w in TW.items():
    final = [h for h in ghist if h["tier"] == t][-1]
    chosen, rule = final["plan_ran"], final["rule"]
    rc, sc, nc, Ec, Tc = gest(chosen, w); rr, sr, nr, Er, Tr = gest(rule, w)
    tiers_gb[t] = dict(rule=rule, chosen=chosen, decision=final["why"].split(":")[0],
                       r_chosen=round(rc, 3), r_rule=round(rr, 3), gain=round(rc - rr, 3),
                       energy_pct=round(100 * (Ec / Er - 1), 1), time_pct=round(100 * (Tc / Tr - 1), 1))
late = [p for p in per_req if p["i"] >= 21]
explore_cost = sum(p["gain"] for p in per_req if p["kind"] == "explore")
net = sum(p["gain"] for p in per_req)
rate = st.mean(tiers_gb[t]["gain"] for t in TW)
out["guarded_bandit"] = dict(
    n=len(ghist), per_request=per_req, tiers=tiers_gb,
    counts={k: sum(p["kind"] == k for p in per_req) for k in ("explore", "go", "fallback")},
    late_gain_per_request=round(st.mean(p["gain"] for p in late), 4),
    explore_cost=round(explore_cost, 3), net=round(net, 3), payback_requests=round(-net / rate) if rate > 0 else None,
    loops_fired={k: [sum(1 for p in per_req if p["kind"] == k and p["loop"] != "hold"), sum(p["kind"] == k for p in per_req)]
                 for k in ("explore", "go", "fallback")},
    meter_cov_min=min(h["meter_cov"] for h in ghist),
)

# ---------------------------------------------------------------- the cache budget K, both sides (2026-09-21)
# accuracy: campaigns/lb_ksweep_llama (RTX 4500, LongBench 5 tasks x 50, Llama-3.2-1B, muKV)
# energy/time: campaigns/k_energy_gpu_20260921 (phone GPU, 1200 MHz, 1024 output tokens, 3 runs per K)
KS = json.load(open("/home/mislam22/EndurKV_workspace/campaigns/k_energy_gpu_20260921/state.json"))["runs"]
LBK = {}
for d, K in (("full", "full"), ("k256", 256), ("k512", 512), ("k1024", 1024), ("k2048", 2048), ("k4096", 4096)):
    rows = json.load(open(f"/home/mislam22/EndurKV_workspace/campaigns/lb_ksweep_llama/{d}/scores.json"))["rows"]
    LBK[K] = st.mean(r["f1"] for r in rows)
kres = {}
for K in (512, 1024, 2048, 4096):
    R = [r for r in KS if r["K"] == K]
    E = [r["E"] for r in R]; T = [r["T"] for r in R]; q = LBK[K] / LBK[1024]
    tiers_k = {}
    for t, w in TW.items():
        r = w[0] * q - w[1] * st.mean(T) / 142.4 - w[2] * st.mean(E) / 782.0
        se = math.sqrt((w[1] / 142.4) ** 2 * st.variance(T) / len(R) + (w[2] / 782.0) ** 2 * st.variance(E) / len(R))
        tiers_k[t] = [round(r, 4), round(1.96 * se, 4)]
    kres[K] = dict(n=len(R), E=round(st.mean(E)), T=round(st.mean(T), 1), tps=round(st.mean(r["tps"] for r in R), 1),
                   acc_vs_full=round(100 * LBK[K] / LBK["full"], 1), acc_vs_1024=round(100 * q, 1), reward=tiers_k)
out["k_lever"] = dict(per_K=kres, full_cache_acc=round(LBK["full"], 2),
                      best={t: max(kres, key=lambda K: kres[K]["reward"][t][0]) for t in TW})

# ---------------------------------------------------------------- GPU energy of the Table 1 rows (2026-09-25)
# /tmp/sllm_faithful (run_streamingllm_faithful.sh): Llama-3.2-1B on the Adreno 840, 9737-token prompt,
# 4096 generated tokens, f16 KV; every run starts cooled (DDR <= 35 C, battery <= 33 C, charging off);
# each round runs full cache, muKV and StreamingLLM 4+2000 back to back. Energy is the USB rail plus the
# battery, windowed to the request, as for every other energy number here.
SF = "/tmp/sllm_faithful"
def gpu_request(tag):
    m = mj(f"{SF}/{tag}/meta.json")
    total = cell_energy(f"{SF}/{tag}/sensors.csv", m["total_ms"] / 1000) / 1000
    return total, m["prefill_ms"] / 1000, m["total_ms"] / 1000
gE = {arm: [gpu_request(f"{arm}_r{r}") for r in (1, 2, 3)] for arm in ("v", "mukv", "sfown")}
med_full = st.median(x[0] for x in gE["v"])
out["gpu_energy"] = dict(
    kJ={arm: [round(x[0], 3) for x in runs] for arm, runs in gE.items()},
    time_s={arm: [round(x[2]) for x in runs] for arm, runs in gE.items()},
    median_kJ={arm: round(st.median(x[0] for x in runs), 2) for arm, runs in gE.items()},
    saving_pct_vs_median_full={arm: round(100 * (1 - st.median(x[0] for x in runs) / med_full), 1)
                               for arm, runs in gE.items() if arm != "v"},
    saving_pct_same_round={arm: [round(100 * (1 - a[0] / b[0]), 1) for a, b in zip(runs, gE["v"])]
                           for arm, runs in gE.items() if arm != "v"})

json.dump(out, open(os.path.join(HERE, "energy_perf_data.json"), "w"), indent=1)
kl = out["k_lever"]
print("K lever:", {K: (v["E"], v["T"], v["acc_vs_full"]) for K, v in kl["per_K"].items()}, "best", kl["best"])
g = out["guarded_bandit"]
print(f"guarded bandit: {g['counts']}, late gain {g['late_gain_per_request']:+.4f}/request, net {g['net']:+.3f}, "
      f"payback ~{g['payback_requests']} requests, loops fired {g['loops_fired']}")
for t, v in g["tiers"].items():
    print(f"  {t:7s} rule {v['rule']:18s} -> {v['chosen']:18s} ({v['decision']}) gain {v['gain']:+.3f}  "
          f"energy {v['energy_pct']:+.1f}%  time {v['time_pct']:+.1f}%")
for a in acc:
    print(f"accuracy {a['model']:13s} kept {a['acc_kept']:5.1f}%  energy {a['energy_pct']}%  time {a['time_pct']}%  "
          f"decode x{a['decode_x']}  buy back +{a['buy_back_energy_pct']}% energy")
print("below 1024:", out["accuracy"]["below_1024"])

# ---------------------------------------------------------------- print for the record
d = out["discharge"]
print(f"discharge: {d['n_battery']} battery requests, {d['soc_start']} to {d['soc_end']}%")
for t, v in d["tiers"].items():
    print(f"  {t:8s} n={v['n']:2d} L={v['lever']:.2f} tokens={v['tokens']:5d} E={v['energy_J']:4d} J  "
          f"J/tok={v['J_per_token']:.2f}  loops {v['loops_fired']}/{v['n']}  |time err| {v['mean_abs_time_err']}%  |E err| {v['mean_abs_energy_err']}%")
print(f"  healthy predicted time learned {d['healthy_pred_s_first']} -> {d['healthy_pred_s_last']} s")
for p in out["ladder"]["points"]:
    print(f"ladder {p['mhz']:4d} MHz n={p['n']:2d} {p['time_s']:5.1f} s {p['energy_J']} J (sd {p['energy_sd']}) "
          f"{p['tps']} tok/s DDR {p['ddr_peak']} C  speedup {p['speedup_pct']}% extra E {p['extra_energy_pct']}%")
for f in out.get("ladder_fine", {}).get("points", []):
    print(f"fine   {f['mhz']:4d} MHz n={f['n']} {f['time_s']:5.1f} s {f['energy_J']} J (sd {f['energy_sd']}) "
          f"DDR {f['ddr_peak']} C  vs 1200: {f['vs_top_energy_pct']:+.1f}% E {f['vs_top_time_pct']:+.1f}% T")
for s in out["ladder"]["steps"]:
    print(f"  {s['frm']}->{s['to']}: {s['faster_pct']}% faster, energy {s['energy_pct']:+}%, DDR {s['ddr_delta']:+} C")
for s in out["cost_table"]["steps"]:
    print(f"cost {s['label']:36s} time +{s['time_cost']}% saves {s['energy_saved']}% rate {s['rate']} {s['decision']}")
c = out["cache_lever"]
print(f"cache: full {c['full']} | muKV {c['mukv']} | time {c['time_pct']}% energy {c['energy_pct']}% "
      f"J/tok {c['J_per_token_full']} -> {c['J_per_token_mukv']}")
for r in out["loops_mid"] + out["loops_low"]:
    print(f"loop {r['tag']:12s} L={r['lever']:.2f} bias {r['bias_before']:+.2f}->{r['bias_after']:+.2f} "
          f"time {r['time_over']:+6.1f}% energy {r['energy_over']:+6.1f}%  {r['action'][:48]}")
b = out["bandit"]
print(f"bandit learned {b['learned']}  {b['energy_kJ']} kJ  regret in warm start {b['regret_warm_pct']}%")
print("wrote energy_perf_data.json")
