#!/usr/bin/env python3
"""Guarded contextual bandit on the phone: run the bandit's plan only when its lower 95% bound
beats the rule plan's upper bound, explore plans that still might win, else use the rule's plan.
Plan costs are shared across battery tiers. Output stays outside /tmp, which clears at boot.
Usage: run_guarded_bandit.py [--requests 30] [--out ...]
"""
import argparse, csv, json, math, os, re, statistics as st, subprocess, sys, time

SERIAL = os.environ.get("ANDROID_SERIAL", "3C15B8003ZA00000")
ROOT = "/data/local/tmp/endurkv"
SCHED = f"{ROOT}/ukv_sched.sh"
PROMPT = f"{ROOT}/corpora/prompt_12k.txt"
TIERS = [("healthy", 80), ("mid", 40), ("low", 15)]
ARMS = ["gpu1200_k1024", "gpu1200d902_k1024", "gpu1200d726_k1024", "gpu902_k1024", "gpu726_k1024"]
E_REF, T_REF = 782.0, 142.4        # as in the first bandit, so rewards are on one scale
Z = 1.96                           # 95% bounds
N_WARM, N_MAX = 2, 6


def adb(cmd, timeout=180, su=False):
    full = f"su -c '{cmd}'" if su else cmd
    for attempt in range(3):
        try:
            r = subprocess.run(["adb", "-s", SERIAL, "shell", full], capture_output=True, text=True, timeout=timeout,
                               stdin=subprocess.DEVNULL)
            return (r.stdout + r.stderr).strip()
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise
            time.sleep(10)
    return ""


def log(msg, out):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    open(os.path.join(out, "log.txt"), "a").write(line + "\n")


def charging(on):
    adb(f"echo {1 if on else 0} > /sys/class/oplus_chg/battery/mmi_charging_enable", su=True)


def pin_cpu():
    adb("echo 1785600 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; echo 1785600 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; "
        "echo 1497600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq; echo 1497600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq", su=True)


def restore(out):
    adb("echo 1996800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; echo 384000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; "
        "echo 2438400 > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq; echo 883200 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq; "
        "echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel", su=True)
    charging(True)
    log("restored CPU caps, GPU cap and charging", out)


def settle(out):
    """Cool gate (DDR <= 35 C, battery <= 33 C), then charging off: the gate turns it back on."""
    for _ in range(3):
        adb(". /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36", timeout=1500, su=True)
        t = adb("echo $(cat /sys/class/thermal/thermal_zone47/temp) $(cat /sys/class/thermal/thermal_zone93/temp)", su=True).split()
        try:
            ddr, batt = int(t[-2]) // 1000, int(t[-1]) // 1000
        except (ValueError, IndexError):
            continue
        charging(False)
        log(f"    settle: ddr={ddr}C batt={batt}C, charging off", out)
        if ddr <= 35 and batt <= 33:
            return ddr, batt
    return None


def dry_run(soc):
    """The rule's plan and the weights at the current lever (the lever carries the loops' bias)."""
    txt = adb(f"cd {ROOT} && sh {SCHED} --prompt {PROMPT} --force-soc {soc} --force-status discharging "
              f"--max-tokens 1024 --ignore-eos --dry-run --tag gb_dry", su=True)
    plan = re.search(r"\[sched\] plan: (\S+)", txt)
    w = re.search(r"weights=\(q ([\d.]+), t ([\d.]+), e ([\d.]+)\)", txt)
    lev = re.search(r"lever=([\d.]+)", txt)
    if not (plan and w):
        raise RuntimeError("dry run failed:\n" + txt[-600:])
    return plan.group(1), tuple(float(x) for x in w.groups()), float(lev.group(1)) if lev else None


def estimate(samples, w):
    """(reward estimate, standard error) from a plan's measured (E, T); None below two samples."""
    if len(samples) < 2:
        return None
    wq, wt, we = w
    E = [s[0] for s in samples]; T = [s[1] for s in samples]; n = len(samples)
    r = wq - wt * st.mean(T) / T_REF - we * st.mean(E) / E_REF
    se = math.sqrt((wt / T_REF) ** 2 * st.variance(T) / n + (we / E_REF) ** 2 * st.variance(E) / n)
    return r, se


def decide(stats, rule, w):
    warm = [a for a in ARMS if len(stats[a]) < N_WARM]
    if warm:
        a = min(warm, key=lambda x: (len(stats[x]), ARMS.index(x)))
        return a, "explore: warm-up", {}
    est = {a: estimate(stats[a], w) for a in ARMS}
    lo = {a: est[a][0] - Z * est[a][1] for a in ARMS}
    hi = {a: est[a][0] + Z * est[a][1] for a in ARMS}
    best = max(ARMS, key=lambda a: est[a][0])
    info = {a: dict(r=round(est[a][0], 4), lo=round(lo[a], 4), hi=round(hi[a], 4), n=len(stats[a])) for a in ARMS}
    if best != rule and lo[best] > hi[rule]:
        return best, "go: bandit beats the rule with confidence", info
    open_ = [a for a in ARMS if a != rule and hi[a] > lo[rule] and len(stats[a]) < N_MAX]
    if open_:
        a = max(open_, key=lambda x: hi[x])
        return a, "explore: could still beat the rule", info
    return rule, "fallback: rule's plan", info


def kill_strays():
    """Only one request may run: a second scheduler instance pkills the first one's power sampler."""
    # bracketed patterns, so pkill cannot match (and kill) the shell that runs it
    for pat in ("ukv_sched.s[h]", "eviction_benc[h]", "sample_sensor[s]"):
        adb(f"pkill -f {pat}", su=True)
    time.sleep(2)


def running(tag):
    return "run" in adb(f"pgrep -f 'ukv_sched.s[h].*--tag {tag} ' >/dev/null && echo run || echo done")


def run_request(tag, soc, plan, rule, out):
    kill_strays()
    adb(f"rm -rf {ROOT}/sched/{tag}; sed -i '/tag={tag} /d' {ROOT}/ukv_sched.log", su=True)
    arg = f"--plan {plan}" if plan != rule else ""
    # Launch fully detached and never re-send it, since a retry would start a second copy.
    # After a timeout, look for the running request instead.
    cmd = (f"su -c 'mkdir -p {ROOT}/sched; cd {ROOT} && setsid nohup sh {SCHED} --prompt {PROMPT} --tag {tag} --ignore-eos "
           f"--max-tokens 1024 --force-soc {soc} --force-status discharging {arg} > {ROOT}/sched/{tag}.stderr 2>&1 < /dev/null &'")
    try:
        subprocess.run(["adb", "-s", SERIAL, "shell", cmd], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        time.sleep(5)
        if not running(tag):
            return None
    for _ in range(150):                      # up to 25 minutes
        time.sleep(10)
        if not running(tag):
            break
    time.sleep(3)
    line = adb(f"grep 'tag={tag} ' {ROOT}/ukv_sched.log | tail -1", su=True)
    stderr = adb(f"cat {ROOT}/sched/{tag}.stderr", su=True)
    d = os.path.join(out, "requests", tag); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "sched_log.txt"), "w").write(line + "\n")
    open(os.path.join(d, "stderr.txt"), "w").write(stderr + "\n")
    adb(f"echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel", su=True)
    kv = dict(re.findall(r"(\w+)=(\S+)", line))
    if "meas_J" not in kv:
        return None
    act = line.split(";")[-1].strip()
    return dict(plan_ran=kv.get("plan"), sched_src=kv.get("src"), sched_rule=kv.get("rule"), E=float(kv["meas_J"]),
                T=float(kv["meas_s"]), pred_E=float(kv["pred_J"]), pred_T=float(kv["pred_s"]),
                ebud=float(kv["ebud"]), tbud=float(kv["tbud"]), bias=kv.get("bias"), lever=kv.get("lever"),
                tps=float(kv.get("tps", 0) or 0), meter_cov=float(kv.get("meter_cov", 0) or 0), loop=act)


def meter_ok(res):
    """Reject a request whose power log did not cover it, or whose energy is implausible for its plan."""
    ratio = res["E"] / res["pred_E"] if res["pred_E"] else 0
    return res["meter_cov"] >= 0.9 and 0.5 <= ratio <= 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=30)
    ap.add_argument("--out", default="/home/mislam22/EndurKV_workspace/campaigns/guarded_bandit_20260921")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sp = os.path.join(a.out, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {"stats": {x: [] for x in ARMS}, "history": []}
    log(f"guarded bandit: {a.requests} requests, arms {ARMS}, serial {SERIAL}, resuming at {len(state['history'])}", a.out)
    if not state["history"]:
        # a clean start, as in the loop experiment: the seed cost table and no lever bias
        adb(f"cp {ROOT}/bak_20260921/ukv_sched_table.txt {ROOT}/ukv_sched_table.txt; rm -f {ROOT}/ukv_lever_bias.txt "
            f"{ROOT}/tables/Llama-3.2-1B-Instruct-Q4_K_M.txt {ROOT}/tables/Llama-3.2-1B-Instruct-Q4_K_M.bias.txt", su=True)
        log("scheduler reset to its seed table with no lever bias", a.out)
    try:
        while len(state["history"]) < a.requests:
            i = len(state["history"]); tier, soc = TIERS[i % 3]
            tag = f"gb{i:02d}_{tier}"
            log(f"request {i}: tier {tier} (SoC {soc}%)", a.out)
            temps = settle(a.out)
            if temps is None:
                log("    phone would not cool; waiting 10 min", a.out); time.sleep(600); continue
            pin_cpu()
            rule, w, lever = dry_run(soc)
            plan, why, info = decide(state["stats"], rule, w)
            log(f"    rule's plan {rule}, weights {w}, lever {lever}; -> {plan} ({why})", a.out)
            res = None
            for attempt in range(3):
                res = run_request(tag if attempt == 0 else f"{tag}r{attempt}", soc, plan, rule, a.out)
                if res is not None and meter_ok(res):
                    break
                log(f"    measurement rejected (coverage {res['meter_cov'] if res else 'none'}, "
                    f"E {res['E'] if res else '-'} J vs predicted {res['pred_E'] if res else '-'} J); re-running", a.out)
                res = None
                settle(a.out); pin_cpu()
            if res is None:
                log(f"    {tag}: no scheduler log line; recorded as failed", a.out)
                state["history"].append(dict(i=i, tier=tier, plan=plan, rule=rule, why=why, failed=True))
                json.dump(state, open(sp, "w"), indent=1); continue
            ran = res["plan_ran"]
            if ran in state["stats"]:
                state["stats"][ran].append([res["E"], res["T"]])
            wq, wt, we = w
            r = wq - wt * res["T"] / T_REF - we * res["E"] / E_REF
            rec = dict(i=i, tier=tier, soc=soc, rule=rule, plan=plan, why=why, weights=w, lever_decided=lever,
                       reward=round(r, 4), ddr_start=temps[0], batt_start=temps[1], estimates=info, **res)
            state["history"].append(rec)
            json.dump(state, open(sp, "w"), indent=1)
            log(f"    ran {ran} (src {res['sched_src']}): E={res['E']:.0f} J T={res['T']:.0f} s "
                f"(budgets {res['ebud']:.0f} J, {res['tbud']:.0f} s) r={r:+.3f}; loops: {res['loop']}", a.out)
    finally:
        kill_strays()
        restore(a.out)
        adb(f"cp {ROOT}/bak_20260921/ukv_sched_table.txt {ROOT}/ukv_sched_table.txt; "
            f"rm -f {ROOT}/ukv_lever_bias.txt; cp {ROOT}/bak_20260921/ukv_lever_bias.txt {ROOT}/ukv_lever_bias.txt 2>/dev/null; "
            f"cp -r {ROOT}/bak_20260921/tables/. {ROOT}/tables/", su=True)
        log("scheduler table and bias restored from bak_20260921", a.out)
    open(os.path.join(a.out, "DONE"), "w").write("done\n")
    log("GUARDED_BANDIT_DONE", a.out)


if __name__ == "__main__":
    main()
