#!/usr/bin/env python3
"""Online contextual bandit trained on the real phone, one pull = one real request.

Context : battery tier (healthy / mid / low), cycled, which sets the reward weights.
Arms    : GPU clock cap {1200, 902, 726} MHz with K held at 1024 (the GPU ladder).
Request : the 9737-token prompt, 1024 output tokens, Llama-3.2-1B on the GPU.
Reward  : r = w_q * 1  -  w_t * T / T_ref  -  w_e * E / E_ref, from the METERED energy (USB rail
          + battery pack, integrated from the 2 Hz sampler) and wall time of that pull.
          E_ref, T_ref are the table's full-performance prediction for this request shape.
Agent   : epsilon-greedy (eps 0.25) with a warm start (every arm once per context), running
          mean per (context, arm). State persists in state.json, so the run is resumable.
Protocol: cool gate (DDR <= 35 C, battery <= 33 C) before every pull, CPU caps pinned at
          1785.6 / 1497.6 MHz, taskset f0 nice -20 for the bench, GPU cap restored after
          each pull, caps and charging restored on exit.
Usage   : run_bandit_online.py [--pulls 24] [--eps 0.25] [--out /tmp/bandit_online]
"""
import argparse, json, os, random, subprocess, sys, time, statistics as st

sys.path.insert(0, "/home/mislam22/EndurKV_workspace/EndurKV/scripts")
from ea_proof_summary import load as load_cell

SERIAL = os.environ.get("ANDROID_SERIAL", "10.0.0.127:5555")
TIERS = {"healthy": (0.55, 0.40, 0.05), "mid": (0.35, 0.20, 0.45), "low": (0.20, 0.10, 0.70)}
ARMS = [1200, 902, 726]
E_REF, T_REF = 782.0, 142.4          # table prediction, GPU 1200 MHz, 9737 prompt + 1024 output
BIN = "/data/local/tmp/ukv"
MODEL = "/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
PROMPT = "/data/local/tmp/endurkv/corpora/prompt_12k.txt"
MU = ("--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 "
      "--snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024")
DEV = "/data/local/tmp/endurkv/logs/bandit"

def adb(cmd, timeout=120, su=False):
    full = f"su -c '{cmd}'" if su else cmd
    for attempt in range(3):
        try:
            r = subprocess.run(["adb", "-s", SERIAL, "shell", full], capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip()
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise
            time.sleep(10)
    return ""

def pull(remote, local):
    subprocess.run(["adb", "-s", SERIAL, "pull", remote, local], capture_output=True, timeout=300)

def log(msg, out):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(out, "log.txt"), "a") as f:
        f.write(line + "\n")

def settle(out):
    for _ in range(3):
        adb(". /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36", timeout=900, su=True)
        adb("prev=999; same=0; for i in $(seq 1 90); do d=$(($(cat /sys/class/thermal/thermal_zone47/temp)/100)); "
            "diff=$((d-prev)); [ $diff -lt 0 ] && diff=$((-diff)); if [ $diff -le 3 ]; then same=$((same+1)); else same=0; fi; "
            "[ $same -ge 3 ] && break; prev=$d; sleep 10; done", timeout=1000, su=True)
        r = adb("echo $(cat /sys/class/thermal/thermal_zone47/temp) $(cat /sys/class/thermal/thermal_zone93/temp)", su=True).split()
        try:
            ddr, batt = int(r[0]) // 1000, int(r[1]) // 1000
        except (ValueError, IndexError):
            continue
        log(f"    settle: ddr={ddr}C batt={batt}C", out)
        if ddr <= 35 and batt <= 33:
            return True
    return False

def pin_cpu():
    adb("echo 1785600 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; echo 1785600 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; "
        "echo 1497600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq; echo 1497600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq", su=True)

def restore():
    adb("echo 1996800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; echo 384000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; "
        "echo 2438400 > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq; echo 883200 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq; "
        "echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel; "
        "echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable", su=True)
    adb("pkill -f sample_sensors", su=True)        # last, on its own: it kills its own shell too

def run_pull(tag, mhz, out):
    """One real request at the given GPU cap. Returns (E_J, T_s, dict) or None."""
    d = os.path.join(out, "pulls", tag); os.makedirs(d, exist_ok=True)
    if not settle(out):
        log(f"    {tag}: phone would not cool; skipped", out); return None
    pin_cpu()
    adb(f"echo {mhz * 1000000} > /sys/class/kgsl/kgsl-3d0/max_gpuclk", su=True)
    adb(f"mkdir -p {DEV}")                       # as the shell user: the bench runs unprivileged and must write here
    adb(f"chmod 777 {DEV}", su=True)              # in case an earlier run created it as root
    # pkill -f matches the shell that carries it, so it must never share a command with the launch
    adb("pkill -f sample_sensors", su=True)
    adb(f"rm -f /data/local/tmp/bandit_{tag}.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/bandit_{tag}.csv --hz 2 >/dev/null 2>&1 &", su=True)
    time.sleep(2)
    if "bandit_" + tag not in adb(f"ls /data/local/tmp/bandit_{tag}.csv 2>/dev/null", su=True):
        log(f"    {tag}: sampler did not start", out)
    cmd = (f"cd {BIN} && LD_LIBRARY_PATH={BIN} taskset f0 nice -n -20 ./eviction_bench --model {MODEL} --prompt {PROMPT} "
           f"--prompt-id {tag} --eval-mode gen --max-tokens 1024 --ignore-eos --ctx-size 16384 --seed 42 --threads 4 "
           f"--n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 {MU} --n-batch 512 --n-ubatch 64 "
           f"--out-meta {DEV}/{tag}.json --out-gen {DEV}/{tag}.gen --out-csv /dev/null > /dev/null 2> {DEV}/{tag}.err")
    adb(cmd, timeout=1200)
    adb("echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk", su=True)
    adb("pkill -f sample_sensors", su=True)        # separate call: it kills its own shell too
    for f, name in ((f"{DEV}/{tag}.json", "meta.json"), (f"{DEV}/{tag}.err", "err"), (f"/data/local/tmp/bandit_{tag}.csv", "sensors.csv")):
        pull(f, os.path.join(d, name))
    if not os.path.exists(os.path.join(d, "meta.json")):
        log(f"    {tag}: no meta pulled; failed", out); return None
    x = load_cell(d); m = json.load(open(os.path.join(d, "meta.json")))
    E = x["prefill_J"] + x["decode_J"]; T = m["total_ms"] / 1000
    if not (E == E and E > 0):
        log(f"    {tag}: energy missing; failed", out); return None
    return E, T, dict(tps=x["tps"], gpu_mhz=x["gpu_mhz"], prefill_s=x["prefill_s"], ddr_peak=x["ddr"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pulls", type=int, default=24); ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--out", default="/tmp/bandit_online"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "pulls"), exist_ok=True)
    sp = os.path.join(a.out, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {"Q": {t: {str(m): [] for m in ARMS} for t in TIERS}, "history": []}
    rng = random.Random(a.seed + len(state["history"]))
    subprocess.run(["adb", "-s", SERIAL, "push", "/home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh", "/data/local/tmp/sample_sensors.sh"], capture_output=True, timeout=120)
    log(f"online bandit: {a.pulls} pulls, eps {a.eps}, arms {ARMS} MHz, contexts {list(TIERS)}, resuming at pull {len(state['history'])}", a.out)
    try:
        while len(state["history"]) < a.pulls:
            i = len(state["history"]); tier = list(TIERS)[i % 3]
            q = state["Q"][tier]
            untried = [m for m in ARMS if not q[str(m)]]
            if untried:
                mhz = untried[0]; why = "warm start"
            elif rng.random() < a.eps:
                mhz = rng.choice(ARMS); why = "explore"
            else:
                mhz = max(ARMS, key=lambda m: st.mean(q[str(m)])); why = "greedy"
            tag = f"pull{i:02d}_{tier}_{mhz}"
            log(f"pull {i}: context {tier}, arm {mhz} MHz ({why})", a.out)
            res = run_pull(tag, mhz, a.out)
            if res is None:
                state["history"].append(dict(i=i, tier=tier, mhz=mhz, failed=True)); json.dump(state, open(sp, "w"), indent=1); continue
            E, T, extra = res
            wq, wt, we = TIERS[tier]
            r = wq * 1.0 - wt * T / T_REF - we * E / E_REF
            q[str(mhz)].append(r)
            state["history"].append(dict(i=i, tier=tier, mhz=mhz, why=why, E=E, T=T, r=r, **extra))
            json.dump(state, open(sp, "w"), indent=1)
            means = {m: (st.mean(q[str(m)]) if q[str(m)] else None) for m in ARMS}
            log(f"    measured E={E:.0f} J T={T:.0f} s (gpu {extra['gpu_mhz']:.0f} MHz, {extra['tps']:.1f} tok/s) -> r={r:+.3f}; "
                f"Q[{tier}] = " + ", ".join(f"{m}:{(f'{v:+.3f}' if v is not None else '-')}" for m, v in means.items()), a.out)
    finally:
        restore()
    log("done. learned greedy policy:", a.out)
    for t in TIERS:
        q = state["Q"][t]; means = {m: st.mean(q[str(m)]) for m in ARMS if q[str(m)]}
        if means:
            best = max(means, key=means.get)
            log(f"  {t:8s}: {best} MHz  (" + ", ".join(f"{m} MHz {v:+.3f} n={len(q[str(m)])}" for m, v in means.items()) + ")", a.out)
    log("BANDIT_DONE", a.out)

if __name__ == "__main__":
    main()
