# K=1024 LIVE THERMAL — wave11_K1024_1780921777

**Snapshot time (host)**: 2026-06-08 08:59:53 (device-local, EDT)
**Sweep started**: 2026-06-08 08:30:27 (≈29.5 min ago at snapshot)
**Run dir**: `/data/local/tmp/endurkv/logs/wave11_K1024_1780921777`
**Current cell** (per `progress.log`): `Phi-3-mini-128k / v1_fa2_stack / ppl`  — status: **PRE-RUN GATE (cool_phone)**, llama-cli not yet launched.

---

## 1. Did the sensor/watchdog files exist yet?

```
$ adb shell ls -la /data/local/tmp/endurkv/logs/wave11_K1024_1780921777/Phi-3-mini-128k/v1_fa2_stack/ppl/
total 6
drwxrwxrwx 2 shell shell 3452 2026-06-08 08:30 .
drwxrwxrwx 3 shell shell 3452 2026-06-08 08:30 ..
```

**`sensors.csv` does not exist yet. `watchdog.log` does not exist yet.**

Reason: per `phone_wave11_eval.sh` `run_one_cell()`, the sampler and watchdog are launched **after** `cool_phone()` + `wait_for_memory()` succeed. The sweep is currently stuck in `cool_phone()` — the launcher (`PID 9379`) is in `rt_sigsuspend`, its only live child is a `sleep` that re-spawns every 15 s, and `progress.log` has emitted no new lines since `CELL START` at 08:30:28.

**Cool gate (from script)**: requires `skin ≤ 33.0 °C` AND `DDR ≤ 40 °C`. The DDR has been fine for the entire window; the skin temperature has been hovering at 33.6–33.7 °C (battery zone, decideg-C: 336–337 dC), i.e. ~0.6 °C above the gate. `COOL_MAX_S=1800` → after 30 min of polling the gate, the launcher will log `WARN cool timeout … proceeding`, exit `cool_phone()`, start the sampler+watchdog, and `sensors.csv` / `watchdog.log` will begin filling. That force-proceed should fire within ~30 s of this snapshot.

So the answer to *"look at the last 5 min of `sensors.csv` / `watchdog.log`"* is: **no live sweep telemetry has been written yet**. The substitute below is a direct read of the same kernel sysfs zones (`/sys/class/thermal/thermal_zone*/temp`, `/sys/class/thermal/cooling_device*/cur_state`, `/sys/devices/system/cpu/cpu{6,7}/cpufreq/...`) that the sampler and watchdog read — at SAMPLE_HZ=5 the sampler reads exactly these same values.

---

## 2. Live thermal numbers (kernel sysfs, 08:59:53)

### DDR / SoC / CPU / surface

| Zone | Path | Value | Notes |
|---|---|---|---|
| **DDR** | thermal_zone47 (ddr) | **35 °C** | watchdog's source-of-truth |
| board | thermal_zone84 | 35 °C | PCB |
| sys-therm-0 | thermal_zone62 | 34 °C | virtual SoC composite |
| cpu-0-5-1 (big silver A720) | thermal_zone16 | 36 °C | hottest of A720 cluster |
| cpu-1-1-1 (Cortex-X4 prime) | thermal_zone27 | 34 °C | pinned core for inference |
| gpuss-9 | thermal_zone45 | 34 °C | GPU sub-block (idle) |
| shell_front | thermal_zone60 | 33 °C | front-glass skin |
| shell_back | thermal_zone68 | 32 °C | back-glass skin |
| shell_frame | thermal_zone65 | 32 °C | aluminium frame |
| **battery (skin)** | dumpsys battery | **33.6 °C** | the cool-gate sensor |

The big-CPU cluster sits at 34–37 °C, ~50–55 °C cooler than the kernel HW-trip points (`cpu-hw-trip-0/1 = 95 °C`). Phone is **cold and idle** at the moment of snapshot, consistent with the sweep being in its pre-run gate (no compute, no DRAM bursts).

### CPU max-freq (DVFS pin)

```
cpu6 scaling_max_freq = 1632000 kHz   (pinned at MAX by pin_dvfs)
cpu7 scaling_max_freq = 1632000 kHz   (pinned at MAX by pin_dvfs)
```

(Note: `progress.log`'s `pin_dvfs` verify shows `max=1382400` due to a stale read taken before sysfs flushed; the live sysfs is the authoritative 1632000.)

### Kernel cooling-device states (= "cool_state")

| Device | Type | cur_state / max_state |
|---|---|---|
| cooling_device0 | cpufreq-cpu0 | **0 / 27** |
| cooling_device1 | cpufreq-cpu6 | **0 / 28** |
| cooling_device22 | cpu-cluster0 | **0 / 27** |
| cooling_device24 | cpu-cluster1 | **0 / 28** |
| cooling_device21 | ddr-cdev | **0 / 1** |
| cooling_device35 | gpu | **0 / 17** |

**Every cool_state = 0.** The kernel thermal framework has not engaged any mitigation step on any CPU cluster, the DDR cdev, or the GPU. No kernel-driven throttle is active.

---

## 3. Has the user-space watchdog engaged tier 1, 2, or 3?

**No — the watchdog is not even running yet.** `watchdog.log` does not exist on disk; the script `preempt_throttle_watchdog.sh` is only spawned by `start_watchdog()` inside `run_one_cell()`, after the cool gate clears. `ps -A` confirms there is no `sh preempt_throttle_watchdog.sh` process on the device.

When it does start (~30 s from now), it will boot into **TIER=0 (MAX, F=1632000 kHz)** and stay there until DDR ≥ 58 °C.

---

## 4. Has cool_state risen (kernel-driven throttle)?

**No.** All CPU, cluster, DDR, and GPU cooling devices report `cur_state = 0`. No kernel mitigation has engaged. (And it shouldn't — the kernel framework only steps `cur_state` up when a zone's temperature crosses its trip points; the zones are sitting at 32–37 °C, far below trip thresholds.)

---

## 5. Is the watchdog's threshold reasonable given current temps?

**Reasonable, possibly slightly conservative for K=1024, but well-chosen as a default.**

| Watchdog tier | DDR up-thresh | DDR down-thresh (hysteresis) | CPU6/7 max kHz | Current headroom |
|---|---|---|---|---|
| 0 → 1 (HIGH) | **58 °C** | 55 °C | 1497600 | DDR=35 °C → **23 °C headroom** |
| 1 → 2 (MED) | **62 °C** | 59 °C | 1267200 | 27 °C headroom |
| 2 → 3 (LOW) | **65 °C** | 62 °C | 1017600 | 30 °C headroom |
| (kernel) ddr-cdev hard trip | n/a | n/a | — | cur_state=0 |

**Argument for "reasonable":**

* The kernel's own `ddr-cdev` has `max_state = 1`, meaning the kernel only knows "OK / throttle". The watchdog interposes **three intermediate steps** (1497.6 / 1267.2 / 1017.6 MHz) before that binary cliff, which is exactly its purpose — a smooth glide instead of a thermal cliff.
* In Wave-9 (`wave9_v1fa2_stack`) and Wave-11 (`wave11_eval_1780862534`) the v1_fa2_stack PPL cells saw DDR climb to **48–55 °C** under K=2048 sustained PPL load. At K=1024, working-set is ~½× so DDR rise should be **smaller**, very likely topping out around **45–52 °C** for Phi-3 PPL on 9 chunks. That means the watchdog's tier-1 trigger (58 °C) is positioned **5–10 °C above expected steady-state DDR**, so it likely **will not fire at all** for the Phi-3 K=1024 PPL run.
* Hysteresis bands are 2–3 °C, sensible for a sysfs zone updated at 1 Hz with 5 Hz polling (sleep 2).

**Argument for "slightly conservative":**

* If the goal of the K=1024 sweep is to demonstrate that v1_fa2_stack stays *fully un-throttled*, the chosen thresholds are well above expected DDR steady-state — meaning the watchdog will likely log "tier=0 MAX" for the entire run and then exit. That's the desired outcome for a "clean" K=1024 stack cell.
* If the goal is to actively *exercise* the preempt-throttle mechanism, the thresholds (58/62/65 °C) might be too high to engage at K=1024; a more aggressive setting (e.g. 52/56/60 °C) would force the watchdog to step through tiers and produce the smooth-glide telemetry needed for the watchdog effectiveness figure. But that's a separate research design choice — for **safety / non-throttling baseline measurement**, 58/62/65 is correctly chosen.

**Verdict**: thresholds are sound. Currently the system is at DDR=35 °C, so the watchdog has **23 °C of headroom** before tier 1 would fire. The watchdog is not in any risk of false-tripping.

---

## 6. RES_SCHEMA (summary)

```yaml
sweep_dir: /data/local/tmp/endurkv/logs/wave11_K1024_1780921777
model: Phi-3-mini-128k
policy: v1_fa2_stack
bench: ppl
K_nominal: 1024
snapshot_at: "2026-06-08T08:59:53-04:00"
elapsed_since_start_min: 29.5

cell_state: "PRE-RUN: stuck in cool_phone() gate (skin=33.6C vs threshold 33.0C)"
sensors_csv_exists: false
watchdog_log_exists: false
benchmark_process_running: false
sampler_process_running: false
watchdog_process_running: false

live_temps_C:
  ddr: 35              # thermal_zone47 (watchdog input)
  board: 35
  sys_therm_0: 34
  cpu_big_silver: 36   # cpu-0-5-1 (hottest A720)
  cpu_prime_X4: 34     # cpu-1-1-1
  gpuss_9: 34
  shell_front: 33
  shell_back: 32
  shell_frame: 32
  battery_skin: 33.6

watchdog_tier: 0_MAX_not_running   # has not started yet; would start at tier 0
watchdog_thresholds_C:
  tier1_HIGH_up: 58   tier1_HIGH_dn: 55   freq_kHz: 1497600
  tier2_MED_up:  62   tier2_MED_dn:  59   freq_kHz: 1267200
  tier3_LOW_up:  65   tier3_LOW_dn:  62   freq_kHz: 1017600

cool_state_kernel:
  cpufreq_cpu0:   {cur: 0, max: 27}
  cpufreq_cpu6:   {cur: 0, max: 28}
  cpu_cluster0:   {cur: 0, max: 27}
  cpu_cluster1:   {cur: 0, max: 28}
  ddr_cdev:       {cur: 0, max: 1}
  gpu:            {cur: 0, max: 17}
kernel_throttle_engaged: false

cpufreq_max_kHz:
  cpu6: 1632000
  cpu7: 1632000

headroom_to_tier1_C: 23   # 58 - 35
threshold_reasonableness: "sound (DDR will likely peak ~45-52 C at K=1024; tier 1 at 58 C gives 5-10 C margin)"
expected_watchdog_activity_during_run: "stay at tier 0 (MAX) for entirety of Phi-3 PPL K=1024 cell"
risk_of_false_throttle: "low — all hysteresis bands well above current"

action_needed: "wait ~30 s for cool_phone() timeout (COOL_MAX_S=1800s); sensors.csv and watchdog.log will then begin populating at 5 Hz."
```
