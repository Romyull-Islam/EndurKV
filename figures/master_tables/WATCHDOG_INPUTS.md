# Watchdog & Sensor Inputs — Architectural Audit

**Files inspected**
- `/home/mislam22/EndurKV_workspace/EndurKV/scripts/android/preempt_throttle_watchdog.sh`
- `/home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh`

---

## 1. What the watchdog actually reads

`preempt_throttle_watchdog.sh` is **strictly single-sensor, DDR-only**.

| Aspect | Value |
|---|---|
| Sensor polled | `${DDR_ZONE}/temp` (default `/sys/class/thermal/thermal_zone47`, divided by 1000 for degC) |
| Other inputs | **None.** No mem_avail, no cool_state, no freq feedback, no SoC/skin/CPU temps |
| Sampling rate | **0.5 Hz** (`sleep 2` between iterations) |
| Actuator | Writes `scaling_max_freq` on **cpu6 and cpu7** only (the big cores) |
| State variable | Single integer `TIER` ∈ {0,1,2,3} |
| Stop condition | Presence of sentinel file `$STOP` |
| Restore on exit | Restores `F_MAX=1632000 kHz` |

### Tier table (state machine)

| Tier | Name | f_max (kHz) | Enter when DDR ≥ (degC) | Exit-up | Exit-down (hysteresis) |
|---|---|---|---|---|---|
| 0 | MAX  | 1,632,000 | — (initial) | T_HIGH_UP = 58 -> tier1 | — |
| 1 | HIGH | 1,497,600 | DDR ≥ 58 | T_MED_UP = 62 -> tier2 | DDR ≤ T_HIGH_DN = 55 -> tier0 |
| 2 | MED  | 1,267,200 | DDR ≥ 62 | T_LOW_UP = 65 -> tier3  | DDR ≤ T_MED_DN  = 59 -> tier1 |
| 3 | LOW  | 1,017,600 | DDR ≥ 65 | — (floor) | DDR ≤ T_LOW_DN  = 62 -> tier2 |

Hysteresis: 3 degC band between up- and down-thresholds (58/55, 62/59, 65/62). This is what produces the "smooth glide" instead of an oscillation.

### Per-tier action (literal)

```sh
set_freq() {
    local f=$1
    for c in cpu6 cpu7; do
        echo "$f" > "/sys/devices/system/cpu/$c/cpufreq/scaling_max_freq" 2>/dev/null
    done
}
```

That is the **only** actuation. No CPU pinning, no governor change, no memory pressure response, no GPU/NPU touch, no I/O throttle.

---

## 2. What the sampler logs (separately)

`sample_sensors.sh` runs at **10 Hz** (configurable; default `HZ=10`, `sleep 0.1`) and is the *observability* path. It feeds the post-hoc analysis (`host_throttle_analyzer.py`, `host_plot_watchdog_actions.py`, etc.), but the watchdog never reads its output.

The sampler captures, per row:

| Group | Signals |
|---|---|
| Thermal (all zones, ~98 of them) | every `/sys/class/thermal/thermal_zone*/temp` with the zone `type` as column name -> includes DDR, SoC, GPU, modem, skin, battery, charger thermals |
| Block I/O | `/sys/block/{sd*,mmcblk*,sda}/stat` cols 5..8 (write_completed, merged, sectors_w, ms_w) |
| Memory | `MemTotal, MemFree, MemAvailable` from `/proc/meminfo` |
| PSI | `cpu/memory/io` avg10 (usually unreadable on Android 16) |
| VM | `pswpout, pgmajfault, pgpgout, pswpin` from `/proc/vmstat` |
| Probe I/O | PID of `entropy_probe`/`attention_probe`/`prune_probe`, plus `write_bytes`/`read_bytes` from `/proc/$pid/io` |
| CPU | `cpu*/cpufreq/scaling_cur_freq` per core |
| Cooling | `cooling_device*/cur_state` for every `cpufreq-cpuN` cooler |
| GPU | `/sys/class/kgsl/kgsl-3d0/gpubusy` (busy_us, total_us) |
| Battery (dumpsys, 1 Hz) | Battery current, Charger voltage, PhoneTemp |
| Power rail (10 Hz, sysfs) | `usb/online`, `usb/current_now`, `usb/voltage_now`, `battery/charge_counter`, `battery/power_now` |

So **the sampler sees everything; the watchdog sees one temperature**.

---

## 3. Architectural design decision

The current split is a deliberate **decouple control from telemetry**:

- **Watchdog = minimal feedback controller.** One scalar input, one scalar output, 0.5 Hz, ~10 lines of decision logic. Cheap (negligible CPU, runs as a sleep loop), predictable, easy to reason about, deterministic enough to put into a paper. DDR was chosen because in the OnePlus-15 / SM8650-class platform the DDR zone is the *upstream* thermal trip that the kernel hits *first* during long KV-cache decode — preempting it before the kernel mitigation kicks in is the entire point of Track-2.
- **Sampler = exhaustive logger.** 10 Hz, ~100 columns, post-hoc fusion in Python. Pays the cost only on disk and during analysis; never in the control loop.

This is a standard "skinny controller / fat logger" pattern. It is intentionally **not** multivariable.

### Strengths of the current design
- Minimal hot-path overhead -> no measurement of the watchdog itself perturbing the workload.
- Trivially reproducible: 4 thresholds + 4 frequencies fully specify policy.
- Single-sensor causal claim is clean: "DDR temp crosses 58 -> we cap cpu6/7" — straightforward to defend.

### Weaknesses for the novelty argument
- The paper claims a "preemptive thermal mitigation" controller, but a one-zone bang-bang-with-hysteresis is **not novel as a control algorithm**. The novelty has to come from *what* is being thermally controlled (KV-cache-driven memory traffic), not from the controller itself.
- DDR-only ignores cases where SoC/skin trips fire before DDR (which the sampler logs already show happens on Phi-3 and during sustained prefill). On those runs the watchdog never engages and Track-2 looks like a no-op.
- No fusion with `mem_avail` or `cool_state` means the watchdog cannot distinguish "hot because of KV memory traffic" (the failure mode we care about) from "hot because of a background process". The sampler has the data to disambiguate, the controller does not use it.

---

## 4. Should it be expanded to multi-sensor?

**Short answer: yes, but carefully — and keep DDR as the dominant input.**

Recommended minimal expansion (still keeps the policy paper-defensible):

1. **Add SoC and CPU big-cluster temps as `max()` inputs**, not independent state machines:
   ```
   T_eff = max(T_DDR, alpha_soc * T_SoC, alpha_cpu * T_CPU_big)
   ```
   with `alpha_*` chosen so DDR remains the controlling sensor in the regime we already validated, but SoC/CPU can preempt when DDR lags. This is one extra line in `read_ddr_c` and zero new tiers — keeps the state machine identical.

2. **Add a `mem_avail` guard as a tier-skip condition**: if `MemAvailable < threshold` AND `T_eff` is rising, jump directly to tier 2 instead of climbing through tier 1. This couples KV-driven memory pressure to the thermal action — that *is* a novel control input in the mobile-LLM literature, because no prior work (PowerInfer-2, MLC-LLM, llama.cpp Android) couples memory headroom to thermal capping.

3. **Do NOT add `cool_state` as an input.** It is a *kernel output* — the kernel's own thermal mitigation decision. Reading it back would create a feedback loop with the very mitigation we are trying to preempt.

4. Keep the 0.5 Hz cadence. The thermal time constant of the DDR die on this SoC is in the multi-second range; faster sampling adds noise, not signal.

### Where the novelty lands

If the expansion is framed as **"memory-pressure-aware preemptive thermal capping for on-device LLM decode"**, the contribution is:

- couples a *workload-aware* signal (`mem_avail`, `pswpout` rate, probe write_bytes) to the *thermal* control loop;
- preempts the kernel's per-zone mitigation by acting on the **fusion** of DDR + SoC + memory pressure;
- the existing single-sensor watchdog becomes the **ablation baseline** ("DDR-only") — which strengthens, rather than weakens, the empirical story because all the existing logs already contain the multi-sensor data, so the ablation is free.

### Concrete recommendation

- Keep `preempt_throttle_watchdog.sh` as the **DDR-only baseline** (rename to `..._ddr_only.sh` for clarity).
- Add `preempt_throttle_watchdog_v2.sh` with `T_eff = max(T_DDR, T_SoC, T_CPU_big)` and a `mem_avail` skip rule.
- Re-run a small subset of waves (Wave 9/10/11 thermal stacks) with v2 to show the multi-sensor variant catches thermal events the DDR-only one misses, then frame the paper claim around v2 with v1 as the ablation.

---

## 5. Bottom line

- **Watchdog today: DDR-only, 0.5 Hz, single-input bang-bang-with-hysteresis on cpu6/cpu7 `scaling_max_freq`.**
- **Sampler: everything else, 10 Hz, observation-only.**
- **For the novelty argument, expanding the watchdog to fuse DDR + SoC + CPU + `mem_avail` is the right move**, and it can be done without invalidating any existing data — the sampler logs already contain the inputs needed to replay/validate the multi-sensor policy offline before committing to a live re-run.
