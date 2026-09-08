# Watchdog v2 Upgrade Note — Multi-Sensor Preempt-Throttle

Status: host-side edit only. Takes effect on the **next launcher push to the
phone** (i.e. after the currently running K=1024 sweep finishes, ~28h from now).
The launcher already on the device is untouched and the K=1024 sweep continues
to use the v1 (DDR-only) watchdog as it was launched with.

Edited file:
- `EndurKV/scripts/android/phone_wave11_eval.sh` :: `start_watchdog()`

After this change, every policy returning true from `needs_watchdog()` uses the
v2 multi-sensor watchdog by default:

- `v1_fa2_stack`
- `v1_fa2_hybrid`
- `v1_fa2_f16`
- `v1_entropy_stack`
- `v1_predictive_stack`
- `endurkv_optimal`

Previously only `endurkv_optimal` was opted into v2; all others used v1.

---

## 1. Why v2 is better — skin leads DDR by ~13 minutes

Empirical finding from
`figures/master_tables/THROTTLE_TRIGGER_EMPIRICAL.md`: in every measured
throttle event on the OnePlus 15 (Snapdragon 8 Elite), the front-skin sensor
(`shell_front_temp_mc`) crosses its trip ~13 minutes BEFORE LPDDR5
(`ddr_temp_mc`) crosses 65 deg C. The kernel cpufreq cliff (883 MHz hard cap)
fires shortly after skin trips, well before DDR reaches the v1 threshold.

The v1 watchdog only watches DDR. It therefore reacts *after* the kernel has
already capped CPU frequency, which is too late — the throttle has already
distorted the measurement. v2 reacts to whichever sensor crosses first
(skin/CPU/battery/DDR), so it pre-empts the kernel cliff instead of chasing it.

## 2. New empirical thresholds (deg C: warn / cut)

| Sensor   | Warn | Cut  | Source trip | Margin below trip |
|----------|------|------|-------------|-------------------|
| DDR      | 51.7 | 56.7 | 65.0        | 8.3               |
| CPU big  | 56.2 | 61.2 | 95.0 (configured, well above measured) | derived from measured range |
| Skin     | 37.7 | 42.7 | 43.0        | 0.3               |
| Battery  | 36.8 | 39.8 | 45.0        | 5.2               |

Warn = soft cap (raises a flag for the runtime); Cut = hard preempt (the
watchdog forces `scaling_max_freq` down to clear the condition before the
kernel does so unilaterally). Thresholds were derived empirically from
the throttle-event runs documented in `THROTTLE_TRIGGER_EMPIRICAL.md` —
each warn sits a few deg below the first crossing observed in the wild,
each cut is at the observed crossing.

## 3. How to fall back to v1 (legacy DDR-only)

Set the env var **before** invoking the launcher:

```sh
WATCHDOG_VERSION=v1 sh scripts/android/phone_wave11_eval.sh ...
```

Any value other than `v1` (including unset) selects v2. The log line emitted
at watchdog start (`starting preempt-throttle watchdog v<N> (root) ...`)
records which version is actually running, so post-hoc audits can tell
from `watchdog.log` which script governed any given cell.

## 4. Why we did NOT push to phone now

The K=1024 sweep is **live on device** and was launched with the v1 watchdog.
Pushing a new launcher mid-sweep would:

- Risk the on-device shell re-sourcing functions mid-run (depending on how
  the wrapper invoked the script), which could swap watchdogs in the middle
  of a cell and produce a mixed v1/v2 trace that is unanalyzable.
- At minimum, contaminate the per-cell `watchdog.log` provenance — some cells
  would be v1, others v2, with no single switchover point.
- Worst case, race-conflict with the running `preempt_throttle_watchdog.sh`
  process if the v2 script were launched while v1 was still active (two
  watchdogs fighting `scaling_max_freq`).

Host-side edit only is therefore the safe move. The K=1024 sweep finishes
the way it started: pure v1, internally consistent.

## 5. When this takes effect

**Next launcher push to the phone**, i.e. after the K=1024 sweep completes
(~28h from now, 2026-06-09 local). At that point the operator does the normal
`adb push` of `phone_wave11_eval.sh` to the device staging dir, and from that
push onward every watchdog-bearing policy runs under v2 by default.

The v1 script (`preempt_throttle_watchdog.sh`) remains on the device for
the `WATCHDOG_VERSION=v1` fallback path and for re-running comparison cells
against the legacy behavior if needed.
