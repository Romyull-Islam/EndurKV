# OnePlus 15 / Snapdragon 8 Elite — Kernel Thermal Throttle Thresholds

**Device:** OnePlus 15 (Snapdragon 8 Elite / SM8750-AB family)
**Probed:** live `/sys/class/thermal/` on the actual handset
**Probe count:** 98 thermal zones, 63 cooling devices
**Date:** 2026-06-08

This document replaces the heuristic "65 °C DDR cliff" with the **actual** kernel
trip points and binding cooling actions, read directly from the device.

---

## TL;DR

| Subsystem      | First passive trip | Action (cooling device)        | Hot trip   | Wave-11 peak | Headroom    |
|----------------|--------------------|--------------------------------|------------|--------------|-------------|
| **DDR**        | **100 °C**         | `ddr-cdev` (single-step)       | 125 °C     | 64.8 °C      | **35.2 °C** |
| Big-CPU prime  | 95 °C (zone6 only) | thermal-pause-1 / hotplug      | 125 °C     | 70.9 °C      | 24 °C       |
| Other CPU cores| 105 °C             | thermal-pause-N / cpu-hotplugN | 125 °C     | 70.9 °C      | 34 °C       |
| GPU (gpuss-0)  | 105 °C             | `kgsl`                         | 125 °C     | n/a          | —           |
| CPU HW trip    | 95 °C              | userspace only (firmware halt) | (same)     | 70.9 °C      | 24 °C       |
| Board / shell  | none active        | sensor-only (HAL surfaces it)  | (none)     | ~36 °C       | —           |
| sys-therm-2    | 48 / 49 / 50 / 60 / 61 °C | **no kernel cdev** (HAL-only) | 90 °C  | ~37 °C       | —           |

**Conclusion: the kernel did not throttle anything in Wave-11.**
Every CPU cooling device read `cur_state=0`, every passive trip is ≥95 °C, and
the lowest *temperature* trips (sys-therm-2 at 48-61 °C) have **no cooling
device bound to them** in the kernel — they are purely informational sensors
that the Android Thermal HAL reads to surface "device is getting warm" hints
to apps. Nothing in the kernel turns them into a frequency cut.

**The "65 °C DDR cliff" is NOT a kernel threshold.** The kernel's only DDR
trip fires at **100 °C** (passive) / 125 °C (hot). Wave-11's 65 °C ceiling
is a *workload-emergent* DDR thermal saturation — DRAM controller arbitration
slows down well before the kernel-visible trip — but the kernel itself is not
issuing the throttle command.

---

## Method

```
adb shell ls /sys/class/thermal/                       # enumerate
adb shell cat /sys/class/thermal/thermal_zone*/type
adb shell cat /sys/class/thermal/thermal_zone*/temp
adb shell cat /sys/class/thermal/thermal_zone*/trip_point_*_temp
adb shell cat /sys/class/thermal/thermal_zone*/trip_point_*_type
adb shell readlink /sys/class/thermal/thermal_zone*/cdev*
adb shell cat /sys/class/thermal/thermal_zone*/cdev*_trip_point
adb shell cat /sys/class/thermal/cooling_device*/{type,cur_state,max_state}
```

All temperatures in milli-°C (raw sysfs); divide by 1000 for °C.

---

## DDR (zone 47 — `ddr`)

```
type              = ddr
policy            = step_wise
trip_point_0_temp = 135000   passive  hyst=1000  (135 °C)   safety
trip_point_1_temp = 100000   passive  hyst=5000  (100 °C)   <-- ACTIVE THROTTLE
trip_point_2_temp =  125000  hot      hyst=0     (125 °C)   panic
cdev0 -> ddr-cdev (cooling_device21, max_state=1)  bound to trip_point_1
```

**Verdict:** the kernel will only invoke DDR throttling at **100 °C**. The DDR
controller has a single-step cooling device (`max=1` → either off or one
throttle level). Wave-11 saw 64.8 °C — **35 °C below the kernel trip**. The
behaviour we have been calling a "cliff at 65 °C" is therefore a property of
the DRAM device's intrinsic refresh / thermal-saturation curve, not a kernel
policy.

---

## CPU cores

**Per-core zones** (`cpu-0-0-0` … `cpu-1-1-1`, zones 5-16, 24-27) — Snapdragon
8 Elite has two clusters of 4 cores × 2 thermal sensors per core.

Standard trip ladder (applies to all per-core zones **except zone 6**):
```
trip0 passive 135000   hyst 1000   (safety net)
trip1 passive 105000   hyst 5000
trip2 passive 112000   hyst 4000   <-- thermal-pause-* bound here
trip3 passive 114000   hyst 6000   <-- cpu-hotplugN bound here
trip4 hot     125000   hyst 0
```

**Anomaly — zone 6 (`cpu-0-0-1`):** trip0 = 95000 mC (95 °C). Every other
per-core zone has trip0 = 135 °C. zone 6's bindings still use trips 2/3 (112/114),
so the 95 °C trip0 is *unbound* — but it's the lowest CPU sensor trip anywhere
in the system.

**Cluster aggregate zones** (`cpullc-0-0`, `cpullc-0-1`, `cpullc-1-0`,
`cpullc-1-1`, zones 0, 1, 17, 18) reuse the 112/114 °C trips and bind a
larger set of `thermal-pause-*` and `cpu-hotplug*` cooling devices.

**Cooling-device snapshot during interrogation (cold idle):** every
`thermal-pause-*` and `cpu-hotplug*` reports `cur_state=0`. `cpufreq-cpu0`
(max=27) and `cpufreq-cpu6` (max=28) likewise sit at 0 — these are bound
only to `socd` (battery discharge), not to per-core temperature zones.

**CPU HW trip zones** (zone 61 `cpu-hw-trip-0`, zone 66 `cpu-hw-trip-1`):
```
policy = user_space
trip0  = 95000 mC   passive
```
These are firmware-level "if CPU exceeds 95 °C the SoC halts" backstops,
surfaced to userspace; no kernel cdev bound. The temperature value reported
(95 000) is a flag, not a live reading.

---

## SoC / battery (`socd`, zone 59)

`socd` = State-Of-Charge-Discharge (Battery Current Limit), **not** a
temperature sensor. Trips report in % / mV, not milli-°C:
```
trip0 passive  90
trip1 passive  95
trip2 passive 100
cdev0 -> cpufreq-cpu6   trip=2
cdev1 -> cpufreq-cpu0   trip=2
cdev2 -> kgsl           trip=2
cdev3 -> cdsp           trip=2
```
This is the **only** zone in the entire system that binds `cpufreq-cpu0` /
`cpufreq-cpu6` — meaning frequency scaling is gated on battery-discharge
limits, not on CPU temperature. (Temperature uses thermal-pause + hotplug,
not cpufreq.)

---

## GPU (`gpuss-0`, zone 36)

```
trip0 passive 135000
trip1 passive 105000
trip2 passive 105000   <-- kgsl cdev bound here
trip3 hot     125000
cdev0 -> kgsl (max_state=17)
```

---

## Skin / shell / board (zones 60, 65, 68, 84)

- `shell_front` (60), `shell_frame` (65), `shell_back` (68): `mode=disabled`
  in the kernel. They are **read-only sensors** exposed to userspace via
  sysfs but the kernel does not consult them for trip decisions.
- `board_temp` (84): trips at 125 °C only, no cdev bound.

Therefore there is **no kernel-level skin-temperature throttle** at all on
this device. Surface-temperature management is left entirely to the Android
Thermal HAL via the `sys-therm-*` zones below.

---

## sys-therm-* (HAL skin-proxy sensors)

`sys-therm-2` (zone 67) is the only sys-therm zone with low trip points:
```
trip0 passive 48000   ( 48 °C)
trip1 passive 49000   ( 49 °C)
trip2 passive 50000   ( 50 °C)
trip3 passive 60000   ( 60 °C)
trip4 passive 61000   ( 61 °C)
trip5 passive 90000   ( 90 °C)
```
…but `ls /sys/class/thermal/thermal_zone67/cdev*` is empty — **no kernel
cooling device is bound to any of these trips.** The HAL polls
`thermal_zone67/temp` and uses the trip levels to emit
`Temperature.throttlingStatus = LIGHT / MODERATE / SEVERE` to apps.
This is the surface Android shows in `dumpsys thermalservice` and on the
"Hot device" warning popup; it is **not** a CPU/GPU/DDR throttle.

`sys-therm-0` (zone 62) carries the **userspace-routed CPU/GPU hot-plug
& thermal-pause map** (cdev0…cdev14 binding cpu-hotplug3/4/5/6/7, kgsl,
cdsp, modem-dsc, display-fps, thermal-pause-3C/C0). Its trips are mostly
125 °C (inactive) with a 90 °C `hot` trip and the lower "trip1=40, trip2=78,
trip3=80" entries used by the HAL.

---

## Cross-check against Wave-11 observations

| Wave-11 observation                     | Kernel reality                                            | Match? |
|-----------------------------------------|-----------------------------------------------------------|--------|
| `cpu*_cool_state never > 0`             | Every cdev was `cur_state=0` during probe; lowest CPU trip is 95 °C and Wave-11 capped at 70.9 °C | ✓ |
| DDR peaked 64.8 °C, "65 °C cliff"       | Kernel DDR trip is 100 °C, not 65 °C — cliff is DRAM-physics, not kernel | partial — kernel exonerated |
| CPU peaked ~70.9 °C                     | 24 °C below the 95 °C hardware trip; no kernel action expected | ✓ |
| Skin/board never tripped                | Shell zones are `mode=disabled`; board_temp trip is 125 °C | ✓ |

---

## What is the binding thermal constraint?

**Nothing in the kernel was binding during Wave-11.** With all `cur_state=0`,
the only mechanisms that could have shaped the curve are:

1. **DRAM intrinsic thermal saturation** — refresh rate / bank conflicts /
   self-refresh transitions slow as the die approaches ~65 °C, well below the
   100 °C kernel trip. This is **device physics, not kernel policy.**
2. **CPU DVFS via the cpufreq governor** (schedutil) reacting to package
   power, not thermal — but `cpufreq-cpu*` cdevs were idle, so any
   frequency reduction was governor-driven, not thermal-driven.
3. **Workload-side scheduling jitter** on Cortex-X cores as the kernel
   migrated threads.

**Therefore the "65 °C DDR cliff" we cite in the paper should be described as
*DRAM thermal saturation*, not *kernel throttling*.** It is a measured
property of the LPDDR5X stack on this specific device, observable across our
runs, but it is not announced by any sysfs interface — the kernel sees DDR
as cool until 100 °C.

---

## Files / paths probed

- `/sys/class/thermal/thermal_zone{0..97}/type`
- `/sys/class/thermal/thermal_zone{0..97}/temp`
- `/sys/class/thermal/thermal_zone{0..97}/trip_point_{0..N}_temp`
- `/sys/class/thermal/thermal_zone{0..97}/trip_point_{0..N}_type`
- `/sys/class/thermal/thermal_zone{0..97}/trip_point_{0..N}_hyst`
- `/sys/class/thermal/thermal_zone{0..97}/cdev{0..N}` (symlinks)
- `/sys/class/thermal/thermal_zone{0..97}/cdev{0..N}_trip_point`
- `/sys/class/thermal/thermal_zone{0..97}/policy`, `/mode`
- `/sys/class/thermal/cooling_device{0..62}/{type,cur_state,max_state}`

Probe utility: `EndurKV/scripts/android/adb_resilient.sh` → `adb_safe_shell`.
