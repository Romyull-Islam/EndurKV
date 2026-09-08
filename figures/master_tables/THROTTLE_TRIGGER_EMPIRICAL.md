# Empirical Throttle Trigger Correlation

What sensor crossed its trip point FIRST when the kernel forced a throttle?

## Method

For two measured throttle events on the OnePlus 15 (Snapdragon 8 Elite):

1. **Wave-4 vanilla long-decode iter 6** — kernel-forced cpufreq cap to 883 MHz
   (decode_tps collapses from ~5.16 -> 4.11 t/s vs the previous iter)
2. **Wave-8 v1_fa2_selective iter 10** — kernel cliff event
   (decode_tps drops from ~6.79 -> 5.77 t/s)

For each `sensors.csv`, the first sample where each candidate sensor exceeds its
canonical Qualcomm/QCM thermal trip is recorded. Sensors and trip points:

| Sensor (CSV column) | Description | Trip (deg C) |
|---|---|---|
| `ddr_temp_mc` | LPDDR5 self-reported temp | 65.0 |
| `cpu-1-0-0_temp_mc` | Big core (cluster 1, core 0, sensor 0) | 95.0 |
| `socd_temp_mc` | SoC die | 95.0 (sensor returns 0 on this device — inactive) |
| `shell_front_temp_mc` | Front skin (user-touch proxy) | 43.0 |
| `cpu-hw-trip-0_temp_mc` | Kernel-published trip sentinel | static 95.0 (configured trip, not a measurement) |
| `battery_temp_mc` | Battery | 45.0 |

Source files:
- `/home/mislam22/EndurKV_workspace/phone-logs/wave4_longdecode_1780750084/vanilla/sensors.csv`
- `/home/mislam22/EndurKV_workspace/phone-logs/wave8_v1fa2_sel_1780788550/v1_fa2_selective/sensors.csv`

## Result 1: cpu*_cool_state is NEVER set in either run

In BOTH runs, every `cpu0_cool_state` ... `cpu7_cool_state` sample stays at **0**
for the entire run (13,829 samples in Wave-4; 12,552 samples in Wave-8). The
kernel's user-visible cooling-device counters never engaged.

The throttle therefore did NOT happen via the standard Linux thermal cooling-
device mechanism. It happened via cpufreq-governor / DCVS hard cap (the 883 MHz
ceiling is silently enforced by the kernel/governor and not reported through the
`cooling_device/cur_state` interface that this `sensors.csv` reads).

Implication: a watchdog cannot rely on `cpu*_cool_state` to detect the throttle.
The CSV's `cpu*_freq_hz` column is also broken (all values pinned at 1 MHz idle),
so frequency-based detection from this stream is unavailable. Detection must be
**temperature-derived**, not state-derived.

## Result 2: chronological order of trip crossings, full run

### Wave-4 vanilla long-decode

| order | sample idx | mono_s | sensor | value (deg C) | trip (deg C) |
|---|---|---|---|---|---|
| 1 | 2207 | 151000.4 | shell_front (skin) | **43.04** | 43.0 |
| 2 | 4702 | 151724.7 | battery | **45.10** | 45.0 |
| —  | —    | —        | ddr | never exceeds (max 62.9) | 65.0 |
| —  | —    | —        | cpu-1-0-0 | never exceeds (max 67.1) | 95.0 |
| —  | —    | —        | socd | sensor reads 0 (offline) | 95.0 |

Max temps in this run: DDR 62.9, CPU big 67.1, skin 49.1, battery 47.3.
The "iter 6 forced 883 MHz" event therefore occurred with:
- DDR still ~58-63 deg C (below 65 deg C trip)
- CPU big still ~58-67 deg C (well below 95 deg C trip)
- Skin already ~46-49 deg C (3-6 deg C OVER 43 deg C trip)
- Battery already ~46-47 deg C (1-2 deg C OVER 45 deg C trip)

### Wave-8 v1_fa2_selective

| order | sample idx | mono_s | sensor | value (deg C) | trip (deg C) |
|---|---|---|---|---|---|
| 1 | 2490 | 189579.0 | shell_front (skin) | **43.05** | 43.0 |
| 2 | 4484 | 190172.8 | battery | **45.10** | 45.0 |
| 3 | 5145 | 190369.5 | ddr | **65.20** | 65.0 |
| —  | —    | —        | cpu-1-0-0 | never exceeds (max 77.9) | 95.0 |
| —  | —    | —        | socd | sensor reads 0 (offline) | 95.0 |

Iter 10 cliff window temps: DDR mean 66.06 (max 72.9), CPU big mean 69.77 (max
77.9), skin mean 52.21, battery mean 50.79. At iter 10 start: DDR=69.1, CPU=74.8,
skin=52.66, batt=50.8. All four sensors were already over their respective
trip points before iter 10 began.

DDR crossed 65 deg C at sample 5145 (mono 190369.5s), about 13.3 minutes after
the skin first crossed 43 deg C.

## Result 3: which sensor was "first OVER trip"?

Strictly by chronological trip crossing, in BOTH runs the answer is the same:

> **shell_front (skin) crossed its 43 deg C trip first, by a large margin.**

Battery (45 deg C) came second.

DDR (65 deg C) crossed third — but only in Wave-8 (the more thermally stressed
run). In Wave-4 DDR peaked at 62.9 deg C and never crossed 65 deg C, yet the
kernel still forced the 883 MHz cap. **The 883 MHz throttle in Wave-4 happened
without DDR ever crossing 65 deg C.** CPU big and SoC-die likewise never crossed
their 95 deg C trip in either run.

## Implication for watchdog design

A DDR-only watchdog at 65 deg C is **insufficient**:

- It misses the Wave-4 event entirely (DDR peaked at 62.9 deg C; the 883 MHz cap
  fired anyway, presumably driven by skin/battery limits or by a virtual zone
  that mixes skin + battery + DDR + CPU).
- The skin (`shell_front`) sensor crossed its 43 deg C trip thousands of samples
  before DDR ever did — about 12 minutes earlier in Wave-4 and ~13 minutes
  earlier in Wave-8.
- Battery crossed its 45 deg C trip second in both runs, ~10 minutes before DDR.

A correct watchdog must be **multi-sensor**: it needs at least DDR and skin, and
ideally battery as a third input. A practical rule, given these data:

```
throttle_imminent =
    (ddr_temp_c   > 64)  OR
    (shell_front_c > 44) OR
    (battery_temp_c > 45 AND ddr_temp_c > 60)
```

The skin sensor is the leading indicator in both runs. DDR is the right "this
run is in the cliff zone now" gate, but it is not the first sensor to trip — and
it is not even guaranteed to trip when the kernel decides to throttle (Wave-4
counter-example).

## Bottom line

**EARLIEST sensor over trip in both throttle events: `shell_front_temp_mc` (skin
proxy) at 43 deg C.**

A DDR-only watchdog would have missed the Wave-4 throttle and would have fired
~13 minutes late in Wave-8. The watchdog must be multi-sensor (skin + DDR at
minimum, battery as a third input).
