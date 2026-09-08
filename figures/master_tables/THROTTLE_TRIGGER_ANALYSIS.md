# Empirical Throttle Trigger Analysis

Question: For the two cells where decode-tps measurably collapsed
(Wave-4 vanilla iter 6: 5.16 -> 4.11 tps; Wave-8 v1_fa2_selective iter 10:
6.81 -> 5.77 tps), which thermal sensor was actually over its threshold at
the moment of throttling, and when did the kernel `cool_state` cooling
device first activate?

All temperatures are in milli-degC as logged; reported here in C. CPU
frequencies in kHz from `cpu*_freq_hz`. Big-core ceiling pinned to
1,632,000 kHz (performance governor) per the `pin_dvfs` shim. Source CSVs
have 138 columns; relevant indices found below.

Column indices used (1-based, identical schema in both files):

| idx | name |
|---|---|
| 1 | wall_clock_s |
| 3 | cpullc-0-0_temp_mc |
| 12 | cpullc-1-0_temp_mc |
| 20 | cpu-1-0-0_temp_mc |
| 45 | ddr_temp_mc |
| 58 | socd_temp_mc (always 0 -- sensor not exposed) |
| 60 | shell_front_temp_mc |
| 67 | sys-therm-2_temp_mc (hottest skin proxy) |
| 97 | battery_temp_mc |
| 114-121 | cpu0..cpu7 freq_hz |
| 122-129 | cpu0..cpu7 cool_state |
| 132 | bat_current_ma |

---

## Cell 1: Wave-4 vanilla, iter 6 (Phi-3-mini Q4_K_M, K=full, 2048 decode steps)

- Wall window: 1780752111..1780752648 s (iter5 end -> iter6 end, 537 s span)
- Decode tps: iter5 5.160 -> iter6 4.110 (-20%)

Max sensor readings within iter6:

| Sensor | iter5 (steady) | iter6 (throttle) | delta |
|---|---|---|---|
| ddr_temp_mc | 62.5 C | 62.9 C | +0.4 |
| cpu-1-0-0 | 66.7 C | 67.1 C | +0.4 |
| cpullc-0-0 | 65.2 C | -- | -- |
| cpullc-1-0 | 66.3 C | -- | -- |
| sys-therm-2 (skin) | 55.6 C | 55.7 C | +0.1 |
| shell_front | 48.96 C | 49.02 C | +0.06 |
| battery_temp | 47.1 C | 47.2 C | +0.1 |
| mean cpu6_freq | 1,382,400 kHz | 1,171,189 kHz | -15.3% |
| mean cpu7_freq | 1,382,400 kHz | 1,171,189 kHz | -15.3% |

Cool-state activity:
- `cpu0_cool_state` ... `cpu7_cool_state` are all 0 for the ENTIRE 13,830-row
  wave4 vanilla run (start to end of 9 iterations, ~68 min). It never rises.
- DDR has no exposed cool_state column in the sensor map.

CPU clock excursions in iter6: min 883,200 kHz, max 1,382,400 kHz on big
cores; ceiling 1,632,000 kHz is never reached even though the governor is
pinned to "performance". Big cores oscillate between 883 MHz and 1.38 GHz
while cool_state == 0, which means the down-clocking is coming from a
mitigation path that does NOT route through the kernel's
`/sys/class/thermal/.../cur_state` interface (LMH / BCL / CPR voltage clamp,
not the `cpu_cooling` thermal-zone governor).

## Cell 2: Wave-8 v1_fa2_selective, iter 10 (Phi-3-mini Q4_K_M, K=512, top_k=32)

- Wall window: 1780791565..1780791956 s (iter9 end -> iter10 end, 391 s span)
- Decode tps: iter9 6.814 -> iter10 5.773 (-15.3%); iter11 recovered to 6.785

Max sensor readings within iter10:

| Sensor | iter9 (steady) | iter10 (throttle) | delta |
|---|---|---|---|
| ddr_temp_mc | 69.9 C | 72.9 C | +3.0 |
| cpu-1-0-0 | 75.2 C | 77.9 C | +2.7 |
| cpullc-0-0 | -- | 78.4 C | -- |
| cpullc-1-0 | -- | 77.5 C | -- |
| sys-therm-2 (skin) | 60.9 C | 62.1 C | +1.2 |
| shell_front | 52.7 C | 53.0 C | +0.3 |
| battery_temp | 50.8 C | 51.1 C | +0.3 |
| mean cpu6_freq | 1,497,600 kHz | 1,355,129 kHz | -9.5% |
| mean cpu7_freq | 1,497,600 kHz | 1,355,129 kHz | -9.5% |
| mean bat_current | 371 mA | 296 mA | -20% |

Cool-state activity:
- All 8 `cpu*_cool_state` columns remain 0 for the ENTIRE wave-8
  v1_fa2_selective run. They never rise.
- HW trip points (`cpu-hw-trip-0/1`) are both at 95 C; cpu-1-0-0 peaked at
  77.9 C, ~17 C below the catastrophic trip.

CPU clock: min 883,200 kHz, max 1,632,000 kHz; mean 1.36 GHz in iter10 vs
1.50 GHz in iter9. The same sub-cooling-device clamp is active. Mean
battery current also drops by 20%, consistent with a BCL-style current
limit being asserted.

---

## Empirical answer to "when did cool_state actually rise above 0?"

It did not. Across both runs (wave-4 vanilla, 9 iters, ~68 min;
wave-8 v1_fa2_selective, 11 iters, ~62 min) every single sample of
`cpu0_cool_state` ... `cpu7_cool_state` is 0. The kernel's thermal
cooling-device path was never triggered.

Despite this, big-core CPU frequency was clamped well below the pinned
ceiling of 1.632 GHz, with mean frequency dropping ~10-15% on the exact
iteration where decode tps collapsed. The throttle is therefore happening
in a Qualcomm-specific mitigation layer (LMH / BCL / CPR), invisible to
`/sys/class/thermal/.../cur_state`.

## Which temperature(s) were over threshold?

Common Snapdragon mitigation knees (typical for SM8650/SM8750-class SoC):

- LMH (Limits Management Hardware) CPU junction warning: ~85-95 C TJ. Hit
  state in our cells: 67-78 C cpu-1-0-0. NOT crossed.
- DDR thermal mitigation knee: typically 80-85 C. Hit state: 63-73 C. NOT
  crossed.
- Skin (sys-therm-2 / shell_front) BCL-style mitigation: vendor-tuned, very
  often 50-55 C shell. Hit state: 49 C (wave4) / 53 C (wave8). CROSSED in
  wave8, AT-ish-threshold in wave4.
- Battery temperature current-limit: typically 45-50 C. Hit state: 47 C
  (wave4) / 51 C (wave8). CROSSED in wave8, CROSSED in wave4 (47 > 45).

## Bottom line on the "is DDR the binding constraint?" question

No. DDR temperature never approached a typical DDR mitigation knee
(~80-85 C); peak observed DDR was 72.9 C and the cell where decode-tps
collapsed by 15-20% had DDR rising by only 0.4 C (wave4) or 3.0 C
(wave8). The dominant covariate of the slowdown is mean big-core CPU
frequency, which drops 10-15% on the throttled iter while the kernel's
cool_state stays at 0. The two sensors that most plausibly crossed a
mitigation threshold are battery_temp_mc (47-51 C) and shell_front /
sys-therm-2 (skin), both of which gate Qualcomm BCL-style current/clock
clamps. DDR temperature rose in lock-step but was not at its own
mitigation knee. Cool_state itself is not a usable throttle proxy on
this device.

## Reproduction commands

```
# wave4 vanilla iter6 window
awk -F, -v s=1780752111 -v e=1780752648 'NR>1 && $1>=s && $1<=e {
  if ($45+0>m_ddr) m_ddr=$45+0
  if ($20+0>m_cpu) m_cpu=$20+0
  if ($67+0>m_st2) m_st2=$67+0
  if ($60+0>m_sf) m_sf=$60+0
  if ($97+0>m_bat) m_bat=$97+0
  c6+=$120; n++
} END {print "ddr=",m_ddr,"cpu100=",m_cpu,"st2=",m_st2,"sf=",m_sf,"bat=",m_bat,"mean_cpu6=",c6/n}' \
phone-logs/wave4_longdecode_1780750084/vanilla/sensors.csv

# cool_state ever > 0 (whole file)
awk -F, 'NR>1 { for(i=122;i<=129;i++) if($i+0>0){print NR,i,$i; exit} }' \
phone-logs/wave4_longdecode_1780750084/vanilla/sensors.csv   # (no output)
awk -F, 'NR>1 { for(i=122;i<=129;i++) if($i+0>0){print NR,i,$i; exit} }' \
phone-logs/wave8_v1fa2_sel_1780788550/v1_fa2_selective/sensors.csv   # (no output)
```
