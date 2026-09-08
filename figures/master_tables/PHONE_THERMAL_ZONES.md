# Phone Thermal Zones, Trip Points, and Cooling-Device Bindings

Device: OnePlus 15 (CPH2749), Android 16. Snapshot collected via
`adb_safe_shell` (see `scripts/android/adb_resilient.sh`) by enumerating
`/sys/class/thermal/thermal_zone$N` for N=0..99 and
`/sys/class/thermal/cooling_device$N`.

Raw captures: `/tmp/thermal_zones_raw.txt`,
`/tmp/cooling_devices_raw.txt`, `/tmp/zone_to_cdev_bindings.txt`.

All temperatures shown in milli-degC as the kernel reports them. Some
zones (`socd`, `vbat`, BCL levels, `pmih010x-ibat-*`, modem `sdr0*`,
`mmw_ific0`) report sentinel values (`0`, `1`, `100`, `-273000`) and use
them as state flags rather than physical temperatures; trip values like
`1` or `100` are state-machine thresholds, not Celsius.

Total: **98 thermal zones** (N=0..97), **63 cooling devices** (N=0..62).

---

## 1. Master zone -> type -> trips -> bound cooling devices

| Zone | Type            | Cur (mC) | Trips (i: type / temp_mC / hyst_mC) | Bound cooling devices (cdev_link @ trip_idx) |
|------|-----------------|---------:|--------------------------------------|----------------------------------------------|
| 0  | cpullc-0-0       |   35500 | 0:passive/95000/1000; 1:hot/125000/0; 2:passive/112000/4000; 3:passive/114000/6000 | cdev13(thermal-pause-10)@2, cdev10(thermal-pause-8)@2, cdev12(thermal-pause-4)@2, cdev18(thermal-pause-20)@2, cdev27(cpu-hotplug2)@3, cdev28(cpu-hotplug3)@3, cdev29(cpu-hotplug5)@3, cdev32(cpu-hotplug4)@3 |
| 1  | cpullc-0-1       |   36600 | 0:passive/135000/1000; 1:passive/112000/4000; 2:passive/114000/6000; 3:hot/125000/0 | cdev13@1, cdev10@1, cdev12@1, cdev18@1, cdev27@2, cdev28@2, cdev29@2, cdev32@2 |
| 2  | qmx-0-0          |   36300 | 0:passive/135000/1000; 1:hot/125000/0 | (none) |
| 3  | qmx-0-1          |   36600 | 0:passive/135000/1000; 1:hot/125000/0 | (none) |
| 4  | qmx-0-2          |   36600 | 0:passive/135000/1000; 1:hot/125000/0 | (none) |
| 5  | cpu-0-0-0        |   36600 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev9(thermal-pause-1)@2, cdev26(cpu-hotplug0)@3 |
| 6  | cpu-0-0-1        |   37700 | 0:passive/95000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev9@2, cdev26@3 |
| 7  | cpu-0-1-0        |   36600 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev16(thermal-pause-2)@2, cdev25(cpu-hotplug1)@3 |
| 8  | cpu-0-1-1        |   38100 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev16@2, cdev25@3 |
| 9  | cpu-0-2-0        |   37000 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev12@2, cdev27@3 |
| 10 | cpu-0-2-1        |   37000 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev12@2, cdev27@3 |
| 11 | cpu-0-3-0        |   36600 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev10@2, cdev28@3 |
| 12 | cpu-0-3-1        |   36600 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev10@2, cdev28@3 |
| 13 | cpu-0-4-0        |   37000 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev13@2, cdev32@3 |
| 14 | cpu-0-4-1        |   37000 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev13@2, cdev32@3 |
| 15 | cpu-0-5-0        |   37400 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev18@2, cdev29@3 |
| 16 | cpu-0-5-1        |   38500 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev18@2, cdev29@3 |
| 17 | cpullc-1-0       |   36800 | 0:passive/135000/1000; 1:passive/112000/4000; 2:passive/114000/6000; 3:hot/125000/5000 | cdev17(thermal-pause-80)@1, cdev20(thermal-pause-40)@1, cdev30(cpu-hotplug6)@2, cdev31(cpu-hotplug7)@2 |
| 18 | cpullc-1-1       |   37600 | 0:passive/135000/1000; 1:passive/112000/4000; 2:passive/114000/6000; 3:hot/125000/5000 | cdev17@1, cdev20@1, cdev30@2, cdev31@2 |
| 19 | qmx-1-0          |   36800 | 0:passive/135000/1000; 1:hot/125000/0 | (none) |
| 20 | qmx-1-1          |   36400 | 0:passive/135000/1000; 1:hot/125000/0 | (none) |
| 21 | qmx-1-2          |   36000 | 0:passive/135000/1000; 1:hot/125000/0 | (none) |
| 22 | qmx-1-3          |   36400 | 0:passive/135000/1000; 1:hot/125000/0 | (none) |
| 23 | qmx-1-4          |   36000 | 0:passive/135000/1000; 1:hot/125000/0 | (none) |
| 24 | cpu-1-0-0        |   36400 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev20@2, cdev30@3 |
| 25 | cpu-1-0-1        |   36400 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev20@2, cdev30@3 |
| 26 | cpu-1-1-0        |   36400 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev17@2, cdev31@3 |
| 27 | cpu-1-1-1        |   36800 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/112000/4000; 3:passive/114000/6000; 4:hot/125000/0 | cdev17@2, cdev31@3 |
| 28 | nsphvx-0         |   35200 | 0:passive/135000/1000; 1:passive/105000/5000; 2:hot/125000/0 | (none) |
| 29 | nsphvx-1         |   35600 | 0:passive/135000/1000; 1:passive/105000/5000; 2:hot/125000/0 | (none) |
| 30 | nsphvx-2         |   35600 | 0:passive/135000/1000; 1:passive/105000/5000; 2:hot/125000/0 | (none) |
| 31 | nsphvx-3         |   35600 | 0:passive/135000/1000; 1:passive/105000/5000; 2:hot/125000/0 | (none) |
| 32 | nsphmx-0         |   34900 | 0:passive/135000/1000; 1:passive/105000/5000; 2:hot/125000/0 | (none) |
| 33 | nsphmx-1         |   35600 | 0:passive/135000/1000; 1:passive/105000/5000; 2:hot/125000/0 | (none) |
| 34 | nsphmx-2         |   35600 | 0:passive/135000/1000; 1:passive/105000/5000; 2:hot/125000/0 | (none) |
| 35 | nsphmx-3         |   35600 | 0:passive/135000/1000; 1:passive/105000/5000; 2:hot/125000/0 | (none) |
| 36 | gpuss-0          |   36300 | 0:passive/135000/1000; 1:passive/105000/5000; 2:passive/105000/0; 3:hot/125000/0 | cdev34(kgsl)@2 |
| 37 | gpuss-1          |   36600 | (same as gpuss-0) | cdev34@2 |
| 38 | gpuss-2          |   36300 | (same) | cdev34@2 |
| 39 | gpuss-3          |   35900 | (same) | cdev34@2 |
| 40 | gpuss-4          |   35900 | (same) | cdev34@2 |
| 41 | gpuss-5          |   36600 | (same) | cdev34@2 |
| 42 | gpuss-6          |   36300 | (same) | cdev34@2 |
| 43 | gpuss-7          |   36300 | (same) | cdev34@2 |
| 44 | gpuss-8          |   37000 | (same) | cdev34@2 |
| 45 | gpuss-9          |   36300 | (same) | cdev34@2 |
| 46 | gpuss-10         |   36600 | (same) | cdev34@2 |
| 47 | ddr              |   36300 | 0:passive/135000/1000; 1:passive/100000/5000; 2:hot/125000/0 | cdev21(ddr-cdev)@1 |
| 48 | mdmss-0          |   36400 | 0:passive/135000/1000; 1:passive/102000/3000; 2:passive/105000/3000; 3:hot/125000/0 | cdev42(modem_lte_dsc)@1, cdev42@2, cdev44(modem_nr_dsc)@1, cdev44@2, cdev46(modem_nr_scg_dsc)@1 |
| 49 | mdmss-1          |   36000 | (same) | cdev42@1, cdev42@2, cdev44@1, cdev44@2, cdev46@1 |
| 50 | mdmss-2          |   35600 | (same) | cdev42@1, cdev42@2, cdev44@1, cdev44@2, cdev46@1 |
| 51 | mdmss-3          |   36000 | (same) | cdev42@1, cdev42@2, cdev44@1, cdev44@2, cdev46@1 |
| 52 | camera-0         |   35600 | 0:passive/135000/1000; 1:hot/125000/0 | (none) |
| 53 | camera-1         |   36000 | 0:passive/135000/1000; 1:hot/125000/0 | (none) |
| 54 | video            |   36400 | 0:passive/135000/1000; 1:hot/125000/0 | (none) |
| 55 | pmh0104_tz       |   37000 | 0:passive/95000/0; 1:hot/115000/0; 2:critical/145000/0 | (none) |
| 56 | pmr735d_tz       |   37000 | (same shape) | (none) |
| 57 | pm8010m_tz       |   37000 | (same shape) | (none) |
| 58 | pmh0101-bcl-lvl0 |       0 | 0:passive/1/0  (state flag, not C) | cdev34(kgsl)@0, cdev62(modem_bcl)@0 |
| 59 | socd             |       0 | 0:passive/100/0; 1:passive/90/0; 2:passive/95/0  (SoC discharge %) | cdev1(cpufreq-cpu6)@2, cdev0(cpufreq-cpu0)@2, cdev34@2, cdev38(cdsp)@2 |
| 60 | shell_front      |   34340 | (no trips exposed) | (none) |
| 61 | cpu-hw-trip-0    |   95000 | 0:passive/95000/0  (HW latch register) | (none) |
| 62 | sys-therm-0      |   34842 | 0:passive/125000/1000; 1:passive/40000/2000; 2:passive/78000/8000; 3:passive/80000/10000; 4:hot/90000/0; 5:passive/125000/1000; 6:passive/125000/1000; 7:passive/125000/1000 | cdev31(cpu-hotplug7)@3, cdev30(cpu-hotplug6)@3, cdev29(cpu-hotplug5)@3, cdev28(cpu-hotplug3)@3, cdev23(display-fps)@5, cdev23@6, cdev23@7, cdev19(thermal-pause-C0)@2, cdev8(thermal-pause-3C)@2, cdev32(cpu-hotplug4)@3, cdev33(thermal-pause-3C)@2, cdev34(kgsl)@2, cdev38(cdsp)@2, cdev42(modem_lte_dsc)@3, cdev46(modem_nr_scg_dsc)@3 |
| 63 | wireless         |   33900 | (no trips) | (none) |
| 64 | pmh0101-bcl-lvl1 |       0 | 0:passive/1/0 | cdev34@0 |
| 65 | shell_frame      |   32893 | (no trips) | (none) |
| 66 | cpu-hw-trip-1    |   95000 | 0:passive/95000/0  (HW latch) | (none) |
| 67 | sys-therm-2      |   36073 | 0:passive/48000/2000; 1:passive/49000/2000; 2:passive/50000/2000; 3:passive/60000/2000; 4:passive/61000/2000; 5:passive/90000/2000 | (none) |
| 68 | shell_back       |   33172 | (no trips) | (none) |
| 69 | pmh0101-bcl-lvl2 |       0 | 0:passive/1/0 | (none) |
| 70 | sys-therm-3      |   34811 | 0:passive/125000/1000; 1:passive/125000/1000 | (none) |
| 71 | vbat             |       0 | 0:passive/2800/100; 1:passive/2600/100; 2:passive/2300/100  (battery mV undervolt) | (none) |
| 72 | sys-therm-4      |   35785 | 0:passive/125000/1000; 1:passive/125000/1000 | (none) |
| 73 | pmih010x-ibat-lvl0 |     0 | 0:passive/13500/200 (mA over-current) | (none) |
| 74 | sys-therm-5      |   35919 | 0:passive/125000/1000; 1:passive/125000/1000 | (none) |
| 75 | pmih010x-ibat-lvl1 |     0 | 0:passive/15000/200 (mA over-current) | (none) |
| 76 | sys-therm-6      |   36050 | 0:passive/125000/1000; 1:passive/125000/1000 | (none) |
| 77 | pmih010x-bcl-lvl0  |     0 | 0:passive/100/0; 1:passive/100/0; 2:passive/1/0 | cdev34(kgsl)@2, cdev62(modem_bcl)@2 |
| 78 | sys-therm-7      |   35176 | 0:passive/125000/1000; 1:passive/125000/1000 | (none) |
| 79 | pmih010x-bcl-lvl1 |     0 | 0:passive/100/0; 1:passive/100/0; 2:passive/1/0 | cdev34@2 |
| 80 | sys-therm-9      |   35915 | 0:passive/125000/1000; 1:passive/125000/1000 | (none) |
| 81 | pmih010x-bcl-lvl2 |     0 | 0:passive/100/0; 1:passive/100/0; 2:passive/1/0 | cdev34@2 |
| 82 | sys-therm-12     |   36624 | 0:passive/125000/1000; 1:passive/125000/1000 | (none) |
| 83 | flash_temp       |   35316 | 0:passive/125000/1000; 1:passive/125000/1000 | (none) |
| 84 | board_temp       |   35550 | 0:passive/125000/1000; 1:passive/125000/1000 | (none) |
| 85 | svooc_mos_btb_usr|   35829 | 0:passive/125000/1000 | (none) |
| 86 | pmh0101_tz       |   35532 | 0:passive/95000/0; 1:hot/115000/0; 2:critical/145000/0 | (none) |
| 87 | pmh0110_d_tz     |   35833 | (same shape) | (none) |
| 88 | pmh0110_f_tz     |   35231 | (same shape) | (none) |
| 89 | pmh0110_g_tz     |   36502 | (same shape) | cdev33(thermal-pause-3C)@0, cdev8(thermal-pause-3C)@0 |
| 90 | pmh0110_i_tz     |   35365 | (same shape) | cdev19(thermal-pause-C0)@0 |
| 91 | pmih010x_tz      |   35130 | (same shape) | (none) |
| 92 | pmih010x_lite_tz |   56620 | 0:passive/125000/0; 1:hot/135000/0; 2:critical/145000/0 | (none) |
| 93 | usb              |    3200 | (no trips; uA-style flag) | (none) |
| 94 | battery          |   33900 | (no trips) | (none) |
| 95 | sdr0_pa          | -273000 | 0:passive/125000/1000; 1:passive/125000/1000 (offline RF PA) | (none) |
| 96 | sdr0             | -273000 | (same; offline RF) | (none) |
| 97 | mmw_ific0        | -273000 | (same; offline mmWave) | (none) |

---

## 2. Cooling devices

CPU/compute mitigation:
| cdev | type                  | max_state | current |
|------|-----------------------|-----------|---------|
| 0    | cpufreq-cpu0          | 27        | 0       |
| 1    | cpufreq-cpu6          | 28        | 0       |
| 22   | cpu-cluster0          | 27        | 0       |
| 24   | cpu-cluster1          | 28        | 0       |
| 3..7,11,14,15 | pause-cpu0..7| 1         | 0       |
| 25..32 | cpu-hotplug0..7     | 1         | 0       |
| 8,9,10,12,13,16,17,18,19,20,33 | thermal-pause-{3C,1,8,10,4,2,80,20,C0,40,3C} | 1 | 0 |
| 34   | kgsl (GPU)            | 17        | 0       |
| 35   | gpu                   | 17        | 0       |
| 37   | cdsp_hw               | 1         | 0       |
| 38   | cdsp                  | 12        | 0       |
| 39   | cdsp_sw_hvx           | 11        | 0       |
| 40   | cdsp_sw_hmx           | 12        | 0       |

I/O, display, audio, modem, RF:
| cdev | type                  | max_state |
|------|-----------------------|-----------|
| 2    | ufs                   | 2         |
| 21   | ddr-cdev              | 1         |
| 23   | display-fps           | 16        |
| 36   | panel0-backlight      | 255       |
| 41   | wsa2                  | 11        |
| 42..47 | modem_{lte,lte_sub1,nr,nr_sub1,nr_scg,nr_scg_sub1}_dsc | 255 |
| 48..53 | pa_{lte,nr}_sdr0_{,scg_,sub1_,scg_sub1_}dsc | 255 |
| 54..61 | mmw0..3_{,sub1_}dsc | 255         |
| 62   | modem_bcl             | 255       |

**Every cooling device read `cur_state=0`** at snapshot time -- consistent
with the earlier finding that under load (Wave-4 / Wave-8) all
`cpu*_cool_state` values also stay at 0 throughout.

---

## 3. Zone -> bound CPU cooling devices (compact map)

CPU cluster 0 (`cpu-0-*`, `cpullc-0-*`) routes to:
- thermal-pause-{1,2,4,8,10,20} (cdev9,16,12,10,13,18) at passive trip
  ~105-112 C
- cpu-hotplug{0..5} (cdev26,25,27,28,29,32) at passive trip ~112-114 C

CPU cluster 1 (`cpu-1-*`, `cpullc-1-*`) routes to:
- thermal-pause-{40,80} (cdev17,20) at passive trip ~105-112 C
- cpu-hotplug{6,7} (cdev30,31) at passive trip ~112-114 C

GPU (`gpuss-*`) routes to: `kgsl` (cdev34) at the **secondary** passive
trip @105000 (hyst 0).

DDR (`ddr` zone 47) routes to: `ddr-cdev` (cdev21) at passive 100 C.

Modem (`mdmss-*`) routes to modem DSC throttles at 102-105 C.

Skin (`sys-therm-0`) is the rich one: it can pause CPUs, drop
display-fps, throttle kgsl, cdsp, and modem -- but its **passive** trips
sit at 40/78/80 C with `hot` at 90 C.

`sys-therm-2` (skin proxy used elsewhere) has the most aggressive trips
on the board: passive at 48, 49, 50, 60, 61 and 90 mC -- yet **no
cooling device is bound to any of its trips**. It is a pure reporting
zone for vendor userspace governors.

---

## 4. Which sensors actually trigger throttle?

Cross-referencing this map with `THROTTLE_TRIGGER_ANALYSIS.md` (Wave-4
vanilla iter6 and Wave-8 v1_fa2_selective iter10, the two empirically
observed throttle events):

- **No kernel cooling device was activated** -- across both runs every
  `cpu*_cool_state` and every snapshot of `cooling_device*/cur_state`
  stayed at 0.
- **All kernel passive trips were untouched.** Highest observed CPU was
  ~78 C vs the lowest kernel passive trip at 95 C (cpu-0-0-1 zone 6 trip 0,
  and cpullc-0-0 zone 0 trip 0) and the cluster-pause trips at 105-112 C.
- **`sys-therm-2` (skin proxy) crossed its first three trips
  (48/49/50 C)** at peak load: 55.7 C (wave4) and 62.1 C (wave8). These
  trips have **no kernel-bound cooling device**, so they fire vendor
  userspace mitigation (Qualcomm thermal-engine / OPlus thermal HAL),
  which clamps big-core frequency via LMH/BCL paths invisible to
  `/sys/class/thermal/.../cur_state`.
- **`shell_front` (zone 60) and `shell_back` (zone 68)** are pure
  reporting zones (no trips, no cdevs). They are used by the userspace
  governor for skin-temperature decisions.
- **`battery` (zone 94)** also has no kernel trips. Peak observed battery
  was 51 C; combined with the observed 20% drop in `bat_current_ma`
  during the wave-8 throttle event, the dominant active mitigation is a
  battery / current limit asserted through BCL (zones 58/64/69 and
  77/79/81) and routed through `modem_bcl` (cdev62) + `kgsl` (cdev34).
- **DDR is not the binding constraint**: peak ddr temp 72.9 C; its
  passive trip @100 C never fires.
- **CPU HW latches** (`cpu-hw-trip-0/1`, zones 61/66) read 95000 mC at
  rest -- these are register-mirror zones, not live sensors; their
  passive trip 95 C is a catastrophic latch, not an operational knee.

**Conclusion:** The throttle that visibly slows decode tps is driven by
the **skin/board/battery family** -- specifically `sys-therm-2` and the
BCL chain -- via a **Qualcomm userspace + LMH/BCL hardware** path that
**does not pass through any of the 63 kernel cooling devices**. The
kernel thermal-zone trip table is essentially a backstop for catastrophic
events (95-125 C); the operational throttle that EndurKV must defend
against runs entirely outside it.
