# On-phone setup

Everything that runs on the phone for μKV: the builds, the per-request scheduler, the two clock
watchdogs, sensor sampling, the cooling gate, the discharge loop and the measurement campaigns.
The phone is a OnePlus 15 (Snapdragon 8 Elite Gen 5, Adreno 840). Some sysfs paths below are
specific to it.

## Requirements

- Host: Android NDK r27c (set `ANDROID_NDK`), CMake, Ninja, `adb`, and Python 3 with matplotlib.
- Phone: USB debugging and root. Root is needed to cap the CPU and GPU clocks, to switch charging
  off (`/sys/class/oplus_chg/battery/mmi_charging_enable`), and to read the thermal and power sensors.
- `adb_resilient.sh` wraps adb calls with timeouts and retries. The campaign scripts also reach an
  adb server on another port through `ANDROID_ADB_SERVER_PORT`.

## Build

```
scripts/android/build_llama_android.sh          # llama.cpp for the CPU
scripts/android/build_llama_android_vulkan.sh   # llama.cpp for the Adreno GPU (Vulkan)
scripts/android/build_probe_android.sh          # eviction_bench and the probes
```

Build the CPU libraries with `-march=armv8.7-a` in the C and C++ flags; without it the
dot-product and int8 matrix-multiply kernels are left out. `entropy_probe/CMakeLists.txt` sets it
for `eviction_bench`. Push each binary together with the `.so` libraries it was built with.

## Layout on the phone

```
/data/local/tmp/endurkv/
  models/               GGUF files
  bin_cpu_ea/           CPU build used by the scheduler
  ukv_sched.sh          scheduler, with its cost table, lever bias and log next to it
  tables/               per-model cost tables and lever bias
  sched/<tag>/          per-request outputs: meta.json, sensors.csv, err
/data/local/tmp/ukv/    Vulkan build used for GPU plans
/data/local/tmp/sample_sensors.sh
```

## Scheduler

```
su -c 'sh /data/local/tmp/endurkv/ukv_sched.sh --prompt FILE [--max-tokens N] [--model NAME] \
       [--plan P] [--no-feedback] [--cpu-only] [--ignore-eos] [--dry-run] [--tag NAME]'
```

For each request the scheduler reads the battery level and charging state, sets the tier and
lever, and walks the model's cost table for a plan: GPU clocks, cache budget and answer cap. It
then runs `eviction_bench` with μKV, meters the request, updates the table, and lets the two
loops move the lever bias. `--dry-run` stops after choosing the plan. `--plan` forces one plan;
the learner (`run_guarded_bandit.py`) uses it.

## Clock watchdogs

- `preempt_throttle_watchdog_v2.sh` steps the CPU clock down from 1497 to 1017 MHz as the battery
  passes 47.0 to 49.5 °C, or the skin 50.0 to 52.5 °C, just below the vendor's 50 °C deep throttle.
- `gpu_watchdog.sh` steps the GPU cap down to 1050, 967 and 902 MHz as DDR passes 60, 62 and
  63.5 °C, just below the vendor's 64 °C cap.

Both run as root in the background, only ever lower a cap, and restore the caps on exit.

## Measurement

- `sample_sensors.sh` records thermal zones, clocks, the USB rail and the battery's coulomb
  counter. The scheduler samples at 2 Hz and the campaigns at up to 5 Hz; the default is 10 Hz.
- `cool_gate.sh` waits until DDR is at most 35 °C and the battery at most 33 °C, with charging
  off, before every timed request.
- `phone_discharge_loop.sh` runs requests back to back through the scheduler, on the phone alone,
  until a stop level, and logs each request with a USB-power flag. `phone_discharge_start.sh`
  waits for a start level first. The cable must be out for a real discharge: with charging off,
  the USB port still supplies most of the power.

## Campaigns

Each `run_*` script drives one experiment from the host and keeps its results on the host.
Examples: `run_clock_fine.py` (GPU clock ladder), `run_k_energy_gpu.py` (cache budget),
`run_guarded_bandit.py` (learner), `run_gpu_watchdog_test.sh` (GPU watchdog),
`run_phi3_gpu_complete.sh` (Phi-3 on the GPU), `run_longbench_wide.sh` and `run_lb_keydiff.sh`
(LongBench), and `run_niah_published_budgets.sh` (needle retrieval).

## Other tools

- `live_monitor/`: a Perfetto capture and a live browser dashboard of power, temperature and clocks.
- `discover_sensors.sh`, `run_one_prompt.sh` and the `host_*.py` scripts come from an earlier
  entropy and attention measurement study. μKV does not use them.

To remove everything from the phone: `adb shell rm -rf /data/local/tmp/endurkv`.
