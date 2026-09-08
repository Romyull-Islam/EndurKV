# OnePlus 15 resource/energy GUI — the Android "Xcode Instruments"

Part of **EndurKV**'s Android device tooling — the GUI front-end to the sensor sampling
in `../sample_sensors.sh` / `../discover_sensors.sh` and the `host_plot_*_thermal.py`
plotters. (PowerInfer-2's Table-8 energy scripts in `../../../../powerinfer2/energy/`
reuse the same coulomb-counter source; this kit is the live/interactive view of it.)

`perfetto` (on-device, v49) + the **https://ui.perfetto.dev** web UI give the same
scrollable timeline as the iPhone/Xcode-Instruments view: per-core CPU usage &
frequency, ~41 thermal zones (CPU / GPU `gpuss` / **NPU** `nsphvx`,`nsphmx`),
whole-device battery power, memory, and per-process CPU — all time-aligned.

## Capture

```bash
# trace exactly spans a phone-side inference command (best):
./capture.sh -- 'cd /data/local/tmp && LD_LIBRARY_PATH=. ./pi-main -m model.gguf -p "..." -n 256 -t 6'

# or fixed window, then drive inference by hand on the phone within it:
./capture.sh -t 60 -o bamboo_decode
```

Then open https://ui.perfetto.dev and drag in the `*.pftrace` file.

## What you get vs. the iPhone shot
| Instruments track            | OnePlus 15 equivalent (in the trace)                     |
|------------------------------|----------------------------------------------------------|
| CPU 0..5 usage               | per-core sched slices + `cpu_frequency` tracks           |
| Thermal State                | `thermal_zone*` (cpu-*, `gpuss`=GPU, `nsphvx`=NPU)        |
| Energy / Metal               | `batt` counters (V·I), GPU/NPU frequency                 |
| App Lifecycle / Time Profiler| per-process slices (`linux.process_stats`)               |

## Energy caveat (honest)
Snapdragon has **no Pixel-style per-rail ODPM**, so "energy" = whole-device
battery power from the coulomb counter (V×I) — the same source as EndurKV's
`sample_sensors.sh` and the PowerInfer-2 Table 8 pipeline. The device also exposes
`battery/power_now` (µW) for an instantaneous reading.

Config: `trace_config.textproto`. Driver: `capture.sh` (env `ADB` to override adb path/serial).

---

## Live dashboard (real-time, in a browser)

Perfetto is capture-then-view. For a *live* view during inference (like watching the
iPhone shot update), use `live_dashboard.py` — a headless-friendly web dashboard that
polls the phone at ~4 Hz over adb and plots time-aligned tracks: battery power (W),
CPU/GPU/NPU/DDR/skin temp, CPU utilization, per-cluster CPU frequency.

```bash
python3 live_dashboard.py                 # serve http://127.0.0.1:8717
python3 live_dashboard.py --port 9000 --csv run1.csv   # also log every sample to CSV
```

Open the printed URL in a browser, then run inference on the phone — the tracks update live.
It auto-excludes the bogus `cpu-hw-trip-*` 95 °C trip points and reads real core sensors.

**Power is real only while DISCHARGING** — the dashboard shows the battery status pill
(green = discharging). Run `./run_on_battery.sh 600` (disables charging for a bounded
window with auto-restore) so the phone runs on battery while staying USB/usbip-tethered;
otherwise wattage reads ~0 (plugged-in/Full).
