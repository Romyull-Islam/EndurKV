# Live power and thermal monitor for the OnePlus 15

An interactive view of the signals that `../sample_sensors.sh` records: a Perfetto capture for
after-the-fact timelines and a live dashboard in the browser.

`perfetto` (on-device, v49) and the https://ui.perfetto.dev web UI give a scrollable
timeline of per-core CPU usage and
frequency, ~41 thermal zones (CPU / GPU `gpuss` / **NPU** `nsphvx`,`nsphmx`),
whole-device battery power, memory, and per-process CPU — all time-aligned.

## Capture

```bash
# trace exactly spans a phone-side inference command (best):
./capture.sh -- '<the command to trace, run on the phone>'

# or fixed window, then drive inference by hand on the phone within it:
./capture.sh -t 60 -o decode_trace
```

Then open https://ui.perfetto.dev and drag in the `*.pftrace` file.

## What the trace contains

| Signal | Track in the trace |
|---|---|
| CPU usage and frequency | per-core sched slices and `cpu_frequency` tracks |
| Temperature | `thermal_zone*` (cpu-*, `gpuss` for the GPU, `nsphvx` for the NPU) |
| Power | `batt` counters (V·I), GPU and NPU frequency |
| Per-process activity | per-process slices (`linux.process_stats`) |

## Energy caveat
Snapdragon has **no Pixel-style per-rail ODPM**, so "energy" = whole-device
battery power from the coulomb counter (V×I), the same source as
`sample_sensors.sh`. The device also exposes
`battery/power_now` (µW) for an instantaneous reading.

Config: `trace_config.textproto`. Driver: `capture.sh` (env `ADB` to override adb path/serial).

---

## Live dashboard (real-time, in a browser)

Perfetto is capture-then-view. For a *live* view during inference, use `live_dashboard.py` — a headless-friendly web dashboard that
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
window with auto-restore) so the phone runs on battery while staying tethered;
otherwise wattage reads ~0 (plugged-in/Full).
