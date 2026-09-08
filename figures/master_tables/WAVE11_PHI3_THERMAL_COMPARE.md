# Wave-11 Phi-3 Thermal Comparison: vanilla vs h2o vs v1_fa2_stack

**Headline:** v1_fa2_stack does NOT meaningfully reduce DDR temp vs vanilla on Phi-3 at K=512 (peak -0.7C, mean-decode -0.1C -- within sensor/run-to-run noise). The DDR-temp watchdog may not be needed at K=512 on this workload; h2o actually runs cooler than both (peak 62.1C, mean 57.7C) while v1_fa2_stack stays comparable to vanilla but with the bus cap reducing burst pressure.

## Sources
- **vanilla**: `/home/mislam22/EndurKV_workspace/phone-logs/wave11_eval_1780862534/Phi-3-mini-128k/vanilla/ppl/sensors.csv` (25035 samples, 7661.1 s)
- **h2o**: `/home/mislam22/EndurKV_workspace/phone-logs/wave11_eval_1780862534/Phi-3-mini-128k/h2o/ppl/sensors.csv` (43136 samples, 6503.4 s)
- **v1_fa2_stack**: `/home/mislam22/EndurKV_workspace/phone-logs/wave9_v1fa2_stack_1780796320/v1_fa2_stack/sensors.csv` (12607 samples, 3842.3 s)

> Note: `v1_fa2_stack` was not run in wave11 (`wave11_eval_1780862534/Phi-3-mini-128k/` only contains `vanilla/` and `h2o/`). The closest comparable v1_fa2_stack data on the exact same Phi-3-mini-128k-instruct Q4_K_M model is from wave9 (`wave9_v1fa2_stack_1780796320/v1_fa2_stack/`), captured on the same OnePlus 15 with the same sensor schema and K=512.

## Thermal + memory summary

| Policy | Peak DDR (C) | Mean DDR decode (C) | Peak CPU (C) | Max swap delta (pages / MB) | Min mem_avail (GB) |
|---|---:|---:|---:|---:|---:|
| vanilla | 64.8 | 59.2 | 70.9 | 40429 / 157.9 | 6.89 |
| h2o | 62.1 | 57.7 | 66.6 | 0 / 0.0 | 6.75 |
| v1_fa2_stack | 64.1 | 59.1 | 69.4 | 0 / 0.0 | 7.56 |

## v1_fa2_stack vs vanilla deltas

- Peak DDR: 64.1 C vs 64.8 C (-0.7 C)
- Mean DDR (decode window, t>=60s): 59.1 C vs 59.2 C (-0.1 C)
- Peak CPU: 69.4 C vs 70.9 C (-1.5 C)

## v1_fa2_stack watchdog (DDR-temp governor) activity

- Source: `/home/mislam22/EndurKV_workspace/phone-logs/wave9_v1fa2_stack_1780796320/watchdog.log`
- Tier transitions: **31**
- Tier events parsed: 32

| Tier | Bus cap (kHz) | Time spent (s) |
|---|---:|---:|
| 0 | 1632000 MAX | 876 (22.8%) |
| 1 | 1497600 HIGH | 1948 (50.7%) |
| 2 | 1267200 MED | 1020 (26.5%) |

## Method

- Peak DDR / CPU: max over all sensor samples for the run (`ddr_temp_mc` for DDR; max across all `cpu*_temp_mc` / `cpullc*_temp_mc` columns for CPU).
- Mean DDR decode: mean of `ddr_temp_mc` for samples with `wall_clock - t0 >= 60 s` (skips warm-up/prefill). For wave11 PPL runs `decode_ms` is recorded as 0 since each iter is teacher-forced, so we use the full sustained portion of the run as the thermally relevant window.
- Max swap delta: `max(vmstat_pswpout) - min(vmstat_pswpout)` over the run (pages, 4 KB each).
- Min mem_avail GB: `min(mem_avail_kb)` / 1024^2.
- Watchdog: parses lines of the form `[ts] DDR=XXC -> tier=N ...` from `watchdog.log`; dwell time per tier accumulates wall-clock seconds between consecutive events.

