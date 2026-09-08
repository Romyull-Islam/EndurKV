# Wave-11 Comprehensive Cell Comparison

Source: `/home/mislam22/EndurKV_workspace/phone-logs/wave11_eval_1780862534`

Methodology: arithmetic mean of per-chunk perplexities (post-fix canonical convention used in `WAVE11_PHI3_PPL_LIVE`); bootstrap 1000-sample 95% CI in log-PPL domain over per-chunk PPLs (deterministic seed). Δ% vs vanilla on **common chunk prefix** (arithmetic mean: `(mean(policy) - mean(vanilla[:n])) / mean(vanilla[:n]) * 100`). Sensors min/max from cell `sensors.csv`; swap MB = `(vmstat_pswpout_last - first) * 4096 / 1e6`. Peak RSS from `stress.csv:peak_rss_kb`. Wall latency = last `t_elapsed_s` in stress.csv. Prefill/Decode in seconds (per-chunk mean from `meta.json:prefill_ms` and `meta.json:(total_ms - prefill_ms)`). Decode tps = `n_decode_steps / decode_s` per chunk, averaged. TTFT = mean(prefill_ms)/1000 (s). Throttled = any `cpu*_cool_state > 0` observed.

| Policy | Model | Bench | n | PPL | CI | Δ% | peak_DDR | peak_CPU | Swap_MB | Min_mem_avail | Peak_RSS | mass_retained | retention_ratio | cache_eff | Prefill_s | Decode_s | Decode_tps | TTFT_s | Total_lat_s | Throttled |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| vanilla | Phi-3-mini-128k | ppl | 8 | 5.707 | [4.590, 6.596] | — | 64.8 | 70.9 | 165.6 | 6.89 | 4.032 | 1.0000 | 1.0000 | 1.000 | 305.0 | 730.8 | 2.967 | 305.0 | 8299 | no |
| h2o | Phi-3-mini-128k | ppl | 7 | 5.769 | [4.689, 6.699] | +2.7% | 62.9 | 67.4 | 0.0 | 6.74 | 4.136 | 0.8094 | 0.2059 | 3.982 | 366.3 | 1157.8 | 2.020 | 366.3 | 10681 | no |
| v1_fa2_stack | Phi-3-mini-128k | ppl | 8 | 6.392 | [5.316, 7.331] | +12.0% | 66.0 | 70.5 | 61.1 | 7.05 | 3.671 | 1.0000 | 1.0000 | 1.000 | 361.5 | 470.3 | 4.984 | 361.5 | 6669 | no |
| tova | Phi-3-mini-128k | ppl | 7 | 5.918 | [4.848, 6.842] | +5.4% | 59.4 | 64.7 | 50.7 | 6.62 | 4.096 | 0.9611 | 0.2138 | 4.543 | 348.4 | 1046.0 | 2.243 | 348.4 | 9772 | no |

## Post-fix canonical numbers (Phi-3-mini-128k, Wave-11)

| Policy | mean PPL (arithmetic) | Δ% vs vanilla (common prefix) | n_chunks |
| --- | ---: | ---: | ---: |
| vanilla | 5.71 | — | 8 |
| h2o | 5.77 | +2.7% | 7 |
| v1_fa2_stack | 6.39 | +12.0% | 8 |
| tova | 5.92 | +5.4% | 7 |

## RES_SCHEMA

Per-cell record:
```
{
  policy:           string,         # vanilla | h2o | v1_fa2_stack | tova
  model:            string,         # Phi-3-mini-128k
  bench:            string,         # ppl
  n:                int,            # n_chunks completed (rows in stress.csv ≈ iter*/meta.json)
  PPL:              float | null,   # arithmetic mean of per-chunk meta.json:perplexity
  CI:               [low, high] | null,  # 95% bootstrap CI (1000 resamples) over log(per-chunk PPL); exponentiated for display; seed=0
  delta_pct:        float | null,   # 100 * (mean(policy[:n]) - mean(vanilla[:n])) / mean(vanilla[:n]); null for vanilla
  peak_DDR:         float | null,   # max(sensors.csv:ddr_temp_mc)/1000  → °C
  peak_CPU:         float | null,   # max over all cpu*_temp_mc columns / 1000 → °C
  Swap_MB:          float | null,   # (vmstat_pswpout_last - first) * 4096 / 1e6
  Min_mem_avail:    float | null,   # min(sensors.csv:mem_avail_kb) / 1024² → GB
  Peak_RSS:         float | null,   # max(stress.csv:peak_rss_kb) / 1024² → GB
  mass_retained:    float | null,   # mean(iter*/meta.json:mean_mass_retained)
  retention_ratio:  float | null,   # mean(iter*/meta.json:mean_retention_ratio)
  cache_eff:        float | null,   # mean(iter*/meta.json:mean_eviction_efficiency)
  Prefill_s:        float | null,   # mean(iter*/meta.json:prefill_ms) / 1000
  Decode_s:         float | null,   # mean((total_ms - prefill_ms) / 1000) per chunk
  Decode_tps:       float | null,   # mean(n_decode_steps / decode_s) per chunk; null if all decode_s ≤ 0
  TTFT_s:           float | null,   # mean(prefill_ms)/1000 — time-to-first-(scored)-token for PPL bench
  Total_lat_s:      int | null,     # last(stress.csv:t_elapsed_s) — cell wall-clock
  Throttled:        "no" | "yes (peak cool_state=N)" | "N/A"
}
```

Field provenance (Wave-11 Phi-3-mini-128k):

| field | source files | aggregation |
| --- | --- | --- |
| `n` | `<cell>/iter*/meta.json` | dir count |
| `PPL` | `<cell>/iter*/meta.json:perplexity` | arithmetic mean (post-fix canonical convention) |
| `CI` | `<cell>/iter*/meta.json:perplexity` | bootstrap 1000-sample 95% CI on log(PPL), exponentiated; seed=0 |
| `Δ%` | per-chunk PPL of policy and vanilla on common prefix `n_p = min(n_policy, n_vanilla)` | `100 * (amean(policy[:n_p]) - amean(vanilla[:n_p])) / amean(vanilla[:n_p])` |
| `peak_DDR` | `<cell>/sensors.csv:ddr_temp_mc` | `max / 1000` (°C) |
| `peak_CPU` | `<cell>/sensors.csv:cpu*_temp_mc` | `max / 1000` across all CPU temp columns (°C) |
| `Swap_MB` | `<cell>/sensors.csv:vmstat_pswpout` | `(last - first) × 4096 / 1e6` |
| `Min_mem_avail` | `<cell>/sensors.csv:mem_avail_kb` | `min / 1024²` (GB) |
| `Peak_RSS` | `<cell>/stress.csv:peak_rss_kb` | `max / 1024²` (GB) |
| `mass_retained`, `retention_ratio`, `cache_eff` | `<cell>/iter*/meta.json:mean_{mass_retained, retention_ratio, eviction_efficiency}` | mean across chunks |
| `Prefill_s` | `<cell>/iter*/meta.json:prefill_ms` | `mean / 1000` |
| `Decode_s` | `<cell>/iter*/meta.json:(total_ms - prefill_ms)` | mean of per-chunk diff (s) |
| `Decode_tps` | `<cell>/iter*/meta.json:n_decode_steps / decode_s` | mean across chunks |
| `TTFT_s` | `<cell>/iter*/meta.json:prefill_ms` | `mean / 1000` (TTFT = prefill end for teacher-forced PPL) |
| `Total_lat_s` | `<cell>/stress.csv:t_elapsed_s` | `last` (cumulative wall-clock) |
| `Throttled` | `<cell>/sensors.csv:cpu*_cool_state` | `OR_reduce(any cool_state > 0)`; reports peak cool_state when throttled |
