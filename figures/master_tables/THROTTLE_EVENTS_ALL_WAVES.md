# Throttle Events: Wave 3 - Wave 11 (master table)

Detection rules (any of):
- (a) cpu*_cool_state went 0 -> >=1 between adjacent iter decode windows
- (b) cpu6_freq_hz median dropped by >=10% between adjacent iters
- (c) decode_tps in stress.csv dropped by >=15% between adjacent iters (iter>=2)

Pre-throttle state = LAST CLEAN sensor sample inside the prior iter's decode window.

Total events detected: **7**

| wave | cell | iter | severity | signature | ddr C | cpu_big C | cpullc C | sysT2 C | skin C | batt C | I_bat mA | cpu6 Hz |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wave3_real | v1_K512 | 2 | severe | (b)cpu6_freq_drop:45.9% | 47.1 | 46.9 | 55.2 | 43.0 | 38.6 | 36.3 | 147 | 1632000 |
| wave4 | v1_fa_K512 | 4 | moderate | (c)decode_tps_drop:25.8% | 56.7 | 60.8 | 59.0 | 49.9 | 44.6 | 42.9 | 101 | 1382400 |
| wave4 | vanilla | 6 | mild | (c)decode_tps_drop:20.3% | 55.6 | 61.2 | 58.7 | 46.8 | 40.2 | 36.7 | 426 | 1632000 |
| wave6 | v1_fa_K512_bounded | 7 | mild | (c)decode_tps_drop:18.4% | 57.5 | 62.8 | 61.0 | 50.2 | 43.7 | 40.9 | 371 | 1497600 |
| wave8 | v1_fa2_selective | 10 | mild | (c)decode_tps_drop:15.3% | 49.0 | 55.0 | 52.5 | 44.6 | 39.0 | 35.5 | 392 | 1632000 |
| wave9 | v1_fa2_stack | 8 | moderate | (c)decode_tps_drop:25.9% | 58.3 | 63.2 | 61.7 | 49.4 | 42.7 | 39.8 | 461 | 1497600 |
| wave9 | v1_fa2_stack | 10 | moderate | (c)decode_tps_drop:25.4% | 58.7 | 64.0 | 61.4 | 49.7 | 42.9 | 39.9 | 461 | 1497600 |

## Empirical percentile distributions of pre-throttle values

| field | p10 (earliest safe trigger) | p50 (median throttle start) | p90 (late) | n |
|---|---:|---:|---:|---:|
| ddr_c | 48.2 | 56.7 | 58.5 | 7 |
| cpu_big_c | 51.8 | 61.2 | 63.5 | 7 |
| cpullc_c | 54.1 | 59.0 | 61.5 | 7 |
| systherm2_c | 44.0 | 49.4 | 50.0 | 7 |
| skin_c | 38.8 | 42.7 | 44.1 | 7 |
| battery_c | 36.0 | 39.8 | 41.7 | 7 |
| battery_current_ma | 128.6 | 392.0 | 461.0 | 7 |

## Recommended watchdog v2 thresholds (5 C below median throttle start)

| key | value |
|---|---:|
| ddr_warn_c | 51.7 |
| ddr_crit_c | 56.7 |
| cpu_warn_c | 56.2 |
| cpu_crit_c | 61.2 |
| skin_warn_c | 37.7 |
| battery_warn_c | 36.8 |
| battery_current_drop_ma | 392.0 |
