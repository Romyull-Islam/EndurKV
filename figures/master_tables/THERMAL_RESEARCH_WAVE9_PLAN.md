## Wave-9 60-Minute Cell Plan

### Configuration
- **Model / workload**: Phi-3-mini-4k-instruct (same GGUF as Wave-8), identical long-decode prompt set, identical seed, target 11+ iters in 60 min.
- **Inherited from Wave-8**: v1_FA2 selective attention, CPU freq pin 1632/1497 MHz, screen off, airplane mode, cold soak to DDR <= 35 C before start.

### Techniques enabled (stacked)
1. **INT8 KV cache** - launcher flags: `--flash-attn --cache-type-k q8_0 --cache-type-v q8_0`. Verify FA2 path engages by grepping `flash_attn = 1` and `cache_type_k = q8_0` in llama.cpp log.
2. **Devfreq DDR bus cap** - pre-flight `ls /sys/class/devfreq/ | grep -E 'llcc|ddr|bw'` to enumerate nodes on 8 Elite Gen 5. For each of `cpu-cpu-llcc-bw` and `cpu-llcc-ddr-bw`: `echo performance > governor`, then write `max_freq = floor(0.75 * peak_available_freq)` and `min_freq = max_freq` to pin. Run a 2 s background asserter loop that re-writes the cap (defeats Qualcomm thermal HAL re-vote).
3. **Closed-loop K controller** - sidecar thread polls `/sys/class/thermal/thermal_zone*/temp` for the DDR zone every 500 ms. States with 5 C hysteresis and 10 s min-dwell:
   - T < 62 C -> K = 512 (Wave-8 value, untouched)
   - 62 <= T < 66 C -> K = 384
   - T >= 66 C -> K = 256
   K is written into the v1_FA2 selective eviction budget via a small ioctl/shared-mem hook to the runtime. KV compaction is incremental (drop lowest-importance tokens; never rebuild the full cache mid-decode).

### Instrumentation
- 1 Hz: DDR temp, CPU big/perf temps, current CPU freq, current DDR/LLCC bus freq, current K, controller state.
- Per-iter: tok/s, PPL on the fixed 256-token held-out continuation.
- Per-token (cheap): emitted-token count, K-at-emit-time.
- Kernel throttle events from `dmesg | grep -iE 'throttle|cooling'`.

### Expected results (point estimates with confidence bands)
| Metric | Wave-8 baseline | Wave-9 target | Wave-9 acceptance band |
| --- | --- | --- | --- |
| Peak DDR | 72.9 C | **68.5 - 70.0 C** | <= 70.5 C |
| Peak CPU | 78.4 C | 77 - 79 C | <= 80 C |
| Mean tok/s | 6.75 | **6.5 - 6.9** | >= 6.2 |
| PPL | 3.56 | 3.56 - 3.61 | <= 3.64 |
| Iters in 60 min | 11 | 11 - 12 | >= 10 |
| Kernel forced 883 MHz throttles | 1 (iter 10) | **0** | <= 1 |

### Failure-mode detection (what tells us we picked wrong)
- **PPL > 3.64 with controller in K=512 most of run** -> Q8 KV broke FA2 path (verify FA flag actually took effect, not a silent fallback).
- **DDR cur_freq above the cap for >5% of samples** -> Qualcomm RPMh/AOP overrode the bus cap; the 2.5 C DDR delta in the stack is fake. Pivot to thermal-zone DDR cooling-device cur_state override in Wave-10.
- **Iters < 10 with throughput < 6.0 tok/s** -> bus cap is too aggressive given Q8's bandwidth reduction. Re-run at 85% of peak bus freq.
- **Controller never leaves K=512** -> DDR thermal zone path wrong, or 75% bus cap already over-cooled and Q8 made it cooler still; either way the Track-2 control story has not been exercised. Re-run with a hotter starting soak or tighter thresholds (60/64 C).
- **PPL > 3.70 with controller frequently at K=256** -> 256 budget is below Phi-3's needle-retrieval floor. Raise the hot setpoint to K=320 in Wave-10.
- **Peak CPU > 81 C** -> Q8 dequant NEON work shifted heat to CPU faster than DDR cooled. Stack with the CPU-side cgroup quota technique in Wave-10.
- **DDR delta < 1.0 C vs Wave-8** -> stack is fighting itself or measurements drifted; run an ablation cell with controller disabled (fixed K=512) to attribute.

### Ablation arms (only if time permits in a follow-up cell)
- A: Q8 KV alone (isolate quantization contribution).
- B: Q8 KV + bus cap, no controller (isolate closed-loop contribution).
- C: full stack (this cell).
This A/B/C decomposition is critical for the dissertation's composability claim.
