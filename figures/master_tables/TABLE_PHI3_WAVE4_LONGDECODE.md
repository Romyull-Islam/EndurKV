# Phi-3 long-decode-dominated workload (Wave-4)

**Setup:** Phi-3-mini-128k-instruct-Q4_K_M, **short prompt (~500 tokens)** +
**2048-token decode** per iter with `--ignore-eos`, 4-thread CPU, DVFS-pinned at
1.63 GHz performance governor, strict cool-down to 33 °C between cells.
60-min budget per cell. `wave4_longdecode_1780750084`.

**Hypothesis tested:** in a regime where decode dominates the cell time and the KV
cache grows continuously (vanilla: 500 → 2548 cells), can a bounded-cache eviction
policy (v1 K=512) prevent the thermal throttle that no-eviction policies hit?

## Per-cell results

| Policy | Iters | Iter 1 → last decode (tok/s) | Decay | Peak DDR | Peak CPU | CPU freq range | Min free RAM | Swap-out |
|---|---|---|---|---|---|---|---|---|
| **vanilla** (FA-on) | 9 | 6.05 → 4.08 | **−33%** | **62.9 °C** ❌ | **66.4 °C** ❌ | **883–1632 MHz** | 7.59 GB | 10 MB |
| **v1 K=512** (FA-off + continuous evict) | 5 | 2.86 → 2.56 | **−10%** | **54.4 °C** ✓ | **57.9 °C** ✓ | **1267–1632 MHz** | 7.08 GB | 0 MB |
| **v1_FA K=512** (FA-off prefill + FA-on decode, frozen mask) | 8 | 5.85 → 4.19 | **−28%** | 62.5 °C ❌ | 66.0 °C ❌ | 883–1632 MHz | 7.37 GB | 0 MB |

## Per-iter decode trajectory (the real story)

```
            iter:    1     2     3     4     5     6     7     8     9
vanilla:           6.05  5.96  5.28  5.18  5.16  4.11  4.69  4.54  4.08
                                                 ↑ kernel throttle event (freq → 1.0 GHz)

v1 K=512:          2.86  2.72  2.55  2.63  2.56                            ← steady, no throttle

v1_FA K=512:       5.85  5.17  5.17  3.83  4.18  4.19  4.60  4.19         ← same pattern as vanilla
                                          ↑ kernel throttle event
```

## Key findings

### 1. v1 successfully prevents the throttle event
- Peak DDR **8.5 °C cooler** than vanilla (54.4 vs 62.9)
- Peak CPU **8.5 °C cooler** (57.9 vs 66.4)
- CPU freq **NEVER drops below 1.27 GHz** vs vanilla's 0.88 GHz throttle floor
- Decode decay only **−10%** vs vanilla's −33%

**This is the dissertation's thermal-control headline finding.** When the bounded cache
keeps DRAM bandwidth bounded, the kernel never needs to throttle.

### 2. v1_FA's frozen-mask design fails in this regime
`v1_FA` uses `no_evict_decode = true` so the eviction mask is frozen at end-of-prefill.
In long-decode workloads, **the cache grows during decode just like vanilla** (peak_kv =
2311 in both cases). The FA-on decode is fast but the bandwidth growth still hits the
thermal envelope, and v1_FA throttles at iter 4 with the same severity as vanilla iter 6.

**Implication:** v1_FA is the right answer for prefill-heavy workloads (Wave-3). v1 is
the right answer for decode-heavy workloads (Wave-4). Track-2 must observe the workload
shape at runtime to pick.

### 3. v1 trades raw throughput for thermal headroom
| Policy | Total tokens per 60 min | Average tok/min |
|---|---|---|
| vanilla | 18,432 | 307 |
| v1 K=512 | 10,240 | 171 |
| v1_FA K=512 | 16,384 | 273 |

vanilla still wins raw throughput-per-wall-clock even while throttling, because vanilla's
FA-on per-step compute is intrinsically faster than v1's FA-off per-step compute.

But:
- vanilla's chip is at 62.9 °C / 66.4 °C → no thermal headroom for other apps
- v1's chip is at 54.4 °C / 57.9 °C → plenty of headroom for background work
- vanilla's freq is unstable (oscillating 0.88-1.6 GHz) → unpredictable latency
- v1's freq is stable (1.27-1.6 GHz) → predictable latency

For mobile workloads where the system is shared between the LLM and other apps
(notifications, foreground apps, etc.) and where latency predictability matters,
v1's bounded-thermal behavior is the practical win.

### 4. Memory pressure is benign in this regime
Min free RAM stayed > 7 GB across all cells. No swap-out activity (vs Wave-3's 528 MB
for v1_FA). This is because the long-decode regime has a much smaller total cache
footprint (~3.7 GB peak for vanilla, ~190 MB for v1).

The file-backed state-swap fix (deployed) won't show measurable benefit here because
no memory pressure was triggered. The fix matters in Wave-3's regime (long prefill,
large cache, narrative QA) where the memory peak forced spillover.

## What the dissertation now says cleanly

| Workload regime | Best policy | Why |
|---|---|---|
| Long context, short answer (Wave-3 narrativeqa) | **v1_FA** | FA-on decode is fast, brief decode doesn't hit thermal limit; state-swap memory cost amortized over long prefill |
| Short prompt, long generation (Wave-4 essay) | **v1** | Continuous decode-time eviction caps DRAM bandwidth, prevents kernel throttle |
| Llama-1B / smaller models | any policy is fine | Workload doesn't push the thermal envelope |
| Multi-turn chat with cache reuse | **v1** | Cache grows continuously, same dynamics as Wave-4 |

## Track-2 closed-loop authority

Wave-4 finally gives Track-2 a real control variable: **vanilla DDR climbed 8.5 °C above v1's**.
Track-2 can now meaningfully:
- Choose K (eviction budget) based on observed DDR temperature
- Switch from v1_FA to v1 when observed cache growth indicates a long-decode trajectory
- Trade decode throughput for thermal stability when the controller's margin shrinks

Without Wave-4, the controller had nothing to grab onto (Wave-3 vanilla and v1 differed
by ~3 °C with no throttle on either). Wave-4 shows the **regime where the controller's
choices matter at the kernel-throttle level**.

## Files

- Source data: `phone-logs/wave4_longdecode_1780750084/`
- Per-cell `stress.csv` (decode_tps per iter), `sensors.csv` (5 Hz thermal/freq/memory)
- Companion docs: `TABLE_PHI3_WAVE3.md`, `MEMORY_SWAP_DISCUSSION.md`
