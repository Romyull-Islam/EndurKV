# Wave-11 Phi-3-mini-128k Q4_K_M — Comprehensive Comparison Table

**Held-out PPL on WikiText-2 chunk-pair (n=8 disjoint pairs). All cells on OnePlus 15 / Snapdragon 8 Elite Gen 5, CPU-only, 4 threads, 1.5 GHz kernel cap.**

| Policy | K | n | PPL | 95% CI | Δ vs vanilla | Peak DDR | Peak CPU | Swap | Min mem | Peak RSS | Prefill | Decode | Decode tps | TTFT | Total Lat | Prefill drift | Honest throttle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **vanilla** | ∞ | 8 | 5.464 | [4.50, 6.55] | — | 64.8°C | 68.6°C | **158 MB** ❌ | 6.89 GB | 4.03 GB | 305s | 731s | 2.97 | **305s** ✓ | 1036s | +17.2% | **MODERATE** |
| **h2o** | 512 | 7 | **5.474** | [4.60, 6.59] | **+0.2%** ✓ | 62.9°C | 65.9°C | **0 MB** ✓ | 6.74 GB | 4.14 GB | 366s | 1158s | 2.02 | 366s | **1524s** ❌ | **+31.2%** ❌ | **SEVERE** |
| **tova** | 512 | 7 | 5.627 | [4.75, 6.73] | +3.0% | **59.4°C** ✓ | **62.8°C** ✓ | 48 MB | 6.62 GB | 4.10 GB | 348s | 1046s | 2.24 | 348s | 1394s | +19.8% | MODERATE |
| **streamingllm** | 512 | 6 | 5.714 | [4.73, 7.00] | +4.6% | 59.4°C | 62.8°C | 72 MB | 6.57 GB | 4.09 GB | 328s | 1039s | 2.26 | 328s | 1367s | **+13.1%** ✓ | **MILD** ✓ |
| **v1_fa2_stack** (ours) | 512 | 8 | 6.082 | [5.12, 7.20] | +11.3% | 66.0°C | 69.0°C | **0 MB** ✓ | **7.05 GB** ✓ | **3.67 GB** ✓ | 362s | **470s** ✓ | **4.98** ✓ | 362s | **832s** ✓ | +18.0% | MODERATE |

## Important methodology note

**`Honest throttle` column** reflects multi-signal analysis (prefill drift across chunks + cpu6_freq distribution + DDR thermal envelope), NOT just kernel `cool_state`.

- `cool_state` stayed 0 in EVERY cell (kernel thermal mitigation never engaged)
- BUT prefill_ms grew +13–31% across chunks, indicating workload-emergent throttle from Qualcomm BCL/DCVS (invisible to /sys/class/thermal)
- v1_fa2_stack ran at **mean 1.298 GHz throughout** (vs 1.5 GHz cap) — 49.5% of CPU6 samples below 1.3 GHz
- streamingllm has the cleanest freq profile (only 0.1% below 1.3 GHz)
- h2o has the WORST prefill drift (+31.2% monotonic) — likely cell aborted before iter 7

## Single-axis winners

| Metric | Winner | Value |
|---|---|---|
| Best PPL | vanilla | 5.464 |
| Best PPL among evicting | **h2o** | 5.474 (matches vanilla) |
| Best DDR thermal | **tova / streamingllm** | 59.4°C |
| Best CPU thermal | tova / streamingllm | 62.8°C |
| Best Swap (endurance) | **h2o / v1_fa2_stack** | 0 MB |
| Best memory headroom | **v1_fa2_stack** | 7.05 GB |
| Best Peak RSS | **v1_fa2_stack** | 3.67 GB |
| Best Decode throughput | **v1_fa2_stack** | 4.98 tps |
| Best TTFT | vanilla | 305s |
| Best Total wall latency | **v1_fa2_stack** | 832s (−20% vs vanilla) |
| Best Prefill drift (least throttle) | **streamingllm** | +13.1% |
| Best honest throttle verdict | **streamingllm** | MILD |

## The 5-policy summary (TL;DR slide)

- **vanilla**: best PPL + fastest TTFT, but uses full cache (158 MB swap)
- **h2o**: matches vanilla PPL but severe prefill drift, slowest total wall
- **tova**: best DDR thermal, mid-range PPL
- **streamingllm**: cleanest throttle behavior, mid-range PPL
- **v1_fa2_stack**: best decode throughput + total wall + memory footprint, +11% PPL cost
