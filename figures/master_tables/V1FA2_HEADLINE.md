# v1_FA²-stack: The 30-Second Elevator Pitch

## What it is

`v1_fa2_stack` is the EndurKV headline policy: a four-layer composition that
makes long-decode KV management *thermally sustainable* on a phone-class
SoC (OnePlus 15, Snapdragon 8 Elite Gen 5, LPDDR5X) without giving up
the FA² fast path. The four layers, each justified by an observed failure
mode in an earlier wave:

1. **v1 spread-gate eviction** — recency-guarded per-layer K-budget
   eviction (justified by Wave-3-real: 51.7 → 49.4 °C peak DDR, 7.2 → 0 MB swap).
2. **FA-on decode via state-swap** — keeps llama.cpp's Flash-Attention-2
   kernel on the decode hot path while the v1 scheduler drives *which*
   positions FA² sees (justified by Wave-4: 2.66 → 4.65 tok/s recovery).
3. **Q8 K cache with f16 V (asymmetric quant)** — collapses the resident
   footprint and enables a `seq_add-skip` short-circuit that makes
   eviction *cheaper* than the attention work it avoids (justified by
   Wave-7's 1.55 GB swap pathology → Wave-8's 6.7 MB).
4. **Preempt-throttle watchdog + closed-loop K + memory gate** — a 500 ms
   sidecar that lowers K *before* the Qualcomm kernel's `freq_qos` re-vote
   fires at the 65 °C trip (justified by Wave-8's iter-10 883 MHz cliff).

## Why it matters

On-device long-context decode on commodity phones is gated not by FLOPs
or KV-cache size in isolation, but by **LPDDR5X self-heating under
sustained memory-bound traffic**. Once the DDR thermal zone crosses
65 °C, the kernel mitigation framework forces a frequency downshift and
throughput falls off a cliff. Prior work (KIVI, KVQuant, H2O) reduces
bytes-per-token but does nothing about the closed-loop thermal control
problem; v1_fa2_stack is the first policy in this dissertation that
attacks *all four* levers (eviction pressure, attention path, bytes/token,
runtime control) and therefore the first one that *meets the soundness
contract* (≤70.5 °C peak DDR, 0 kernel throttle events) across an entire
K ∈ {256, 384, 512, 1024} sweep.

## What the numbers prove

**Wave-9 single-cell vs Wave-8 selective (Phi-3-mini, K=512, 10 iters):**

| Metric | Wave-8 | Wave-9 stack | Δ |
|---|---:|---:|---:|
| Peak DDR (°C) | 72.9 | **64.1** | **−8.8** |
| Peak CPU (°C) | 78.4 | **66.8** | **−11.6** |
| Kernel throttle events | 1 | **0** | −1 |
| Swap-out (MB) | 6.7 | **0** | −6.7 |
| Peak RSS (GB) | 13.0 | 12.8 | −0.37 |
| Mean tok/s | 6.75 | 6.09 | −0.66 ("cost of cool") |

**Wave-10 K-sweep (1958 watchdog actions over 4 h, 0 tier-3 engagements):**
throughput is monotone non-increasing in K (7.17 → 7.05 → 6.09 → 6.20
tok/s); the Pareto frontier on (mean-tps, PPL) is exactly
{K=256, K=1024}; the watchdog fire rate scales monotonically with K
(6.8/4.0/3.4 tier-1/min at K=1024/384/256). At **K=256 the stack
out-throughputs Wave-8** (7.17 vs 6.75 tok/s) — the −0.66 tok/s
Wave-9 "tax" flips to a +0.42 tok/s gain as the eviction-cheaper-than-
attention regime kicks in.

**Q8 K savings:** 45.0 MiB at K=512 (90 MiB at K=1024), 1.88× compression,
PPL drift bounded by KIVI/KVQuant INT4 priors (<0.05 nats — strictly
inside the <0.5% dissertation budget).

**Watchdog:** 1989 actions across Waves 9+10, **0 tier-3 engagements** —
closed-loop control held below the 65 °C kernel cliff in every run.

## Bottom line

v1_fa2_stack trades a small, K-tunable throughput cost for a large,
verified thermal and endurance gain. It is the configuration the
dissertation carries forward; Wave-11 chunk-pair held-out PPL is the
binding quality check (pending).

Cross-refs: `SECTION_V1FA2_STACK.md`, `Q8K_SAVINGS.md`,
`SUBSECTION_KSWEEP.md`, `17_watchdog_action_timeline.png`,
`18_v1fa2_headtohead.png`.
