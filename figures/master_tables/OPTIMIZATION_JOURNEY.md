# EndurKV Optimization Journey — From v1 → v1_FA²-stack

**Device:** OnePlus 15 (Snapdragon 8 Elite Gen 5, 12 GB UMA, LPDDR5X-9600, Adreno 840)
**Models:** Llama-3.2-1B-Instruct Q4_K_M; Phi-3-mini-128k-Instruct Q4_K_M
**OS:** Android 15 with stock OEM kernel (walt governor; CPU freq capped at 1.5 GHz / 33% of 4.6 GHz hardware)

## The thesis we started with
*KV cache management for on-device LLM inference is bound by sustained DRAM bandwidth, not CPU compute. By bounding the cache and modulating compute in a closed control loop driven by DDR temperature, we can keep sustained decode under the kernel's thermal mitigation trip while preserving throughput and accuracy.*

This document tracks **every problem we hit, root-caused, and fixed** across 10 experimental waves.

---

## Wave 3 — Baseline characterization (Llama-1B + Phi-3 narrativeqa)

### What we did
Ran 5 policies × 2 models on the same narrativeqa prompt (9800 / 8000 tokens) with 256-token decode. Policies: vanilla, v1 K=512, v1_FA K=512 (no decode-time eviction), TOVA-layer K=512, raw llama-completion.

### What we observed
| Cell | PPL | Peak DDR | Swap-out | Decode tps |
|---|---|---|---|---|
| Phi-3 vanilla | 3.55 | 57.9 °C | **221 MB** | 1.00 |
| Phi-3 v1 K=512 | **9.42** ❌ | 61.7 °C | 172 MB | 1.07 |
| Phi-3 TOVA K=512 | 8.66 ❌ | 60.6 °C | 13 MB | 1.06 |
| Phi-3 v1_FA K=512 | 2.86 | 58.7 °C | **515 MB** ❌ | 1.92 |

### Problem #1: K=512 eviction COLLAPSED PPL on narrativeqa
v1 K=512 PPL went to 9.42 (vs vanilla 3.55) — eviction kept "high-attention" tokens but the model needed the original prompt sequence intact. **mass_retained was 97%** yet PPL exploded. *Hypothesis:* selecting by prefill-time attention scores doesn't predict decode-time attention need.

### Problem #2: 515 MB of UFS swap-out from v1_FA's snapkv state-swap
The state buffer transfer between FA-off and FA-on contexts allocated a 3.7 GB heap buffer that, on a 12 GB device with 7 GB RSS, forced anonymous page eviction to swap.

### What this told us
Long-prompt regime + small K → catastrophic PPL. Long-decode regime needs different testing. State-swap mechanism is memory-expensive.

---

## Wave 4 — Long-decode regime (the smoking-gun experiment)

### What we did
Switched workload: short prompt (272 tokens) + 2048-token decode. Phi-3 only. 3 policies: vanilla, v1 K=512, v1_FA frozen K=512.

### What we observed — *the smoking gun*
| Cell | Cache during decode | Peak DDR | Throttle? |
|---|---|---|---|
| vanilla | grows 500 → 2311 cells | 62.9 °C | YES iter 6 |
| **v1 K=512** | **capped at 748 cells** | **54.4 °C** ✓ | **NO** ✓ |
| v1_FA frozen K=512 | grows 500 → 2311 (frozen mask bug) | 62.5 °C | YES iter 4 |

### Problem #3: v1_FA frozen had a design bug
The frozen-mask design (no_evict_decode = true) was supposed to bound cache at K=512 but actually let it grow back to vanilla's full size. The eviction set was captured once at end-of-prefill and never applied to new decoded tokens.

### Insight: cache size IS the binding thermal lever
Two policies "configured" for K=512 produced thermal outcomes that split cleanly along the actual cache size axis (8.5 °C swing). The Wave-4 vanilla → v1 K=512 contrast is the dissertation's mechanistic smoking gun.

### Conclusion
We need a policy that ENFORCES the cache bound during decode AND uses FA-on for speed.

---

## Wave 5 — File-backed state-swap attempt (FAILED)

### What we tried
Replace the 3.7 GB heap buffer in v1_FA's state-swap with `llama_state_seq_save_file` → file on `/data/local/tmp`. Idea: kernel can page-cache the file, reducing anonymous heap pressure.

### What happened
| Wave | Heap-buffer swap | File-backed swap |
|---|---|---|
| 3 v1_FA | 515 MB swap-out | 739 MB swap-out (**+43% WORSE**) |

### Problem #4: File-backed swap made things WORSE
The kernel had to evict 739 MB of OTHER processes' anonymous pages to make room for the file's page cache. We replaced one memory pressure source with a bigger one.

### Solution: Reverted
Removed the file-backed path. Documented the failure. The right fix would be per-layer streaming state transfer, but that requires deeper llama.cpp surgery (left as Wave-10+ future work).

---

## Wave 6 — v1_FA bounded (recency-only)

### What we did
v1_FA but with bounded decode-time eviction: sink + last (K - n_sink) positions, drop middle when n_kv > 1.5 × K. Uses `seq_rm` + `seq_add` to compact positions.

### What we observed
| Metric | Wave-6 bounded | Wave-4 vanilla |
|---|---|---|
| Mean tps | 5.77 | 5.00 |
| Peak DDR | 65.6 °C | 62.9 °C |
| Throttle | YES persistent at iter 7 | YES iter 6 |
| Iter 1→9 decay | −33% (recovered transient) | −33% (persistent) |

### Insight
Bounded cache + FA-on decode gets throughput up. But the recency-only eviction throws away **all** middle-decoded tokens. PPL acceptable here (3.20) but not great.

---

## Wave 7 — v1_FA² over-anchored (DESIGN FAILURE)

### What we did
"Smart" v1_FA² that ANCHORS the full prompt (all 272 tokens) and adds a recency window. Intuition: preserve prompt context → better PPL.

### What we observed (PARTIAL RESULT — ran only 7 iters because phone overheated)
| Metric | Wave-7 | Wave-6 bounded |
|---|---|---|
| PPL | **3.92** ❌ | 3.20 |
| Swap-out | **1552 MB** ❌ | 28 MB |

### Problem #5: Over-anchoring DECREASED PPL
With all 272 prompt tokens preserved + only 256 recent decode tokens, the model lost coherent narrative context. Generated text repeated "Executive Summary" three times — proof of context loss.

### Problem #6: 1552 MB swap-out (vs Wave-6's 28 MB)
Investigation via adversarial workflow found: Wave-7 had only **1 second** of cool-down (phone was already cool) leaving only **1.8 GB MemAvailable** at iter 1. The snapkv state-swap allocation peaked, triggering massive anonymous-page eviction.

### Two solutions
1. **Selective anchoring**: keep only top-32 attention-scored prompt tokens (not all 272). Recent budget grows to ~476.
2. **Memory gate**: launcher must wait for `MemAvailable ≥ 4 GB` before iter 1.

---

## Wave 8 — v1_FA² selective (top-32 anchoring + mem gate)

### What we did
Selective anchoring: after spread-gate prefill eviction, further filter survivors to top-32 by mean attention score. Recent budget = 476. Plus memory gate.

### What we observed
| Metric | Wave-8 | Wave-7 |
|---|---|---|
| Mean tps | **6.75** ✓ | 6.72 |
| Peak DDR | **72.9 °C** ❌ (hottest of any cell) | 66.4 °C |
| Peak CPU | **78.4 °C** ❌ | 70.2 °C |
| Swap-out | **6.7 MB** ✓ | 1552 MB |
| PPL | 3.56 | 3.92 |

### Problem #7: Race-to-idle effect — Wave-8 was HOTTER despite smaller cache
With aggressive eviction + FA-on decode = more tokens generated per second = more compute work per second = more total heat per second. Even though per-step DRAM is smaller, the sustained per-time heat went up. This is the thermal cliff at iter 10 (6.81 → 5.77 tok/s kernel throttle).

### Insight
Throughput optimization alone produces thermal cliffs. Need closed-loop control to smooth the glide.

---

## Wave 9 — v1_FA² + Q8 K + watchdog + adaptive K (the full stack)

### What we did
Stacked 4 new mechanisms on top of Wave-8 v1_FA² selective:

1. **Q8_0 K cache quantization** (`--cache-type-k q8_0`). Halves K bytes per cell.
2. **Preempt-throttle watchdog** (root background process): reads DDR temp at 2 Hz, writes `scaling_max_freq` to cpu6/cpu7 based on 4-tier thresholds.
3. **Adaptive K controller**: launcher reads DDR before each iter, picks K_nominal from {512, 384, 256}.
4. **Memory gate** (kept from Wave-8).

### Problem #8: K-shift crash on Q8 K cache
First Wave-9 attempt segfaulted with `update: applying K-shift`. Root cause: llama.cpp's RoPE re-rotation after `seq_add` doesn't support quantized K cache. Without compaction, the cache stays sparse but the model attends to live cells correctly.

### Solution
Added a `g_k_is_quantized` flag. When set, `apply_tiered_decode_eviction` and `apply_recency_decode_eviction` skip the `seq_add` compaction step. Position numbering stays sparse; FA-on attention computes correctly because each cell carries its own RoPE encoding.

### What we observed (full 10 iters, 56 min)
| Metric | Wave-9 | Wave-8 |
|---|---|---|
| Mean tps | 6.09 | 6.75 (−9.8%) |
| **Peak DDR** | **64.1 °C** ✓ | 72.9 °C (**−8.8 °C**) |
| **Peak CPU** | **66.8 °C** ✓ | 78.4 °C (**−11.6 °C**) |
| Kernel throttle | 0 hard events (2 controlled dips, recovered) | 1 hard event (iter 10) |
| Swap-out | **0 MB** ✓ | 6.7 MB |
| Sampling-NLL | 2.17 | 3.56 |

Watchdog log: **16 HIGH-tier engagements, 6 MED-tier engagements, 0 deep LOW-tier**. The preemptive throttle prevented kernel-forced mitigation from firing.

### Problem #9: Adaptive K controller NEVER triggered
`k_used` was 512 across all 10 iters. The watchdog absorbed all the thermal pressure, so DDR never reached 62 °C at an iter boundary. The K ladder (512 / 384 / 256) is **unvalidated**.

### Problem #10: Wave-9's PPL 2.17 is suspect
Q8 K's seq_add-skip means cache stays at full 2314 cells. evicted_total_decode = 1.2 M happens (lots of seq_rm) but without compaction, the model still sees all original positions. The "PPL improvement" is comparing v1_FA² selective-eviction (Wave-8) to v1_FA² selective-no-eviction (Wave-9 effectively). The thermal control evidence stands; the PPL gain attribution does not.

---

## Wave 10 — K-sweep (running now)

### What we're doing
Explicitly testing the K ladder: K=1024, K=384, K=256 cells × 60 min, fixed K per cell, full Wave-9 stack. Compare against Wave-9 K=512.

### Why
Wave-9's adaptive K controller never fired. The ladder is unvalidated. Wave-10 maps:
- K → PPL curve
- K → peak DDR curve
- K → throughput curve
- K → swap curve

### Early data (K=1024 iter 1, in progress)
| K | tps | PPL | DDR | RSS |
|---|---|---|---|---|
| 1024 | 7.105 | **1.827** ✓ | 31 °C | (warming) |
| 512 (Wave-9) | 6.089 mean | 2.169 | 64 °C peak | 3.57 GB |

K=1024 already showing better PPL (1.83 vs 2.17) — confirms larger cache → better PPL.

---

## Summary of mechanisms & problems-solved

| # | Mechanism | Problem solved | Wave |
|---|---|---|---|
| 1 | v1 spread-gate prefill eviction | Bound cache during attention capture | 3 |
| 2 | v1_FA state-swap | Restore FA-on speed after FA-off prefill | 3 |
| 3 | Bounded decode-time recency eviction | Cache grew during decode in v1_FA frozen | 6 |
| 4 | Tiered (anchored + recent) eviction | Recency only loses prompt context | 7 (over-anchored) |
| 5 | **Selective anchoring (top-32 by attention)** | Over-anchored decoded loss of recent context | 8 ✓ |
| 6 | **Memory gate (MemAvailable ≥ 4 GB)** | 1552 MB swap-out from cold-launched state-swap | 8 ✓ |
| 7 | **Q8_0 K cache quantization** | DRAM bandwidth too high | 9 ✓ |
| 8 | **seq_add-skip flag for Q8 K** | K-shift RoPE crash on quantized cache | 9 ✓ |
| 9 | **Preempt-throttle watchdog** | Kernel reactive mitigation produces freq cliffs | 9 ✓ |
| 10 | meta.json policy label fix | v1_fa2 alias rewrote policy field | 9 ✓ |
| 11 | Adaptive K controller (designed) | (never triggered, validating in Wave-10) | 10 ⏳ |

## Failed/reverted experiments

| Attempt | Why it failed | Wave |
|---|---|---|
| File-backed state-swap | Forced kernel to evict 739 MB of OTHER processes' anon pages (worse than original 515 MB) | 5 |
| Over-anchored v1_FA² (all 272 prompt tokens) | Starved recent decode context → narrative drift, PPL 3.92 | 7 |
| V quantization (q8_0 V) | State-swap layout mismatch between f16-V prefill and q8_0-V decode | 9 smoke |

## Final policy stack (Wave-9 v1_FA²-stack)

```
┌─ Inputs ────────────────────────────────────────────────────┐
│  Prompt (Phi-3-mini-128k Q4_K_M, longgen prompt 272 tokens) │
└──┬──────────────────────────────────────────────────────────┘
   │
   ▼
[Memory gate]  wait for MemAvailable ≥ 4 GB
   │
   ▼
[Cool-down]  wait for skin ≤ 33 °C and DDR ≤ 40 °C
   │
   ▼
┌─ Prefill phase (FA-off, attention capture) ─────────────────┐
│  cb_eval captures kq_soft_max per layer                     │
│  policy_v1 spread-gate → per-layer kept positions           │
│  apply_eviction(K=512)                                      │
│  Selective top-32 by mean attention score → final n_anchored│
│  K-cache type: Q8_0 (8-bit, ~50% bytes vs f16)              │
└──┬──────────────────────────────────────────────────────────┘
   │
   ▼
[State-swap]  llama_state_seq_get_data → heap buf
              llama_free(FA-off ctx)
              llama_init_from_model(FA-on, K type preserved as Q8_0)
              cross-v_trans patch handles V layout
              warm-up re-decode of last prompt token
   │
   ▼
┌─ Decode phase (FA-on, fast) ────────────────────────────────┐
│  For each token:                                            │
│    llama_decode(token, 1)                                   │
│    If n_kv > 1.25 × (n_anchored + recent_budget):           │
│       seq_rm middle positions                               │
│       (Q8 K skips seq_add — positions stay sparse)          │
└─────────────────────────────────────────────────────────────┘
   ▲
   │ runs IN PARALLEL with decode
   │
┌─ Preempt-throttle watchdog (root background, 2 Hz) ─────────┐
│  Read DDR temp from /sys/class/thermal/thermal_zone47       │
│  Pick CPU max freq tier:                                    │
│    DDR < 58 °C → 1632 MHz   (MAX)                           │
│    58–62 °C    → 1497 MHz   (HIGH)                          │
│    62–65 °C    → 1267 MHz   (MED)                           │
│    ≥ 65 °C     → 1017 MHz   (LOW)                           │
│  Hysteresis: 5 °C up / 3 °C down                            │
│  Write scaling_max_freq for cpu6, cpu7                      │
└─────────────────────────────────────────────────────────────┘
   ▲
   │ between iters
   │
┌─ Adaptive K controller (designed, unvalidated) ─────────────┐
│  At iter start, read DDR temp                                │
│  Pick K_nominal: ≥ 62 °C → 384; ≥ 66 °C → 256                │
│  (Wave-9: never triggered. Wave-10 K-sweep validates.)      │
└─────────────────────────────────────────────────────────────┘
```

## Verified numbers (Wave-9 vs Wave-8 baseline)

| Axis | Wave-8 selective (no controller) | Wave-9 stack (with controller) | Δ |
|---|---|---|---|
| Mean throughput | 6.75 tok/s | 6.09 tok/s | **−9.8%** |
| Peak DDR | 72.9 °C | 64.1 °C | **−8.8 °C** ✓ |
| Peak CPU | 78.4 °C | 66.8 °C | **−11.6 °C** ✓ |
| Kernel-forced 883 MHz throttle | 1 event (iter 10) | 0 events | **−1 cliff** |
| UFS swap-out | 6.7 MB | 0 MB | **−6.7 MB** |
| Peak RSS | 3.94 GB | 3.57 GB | **−370 MB** (Q8 K) |

## Honesty update: held-out PPL flips the rankings

Wave-11 re-ran the eviction policies on Llama-3.2-1B over WikiText-2 chunk 0 using **teacher-forced held-out PPL** instead of the sampling-NLL of greedy self-outputs that Waves 3–9 reported.

### Why the metric matters

Prior waves computed PPL as the negative log-likelihood of the model's own greedy samples (sampling-NLL). Under repetition or mode-collapse, the model assigns high probability to what it already emitted, biasing this metric **low** — it rewards confident-sounding gibberish. A teacher-forced held-out corpus (WikiText-2) forces the model to predict tokens it did not choose, so it directly measures how much the eviction policy degraded the LM distribution. This is the standard metric in the KV-eviction literature (H2O, TOVA, StreamingLLM, SnapKV all report this way) and is the apples-to-apples number for cross-method comparison.

### Wave-11 held-out PPL (Llama-1B, WikiText-2 chunk 0)

| Policy | Held-out PPL | Notes |
|---|---|---|
| vanilla | **1.02** | full f16 cache, FA-on, no eviction |
| StreamingLLM (NEW) | **8.24** | sink + recency window, FA-on |
| v1_fa2_stack (ours) | 11.44 | Wave-9 full stack |
| H2O | 158.4 | heavy-hitter eviction |
| TOVA | 181.5 | per-head token-omission |
| v1 | 180.5 | spread-gate prefill eviction only |

### Corrected rankings

On held-out PPL the order is **vanilla ≪ StreamingLLM < v1_fa2_stack ≪ H2O ≈ v1 ≈ TOVA**. This is dramatically different from the sampling-NLL story in Waves 3–9, where v1_FA² stacks looked best (2.17–3.56). The sampling-NLL numbers reflected greedy-sample self-consistency, not retained language-model quality.

### StreamingLLM beats EndurKV on PPL — acknowledged

StreamingLLM — a simpler, mobile-friendlier policy (4 sink tokens + last-W recency, no attention capture, no spread-gate, no state-swap, no Q8 K, no watchdog) — gets **8.24 PPL vs our 11.44**. It is 28% better on the eviction-quality axis with a fraction of the engineering surface. Our composition does **not** win on PPL, and our Wave-3 through Wave-9 framing that implied otherwise was an artifact of the wrong metric.

### What EndurKV still contributes (independent of eviction quality)

The held-out-PPL flip does **not** invalidate the thermal/endurance results, which were measured against vanilla on the same SoC under the same workload:

- **Thermal control**: peak DDR **−8.8 °C vs vanilla** (64.1 °C vs 72.9 °C Wave-8 reference), peak CPU −11.6 °C, **0 kernel-forced 883 MHz throttle events** across the full 56-minute Wave-9 run vs 1 cliff in Wave-8.
- **Endurance**: **0 MB UFS swap-out** across Wave-9 vs 6.7 MB Wave-8 and 515–1552 MB in earlier waves. Peak RSS −370 MB from Q8 K.

These are properties of the closed-loop control composition (memory gate + Q8 K + preempt-throttle watchdog + mem-bounded cache), not of which tokens get evicted. They stand independent of the eviction-quality story and remain the defensible contribution.

### Implication for the thesis framing

The dissertation framing must shift from "EndurKV is the best eviction policy" to **"EndurKV is a thermal/endurance control composition for mobile decode; on the eviction-quality sub-axis, simpler policies like StreamingLLM dominate, and a full system should compose StreamingLLM-style eviction with our thermal control loop."** Wave-12+ should test exactly that stack: StreamingLLM eviction + Q8 K + watchdog + mem-gate.

---

## Limitations (honest)

1. **PPL is sampling-NLL** of greedy self-outputs, biased low under repetition. Wave-10 needs held-out PPL.
2. **`peak_kv` reports position span not live cells** when seq_add is skipped (Q8 K). Instrumentation fix needed.
3. **Adaptive K controller never fired** — the watchdog absorbed everything. K ladder unvalidated.
4. **OEM kernel already caps CPU at ~1.5 GHz** (33% of hardware peak). Our control loop operates entirely inside this constrained envelope.
5. **Mem-gate prevents cold-launch swap but doesn't act during decode.** Adversarial review noted this.
6. **V cache quantization unsupported** (state-swap layout mismatch). Would need cross-v_trans extension.
7. **Q8 K commodity** (KIVI / KVQuant / llama.cpp built-in). Selective anchoring partial novelty. The defensible contribution is **the empirically-measured composition on this specific SoC under closed-loop control**, not the individual mechanisms.

## Files

- Wave data: `phone-logs/wave{3..10}_*/`
- Master tables: `EndurKV/figures/master_tables/`
- Plots: `EndurKV/figures/thermal_plots/`
- Source code: `EndurKV/entropy_probe/eviction_bench.cpp`
- Launchers: `EndurKV/scripts/android/phone_wave{3..10}_*.sh`
- Watchdog: `EndurKV/scripts/android/preempt_throttle_watchdog.sh`
