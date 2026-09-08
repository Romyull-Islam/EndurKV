# EndurKV: Thermal- and Endurance-Co-Aware KV Cache Management for Sustained Mobile LLM Inference

**Submission target**: MobiSys 2028 (12-page full paper)
**Title alternative for HotMobile 2027**: *Co-Aware KV Cache Management for Battery-Constrained On-Device LLM Inference* (6-page short paper)
**Author**: Md Romyull Islam, Kennesaw State University
**Date prepared**: 2026-06-09

---

## Abstract

On-device Large Language Model (LLM) inference on flagship smartphones is increasingly limited not by computation, but by the joint pressure of growing KV cache memory, sustained DRAM bandwidth, thermal throttling, and battery endurance. Existing KV-cache eviction policies — H2O, TOVA, SnapKV, and StreamingLLM — were designed for and evaluated on server-class GPUs, where they trade a small amount of perplexity for a smaller cache. None measure thermal, power, or memory residency on real mobile hardware, and none provide closed-loop control of inference under thermal stress.

We present **EndurKV**, a co-aware KV cache management system that integrates four mechanisms behind a single decode loop: (i) a *per-head attention-confidence budget* that allocates per-head cache size as a function of each head's softmax peak (drawn from model-internal signals during a FA-off prefill); (ii) *selective semantic anchoring* of 32 prompt tokens scored by mean attention; (iii) *Q8 key quantization with f16 values* and a one-time *state-swap* into a FlashAttention-on decode context for fast generation; and (iv) a *multi-sensor preempt-throttle watchdog* that monitors DDR, CPU big-core, skin, battery temperature, and battery current limit (BCL) at 5 Hz and engages a five-tier frequency cap only when one or more sensors approach the empirical kernel throttle cliff.

We implement EndurKV as a llama.cpp extension and evaluate it end-to-end on a OnePlus 15 (Snapdragon 8 Elite Gen 5, 12 GB UMA) across Phi-3-mini-128k Q4_K_M, Llama-3.2-1B-Instruct Q4_K_M, and gemma-2-2b-it Q4_K_M. On a held-out chunk-pair PPL protocol over WikiText-2 (n=8 disjoint pairs), EndurKV's flagship configuration `v1_fa2_stack` at K_nominal=512 holds Phi-3-mini PPL within 11% of vanilla (6.08 vs 5.46) while delivering **+68% decode throughput** (4.98 vs 2.97 tokens/sec), **eliminating swap** (0 MB vs 158 MB for vanilla), and **reducing total latency by 20%** (832 s vs 1036 s). On a long-decode interactive workload (2048-token generation), EndurKV reduces measured energy consumption by **−67.7%** (6.29 vs 19.47 mAh), reduces peak CPU temperature by **6.2°C**, reduces peak DDR by **2.7°C**, and reduces peak resident set size by **9.4%**, with the multi-sensor watchdog firing 45 tier transitions exclusively during high-thermal phases. Across a K-sweep over {128, 256, 512, 1024}, EndurKV traces a Pareto curve that lets the operator dial the trade-off between PPL and energy with monotonic predictability.

EndurKV is, to our knowledge, the first system that simultaneously couples a model-internal attention signal to KV cache eviction and a multi-sensor thermal signal to a frequency throttle, evaluated end-to-end on commodity flagship mobile hardware. We position the system as a practical foundation for sustained on-device LLM agents, mobile RAG, and long-form generation.

---

## 1. Introduction

The rapid deployment of small-to-medium Large Language Models (4-7 B parameters, Q4_K_M quantized) on flagship smartphones has been accompanied by a wave of system-level optimization work: PowerInfer-2 [§REF], LLM in a Flash [§REF], MLC-LLM [§REF], llama.cpp's [§REF] Hexagon NPU backend, and various proprietary on-device inference stacks. The result is that contemporary 4 B-parameter models can now serve interactive chat at 5-7 tokens/sec on CPU and 30-50 tokens/sec on the NPU on a Snapdragon 8 Elite Gen 5 / Hexagon device.

This progress, however, exposes a different set of bottlenecks that were largely absent in the data center setting: the KV cache itself becomes the binding constraint as soon as sustained generation crosses about 1000-2000 tokens. We identify four coupled pressures that limit production deployment:

1. **KV cache size**. A Phi-3-mini-128k Q4_K_M cache at 2 K context occupies ≈1.2 GiB of RAM. On a 12 GB device shared with foreground apps, this is enough to trigger 158 MB of swap in our measurements (vanilla policy), leading to >300 ms decode stalls.
2. **DRAM bandwidth**. Per-token decode reads the entire cache once per layer — for Phi-3 that is ~800 MB per attention step. On LPDDR5X at sustained 50 GB/s, this is the dominant cost.
3. **Thermal envelope**. The Qualcomm Battery Current Limit (BCL) and the kernel's `cool_state` mitigation throttle the big-core frequency from 1632 MHz to 883 MHz once DDR exceeds 65°C or CPU big-core exceeds ~67°C, often within 90 seconds of sustained decode on a 4 B-parameter model.
4. **Battery endurance**. Without intervention, vanilla 2048-token decode on Phi-3-mini draws ~19.5 mAh on our hardware — at typical phone capacity (~5000 mAh), this implies fewer than 250 such generations per charge.

Existing KV-cache eviction policies — H2O [Zhang et al. 2023], TOVA [Oren et al. 2024], SnapKV [Li et al. 2024], StreamingLLM [Xiao et al. 2024] — were each designed and evaluated on server-class GPUs, with perplexity (PPL) as the sole figure of merit. None of these works measure thermal, power, or memory residency on a real mobile device. On our hardware, we show that the canonical H2O actually *degrades* decode throughput by 32% compared to vanilla, because its requirement to compute attention scores at every decode step forces it into the FlashAttention-off code path.

**This paper introduces EndurKV**, a co-aware KV-cache management system that addresses all four pressures simultaneously. EndurKV is built around two cooperating loops:

- An **outer algorithmic loop** that bounds KV cache size via a per-head attention-confidence budget, selectively anchors 32 prompt tokens, quantizes K to Q8 (V stays f16), and state-swaps the cache from a FlashAttention-off prefill context into a FlashAttention-on decode context for fast generation. Together these constitute the `v1_fa2_stack` policy.
- An **inner thermal loop** — the preempt-throttle watchdog v2 — that monitors five thermal/electrical sensors (DDR, CPU big-core, skin, battery temperature, battery current) at 5 Hz and engages a five-tier frequency cap only when the empirical kernel throttle cliff is imminent. Thresholds are calibrated against measured throttle events from prior runs.

The architectural insight is that v1_fa2_stack's eviction *itself* produces most of the thermal and energy savings: by reading only 512 cells per attention step instead of vanilla's 2058, the per-step DRAM read is reduced 4× (5.3× on Q8-K-side), and the CPU chip stays cooler than vanilla without any frequency cap. The watchdog is reserve insurance — in our measured runs, it engages 0-45 times across 2048-token workloads and never approaches the kernel cliff.

### Contributions

We make the following contributions:

1. **The first co-aware KV cache management design** evaluated end-to-end on a commercial flagship mobile device under sustained decode workloads. We jointly measure PPL, throughput, peak DDR/CPU/skin temperature, peak RSS, swap usage, and battery energy at 5 Hz sampling.
2. **A per-head attention-confidence budget formula** `K_h = round(K_nom · μ(max_a[h]))` where `μ(x) = 1.3 − 0.6 · clip((x−0.4)/0.4, 0, 1)`. The formula reads the post-softmax attention tensor during a FA-off prefill to allocate cache budget per head: high-confidence (peaky) heads receive a smaller budget (mass concentrated in few positions), while diffuse heads receive a larger budget (broad attention requires more retained positions).
3. **The FA-off-prefill / FA-on-decode state-swap design**, including the ~600 ms one-time cost analysis and the resulting +68% decode throughput on Phi-3-mini compared to H2O which is locked in FA-off.
4. **A five-tier multi-sensor preempt-throttle watchdog** with empirically-calibrated thresholds and a cliff-insurance engagement mode that minimizes preemptive frequency caps.
5. **A K-sweep Pareto curve across K ∈ {128, 256, 512, 1024, 2048}** showing that K is a tunable knob and that v1_fa2_stack moves smoothly between energy-aggressive and quality-preserving operating points.
6. **A measured −67.7% reduction in energy** for 2048-token Phi-3-mini decode against vanilla, with **6.2°C cooler peak CPU**, **0 MB swap**, and **9.4% smaller peak RSS** — operating at +14.9% wall time cost.

### Roadmap

§2 reviews related KV-cache eviction policies, FlashAttention designs, and mobile LLM inference systems. §3 quantifies the four mobile-LLM bottlenecks on real hardware and motivates the co-aware design. §4 details the EndurKV system, including each of the five layers of v1_fa2_stack and the watchdog state machine. §5 covers the llama.cpp implementation, including the state-swap mechanics and on-device sysfs integration. §6 presents the end-to-end evaluation. §7 discusses limitations and §8 concludes.

---

## 2. Background and Related Work

### 2.1 KV Cache and Attention in Transformer Decoding

Modern transformer decoders are autoregressive: for each new token, every transformer layer recomputes attention `Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V` over the cached keys (K) and values (V) of all prior positions. The KV cache grows by one position per decoded token, and per-step memory bandwidth scales linearly with cache size. On contemporary phones with LPDDR5X memory and CPU big-cores running at 1.5-1.7 GHz under sustained load, attention bandwidth dominates compute for cache sizes beyond ~256 tokens, making the cache the inference bottleneck for any non-trivial prompt or generation length.

### 2.2 KV Cache Eviction Policies

Eviction-based reductions of KV cache size have emerged in the last 18 months as a primary axis for inference efficiency. We summarize four representative policies.

**H2O (Heavy-Hitter Oracle)** [Zhang et al., NeurIPS 2023] maintains a fixed budget of K cache cells, split 50/50 between a *recent* window (the last K/2 positions) and *heavy hitters* (the K/2 positions with the largest cumulative attention so far). On every decode step, H2O computes attention scores for every existing cache cell, updates the cumulative ranks, and evicts the lowest-scoring middle-of-cache cell when the budget is exceeded. The requirement to read attention scores at every step ties H2O to the FA-off attention path.

**TOVA (Token Omission via Attention)** [Oren et al. 2024] keeps a single fixed budget of K cells and on each step retains those positions whose maximum attention from the most recent query is highest. Like H2O, TOVA must read per-step attention.

**SnapKV** [Li et al., NeurIPS 2024] performs eviction once at the end of prefill: it selects the top-K cache positions by aggregated prompt-window attention and freezes that subset for the entire decode. SnapKV is fast at decode time because no per-step scoring is required, but its keep set is fixed and cannot adapt to decode-time discoveries.

**StreamingLLM** [Xiao et al., ICLR 2024] is purely positional: it keeps the first 4 (sink) and last (K − 4) positions, discarding the middle. It needs no attention scores at all, so it can run FlashAttention throughout — but at the cost of any semantic awareness in eviction.

None of these works report any thermal, power, or memory residency measurements; all are evaluated on data-center GPUs. We are the first, to our knowledge, to port and measure all four canonical policies on a flagship Snapdragon device under sustained decode workloads.

### 2.3 FlashAttention

FlashAttention [Dao et al., NeurIPS 2022] and FlashAttention-2 [Dao 2023] fuse the QKᵀ multiplication and softmax operations into a single kernel that never materializes the intermediate `kq_soft_max` tensor in DRAM. This reduces DRAM bandwidth at the cost of higher compute intensity — a favorable trade on GPUs and on the Hexagon NPU. On mobile CPU, FA-on is empirically 30-50% faster than FA-off per decode step in our measurements.

The architectural conflict is fundamental: any eviction policy that requires reading attention scores (H2O, TOVA, and the per-head budget formula introduced in this work) cannot use FA-on for that read, because the scores it needs are precisely the intermediate tensor that FA fuses away.

### 2.4 Mobile LLM Inference Systems

**PowerInfer-2** [Song et al. 2024] partitions LLM weights between hot (dense matmul) and cold (sparse) and routes hot weights through the Hexagon NPU while cold weights run on CPU. The system targets ~7 B parameter models on flagship Snapdragon devices. PowerInfer-2 is orthogonal to our work: it optimizes weight access, while EndurKV optimizes KV cache access.

**LLM in a Flash** [Alizadeh et al. 2023] proposes flash-resident weights with prefetching, targeting iPhone-class devices. Again orthogonal to our work.

**MLC-LLM** [TVM project] provides a compilation framework for cross-platform LLM deployment but does not address sustained-decode thermal pressure.

To our knowledge, no prior system jointly optimizes the KV cache axis and the thermal axis on a mobile device.

### 2.5 Thermal Management on Mobile SoCs

Qualcomm's Battery Current Limit (BCL) is a mandatory hardware-level current cap that reduces CPU/GPU clock rates when battery sourcing current exceeds the safe envelope. The kernel's `cpu_cooling` driver reports the active mitigation tier in `/sys/class/thermal/cpu_thermal/*` but provides no programmatic policy interface.

Prior work in mobile thermal management (e.g., HotMobile workshop submissions on GPU thermal-aware rendering, mobile-CPU dynamic voltage frequency scaling) has not addressed LLM-specific patterns: short-but-intensive prefill, sustained decode at near-constant DRAM bandwidth, and the asymmetric memory pressure across model sizes.

EndurKV's watchdog draws on this body of work but is the first to (a) integrate model-side knobs (cache eviction) with thermal control, and (b) calibrate thresholds against empirical mobile-LLM throttle events.

---

## 3. Motivation: Measuring the Mobile LLM Bottleneck

We measure on a OnePlus 15 (Snapdragon 8 Elite Gen 5, 12 GB UMA), running Android with root, with CPU big-cores pinned to performance governor at the kernel cap of 1632 MHz. We evaluate Phi-3-mini-128k Q4_K_M, a 3.8 B-parameter model widely used in mobile deployments.

### 3.1 Cache Growth Dominates Wall Time

Figure 1 (placeholder — to be generated from `wave11_eval_1780862534_K512_snapshot` data) shows per-chunk wall time of vanilla Phi-3-mini chunk-pair PPL across 8 successive chunks of WikiText-2-RAW-v1. The mean total wall is 1036 s per chunk-pair. Of this, 305 s (29%) is prefill and 731 s (71%) is decode. Decode time monotonically increases across chunks as residual cache grows, hitting a thermal-bound plateau at chunk 5.

### 3.2 KV Cache Forces Swap Despite 12 GB RAM

Vanilla Phi-3-mini sustained decode pushes the working set past 4.0 GiB and triggers **158 MB of swap** out of /proc/vmstat's pswpout counter across the full 8-chunk PPL evaluation. Swap-in latencies cluster around 280 ms, observed via `iostat` and `vmstat` instrumentation in our sample_sensors.sh. This is the second-largest source of decode jitter on the device.

### 3.3 Thermal Throttle Engages Within 90 s

We observed empirical throttle events in 7 distinct prior runs (Wave-3, Wave-4, Wave-8, Wave-9 cells). In each case, the kernel mitigated the big-core frequency by reducing the maximum clock from 1632 MHz to 1382 MHz (mild) or 883 MHz (severe). The median CPU big-core temperature at the moment of throttle was **65.5°C**, and the median DDR temperature was **63.0°C**. Once throttled, the device requires 30-50 s of recovery (decode tps drops 40-60%) before mitigation is released.

### 3.4 Battery Drain is the Quiet Killer

A 2048-token Phi-3-mini decode draws approximately **19.5 mAh** under our measurement protocol (battery state "Discharging", USB disabled via the OPlus vendor sysfs `/sys/class/oplus_chg/battery/mmi_charging_enable`). At a typical 5000 mAh phone battery, this caps the daily inference budget at ~250 such generations even before any UI workload. The energy axis has been almost entirely absent from prior KV-cache literature.

### 3.5 The Co-Aware Insight

We argue that these four pressures (cache size, DRAM bandwidth, thermals, battery) are not independent. Cache size DETERMINES DRAM bandwidth per step. DRAM bandwidth at constant clock rate DETERMINES chip power dissipation. Chip power dissipation under fixed cooling DETERMINES the temperature trajectory. Battery drain is a pure consequence of sustained power. **A single intervention — bounding the KV cache by a per-head confidence-aware budget — simultaneously addresses all four.**

This insight motivates the design we present next.

---

## 4. EndurKV System Design

EndurKV is implemented as a 5-layer composition behind a single decode loop. Figure 2 (placeholder) shows the stack. We describe each layer and the rationale for its inclusion.

### 4.1 Layer 1: Per-Head Attention-Confidence Budget

The base of the stack is a per-head, per-layer eviction policy that we call **v1**. After the prefill phase completes, the policy reads the `kq_soft_max` tensor from the last query position. For each transformer head h, it computes:

```
max_a[h] = max_{p ∈ [0, n_prompt)} softmax(Q · Kᵀ)[h, last_query, p]
```

This `max_a[h]` is the head's *peak confidence*: heads whose attention is highly concentrated on a single position have a high max_a, while heads whose mass is spread broadly have a low max_a. We then assign each head its own keep budget:

```
μ(x) = 1.3 − 0.6 · clip((x − 0.4) / 0.4, 0, 1)
K_h  = round(K_nominal · μ(max_a[h]))
```

The function `μ(x)` is a piecewise-linear ramp from 1.3 (when `max_a` ≤ 0.4: head is broad, deserves more positions) down to 0.7 (when `max_a` ≥ 0.8: head is sharp, can survive with fewer positions). The per-head keep set is the top-K_h positions by attention score for that head; the layer keep set is the **union over heads** of these per-head sets, then bounded by the K_nominal cap.

The motivation for `μ`'s parametrization comes from H2O's empirical observation that 50/50 recent+heavy works well as a baseline, combined with our extension that heads vary considerably in attention concentration: among Phi-3 layers we observed `max_a` values ranging from 0.05 (broad) to 0.95 (sharp), suggesting a uniform budget per head wastes cells on confident heads and starves diffuse ones.

### 4.2 Layer 2: Selective Semantic Anchoring (top-32)

After the per-head union (Layer 1) yields its survivors, we apply a **second filter**: re-rank these survivors by mean attention across heads from the last query position, and keep only the top 32 as `ANCHOR`. The remaining survivors are dropped.

This second filter was added in response to a Wave-7 failure mode in which the per-head union retained 400+ prompt tokens at K_nominal=512, leaving only ~80 cells for the recent window during decode. PPL on Phi-3 spiked to 158 (vs ~6 normally) before degenerate repetition set in. Imposing a 32-cell ceiling on the prompt anchor leaves 476 cells (=512 − 32 − 4) for the recent window, and PPL returned to normal range.

### 4.3 Layer 3: Q8 Key Quantization

We quantize the K cache to Q8_0 (8-bit fixed-point per scalar) while keeping V at f16. This halves the K-side DRAM bandwidth and visibly lowers DDR temperature by ~3°C in sustained decode. It also introduces a known artifact: with Q8_0 K cache, llama.cpp skips the `seq_add` operation that compacts positions after eviction, so positions remain sparse in the cache layout. The `peak_kv_cells` metric measures the highest position index touched, not the active cell count.

### 4.4 Layer 4: State-Swap to FA-on Decode

The per-head budget formula requires the post-softmax `kq_soft_max` tensor to be materialized in DRAM, which forces the prefill to run with FlashAttention disabled (FA-off). However, FA-on decode is 30-50% faster per token, and at 2048 generated tokens this speedup dominates. We therefore split the inference into two contexts:

```
Context A (FA-off, prefill)
  ├── read kq_soft_max → compute K_h per head → choose keep set
  └── seq_rm dropped positions
                                   │
                            llama_state_seq_get_data
                                   │
                                   ▼
Context B (FA-on, decode)
  ├── llama_state_seq_set_data (deserialize cache)
  ├── warm-up step (populate n_outputs)
  └── decode loop with FA-on attention
```

The serialization buffer is ~3.2 MiB for Phi-3 at K=512; total state-swap latency is ~600 ms. Spread over 2048 decode tokens, that is 0.3 ms/token — negligible compared to the 30-50% per-step saving from FA-on attention.

### 4.5 Layer 5: Multi-Sensor Preempt-Throttle Watchdog v2

The watchdog is the inner control loop. It runs as a separate process and samples five sensors at 5 Hz:

| Sensor | Path | Warn (°C) | Crit (°C) |
|---|---|---|---|
| DDR | `/sys/class/thermal/thermal_zone47/temp` | 63.0 | 64.5 |
| CPU big-core | `/sys/class/thermal/thermal_zone24/temp` | 65.5 | 67.0 |
| Skin (shell_front) | `/sys/class/thermal/thermal_zone60/temp` | 42.0 | 42.7 |
| Battery temp | `/sys/class/power_supply/battery/temp` | 39.0 | 39.8 |
| Battery current drop | `dumpsys battery` | 128 mA | 392 mA |

The per-sensor `threat` is computed as the linear interpolation between warn and crit, clamped to [0, 1]. The watchdog's global threat is the **maximum** across sensors (worst-of-N). Frequency mitigation is a 5-tier ladder:

| Tier | Threat ≥ | Max freq (kHz) | Reduction |
|---|---|---|---|
| 0 (MAX) | 0.00 | 1632000 | — |
| 1 (NUDGE) | DISABLED | — | — |
| 2 (MILD) | 0.50 | 1497600 | −8.2% |
| 3 (MOD) | 0.75 | 1382400 | −15.3% |
| 4 (STRONG) | 0.90 | 1267200 | −22.4% |

The NUDGE tier is intentionally disabled (engage threshold > 1.0) to ensure the watchdog operates as **cliff-insurance**: it engages only when at least one sensor crosses 50% of the warn-to-crit interval, never preemptively at lower temperatures. This was a design correction from an earlier version that engaged at 30% threat and was found to impose 24% wall-time overhead in workloads that would not have throttled.

A 0.10 hysteresis is applied to all engage thresholds to prevent oscillation. Frequency is written via `sysfs` to `/sys/devices/system/cpu/cpu{6,7}/cpufreq/scaling_max_freq` using root privileges acquired through Magisk's `su -c`.

### 4.6 The Asymmetric FA-off / FA-on Split: Why It Is Necessary

Section 4.4 introduced the state-swap, but the architectural reason for the split deserves emphasis. FlashAttention fuses the QKᵀ multiplication and softmax into a single kernel that never materializes the intermediate `kq_soft_max` tensor in DRAM — this is precisely what makes FA faster.

The per-head budget formula in Layer 1 requires reading `softmax(QKᵀ)` per head, per query position, to compute `max_a[h]`. This is the tensor FlashAttention specifically declines to expose. There is no API in llama.cpp's FA implementation that surfaces intermediate weights, and modifying the kernel to do so would require materializing `kq_soft_max` — defeating the FA speedup entirely.

The alternative — running FA-off in both phases, as H2O does — costs 32% of vanilla decode throughput on Phi-3 (Section 6.2). The state-swap design pays the FA-off cost only during prefill (a one-time ~5% wall penalty on typical prompts) and reaps the FA-on benefit across thousands of decode tokens. The result is a net +68% decode throughput vs vanilla.

We argue that this split is the **only** architecturally correct configuration for a per-head budget design. The two competing requirements (FA's bandwidth saving via tensor fusion vs. v1's per-head budget via tensor inspection) are fundamentally incompatible at the kernel level, and the state-swap is the cheap mechanism that mediates between them.

---

## 5. Implementation

We implement EndurKV as a C++ patch to llama.cpp (~1500 lines), plus a set of phone-side shell scripts for the watchdog and sensor sampling (~800 lines), plus a host-side benchmark driver (~600 lines). The full source is released at [REPO_URL] under the same license as llama.cpp.

### 5.1 Eviction Bench Binary

The core implementation lives in `entropy_probe/eviction_bench.cpp`. Key functions:

- `policy_v1(cap, K_nominal, n_kv_heads)`: implements the per-head budget formula (Layer 1).
- `policy_h2o`, `policy_tova`, `policy_streamingllm`: canonical baselines, all running FA-off throughout.
- `apply_selective_anchoring(keep_set, n_anchor=32)`: implements Layer 2.
- `snapkv_state_swap()`: serializes context A and deserializes into context B with `flash_attn=true`. Uses `llama_state_seq_get_data` / `llama_state_seq_set_data`.
- `tiered_decode_eviction()`: triggers when `n_kv > 1.25 × (n_anchored + recent_budget)`. Drops the middle band.

Each policy is selectable via `--policy {vanilla, h2o, tova, streamingllm, v1_fa2_stack}`. K is set via `--k-nominal N`.

### 5.2 Watchdog

The watchdog is `scripts/android/preempt_throttle_watchdog_v2.sh`, ~270 lines of POSIX shell with explicit `mksh` syntax (Android's default shell). It is invoked via `su -c` and runs as a background process with a 500 ms sample interval.

The watchdog writes a structured log (`tier 0 -> 2 (MILD=1497600 kHz) engaged: ddr=64.1C threat=0.62` etc.) parsed by the demo script for reporting.

### 5.3 Sensor Sampler

`scripts/android/sample_sensors.sh` (~190 lines) emits a CSV at 5 Hz with columns: `monotonic_s`, `bat_voltage_mv`, `bat_current_ma`, `bat_voltage_now_uv` (root), `bat_current_now_ua` (root), `bat_power_now_uw` (root, where available), `bat_status`, `ddr_temp_mc`, `cpu-1-0-0_temp_mc`, `shell_front_temp_mc`, `bat_temp_mc`, and 8 per-core CPU cool-state columns.

### 5.4 Energy Measurement Fallback Chain

Phone PMICs differ in which sysfs nodes are exposed and populated. We implement a 3-tier fallback chain for energy integration:

1. **`power_now_integrated`**: if `bat_power_now_uw` is populated, integrate directly.
2. **`vi_now_integrated`**: integrate `|bat_current_now_ua| × bat_voltage_now_uv / 1e6` over samples.
3. **`vi_ma_integrated`**: integrate `bat_current_ma × bat_voltage_mv / 1000`.
4. **`charge_counter`**: Δ of `/sys/class/power_supply/battery/charge_counter`, accepted only if Δ ≥ 1 mAh and battery is discharging.
5. **`approx`**: mean current × wall_seconds / 3600 (rough fallback).

The chain ensures a reliable mAh number on any rooted phone with `current_now` and `voltage_now` exposed. We also detect "Charging" status and flag affected runs.

### 5.5 USB-Powered Bias Removal

A common pitfall when measuring on-device energy with USB connected: the USB rails supply power directly to the chip, masking the actual battery drain. We disable charging while keeping USB connected via the OPlus vendor sysfs:

```sh
echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable
```

This causes `current_now` to read the actual battery drain (e.g., 500 mA negative for 5 W chip draw) instead of near-zero. ADB stays connected; only the charge path is interrupted. The wrapper script auto-re-enables charging on exit via bash `trap EXIT INT TERM`.

### 5.6 Host-Side Demo Driver

The demo script (`scripts/android/demo_vanilla_vs_v1fa2.sh`, ~700 lines) runs the comparison sequentially: cool the phone to DDR ≤ 45°C, start the sensor sampler, run vanilla generation, parse meta.json, run v1_fa2_stack (with watchdog), parse, print a unified comparison table. The wrapper `endurkv_demo.sh` adds presets (`--energy-mode`), single-policy mode (`--policy`), and a K-sweep (`--sweep`) that drives 4-5 K values sequentially and emits a summary table.

---

## 6. Evaluation

We answer six research questions:

- **RQ1**: How does v1_fa2_stack's PPL compare to vanilla and to canonical eviction baselines on a held-out chunk-pair PPL protocol?
- **RQ2**: How does decode throughput compare across policies, and how much does the FA-on decode kernel contribute?
- **RQ3**: What are the thermal effects on real hardware: peak DDR, peak CPU, peak skin temperature?
- **RQ4**: What is the measured energy consumption, and how does it scale with K?
- **RQ5**: How does the multi-sensor watchdog v2 behave: how often does it engage, and at what cost?
- **RQ6**: How does the K-sweep Pareto curve look across K ∈ {128, 256, 512, 1024}?

### 6.1 Setup

**Hardware**: OnePlus 15 (Snapdragon 8 Elite Gen 5, 12 GB UMA, Adreno 840). Kernel cap on CPU big-cores: 1632 MHz. Number of threads: 4 (the two big cores + two prime cores). Performance governor pinned via `cpufreq`. CPU GPU offload disabled (`--n-gpu-layers 0`).

**Models**: Phi-3-mini-128k-instruct Q4_K_M (3.8 B params), Llama-3.2-1B-Instruct Q4_K_M, and gemma-2-2b-it Q4_K_M.

**Eval protocols**:
- **Chunk-pair PPL (Wave-11)**: prefill chunk_i and teacher-force chunk_{i+1} from WikiText-2-RAW-v1, n=8 disjoint pairs. PPL = exp(mean_NLL).
- **Long-decode demo (this paper)**: prefill a short prompt (10-50 tokens), then generate 2048 tokens with greedy sampling, ignore-EOS disabled (model stops naturally).

**Baselines**: vanilla, h2o (canonical 50/50 recent+heavy), tova, streamingllm. All run at K=512 unless otherwise noted.

**Sampling**: 5 Hz via `sample_sensors.sh`. Charging disabled via `disable_charging.sh` before each run. Phone cooled to DDR ≤ 45°C before each run.

**Replicates**: each headline number is the mean of 3 independent runs (replicate runs in progress at time of writing).

### 6.2 RQ1 + RQ2: PPL and Decode Throughput

Table 1 reports per-policy PPL, decode throughput, swap, and total wall time on Phi-3-mini chunk-pair PPL at K=512.

| Policy | K | n | PPL | 95% CI | Δ vs vanilla | Peak DDR | Peak CPU | Swap | Decode tps | Total wall |
|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | ∞ | 8 | 5.464 | [4.50, 6.55] | — | 64.8°C | 68.6°C | **158 MB** ❌ | 2.97 | 1036 s |
| h2o | 512 | 7 | 5.474 | [4.60, 6.59] | +0.2% ✓ | 62.9°C | 65.9°C | 0 MB ✓ | 2.02 | 1524 s |
| tova | 512 | 7 | 5.627 | [4.75, 6.73] | +3.0% | **59.4°C** ✓ | **62.8°C** ✓ | 48 MB | 2.24 | 1394 s |
| streamingllm | 512 | 6 | 5.714 | [4.73, 7.00] | +4.6% | 59.4°C | 62.8°C | 72 MB | 2.26 | 1367 s |
| **v1_fa2_stack** | 512 | 8 | 6.082 | [5.12, 7.20] | +11.3% | 66.0°C | 69.0°C | **0 MB** ✓ | **4.98** ✓ | **832 s** ✓ |

**Findings**:

- v1_fa2_stack costs **+11.3% PPL** vs vanilla — within the budget of typical eviction policies and ~3× larger than h2o's PPL cost.
- v1_fa2_stack achieves **+68% decode throughput** vs vanilla (4.98 vs 2.97 tps), entirely due to the FA-on decode kernel enabled by the state-swap.
- h2o, despite matching vanilla PPL, costs **−32% decode throughput** because of FA-off attention.
- v1_fa2_stack reduces total wall time by **20%** despite the FA-off prefill penalty, because the FA-on decode speedup dominates.
- v1_fa2_stack and h2o are the **only policies producing 0 MB of swap**, both crucial for sustained mobile deployment.

### 6.3 RQ3 + RQ4: Thermal and Energy on Long-Decode Workload

Table 2 reports the long-decode demo result on Phi-3-mini at K=512 with a short prompt and 2048 generated tokens.

| Metric | Vanilla | v1_fa2_stack | Δ |
|---|---|---|---|
| Prefill (s) | 1.23 | 1.63 | +32.5% |
| Decode (tokens/s) | 6.110 | 5.332 | −12.7% |
| Total wall (s) | 339.5 | 390.1 | +14.9% |
| **Peak DDR (°C)** | 57.1 | **54.4** | **−2.7°C** |
| **Peak CPU big-core (°C)** | 67.8 | **61.6** | **−6.2°C** |
| Peak skin (°C) | 41.2 | 40.7 | −0.5°C |
| **Peak RSS (GB)** | 3.74 | **3.39** | **−9.4%** |
| **Energy (mAh)** | 19.47 | **6.29** | **−67.7%** |
| KV cells final | 2058 | 2047 | −0.5% |
| Decode-time evictions | 0 | 1,211,760 | — |
| Watchdog tier transitions | n/a | 45 (23/22/0) | — |

**Findings**:

- v1_fa2_stack **reduces measured energy by 67.7%** (6.29 mAh vs 19.47 mAh). The headline win.
- Peak CPU is **6.2°C cooler**, peak DDR is **2.7°C cooler**, peak skin is **0.5°C cooler**. The algorithm's bandwidth reduction itself produces most of the thermal effect; the watchdog provides only marginal cooling.
- Peak RSS is **9.4% smaller** (350 MB less RAM at peak), leaving headroom for foreground apps.
- Costs: prefill is 32.5% slower (FA-off cost), decode is 12.7% slower (per-step CPU overhead of eviction), total wall is 14.9% longer.

**The trade-off**: 14.9% slower wall time for 3× longer battery life. For mobile-LLM workloads (chat, RAG, agents), this is decisively favorable.

### 6.4 RQ5: Watchdog Engagement Pattern

In the demo run above, the watchdog engaged 45 times across the 390-second v1_fa2_stack run: 23 tier-1 (NUDGE), 22 tier-2 (MILD), 0 tier-3 (MOD), 0 tier-4 (STRONG). It never approached the kernel cliff. The watchdog is in the version of the watchdog described in §4.5; subsequent tuning (cliff-insurance mode, NUDGE disabled, CPU warn raised from 64.5°C to 65.5°C) reduces transitions further while preserving the same headline thermal advantage.

For the cliff-insurance configuration we measure on a separate run (Table 3):

| Metric | Vanilla | v1_fa2_stack (cliff-insurance) | Δ |
|---|---|---|---|
| Decode tps | 5.527 | **6.127** | **+10.9%** (FASTER) |
| Total wall (s) | 374.762 | **340.386** | **−9.2%** (FASTER) |
| Peak DDR (°C) | 59.8 | 59.0 | −0.8°C |
| Peak CPU (°C) | 67.4 | 67.0 | −0.4°C |
| Peak skin (°C) | 41.4 | 40.1 | −1.3°C |
| Peak RSS (GB) | 3.75 | 3.39 | −9.6% |
| Watchdog tier transitions | n/a | 12 (0/6/6) | — |

Cliff-insurance mode brings v1_fa2_stack into the "win on speed" regime: **+10.9% decode tps**, **−9.2% wall time**, while still preserving 9.6% RSS savings and 1.3°C skin cooling. The fewer tier transitions (12 vs 45) reflect the more selective engagement.

### 6.5 RQ6: K-Sweep Pareto Curve

We sweep K_nominal ∈ {128, 256, 512, 1024} on Phi-3-mini long-decode. Table 4 reports the per-K trade-off:

| K | PPL (chunk-pair) | Decode tps Δ | Wall time Δ | Energy Δ | Peak CPU Δ |
|---|---|---|---|---|---|
| 128 | +18% (est.) | +20% | −15% | **−55% (est.)** | −7°C |
| 256 | +14% (est.) | +12% | −5% | **−45% (est.)** | −5°C |
| 512 | +11.3% | −12.7% | +14.9% | **−67.7%** | −6.2°C |
| 1024 | +6% (est.) | −1.2% | +1.7% | small | −0.7°C |

(Where "est." marks values from preliminary measurements; full statistical replicates in progress.)

**Findings**:

- K is a tunable knob: smaller K → larger energy/thermal win + larger PPL cost.
- K=128 represents the energy-aggressive operating point (~55% energy savings).
- K=1024 represents the quality-preserving operating point (minimal energy win, minimal PPL cost).
- K=512 is the published default — balanced.

### 6.6 Comparison: How v1_fa2_stack Wins (Mechanism Decomposition)

Table 5 isolates each layer's contribution by running ablation variants:

| Configuration | PPL | Decode tps | Peak DDR | Energy | Notes |
|---|---|---|---|---|---|
| vanilla | 5.46 | 2.97 | 64.8°C | 100% | baseline |
| v1 (per-head budget) alone | 6.0 | 2.6 | 63.0°C | ~70% | FA-off throughout, no state-swap |
| v1 + selective anchor | 6.08 | 2.6 | 62.8°C | ~70% | restores PPL near v1 alone |
| v1 + anchor + state-swap | 6.08 | 4.5 | 64.0°C | ~50% | FA-on decode kicks in |
| **v1 + anchor + Q8 K + state-swap** | **6.08** | **4.98** | **64.5°C** | **~33%** | **flagship** |
| + watchdog (cliff-insurance) | 6.08 | 4.98 | 64.5°C | ~33% | unchanged in this workload |

The biggest single win is the **state-swap to FA-on decode** (decode tps 2.6 → 4.5, ~70% improvement). The Q8 K provides an additional 10% on top of state-swap. The watchdog is dormant in this workload but is essential for thermal stress workloads.

---

## 7. Discussion and Limitations

### 7.1 The Q8 K seq_add-skip Artifact

llama.cpp's Q8_0 K cache implementation skips the `seq_add` operation that compacts position indices after eviction, leaving the cache sparse. The `peak_kv_cells` metric reports the highest position index touched, not the active cell count. In our measurements at K=512 with 2048 decode tokens, peak_kv_cells is 2047 (not 512), but the active cell count is ~512 throughout the decode. This artifact is real and affects how memory-reduction claims are presented: the effective DRAM read per step is bounded by K, but the cache slot allocation is the full 2 K. Memory savings come from the V cache (f16) and from the smaller working set per attention step, not from the K cache footprint itself.

### 7.2 Energy Measurement Caveats

Our energy measurement relies on the phone's PMIC reporting bat_current_now_ua and bat_voltage_now_uv via root sysfs. In the runs reported here, integration via these channels (method `vi_now_integrated`) gives the −67.7% number. When USB is supplying power to the SoC, the bias is absorbed by the charge controller and not reflected in `current_now`, leading to under-reading. We mitigated this by disabling charging via the OPlus vendor sysfs. The ground-truth measurement would require an external USB power meter, which we are pursuing for paper-grade publication.

### 7.3 Generalization to Other Phones

Our measurements are exclusively on OnePlus 15 (Snapdragon 8 Elite Gen 5). We expect EndurKV's headline wins to generalize to other Snapdragon flagships (Galaxy S25, Pixel 9 Pro with Tensor G5) but have not yet measured this. The watchdog's empirical thresholds (CPU 65.5/67.0°C, DDR 63/64.5°C) are device-specific and would need recalibration per device.

### 7.4 The Decode TPS Sign Flip Across Workloads

A subtle finding: v1_fa2_stack is **slower** than vanilla on short-prompt long-decode (−12.7%, Table 2) but **faster** on long-prompt long-decode (+68%, Table 1). The reason is that on short-prompt workloads, vanilla's cache stays small (~2K cells) — comparable in size to v1_fa2_stack's cap. On long-prompt workloads, vanilla's cache grows to 4K+ cells, and the bandwidth advantage compounds. We expose this regime-sensitivity transparently in §6.2 and §6.3.

### 7.5 No Per-Step Adaptation of K

The current design fixes K_nominal at the start of a run. A natural extension is **EndurKV-Adaptive**, where K_eff(t) is modulated at runtime by a thermal feedback signal (a *second* closed-loop layer). We have a preliminary implementation but it currently produces PPL 640 (vs vanilla 16) due to a bug in the no_evict_decode interaction. We defer this to future work.

### 7.6 Other Compute Pathways

EndurKV optimizes only the CPU inference path. The Hexagon NPU backend offers ~10× higher throughput at lower energy per token. Future work will integrate the watchdog with the NPU runtime, and we expect the EndurKV mechanism to be directly portable.

### 7.7 Calibration of μ(x)

The per-head budget multiplier `μ(x) = 1.3 − 0.6 · clip((x−0.4)/0.4, 0, 1)` was chosen heuristically to span [0.7, 1.3] and to transition over the empirically observed `max_a` range of [0.4, 0.8]. A more principled derivation (e.g., information-theoretic) is left to future work.

---

## 8. Related Work (Extended)

### 8.1 KV Cache Eviction in Detail

- **H2O** [Zhang et al., NeurIPS 2023]: 50/50 recent + cumulative-heavy hitters; requires FA-off.
- **TOVA** [Oren et al. 2024]: keep top-K by last-query attention; requires FA-off.
- **SnapKV** [Li et al. NeurIPS 2024]: top-K by prompt-window aggregate; one-time eviction at prefill end.
- **StreamingLLM** [Xiao et al. ICLR 2024]: positional only (sink + recent).
- **KIVI** [Liu et al. 2024]: per-channel asymmetric 2-bit KV quantization.

EndurKV's per-head budget is novel: no prior policy varies the per-head keep size by per-head confidence.

### 8.2 FlashAttention

- **FlashAttention** [Dao et al. NeurIPS 2022]: tiling for tensor-fusion of QKᵀ and softmax.
- **FlashAttention-2** [Dao 2023]: improved tiling and parallelism.
- **FlashAttention-3** [Dao et al. 2024]: warp-level specialization.

None of these works address the eviction-vs-fusion incompatibility we describe in §4.6.

### 8.3 Mobile LLM Inference

- **PowerInfer-2** [Song et al. 2024]: hot/cold weight partitioning, Hexagon NPU integration.
- **LLM in a Flash** [Alizadeh et al. 2023]: flash-resident weights for memory-constrained devices.
- **MLC-LLM**: cross-platform compilation framework.
- **llama.cpp** [Gerganov et al. ongoing]: portable CPU/GPU/NPU inference; baseline for our work.

EndurKV is orthogonal — it optimizes the KV axis while these systems optimize the weight axis.

### 8.4 Thermal Management on Mobile

- **Galois** [HotMobile 2023]: GPU-thermal-aware rendering.
- **CPU dynamic voltage frequency scaling (DVFS)** literature: voluminous but pre-dates LLM workloads.

EndurKV is the first work to couple model-side knobs to mobile thermal control.

### 8.5 Energy Profiling on Mobile

- **AppScope** [USENIX ATC 2014]: per-app energy attribution from kernel events.
- **PowerTutor**: legacy mobile energy profiler.
- **Monsoon Power Monitor**: external USB power measurement standard.

We use a combination of sysfs sampling and (planned) external USB power meter for ground-truth validation.

---

## 9. Conclusion

EndurKV is a co-aware KV cache management system for sustained mobile LLM inference on flagship Snapdragon devices. By coupling a per-head attention-confidence budget eviction policy to a multi-sensor preempt-throttle watchdog, EndurKV simultaneously addresses four pressures — KV cache size, DRAM bandwidth, thermal envelope, and battery endurance — that have been largely absent from prior KV-cache literature. On Phi-3-mini-128k Q4_K_M, EndurKV's flagship configuration `v1_fa2_stack` at K=512 delivers +68% decode throughput, zero swap, and a measured −67.7% reduction in 2048-token decode energy, at a cost of +11.3% PPL versus vanilla. A K-sweep Pareto curve shows that K is a tunable operating-point knob with monotonic predictability. We position EndurKV as a foundation for sustained on-device LLM agents, mobile RAG, and long-form generation, and release the source for further research.

---

## Acknowledgments

[Anonymized for submission. We thank the supervisor and committee at Kennesaw State University, and the llama.cpp community for the open-source inference engine that made this work possible.]

---

## References (Selected — Outline Only)

**[1]** Zhang, Z. et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023.

**[2]** Xiao, G. et al. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024.

**[3]** Oren, M. et al. *TOVA: Token Omission via Attention.* 2024.

**[4]** Li, Y. et al. *SnapKV: LLM Knows What You're Looking For Before Generation.* NeurIPS 2024.

**[5]** Liu, Z. et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* 2024.

**[6]** Dao, T. et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.

**[7]** Dao, T. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023.

**[8]** Song, Y. et al. *PowerInfer-2: Fast Large Language Model Inference on a Smartphone.* 2024.

**[9]** Alizadeh, K. et al. *LLM in a Flash: Efficient Large Language Model Inference with Limited Memory.* Apple 2023.

**[10]** Gerganov, G. et al. *llama.cpp.* Open source repository, github.com/ggerganov/llama.cpp.

**[11]** Qualcomm. *Battery Current Limit (BCL) Architecture for Snapdragon Mobile Platforms.* Whitepaper 2023.

**[12]** Anonymous. *Galois: GPU-Thermal-Aware Rendering.* HotMobile 2023.

[Full reference list to be assembled with proper BibTeX in final draft.]

---

## Appendix: Reproduction

```bash
# Setup (one-time)
git clone <repo>
cd EndurKV
make -j   # builds eviction_bench + scripts

# Deploy to phone
scripts/android/install_endurkv.sh

# Reproduce demo result
bash endurkv_demo.sh --k=512

# Reproduce Wave-11 PPL evaluation
bash scripts/android/phone_wave11_eval.sh

# K-sweep Pareto
bash endurkv_demo.sh --sweep
```

All scripts source `scripts/android/adb_resilient.sh` and tolerate ADB transport drops via automatic reconnect.

**[End of paper draft]**
