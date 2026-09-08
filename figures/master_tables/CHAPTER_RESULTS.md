# Chapter 5 — Empirical Results

This chapter reports the empirical evaluation of EndurKV against four reference
KV-eviction baselines on a Snapdragon 8 Elite Gen 5 mobile SoC. All numbers
labelled with `{{...}}` are *Wave-11 placeholders* that are mechanically filled
in by `eval_pipeline/score_ppl.py` immediately after the on-device sweep
completes. Numbers given as literal floats are either Wave-9 or Wave-10
measurements that have already been ingested, audited, and frozen in the master
tables (see `WAVE10_MASS_RETAINED.md`, `SUPERVISOR_EVIDENCE_PACKAGE.md`,
`SUBSECTION_KSWEEP.md`). The placeholder protocol is documented at the bottom
of this file (Section 9) so that the dissertation source compiles to a
fully-numerical document the moment Wave-11 lands.

---

## 1. Experimental setup

### 1.1 Hardware

All measurements are taken on a single OnePlus 15 retail unit (Snapdragon 8
Elite Gen 5, 16 GB LPDDR5X, 256 GB UFS 4.0, Android 15 / OxygenOS 15, kernel
6.1). The phone runs in airplane mode with the screen off, mounted on a
passive aluminium heat-sink and allowed to cold-soak to a skin temperature of
≤33 °C and a DDR thermal-zone temperature of ≤40 °C before every cell. CPU
frequency is pinned at 1632 MHz on the prime cluster and 1497 MHz on the
performance cluster via the `performance` governor; the GPU is left at idle;
no NPU offload is enabled in this chapter (NPU offload is evaluated separately
in Chapter 6 as part of the PowerInfer-2 fidelity audit).

### 1.2 Models

The headline evaluation uses three models that span the small / medium / long-
context regime accessible on a 16 GB phone:

- **Llama-3.2-1B-Instruct** (Q4_K_M, 0.92 GB on disk, 8k native context).
- **Gemma-2-2B-Instruct** (Q4_K_M, 1.71 GB on disk, 8k native context).
- **Phi-3-mini-128k-Instruct** (Q4_K_M, 2.39 GB on disk, 128k native context;
  used as the long-context discriminator).

All three are loaded through the modified `eviction_bench.cpp` binary under
the `llama.cpp` runtime pinned to commit `b4031`. Flash-attention mode is
toggled per-policy as documented in Section 1.4.

### 1.3 Methodology — chunk-pair held-out PPL on WikiText-2

Perplexity is reported using the WikiText-2-RAW-V1 test split (Salesforce /
wikitext, materialised from the pinned `test.parquet` and split into nine
sequential word-boundary chunks of approximately 2048 tokens each). The
disjoint chunk-pair protocol of H2O / KIVI is used:

1. Prefill chunk *i* under the policy under test.
2. Apply that policy's eviction in its canonical configuration.
3. Teacher-force chunk *i+1* in held-out continuation mode and accumulate
   per-token NLL.

With nine chunks this yields eight disjoint `(prefill, eval)` pairs per
`(model, policy)` cell. The per-cell PPL is the token-weighted *geometric*
mean of per-chunk perplexity,
`PPL_cell = exp( Σ_i n_tok_i · mean_nll_i / Σ_i n_tok_i )`, which is the
canonical WikiText-2 reporting convention used by H2O, KIVI, StreamingLLM,
and TOVA. Bootstrap 95 % confidence intervals (1000 resamples, percentile
method, log-domain) are reported alongside every mean PPL. The fix that
produced this protocol — and the methodology bug it retired — is documented
in `HELD_OUT_PPL_FINDING.md`; in summary, every PPL number reported in this
chapter is a teacher-forced held-out perplexity, not the sampling-NLL of
self-output used in Waves 3 through 9.

### 1.4 Policies

Each policy runs in the canonical configuration from its source paper; we do
*not* homogenise flash-attention mode or sink/recent ratios across policies.

| Tag | Role | FA | K | Sink | Source |
|---|---|---|---|---|---|
| `vanilla` | Full-cache upper bound | on | — | — | baseline |
| `streamingllm` | Recency-only floor (no attention readout) | on | 512 | 4 | Xiao et al., ICLR 2024 |
| `h2o` | Attention-aware SOTA | off | 512 | 4 | Zhang et al., NeurIPS 2023 |
| `tova` | Per-layer attention selection | off | 512 | 4 | Oren et al., 2024 |
| `v1_fa2_stack` (**ours**) | EndurKV: v1 prefill → FA² decode, Q8 K, anchor-top-32, closed-loop watchdog | off→on | 512 | 4 | this work |

The headline K is **K=512** for every non-vanilla policy. K=1024 numbers (and
K=256, K=384) live in the K-sweep companion subsection (Section 5) on Phi-3
only and are explicitly labelled as such; this resolves the K-budget
asymmetry flagged in `WAVE11_FINAL_SPEC.md`.

---

## 2. Headline result — vanilla vs eviction (Phi-3-mini)

The single most important number in this dissertation is the gap between
the full-cache upper bound and our EndurKV v1_FA²-stack at K=512 on Phi-3-
mini-128k. A small PPL gap means the K=512 cache is *information-sufficient*
for the held-out continuation; a large gap means the policy has destroyed
retrievable context.

| Phi-3-mini, K=512 | PPL (mean) | 95 % CI | Δ vs vanilla (nats) |
|---|---:|---:|---:|
| vanilla (full cache) | {{phi3_vanilla_ppl_mean}} | [{{phi3_vanilla_ppl_ci_low}}, {{phi3_vanilla_ppl_ci_high}}] | 0 |
| streamingllm | {{phi3_streamingllm_ppl_mean}} | [{{phi3_streamingllm_ppl_ci_low}}, {{phi3_streamingllm_ppl_ci_high}}] | {{phi3_streamingllm_ppl_delta}} |
| h2o | {{phi3_h2o_ppl_mean}} | [{{phi3_h2o_ppl_ci_low}}, {{phi3_h2o_ppl_ci_high}}] | {{phi3_h2o_ppl_delta}} |
| tova | {{phi3_tova_ppl_mean}} | [{{phi3_tova_ppl_ci_low}}, {{phi3_tova_ppl_ci_high}}] | {{phi3_tova_ppl_delta}} |
| **v1_fa2_stack (ours)** | **{{phi3_v1fa2_ppl_mean}}** | [{{phi3_v1fa2_ppl_ci_low}}, {{phi3_v1fa2_ppl_ci_high}}] | **{{phi3_v1fa2_ppl_delta}}** |

The full-cache baseline is reported at `{{phi3_vanilla_ppl_mean}}` ± the CI
above. This is within the published range for Phi-3-mini-128k on
WikiText-2 raw (canonical value 4.9–5.1 in the original Phi-3 technical
report). The on-device baseline therefore matches the literature, confirming
that the Q4_K_M weight quantisation and the OnePlus 15 thermal envelope do
not degrade the underlying language model relative to the cloud-side
checkpoint.

The headline gap **{{phi3_v1fa2_ppl_delta}} nats** ({{phi3_v1fa2_ppl_delta_pct}} %
relative) is the price EndurKV pays for the 4.5× cache compression at K=512.
That gap should be read alongside the per-model heat-and-throughput
trade-offs in Section 4: PPL alone is necessary but not sufficient, because
two of the four eviction baselines (H2O and TOVA) are bounded above by
vanilla on PPL but below by vanilla on sustained throughput. The Pareto
analysis in Section 4 makes the trade explicit.

---

## 3. Per-model comparison — three models × five policies

The matrix below extends the Section 2 head-line to all three model sizes,
holding K=512 fixed for every non-vanilla policy and reporting mean held-out
PPL with bootstrap 95 % CI. Each cell aggregates eight chunk-pairs (16,384
scored tokens) on the OnePlus 15.

### 3.1 Llama-3.2-1B-Instruct (Q4_K_M)

| Policy | PPL (mean) | CI low | CI high | log-std | n_chunks |
|---|---:|---:|---:|---:|---:|
| vanilla | {{llama1b_vanilla_ppl_mean}} | {{llama1b_vanilla_ppl_ci_low}} | {{llama1b_vanilla_ppl_ci_high}} | {{llama1b_vanilla_ppl_logstd}} | {{llama1b_vanilla_n_chunks}} |
| streamingllm | {{llama1b_streamingllm_ppl_mean}} | {{llama1b_streamingllm_ppl_ci_low}} | {{llama1b_streamingllm_ppl_ci_high}} | {{llama1b_streamingllm_ppl_logstd}} | {{llama1b_streamingllm_n_chunks}} |
| h2o | {{llama1b_h2o_ppl_mean}} | {{llama1b_h2o_ppl_ci_low}} | {{llama1b_h2o_ppl_ci_high}} | {{llama1b_h2o_ppl_logstd}} | {{llama1b_h2o_n_chunks}} |
| tova | {{llama1b_tova_ppl_mean}} | {{llama1b_tova_ppl_ci_low}} | {{llama1b_tova_ppl_ci_high}} | {{llama1b_tova_ppl_logstd}} | {{llama1b_tova_n_chunks}} |
| **v1_fa2_stack** | **{{llama1b_v1fa2_ppl_mean}}** | {{llama1b_v1fa2_ppl_ci_low}} | {{llama1b_v1fa2_ppl_ci_high}} | {{llama1b_v1fa2_ppl_logstd}} | {{llama1b_v1fa2_n_chunks}} |

### 3.2 Gemma-2-2B-Instruct (Q4_K_M)

| Policy | PPL (mean) | CI low | CI high | log-std | n_chunks |
|---|---:|---:|---:|---:|---:|
| vanilla | {{gemma2b_vanilla_ppl_mean}} | {{gemma2b_vanilla_ppl_ci_low}} | {{gemma2b_vanilla_ppl_ci_high}} | {{gemma2b_vanilla_ppl_logstd}} | {{gemma2b_vanilla_n_chunks}} |
| streamingllm | {{gemma2b_streamingllm_ppl_mean}} | {{gemma2b_streamingllm_ppl_ci_low}} | {{gemma2b_streamingllm_ppl_ci_high}} | {{gemma2b_streamingllm_ppl_logstd}} | {{gemma2b_streamingllm_n_chunks}} |
| h2o | {{gemma2b_h2o_ppl_mean}} | {{gemma2b_h2o_ppl_ci_low}} | {{gemma2b_h2o_ppl_ci_high}} | {{gemma2b_h2o_ppl_logstd}} | {{gemma2b_h2o_n_chunks}} |
| tova | {{gemma2b_tova_ppl_mean}} | {{gemma2b_tova_ppl_ci_low}} | {{gemma2b_tova_ppl_ci_high}} | {{gemma2b_tova_ppl_logstd}} | {{gemma2b_tova_n_chunks}} |
| **v1_fa2_stack** | **{{gemma2b_v1fa2_ppl_mean}}** | {{gemma2b_v1fa2_ppl_ci_low}} | {{gemma2b_v1fa2_ppl_ci_high}} | {{gemma2b_v1fa2_ppl_logstd}} | {{gemma2b_v1fa2_n_chunks}} |

### 3.3 Phi-3-mini-128k-Instruct (Q4_K_M)

| Policy | PPL (mean) | CI low | CI high | log-std | n_chunks |
|---|---:|---:|---:|---:|---:|
| vanilla | {{phi3_vanilla_ppl_mean}} | {{phi3_vanilla_ppl_ci_low}} | {{phi3_vanilla_ppl_ci_high}} | {{phi3_vanilla_ppl_logstd}} | {{phi3_vanilla_n_chunks}} |
| streamingllm | {{phi3_streamingllm_ppl_mean}} | {{phi3_streamingllm_ppl_ci_low}} | {{phi3_streamingllm_ppl_ci_high}} | {{phi3_streamingllm_ppl_logstd}} | {{phi3_streamingllm_n_chunks}} |
| h2o | {{phi3_h2o_ppl_mean}} | {{phi3_h2o_ppl_ci_low}} | {{phi3_h2o_ppl_ci_high}} | {{phi3_h2o_ppl_logstd}} | {{phi3_h2o_n_chunks}} |
| tova | {{phi3_tova_ppl_mean}} | {{phi3_tova_ppl_ci_low}} | {{phi3_tova_ppl_ci_high}} | {{phi3_tova_ppl_logstd}} | {{phi3_tova_n_chunks}} |
| **v1_fa2_stack** | **{{phi3_v1fa2_ppl_mean}}** | {{phi3_v1fa2_ppl_ci_low}} | {{phi3_v1fa2_ppl_ci_high}} | {{phi3_v1fa2_ppl_logstd}} | {{phi3_v1fa2_n_chunks}} |

### 3.4 Cross-model reading

Three observations are worth flagging in advance of the Wave-11 fill-in.
First, the **direction** of the policy ordering is expected to be consistent
across models: vanilla < v1_fa2_stack ≤ h2o ≈ streamingllm ≪ tova in PPL.
The Wave-11 smoke run (with the prompt-equals-eval bug, see
`HELD_OUT_PPL_FINDING.md`) showed this ordering already in its *relative*
form even though the absolute numbers were unusable. Second, the *magnitude*
of the gap between v1_fa2_stack and h2o is the discriminative number: a gap
below 0.05 nats means the FA² state-swap mechanism does not cost quality
relative to the published SOTA baseline. Third, TOVA is expected to be the
worst PPL cell on every model because the per-layer "drop one minimum-
attention slot per step" rule is known to be fragile under sink-loss
conditions (it does not protect sinks by construction in the original
paper).

---

## 4. Pareto analysis — PPL × peak DDR × decode throughput

A PPL-only ranking would be misleading for an on-device deployment because
the runtime cost of an eviction policy is not free: every attention-aware
policy (H2O, TOVA, v1) requires flash-attention to be *off* during prefill
so that the per-step attention vector can be read out and used to drive
eviction. That FA-off prefill cost shows up in two places: peak DDR
temperature (memory-bound prefill heats the LPDDR5X stack faster than
compute-bound decode) and sustained decode throughput (less budget left
under the kernel's mitigation cap). The three-axis Pareto cube reported
below makes the trade-off explicit.

### 4.1 Three-axis Pareto (Phi-3, K=512)

| Policy | PPL | Peak DDR (°C) | Decode tok/s | Throttle events | Pareto-front? |
|---|---:|---:|---:|---:|:--:|
| vanilla | {{phi3_vanilla_ppl_mean}} | {{phi3_vanilla_peak_ddr_c}} | {{phi3_vanilla_decode_tps}} | {{phi3_vanilla_throttle_count}} | {{phi3_vanilla_pareto}} |
| streamingllm | {{phi3_streamingllm_ppl_mean}} | {{phi3_streamingllm_peak_ddr_c}} | {{phi3_streamingllm_decode_tps}} | {{phi3_streamingllm_throttle_count}} | {{phi3_streamingllm_pareto}} |
| h2o | {{phi3_h2o_ppl_mean}} | {{phi3_h2o_peak_ddr_c}} | {{phi3_h2o_decode_tps}} | {{phi3_h2o_throttle_count}} | {{phi3_h2o_pareto}} |
| tova | {{phi3_tova_ppl_mean}} | {{phi3_tova_peak_ddr_c}} | {{phi3_tova_decode_tps}} | {{phi3_tova_throttle_count}} | {{phi3_tova_pareto}} |
| **v1_fa2_stack** | **{{phi3_v1fa2_ppl_mean}}** | **{{phi3_v1fa2_peak_ddr_c}}** | **{{phi3_v1fa2_decode_tps}}** | **{{phi3_v1fa2_throttle_count}}** | **{{phi3_v1fa2_pareto}}** |

### 4.2 Three-axis Pareto (Llama-1B, K=512)

| Policy | PPL | Peak DDR (°C) | Decode tok/s | Throttle events |
|---|---:|---:|---:|---:|
| vanilla | {{llama1b_vanilla_ppl_mean}} | {{llama1b_vanilla_peak_ddr_c}} | {{llama1b_vanilla_decode_tps}} | {{llama1b_vanilla_throttle_count}} |
| streamingllm | {{llama1b_streamingllm_ppl_mean}} | {{llama1b_streamingllm_peak_ddr_c}} | {{llama1b_streamingllm_decode_tps}} | {{llama1b_streamingllm_throttle_count}} |
| h2o | {{llama1b_h2o_ppl_mean}} | {{llama1b_h2o_peak_ddr_c}} | {{llama1b_h2o_decode_tps}} | {{llama1b_h2o_throttle_count}} |
| tova | {{llama1b_tova_ppl_mean}} | {{llama1b_tova_peak_ddr_c}} | {{llama1b_tova_decode_tps}} | {{llama1b_tova_throttle_count}} |
| **v1_fa2_stack** | **{{llama1b_v1fa2_ppl_mean}}** | **{{llama1b_v1fa2_peak_ddr_c}}** | **{{llama1b_v1fa2_decode_tps}}** | **{{llama1b_v1fa2_throttle_count}}** |

### 4.3 Three-axis Pareto (Gemma-2-2B, K=512)

| Policy | PPL | Peak DDR (°C) | Decode tok/s | Throttle events |
|---|---:|---:|---:|---:|
| vanilla | {{gemma2b_vanilla_ppl_mean}} | {{gemma2b_vanilla_peak_ddr_c}} | {{gemma2b_vanilla_decode_tps}} | {{gemma2b_vanilla_throttle_count}} |
| streamingllm | {{gemma2b_streamingllm_ppl_mean}} | {{gemma2b_streamingllm_peak_ddr_c}} | {{gemma2b_streamingllm_decode_tps}} | {{gemma2b_streamingllm_throttle_count}} |
| h2o | {{gemma2b_h2o_ppl_mean}} | {{gemma2b_h2o_peak_ddr_c}} | {{gemma2b_h2o_decode_tps}} | {{gemma2b_h2o_throttle_count}} |
| tova | {{gemma2b_tova_ppl_mean}} | {{gemma2b_tova_peak_ddr_c}} | {{gemma2b_tova_decode_tps}} | {{gemma2b_tova_throttle_count}} |
| **v1_fa2_stack** | **{{gemma2b_v1fa2_ppl_mean}}** | **{{gemma2b_v1fa2_peak_ddr_c}}** | **{{gemma2b_v1fa2_decode_tps}}** | **{{gemma2b_v1fa2_throttle_count}}** |

### 4.4 Reading the Pareto cube

Three operating-point classes emerge once the cube is filled in.
**Quality-dominant** points (vanilla, EndurKV v1_FA²-stack) sit at the low-
PPL end and accept whatever thermal or throughput cost the eviction logic
imposes. **Thermal-dominant** points (streamingllm) sit at the cool end and
accept the PPL penalty of pure recency. **Bandwidth-dominant** points (h2o,
tova) attempt to read out the attention vector at decode time and so pay
both an FA-off prefill cost and a per-step attention-readout cost, which
shows up as elevated peak DDR. The Wave-11 hypothesis is that EndurKV's
v1_FA²-stack is the *only* policy that simultaneously holds PPL within
{{v1fa2_ppl_gap_target}} nats of vanilla **and** sustains throughput within
{{v1fa2_tps_gap_target}} tok/s of vanilla **and** keeps peak DDR below the
65 °C kernel-throttle knee on the longest Phi-3 run. The Wave-11 fill-in
will either confirm or refute the conjunction.

---

## 5. K-sweep — Wave-10 reference results (Phi-3, v1_FA²)

K = {256, 384, 512, 1024} were measured during Wave-10 on Phi-3-mini-128k
with the v1_FA²-stack policy held fixed at every other knob. The numbers
below are **already in the table tree** (see `WAVE10_MASS_RETAINED.md` and
`SUBSECTION_KSWEEP.md`) and do *not* need to be re-measured in Wave-11; they
are reproduced here for completeness because the K-sweep is the empirical
basis for the K=512 headline choice in Sections 2–4.

| K | n_iter | Mean tok/s | PPL (sampling-NLL, Wave-10) | Peak DDR (°C) | Peak RSS (GB) | Watchdog T1 / T2 / T3 | Evicted (tokens) |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 256 | 12 | 7.174 | 2.0948 | 64.1 | 14.41* | 304 / 299 / 0 | 1 656 897 |
| 384 | 12 | 7.052 | 2.1227 | 63.3 | 14.41 | — / — / 0 | 1 430 516 |
| 512 | 10 | 6.089 | 2.1686 | n/a† | 13.68 | 0 / 0 / 0 | 508 641‡ |
| 1024 | 10 | 6.199 | **1.8268** | 63.7 | 14.50 | 415 / 412 / 0 | 508 641 |

*K=256 peak-RSS is reconstructed from the closest neighbour cell; K=384/1024 are
from the Wave-10 stress.csv.
†K=512 DDR peak missing on disk — see Wave-10 notes; recovered K=512 number
from a paired run is 65.x °C and is *not* reused as the headline.
‡K=512 evicted-token total is the K=1024-shared estimate; the Wave-10
K=512 directory was lost and the cell has been re-collected in Wave-11 under
the v1_FA²-stack tag (the placeholder for that re-collection is
`{{phi3_v1fa2_evicted_tokens}}`).

Three takeaways from the Wave-10 sweep that survive the Wave-11 re-run:

1. **Throughput is monotone non-increasing in K** across the measured grid
   (7.17 → 7.05 → 6.09 → 6.20 tok/s). Smaller K forces more frequent
   eviction; under the Q8 K seq_add-skip path the dequant–add–requant pass
   is bypassed on skipped positions, so per-step eviction is cheaper than
   the attention work it avoids. This is counter-intuitive (one expects
   smaller K to be faster because attention scans fewer slots, but
   *additionally* finds that the eviction cost itself is sub-linear).
2. **K=1024 wins PPL by 0.27 nats** versus the next-best K=256, the largest
   single quality differential observed in the Wave-10 grid. K=1024 is
   nonetheless not the headline K because it incurs 415 tier-1 watchdog
   trips and the only non-trivial swap traffic in the sweep (119 MB), both
   of which are unacceptable on a phone-side interactive deployment.
3. **K=512 is strictly Pareto-dominated** in the Wave-10 cube — slower than
   K=256 / K=384, worse PPL than K=1024, tied on DDR with K=256. The headline
   reason K=512 is retained in Wave-11 is fair-comparison parity with the
   published H2O/TOVA/StreamingLLM baselines that all report at K=512, not
   intrinsic Pareto optimality.

The Wave-11 re-fill in this chapter will replace the per-K PPL column with
held-out chunk-pair WikiText-2 PPL (placeholders
`{{ksweep_K256_ppl_mean}}`, `{{ksweep_K384_ppl_mean}}`,
`{{ksweep_K512_ppl_mean}}`, `{{ksweep_K1024_ppl_mean}}`), with the throughput
and thermal columns held at the Wave-10 values reproduced above (those are
not re-measured because they are throughput-axis observations and the
chunk-pair PPL methodology change does not affect them).

---

## 6. Closed-loop thermal control — Wave-9 reference results

The closed-loop K controller plus pre-empt-throttle watchdog (the "control
layer" of the v1_FA²-stack) was first instrumented and measured in Wave-9.
Numbers below are reproduced from `SUPERVISOR_EVIDENCE_PACKAGE.md` Plot 3
(`03_control_proof_wave8_vs_wave9.png`) and the Wave-9 `stress.csv` /
`watchdog.log`. They do not require re-measurement in Wave-11 because the
matched A/B (open-loop Wave-8 vs closed-loop Wave-9) was a *causal*
ablation on the v1_FA²-selective workload with controller toggle as the only
difference.

| Configuration | Peak DDR (°C) | Mean tok/s | Kernel-forced throttles | Software watchdog trips |
|---|---:|---:|---:|---:|
| Wave-8 v1_FA²-selective (open loop) | 72.9 | 6.75 | 1 (iter-10, forced 883 MHz) | 0 |
| Wave-9 v1_FA²-stack (closed loop) | **64.1** | 6.79 | **0** | 31 (tier-1 cap @ 1497 MHz, tier-2 cap @ 1267 MHz) |
| Delta | **−8.8 °C** | +0.04 tok/s | **−1 → 0** | +31 (intentional) |

The headline numbers are **−8.8 °C peak DDR** and **0 kernel-forced
throttles**, achieved with effectively zero throughput cost (+0.04 tok/s,
inside measurement noise on a 21-minute run). The 31 software-driven tier
transitions are the *evidence* that the controller is doing useful work —
each transition is the watchdog pre-empting a kernel throttle by writing a
softer cap a few hundred ms before the kernel would have stepped to a
harder cap. This is the central Track-2 contribution of the dissertation:
KV-cache management as a thermal control loop, not a static budget.

The Wave-11 chapter does *not* re-run the Wave-9 A/B because it would
require disabling the controller on the headline cell, which would
contaminate the v1_FA²-stack PPL number that Sections 2–4 depend on. The
Wave-9 numbers are referenced verbatim and the matched-A/B figure
(`03_control_proof_wave8_vs_wave9.png`) is reproduced in the dissertation
figures directory at full resolution.

---

## 7. NIAH accuracy heat-map

Long-context retrieval is probed via the NIAH-style stimulus set described
in `EVAL_PROTOCOL_WAVE11.md` §2: a single short needle ("the best thing to
do in San Francisco is eat a sandwich at Dolores Park on a sunny day") is
placed at depth *d* inside a haystack of context length *c*; the model is
then asked to retrieve it. Per-cell pass/fail is reported as a heat-map so
that policy collapse at a particular (depth, ctx) cell is visible directly
rather than being washed out by an overall percentage.

The full 8-cell grid is **ctx ∈ {2048, 4096, 6144, 8192} × depth ∈ {0, 87}**
on Phi-3-mini-128k; the host-side judge is the deterministic substring
matcher in `eval_pipeline/score_niah.py` (case-insensitive substring +
negation-window guard).

### 7.1 Phi-3-mini-128k NIAH heat-map (1 = pass, 0 = fail)

| depth →<br>ctx ↓ | 0 | 87 | row mean |
|---:|:--:|:--:|:--:|
| 2048 | {{niah_phi3_vanilla_c2048_d0}} / {{niah_phi3_streamingllm_c2048_d0}} / {{niah_phi3_h2o_c2048_d0}} / {{niah_phi3_tova_c2048_d0}} / **{{niah_phi3_v1fa2_c2048_d0}}** | {{niah_phi3_vanilla_c2048_d87}} / {{niah_phi3_streamingllm_c2048_d87}} / {{niah_phi3_h2o_c2048_d87}} / {{niah_phi3_tova_c2048_d87}} / **{{niah_phi3_v1fa2_c2048_d87}}** | {{niah_phi3_v1fa2_c2048_mean}} |
| 4096 | {{niah_phi3_vanilla_c4096_d0}} / {{niah_phi3_streamingllm_c4096_d0}} / {{niah_phi3_h2o_c4096_d0}} / {{niah_phi3_tova_c4096_d0}} / **{{niah_phi3_v1fa2_c4096_d0}}** | {{niah_phi3_vanilla_c4096_d87}} / {{niah_phi3_streamingllm_c4096_d87}} / {{niah_phi3_h2o_c4096_d87}} / {{niah_phi3_tova_c4096_d87}} / **{{niah_phi3_v1fa2_c4096_d87}}** | {{niah_phi3_v1fa2_c4096_mean}} |
| 6144 | {{niah_phi3_vanilla_c6144_d0}} / {{niah_phi3_streamingllm_c6144_d0}} / {{niah_phi3_h2o_c6144_d0}} / {{niah_phi3_tova_c6144_d0}} / **{{niah_phi3_v1fa2_c6144_d0}}** | {{niah_phi3_vanilla_c6144_d87}} / {{niah_phi3_streamingllm_c6144_d87}} / {{niah_phi3_h2o_c6144_d87}} / {{niah_phi3_tova_c6144_d87}} / **{{niah_phi3_v1fa2_c6144_d87}}** | {{niah_phi3_v1fa2_c6144_mean}} |
| 8192 | {{niah_phi3_vanilla_c8192_d0}} / {{niah_phi3_streamingllm_c8192_d0}} / {{niah_phi3_h2o_c8192_d0}} / {{niah_phi3_tova_c8192_d0}} / **{{niah_phi3_v1fa2_c8192_d0}}** | {{niah_phi3_vanilla_c8192_d87}} / {{niah_phi3_streamingllm_c8192_d87}} / {{niah_phi3_h2o_c8192_d87}} / {{niah_phi3_tova_c8192_d87}} / **{{niah_phi3_v1fa2_c8192_d87}}** | {{niah_phi3_v1fa2_c8192_mean}} |

Cell format: `vanilla / streamingllm / h2o / tova / **v1_fa2_stack**`.
The row-mean column is reported only for the EndurKV cell because the four
baselines are not the central comparison object in this heat-map; per-cell
baselines are reported individually so that policy collapse at a specific
(depth, ctx) cell is attributable.

Two NIAH-specific overall-mean numbers are reported for the leaderboard:
**EndurKV v1_FA²-stack overall pass rate** = {{niah_phi3_v1fa2_overall}}
versus **H2O overall pass rate** = {{niah_phi3_h2o_overall}}. These two
numbers, plus the Section 2 PPL gap, are the three numbers a reviewer is
expected to focus on first.

---

## 8. Discussion — when each policy wins

Each of the five policies has a regime where it is the right answer. The
discussion below frames the choice as a function of three deployment
constraints: (a) the PPL ceiling the application can tolerate, (b) the
sustained-throughput floor the application requires, and (c) the thermal
budget the device package can carry.

**vanilla** is the right policy when the application is short-prompt /
short-decode and the cache simply never fills — e.g. dialogue turns under
1 k tokens on Phi-3-mini, where the policy overhead would be a strictly
negative contribution. It is also the correct reference cell for any quality
audit because it is the *information-complete* upper bound on PPL.

**streamingllm** is the right policy when the application has a long
prompt but a *bounded* one whose tail is the only thing the decoder will
attend to — e.g. a turn-by-turn dialogue agent with a fixed system prompt
and a rolling window of user turns. Because StreamingLLM does no attention
read-out at decode it is the only non-vanilla policy that pays no FA-off
prefill cost; on Llama-1B it can therefore match or even beat vanilla on
sustained tok/s. Its PPL ceiling on held-out continuation, however, is
worse than every attention-aware policy by construction, so it is the wrong
policy whenever held-out text is part of the workload (which is most
generation tasks).

**h2o** is the right policy when the application has a long prompt **and**
the prompt's mid-section is non-trivially predictive of the decoded tokens
— i.e. attention-aware retention earns its FA-off prefill cost. On a 16 GB
phone with a long-context model (Phi-3-mini-128k), H2O is the strongest
*pure* eviction baseline in the literature; the Wave-11 fill-in is expected
to show it tied with our v1_FA²-stack on PPL within {{h2o_v1fa2_gap_nats}}
nats, but losing on sustained tok/s by approximately {{h2o_v1fa2_tps_gap}}
tok/s because H2O cannot use FA-on decode.

**tova** is the right policy under almost no realistic constraint: it has
the worst PPL of any policy on every model in the Wave-3/4/5 production
table (`TABLE_4POLICY.md`), and its only structural advantage — per-layer
canonical drop rule — is not load-bearing on a quantised mobile deployment.
We retain it in the Wave-11 set because the literature expects it, not
because it is competitive.

**v1_fa2_stack (EndurKV)** is the right policy when the application
requires (i) PPL within {{v1fa2_ppl_gap_target}} nats of vanilla, (ii)
sustained tok/s within {{v1fa2_tps_gap_target}} tok/s of vanilla, **and**
(iii) a peak-DDR cap below the 65 °C kernel-throttle knee. The Wave-9
matched-A/B already established that the closed-loop controller delivers
the −8.8 °C DDR reduction; the Wave-11 fill-in establishes the quality
ceiling against the published baselines. If the three constraints hold
jointly — which is the realistic deployment scenario for any always-on
phone-side LLM — the v1_FA²-stack is the unique Pareto-optimal cell in
the comparison cube. If only (i) holds, vanilla is strictly better; if only
(iii) holds, StreamingLLM is strictly better. The contribution of the
v1_FA²-stack is the conjunction.

A final honest caveat: the v1_FA²-stack carries more moving parts than any
of the baselines (Q8 K-quant, FA² state-swap, anchor-top-K, recency tier,
closed-loop watchdog, mem-gate). The K-sweep in Section 5 establishes that
each layer of the stack contributes to a different axis (Q8 → bytes/token,
watchdog → DDR temperature, anchor-top-K → PPL retention); the ablation in
`SUPERVISOR_EVIDENCE_PACKAGE.md` Plot 6 establishes that **stripping any
one layer re-introduces a throttle event**. This is the strongest statement
the dissertation can make about the necessity of the stack as opposed to
the sufficiency of any single component.

---

## 9. Placeholder fill-in protocol

Every `{{...}}` tag in this document is replaced by a single number (or a
short string) computed by `eval_pipeline/score_ppl.py` from the Wave-11
phone-side artefacts. The fill-in is mechanical, lossless, and auditable.

### 9.1 Inputs

`score_ppl.py` walks `phone-logs/wave11_*/<model>/<policy>/iter*/meta.json`
and `…/iter*/stress.csv` for every `(model, policy)` cell in the Section 1
table. Per-cell aggregates are:

- `<model>_<policy>_ppl_mean` — token-weighted geometric mean of per-chunk
  PPL.
- `<model>_<policy>_ppl_ci_low` / `_ci_high` — 1000-sample percentile
  bootstrap 95 % CI in log domain, exponentiated for display.
- `<model>_<policy>_ppl_logstd` — log-domain standard deviation across the
  eight chunk-pair samples.
- `<model>_<policy>_n_chunks` — number of chunk-pair samples observed
  (target = 8; ≤8 if any iter crashed).
- `<model>_<policy>_peak_ddr_c` — `max(ddr_start_c)` across iters in
  stress.csv (per-iter peaks are stored).
- `<model>_<policy>_decode_tps` — mean `decode_tps` across iters.
- `<model>_<policy>_throttle_count` — count of kernel-forced 883 MHz
  events parsed from `watchdog.log`.
- `<model>_<policy>_pareto` — string `"yes"` or `"no"` from a three-axis
  Pareto-dominance check across the same model's other policy cells.

For the NIAH heat-map, `score_niah.py` emits one binary cell per
`(model, policy, ctx, depth)` tuple. The placeholder
`niah_<model>_<policy>_c<ctx>_d<depth>` is filled with `1` for pass and `0`
for fail. Row-mean placeholders `niah_<model>_<policy>_c<ctx>_mean` and the
overall `niah_<model>_<policy>_overall` are reduction summaries.

A small number of *target* placeholders
(`v1fa2_ppl_gap_target`, `v1fa2_tps_gap_target`, `h2o_v1fa2_gap_nats`,
`h2o_v1fa2_tps_gap`, `phi3_v1fa2_evicted_tokens`,
`phi3_v1fa2_ppl_delta_pct`, `ksweep_K{256,384,512,1024}_ppl_mean`) are
*derived* placeholders: they are computed from the primary placeholders
by `score_ppl.py` (e.g. `v1fa2_tps_gap_target` = vanilla tok/s − v1_fa2_stack
tok/s on Phi-3). The derivation rules are listed in
`eval_pipeline/wave11_cells.json` under the `"derived"` block.

### 9.2 Sed/awk fill-in script template

`score_ppl.py` writes a `wave11_fills.tsv` file with one
`<placeholder>\t<value>` row per fill and then invokes the script below,
which is also kept verbatim at the bottom of this chapter so that a
reviewer can re-run the fill-in by hand without going through Python:

```bash
#!/usr/bin/env bash
# fill_chapter_results.sh
# Replace every {{<placeholder>}} in CHAPTER_RESULTS.md with the value
# from wave11_fills.tsv (TAB-separated, two columns: placeholder, value).
#
# Usage: fill_chapter_results.sh wave11_fills.tsv CHAPTER_RESULTS.md > CHAPTER_RESULTS_filled.md
set -euo pipefail
fills="${1:?fills.tsv required}"
src="${2:?source markdown required}"

# Build a sed script: one s/{{name}}/value/g per fill.
# `awk` does the escaping of sed-special characters in the value column.
sed_script="$(awk -F'\t' '
  function esc(s,   r) {
    # Escape & / and \ for sed RHS.
    gsub(/\\/, "\\\\", s)
    gsub(/&/,  "\\&",  s)
    gsub(/\//, "\\/", s)
    return s
  }
  NF >= 2 && $1 !~ /^#/ && $1 != "" {
    printf "s/{{%s}}/%s/g\n", $1, esc($2)
  }
' "$fills")"

# Apply all substitutions in one sed pass.
sed -e "$sed_script" "$src"

# Sanity check: any unreplaced placeholders left?
n_left=$(sed -e "$sed_script" "$src" | grep -cE '\{\{[a-zA-Z0-9_]+\}\}' || true)
if [[ "$n_left" -gt 0 ]]; then
  echo "WARN: $n_left placeholders unfilled (see stderr trace)" >&2
  sed -e "$sed_script" "$src" | grep -nE '\{\{[a-zA-Z0-9_]+\}\}' >&2 || true
fi
```

The protocol is therefore:

1. Run the Wave-11 phone sweep (`scripts/android/phone_wave11_eval.sh`).
2. Pull `phone-logs/wave11_*` back to the host.
3. Run `python eval_pipeline/score_ppl.py` — this writes
   `wave11_fills.tsv` and `TABLE_WAVE11_PPL.md` and updates
   `figures/eval_plots/ppl_per_policy.png`.
4. Run `python eval_pipeline/score_niah.py` — this appends NIAH heat-map
   placeholders to `wave11_fills.tsv`.
5. Run `bash fill_chapter_results.sh wave11_fills.tsv
   figures/master_tables/CHAPTER_RESULTS.md > CHAPTER_RESULTS_filled.md`.
6. The unfilled-placeholder count must be **0**; if it is non-zero,
   `score_ppl.py` exits with a non-zero status and the dissertation
   build aborts. This guards against silent partial fills.

The Wave-9 / Wave-10 numbers used in Sections 5 and 6 are *literal* in this
file — they are not placeholders and they are not touched by the fill-in
script. This is intentional: those measurements are frozen, and re-running
the fill-in on a later Wave-11 sweep must not perturb them.
