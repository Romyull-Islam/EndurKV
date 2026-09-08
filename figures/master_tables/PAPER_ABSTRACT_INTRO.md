# Thermal and Endurance Co-Aware KV Management for Sustained Mobile LLM Inference Driven by Model-Internal Signals

**Venue:** HotMobile 2027
**Hardware:** OnePlus 15, Snapdragon 8 Elite Gen 5, Adreno 840, 12 GB UMA

---

## Abstract

On-device large language model (LLM) inference on flagship smartphones is no longer
bound by arithmetic throughput: with weights in 2-4 bit quantisation and unified
memory architectures (UMA) crossing 12-16 GB, the binding constraint is *sustained
DRAM bandwidth under thermal load*. We measure, on a OnePlus 15 (Snapdragon 8
Elite Gen 5, Adreno 840), that vanilla llama.cpp running Phi-3-mini-128k at
Q4_K_M holds {{wave11_phi3_vanilla_peak_ddr}} °C peak DDR and triggers kernel-side
preempt-throttle to 883 MHz between iterations 6 and 10 of a sustained
narrative-QA stream — capping usable throughput well below the cold-start figure
the user sees in the first second. Existing KV eviction policies (TOVA, H2O,
StreamingLLM) target cloud GPU memory pressure and ignore the phone's thermal
floor. We present **EndurKV**, a closed-loop preempt-throttle watchdog composed
with a bounded KV eviction policy whose budget is driven by model-internal
signals (attention concentration, Q8 K-cache footprint) and a memory-pressure
gate. On Phi-3-mini-128k narrative-QA, the composed v1_FA²-stack achieves a peak
DDR drop of {{wave9_dd_drop_C}}=8.8 °C versus vanilla with
{{wave9_throttle_count}}=0 kernel throttle events over a 10-iteration sustained
run, while held-out WikiText-2 perplexity remains within bounds of the strongest
prior policy. To our knowledge this is the first on-device measurement on
Snapdragon 8 Elite Gen 5 isolating *cache size* as the binding thermal lever
(Wave-4: 8.5 °C swing between K=512 and unbounded at fixed compute).

---

## 1. Introduction

### 1.1 Smartphone LLM inference is now real

In 2026, on-device LLM inference graduated from a research curiosity to a
shipping primitive. Flagship phones routinely carry 12-16 GB of unified memory,
Q2-Q4 weight quantisation has been validated end-to-end on production models,
and decoder-only stacks of 3-8 B parameters now fit comfortably in user-space
with hundreds of megabytes of slack. The OnePlus 15 we use in this paper packs a
Snapdragon 8 Elite Gen 5 SoC, Adreno 840, and 12 GB of LPDDR5X — running Phi-3
mini-128k-Instruct at Q4_K_M leaves enough headroom for a 9.8 K-token narrative
prompt plus a KV cache approaching 2 GB. The naive read of this state of the
world is that we have won: the model fits, the model runs, ship it.

We did not win. We measured.

### 1.2 The thermal cliff

When we run Phi-3-mini-128k Q4_K_M on the OnePlus 15 for a sustained narrative-QA
workload (Wave-3, 9794-token prompt, 2-iteration stream), peak DDR temperature
reaches {{wave11_phi3_vanilla_peak_ddr}} °C and the userspace scheduler caps the
two performance clusters at 883 MHz once the SoC crosses its 65 °C DDR
soft-trip. The first iteration prints at a respectable cold-start rate; the
sixth through tenth iterations crawl. This is the *thermal cliff*: a sustained
inference pipeline that looked viable in a 30-second demo collapses on the
fifth minute of the same workload, because the binding resource is not compute
and not memory capacity but sustained DRAM bandwidth under thermal load. In
our Wave-4 long-decode regime we see the cliff clearly: unbounded-cache vanilla
hits 62.9 °C peak DDR; the same model with the KV cache held at K=512 hits 54.4
°C — a **8.5 °C swing at fixed compute and fixed quantisation**. The KV cache
size, not arithmetic intensity, is the thermal lever.

### 1.3 Existing KV eviction does not solve this

The recent flood of KV eviction work — TOVA, H2O, StreamingLLM, KIVI, and the
SnapKV family — was overwhelmingly designed against a *cloud GPU* objective:
trim a 2 M-token cache so it fits the HBM budget on an A100, with the
secondary effect of recovering compute. The selection criteria are
attention-mass or recency-based and the policies are evaluated against
perplexity-neutrality on long-context benchmarks. None of these policies are
evaluated against an on-device thermal budget, none of them close a loop on
the DRAM-bandwidth signal that the *phone* actually constrains, and crucially
none of them treat the cache budget K as a control variable driven by
*model-internal* signals (attention concentration, Q8 K-cache footprint
spread) coupled to a *device-internal* signal (DDR temperature, pswpout
pressure). When we ran TOVA at K=512 on Phi-3 (Wave-3, narrative-QA), it gave
us 60.6 °C peak DDR — *hotter* than vanilla, because the prefill-side eviction
overhead dominates the cache-shrink benefit on phone-class memory hierarchies.
Cloud policies do not transfer.

### 1.4 EndurKV thesis: cache size is the binding thermal lever

The smoking gun is Wave-4. Holding everything else fixed — model, quantisation,
prompt, hardware, ambient — varying *only the KV budget* moves peak DDR by 8.5
°C and converts a 5-iteration thermal-throttled run into an 8-iteration
non-throttled run. This says that the right object for an on-device KV policy
to control is not "which tokens to keep" (the cloud question) but **what the
total cache footprint is, as a function of the device's current thermal
state**. EndurKV is built on that thesis.

Concretely, EndurKV composes three layers. (1) A **Q8 K-cache** that halves
K-cache DRAM traffic with negligible attention-quality cost (Wave-8). (2) A
**preempt-throttle watchdog** that reads DDR temperature at 1 Hz and applies
a CPU-frequency cap *before* the kernel does, avoiding the bimodal
freq-collapse seen in vanilla (Wave-9). (3) A **memory-pressure gate** that
reads `vmstat_pswpout` and tightens K when swap activity rises, eliminating
the 1.55 GB swap-out we saw in unconstrained v1_FA² (Wave-7 → Wave-9: 1551.8
MB → 0.0 MB). The composed stack — v1_FA²-stack — delivers the headline
{{wave9_dd_drop_C}}=8.8 °C peak DDR drop versus vanilla with
{{wave9_throttle_count}}=0 kernel-side throttle events over a 10-iteration
sustained run.

### 1.5 Contributions

This paper makes three contributions:

**(a) First on-device thermal measurement on Snapdragon 8 Elite Gen 5
(OnePlus 15).** We instrument every DDR/CPU thermal zone, every vmstat
counter, and every per-iter llama.cpp timing field, and we release a 9-wave
trace (Wave-3 through Wave-11) covering vanilla, three cloud baselines (H2O,
TOVA, StreamingLLM), and the EndurKV composed stack across narrative-QA,
long-decode, held-out WikiText-2 PPL, and Needle-in-a-Haystack. To our
knowledge no prior on-device LLM paper isolates the DDR-temperature signal as
a controlled variable on this SoC class.

**(b) The v1_FA²-stack: a composed, model-internally-driven KV policy with a
{{wave9_dd_drop_C}}=8.8 °C peak-DDR delta against vanilla.** The stack is the
first on-device KV policy to (i) close a loop on DDR temperature, (ii) drive
K from a model-internal attention-concentration signal, and (iii) compose
quantisation, eviction, and thermal control as a *single* control surface
rather than three orthogonal knobs. Wave-10's K-sweep confirms the policy is
robust to K ∈ {256, 384, 512, 1024} with monotone-non-increasing throughput
in K, validating that the cache-size lever is the binding one.

**(c) Held-out PPL evaluation revealing the true cost of eviction.** Wave-11
re-evaluates all five policies under the held-out, teacher-forced WikiText-2
chunk-pair protocol used by the H2O/KIVI/StreamingLLM/TOVA papers — replacing
the biased-low sampling-NLL we (and, we suspect, several prior on-device
papers) used through Wave-10. This is, to our knowledge, the first on-device
KV-policy comparison under the published-grade PPL protocol on Phi-3
Q4_K_M, and we report perplexity, NIAH retrieval accuracy, peak DDR,
peak CPU, and swap together per policy.

### 1.6 Roadmap

Section 2 characterises the thermal floor of the OnePlus 15 and reproduces
the Wave-4 8.5 °C cache-size swing that motivates the work. Section 3
describes the EndurKV design — the Q8 K-cache, the preempt-throttle
watchdog, and the memory-pressure gate — and how the three layers compose.
Section 4 presents the Wave-9 thermal headline and the Wave-10 K-sweep.
Section 5 presents the Wave-11 held-out PPL and NIAH evaluation against
H2O, TOVA, and StreamingLLM. Section 6 discusses limitations (single SoC,
single ambient, single model family) and what we believe transfers.
Section 7 surveys related work in on-device LLM systems and KV eviction.
Section 8 concludes.
