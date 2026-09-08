# TTFT Path Forward — Synthesis

**Status:** Decision document
**Inputs:** TTFT_VS_K_EMPIRICAL (K-sweep), snapkv_feasibility_assessment (eviction_bench.cpp architectural review), Mobile LLM TTFT Field Comparison (published baselines).

---

## 1. Will K=1024 fix TTFT?

**Almost certainly NO.** Both physics and empirics agree:

- **Empirical (decisive):** Across K in {256, 384, 512, 1024} on Phi-3-mini with a 272-tok prompt under v1_fa2, prefill_ms varies by **0.27%** total (33457 / 33455 / 33534 / 33444 ms). Across K in {512, 2048} on Llama-3.2-1B with an 8007-tok prompt under v1, it varies by **2.47%**. Both spreads sit *far* below the 17.2% same-K cross-wave noise floor.
- **Mechanism:** Prefill is O(prompt_len^2 * d_model) dense attention. K-dependent work is one batched O((prompt_len - K) * n_layers) eviction at the tail. That is negligible against the dense quadratic. v1_fa K=512 (33491.7 ms) and v1_fa2 K=512 (33533.7 ms) are indistinguishable — the FA2 stack adds zero prefill cost.
- **Consequence:** Raising K from 512 -> 1024 -> 2048 *will* help long-context PPL (canonical SnapKV/H2O quality scaling) but it **cannot** materially change TTFT. The K knob is decode-side only.

**Verdict:** Running a K=1024 sweep is worthwhile only as a PPL-closure data point. It is *not* a TTFT remediation.

---

## 2. Will canonical SnapKV fix TTFT?

**Yes — but only because canonical SnapKV replaces the FA-OFF prefill path with a fast FA-ON prefill, not because of anything K-related.**

From the snapkv_feasibility_assessment:

- **Canonical-spirit path (recommended, 6-10 h, low risk):** *Keeps FA-OFF prefill.* Adds correct observation-window capture + 1D-conv pooling + top-K + recent-window selection. Reuses the proven v1_fa state-swap. This **does not lower TTFT** — same prefill cost as v1_fa. It only fixes the *eviction algorithm* (and thus PPL/quality).
- **True canonical path (16-24 h, medium-high risk):** *FA-ON prefill* + separate re-attention pass via shadow-context state-swap. This is the only path that *could* deliver lower TTFT (~1.5-2x prefill speedup from FA-ON prefill), but it depends on cross-FA-mode `llama_state_seq_get/set_data` being reliable — currently flagged as unreliable at eviction_bench.cpp:1115 ("state_set failed (likely KV layout differs across FA modes)"). The eviction_bench today only does FA-OFF -> FA-ON transfer; the reverse direction is unvalidated.

**Verdict:** SnapKV "fixes TTFT" only if we take Path 2 (FA-ON prefill, high risk, 2-3 days). Path 1 ships the correct eviction algorithm with the SAME TTFT as v1_fa. **Neither path can match vanilla (no-eviction) TTFT** — vanilla skips the eviction machinery entirely.

---

## 3. Is our vanilla 305s TTFT competitive vs published Phi-3 TTFT?

**No, not against any optimized on-device stack — but yes, as a "vanilla baseline" reference number.**

From the Mobile LLM TTFT Field Comparison:

| Stack | Model | Prompt | TTFT |
|---|---|---|---|
| llama.cpp Hexagon NPU | Llama-3.2-1B Q4 | 512 tok | **~3 s** |
| mllm-NPU (SD 8 Gen 3) | Llama-2-7B | 1024 tok | ~27-54 ms prefill block |
| MLC-LLM (iPhone 13 Pro Max) | Llama-2-7B 3bit | 512 tok | ~9.4 s |
| PowerInfer-2 (SD 8 Gen 3) | TurboSparse-Mistral-7B Q4 | 512 tok | "a few seconds" |
| **llama.cpp CPU (vanilla)** | Llama-2-7B Q4 | 512 tok | **170-300+ s** |
| **Our vanilla** | Phi-3-mini Q4 | (prompt) | **305 s** |

- **Our 305 s is in-line with the unoptimized llama.cpp CPU baseline** on Snapdragon-class hardware for 7B-class Q4 long-prompt workloads (PalmBench, "Understanding LLMs in Your Pockets" both report 170-300+ s). It is a believable vanilla number, not a measurement bug.
- **It is 30-100x slower than the on-device SOTA** (Hexagon NPU, mllm-NPU, PowerInfer-2, MLC-GPU, MNN-LLM) — all of which use NPU/GPU offload, sparse activation, or operator-optimized engines that EndurKV's CPU path does not.
- **For the paper:** the 305 s number is publishable *as the unoptimized CPU baseline*, but it must be explicitly framed that way. EndurKV's contribution is *decode-side endurance + long-decode quality at fixed budget*, not prefill speed.

---

## 4. Path of Least Regret

Three options on the table:

### (a) Run K=1024 sweep
- **Cost:** ~1 wave run.
- **Gains:** Confirms the SnapKV/H2O PPL gap closes at larger K. Strengthens "K is decode-side" empirical evidence (already at 0.27% spread; adding K=2048 on Phi-3 would seal it).
- **Misses:** Does nothing for TTFT.

### (b) Implement canonical SnapKV first
- **Cost:** 6-10 h dev (Path 1, canonical-spirit) + 4 h smoke + relaunch. Path 2 (true canonical, FA-ON prefill) is 16-24 h and medium-high risk.
- **Gains:** Path 1 fixes the eviction algorithm (correct observation-window, 1D pooling, top-K, recent window) -> closes SnapKV PPL gap at LOW K. Path 2 *additionally* would lower SnapKV's TTFT toward vanilla.
- **Misses:** Path 1 does NOT lower v1_fa2 TTFT. Path 2 might, but risks blowing the timeline on a state-swap bug we've already flagged.

### (c) Frame the paper honestly
- **Cost:** Writing time only.
- **Gains:** v1_fa2_stack wins long-decode throughput + endurance + mass-retained quality at fixed K. Vanilla wins TTFT (because it does no eviction work). These are *complementary* claims, not contradictory ones. The TTFT gap is an honest accounting of the eviction-bookkeeping cost.
- **Misses:** Doesn't close the SnapKV PPL gap.

---

## Recommendation

**Sequence: (b-Path-1) -> (a) -> (c). Do NOT attempt Path 2 unless we have a clear ~3-day budget AND have first validated cross-FA-mode state-swap on a microbench.**

1. **First (6-10 h, low risk): Implement canonical SnapKV Path 1.** This is the *correct* scientific comparison — our current SnapKV implementation is not faithful to the paper (no observation-window pooling, no proper top-K-by-pooled-attention). Fixing this is non-negotiable for publication regardless of the TTFT story. It reuses the proven v1_fa state-swap so risk is low. The fix is a ~50-100 LOC patch in eviction_bench.cpp around lines 247-273 (extract last-N rows in eval_callback) + a new selection routine before apply_eviction.
2. **Then (1 wave, ~hours): Run a K=1024 (and ideally K=2048) sweep** on Phi-3 with the corrected SnapKV. Two purposes: (i) confirm SnapKV PPL closes at larger K with the canonical algorithm, (ii) put a third Phi-3 data point on the TTFT-vs-K empirical curve to make the "K is decode-side" claim airtight.
3. **Then frame the paper:** position 305 s explicitly as the unoptimized llama.cpp CPU baseline (cite PalmBench / "Understanding LLMs in Your Pockets" / SD-8-Gen-3 numbers). Lead with v1_fa2_stack's wins: long-decode throughput, thermal endurance, mass-retained quality at fixed K. Concede the TTFT cost honestly: it is the price of decode-side eviction bookkeeping, and it is *orthogonal* to the NPU/sparse/GPU axis that the SOTA stacks compete on.

**Explicitly DO NOT:** burn cycles trying to make v1_fa2_stack TTFT match vanilla. Physics (full attention is still full attention) and the empirical K-invariance both say there is no eviction-side knob that lowers prefill cost. The only thing that lowers prefill cost is changing the prefill path itself (FA-ON, NPU offload, sparse activation), and that is out of scope for this paper.

**Expected outcomes after sequence (b1 -> a -> c):**
- SnapKV PPL closes at canonical K (paper-faithful comparison).
- Three Phi-3 K points (256/512/1024 or 512/1024/2048) all within ~1% TTFT — empirical K-invariance is unimpeachable.
- Paper's headline is honest, defensible, and complementary to (not in competition with) PowerInfer-2 / mllm-NPU / Hexagon-NPU TTFT numbers.

---

## Summary table

| Option | Fixes TTFT? | Fixes PPL? | Cost | Risk |
|---|---|---|---|---|
| (a) K=1024 sweep | No | Partial (closes gap at high K) | ~hours | Low |
| (b1) SnapKV Path 1 (FA-OFF) | No | **Yes** (canonical algorithm) | 6-10 h | Low |
| (b2) SnapKV Path 2 (FA-ON) | **Yes** (1.5-2x prefill) | Yes | 16-24 h | Medium-High (state-swap reliability) |
| (c) Honest framing | No (frames it) | No | Writing only | None |

**Path of least regret:** (b1) -> (a) -> (c). Total ~1.5-2 days of work, low risk, paper-faithful, defensible TTFT story.
