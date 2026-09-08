# Paper C — Combined Per-Head + Entropy-Gated Adaptive Eviction

**Status:** Future work (documented here so it does not get lost).
**Target venue:** MobiSys 2028 (full paper) or NSDI (control-theoretic framing).
**Relationship to Paper A:** Paper A's v1_fa2_stack is the per-head fixed-budget half. Paper C adds the per-step entropy gate on top, fixing Paper A's NIAH catastrophic failure.
**Relationship to Paper B:** Paper B is entropy gate alone (per-step uniform). Paper C is the combined per-head AND per-step adaptive — strictly stronger than either.

---

## Core claim

```
K_h(t) = round( K_nom · μ(m_h) · (1 − H̃(t)) )
         └─────────────┘   └────────────┘
         Paper A's per-head      Paper B's entropy gate
         (fixed at prefill)      (per-step time-varying)
```

This occupies the **top-left quadrant** of the design space (per-head AND per-step adaptive). No published policy reaches this quadrant.

## Why this fixes the NIAH catastrophe

On a needle-in-haystack query, the model's output entropy spikes when it cannot find the answer (uncertain across many candidate tokens). The factor `(1 − H̃(t))` shrinks the prune budget to near-zero exactly when uncertainty is high, preserving cells until the needle is located. Once entropy drops (committed), pruning resumes at full rate.

Predicted outcome: **NIAH 0/8 → ~6+/8** on Phi-3 and Llama-1B, at small PPL cost (entropy-spike rate is ~5–15% of decode steps on long-form tasks per the position-distribution study).

## What is already in hand

- **Server-side correlation evidence**: ρ = −0.37, n = 1670, p < 10⁻⁵⁵ across 7 long-form LongBench/HELM tasks (Llama-3.1-8B at 4–12K ctx). See proposal slide 22.
- **Probe instrumentation**: `entropy_probe/` standalone C++ binary, <1% overhead, links unmodified `libllama`.
- **Position-distribution study**: First-4 tokens (sinks) carry ~60% mass, recent-16 carries <1% across all entropy quintiles. Disconfirmed a competing hypothesis; lets us reallocate the recent window.
- **Paper A plumbing**: sensor sampler, watchdog tiers, llama.cpp eviction patch, disable-charging protocol — all reusable.

## What needs to be built

| Item | Effort | Risk |
|---|---|---|
| On-phone entropy probe integration (hook `llama_get_logits_ith`) | 2 days | Low (already proven on server) |
| Rolling-window normalization `H̃(t)` over 64 steps | 1 day | Low |
| Modulation of `prune_budget` and `recent_window` by `(1 − H̃(t))` | 2 days | Low |
| Re-run Wave-11 PPL + NIAH for v1_fa2_stack + v1_fa2_stack_entropy on 3 models × 7 policies | 1 week | Medium (phone time) |
| Server-side validation of phone signal preservation (ρ test) | 2 days | Medium — proposal Month 1 fallback if ρ_phone < −0.20 |
| **Tier-2 NIAH on mobile**: S-NIAH-2, S-NIAH-3, MK-NIAH-1, MQ-NIAH, VT × 3 policies × 2 models | 2 weeks | Medium (phone time) |
| **LongBench full server-side at Llama-3.1-8B**: 6 task domains × 4 policies × 3 budgets (b={10%, 20%, 40%}) for AdaKV apples-to-apples | 1 week | Low |
| **Budget sweep** b ∈ {10%, 20%, 40%, 60%} on Phi-3 + Llama-1B | 1 week | Medium (phone time) |
| **Question-aware variant of μ(m_h)** — query-conditioned attention at prefill | 3 days | Low |

**Total: ~8 weeks** assuming the entropy-attention correlation survives on-phone with 4-bit KV and thermal throttling. Fallback per the proposal: if ρ_phone < −0.20, drop the gate; the paper becomes Paper A + on-phone confirmation of the negative result, still publishable as honest discovery.

## Why this evaluation scope (vs μKV's narrower one)

The HotMobile μKV paper deliberately restricts to S-NIAH-1 + Wave-11 PPL because of the 6-page limit. Paper C is a full-conference paper where reviewers (especially anyone familiar with Feng et al. NeurIPS 2025) will expect a Ruler+LongBench-grade quality eval. Specifically:
- AdaKV's headline (question-agnostic, Llama-3.1-8B, LongBench-avg, b=20%): Ada-SnapKV **42.87** vs SnapKV **41.29** vs full-cache **49.20** (~87% retained)
- Paper C must reproduce this for our μKV+entropy variant on the same server protocol
- Without it, the "what's the contribution beyond systems engineering" attack lands
- With it, the paper shows: μKV+entropy preserves AdaKV-grade quality at higher throughput on mobile, AND fixes the NIAH retrieval gap (which AdaKV does have, but at the cost of FA-off lock-in)

## Paper outline (MobiSys 2028 form factor, ~14 pages)

```
1. Introduction
   - The workload-dependence problem: one budget, two failure modes
   - Paper A's v1_fa2_stack at K=512 wins throughput but 0/8 NIAH retrieval
   - Per-step output entropy as the missing adaptation signal
   - Contributions: combined per-head × per-step policy, fixes the catastrophe,
     no FA-off lock-in (unlike AdaKV), measured cross-model

2. Background and Motivation
   - KV eviction taxonomy: fixed K vs per-head adaptive vs proposed per-step adaptive
   - Why entropy is the right signal: position-distribution study (sinks/recent/middle)
   - Why Paper A's per-head alone fails: needle survives in no head's retained set
     (eviction count: 3944 cells / stim vs vanilla 0)

3. Design: Entropy-Gated Per-Head Eviction
   - Formal: K_h(t) = K_nom · μ(m_h) · (1 − H̃(t))
   - μ(·) per-head from Paper A
   - H̃(t) per-step output entropy, rolling 64-step normalization
   - Composition with state-swap, sink, anchor, recent window
   - Algorithm pseudocode (modify Paper A's Alg 1 line for K_i and Alg 2 trigger)

4. Implementation
   - llama.cpp probe (reuses entropy_probe/)
   - On-phone integration with Paper A's eviction_bench_v8
   - Negligible overhead (<1% from probe + O(1) modulation)

5. Evaluation
   - RQ1: NIAH retrieval — does the gate fix the 0/8 catastrophe?
     Hypothesis: vanilla 8/8, AdaKV 8/8, Paper A 0/8, Paper C 6+/8
   - RQ2: Wave-11 PPL — does the gate cost quality on creative-continuation?
   - RQ3: Decode tps — does the gate cost throughput?
   - RQ4: Cross-model (3 models × 7 policies including Paper C)
   - RQ5: Phone-vs-server signal preservation (ρ test)
   - RQ6: Per-task breakdown — where does the gate help (long-form summary)
     and where is it null (code, few-shot)
   - RQ7: Ablation — entropy gate alone (Paper B) vs combined (Paper C)

6. Related Work
   - KV eviction (per-head adaptive: AdaKV, AhaKV, KeepKV, CriticalKV)
   - Entropy signals in inference (AdaEDL, EASD, Step-Entropy — used for
     compute decisions, never for KV state)
   - Mobile thermal control (zTT, FUSE, ZeroDVFS)

7. Discussion
   - Failure modes: tasks where ρ collapses (lcc, trec, triviaqa at long ctx)
   - Fallback: thermal-and-endurance-only controller (Paper A baseline)
   - Path forward: 3-action selector (offload + quantization + eviction)

8. Conclusion
   - Per-step entropy as a model-internal control signal for KV state
   - Combined adaptation (head × time) is strictly stronger than either alone
```

## Critical-path artifacts already on disk (do not lose)

- `EndurKV/scripts/android/phone_wave11_eval.sh` — phone-side launcher
- `EndurKV/figures/master_tables/PAPER_DRAFT_MOBISYS_2028.tex` — Paper A draft
- `EndurKV/figures/master_tables/niah_tier1_6policy_aggregate.json` — full NIAH matrix
- `EndurKV/figures/master_tables/llama1b_wave11_aggregate.json` — Llama-1B PPL
- `EndurKV/figures/master_tables/energy_vs_bandwidth_correlation.pdf` — measured energy decomposition
- `entropy_probe/` directory (per proposal slide 21) — probe code with <1% overhead

## Success criteria (binary, measurable, before submission)

1. ρ_phone ≥ −0.20 (signal survives mobile deployment) — proposal Month 1 gate
2. NIAH Phi-3 ≥ 4/8 (gate measurably helps) — vs current 0/8
3. Wave-11 PPL ≤ +18% vs vanilla (gate does not destroy quality) — vs current +15.5%
4. Decode tps ≥ 4.5 tps Phi-3 long-decode (gate does not destroy throughput) — vs Paper A 4.98

If any fails, paper becomes "negative result on mobile" — still publishable, just smaller venue.

## Estimated completion: 8 weeks from start of implementation.
