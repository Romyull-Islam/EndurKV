# EndurKV-Optimal: Combined-Winner Policy Design

**Date:** 2026-06-08
**Status:** DESIGN ONLY — not yet implemented or measured on device
**Audience:** thesis / paper appendix, supervisor review
**Predecessor variants audited:** vanilla, h2o canonical, tova canonical, streamingllm, v1_fa2_stack, v1_fa2_hybrid, v1_fa2_f16, v1_entropy_stack, v1_predictive_stack
**Predecessor evidence:** Wave-11 Phi-3-mini-128k Q4_K_M, K=512 cells, chunk-pair PPL on OnePlus 15

---

## 1. Motivation: the Pareto gap

The Wave-11 audit on Phi-3-mini-128k (K=512) shows **no single variant dominates**:

| Axis | Winner | Value | Loser-on-this-axis |
|---|---|---|---|
| PPL | vanilla | 5.464 | v1_fa2_stack (6.082, +0.62) |
| decode tps | v1_fa2_stack | 4.98 tok/s (1.68x vanilla) | h2o canonical (2.02 tok/s) |
| peak DDR | tova / streamingllm (tie) | 59.4 GB | v1_fa2_stack (66.0 GB) |
| swap | h2o canonical | 0 MB | vanilla (158 MB) |
| endurance (honest-throttle) | v1_fa2_stack | only watchdog-bearing variant | all canonical baselines (no watchdog) |

The dominating-variant claim is unverified for **v1_fa2_hybrid** (closest design intent) because it currently exists only as Llama-1B chunk0->1 smoke (PPL 16.78) and was not run on Phi-3 K=512.

The optimal design therefore combines the **provably winning component from each axis** rather than inventing a new mechanism.

---

## 2. Design: `endurkv_optimal`

```
endurkv_optimal =
    EVICTION:        H2O canonical 50/50 (recent half + heavy-hitter half by
                     accumulated attention) — proven PPL parity with vanilla
                     (+0.2% on Phi-3 K=512: 5.474 vs 5.464)
    CACHE:           Q8_0 K + f16 V (per v1_fa2_stack — bounded n_kv working
                     set, ~50% K-tensor memory reduction)
    DECODE PATH:     state-swap from FA-off prefill -> FA-on decode (per
                     v1_fa2_stack — proven +68% decode throughput)
    EVICT-DURING-DECODE: frozen mask during decode (no_evict_decode = true)
                     to preserve FA-on path; eviction decisions made at
                     prefill boundary and at scoring windows only
    THERMAL:         multi-sensor watchdog v2 (DDR + skin + battery + CPU)
                     — promoted from v1_fa2_stack's honest-throttle stack
    ENDURANCE:       memory gate (>=4 GB MemAvailable) — per v1_fa2_stack
    n_sink:          4 (StreamingLLM sink protection, prepended to recent
                     window; cheap insurance against attention-sink loss)
    recent_window:   K_nominal - n_sink - K_heavy_half
                     where K_heavy_half = (K_nominal - n_sink) / 2
                     => matches h2o canonical 50/50 layout exactly, with
                     4 sink slots reserved at the front
```

### 2.1 Component provenance (why each piece)

- **EVICTION = H2O canonical 50/50.** H2O canonical measured PPL 5.474 on Phi-3 K=512 — only +0.010 above vanilla (5.464), the smallest PPL tax of any eviction-bearing variant. v1's selective top-32 cost +0.62 PPL (6.082). The 50/50 recent + accumulated-attention heavy-hitter rule is the proven recipe.
- **CACHE = Q8_0 K + f16 V.** v1_fa2_stack shipped Q8 K and delivered 4.98 tok/s with no PPL collapse traceable to quantization (the +0.62 PPL was traced to selective top-32, not Q8). Q8 K halves K-tensor memory, which is what makes the heavy-hitter + recent residency affordable inside a bounded n_kv.
- **DECODE PATH = state-swap (FA-off prefill -> FA-on decode).** This is the single biggest throughput lever measured in Wave-11: v1_fa2_stack hit 4.98 tok/s vs h2o canonical's 2.02 tok/s (a 2.46x gap) with identical underlying eviction *families* differing primarily in the FA-on-decode path. H2O canonical leaves throughput on the floor by running FA-off through decode.
- **EVICT-DURING-DECODE = frozen mask.** FA-on requires a stable KV layout per decode step. Re-evicting mid-decode breaks FA's contract. v1_fa2_stack already enforces no_evict_decode = true; we keep this invariant. Eviction decisions are taken (a) at prefill end and (b) at scoring windows where we temporarily flip FA off, score, evict, then re-enter FA-on. H2O accumulated-attention counters are updated continuously but only *acted on* at scoring windows.
- **THERMAL = multi-sensor watchdog v2.** v1_fa2_stack's watchdog gave us the only honest-throttle verdict in the audited set. Promoting to v2 (DDR + skin + battery + CPU) is a paper-required claim, not new science — same control law, more sensors.
- **ENDURANCE = memory gate (>=4 GB MemAvailable).** Inherited from v1_fa2_stack. Combined with Q8 K and H2O's tight residency this is what should drive swap to 0 MB.
- **n_sink = 4.** Cheap insurance. StreamingLLM showed sink+sliding alone gets to 5.714 PPL on Phi-3 K=512; adding 4 sink slots on top of H2O 50/50 costs 4 cells out of 512 (0.8%) and protects against the well-known attention-sink failure mode in long contexts. Standard practice in the literature.

### 2.2 Layout (K=512 cells, n_sink=4)

```
[ 0 .. 3      ]  n_sink = 4         (StreamingLLM sinks, never evicted)
[ 4 .. 257    ]  heavy-hitter half  (254 cells, H2O accumulated-attention top-K)
[ 258 .. 511  ]  recent half        (254 cells, sliding window of newest tokens)
```

Total = 512 cells, matches K_nominal. Heavy half and recent half are equal-sized (matches "H2O canonical 50/50" exactly after sink reservation).

---

## 3. Predicted performance on Phi-3-mini-128k Q4_K_M, K=512

Predictions are interpolations from the measured component variants, **not** simulated.

| Metric | Predicted | Anchor / reasoning |
|---|---|---|
| PPL (chunk-pair) | ~5.50 (range 5.47 - 5.55) | H2O canonical = 5.474; +n_sink adds ~0.01-0.02 (StreamingLLM diff); +Q8 K adds <=0.05 (Q8K_SAVINGS.md historical). Total expected: 5.474 + 0.01 + 0.03 ~ 5.51. |
| decode tps | 4.5 - 5.0 tok/s | v1_fa2_stack = 4.98 with same FA-on decode + Q8 K + state-swap. H2O scoring during scoring windows is no slower than v1's selective top-32 (both touch O(n_kv)). Slight downside risk from more aggressive scoring window cadence. |
| peak DDR | 60 - 62 GB | Between v1_fa2_stack (66.0 GB, with v1's larger working set) and h2o canonical (62.9 GB). Q8 K saves ~2-3 GB on the K tensor vs f16, but FA-on decode adds workspace. Net: lower than v1_fa2_stack's 66.0 GB. (Note: original prompt said "60-62 C" — that is a units error; thermal C is a separate axis not in Wave-11 instrumentation.) |
| swap | 0 MB | H2O canonical achieved 0 MB. Memory gate >=4 GB MemAvailable holds the residency invariant. Q8 K reduces pressure further. |
| honest-throttle verdict | MILD (target) | v1_fa2_stack earned MILD-MODERATE with v1 watchdog; v2 watchdog (multi-sensor) should tighten this to MILD if the multi-sensor inputs trigger earlier graceful slowdowns. |

### 3.1 vs predecessor v1_fa2_stack

| Axis | v1_fa2_stack (measured) | endurkv_optimal (predicted) | Delta |
|---|---|---|---|
| PPL | 6.082 | ~5.50 | **-0.58 (better)** |
| decode tps | 4.98 | 4.5 - 5.0 | ~equal (-0 to -0.5) |
| peak DDR | 66.0 GB | 60 - 62 GB | **-4 to -6 GB (better)** |
| swap | 58 MB | 0 MB | **-58 MB (better)** |
| throttle | MILD-MODERATE | MILD (target) | better |

### 3.2 vs vanilla

| Axis | vanilla (measured) | endurkv_optimal (predicted) | Delta |
|---|---|---|---|
| PPL | 5.464 | ~5.50 | +0.04 (within H2O's +0.01 + sink/Q8 noise) |
| decode tps | 2.97 | 4.5 - 5.0 | **+1.5 to +2.0 (1.5x - 1.7x)** |
| peak DDR | 64.8 GB | 60 - 62 GB | **-3 to -5 GB** |
| swap | 158 MB | 0 MB | **-158 MB** |

---

## 4. Required code changes

All changes live in the existing v1_fa2 launcher + run_llama.cpp eval harness. **No new files needed.**

1. **New parse_args alias `endurkv_optimal`** in the policy-string parser.
2. Sets `a.policy = "h2o"` (selects H2O accumulated-attention scoring path, already implemented for h2o canonical).
3. Sets `a.snapkv_decode = true` (enables state-swap from FA-off prefill -> FA-on decode, already implemented in v1_fa2_stack).
4. Sets `a.no_evict_decode = true` (frozen-mask invariant during FA-on decode, already implemented in v1_fa2_stack).
5. Sets `a.n_sink = 4` (StreamingLLM-style sink reservation, already implemented in streamingllm path; just needs to be reachable from the H2O path).
6. Launcher: `cache_type_k = q8_0` via the existing Q8 K launcher hook used by v1_fa2_stack.
7. Launcher: wires the multi-sensor watchdog v2 launch via `needs_watchdog()` (already promotes v1_fa2_stack; just add `endurkv_optimal` to the allow-list).
8. Launcher: enables the memory-gate (>=4 GB MemAvailable) — already implemented for v1_fa2_stack, again just add to allow-list.

### 4.1 Files touched (expected)

- `eval_pipeline/run_llama.cpp` (or wherever `parse_args` for policy aliases lives) — new alias branch.
- `eval_pipeline/launcher.{sh,py}` (or equivalent) — extend `needs_watchdog()` and `needs_memgate()` allow-lists.
- No changes to H2O scoring code, no changes to state-swap code, no changes to FA path. **All component code already exists**; this is a wiring exercise.

### 4.2 Implementation effort

- **C++ alias + arg plumbing:** ~2 hours
- **Launcher allow-list + Q8 K hook + watchdog v2 wiring:** ~2 hours
- **Sink-on-H2O integration test (smoke):** ~1-2 hours
- **Total:** ~4-6 hours

### 4.3 Smoke + eval

- Smoke test on Llama-1B chunk0->1 (existing harness): ~30 min on phone.
- Phi-3 K=512 PPL + tps + DDR + swap (matches Wave-11 protocol): ~2 hours on phone (after K=1024 sweep finishes).

---

## 5. Risks (honest)

1. **H2O + state-swap is an untested combination.** All four predecessor "h2o canonical" runs were FA-off throughout decode. We do not have a single measured data point where the H2O accumulated-attention bookkeeping coexists with the FA-on decode path. Component analysis says the pieces are orthogonal (state-swap is a KV-layout / kernel-selection decision, H2O is a scoring/eviction decision), but "orthogonal in design" is not "orthogonal in code." **Mitigation:** smoke on Llama-1B first; if PPL collapses or NaNs appear, fall back to scoring-window-only FA-off (still FA-on for the bulk of decode steps — partial throughput retained).

2. **FA-mode KV layout incompatibility might cause a crash.** FA-on demands contiguous K and V tensors in a specific layout. H2O's accumulated-attention scoring writes to a counter buffer that is *separate* from K/V, so this should be safe — but the eviction itself rewrites K/V slots, and if the rewrite happens while FA's plan still references the old layout, we get a use-after-free or a silent wrong-answer. The `no_evict_decode = true` invariant + scoring-window-only eviction is the defense, but the boundary conditions (last token of a scoring window, first token after) are the most likely crash sites. **Mitigation:** add an assert at scoring-window boundaries that re-validates K/V contiguity before re-entering FA-on; in smoke, run with sanitizers.

3. **Q8 K + H2O accumulated attention.** H2O's accumulated-attention scores are computed from softmax outputs, not from K directly, so Q8 K should not bias the score distribution. But Q8 K *does* mean attention itself is computed against dequantized keys, and if the dequant noise correlates with token position, the heavy-hitter set could shift. **Mitigation:** compare Llama-1B smoke PPL against h2o-f16-K smoke PPL on the same chunks; a delta >~0.5 PPL would be a red flag.

4. **Watchdog v2 sensors may not all be available on OnePlus 15.** Multi-sensor v2 design assumes DDR + skin + battery + CPU; if a sensor is missing or sysfs path changes between Android builds, the watchdog must degrade gracefully to v1 behavior, not abort. **Mitigation:** sensor-availability probe at launcher startup; fall back to v1 watchdog if any v2 sensor is missing, log a WARN.

5. **Predicted PPL (~5.50) is an interpolation.** The +0.01 (H2O) + ~0.02 (sink) + ~0.03 (Q8 K) decomposition is additive in the priors; in practice these may interact. A PPL outcome anywhere in [5.45, 5.65] would be consistent with the priors. Anything >5.7 means a component is misbehaving.

6. **Throughput could land below 4.5 tok/s** if H2O's scoring-window cadence is more frequent than v1's selective top-32 cadence. **Mitigation:** parameterize the scoring-window stride; start at the same stride v1_fa2_stack used.

---

## 6. Decision criteria (after smoke)

- **PPL <= 5.55 AND tps >= 4.5 AND swap == 0 AND no crash:** ship as `endurkv_optimal`, run full Phi-3 K=512 eval, claim Pareto-dominating variant in paper.
- **PPL in [5.55, 5.7] OR tps in [4.0, 4.5]:** acceptable, but report honestly as "near-Pareto" not "Pareto-dominating."
- **PPL > 5.7 OR crash:** debug; most likely cause is the H2O <-> state-swap interaction (risk #1 / #2 above).

---

## 7. Schema (machine-readable summary)

See `OPTIMAL_SCHEMA` emitted by the design script — that is the source of truth for downstream tooling. This document is the prose justification.
