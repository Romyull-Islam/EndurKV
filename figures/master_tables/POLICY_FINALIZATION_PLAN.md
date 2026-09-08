# Policy Finalization Plan — Wave-11 Staged Strategy

**Created:** 2026-06-08
**Status:** ACTIVE — K=128 sweep CANCELED in favor of staged plan
**Rationale:** A K-sweep across 5 candidate novel policies is wasteful when 4 will lose.
Finalize the policy FIRST, then K-sweep only the winner.

---

## PHASE A — Wait for K=1024 sweep to finish (~25h from now)

The Wave-11 K=1024 sweep is currently running on the OnePlus 15 and produces clean
data for the 5 published baselines:

- `vanilla` (full-cache FA-on reference)
- `h2o` (canonical 50/50 budget/recent)
- `tova` (top-K attention)
- `streamingllm` (sink + window)
- `v1_fa2_stack` (our current best published variant)

across **Phi-3-mini, Llama-3.2-1B, Gemma-2-2B**. Combined with the existing K=512
data, this gives us **two K points (512, 1024) for 5 baselines across 3 models** —
already paper-grade evidence for the baseline section of the chapter.

**Deliverables at end of Phase A:** 15 K=1024 cells, thermal traces, peak DDR,
swap, PPL, decode tps. Logs in `figures/master_tables/K1024_LIVE_THERMAL.md`.

---

## PHASE B — Smoke-test ALL novel candidates on one model (~10h)

Once the phone is free, push the new `eviction_bench_v4` binary (which contains
the `endurkv_optimal` policy plus the four alternates) to the device. Run a
**single-model chunk-pair smoke test on Phi-3 at K=512, 5 chunks each** for all 5
candidate variants:

1. `endurkv_optimal` — H2O 50/50 + state-swap to FA-on + Q8 K-cache + multi-sensor v2 watchdog
2. `v1_fa2_hybrid` — H2O 50/50 + watchdog + f16 K (no state-swap)
3. `v1_fa2_f16` — v1 stack + state-swap + watchdog + f16 K (no Q8)
4. `v1_entropy_stack` — entropy-weighted eviction + Q8 + state-swap + watchdog
5. `v1_predictive_stack` — trajectory-prediction eviction + Q8 + state-swap + watchdog

Approx. budget: **5 cells x ~2h each = ~10h**.

---

## PHASE C — Decision: Pareto-composite pick (immediate after Phase B)

We score each surviving candidate across four axes, ranked in importance:

1. **PPL gate (HARD)** — `policy_PPL <= 1.05 x vanilla_PPL`. Any policy that
   exceeds 5% PPL inflation versus vanilla on Phi-3 is **disqualified**, no
   exceptions.
2. **Decode throughput** — `decode_tps >= 4.0` tok/s (matches the v1_fa2_stack
   reference of 4.98 tok/s). Below this is disqualified.
3. **Peak DDR temperature** — `peak_DDR <= 63 C` (within the canonical H2O
   thermal envelope). Above this is disqualified.
4. **Swap pressure** — `swap_MB == 0`. Any non-zero swap is disqualified.

Among the candidates that clear all four gates, the **tiebreaker** is the
composite product:

    score = total_wall_latency_seconds * peak_DDR_celsius

Lower is better. This jointly rewards fast and cool — the two axes that matter
most for on-device deployment. The candidate with the smallest score is the
chosen finalist policy.

---

## PHASE D — Full K-sweep on the chosen policy ONLY (~60h)

Sweep `K in {128, 256, 512, 1024, 2048}` for the winning policy across all 3
models = **15 cells**. For fairness we also re-run `v1_fa2_stack` at the same K
values as an ablation/control = **15 more cells**. Total: **30 cells x ~2h = ~60h**.

This gives a clean K-sensitivity curve for both the new finalist and the prior
published baseline, on identical chunks/seeds/eval splits.

---

## PHASE E — Final paper writeup

At end of Phase D we have:

- K-sensitivity curve (5 K points) for finalist policy + v1_fa2_stack control
- Multi-policy comparison across 5 baselines + finalist at K=512 and K=1024
- Thermal traces and peak DDR per (model, policy, K)
- Endurance data (24h soak from Wave-9 + chunk-pair Wave-11)
- Held-out PPL on WikiText-103 + PG-19
- Optional: NIAH ablation if any wall-clock budget remains

Produces: comprehensive results table, Pareto plots (PPL vs decode tps, peak DDR
vs decode tps), K-sensitivity plot, and the discussion section.

---

## TOTAL TIMELINE FROM NOW

| Milestone | ETA from 2026-06-08 |
|---|---|
| K=1024 sweep finishes (Phase A) | +25h |
| Phase B smoke-test complete | +35h |
| Phase C decision | immediate |
| Phase D K-sweep complete | +95h |
| **End-to-end (Phase E ready)** | **~4 days from now** |

---

## DECISION CRITERIA (canonical form)

```
GATE 1 (PPL):    policy_PPL <= 1.05 * vanilla_PPL_on_same_model     [hard]
GATE 2 (TPS):    decode_tps  >= 4.0 tok/s                            [hard]
GATE 3 (THERM):  peak_DDR    <= 63 C                                 [hard]
GATE 4 (SWAP):   swap_MB     == 0                                    [hard]

TIEBREAKER:      minimize  total_wall_latency_s * peak_DDR_C
```

Gate 1 is the FIRST gate and is absolute: any policy that inflates PPL above 5%
of vanilla is disqualified regardless of how fast or cool it runs. Among
survivors, the latency * temperature product picks the single finalist that is
both fastest and coolest — the two operational axes that matter on-device.

---

## CANCELED WORK

- K=128 host watcher (`/tmp/k128_watcher.pid`, `/tmp/watch_k1024_then_fire_k128.sh`) — REMOVED.
- Phone-side wrapper (`/data/local/tmp/endurkv/scripts/wave11_K128_wrapper.sh`) — REMOVED.
- Rationale: K=128 across all 5 candidates is premature; do the K-sweep only on
  the Phase-C winner to avoid burning ~50h on policies that lose Gate 1 anyway.
