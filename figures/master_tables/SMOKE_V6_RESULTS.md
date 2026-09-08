# SMOKE_V6_RESULTS

Smoke benchmark using `eviction_bench_v6` against Llama-3.2-1B-Instruct-Q4_K_M on
wiki.test.raw.chunk1 (PPL eval) with chunk0 as prompt. Settings: ctx=4096, K=512,
n_sink=4, seed=42, threads=4, n-gpu-layers=0, n-batch=512, ubatch=64, greedy,
max-tokens=32, ignore-eos. Vanilla reference PPL = 16.28; GATE-1 threshold = 17.10.

Composite score = 0.5 * normalized PPL Δ + 0.5 * normalized peak_kv_cells
(both normalized 0..1, lower is better). Note: all 6 policies report
peak_kv_cells = 1894 (full prompt length stays resident pre-eviction-tick),
so the composite is effectively driven by the PPL Δ term alone.

Ranked by composite (best first):

| Rank | Policy              | Smoke PPL | Δ vs vanilla 16.28 | GATE-1 (≤17.10) | evicted_prefill | peak_kv_cells | Notes |
|------|---------------------|-----------|--------------------|-----------------|-----------------|---------------|-------|
| 1    | v1_fa2_hybrid       | 16.7817   | +0.5017            | PASS            | 0               | 1894          | H2O canonical 50/50 inside v1_FA² thermal stack; no state-swap, f16/f16; decode-time eviction (evicted_total_decode=976,899). Best PPL of the cohort. |
| 2    | v1_predictive_stack | 17.2849   | +1.0049            | FAIL            | 737             | 1894          | Temporal-trajectory (slope) scoring; FA-off decode + per-step eviction (evicted_total_decode=1,558,946). Close to gate but misses. |
| 3    | endurkv_optimal     | 17.7318   | +1.4518            | FAIL            | 1536            | 1894          | Reference / pre-existing smoke. Canonical H2O eviction + state-swap → FA-on decode + q8_0 K / f16 V. |
| 4    | v1_fa2_f16          | 18.2888   | +2.0088            | FAIL            | 2171            | 1894          | v1_FA² spread-gate + selective anchoring (top-32) + state-swap, f16/f16 K cache. Isolates "what does dropping q8_0 K cost?" |
| 5    | v1_entropy_stack    | 18.3773   | +2.0973            | FAIL            | 2492            | 1894          | Entropy-weighted attention scoring + thermal stack (q8_0 K / f16 V). Worst tier — entropy scoring underperforms tiered anchoring. |
| 5    | v1_fa2_stack        | 18.3773   | +2.0973            | FAIL            | 2170            | 1894          | v1_FA² tiered (top-32 anchor, recent_budget=476) + state-swap + q8_0 K / f16 V. Binary CLI alias: passed as `--policy v1_fa2` (the canonical name; `v1_fa2_stack` is rejected). Ties v1_entropy_stack on PPL. |
| 7    | endurkv_adaptive    | 640.6107  | +624.3307          | FAIL            | 2176            | 1894          | v1 per-head spread-gate + AdaptiveKController (DDR/CPU/skin/battery → μ_thermal); f16 K + f16 V, NO state-swap, selective anchoring top-32. Binary maps `endurkv_adaptive`→`v1_adaptive`. No K_effective/μ_thermal log lines emitted by the controller in this smoke (only the standard `policy=v1_adaptive K=512 n_sink=4 FA=OFF` line). Catastrophic PPL — eviction policy is destroying eval-target context; well outside GATE-1 and worse than every other Wave-11 cohort policy by ~620 PPL. |

## Key observations

- Only `v1_fa2_hybrid` (canonical H2O eviction wrapped with the v1_FA² thermal
  watchdog/memory gate, no state-swap, f16/f16) passes GATE-1.
- The four q8_0-K stacks (`endurkv_optimal`, `v1_fa2_stack`, `v1_entropy_stack`,
  `v1_predictive_stack`) and the f16/f16 v1_FA² variant (`v1_fa2_f16`) all fail
  GATE-1 by 0.18–1.28 PPL points.
- `v1_predictive_stack` is the closest of the failing policies — within 0.19
  PPL of the gate.
- `v1_fa2_stack` and `v1_entropy_stack` produce identical PPL (18.377280) and
  near-identical evicted_prefill (2170 vs 2492). The selective anchoring +
  tiered eviction in `v1_fa2` is no better than entropy-weighted scoring at
  this prompt / K=512.
- Every policy reaches the same peak_kv_cells = 1894 (full prompt length),
  meaning all eviction happens after prefill ingestion — the working-set
  memory footprint at peak is identical across the cohort. Differentiation
  comes from PPL quality, not memory savings at peak.

## CLI deviation noted

The task specification listed a policy literal `v1_fa2_stack`, but the
`eviction_bench_v6` binary does not accept that token (`bad policy:
v1_fa2_stack`). Per `entropy_probe/eviction_bench.cpp:173-178`, the accepted
v1_FA² aliases are `v1_fa2`, `v1_fa2_hybrid`, `v1_fa2_f16`. The original `v1_fa2`
already implements the "stack" configuration (state-swap, tiered decode-time
eviction, selective anchoring), so the smoke for `v1_fa2_stack` was run with
`--policy v1_fa2` and the requested `--cache-type-k q8_0 --cache-type-v f16
--anchor-top-k 32 --recent-budget 476` flags. Output files use the
`v1_fa2_stack_*` naming the task requested; `meta.json.policy` reads `v1_fa2`.

## Source files

- `/tmp/smoke_v6_local/smoke_v6/endurkv_optimal_meta.json`
- `/tmp/smoke_v6_local/smoke_v6/v1_fa2_hybrid_meta.json`
- `/tmp/smoke_v6_local/smoke_v6/v1_fa2_f16_meta.json`
- `/tmp/smoke_v6_local/smoke_v6/v1_entropy_stack_meta.json`
- `/tmp/smoke_v6_local/smoke_v6/v1_predictive_stack_meta.json`
- `/tmp/smoke_v6_local/smoke_v6/v1_fa2_stack_meta.json`
- `/sdcard/smoke_v7/endurkv_adaptive_meta.json` (Wave-11 endurkv_adaptive smoke, eviction_bench_v7, 2026-06-09)
