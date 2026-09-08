# Wave-11 Evaluation Protocol — EndurKV vs other KV-cache policies

**Date:** 2026-06-07
**Goal:** Empirically evaluate the EndurKV v1_FA²-stack policy against canonical published baselines (vanilla, v1, TOVA, H2O, optionally pyramid) on two standard KV-eviction benchmarks (WikiText-2 PPL + Needle-in-a-Haystack accuracy) under matched on-device conditions.

## Key principle (user directive)
**Each policy runs in its ORIGINAL canonical configuration from its source paper.** We do NOT homogenize FA mode or eviction parameters across policies. This is more rigorous than artificially constraining each policy: each represents what its authors actually proposed.

## Ultimate policy (the headline)
`v1_FA²-stack K=512` (Wave-9 EndurKV closed-loop), as already documented in [OPTIMIZATION_JOURNEY.md](OPTIMIZATION_JOURNEY.md).

Configuration:
- `--policy v1_fa2 --k-nominal 512 --anchor-top-k 32 --recent-budget 476`
- `--cache-type-k q8_0 --cache-type-v f16`
- `--n-sink 4 --max-tokens 2048 --ignore-eos`
- Sidecar: `preempt_throttle_watchdog.sh` (DDR-driven 4-tier CPU freq cap)
- Launcher: mem-gate (MemAvailable ≥ 4 GB), cool-down to skin ≤ 33 °C, DDR ≤ 40 °C

## Two benchmarks (chosen from KV-eviction literature)

### Benchmark A: Perplexity (WikiText-2 raw)

**Dataset:** wiki.test.raw, 287K-token test set.
**Sample:** 8 non-overlapping chunks × 2048 tokens = 16,384 scored tokens per cell.
**Method:** teacher-forced via `--eval-mode ppl --eval-text` in eviction_bench (lines 1046-1127). Per chunk:
  1. Prefill 2048 tokens
  2. Apply policy's eviction
  3. Teacher-force the next 2048 tokens, accumulate per-token NLL
  4. Report `exp(mean_nll)` as PPL

**Why:** H2O (NeurIPS 2023), KVQuant (NeurIPS 2024), KIVI (ICML 2024), AdaKV (NeurIPS 2024) all use WikiText-2 PPL as their primary intrinsic eval. Matches llama.cpp's `llama-perplexity` tool exactly.

### Benchmark B: Needle-in-a-Haystack (NIAH) accuracy

**Dataset:** Greg Kamradt's standard NIAH formulation (https://github.com/gkamradt/LLMTest_NeedleInAHaystack).
**Sample:** 32 trials per cell = 4 context lengths {2k, 4k, 6k, 8k tokens} × 8 depth percentiles {0%, 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%}.
**Method:**
  - Stuff prompt with Paul Graham essays + a single needle ("The best thing to do in San Francisco is eat a sandwich at Dolores Park on a sunny day.") at the specified depth.
  - Run prefill + decode under the policy.
  - Grade by GPT-4 (OpenAI API on host, post-hoc) on whether the needle appears in the generated answer.
  - Report mean accuracy + per-depth heatmap.

**Why:** Standard for long-context KV eviction. SnapKV, AdaKV, R-KV, KVzip all use NIAH variants. Directly probes whether eviction destroys retrievable information.

## Comparison set (canonical configurations)

| Policy | Eviction signal | FA mode | K | Paper |
|---|---|---|---|---|
| **vanilla** | none (full cache) | FA-on | n/a | baseline |
| **llama.cpp stock** | none | FA-on | n/a | harness control |
| **v1** | per-step spread-gate attention top-K | **FA-off** (paper requires per-step attention) | 512 | EndurKV Wave-3 |
| **TOVA** | per-layer mean attention (last query), drop min | **FA-off** (Oren et al. ACL 2024 used standard attention) | 512 | arXiv:2401.06104 |
| **H2O** | accumulated attention + recent window | **FA-off** (Zhang et al. NeurIPS 2023 used standard attention) | 512 | arXiv:2306.14048 |
| **pyramid** | layer-wise pyramid retention | **FA-off** | 512 | arXiv:2405.12532 |
| **EndurKV v1_FA²-stack** | v1 spread-gate prefill + top-32 attention anchor + recency tier + Q8 K + watchdog | **FA-off prefill, FA-on decode** (state-swap) | 512 | this work |

## Models

| Model | Path | Role |
|---|---|---|
| Phi-3-mini-128k-Instruct Q4_K_M | `/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf` | **Primary** (headline numbers) |
| Llama-3.2-1B-Instruct Q4_K_M | `/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` | Secondary (smaller-architecture generalization) |

## Fair-comparison invariants (must hold across all cells)

1. **Same model weights** within a comparison (no quantization variation)
2. **Same WikiText-2 chunks** (chunk indices 0-7 from wiki.test.raw)
3. **Same NIAH stimuli** (same essays, same needle, same depth/length grid)
4. **Same DVFS pin attempt** (1632 MHz scaling_max via performance governor)
5. **Same mem-gate** (MemAvailable ≥ 4 GB before iter 1)
6. **Same cool-down target** (skin ≤ 33 °C, DDR ≤ 40 °C)
7. **Same threads** (4)
8. **Same ctx_size** (12288 — fits longest WikiText chunk + decode)
9. **Same seed** (1337 for the primary run; secondary seeds 2718 + 42 if time permits)

What we deliberately do NOT homogenize (per user directive):
- **FA mode** (each policy uses what its paper specifies)
- **Eviction parameters** beyond K (each policy uses canonical defaults)
- **Sink/recent window sizes** (StreamingLLM uses 4 sinks, H2O uses ~50% recent budget — keep their conventions)

## Tier-1 (Minimal Viable, ~16 hours phone time)

**Goal:** Headline PPL + NIAH numbers across 5 policies on Phi-3 at K=512.

| Cell | Policy | Phi-3 PPL probe | Phi-3 NIAH probe | Approx hours |
|---|---|---|---|---|
| W11-A | vanilla (FA-on) | 8 chunks | 32 trials | 2.5 |
| W11-B | v1 K=512 (FA-off) | 8 chunks | 32 trials | 4 |
| W11-C | TOVA K=512 (FA-off, canonical per-layer) | 8 chunks | 32 trials | 4 |
| W11-D | H2O K=512 (FA-off) | 8 chunks | 32 trials | 4 |
| W11-E | EndurKV v1_FA²-stack K=512 (FA-off→FA-on) | 8 chunks | 32 trials | 3 |
| Tier 1 total | | | | **~17.5 hours** |

Note: FA-off cells take ~50% longer than FA-on due to attention materialization overhead during prefill.

## Tier-2 (Standard, +18 hours)

Add K-sweep {256, 384, 1024} for the 5 policies on Phi-3 (skip TOVA at K=1024 since it would be no-op — TOVA paper caps at full context).

## Tier-3 (Comprehensive, +30 hours)

Add Llama-3.2-1B with the same protocol. + StreamingLLM port (1 day eng work).

## Success criteria

- All 5 Tier-1 policies produce both PPL and NIAH numbers.
- v1_FA²-stack PPL ≤ vanilla + 0.3 (within noise).
- v1_FA²-stack NIAH accuracy ≥ 0.85 × vanilla NIAH.
- v1_FA²-stack peak DDR ≤ Wave-9 measurement (64.1 °C) under both probes.
- Zero kernel-forced 883 MHz throttle events in v1_FA²-stack runs.

## Statistical tests

- 95% CI on PPL means via chunk-bootstrap.
- McNemar's test on paired NIAH accuracy (vanilla vs each policy on same trials).
- ANOVA across K-sweep PPL within each policy (Tier 2).

## Threats to validity (incorporated from adversarial review)

1. **FA-mode confound:** Documented; each policy runs in canonical mode. We report FA mode as a column in the comparison table, not as a confound.
2. **Q8 K seq_add-skip:** documented; affects only v1_FA²-stack; PPL probe is teacher-forced so sparse position layout is irrelevant for next-token logits.
3. **Phone warmth carryover:** mitigated by 33 °C cool-down + DDR ≤ 40 °C between cells.
4. **Kernel mitigation re-vote:** Watchdog re-asserts scaling_max_freq every 2 sec; kernel can still re-vote but we log and report any race.
5. **NIAH judge bias:** Use GPT-4 with chain-of-thought + 3-way consistency (judge twice with shuffled context, accept on agreement); could also use rule-based string-match as a secondary judge.

## Deliverables

1. `phone-logs/wave11_eval_<TS>/` per-cell raw data
2. `EndurKV/figures/eval_plots/`:
   - `ppl_vs_policy.png` (bar chart with 95% CI)
   - `niah_heatmap_per_policy.png` (4-row × 8-col grid per policy)
   - `niah_accuracy_summary.png` (bar chart)
   - `pareto_eval.png` (PPL × NIAH × thermal × throughput)
3. `EndurKV/figures/master_tables/TABLE_WAVE11_EVAL.md`: comprehensive comparison table
4. `eval_pipeline/score_ppl.py`: per-cell PPL aggregator
5. `eval_pipeline/score_niah.py`: GPT-4 judge wrapper for NIAH
6. `EndurKV/docs/REPRODUCIBILITY.md`: complete reproduction recipe

## Phone hours

- Tier 1: ~17.5
- Tier 2: ~35.5
- Tier 3 (with Llama-1B): ~65.5

**Recommendation:** Run Tier 1 immediately after Wave-10 K-sweep completes. Schedule Tier 2 as overnight follow-up. Tier 3 is conditional on dissertation chapter feedback.
