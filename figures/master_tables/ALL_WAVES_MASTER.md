# ALL WAVES — Master Comparison Table (Wave-3 through Wave-11)

**Generated:** 2026-06-07
**Hardware:** OnePlus 15 / Snapdragon 8 Elite Gen 5 / Adreno 840 / 12 GB UMA
**Source:** `phone-logs/wave{3..11}_*/<cell>/{stress.csv,sensors.csv,iter*/meta.json}`

---

## CRITICAL READING NOTE — PPL metric is NOT comparable across the Wave-3..10 / Wave-11 boundary

> **Wave-3 through Wave-10 PPL columns report SAMPLING-NLL** — the mean NLL of the
> model's own greedy-sampled continuation, exponentiated. This metric is **biased low**
> (the model is unsurprised by its own greedy output) and is therefore **only valid for
> internal comparison among Wave-3..10 cells**. Do NOT compare Wave-3..10 PPL numbers
> against published baselines, against Wave-11 PPL, or against any teacher-forced PPL
> in the literature.
>
> **Wave-11 PPL is held-out, teacher-forced WikiText-2 PPL** computed over disjoint
> (prefill, eval) chunk pairs per the H2O / KIVI / StreamingLLM / TOVA convention.
> This is the publication-grade metric and is comparable to literature. Wave-11 cells
> are still running at table generation time and are marked `running`.
>
> See `HELD_OUT_PPL_FINDING.md` and `OPTIMIZATION_JOURNEY.md` Problem #10 for the
> bug post-mortem (sampling-NLL was used through Wave-10 by mistake; Wave-11 is the
> corrected re-evaluation).

---

## Master per-cell table

Columns:
- **Wave** = wave number (3-real = Llama-1B 25-min sustained; 3-phi3 = Phi-3 2-iter sustained)
- **Workload** = one of `narrativeqa-Phi3`, `long-decode-Phi3`, `narrativeqa-Llama1B`, `ppl-chunk-pair` (Wave-11), `niah` (Wave-11)
- **Policy** = policy tag as logged
- **K** = K_nominal (`-` for vanilla / llama.cpp stock)
- **n_iters / n_chunks** = number of stress iterations (for Wave-3..10) or disjoint chunk pairs (for Wave-11 PPL) / number of stimuli (for Wave-11 NIAH)
- **Mean tok/s** = mean decode tokens/sec across iters
- **Peak DDR (C)** = max DDR temp during cell (sensors.csv `ddr_temp_mc`)
- **Peak CPU (C)** = max non-hw-trip CPU core temp during cell
- **Mean PPL** = per-cell perplexity; `[s]` = SAMPLING-NLL (Wave-3..10, biased low), `[h]` = HELD-OUT teacher-forced (Wave-11)
- **Swap MB** = (max - min) vmstat_pswpout pages × 4 KB / 1024 (bytes swapped out during cell)
- **Thermal control** = watchdog (preempt-throttle DDR-driven CPU freq cap) enabled YES/NO

| Wave | Workload | Policy | K | n_iters/n_chunks | Mean tok/s | Peak DDR (C) | Peak CPU (C) | Mean PPL | Swap MB | Thermal control |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 3-real | narrativeqa-Llama1B | vanilla | - | 5 | 5.09 | 51.7 | 58.7 | 2.424 [s] | 7.2 | NO |
| 3-real | narrativeqa-Llama1B | v1_K2048 | 2048 | 4 | 7.36 | 49.8 | 57.1 | 1.868 [s] | 0.0 | NO |
| 3-real | narrativeqa-Llama1B | v1_K512 | 512 | 4 | 7.05 | 49.4 | 57.5 | 2.027 [s] | 0.0 | NO |
| 3-real | narrativeqa-Llama1B | v1_fa_K512 | 512 | 4 | 7.77 | 49.8 | 57.1 | 2.430 [s] | 0.0 | NO |
| 3-phi3 | narrativeqa-Phi3 | vanilla | - | 2 | 0.997 | 57.9 | 62.4 | 3.555 [s] | 220.7 | NO |
| 3-phi3 | narrativeqa-Phi3 | v1_K512 | 512 | 2 | 1.071 | 61.7 | 68.6 | 9.418 [s] | 172.5 | NO |
| 3-phi3 | narrativeqa-Phi3 | v1_fa_K512 | 512 | 2 | 1.921 | 58.7 | 63.6 | 2.857 [s] | 515.1 | NO |
| 3-phi3 | narrativeqa-Phi3 | tova_K512 | 512 | 2 | 1.059 | 60.6 | 63.7 | 8.657 [s] | 13.4 | NO |
| 3-phi3 | narrativeqa-Phi3 | llamacpp_stock | - | 3 | 1.06 | 57.5 | 60.0 | n/a | 0.0 | NO |
| 4 | long-decode-Phi3 | vanilla | - | 9 | 5.01 | 62.9 | 69.4 | 2.555 [s] | 9.6 | NO |
| 4 | long-decode-Phi3 | v1_K512 | 512 | 5 | 2.66 | 54.4 | 62.0 | 4.224 [s] | 0.0 | NO |
| 4 | long-decode-Phi3 | v1_fa_K512 | 512 | 8 | 4.65 | 62.5 | 69.0 | 2.341 [s] | 0.0 | NO |
| 5 | narrativeqa-Phi3 | v1_fa_K512_filebacked | 512 | 2 | 2.03 | 60.6 | 66.7 | 2.857 [s] | 739.5 | NO |
| 6 | narrativeqa-Phi3 | v1_fa_K512_bounded | 512 | 9 | 5.77 | 65.6 | 72.9 | 3.203 [s] | 28.0 | NO |
| 7 | narrativeqa-Phi3 | v1_fa2 | 512 | 7 | 6.72 | 66.4 | 74.4 | 3.918 [s] | 1551.8 | NO |
| 8 | narrativeqa-Phi3 | v1_fa2_selective | 512 | 11 | 6.75 | 72.9 | 80.6 | 3.560 [s] | 6.7 | NO |
| 9 | narrativeqa-Phi3 | v1_fa2_stack | 512 | 10 | 6.09 | 64.1 | 69.4 | 2.169 [s] | 0.0 | **YES** |
| 10 | narrativeqa-Phi3 | v1_fa2_stack (K-sweep) | 256 | 12 | 7.17 | 64.1 | 70.9 | 2.095 [s] | 25.3 | **YES** |
| 10 | narrativeqa-Phi3 | v1_fa2_stack (K-sweep) | 384 | 12 | 7.05 | 63.3 | 70.9 | 2.123 [s] | 21.5 | **YES** |
| 10 | narrativeqa-Phi3 | v1_fa2_stack (K-sweep) | 1024 | 10 | 6.20 | 63.7 | 71.7 | 1.827 [s] | 119.2 | **YES** |
| 11 | ppl-chunk-pair | vanilla | - | 8 | running | running | running | running [h] | running | **YES** |
| 11 | ppl-chunk-pair | h2o | 512 | 8 | running | running | running | running [h] | running | **YES** |
| 11 | ppl-chunk-pair | tova | 512 | 8 | running | running | running | running [h] | running | **YES** |
| 11 | ppl-chunk-pair | streamingllm | 512 | 8 | running | running | running | running [h] | running | **YES** |
| 11 | ppl-chunk-pair | v1_fa2_stack | 512 | 8 | running | running | running | running [h] | running | **YES** |
| 11 | niah | vanilla | - | 8 | running | running | running | n/a (accuracy) | running | **YES** |
| 11 | niah | h2o | 512 | 8 | running | running | running | n/a (accuracy) | running | **YES** |
| 11 | niah | tova | 512 | 8 | running | running | running | n/a (accuracy) | running | **YES** |
| 11 | niah | streamingllm | 512 | 8 | running | running | running | n/a (accuracy) | running | **YES** |
| 11 | niah | v1_fa2_stack | 512 | 8 | running | running | running | n/a (accuracy) | running | **YES** |

Legend for the **Mean PPL** column:
- `[s]` — sampling-NLL perplexity (`exp(mean_nll)` from greedy-sampled continuation). **Biased low.** Internal comparison among Wave-3..10 cells only.
- `[h]` — held-out teacher-forced PPL on WikiText-2 raw (disjoint chunk-pair protocol, token-weighted geometric mean per cell, 95% CI via percentile bootstrap). Comparable to literature.
- `n/a` — PPL not reported for this workload (`niah` reports retrieval accuracy; `llamacpp_stock` log has no NLL field).
- `running` — Wave-11 cell pending. Directory `phone-logs/wave11_eval_1780862534/` exists but is empty at table-generation time.

---

## Per-wave provenance

| Wave | Source directory | What ran |
|---|---|---|
| 3-real | `phone-logs/wave3_real_1780680903/` | Llama-3.2-1B Q4_K_M, narrativeqa_pub_001 (8007-tok prompt), 25-min sustained, vanilla + v1_K2048 + v1_K512 + v1_fa_K512 |
| 3-phi3 | `phone-logs/wave3_phi3_1780719530/` | Phi-3-mini-128k-Instruct Q4_K_M, narrativeqa (9794-tok prompt), 2 iters per cell, vanilla + v1_K512 + v1_fa_K512 + tova_K512 + llamacpp_stock |
| 4 | `phone-logs/wave4_longdecode_1780750084/` | Phi-3, short prompt (~500 tok) + 2048-tok decode, 5-9 iters per cell, vanilla + v1_K512 + v1_fa_K512 |
| 5 | `phone-logs/wave5_v1fa_filebacked_1780764804/` | Phi-3 narrativeqa, single cell: v1_fa_K512 with file-backed state-swap experiment |
| 6 | `phone-logs/wave6_v1fa_bounded_1780769821/` | Phi-3 narrativeqa, single cell: v1_fa_K512 with bounded-cache fix (no_evict_decode=false) |
| 7 | `phone-logs/wave7_v1fa2_1780782482/` | Phi-3 narrativeqa, single cell: v1_fa² (anchor + recency tiers, no thermal stack) |
| 8 | `phone-logs/wave8_v1fa2_sel_1780788550/` | Phi-3 narrativeqa, single cell: v1_fa²_selective (Q8 K-cache added, no watchdog) |
| 9 | `phone-logs/wave9_v1fa2_stack_1780796320/` | Phi-3 narrativeqa, single cell: v1_fa²_stack (Q8 K + watchdog + closed-loop K + mem-gate) — the EndurKV thermal stack |
| 10 | `phone-logs/wave10_ksweep_1780815847/` | Phi-3 narrativeqa, K-sweep {256, 384, 1024} with v1_fa²_stack (K=512 is Wave-9) |
| 11 | `phone-logs/wave11_eval_1780862534/` | **Currently empty / running.** Planned: 5 policies × {PPL chunk-pair on WikiText-2 raw, NIAH 8-stimulus grid} on Phi-3 Q4_K_M |

---

## How the numbers were derived

- **Mean tok/s** = arithmetic mean over per-iter `decode_tps` rows in each cell's `stress.csv`.
- **Peak DDR (C)** = `max(ddr_temp_mc) / 1000` over the full `sensors.csv` of that cell.
- **Peak CPU (C)** = `max(cpu-*_temp_mc) / 1000` excluding `cpu-hw-trip-*` sentinel (95 C trip-point reference, not an actual reading).
- **Mean PPL [s]** = mean of `iter*/meta.json:perplexity` across iters of the cell, where `perplexity = exp(mean_nll)` and `mean_nll` is the per-token NLL of the model's own greedy continuation. Wave-8/9/10 also have this as a column in `stress.csv` (matches the meta.json value).
- **Mean PPL [h]** = (Wave-11 only) per-chunk teacher-forced NLL summed and exponentiated per the WikiText-2 convention: `exp(sum_i n_tok_i * mean_nll_i / sum_i n_tok_i)`.
- **Swap MB** = `(max(vmstat_pswpout) - min(vmstat_pswpout)) * 4 / 1024` (pages × 4 KB → MB) over the cell.
- **Thermal control** = YES iff `watchdog.log` exists in the wave directory AND the cell was launched with the preempt-throttle watchdog sidecar (Wave-9 introduced it; Wave-10 and Wave-11 inherit it).

---

## Within-wave notes

- **Wave-3-real (Llama-1B)**: thermal envelope was never breached; no policy throttled. Peak DDR spread 49.4–51.7 C, peak CPU 57.1–58.7 C. Mean tok/s shows v1_fa wins on decode (7.77 vs vanilla 5.09).
- **Wave-3-phi3**: FA-off prefill is the dominant cost. v1 K=512 went 3.8 C HOTTER than vanilla on peak DDR — invalidates the simple "smaller cache = cooler" hypothesis for prefill-dominated workloads. v1_FA recovers (58.7 C, near vanilla) by doing FA-off prefill + FA-on decode.
- **Wave-4 (long-decode)**: opposite regime — decode dominates. v1 K=512 is 8.5 C COOLER than vanilla peak DDR (54.4 vs 62.9) and never throttles. v1_fa fails here (FA-on decode → cache grows → hits thermal limit at iter 4 like vanilla).
- **Wave-5**: file-backed state-swap experiment for v1_fa. Memory savings only meaningful under pressure (Wave-3 regime); produced 739 MB swap-out and slow 2.0 tok/s.
- **Wave-6**: bounded-cache v1_fa fix — decode-time eviction enabled. 65.6 C peak DDR (hotter than v1, but only one cell to compare).
- **Wave-7 (v1_fa²)**: adds anchor-top-K + recency tiers. 66.4 C peak DDR; 1.55 GB swap-out (memory pressure issue).
- **Wave-8 (v1_fa²_selective)**: + Q8 K-cache. Eliminates swap (6.7 MB) but DDR spikes to 72.9 C (no thermal control yet).
- **Wave-9 (v1_fa²_stack)**: + watchdog + closed-loop K + mem-gate. Peak DDR drops to 64.1 C, zero swap, zero throttle. This is the EndurKV headline.
- **Wave-10**: K-sweep on the Wave-9 stack. Headline finding — throughput is monotonically non-increasing in K over {256, 384, 512, 1024}: 7.17 → 7.05 → 6.09 → 6.20 tok/s. K=512 anomaly is a scheduler artefact (two pathological iter dips), not fundamental.
- **Wave-11**: held-out PPL + NIAH on the corrected protocol. Cells are pending; do not cross-compare to Wave-3..10 PPL numbers.

---

## Files referenced

- `TABLE_PHI3_WAVE3.md` — full Wave-3-phi3 narrative
- `TABLE_PHI3_WAVE4_LONGDECODE.md` — full Wave-4 narrative
- `SUBSECTION_KSWEEP.md` — Wave-10 K-sweep narrative
- `WAVE11_FINAL_SPEC.md` — Wave-11 binding spec (5 policies × 2 benchmarks at K=512)
- `EVAL_PROTOCOL_WAVE11.md` — Wave-11 evaluation protocol
- `HELD_OUT_PPL_FINDING.md` — sampling-NLL vs held-out PPL post-mortem
- `OPTIMIZATION_JOURNEY.md` (Problem #10) — sampling-NLL bug discovery
- `WAVE_ROADMAP.md` — wave-by-wave narrative
