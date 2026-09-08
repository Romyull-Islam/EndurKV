# Llama-3.2-1B-Instruct (Q4_K_M) — Comprehensive Metrics Table

**Device:** OnePlus 15 (Snapdragon 8 Elite Gen-5, 16 GB LPDDR5X), CPU-only inference
**Model:** `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, ctx=12 288, n_layers=16, n_kv_heads=8, head_dim=64
**System data source:** `phone-logs/wave3_real_1780680903/{vanilla, v1_K512, v1_K2048, v1_fa_K512}/`
**Held-out PPL source:** smoke run `2026-06-07` (Llama-1B chunk-pair sliding-window held-out)

---

## Metric-type legend

| Tag | Meaning |
|---|---|
| **sampling-NLL** | Per-token NLL on the *sampled* (model-generated) decode tokens during the Wave-3 narrativeqa_pub_001 prompt. Deterministic-greedy across iters, so PPL is identical iter-to-iter. Lower is better but it is **NOT** a held-out language-modeling PPL; it measures the model's confidence in its own greedy continuation. |
| **held-out** | True next-token PPL on a sliding chunk-pair held-out evaluation (smoke run, 2026-06-07). This is the comparable LM PPL used for cross-policy quality claims. |

> The Wave-3 sampling-NLL is reported for completeness because it is the only PPL emitted by the on-device log; **all quality conclusions must be drawn from the held-out column**.

---

## Comprehensive table

| Policy | K | Workload | PPL (metric type) | Peak DDR | Peak CPU | Swap MB | Min mem_avail GB | Peak RSS GB | Prefill ms | Decode tps | TTFT s | Total wall s | Throttled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla (FA-vanilla) | — (full) | narrativeqa_pub_001, n_prompt=8007, n_decode=22, 5 iters | **2.42** (sampling-NLL, Wave-3) / **16.28** (held-out, smoke 06-07) | 51.7 °C | 70.2 °C | 7.2 | 7.69 | 1.182 | 324 956 (mean; 275 405 – 367 216) | 5.088 (mean; 4.571 – 5.597) | 324.96 (mean) | 329.3 (mean), 1 646.5 (sum-5) | No (cool_state=0) |
| streamingllm (v1) | 512 | narrativeqa_pub_001, n_prompt=8007, n_decode=3 (early-EOS), 4 iters | **2.03** (sampling-NLL, Wave-3) / **17.12** (held-out, smoke 06-07) | 49.4 °C | 59.6 °C | 0.0 | 7.59 | 1.290 | 418 254 (mean; 386 273 – 463 323) | 7.051 (mean; 6.627 – 7.745) | 418.25 (mean) | 418.7 (mean), 1 674.8 (sum-4) | No (cool_state=0) |
| streamingllm (v1) | 2048 | narrativeqa_pub_001, n_prompt=8007, n_decode=3 (early-EOS), 4 iters | **1.87** (sampling-NLL, Wave-3) / n/a (not in smoke 06-07) | 49.8 °C | 59.2 °C | 0.0 | 7.60 | 1.290 | 428 593 (mean; 388 872 – 463 837) | 7.364 (mean; 6.787 – 7.750) | 428.59 (mean) | 429.0 (mean), 1 716.1 (sum-4) | No (cool_state=0) |
| v1_fa2_stack (no_evict_decode) | 512 | narrativeqa_pub_001, n_prompt=8007, n_decode=22, 4 iters | **2.43** (sampling-NLL, Wave-3) / **11.44** (held-out, smoke 06-07) | 49.8 °C | 59.2 °C | 0.0 | 7.42 | 1.290 | 408 582 (mean; 386 401 – 433 038) | 7.774 (mean; 6.958 – 8.340) | 408.58 (mean) | 412.2 (mean), 1 649.0 (sum-4) | No (cool_state=0) |

### Notes on each column

- **PPL (metric type)** — **two values per row**. The Wave-3 sampling-NLL is identical across the 4–5 iters of each policy because decoding is greedy and the prompt is fixed; we report the single deterministic value. The smoke-06-07 held-out PPL is the apples-to-apples LM-quality number; **v1_fa2_stack at 11.44 is the headline quality improvement** vs vanilla 16.28 and streamingllm 17.12. H2O / TOVA from the same smoke run were **invalid pre-fix** (canonical-fix landed after 06-07) and are omitted here.
- **Peak DDR / Peak CPU** — Max across the run window from `sensors.csv`. CPU peak excludes the `cpu-hw-trip-*` sentinel columns (constant 95 000 m°C = trip-point threshold, not a reading); we take the max over real per-core `cpu-*_temp_mc` and `cpullc-*_temp_mc` columns.
- **Swap MB** — Δ of `vmstat_pswpout` (pages-out, 4 KB pages) across the run. Only **vanilla** spilled to swap (~7 MB); the v1 family stayed entirely resident. Counterpart `vmstat_pswpin` Δ was ≤ 1.1 MB for all rows (effectively zero).
- **Min mem_avail GB** — Min of `mem_avail_kb` across the run. All policies kept ≥ 7.4 GB free on a 16 GB device → **memory headroom was never the bottleneck**.
- **Peak RSS GB** — Max of `peak_rss_kb` from per-iter `meta.json`. Vanilla is ~108 MB lower than the v1 family; the extra is the v1 selector / index footprint.
- **Prefill ms** — Per-iter `prefill_ms` from `meta.json`; we report mean and (min – max) range across iters because thermal soak grows prefill ~25 % from iter 1 → iter 5.
- **Decode tps** — Per-iter `decode_tps` (sampled tokens / decode wall-s). v1 family is 38–53 % faster decode than vanilla because of the smaller live-KV window.
- **TTFT s** — Time-to-first-token. For a CPU-only llama.cpp run with no streaming-prefill, TTFT ≈ `prefill_ms` (first sampled token is emitted after the prefill batch completes). Mean over iters.
- **Total wall s** — Per-iter `total_ms` (prefill + decode). We report mean and the 4- or 5-iter sum (total stress-loop duration).
- **Throttled** — `True` iff any `cpu*_cool_state > 0` in `sensors.csv`. **No policy throttled** in this run — peak CPU stayed below 71 °C and DDR below 52 °C, well under the 95 °C trip-point.

---

## Cross-policy take-aways (Llama-1B, Wave-3)

1. **v1_fa2_stack is the headline win**: same quality regime as vanilla on Wave-3 sampling-NLL (2.43 vs 2.42) but **−4.84 PPL on held-out** (11.44 vs 16.28), with **+53 % decode tps** (7.77 vs 5.09) and **zero swap** vs vanilla's 7.2 MB spill.
2. **streamingllm trades a small held-out-PPL regression** (17.12 vs 16.28) for similar speed gains — v1_fa2_stack dominates it on quality at matched K=512.
3. **Sampling-NLL is misleading**: at K=2048 streamingllm shows the *lowest* sampling-NLL (1.87) because its early-EOS path is more confident on its own short greedy continuation; this is not a quality win, which is why we anchor all claims on the held-out column.
4. **Thermal headroom**: no throttling at 16 GB / SD8 Elite Gen-5. The thermal story is more interesting on Phi-3 (Wave-11) and not on Llama-1B.

---

## Provenance

- Wave-3 raw: `phone-logs/wave3_real_1780680903/{vanilla,v1_K512,v1_K2048,v1_fa_K512}/{stress.csv, sensors.csv, iter*/meta.json}`
- Wave-3 summary: `phone-logs/wave3_real_1780680903/WAVE3_REAL_SUMMARY.md`
- Smoke held-out: 2026-06-07 Llama-1B chunk-pair smoke (vanilla 16.28, streamingllm 17.12, v1_fa2_stack 11.44); H2O / TOVA invalid pre-fix
- Build script: this file generated by tool-driven aggregation 2026-06-08

