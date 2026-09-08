# EndurKV-Evict — Phone Deployment Design & Progress Log

**Last updated:** 2026-05-29 (continuous-log; append-only as work proceeds)
**Author:** Md Romyull Islam (Kennesaw State University, PhD CS)
**Dissertation:** "Thermal and Endurance Co-Aware KV Management for Sustained Mobile LLM Inference Driven by Model-Internal Signals"

---

## 1. Goal

Deploy our KV-cache eviction policies on **OnePlus 15 (Snapdragon)** and measure **latency, perplexity, accuracy, memory, thermal, energy** end-to-end on the phone. Beat all existing per-head eviction papers (AdaKV, HeadKV, DuoAttention, PyramidKV, SnapKV, H2O, StreamingLLM) on apples-to-apples phone comparisons.

**Primary benchmark:** LongBench (long-context QA tasks).
**Secondary:** NIAH, reasoning (R1-distill GSM8K), short context.

---

## 2. Policies to compare (priority order)

1. **EndurKV-Evict v1 (ours)** — linear-clipped per-head gate, α=1.3, β=0.6, on max-attention signal. *Final formula in §6.*
2. **Vanilla llama.cpp** — un-patched llama.cpp with FIFO `ctx-shift` on overflow. The default behavior users get out of the box.
3. **TOVA** (Oren EMNLP'24) — fixed K per-head, top-K by current attention.
4. **PyramidKV** (Zhang 2024) — linear K_max→K_min across depth, TOVA selection within layer.
5. **AdaKV** (Feng 2024) — entropy-proportional per-kv-head budget.
6. **H2O** (Zhang NeurIPS'23) — K/2 heavy hitters by cumulative attention + K/2 recent.
7. **EndurKV-Evict v1-adaptive (ours)** — layer-sharpness-gated v1↔TOVA blend (only deploy if reasoning sim confirms benefit; currently running on host).

---

## 3. GQA strategy (decision)

**llama.cpp stores K/V as `[n_kv, n_head_kv, head_dim]` per layer.** For GQA models (Llama, Mistral, Qwen2, Gemma-2), multiple query heads share one kv-head.

| Model | n_query | n_kv | GQA ratio |
|-------|---------|------|-----------|
| Llama-3.2-1B | 32 | 8 | 4:1 |
| Llama-3.1-8B | 32 | 8 | 4:1 |
| Mistral-7B | 32 | 8 | 4:1 |
| Qwen2-7B | 28 | 4 | 7:1 |
| Gemma-2-2B | 8 | 4 | 2:1 |
| Phi-3-mini | 32 | 32 | MHA |
| R1-Distill-8B | 32 | 8 | 4:1 |

**Chosen approach: Aggregate-OR at kv-head level.**
- Compute per-(query-head) decision via our v1 gate (as in the offline sim).
- For each (layer, kv_head, position): **keep position if ANY query head sharing this kv-head wants it.**
- This is the SOTA-standard handling (used in AdaKV, HeadKV).
- Slightly more conservative than the per-query-head sim (keeps a few more positions) → real-phone memory savings will be a bit smaller than sim suggests.
- **Reviewers expect this distinction.** Report both numbers: "simulated upper bound" vs "realistic GQA-aware".

**Rationale for NOT choosing mask-only (B):** mask-only doesn't free memory — only saves attention computation. The mobile inference story needs REAL memory reduction.

---

## 4. Status / progress so far

### Offline simulation (host PC) — DONE

- 168 cells (7 models × 5 datasets × 2 K budgets) tested
- v1 wins 144/168 (86%), median Δ TOVA = −11.3%, cache× = 1.04
- v1-adaptive currently running (background, ~30 min remaining at time of writing)
- All offline results saved at: `EndurKV/figures/unified_all_models_results.csv` (1008 rows)

### Build infrastructure — DONE

- llama.cpp built for ARM aarch64 Android: `EndurKV/llama.cpp/build-android/bin/`
  - `llama-completion` (formerly llama-cli, stripped to 2.7 MB)
  - `llama-perplexity` (2.7 MB)
  - `llama-bench` (717 KB)
  - `libllama.so` (3.0 MB) + `libggml*.so`
- NDK location: `/home/mislam22/tools/ndk/android-ndk-r27c`
- Build script: `EndurKV/scripts/android/build_llama_android.sh`

### Phone benchmark scripts — DONE (vanilla baseline)

- `EndurKV/scripts/android/phone_bench_vanilla.sh` — runs llama-completion on a prompt, captures latency + memory + thermal
- `EndurKV/scripts/android/phone_bench_perplexity.sh` — runs llama-perplexity on a corpus
- `EndurKV/scripts/android/push_to_phone.sh` — host-side adb push helper
- `EndurKV/scripts/android/sample_sensors.sh` — existing 10 Hz thermal+battery sampler

### v1 C++ patch — NOT YET STARTED (this is the next major work)

- Integration point identified: `llama-kv-cache.cpp::seq_rm()` (line 330) is the eviction primitive
- Attention capture: existing `cb_eval` callback (used by entropy_probe) gives us per-step attention
- New CLI arg needed: `--kv-policy {vanilla,tova,pyramidkv,adakv,h2o,v1,v1-adaptive} --k-nominal N`

### v1-adaptive sim — RUNNING

- Single-threaded version: `EndurKV/scripts/android/host_v1_adaptive_v2.py`
- Process PID 11044, started ~5 min before this doc was written
- Output: `EndurKV/figures/v1_adaptive_v2.log` (progress) + `v1_adaptive_results.csv` (when done)
- Decides whether v1-adaptive deserves to be in the phone benchmark suite

---

## 5. Implementation plan (3-week realistic timeline)

| Week | Tasks |
|------|-------|
| **Week 1 (engineering)** | Write v1 C++ patch (GQA-aware aggregate-OR) · Integrate cb_eval hook for attention · Add CLI flag for policy selection · Build patched llama.cpp for Android · Smoke test on host first |
| **Week 2 (TOVA + PyramidKV + scripts)** | Implement TOVA + PyramidKV as same-pattern policy plugins · Build new Android binaries · Extend `phone_bench_vanilla.sh` → `phone_bench_with_policy.sh --policy v1/tova/pyramidkv/vanilla` · Bundle LongBench prompts for phone (extract from JSONL → .txt files) · Bundle WikiText-2 for perplexity |
| **Week 3 (run + analyze)** | Push everything to phone · Run benchmark sweep (5 models × 3 policies × 10 LongBench tasks × 2 K budgets × 3 repeats = ~900 runs · ~12 hr wall time with thermal cooldowns) · Pull logs, aggregate, plot |
| **Stretch** | AdaKV + H2O integration if time permits · v1-adaptive deployment (if sim confirms it wins reasoning) · Pivot A: thermal-coupled α/β (the real dissertation novelty) |

---

## 6. The v1 algorithm — final formal spec

```
INPUT:  attention values a[s, l, q_head, position] from step s, layer l
        K_nominal  (per-head cache budget)
        n_head_kv  (model's KV head count; for GQA aggregation)

FOR each decode step s:
    FOR each layer l in 0..n_layers-1:

        // (1) per-query-head gate
        FOR each query head q in 0..n_query_heads-1:
            max_a[q]   = max over positions of a[s, l, q, :]
            norm[q]    = clip( (max_a[q] - 0.4) / 0.4 , 0, 1 )
            mult[q]    = 1.3 - 0.6 * norm[q]                    // multiplier ∈ [0.7, 1.3]
            K_h[q]     = round( K_nominal * mult[q] )
            keep_q[q]  = top-K_h[q] indices by a[s, l, q, :]    // per-query-head set

        // (2) GQA aggregate-OR: union over query heads sharing each kv-head
        FOR each kv_head kv in 0..n_head_kv-1:
            sharing_qs = { q : q % n_head_kv == kv }   // standard GQA mapping
            keep_kv[kv] = ⋃ keep_q[q] for q in sharing_qs

        // (3) evict positions NOT in keep_kv[kv] from kv-cache layer l
        FOR each kv_head kv:
            evict_set = { 0..n_kv-1 } - keep_kv[kv]
            CALL llama_kv_cache::seq_rm(seq_id=0, p0=p, p1=p+1) for each p in evict_set
            // (or batch as a contiguous range when possible)
```

Constants: α=1.3, β=0.6, thresh_low=0.4, thresh_high=0.8.
Same code shape works for MHA (n_head_kv == n_query_heads → no aggregation needed).

---

## 7. Phone-side measurement plan

For each (model, policy, K_nominal, task) combination, capture:

| Metric | Captured from | File |
|--------|---------------|------|
| Prefill latency (ms) | `llama-completion` stderr `t_p_eval_ms` | `<run>/<prompt_id>.timing.txt` |
| Decode tok/s | `llama-completion` stderr `t_eval_ms` / n_predict | same |
| Per-token latency p50/p95 | parse timing log per step | derived |
| Perplexity | `llama-perplexity` stdout | `<run>/<tag>.ppl.txt` |
| Generated text (for accuracy F1/EM/ROUGE) | `llama-completion` stdout | `<run>/<prompt_id>.gen.txt` |
| Peak RSS | `/proc/$PID/status` at 5 Hz | `<run>/<prompt_id>.mem.csv` |
| KV cache MB | llama.cpp's `llama_get_kv_cache_used_cells` * bytes/cell | derived from meta |
| Thermal (junction temp) | `sample_sensors.sh` 10 Hz | `<run>/<prompt_id>.sensors.csv` |
| Battery V, mAh, charging state | same sampler | same |
| Throttle state | `/sys/class/thermal/cooling_device*/cur_state` (in sensors.csv) | same |

**Energy mWh/1K tokens:** integrate (battery_voltage × current_mA × dt) over decode duration, normalize.

---

## 8. Datasets to bundle on phone (LongBench first)

Per the user's directive: **LongBench is the primary benchmark.**

### LongBench tasks (16 total in benchmark)
- Single-doc QA: `narrativeqa`, `qasper`, `multifieldqa_en`, `multifieldqa_zh`
- Multi-doc QA: `hotpotqa`, `2wikimqa`, `musique`
- Summarization: `gov_report`, `qmsum`, `multi_news`
- Few-shot: `trec`, `triviaqa`, `samsum`
- Synthetic: `passage_count`, `passage_retrieval_en`
- Code: `lcc`, `repobench-p`

For phone deployment: pick **5 representative tasks** to start (matching what we already have prompts for):
- narrativeqa (single-doc QA)
- qasper (single-doc QA scientific)
- hotpotqa (multi-doc QA)
- gov_report (summarization)
- multifieldqa_en (mixed)

5 tasks × 5 prompts each = 25 prompts per (model, policy, K) cell.

Stored: `EndurKV_workspace/prompts/prompts_pub_longbench.jsonl` (420 prompts).
For phone: extract per-task subset to `phone-deploy-stage/prompts/<task>_<id>.txt`.

---

## 9. Models to bundle on phone

Priority order (start with #1, expand):

1. **Llama-3.2-1B** — small, fast iteration (~1 GB on phone). Q4_K_M quantized.
2. **Llama-3.1-8B** (R1-Distill base) — primary mobile target (~5 GB).
3. **Mistral-7B** — strong GQA-4:1 model (~4.4 GB).
4. **Qwen2-7B** — different GQA ratio (7:1) for stress-test (~4.7 GB).
5. Gemma-2-2B, Phi-3-mini — secondary.

Total phone storage needed if all 5: ~16 GB. Phone has typically 256+ GB, so fine.

---

## 10. Where artifacts live (resume guide)

| Artifact | Location |
|----------|----------|
| **This document** | `EndurKV/docs/DEPLOYMENT_PLAN.md` |
| Host simulator | `EndurKV/scripts/android/host_simulate_kv_baselines.py` |
| Unified offline results | `EndurKV/figures/unified_all_models_results.csv` |
| v1-adaptive results (when done) | `EndurKV/figures/v1_adaptive_results.csv` |
| Android build script | `EndurKV/scripts/android/build_llama_android.sh` |
| Phone benchmark scripts | `EndurKV/scripts/android/phone_bench_*.sh` |
| Push script | `EndurKV/scripts/android/push_to_phone.sh` |
| Phone-deploy staging (with Android binaries) | `phone-deploy/bin/` (symlinks to build-android/) |
| **v1 patch (when written)** | `EndurKV/llama.cpp/src/llama-kv-evict-policy.{h,cpp}` (TBD) |
| **Patched llama-completion (when built)** | `EndurKV/llama.cpp/build-android-v1/bin/llama-completion` (TBD) |
| Slide-ready figures | `EndurKV/figures/architecture_fig/` |

---

## 11. Decision log (append-only)

- **2026-05-29:** Chose Aggregate-OR for GQA handling (option A). Faithful to SOTA pattern, real memory savings, reviewer-expected.
- **2026-05-29:** Selected LongBench as primary deployment benchmark. NIAH/reasoning are secondary.
- **2026-05-29:** Decided v1_linear (α=1.3, β=0.6) is the headline policy. v1-sigmoid dropped (overfit to long-context only, lost on short/llama-long/reasoning).
- **2026-05-29:** Decided to report BOTH simulated upper-bound AND realistic GQA-aware deployment numbers in paper, to be transparent about the per-head-to-per-kv-head conversion.
- **2026-05-29:** Vanilla llama.cpp baseline = un-patched, FIFO ctx-shift on overflow (this IS the implicit eviction in real-world usage).
- **2026-05-29:** v1-adaptive sim completed (1786s, 168 cells). Result: v1_adaptive does **not** beat v1_linear anywhere. On reasoning specifically (the regime it was designed to help), it's +37.7% worse than TOVA and +35% worse than v1_linear. The layer-sharpness threshold tuning was wrong — it doesn't collapse to TOVA early enough at low sharpness. **DROPPED from deployment.** v1_linear stays as the only ours-policy.
- **2026-05-29:** Final deployment policy list (priority order): (1) v1_linear, (2) vanilla llama.cpp, (3) TOVA, (4) PyramidKV. Optional add-ons if time: AdaKV, H2O.
- **2026-05-30:** Four pillars of the paper claim, in priority order:
  1. **Maximum memory savings** (via KV eviction)
  2. **Reduce thermal throttling** (dissertation novelty)
  3. **Preserve quality** (within 5% of vanilla)
  4. **Latency: reduce OR match vanilla** (the connection: cold-state may trail; **sustained** must win due to less throttling)
  The dissertation's novel claim hinges on (2)+(4): v1 trails vanilla in cold burst tok/s but **beats vanilla in sustained tok/s** under thermal pressure. This is why eviction matters specifically for mobile.
- **2026-05-30:** A/B #1 (1B, K=512) confirmed eviction policies all degenerate at 15.6× compression. A/B #2 (8B, K=1024) confirmed eviction adds latency overhead (+25% prefill, +28% decode) but model + repetition penalty fix needed.
- **2026-05-30:** Added to eviction_bench.cpp: --n-sink (StreamingLLM sink-token protection, default 4), --repeat-penalty (default 1.1), --no-fa-vanilla (vanilla FA-on by default for realistic baseline), --no-evict-decode (SnapKV-style frozen mask). Rebuilding for Android.
- **2026-05-30:** Experimental design split into 4 sub-experiments aligned with 4 pillars:
  - (M) Memory — long-context capability test at fixed ctx_size
  - (T) Thermal — sustained 15-min sessions, no cool-down
  - (Q) Quality — 25 prompts × cool-down × multi-replicate
  - (P) Perplexity — WikiText-2 corpus PPL per (model, policy, K)

---

## 12. Risks & open questions

| Risk | Mitigation |
|------|------------|
| v1 C++ patch is harder than budgeted (touches concurrent decode + memory mgmt) | Phase the work: first MHA-only, then GQA aggregation, then optimize |
| Phone thermal throttling makes runs non-reproducible | Cool to 35°C between runs, use phone-side fan if possible, repeat 3× |
| Eviction creates cache fragmentation → real RSS unchanged | Periodic compaction; report both "logical cache" and "physical RSS" |
| llama.cpp upstream changes break our patch | Pin to current commit; maintain rebase notes |
| v1-adaptive doesn't help reasoning | Drop from deployment, paper claims v1 only |

---

## 13. Next concrete action (when resuming)

1. **Wait for v1-adaptive sim** to finish (~30 min). Confirm it beats v1 on reasoning. If yes, add to deployment policy list. If no, drop.
2. **Start writing the v1 C++ patch** in `EndurKV/llama.cpp/src/llama-kv-evict-policy.h` + `.cpp`. Reference design in §6 above. Aggregate-OR for GQA.
3. **Build patched llama.cpp for Android.** Place output in `build-android-v1/`.
4. **Extend `phone_bench_vanilla.sh` → `phone_bench_with_policy.sh`** to take `--policy` and `--k-nominal` args.
5. **Bundle LongBench prompts** (5 tasks × 5 prompts each = 25 .txt files).
6. **First on-phone run:** Llama-3.2-1B + v1 policy + narrativeqa task + K=512.

---
