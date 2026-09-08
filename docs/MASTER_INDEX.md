# EndurKV — Master Index & Decision Log

**Single entry point for the entire project.** Read this first; it points to every artifact, decision, and result. Append-only log of why we did what.

---

## 1. The thesis in 3 lines

> Per-head adaptive KV-cache eviction policy (`EndurKV-Evict`) driven by
> max-attention concentration, deployed on Snapdragon 8 Elite Gen 5 phone.
> Targets: max memory savings · minimum thermal throttling · quality ≥ vanilla
> · latency ≥ vanilla (cold) or ≥ best published (sustained).

The full proposal is in [`EndurKV/slides/proposal_progress.md`](../slides/proposal_progress.md).

---

## 2. The headline policy formula (v1 / `EndurKV-Evict`)

```
For each layer ℓ, head h, decode step:
  max_a[h] = max over positions of current attention from head h
  norm[h]  = clip( (max_a[h] − 0.4) / 0.4 , 0, 1 )
  μ[h]     = 1.3 − 0.6 · norm[h]                         # multiplier ∈ [0.7, 1.3]
  K_h      = round( K_nominal · μ[h] )                   # per-head budget
  kept[h]  = top-K_h indices by current attention a[h, :]
```

**Why these constants?** See §6 (sweep results).

---

## 3. Four pillars (paper-claim framework)

| Pillar | Metric | Threshold |
|--------|--------|-----------|
| 1. Max memory savings (effective KV) | mass_retained ÷ retention_ratio | efficiency > 5× uniform |
| 2. Reduce thermal throttling | peak_skin_C · throttle_level · sustained tok/s | lower onset, higher sustained tok/s than vanilla |
| 3. Preserve quality | F1, PPL, needle-recall | within 5% of vanilla |
| 4. Latency: ≥ vanilla cold OR best baseline sustained | decode_tps cold + sustained | match or beat |

Full mapping in [`DEPLOYMENT_PLAN.md`](DEPLOYMENT_PLAN.md) §11 decision log.

---

## 4. Documentation map

### Master state docs

| Doc | What's in it |
|-----|--------------|
| [`PROJECT_STATE.md`](../../PROJECT_STATE.md) (top of workspace) | Mission · formula · dual-paper plan · milestone (rank-1 in 31/31 cells, offline) |
| [`EndurKV/docs/DEPLOYMENT_PLAN.md`](DEPLOYMENT_PLAN.md) | This session's phone deployment design · GQA strategy · 4 pillars · decision log |
| [`EndurKV/ENV.md`](../ENV.md) | Build environment, NDK, dependencies |

### Earlier (offline simulation) results

| Doc | What's in it |
|-----|--------------|
| [`logs/EXPERIMENTS.md`](../../logs/EXPERIMENTS.md) | Per-experiment numbers from earlier offline sweeps |
| [`logs/STRATEGY.md`](../../logs/STRATEGY.md) | SOTA positioning per axis |
| [`logs/MOBILE_KV_LANDSCAPE.md`](../../logs/MOBILE_KV_LANDSCAPE.md) | Every published baseline we compared against |
| [`logs/COMPARISON.md`](../../logs/COMPARISON.md) | Cross-policy comparison notes |
| [`logs/MODEL_COVERAGE_MATRIX.md`](../../logs/MODEL_COVERAGE_MATRIX.md) | 7 models × 5 datasets coverage grid |
| [`figures/final_evaluation_tables.md`](../figures/final_evaluation_tables.md) | 22-system results table — v1 wins 144/168 cells (86%) |
| [`figures/comprehensive_results.md`](../figures/comprehensive_results.md) | Per-(model, K, policy) LongBench numbers |

### Slide-ready figures (in `figures/architecture_fig/`)

| Figure | What it shows |
|--------|----------------|
| `perhead_v1_clean.png` | v1's 5-stage architecture diagram |
| `perhead_v1_architecture.png` | older detailed architecture |
| `perhead_v1_vs_tova_comparison.png` | side-by-side with TOVA |
| `design_space_4axis.png` | 4-axis design space (Signal × Selection × Budget × Inter-head) |
| `pareto_cache_vs_kl_all.png` | Pareto frontier across 168 cells (offline) |
| `win_rate_per_model.png` | v1 wins per-model breakdown |
| `win_rate_per_dataset.png` | v1 wins per-dataset breakdown |
| `median_delta_tova_grid.png` | heatmap of median Δ KL per (model, policy) |

---

## 5. Code map

### Offline simulator (host, Python)

| Script | Purpose |
|--------|---------|
| `host_simulate_kv_baselines.py` | Core per-head simulator. Loads .attn.bin + .kv.bin, applies any policy, computes KL/mass/retention. |
| `host_simulate_eviction_perhead.py` | Per-head policy implementations (TOVA, AdaKV, HeadKV, DuoAttention) |
| `host_simulate_eviction_policies_v2.py` | Global policies (H2O, SnapKV, Scissorhands, LWKD, AhaKV, LazyEviction) |
| `host_unified_baselines.py` | Cross-policy comparison runner (14 policies × 5 models × 2 K) |
| `host_v1_adaptive.py` / `host_v1_adaptive_v2.py` | Layer-adaptive blend (v1↔TOVA via L_sharp) |
| `host_signal_selection_search.py` | Token-ranking axis sweep |
| `host_gate_search_mp.py` | 69-config gate-shape × (α, β) sweep (multiprocess) |

### Hyperparameter sweeps (saved results)

| Sweep | Results CSV | Config |
|-------|-------------|--------|
| **v1 5×5 (α, β)** | `figures/sweep/sweep_quick_results.csv` (1000 rows) | α ∈ {1.1..1.5}, β ∈ {0.4..0.8}; **(1.3, 0.6) chosen** (see §6) |
| **v2 hyperparam** | `figures/v2_sweep/v2_sweep_results.csv` | participation-resonance gate ablation (v2 dropped) |
| **Gate-shape (69 configs)** | `figures/gate_search_mp_results.csv` (690 rows) | linear-clipped · sigmoid · quadratic · inv-quadratic · step · per-α/β |
| **Signal+selection** | `figures/signal_selection_ranking.csv` | 28 (signal × selection) configs at fixed best gate; **negative result** — TOVA's signal/selection is optimal |

### Phone deployment (the new work — this session)

| File | Purpose |
|------|---------|
| `entropy_probe/eviction_bench.cpp` | On-phone benchmark binary: implements all 4 policies (vanilla, v1, tova, pyramid) with sink-token, repetition penalty, FA-on-vanilla, effective-KV metrics |
| `entropy_probe/attention_probe.cpp` | Earlier attention capture probe (used for offline KV traces) |
| `scripts/android/build_llama_android.sh` | Build llama.cpp + tools for Android arm64 (CPU only) |
| `scripts/android/build_llama_android_vulkan.sh` | Same with GGML_VULKAN=ON (Adreno 840 GPU) |
| `scripts/android/build_probe_android.sh` | Build entropy_probe + eviction_bench for Android |
| `scripts/android/phone_bench_vanilla.sh` | Single-prompt vanilla baseline runner |
| `scripts/android/phone_bench_perplexity.sh` | llama-perplexity on a corpus |
| `scripts/android/phone_run_longbench_sweep.sh` | Master multi-prompt sweep |
| `scripts/android/phone_sweep_cooled.sh` | Sweep with cool-down + replicates |
| `scripts/android/phone_cool_then_run.sh` | Cool-and-measure wrapper |
| `scripts/android/phone_queue_after_ab3.sh` | Queued runner: WT2-PPL → A/B #4 after A/B #3 |
| `scripts/android/sample_sensors.sh` | 10 Hz thermal + battery + freq + cool-state sampler |
| `scripts/android/push_to_phone.sh` | Host-side adb push helper |
| `scripts/android/host_aggregate_full.py` | Joins meta.json + steps.csv + sensors.csv → single table |
| `scripts/android/host_score_phone_runs.py` | F1 / EM / ROUGE-L / needle-recall scorer (vs ground-truth from JSONL) |

### Compiled artifacts

| Artifact | Location |
|----------|----------|
| Host CPU llama.cpp | `llama.cpp/build/bin/` (x86_64 Linux) |
| Android CPU llama.cpp | `llama.cpp/build-android/bin/` (aarch64) |
| Android Vulkan llama.cpp | `llama.cpp/build-android-vulkan/bin/` (aarch64 + Adreno) |
| Host eviction_bench | `entropy_probe/build-host/eviction_bench` |
| Android eviction_bench | `entropy_probe/build-android/eviction_bench` (66 KB stripped) |

---

## 6. Decision timeline + parameter justifications

### 2026-05-22 — initial captures

- Captured `.attn.bin` files for 5 LongBench-active paper models (Phi-3, Mistral-7B, Qwen2-7B, Gemma-2-2B, R1-Distill-8B) + Llama-1B + Llama-8B on the OnePlus 15 phone.
- These captures are at `logs/study_phone_*_longbench/`, `_niah/`, `_reasoning/`.
- Capture mechanism: `attention_probe` binary running on phone with `cb_eval` hook collecting per-layer `kq_soft_max` tensors to disk.

### 2026-05-23 → 2026-05-24 — first offline sweep

- Built the offline simulator (`host_simulate_kv_baselines.py`) that replays attention captures through any policy and computes KL / mass / retention.
- **5×5 (α, β) sweep on v1 spread gate**: tested α ∈ {1.1, 1.2, 1.3, 1.4, 1.5} × β ∈ {0.4, 0.5, 0.6, 0.7, 0.8}.
  - Result CSV: `figures/sweep/sweep_quick_results.csv`
  - **(α=1.3, β=0.6) chosen** — Pareto-optimal at the "moderate cache overshoot, large quality win" point (~15% better KL than TOVA at 1.15× cache). Alternative (1.2, 0.8) wins at strict-K but underspends; (1.5, 0.4) wins at max-quality but uses 40% extra cache. (1.3, 0.6) is the headline.
- Initial validation: 31 cells, 7 architectures, 48 policies, 106 (cell, K) pairs. v1 ranked 1st in **all 31 cells**.

### 2026-05-25 → 2026-05-26 — expansion sweeps

- **69-config gate-shape sweep** (`host_gate_search_mp.py`): linear-clipped, sigmoid, quadratic, inv-quadratic, step. Output `figures/gate_search_mp_results.csv`.
  - Best: sigmoid (α=1.5, β=0.5, c=0.4, γ=8.0) → −12.7% KL at 1.06× cache. But Method-A vs Method-B cache-adj reconciliation showed it doesn't beat v1 strictly.
- **Signal+selection sweep** (`host_signal_selection_search.py`): 28 (signal × selection) configs.
  - **Negative result**: TOVA's `current attention` + `argmax` is already optimal — no exotic signal (heat-diffusion, FFT, hybrid, cum_max) beats it.
- **v2..v7 ablations** (participation-resonance, FFT, Newton-Raphson, Rényi, Leidenfrost, logistic): all LOSE to v1 except v6 (logistic) which ties.

### 2026-05-27 → 2026-05-28 — comprehensive eval

- Wikitext-2 corpus PPL via `llama-perplexity`.
- **`unified_all_models_results.csv`** (1008 rows): 7 models × 5 datasets × 6 policies × 2 K.
  - v1 wins **144 / 168 cells (86%)** on raw KL.
  - Median Δ TOVA = −11.3%.
  - Cache× = 1.04 (essentially cache-neutral on Llama).
  - Beats AdaKV, PyramidKV, TOVA, KVzip-decode, H2O, SnapKV, StreamingLLM, etc.

### 2026-05-29 — 4 pillars + phone deployment plan

- Locked the 4-pillar framework (memory · thermal · quality · latency).
- Sub-experiments defined: (M) memory long-context · (T) thermal sustained · (Q) 25-prompt quality sweep · (P) WikiText-2 corpus PPL.
- Wrote [`DEPLOYMENT_PLAN.md`](DEPLOYMENT_PLAN.md).
- Built CPU `llama.cpp` + `eviction_bench` for Android arm64.
- Pushed binaries + 25 LongBench prompts + WikiText-2 to phone (`/data/local/tmp/endurkv/`).

### 2026-05-30 — phone A/Bs + Vulkan

- **A/B #1**: Llama-3.2-1B, K=512, FA-off all. All eviction policies degenerated (15.6× compression too aggressive for 1B model). Vanilla generated coherent text.
- **A/B #2**: Llama-3.1-8B, K=1024, FA-off all. v1 and TOVA both produced repetitive output ("The lying scribe..."). Diagnosed root cause: greedy decoding + no sink-token protection + no rep-penalty.
- **Added to `eviction_bench.cpp`**:
  - `--n-sink 4` (StreamingLLM-style sink-token protection)
  - `--repeat-penalty 1.1` (llama.cpp standard)
  - `--no-fa-vanilla` (vanilla gets FA-on by default for production-realistic baseline)
  - `--no-evict-decode` (SnapKV-style frozen-mask option)
  - `mean_mass_retained`, `mean_retention_ratio`, `mean_eviction_efficiency` (your "max hit rate / min retention" metric)
- **A/B #3 (running now)**: same config as #2 but with all fixes. v1 + vanilla done, TOVA mid-run, pyramid pending.
  - v1's effective-KV: **mean_mass_retained = 98.6% at retention_ratio = 12.2%** → **efficiency = 8.12×**.
  - Vanilla generated correct answer: "The lying scribe. (line 620)..."
- **Vulkan llama.cpp built** for Adreno 840 (Snapdragon 8 Elite Gen 5):
  - `build-android-vulkan/bin/` has `llama-completion`, `llama-perplexity`, `llama-bench` + `libggml-vulkan.so` (30 MB stripped).
  - Standard llama.cpp practice: `GGML_VULKAN=ON` + `-ngl N` runtime flag.
- **Queued runner** (`phone_queue_after_ab3.sh`, PID 22691): will fire WikiText-2 PPL + A/B #4 (K=2048) as soon as A/B #3 ends.
- **K=2048 = 25% retention** chosen because that matches the published-baseline standard (H2O / TOVA / SnapKV all benchmark at 20-50%). K=512 / K=1024 were stress tests (12-15% retention, too aggressive).

---

## 7. On-phone current state (as of 2026-05-30 evening)

| Resource | Status |
|----------|--------|
| Path on phone | `/data/local/tmp/endurkv/` |
| Binary in use | `bin/eviction_bench` (CPU, with all fixes — 66 KB) |
| Libs | `bin/libllama.so` + `libggml*.so` (CPU-only) |
| Models pushed | Llama-3.2-1B (770M) · Llama-3.1-8B (4.5G) · Mistral-7B (4.0G) · Phi-3-mini (2.2G) · Qwen2-7B (4.3G) · Gemma-2-2B (1.5G) · R1-Distill-8B (4.5G) — total ~22 GB |
| Prompts | 25 LongBench .txt files at `prompts/longbench/` |
| Corpus | `corpora/wiki.test.raw` (1.2 MB) for PPL |
| Sensors sampler | `scripts/sample_sensors.sh` (10 Hz, 130+ thermal/battery/freq cols) |
| Cool-down wrapper | `scripts/phone_cool_then_run.sh` |
| Logs dir | `logs/` (each run goes to `ab_*` or `ab4_*` etc.) |

Stage in pipeline (current run):
- **A/B #3** still running TOVA → pyramid (K=1024) — saved to `logs/ab4_*`
- **Queued**: WikiText-2 PPL → `logs/wt2_ppl/`
- **Queued**: A/B #4 (K=2048) → `logs/ab5_*`
- **Ready to deploy after queue**: Vulkan llama.cpp for GPU acceleration test

---

## 8. Reproducibility checklist

Everything to reproduce one experiment end-to-end:

1. Read [`DEPLOYMENT_PLAN.md`](DEPLOYMENT_PLAN.md) §10 artifact map
2. Build host llama.cpp: `cd llama.cpp && cmake -S . -B build && cmake --build build`
3. Build Android llama.cpp: `bash scripts/android/build_llama_android.sh`
4. Build eviction_bench: `bash scripts/android/build_probe_android.sh`
5. Strip binaries with NDK llvm-strip
6. Push via `adb push` (or via the deploy script)
7. Run: `adb shell sh /data/local/tmp/endurkv/scripts/phone_sweep_cooled.sh --model models/X.gguf`
8. Pull results: `adb pull /data/local/tmp/endurkv/logs ./phone-logs/`
9. Aggregate: `python EndurKV/scripts/android/host_aggregate_full.py --in-dir phone-logs/`
10. Score: `python EndurKV/scripts/android/host_score_phone_runs.py --runs-dir phone-logs/`

---

## 9. Outstanding work (priority order)

| Task | Why | Owner |
|------|-----|-------|
| **Wait for A/B #3 to finish** | TOVA + Pyramid still pending | phone |
| **WikiText-2 PPL** (queued) | Paper-grade PPL benchmark | phone |
| **A/B #4 at K=2048** (queued) | Fair-K comparison vs published baselines | phone |
| **Compare CPU vs Vulkan** | Validate GPU speedup before committing to big sweep | phone |
| **Sub-exp T: thermal sustained** | The dissertation's headline | phone (long run) |
| **Sub-exp M: long-context capability** | Real memory savings demo | phone |
| **Sub-exp Q: 25-prompt quality sweep** | F1/PPL/needle-recall paper table | phone (long) |
| **Sub-exp P: WikiText-2 per-policy** | Need eviction_bench in PPL mode (small C++ change) | host code |
| **Final 5-figure paper layout** | After all runs complete | host analysis |

---

**Last updated:** 2026-05-30 20:25
**Append next entry below.**
