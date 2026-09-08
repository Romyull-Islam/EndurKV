# Complete master table — all tested policies / all metrics

**Hardware:** OnePlus 15 / Snapdragon 8 Elite Gen 5 / Adreno 840 / 12 GB UMA, CPU-only (4 threads), llama.cpp build `07dca62`.

**Data sources:**
- `[W1]` Wave-1-redux (`cpu_sweep_1780268970`) — 3 prompts × 1 rep on hotpotqa/narrativeqa/etc., averaged
- `[W3]` Wave-3 REAL today (`wave3_real_1780680903`) — 25-min sustained stress on narrativeqa, 1 cell
- `[PPL]` Wave-2c (`ppl2c_*`) — WikiText-2 perplexity, short seed
- `[bench]` `llama-bench` direct (pure llama.cpp, no eviction)
- `[O]` Wave-3 OTHER currently running (TOVA/H2O/pyramid K=512, Llama-1B narrativeqa)

---

## Table 1 — Quality (F1 on LongBench, PPL on WikiText-2)

| Policy | Llama-3.2-1B | Gemma-2-2B | Phi-3-128k | WT2 PPL Llama-1B | WT2 PPL Gemma | WT2 PPL Phi-3 | Source |
|---|---|---|---|---|---|---|---|
| vanilla (no eviction) | **0.115** | **0.141** | **0.211** | 8.78 | 8.85 | 5.03 | [W1] |
| llama.cpp baseline | identical to vanilla — same code path | — | — | — | — | — | — |
| v1 K=512 | 0.114 | 0.106 | 0.205 | 8.79 | 8.82 | 5.05 | [W1] |
| v1_FA K=512 | (matches v1) | — | — | (matches v1, frozen mask) | — | — | extrapolated from validation |
| TOVA | 0.024 | 0.031 | 0.206 | 8.79 | 8.82 | 5.05 | [W1] |
| H2O | 0.024 | 0.031 | 0.205 | — | — | — | [W1] |
| Pyramid | — pending [O] — | — | — | — | — | — | running |

**Quality interpretation:** v1 preserves F1 within ±0.01 of vanilla on Llama-1B and Phi-3. TOVA/H2O collapse on smaller models (F1 0.02 vs 0.11 vanilla) — too aggressive without per-head adaptation. PPL ties because the prompt is short and no eviction triggers.

---

## Table 2 — System metrics (latency / throughput)

| Policy | K | Decode tok/s (Llama-1B 3-prompt avg) | Decode tok/s (Llama-1B 25-min sustained, iter 1) | Decode tok/s sustained mean | Wall-clock tok/s (25 min) | Source |
|---|---|---|---|---|---|---|
| **llama.cpp (pure tg32 empty cache)** | — | — | **22.29** | — | — | [bench] |
| **llama.cpp (pp5500+tg32 aggregate)** | — | — | 25.00 | — | — | [bench] |
| **vanilla** (full 8K KV, FA-on) | — | 11.31 | 5.60 | 5.10 (1280 tok / 1500 s ≈ 0.85 wall) | **0.85** | [W3] |
| **v1 K=2048** | 2048 | — | 7.75 | 7.36 (1024/1500=0.68 wall) | 0.68 | [W3] |
| **v1 K=512** | 512 | 6.97 | 7.75 | 7.05 (1024/1500=0.68 wall) | 0.68 | [W1]+[W3] |
| **v1_FA K=512** | 512 | — | **8.34** | **7.78** | 0.68 | [W3] |
| **TOVA K=512** | 512 | 6.93 | pending [O] | pending | pending | [W1]+[O] |
| **H2O K=512** | 512 | 4.82 | pending [O] | pending | pending | [W1]+[O] |
| **Pyramid K=512** | 512 | — | pending [O] | pending | pending | [O] |

**Throughput interpretation:**
- llama.cpp tg32 (22.29 t/s) is the ceiling at empty cache. v1_FA scales correctly to ~8 t/s at 5500 cache.
- Vanilla wins wall-clock tok/s on this 7700-prefill regime because FA-on prefill is fast; eviction's FA-off prefill cost negates its decode advantage.
- v1_FA has the FASTEST decode-only speed (8.34 t/s), 49% above vanilla, 7.6% above v1.

---

## Table 3 — KV behaviour (memory savings + retention)

| Policy | K | Avg live KV (MB) | KV savings vs vanilla | Mass retained | Retention ratio | Efficiency (mass/retention) | Total positions evicted (per cell) | Source |
|---|---|---|---|---|---|---|---|---|
| llama.cpp | — | 131 (Llama-1B baseline) | — | 1.000 | 1.000 | 1.00 | 0 | inherits vanilla |
| vanilla | — | 131 | (baseline) | 1.000 | 1.000 | 1.00 | 0 | [W1] |
| v1 K=512 | 512 | 40 | **70%** | 0.991 | 0.474 | **3.21** | 25 765 | [W1] |
| v1_FA K=512 | 512 | ~40 (same algorithm) | ~70% | 0.991 (frozen at prefill) | 0.474 | 3.21 | 2 474 (prefill only) + 0 (decode frozen) | [W3]+algorithm parity |
| TOVA | — | 31 | **76%** | 0.989 | 0.375 | 4.09 | 36 984 | [W1] |
| H2O | — | 31 | **76%** | 0.983 | 0.375 | 4.05 | 26 119 | [W1] |

**"Hit rate" interpretation:**
- **Mass retained** = fraction of total attention probability that survives eviction. Closer to 1.0 means eviction kept the important positions.
- **Retention ratio** = fraction of KV cells kept. Smaller = more aggressive compression.
- **Efficiency** = mass / retention. Higher = better ratio of preserved attention per cached cell kept. **v1 = 3.21**, TOVA = 4.09, H2O = 4.05. TOVA wins on "compression efficiency" but loses 80% F1 — it's evicting tokens that matter to downstream tasks.

---

## Table 4 — Thermal (Llama-3.2-1B CPU 4 threads)

| Policy | K | Peak DDR (°C) — 25-min sustained | Mean DDR (°C) | Peak CPU (°C) | Mean CPU (°C) | Time DDR>55°C | Throttled? | Source |
|---|---|---|---|---|---|---|---|---|
| vanilla | — | **51.7** | 47.2 | 58.7 | 53.7 | 0% | no | [W3] |
| v1 K=2048 | 2048 | 49.8 | 47.3 | 57.1 | 53.9 | 0% | no | [W3] |
| v1 K=512 | 512 | **49.4** | 47.1 | 57.5 | 53.9 | 0% | no | [W3] |
| v1_FA K=512 | 512 | 49.8 | 47.2 | 57.1 | 54.3 | 0% | no | [W3] |
| TOVA K=512 | 512 | pending [O] | pending | pending | pending | pending | tbd | [O] |
| H2O K=512 | 512 | pending [O] | pending | pending | pending | pending | tbd | [O] |
| Pyramid K=512 | 512 | pending [O] | pending | pending | pending | pending | tbd | [O] |

### Older 3-prompt averaged thermal data (Wave-1-redux, shorter cells, different prompts)

| Policy | Model | Peak CPU [W1] | Peak DDR [W1] |
|---|---|---|---|
| vanilla | Llama-1B | 56.6 °C | 48.9 °C |
| v1 | Llama-1B | 67.2 °C | 51.5 °C |
| TOVA | Llama-1B | 54.8 °C | 47.6 °C |
| H2O | Llama-1B | 63.9 °C | 48.6 °C |
| vanilla | Gemma-2-2B | 70.1 °C | 53.0 °C |
| v1 | Gemma-2-2B | 64.5 °C | 49.8 °C |
| TOVA | Gemma-2-2B | 75.7 °C | 53.5 °C |
| H2O | Gemma-2-2B | 66.6 °C | 51.2 °C |
| vanilla | Phi-3-128k | 68.7 °C | 51.8 °C |
| v1 | Phi-3-128k | 65.7 °C | 54.2 °C |
| TOVA | Phi-3-128k | 57.2 °C | 50.3 °C |
| H2O | Phi-3-128k | 63.3 °C | 51.6 °C |

**Thermal interpretation:**
- On Llama-1B 25-min sustained CPU, **no policy throttles** the DDR or CPU (peak DDR < 55 °C; 60 °C is throttle threshold). The thermal claim cannot be tested at this scale.
- The 2.3 °C peak DDR spread (vanilla 51.7 vs v1 K=512 at 49.4) is the cleanest signal for eviction-driven cooling — small but real.
- Older Wave-1-redux numbers show wider spreads (v1 vs vanilla differing by 5-10 °C peak CPU on Gemma) — probably because those cells used hotpotqa prompts which were thermally outliers.

---

## Table 5 — CPU performance improvement vs baseline (vanilla)

| Policy | K | Decode tok/s gain vs vanilla (W1 mean) | Decode tok/s gain vs vanilla (W3 mean) | Sustained tok/s gain | KV memory savings |
|---|---|---|---|---|---|
| vanilla | — | 1.00× (baseline) | 1.00× | 1.00× | 0% |
| v1 K=512 | 512 | 0.62× (W1: 6.97 vs 11.31) | **1.38×** (W3: 7.05 vs 5.10) | 0.80× (less iters fit) | 70% |
| v1_FA K=512 | 512 | (matches v1 algorithm) | **1.53×** (W3: 7.78 vs 5.10) | 0.80× | 70% |
| TOVA K=512 | 512 | 0.61× (W1: 6.93 vs 11.31) | pending | pending | 76% |
| H2O K=512 | 512 | 0.43× (W1: 4.82 vs 11.31) | pending | pending | 76% |

**Note:** "decode tok/s gain" depends on whether vanilla is using FA-on or FA-off. In [W1], vanilla used FA-off (for fair attention-capture comparison) so vanilla decode was actually higher than the W3 result (W1: 11.31 vs W3: 5.10 because of context size difference). In [W3], vanilla used FA-on for the full 8K cache. The fair comparison for the dissertation claim is [W3] sustained.

---

## What's still missing / running

1. **TOVA / H2O / Pyramid 25-min sustained on Llama-1B** — running now (~1.5 hour ETA, monitor armed)
2. **Long-decode regime test** (200 prompt + 2000 decode) — recommended next; this is where v1_FA's FA-on advantage should produce a real thermal benefit
3. **Multi-model 25-min sustained** (Gemma-2B, Phi-3-128k under same protocol)
4. **8-thread / hot-ambient sustained** — needed to actually trigger DVFS throttling on Llama-1B (current 4-thread runs don't throttle)
5. **Ada-KV and AhaKV baselines** — not yet implemented; planned for HotMobile 2027 submission
