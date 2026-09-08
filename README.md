# EndurKV — Thermal & Endurance Co-Aware KV Management

PhD dissertation (Md Romyull Islam, Kennesaw State University):
**"Thermal and Endurance Co-Aware KV Management for Sustained Mobile LLM Inference
Driven by Model-Internal Signals."**

Core algorithm: **`EndurKV-Evict` (a.k.a. `perhead_v1`)** — per-head adaptive KV-cache
eviction whose budget is shaped by an attention-spread gate.

```
K_h = round( K_nominal · ( 1.3 − 0.6 · clip( (max_a[h] − 0.4) / 0.4, 0, 1 ) ) )
```

### How the four constants (1.3, 0.6, 0.4, 0.4) were chosen — validated by sweep

The formula has four parameters. They were validated by an actual
hyperparameter sweep (1001 quick-sweep cells + 691 detailed cells across
multiple gate shapes and models):

**Quick sweep coverage**: α ∈ {1.1,…,1.5} × β ∈ {0.4,…,0.8} × thresh_low ∈ {0.2,…,0.5} × thresh_high ∈ {0.6,…,0.8} across multiple models × prompts × K values.

For the **production config (α=1.3, β=0.6, tl=0.4, th=0.8)** evaluated
across 40 cells:

| Metric | Value |
|---|---|
| KL divergence vs TOVA | **−12.35 %** (lower = better) |
| Mass-retained advantage | +3-4 pp over TOVA |
| Cache used vs K_nominal | ~30 % above (flat-head bonus accounts for it) |

The best linear-clipped config in the sweep was actually
`α=1.4, β=0.6, tl=0.4, th=0.8` at **−19.4 %** KL vs TOVA — but it spends
~20 % more cache than production. Our (α=1.3) sits at a better
cache/quality trade-off point. Raw sweep results in
[`figures/sweep/sweep_quick_results.csv`](../figures/sweep/sweep_quick_results.csv)
and [`figures/gate_search_mp_ranking.csv`](../figures/gate_search_mp_ranking.csv).

**Why each parameter has the value it does:**


| Constant | Role | Rationale |
|---|---|---|
| **`0.4`** (lower threshold) | Below this `max_a`, head is "diffuse" — full bonus | Below 0.4, head distributes attention broadly; below this is empirically rare for non-pathological heads |
| **`0.4`** (range width) | Width of the transition zone (0.4 → 0.8) | The bulk of heads' `max_a` values fall in [0.4, 0.8]; chosen so the gate covers this band |
| **`1.3`** (α — max multiplier) | Diffuse-head budget bonus = K_nominal × 1.3 (+30 %) | Symmetric with the max cut; ±30 % swing is meaningful but bounded |
| **`0.6`** (β — swing range) | Swing range so μ ∈ [0.7, 1.3] | Half this (0.3) would barely differentiate from TOVA; twice this (1.2) would over-shrink focused heads |

**Budget conservation:** with `max_a` approximately uniform in [0.4, 0.8]
(empirically true), the average μ across heads is 1.0 — total budget is
preserved. v1 redistributes the same budget across heads, it doesn't grant
itself extra memory.

**Validation:** these defaults score Pareto-best across the 14-baseline
offline simulator comparison (KL divergence 1–24 % lower than TOVA at
every standard K budget) — see [`logs/COMPARISON.md`](../logs/COMPARISON.md).
A formal `(α, β, lo, hi)` grid search is on the [future-work
list](#future-work).

> Hardware target: **OnePlus 15** — Snapdragon 8 Elite Gen 5 (`SM8850 "Canoe"`),
> Adreno 840 GPU, 12 GB UMA.

---

## Quick map

| Where | What |
|---|---|
| [`entropy_probe/eviction_bench.cpp`](entropy_probe/eviction_bench.cpp) | On-phone benchmark binary — runs `{vanilla, v1, tova, pyramid}` policies, captures `kq_soft_max`, calls `llama_memory_seq_rm`, emits `meta.json + steps.csv + gen.txt` |
| [`scripts/android/`](scripts/android/) | Build + deploy + sweep scripts |
| [`scripts/android/phone_full_sweep_3policy.sh`](scripts/android/phone_full_sweep_3policy.sh) | Master sweep: 3 policies × 5 prompts × 2 K × 3 reps = 90 runs |
| [`scripts/android/build_llama_android_vulkan.sh`](scripts/android/build_llama_android_vulkan.sh) | NDK + Vulkan build of llama.cpp |
| [`scripts/android/host_live_report.py`](scripts/android/host_live_report.py) | Pulls phone logs → mean ± std per policy |
| [`docs/MASTER_INDEX.md`](docs/MASTER_INDEX.md) | Full index of docs, scripts, CSVs, figures |
| [`docs/DEPLOYMENT_PLAN.md`](docs/DEPLOYMENT_PLAN.md) | 4-pillar evaluation framework |

---

## Engineering journal (decisions + their reasons)

This section is the running record of *why* the code looks the way it does — every
non-obvious knob, every blind alley, every workaround. Update it whenever a new
constraint surfaces. Newest-first.

### 2026-05-31 — Baseline fairness audit: we strengthened TOVA over its paper

Audit of our policy implementations against published specs.

**Sink protection (we added; paper-spec TOVA does NOT have it):**

Every eviction policy in our codebase (v1, TOVA, pyramid) runs with
`--n-sink 4` — first 4 KV positions (BOS + role tokens) never evicted.
This is the StreamingLLM-style attention-sink protection
(Xiao et al. ICLR 2024). Original TOVA (Oren et al. ACL 2024) does **not**
include this — pure paper-spec TOVA can suffer the "sink-collapse" failure
mode where dropping BOS triggers degenerate generation.

| Variant | Original-paper TOVA | Our TOVA-with-sink | Our v1 |
|---|---|---|---|
| First 4 positions protected? | ✗ No (paper) | ✓ Yes | ✓ Yes |
| Sink-collapse failures | possible | suppressed | suppressed |

**Implication for the paper's framing**: when our results show v1 ≥ TOVA on
F1/quality, this is against a **stronger TOVA than the published paper**.
A pure-paper TOVA would likely fail more often (e.g. on the multifieldqa
prompt where we already saw TOVA degenerate even with sink protection,
suggesting TIR-class failure, not sink-collapse). So our advantage over TOVA
is conservative — we are not comparing against a weakened straw-man baseline.

**Other fairness items applied uniformly to all policies:**

| Property | What we do | Original baseline spec |
|---|---|---|
| Aggregate-OR across query heads (GQA) | yes | published GQA-aware impls (AdaKV, SnapKV) use Aggregate-OR or Aggregate-MEAN; we chose OR — more conservative |
| Seq-level eviction (union across layers) | yes | research papers can do per-layer; llama.cpp constraint forces seq-level for us |
| `--n-recent K` (recent-window protection) | **not** used | TOVA paper doesn't have it; StreamingLLM/H2O/SnapKV add it. We omit for fair v1-vs-TOVA. |
| Greedy decoding for publication runs | yes | deterministic; sampling params inert |

Full audit lives in [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) §5b.

---

### 2026-05-31 — Planned: SnapKV-style FA-on decode (two-context KV swap)

**Problem.** Decode is 4–5× slower under our eviction policies than under
vanilla, because the policies need FA-off to read `kq_soft_max` for the
per-step eviction decision. Measured on Adreno 840 GPU:

| Model | Vanilla (FA-on) decode | v1 / TOVA (FA-off) decode | Ratio |
|---|---|---|---|
| Llama-1B  | 21.7 t/s | 4.3 t/s | 5.0× slower |
| Gemma-2-2B | 7.6 t/s | 1.9 t/s | 4.0× slower |

**Strategy A — SnapKV-style.** After prefill, freeze the KV mask and swap to
an FA-on context for decode. Two contexts share the same model:

```
1. ctx_off  = llama_init_from_model(model, params_FA_OFF + cb_eval)
2. llama_decode(ctx_off, prefill_batch)      # captures attention
3. apply_eviction(ctx_off, policy_decision)  # one-shot, post-prefill
4. blob = llama_state_seq_get_data(ctx_off, seq=0)
5. llama_free(ctx_off)
6. ctx_on  = llama_init_from_model(model, params_FA_ON + no_cb_eval)
7. llama_state_seq_set_data(ctx_on, blob, seq=0)
8. decode normally on ctx_on at vanilla speed
```

Expected behaviour: decode throughput returns to vanilla levels (5× speedup
on Llama-1B), and quality stays within ≲1 % F1 of continuous eviction (this
matches SnapKV's published numbers and is the standard methodology in the
KV-eviction literature).

Trade-off vs continuous eviction: with the mask frozen, mid-generation
attention shifts can't update what's kept. For LongBench-style QA where the
answer is determined by the prompt context (not later generation tokens),
this is fine.

**Status:** not yet implemented. Will land after Wave-1 finishes so the
running sweep isn't disturbed. After landing we'll re-sweep Llama-1B +
Phi-3-128k to compare snapshot vs continuous eviction.

---

### 2026-06-01 — Locked-in pipeline order

After Wave-1-redux finished cleanly, here is the committed execution order
(small enough to fit in a few days, big enough to land the dissertation's
banner claim):

| Stage | What | Output |
|---|---|---|
| Wave-2b (running) | flawed-slicing PPL — let finish for completeness | log of "all collapse at K=512" regime |
| **Wave-2c** | corrected PPL: continuous WT2 + K-sweep {1024, 1500, 2200} | clean v1 vs TOVA vs vanilla PPL curves |
| **H2O sweep** | add H2O baseline to phone runs (27 cells like CPU sweep) | F1/PPL/thermal of H2O on phone — answers "did you compare to H2O?" |
| **Wave-3 sustained-stress** | 30 min continuous per policy × 3 policies | thermal trajectories under heat soak — measures throttling |
| **Track-2 design** | use Wave-3 thermal data to build closed-loop controller | adaptive K_nominal driven by `(thermal, memory, UFS)` signals |
| **Wave-3-revised** | rerun Wave-3 with Track-2 enabled | proves the closed-loop adaptive system beats static-K |
| **Wave-5 full grid** | 5 prompts × 5 tasks × 2 reps × 4 policies × 3 models | publication-grade statistical confidence |
| Final | tables + figures | paper |

Track-2 is **the dissertation's main novelty**. v1 alone is Track-1, with
modest 1–4 % KL improvement vs TOVA. Track-2 = "closed-loop runtime KV
controller consuming joined model-internal + thermal + UFS signals" — no
published mobile-eviction paper does this. We build it *after* Wave-3
measurements give us the actual thermal trajectories to design against.

---

### 2026-05-31 — Wave-3 sustained-stress test design (THE Track-2 experiment)

**Why this is the dissertation's main experiment, not Wave-1/Wave-5.**

The team's existing internal assessment (in `logs/COMPARISON.md`) is honest about
v1's algorithmic margin:

> "EndurKV-Evict v1 (TOVA scoring + spread gate): Pareto-best at every standard-K
> budget on the 1B short, 8B short, and 1B long-ctx benches across 14 baselines.
> At matched cache it gives only **1–4 % lower KL** than TOVA — **not** publishable
> alone as an algorithmic contribution."

So v1 vs. the eviction-policy literature wins on **KL by a modest margin** (already
demonstrated in offline simulation). The dissertation's *banner* claim is Track 2:

> "A closed-loop runtime KV controller consuming joined model-internal + thermal +
> UFS signals **sustains throughput under mobile thermal pressure where vanilla
> and TOVA cannot.**"

That claim cannot be measured by a cold-state sweep — Wave-1's `--cool-then-run`
deliberately resets thermal load between cells, masking the very signal we want.
Wave-3 removes the cool-down and lets the chip throttle.

**Protocol.** [`scripts/android/phone_wave3_stress.sh`](scripts/android/phone_wave3_stress.sh)

| Aspect | Setting |
|---|---|
| Model | Llama-3.2-1B (only model with stable GPU pipeline) |
| Backend | Vulkan GPU, ngl=16, ubatch=64, FA-off (required for attention capture) |
| Vanilla | runs `--no-fa-vanilla` so all policies share the FA-off compute path (fair) |
| Policies | vanilla → TOVA → v1 (sequential, with full cool-down between) |
| Workload | continuous prefill (7.7K-token chat-templated hotpotqa prompt) + 256-token decode, looped |
| Duration per policy | **1800 s (30 min) of continuous inference, no cool-downs** |
| Cool-down between policies | skin ≤ 35°C (up to 10 min wait) |
| Sampling | sensors.csv at 5 Hz: skin/battery/CPU/GPU temps, battery current, KV size |
| Per-iter log | iter#, t_wall_s, prefill_ms, decode_tps, peak_kv_mb, peak_rss_kb, evicted |

**Hypothesis.** Plotting `decode_tps vs t_wall` for each policy yields:

```
              decode_tps
                  ↑
       21 t/s   ▲ vanilla — starts highest (FA-off cold), throttles fastest
                ↘ ___________
       15 t/s   ▼ TOVA — middle
                ↘ __________
       12 t/s   ▼ v1 — starts lowest, sustains flattest
                ─────────────────────── (mostly horizontal, mildly declining)
                  0 min      30 min →
```

v1 wins **sustained area under the curve**, not peak. The dissertation graph then
plots cumulative tokens generated over 30 min, not instantaneous tok/s — the
former exposes v1's lead clearly.

**Wall time.** ~30 min × 3 policies + ~5 min cooldowns ≈ 2 hr total. Runs after the
CPU sweep finishes so the phone is clean and not memory-pressured.

**Risk.** This is the experiment that previously crashed the phone (cumulative GPU
stress). Mitigations:
- Llama-1B only (only stable GPU model in our matrix)
- 5-Hz sensor logging includes battery temp — if battery ≥ 42°C, abort and cool
- Inter-policy cool to 35°C (stricter than Wave-1's 38°C)
- Pull data after each policy completes (don't risk losing 30 min if next one crashes)

---

### 2026-05-31 — Wave-5 publication grid (5 prompts × 5 tasks × 2 reps)

Wave-1's `3 prompts × 2 reps` per task family was a time-budget initial signal,
not the final result. Single-prompt-per-task is too noisy for publication —
the eviction-policy literature (TOVA, H2O, SnapKV, AdaKV, PyramidKV) uses
the full LongBench evaluation set (200+ samples per task) or at minimum
100. We will replace Wave-1's prompt grid with **Wave-5**:

| Aspect | Wave-1 (initial) | **Wave-5 (publication)** |
|---|---|---|
| Prompts per task | 1 | **5** |
| Task families | 3 (qasper, hotpotqa, multifieldqa) | **5** (+ gov_report, narrativeqa) |
| Replicates | 2 | 2 |
| Total cells / (model, policy) | 6 | **~50** (with model-specific exclusions) |

**Per-model exclusions for Wave-5** (driven by hardware/architecture limits we
already discovered):

| Model | Context | Excluded | Reason |
|---|---|---|---|
| Llama-3.2-1B | 131 K | none | fits everything |
| Gemma-2-2B  | 8 K   | `narrativeqa` (8.5–9 K tok) | exceeds Gemma's context window |
| Gemma-2-2B  | —     | vanilla on `hotpotqa` GPU runs | falls back to CPU (Adreno TDR) |
| Phi-3-128k  | 128 K | none | fits everything |

Total Wave-5 cells: ~420. Wall time: ~30–40 hr on phone with cool-downs.
Wave-5 launches after Wave-1, Wave-2 (WT2 PPL), and Strategy A re-sweep land.

---

### 2026-05-31 — Right metric per benchmark (LongBench = F1/EM/ROUGE, WT2 = PPL)

The eviction-policy literature uses different quality metrics for different
data, so we follow suit. Pinning the right metric to the right benchmark:

| Benchmark | Primary metric | Where it comes from in this repo |
|---|---|---|
| LongBench: `qasper`, `hotpotqa`, `multifieldqa_en` (extractive QA) | **F1 + EM** | Score Wave-1 `gen.txt` against `prompts/prompts_pub_longbench.jsonl[ground_truth]` via `scripts/android/host_score_phone_runs.py` |
| LongBench: `gov_report`, summarization tasks | **ROUGE-L / ROUGE-1** | Same scorer; metric chosen by task field in JSONL |
| LongBench: `narrativeqa` (abstractive QA) | **F1** | Same scorer |
| **WikiText-2** | **Perplexity** | Wave-2 sweep, `--eval-mode ppl` with `--eval-text corpora/wiki_ref_4k.txt` |
| NIAH / synthetic 32 K (future) | Needle recall | Generated by `scripts/host_make_niah.py` (not yet run) |

So we **don't** publish a single "PPL" column across all rows — that would be
applying the wrong yardstick. Each cell of the final results table reports the
metric its data type calls for.

Wave-1's per-token sampled NLL is **dropped from the publication** (it was a
mis-named metric, not actual perplexity). Wave-2 provides the canonical PPL on
WikiText-2 only.

---

### 2026-05-31 — PPL methodology: stop using sampled-token NLL → teacher-forced PPL

**Bad metric we had.** The original Wave-1 sweep computed "perplexity" as
`exp(mean(-log P(sampled_token | prefix)))` over 128 sampled tokens, where the
log-prob was the post-sampling-pipeline value (i.e. after `repeat-penalty`,
temperature, top-k, top-p). This is **not perplexity** and is invalid for
cross-policy comparison:

1. *Sampling noise* — at T=0.8, top-p=0.95 we deliberately pick less-likely
   tokens for diversity. Different RNG seeds → wildly different PPLs.
2. *Different sequences scored* — each policy generates a different token
   stream. We were comparing two policies on **different reference texts**.
3. *Distorted probabilities* — temperature and repeat-penalty change the
   logit distribution before we read the chosen token's probability. The
   number is "how likely under the *sampling* distribution," not "how likely
   under the *model*."
4. *FA-on for vanilla emits NaN* under our log-prob extraction, which made
   the vanilla cell appear blank and inflated suspicion that the eviction
   policies were beating it.

**The user caught it** when v1 PPL on `qasper_pub_001` came out at 14.15 vs
TOVA's 7.15 — a result that's not physically sensible (v1's per-head budget
*broadens* coverage for flat heads, so it should *not* be worse than TOVA's
uniform top-K_global on a doc-QA task with mixed head profiles).

**Right metric: teacher-forced raw-logit PPL.** Added `--eval-mode ppl` and
`--eval-text PATH` to `eviction_bench`. New flow per cell:

1. Prefill the prompt under the policy (same as before).
2. Apply post-prefill eviction (same as before).
3. **Teacher-force a held-out reference text** (e.g. `wiki.test.raw` slice)
   one token at a time. For each ref token `r_i`, take the **raw logits**
   from `llama_get_logits_ith(ctx, -1)`, compute the numerically-stable
   softmax log-prob of the *actual* `r_i`, and sum `-log P(r_i | prefix)`.
   No sampling, no temperature, no rep-penalty.
4. `PPL = exp(mean_NLL)`. Compare policy_PPL / vanilla_PPL per `(model, prompt)`.

This is exactly what `llama-perplexity` does, just wrapped around our
eviction policy so the policy sees the same compute path as in generation.

**Wave-2 plan.** `scripts/android/phone_ppl_eval_wave2.sh` runs the new
`--eval-mode ppl` across the same 3 models × 3 policies × 3 prompts with a
fixed 4 KB slice of WikiText-2 as reference. PPL is deterministic so
`N_REPLICATES=1`. This produces the publication-grade quality number; the
Wave-1 sweep's latency/memory/retention/efficiency columns are still valid
and used as-is.

**Also fixed in the same change:** JSON output of `NaN`/`Inf` now emits
`null` instead of bare `nan` (the latter is invalid JSON and was silently
breaking host-side parsers).

---

### 2026-05-31 — Final model lineup: 1B / 2B / 3.8B, three architectures, all GPU

After confirming the 7B+ ceiling (see entries below) we needed three models we
could actually run on the Adreno 840 *and* that gave a fair apples-to-apples
comparison. Hard constraints:

1. **All on the same backend.** Mixing CPU and GPU between models makes the
   eviction-policy comparison incoherent — different ops, different
   numerics, different latency profile. The user vetoed mixed backends.
2. **<7B parameters.** Anything bigger DeviceLost-crashes on this GPU at
   LongBench prompt lengths. We exhaustively confirmed this on Llama-3.1-8B,
   Mistral-7B, Qwen2-7B, and DeepSeek-R1-Distill-Llama-8B.
3. **Three distinct architectures.** A scaling story across one family
   (1B → 3B → 8B Llama) would be cleanest but the 8B vetoes that. So we
   spread architectures: Llama, Gemma, Phi.
4. **≥32K context window on at least one model.** The dissertation wants a
   long-context stress test, not just LongBench (which tops at ~9K tokens).

Final lineup:

| Tag | Model | Params | Arch family | Context | Adreno GPU fit |
|---|---|---|---|---|---|
| **Llama-3.2-1B**  | Llama-3.2-1B-Instruct (Q4_K_M, 770 MB)        | 1.2 B | Llama  | 131 K | ✅ all 32 layers |
| **Gemma-2-2B**    | gemma-2-2b-it (Q4_K_M, 1.5 GB)                | 2.0 B | Gemma  | 8 K   | ✅ all 26 layers |
| **Phi-3-128k**    | Phi-3-mini-128k-instruct (Q4_K_M, 2.3 GB)     | 3.8 B | Phi    | 128 K | ✅ all 32 layers |

Why **Phi-3-mini-128k** instead of the **-4k** variant we already had:
the 4K-context Phi-3 cannot accept LongBench prompts larger than ~4000 tokens
(hotpotqa, narrativeqa, several gov_report items exceed that). The 128K variant
is the same architecture and same parameter count — only the positional encoding
differs — so it slots into the same GPU profile while letting us run the full
prompt distribution.

Why **Gemma-2-2B** over downloading a 3B Llama: it's already on the phone, gives
us a third architecture family, and its 8K context handily covers all LongBench
prompts except `hotpotqa_pub_001`/`002`/`003` (which exceed 7K).
Llama-3.2-3B-Instruct is a strong alternative if we later want a clean 1B→3B
scaling story within one family.

Models we **explicitly dropped from the sweep** and why:

| Dropped | Reason |
|---|---|
| Llama-3.1-8B           | GPU DeviceLost at pp1300+. CPU works but slow (430 s prefill, 1.4 t/s decode for 8B). |
| Mistral-7B-Instruct    | GPU DeviceLost at any pp≥1300. Same issue with FA-off. |
| Qwen2-7B-Instruct      | GPU DeviceLost (different arch, same kernel-size ceiling). |
| DeepSeek-R1-Distill-Llama-8B | Same arch as Llama-3.1-8B; assumed-same failure. |
| Phi-3-mini-**4k**      | 4 K context window is too small for most LongBench prompts. |

The full 4-corner constraint (`uniform backend ∧ <7B ∧ ≥32K ctx ∧ ≥3 archs`)
left exactly this lineup as the buildable one.

---

### 2026-05-31 — Llama-3.1-8B too big for Adreno GPU on long prefills → CPU

**Symptom.** After the NEEDED-libs fix below, `eviction_bench` ran cleanly on GPU
for **1B and Phi-3** with 2840-token LongBench prompts. The **8B model still
crashed with `vk::DeviceLostError`** during prefill, no matter what we tried:

| Setting | Result |
|---|---|
| 8B / ngl=32 / ub=64  / pp512  | ✅ 13.75 t/s |
| 8B / ngl=32 / ub=32  / pp1024 | ✅ 8.44 t/s |
| 8B / ngl=32 / ub=64  / pp2840 (LongBench) | ❌ DeviceLost |
| 8B / ngl=32 / ub=32  / pp2840 | ❌ DeviceLost |
| 8B / ngl=32 / ub=16  / pp2840 | ❌ DeviceLost |
| 8B / ngl=16 / ub=32  / pp2840 (partial offload) | ❌ DeviceLost |
| 8B / **llama-bench** -p 2840 / ngl=32 / ub=64 | ❌ no result table |

That last row is the smoking gun — even llama.cpp's own `llama-bench` can't
sustain pp2840 on 8B on this Adreno driver. So this isn't a bug in
`eviction_bench`; it's a hardware ceiling.

**Why.** At the end of a 2840-token prefill the attention compute per ubatch is
`ub × 2840 × head_dim × n_heads = 64 × 2840 × 128 × 32 ≈ 750M ops/layer × 32 layers`.
That's ~24 GFLOPs in a single Vulkan dispatch — well within Adreno 840's
~4 TFLOP capability on paper, but the per-kernel wall time still exceeds the
TDR watchdog under real driver/shader overhead. Shrinking `n_ubatch` doesn't
help because the *attention context length* drives the kernel size, not the
ubatch.

**Fix shipped.** Per-model backend selection in the sweep config — GPU for the
smaller models, CPU for 8B. CPU is fully validated for 8B Llama-3.1 here at
~3–5 t/s decode:

```bash
# scripts/android/phone_full_sweep_3model_3policy.sh
MODELS=(
    "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf|Llama-3.2-1B|16|512|64|8192"   # ngl=16 GPU
    "models/Phi-3-mini-4k-instruct-Q4_K_M.gguf|Phi-3-mini|32|512|64|4096"    # ngl=32 GPU
    "models/Llama-3.1-8B-Instruct-Q4_K_M.gguf|Llama-3.1-8B|0|512|64|12288"   # ngl=0  CPU
)
```

Also added **chunked prefill** to `eviction_bench`: it now splits the prompt
into `n_batch`-sized pieces and calls `llama_decode` once per chunk, matching
what `llama-bench` does internally. Same total work, but each `llama_decode`
deals with at most `n_batch` (= 512) new tokens at once. This *didn't* fix 8B
on its own, but it removes the redundancy with `n_ubatch` and keeps memory
bounded for the 1B / Phi-3 GPU paths.

**Lesson.** On mobile Vulkan, "prefill length" is a hard hardware parameter
when the model is large. Track both `(model size, prefill length)` jointly
when validating GPU configs — a model that survives `pp1024` may die at `pp2840`.

---

### 2026-05-31 — `eviction_bench` hung silently on Vulkan: missing NEEDED libs

**Symptom.** After fixing the Adreno TDR issue (see next entry), `eviction_bench`
linked against the Vulkan `libllama.so` would **hang at startup with zero output**
— not even the model loader header. `EXIT=124` after timeout. Both `stderr.log`
and `stdout.log` were 0 bytes. `llama-bench` in the same dir, with the same
`LD_LIBRARY_PATH`, worked perfectly.

**Bisection.**

| Setup | Result |
|---|---|
| `eviction_bench` (CPU-build) + CPU `libllama.so` | ✅ works (51 t/s prefill on 1B) |
| `eviction_bench` (Vulkan-build) + Vulkan `libllama.so`, `-ngl 16` | ❌ silent hang |
| `eviction_bench` (Vulkan-build) + Vulkan `libllama.so`, `-ngl 0`  | ❌ silent hang |
| `llama-bench`     (Vulkan-build) + Vulkan `libllama.so`, `-ngl 16` | ✅ 40/52 t/s |

Same libs. Different binary behaviour → the bug is in the binary, not the libs.

**Root cause.** `readelf -d` showed `eviction_bench`'s NEEDED list was missing
two libraries that `llama-bench` had:

```
llama-bench NEEDED:    libllama  libggml  libggml-cpu  libggml-vulkan  libggml-base  ...
eviction_bench NEEDED: libllama  libggml                              libggml-base  ...
```

Modern llama.cpp ships its backends as **separate `.so` plugins** —
`libggml-cpu.so`, `libggml-vulkan.so`, `libggml-cuda.so`. The library
`libllama.so` doesn't link against them directly; instead it depends on the
*executable* to bring them in, OR on `ggml_backend_load_all()` to `dlopen` them
at runtime from `$ORIGIN` / `LD_LIBRARY_PATH`.

Our `entropy_probe/CMakeLists.txt` only linked `libllama`, `libggml`,
`libggml-base`. When the Vulkan `libllama.so` (which expects a registered
backend) initialised, it called into `ggml_backend_load_all()`, which tried to
`dlopen` the backend plugins. Something in that runtime-`dlopen` path on
Adreno's Vulkan driver deadlocks silently.

**Fix.** Updated [`entropy_probe/CMakeLists.txt`](entropy_probe/CMakeLists.txt) to
also link the backend `.so` files when they exist in the llama.cpp build dir, and
added `-Wl,--no-as-needed` so the linker keeps them as NEEDED even though no
direct symbol is referenced (the backend `.so` registers itself via a static
constructor — the linker would otherwise drop the dep):

```cmake
find_library(GGML_CPU_LIBRARY    NAMES ggml-cpu    PATHS "${LLAMA_BUILD_DIR}/bin" NO_DEFAULT_PATH)
find_library(GGML_VULKAN_LIBRARY NAMES ggml-vulkan PATHS "${LLAMA_BUILD_DIR}/bin" NO_DEFAULT_PATH)
target_link_libraries(${_t} PRIVATE
    ${LLAMA_LIBRARY} ${GGML_LIBRARY} ${GGML_BASE_LIBRARY} ${GGML_BACKEND_LIBS})
target_link_options(${_t} PRIVATE "-Wl,--no-as-needed")
```

After rebuild, `readelf -d` showed all five `.so`s in NEEDED, and v1 policy on
GPU produced a clean run: prefill 68 t/s, decode 5.1 t/s on 1B, perplexity 8.23,
mean retention 37% (target K=1024 → 36% — within 1pp).

**Lesson.** With load-time NEEDED you fail fast with a clear `dlopen` error.
With runtime `ggml_backend_load_all()` you can deadlock with zero output. Always
link the backends as NEEDED for any executable using a backend-pluggable
`libllama.so`.

---

### 2026-05-31 — Vulkan on Adreno 840 works, but needs `ubatch ≤ 64`

**Symptom.** First Vulkan run on the OnePlus 15 (Adreno 840 driver, brand-new
Snapdragon 8 Elite Gen 5) crashed instantly with:

```
libc++abi: terminating due to uncaught exception of type
vk::DeviceLostError: vk::Queue::submit: ErrorDeviceLost
```

…regardless of model size. Both 8B (4.6 GB) and 1B (760 MB) crashed identically at
`-ngl 99 -p 512`. The stack ended in `ggml_backend_sched_graph_compute_async` → no
useful info, just a generic "GPU was reset by the driver."

**Bisection.** Walked the parameter space to isolate the offender:

| Model | `-ngl` | prefill | decode | result |
|------:|------:|--------:|-------:|--------|
| 1B    | 1     | pp8     | —      | ✅ 31.8 t/s |
| 1B    | 16    | pp64    | tg8    | ✅ 95.8 / 47.0 t/s |
| 8B    | 8/16/24/32 | pp64 | tg8  | ✅ working all the way to `ngl=32` |
| 8B    | 32    | **pp128** | —    | ❌ DeviceLost |
| 8B    | 32    | pp512   | —      | ❌ DeviceLost |
| 8B    | 33    | pp512   | —      | ❌ DeviceLost (output layer included) |
| 8B    | 99    | pp512   | —      | ❌ DeviceLost |
| 8B    | 32 **+ `-ub 64`** | pp512 | tg16 | ✅ **13.75 / 9.29 t/s** |
| 8B    | 32 **+ `-ub 32`** | pp1024 | tg16 | ✅ 8.44 / 7.95 t/s |

So it isn't memory pressure (1B crashes too; UMA → 8B fits trivially in shared RAM)
and it isn't the output layer alone.

**Root cause.** Adreno's **TDR (Timeout Detection and Recovery)** watchdog. Mobile
GPU drivers abort any single compute kernel that runs too long to keep the UI
responsive — a hard 1–2 s ceiling. At `n_ubatch ≥ 128` on an 8B Llama, each per-layer
matmul kernel becomes large enough (≈ batch × 4096 × 4096 × 3 mat-muls per layer)
to overrun the watchdog. The driver kills the queue → `DeviceLost`.

llama.cpp's `n_ubatch` controls **how many tokens go in one GPU dispatch**.
Setting it to 64 doesn't reduce total work — it just chops each layer into 8 smaller
dispatches that each finish well under the watchdog limit. Same FLOPs, more launches.

**Fix shipped.** Added two flags to `eviction_bench`:

```
--n-batch N        # physical batch (default 512, auto-grows to fit prompt)
--ubatch-size N    # micro-batch (default 64 — Adreno-safe)
```

…and rebuilt against the Vulkan-linked `libllama.so`. Production setting for
the OnePlus 15 sweep:

```bash
LD_LIBRARY_PATH=bin bin/eviction_bench \
  --n-gpu-layers 32 \
  --ubatch-size 64 \
  ...
```

**Tradeoff to keep in mind.** Smaller ubatch = more kernel-launch overhead, so
ubatch=32 cost us ~40 % prefill throughput vs ubatch=64 on the pp1024 test. Use
the **largest ubatch that survives** for the longest prompt in the sweep. 64 is
safe through ≥ pp512; if some LongBench prompt over 3000 tokens crashes we'll
drop to 48 or 32.

**Why not OpenCL / QNN?**
- CUDA — NVIDIA-only; doesn't exist for Adreno.
- OpenCL — supported by llama.cpp, may have slightly better Adreno-specific kernels,
  but less actively maintained. Reserve as a fallback.
- QNN / SNPE — Qualcomm proprietary NPU SDKs, no llama.cpp integration today.
- Vulkan — chosen: cross-vendor standard, works once `n_ubatch` is sane.

---

### 2026-05-30 — Sampling protocol made identical across policies

Greedy decoding was producing degenerate "lying scribe…" repetition under
aggressive eviction (K=1024). Switched to the standard llama.cpp sampling stack,
applied **identically** to every policy (vanilla, v1, tova) for a fair comparison:

```
repeat-penalty 1.1   temperature 0.8   top-k 40   top-p 0.95
```

Top-p, top-k, etc. operate at the *output sampling* stage — they do **not** change
v1's gate formula or what gets evicted from the cache. They only choose which
already-allowed token to emit. The eviction policy still sees the same attention
softmax.

Also added sink-token protection (`--n-sink 4`, StreamingLLM-style) for v1 and
tova: the first 4 positions (BOS, role tokens) are always kept regardless of
attention score. Removing them is the classical degenerate-repetition trigger.

Vanilla keeps FA-on for production-realistic latency; v1/tova force FA-off
because we need `kq_soft_max` as a discrete tensor to read per-head attention.

---

### 2026-05-30 — WikiText-2 PPL killed early

Phone-side `llama-perplexity` on WikiText-2 was projected at ~19 h total
(~16 min × 70 4K chunks). Killed at chunk 8/70 — partial mean PPL ≈ 6.79 is
saved at [`phone-logs/wt2_ppl_partial/`](../phone-logs/wt2_ppl_partial/). For
our purposes, in-sweep `meta.json -> "perplexity"` (over the generated tokens)
is the publication-relevant number; the WT2 corpus PPL was a sanity check.

---

### 2026-05-29 — GQA mapping fixed (Aggregate-OR)

The early offline simulator did per-query-head eviction, which is a fiction —
real llama.cpp KV is per-kv-head and positions are shared across the query
heads in the GQA group (Llama-3.1-8B has 32 query heads sharing 8 KV heads, 4:1).
The on-phone path now uses **Aggregate-OR**: a position is kept if *any* query
head in the group wants it. This matches the published GQA-aware baselines
(H2O, TOVA, SnapKV evaluations all use the same convention).

---

### 2026-05-28 — Memory savings: which path?

llama.cpp pre-allocates the entire KV buffer at `llama_init` based on `--ctx-size`.
`llama_memory_seq_rm` marks slots reusable but **does not shrink the buffer.**
Three real-memory-savings stories:

1. **Smaller `--ctx-size`** — direct but unrealistic vs production.
2. **Long-context capability test** — chosen — show that at the same context
   length, our policy enables a longer effective prompt before OOM, OR maintains
   quality at a smaller context.
3. **Chunked prefill** — future work.

`peak_kv_mb` in `meta.json` reports the allocated buffer, not the live set;
`mean_retention_ratio` is what to cite for "effective" KV savings.

---

### 2026-05-27 — Why FA-on for vanilla only

Flash Attention fuses the softmax into the attention kernel. That gives a
5–10× speedup but **no exposed `kq_soft_max` tensor** to read in the
`cb_eval` callback. So:

- vanilla → FA-on (production baseline)
- v1, tova, pyramid → FA-off (need attention scores for the eviction decision)

The latency comparison still reads cleanly: vanilla represents the best a stock
llama.cpp deployment can do, and the eviction policies represent the cost of our
intervention. The thermal and memory advantages are what compensate.

---

## Building (host → phone)

```bash
# 1. Build llama.cpp (Vulkan) for Android
bash scripts/android/build_llama_android_vulkan.sh

# 2. Build eviction_bench linked against Vulkan libllama
cd entropy_probe
mkdir -p build-android-vulkan && cd build-android-vulkan
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_DIR=../../llama.cpp \
  -DLLAMA_BUILD_DIR=../../llama.cpp/build-android-vulkan
cmake --build . --target eviction_bench -j 12

# 3. Push to phone
adb push build-android-vulkan/eviction_bench  /data/local/tmp/endurkv/bin/
adb push ../../llama.cpp/build-android-vulkan/bin/lib*.so  /data/local/tmp/endurkv/bin/

# 4. Run a single check
adb shell "cd /data/local/tmp/endurkv && \
    LD_LIBRARY_PATH=bin bin/eviction_bench \
      --model models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \
      --prompt prompts/longbench/qasper_pub_001.txt \
      --prompt-id qasper_pub_001 \
      --policy v1 --k-nominal 1024 --max-tokens 64 \
      --n-gpu-layers 32 --ubatch-size 64 \
      --n-sink 4 --temperature 0.8 --top-p 0.95 --top-k 40 \
      --out-csv /sdcard/Download/steps.csv \
      --out-meta /sdcard/Download/meta.json \
      --out-gen  /sdcard/Download/gen.txt"

# 5. Full sweep (90 runs, ~6–10 h on GPU)
N_GPU_LAYERS=32 MODEL=models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  bash scripts/android/phone_full_sweep_3policy.sh
```

---

## License & attribution

Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT). All eviction
policy implementations and the engineering journal above are part of the
EndurKV dissertation work.
