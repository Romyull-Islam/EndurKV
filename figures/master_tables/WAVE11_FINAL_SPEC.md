# Wave-11 Final Specification

Date: 2026-06-07
Status: APPROVED (incorporates verification fixes)
Supersedes: prior Wave-11 plan that listed `StreamingLLM` as "SOTA-mobile" and proposed K=1024 for EndurKV.

This document is the binding spec for the Wave-11 phone-side sweep. It addresses each of the three verification findings in `Verifications`:

1. SOTA naming — `StreamingLLM` was overclaimed as "SOTA-mobile". Final spec demotes it to "recency-only floor / control baseline" and adds H2O as the actual "attention-aware SOTA" reference baseline. H2O is already implemented in `eviction_bench.cpp` (`policy_h2o`, `H2OState` near line 290), so it costs zero porting time. StreamingLLM is still ported (≈30 LoC) because it gives a deterministic recency floor that no other policy provides, and because `eval_pipeline` and reviewers expect it.
2. Accuracy benchmark — NIAH stays, but framed honestly as "long-context retrieval probe with deterministic substring judge". The frozen 8-stimulus Tier-1 grid (ctx ∈ {2048, 4096, 6144, 8192} × depth ∈ {0, 25, 50, 75, 87}) is kept because the stimuli are already on the phone and host-side scorers are already wired up — re-freezing RULER would violate the "only the device changes" fidelity rule. The single-needle saturation concern is mitigated by reporting per-(ctx, depth) cells (not just overall %) so reviewers can see exactly where each policy collapses; a follow-up Wave-12 will add RULER-multi-key once stimuli are frozen.
3. K choice — K=512 is the binding K for every policy except `vanilla`. K=1024 was a cross-corpus cherry-pick (Phi-3 longgen sampling-NLL vs Llama-1B WikiText-2 teacher-forced) and would let EndurKV use 2× the cache the baselines are limited to. Headlines run at K=512 for fair comparison; K=256/384/1024 are reported only inside the K-sweep companion table (Phi-3 only) and labeled as such.

---

## 1. Final comparison set (5 policies)

All cells use `--threads 4 --n-gpu-layers 0 --n-batch 512 --ubatch-size 64 --seed 42 --greedy --ignore-eos` (omitted below for brevity). `K_NOMINAL=512`, `N_SINK=4`, `ANCHOR_TOP_K=32`, `RECENT_BUDGET = K_NOMINAL - N_SINK = 508` (for streamingllm) or `K_NOMINAL - N_SINK - ANCHOR_TOP_K = 476` (for v1_fa2_stack).

| # | Policy tag (in launcher) | Role | Full CLI flags (per-policy delta) | Runnable today |
|---|---|---|---|---|
| 1 | `vanilla` | Production baseline; FA-on; full cache; the upper bound on quality and the thermal floor reference. | `--policy vanilla --cache-type-k f16 --cache-type-v f16` | yes |
| 2 | `h2o` | **Attention-aware SOTA baseline** (Zhang et al., NeurIPS 2023). Heavy-hitter + recency + sink. The reviewer-expected strong eviction baseline. Already implemented (`policy_h2o`, `H2OState`). FA-off prefill & decode. | `--policy h2o --k-nominal 512 --n-sink 4 --cache-type-k f16 --cache-type-v f16` | yes |
| 3 | `tova` | Single-token attention selection baseline (Oren et al., 2024). Per-layer canonical, FA-off. Already implemented. | `--policy tova --k-nominal 512 --n-sink 4 --cache-type-k f16 --cache-type-v f16` | yes |
| 4 | `streamingllm` | **Recency-only floor / mobile-friendly control** (Xiao et al., ICLR 2024). Sink + sliding window; no attention readout. FA-on at both prefill and decode — the only policy in the set besides vanilla that pays no FA-off prefill cost. NOT labeled "SOTA"; labeled "no-attention control" in the paper. | `--policy streamingllm --k-nominal 512 --n-sink 4 --recent-budget 508 --cache-type-k f16 --cache-type-v f16` | needs porting (≈30 LoC) |
| 5 | `v1_fa2_stack` (**our**) | **EndurKV headline configuration.** v1_FA² with the Wave-9 thermal stack: Q8 K-cache, anchor-top-k=32, recent-budget=476, sink=4, preempt-throttle watchdog (cell-only), closed-loop K controller, memory-pressure gate. FA-off prefill → FA-on decode via state-swap. **K=512** for fair comparison with the baselines (not K=1024 as previously proposed). | `--policy v1_fa2 --k-nominal 512 --anchor-top-k 32 --recent-budget 476 --n-sink 4 --cache-type-k q8_0 --cache-type-v f16` | yes |

### Why this set, not the previous "vanilla + v1 + tova + h2o + v1_fa2_stack" set?

- **Adds StreamingLLM** as a published-paper recency control, replacing the unpublished internal `v1` (which is a stepping-stone variant of our own work; keeping it alongside `v1_fa2_stack` would be self-comparison and double-counting). The internal `v1` cells are still run but stored under `figures/internal_ablations/` and not used in the headline table.
- **Adds H2O explicitly as the SOTA baseline** instead of leaving `StreamingLLM` ambiguously labeled. This matches the verification recommendation: "If only ONE baseline is feasible in the time budget, pick H2O".
- **Drops K=1024 for EndurKV** in the headline. K=512 is used for every non-vanilla policy. K=1024 numbers (and K=256, K=384) live in the K-sweep companion subsection on Phi-3 only.

---

## 2. Final benchmarks

```json
{
  "ppl": {
    "dataset": "WikiText-2-RAW-V1 test split (Salesforce/wikitext on HuggingFace, public, MIT/Wikipedia-CC-BY-SA). Materialized from the pinned test.parquet via eval_pipeline/data/build_wiki_chunks.py and split into 9 sequential word-boundary chunks (~2048 tokens each) at /data/local/tmp/endurkv/eval_data/wiki.test.raw.chunk{0..8}. Frozen via Wave-11 manifest (wiki.test.raw.sha256). Provenance: see eval_pipeline/data/PROVENANCE.md. Earlier Wave-11 drafts used the tokenized wikitext-2-v1 with <unk> substitutions — that file has been replaced; PPL numbers from before this fix are not comparable to literature.",
    "sample": "Per (model, policy) cell: prefill chunk i, teacher-force chunk i+1 (KIVI/H2O disjoint-pair protocol). With 9 chunks this yields 8 disjoint (prefill, eval) pairs per cell. Per-cell PPL is the TOKEN-WEIGHTED GEOMETRIC MEAN of per-chunk perplexity (exp(sum_i n_tok_i * mean_nll_i / sum_i n_tok_i)), the conventional WikiText-2 reporting convention used by H2O/KIVI/StreamingLLM/TOVA. 95% bootstrap CI is computed in log domain (per eval_pipeline/score_ppl.py)."
  },
  "accuracy": {
    "dataset": "NIAH synthetic retrieval probe — NOT the canonical Kamradt NIAH. The haystack is a single ~1300-character LLM-written paragraph (FILLER in eval_pipeline/data/build_niah_stimuli.py) repeated to fill the context window. The original WAVE11 draft claimed `Paul Graham essays (public domain)`; that claim was inaccurate and is retracted. See eval_pipeline/data/NIAH_HONESTY.md for the disclosure and the recommended fix (downloading the real Paul Graham essay corpus from Kamradt's reference harness).",
    "sample": "8 stimuli per (model, policy) cell = 4 ctx levels {2048, 4096, 6144, 8192} × 2 depth levels per ctx (a subsampled grid of {0, 25, 50, 75, 87} — the full 32-cell grid is recommended for publication, see NIAH_HONESTY.md). Scoring: case-insensitive substring match for \"sandwich at dolores park\" in gen.txt, guarded by a negation-window check (see score_niah.py rule_based_judge). Optional GPT-4o judge runs when OPENAI_API_KEY is set; agreement statistics are reported alongside both judges. Per-(ctx, depth) pass/fail reported as a heatmap."
  }
}
```

### Why we did NOT switch to RULER

- The verification correctly notes that RULER would be a stronger discriminator at >32k contexts. But Phi-3-mini-128k Q4_K_M on the OnePlus 15 thermal envelope cannot sustain 32k+ prefill in one window, and our other two models (Llama-3.2-1B, Gemma-2-2B) cap at 8k/8k native context. RULER's discriminative regime is above our usable context window.
- Switching the stimuli set mid-protocol would break the "only the device changes" fidelity rule. The NIAH stimuli are already frozen on the phone with the same SHA-256 manifest used in Wave-9/10 phone runs.
- The single-needle saturation problem is real but manageable: we report per-(ctx, depth) cells, and the discriminative power lives in the per-cell heatmap, not in the overall %. H2O/SnapKV-style policies cluster near 100% only on the overall mean; their per-cell scores at depth=0 and depth=87 separate cleanly.
- RULER will be added in Wave-12 with a separately frozen stimulus set.

---

## 3. Launcher update plan (concrete edits to `phone_wave11_eval.sh`)

File: `/home/mislam22/EndurKV_workspace/EndurKV/scripts/android/phone_wave11_eval.sh`

### CHANGE_POLICIES (line 102)
Replace:
```sh
POLICIES=${POLICIES:-"vanilla v1 tova h2o v1_fa2_stack"}
```
with:
```sh
POLICIES=${POLICIES:-"vanilla h2o tova streamingllm v1_fa2_stack"}
```
Rationale: drop internal `v1` (self-comparison), add `streamingllm`. Order is "baselines first, ours last" for log readability.

### CHANGE_K (lines 55-59)
Keep K=512 as-is (already correct). Add an inline comment block above line 55 documenting the K-choice provenance so a reviewer reading the launcher sees that K=512 is intentional:
```sh
# K=512 binds every non-vanilla policy. K=1024 was rejected for the headline
# because Wave-10 K=1024 PPL came from Phi-3 longgen sampling-NLL (suspect
# metric per OPTIMIZATION_JOURNEY.md Problem #10) and cannot be compared to
# wave-11's teacher-forced WikiText-2 PPL. K-sweep {256, 384, 1024} runs as a
# separate Phi-3-only companion sweep, NOT inside this launcher.
K_NOMINAL=${K_NOMINAL:-512}
```

### ADD_PORT_FOR_SOTA (entry in `policy_flags()` at line 205-231)
Add a new `streamingllm)` case between `h2o)` and `v1_fa2_stack)`:
```sh
        streamingllm)
            # StreamingLLM (Xiao et al., ICLR 2024): sink + sliding-window
            # recency mask; no attention readout; FA-on prefill + decode.
            # Implemented in eviction_bench by --policy streamingllm wired to
            # apply_recency_decode_eviction with n_recent = K - n_sink.
            printf -- "--policy streamingllm --k-nominal %s --n-sink %s --recent-budget %s --cache-type-k f16 --cache-type-v f16" \
                "$K_NOMINAL" "$N_SINK" "$(( K_NOMINAL - N_SINK ))"
            ;;
```

### PORT_IN_EVICTION_BENCH (`/home/mislam22/EndurKV_workspace/EndurKV/entropy_probe/eviction_bench.cpp`)
1. Add `"streamingllm"` to the policy allowlist at lines 171-173.
2. Add `policy_streamingllm()` after `policy_pyramid` near line 538 (~30 LoC, mirrors `policy_tova` structure but ignores `attn` and emits `[0, n_sink) ∪ [n_kv - (K_nominal - n_sink), n_kv)` for every layer). Reuses existing `protect_sink` helper at line 314 and the `apply_recency_decode_eviction` mechanic at line 583.
3. Wire dispatch at the 4 sites: prefill (lines 869-944), decode-time eviction (lines 1102-1106 and 1194-1198), and the args-validator.
4. Keep `--decode-bound` enabled by default for `streamingllm` so the cache stays capped during long decodes (same trick as `v1_fa`).

No CMakeLists.txt changes. No new dependencies.

### ADD_K_SWEEP_COMPANION (separate file, NOT the main launcher)
Create `/home/mislam22/EndurKV_workspace/EndurKV/scripts/android/phone_wave11_ksweep.sh` that runs Phi-3 only at `K_NOMINAL ∈ {256, 384, 512, 1024}` for `vanilla` and `v1_fa2_stack` only, on the same WikiText-2 chunks. This produces the K-sweep companion subsection. Do NOT include it in the headline table — the headline table is K=512 across every policy.

### REMOVE_INTERNAL_V1_FROM_HEADLINE
Internal `v1` cells (the FA-off stepping-stone variant of our own work) are retained under `OUT_DIR/<MODEL>/v1/` if `POLICIES` is overridden to include `v1`, but the default `POLICIES` no longer lists it. Internal ablations stored in `figures/internal_ablations/`, not in `figures/master_tables/`.

### NO CHANGE NEEDED
- Watchdog logic (lines 160-198) — still scoped to `v1_fa2_stack` only.
- Cool/mem gates, DVFS pin, sensor sampler — unchanged.
- PPL chunk count (8), NIAH stimulus count (8) — unchanged.
- Per-cell deliverables (`stress.csv`, `sensors.csv`, `iter*/`) — unchanged.

---

## 4. Expected Wave-11 wall time

Per `wave11_cells.json` `total_expected_minutes` for the original 5-policy 240-cell grid: **1395.6 minutes ≈ 23.3 hours** end-to-end, derived from Wave-10 per-cell measurements.

Updated estimate for the **revised policy set**:

- `vanilla` (FA-on): unchanged. 8 PPL chunks × ~9.8 min + 8 NIAH × ~11 min (mean over ctx levels) ≈ 165 min per model.
- `h2o` (FA-off): unchanged. 8 × 7.0 min + 8 × 8.0 min ≈ 120 min per model.
- `tova` (FA-off): unchanged. ≈ 120 min per model.
- `streamingllm` (FA-on, no attention readout): **faster than h2o/tova** because FA-on prefill matches vanilla's path. Estimated at ≈ 8 × 9.5 min + 8 × 10 min ≈ 156 min per model (slightly slower than vanilla due to per-step recency mask, slightly faster than tova/h2o due to no FA-off prefill).
- `v1_fa2_stack` (FA-off prefill, FA-on decode, Q8 K, watchdog): unchanged. ≈ 8 × 7.0 min + 8 × 9.0 min ≈ 128 min per model.

Per model: 165 + 120 + 120 + 156 + 128 = **689 min ≈ 11.5 h**.
Three models (Phi-3-mini-128k, Llama-3.2-1B, Gemma-2-2B): **34.4 h** end-to-end, plus ~10% slack for cool gates and memory waits ≈ **38 h**.

Realistic budget given OnePlus 15 thermal recovery between cells: **38-42 hours**, run as 2 overnight sessions (24h + 14-18h) with the detached `setsid` pattern already in the launcher. The Phi-3 K-sweep companion sweep (separate launcher) adds ~6 h on top.

**Headline estimate: 40 hours for the headline table; +6 hours for the Phi-3 K-sweep companion. Total ≈ 46 hours of phone time across 2-3 thermal sessions.**

---

## 5. Deliverables

The full sweep produces, under `/data/local/tmp/endurkv/logs/wave11_eval_<ts>/`:

### Per-cell raw artifacts (120 cells = 3 models × 5 policies × (8 PPL + 8 NIAH))
- `stress.csv` — one row per chunk/stimulus with `prefill_ms, decode_tps, n_decode_steps, peak_kv_cells, peak_rss_kb, evicted, ppl, niah_correct, k_used, ddr_start_c, mem_free_gb_start`.
- `sensors.csv` — 5 Hz thermal + CPU/GPU frequency sampling.
- `iter*/steps.csv` — per-decode-step timing and KV stats.
- `iter*/meta.json` — per-iter `prefill_ms, decode_tps, n_decode_steps, peak_kv_cells, peak_rss_kb, evicted_total_decode, perplexity, live_cache_cells`.
- `iter*/gen.txt` — NIAH only; host-side judge reads this.
- `iter*/stderr.log`, `iter*/stdout.log` — debugging.
- `watchdog.log` — v1_fa2_stack cells only.

### Host-side aggregations (run after pulling logs)
- `figures/master_tables/table1_quality_and_system.csv` — 5×3 grid of (PPL mean ± stderr, NIAH overall %, decode tps, prefill ms, peak RSS, peak DDR °C, peak skin °C).
- `figures/master_tables/table2_thermal.csv` — peak/mean DDR + skin + battery over the full sweep, per policy.
- `figures/master_tables/TABLE_WAVE11_PPL.md` — rendered PPL table with seq_add-skip caveat and live-cache-vs-K_nominal column.
- `figures/master_tables/TABLE_4POLICY.md` — refreshed 4-policy comparison incorporating Wave-11 numbers (despite the name, now 5 policies).
- `figures/niah_heatmaps/{model}_{policy}.png` — 4×2 NIAH pass/fail heatmaps per (model, policy), showing per-(ctx, depth) cells.
- `figures/pareto/wave11_pareto.png` — quality (PPL or NIAH) × throughput (decode tps) × thermal (peak DDR °C) Pareto front.
- `figures/internal_ablations/v1_*.csv` — internal `v1` cells (only if `POLICIES` was overridden to include it).
- `figures/master_tables/SUBSECTION_KSWEEP.md` — refreshed K-sweep companion section using ONLY the Phi-3 K-sweep companion sweep (no more wave-9/wave-10 splicing). K=512 is the headline; K=256/384/1024 are companion rows.

### Statistical robustness
- PPL: 8 chunks × 3 models × 5 policies = 120 PPL measurements per policy slot (across models). Report mean ± stderr per (model, policy).
- NIAH: 8 stimuli × 3 models × 5 policies = 120 binary trials per policy slot. Report (overall %, per-ctx %, per-depth %, 4×2 heatmap).

### Reproducibility manifest
- `figures/master_tables/WAVE11_FINAL_SPEC.md` (this file) — the binding spec.
- `eval_pipeline/wave11_cells.json` — updated to drop `v1`, add `streamingllm`. SHA-256 stamped.
- `eval_pipeline/data/niah/MANIFEST.txt` — frozen NIAH stimulus hashes (unchanged from Wave-9/10).
- `eval_data/wiki.test.raw.chunk{0..7}.sha256` — frozen WikiText-2 chunk hashes (unchanged from Wave-9/10).

---

## 6. Acceptance criteria for the Wave-11 headline

A Wave-11 sweep is accepted as the headline if:
1. All 120 cells complete with non-sentinel `stress.csv` rows.
2. Per-cell live cache (`peak_kv_cells` for eviction policies) is reported alongside `K_NOMINAL` — addressing the "live cache != K_nominal" concern in the K-choice verification.
3. PPL is teacher-forced WikiText-2 (not sampling-NLL); the headline PPL is the arithmetic mean of 8 chunks.
4. NIAH is scored by deterministic substring match; per-(ctx, depth) heatmaps are produced.
5. Headline table reports K=512 for every non-vanilla policy. K=1024 numbers appear only in `SUBSECTION_KSWEEP.md` and are labeled "K-sweep companion (Phi-3 only)".
6. The paper text describes StreamingLLM as a "recency-only floor / mobile-friendly control" and H2O as the "attention-aware SOTA baseline". StreamingLLM is NOT labeled "SOTA".
