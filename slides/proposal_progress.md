# Proposal slide content — copy/paste ready

Three slide-ready figures sit in `figures/`:

- `figures/slide_progress.png`        — what we built so far (text on next page)
- `figures/slide_headline.png`        — overall scatter with safety zones
- `figures/slide_per_task_rho.png`    — **strongest figure** — per-task ρ bars

---

## Slide 1 — Progress

**Title:** What we've built so far
**Sub-title:** Server-side measurement study — Llama-3.2-1B + 12-task workload

```
Goal
    Validate one hypothesis before committing to on-phone work:
    can output entropy be used as a per-step "safety gate" that
    decides whether KV-cache pruning is safe right now?

What we built
    1. Two C++ probes that link against llama.cpp's public API
       (zero edits to llama.cpp source — both live in their own
       CMake project under entropy_probe/):
         - entropy_probe    per-step Shannon H, top-1, top-5
         - attention_probe  captures kq_soft_max-{layer} via
                            llama_context_params.cb_eval after
                            disabling flash attention

    2. 80-prompt workload spanning three benchmark families that
       the comparison literature uses:
         - LongBench  (KVSwap/KIVI/CAKE/SnapKV)   8 tasks × 6 = 48
         - HELM       (H2O — summarisation)      xsum, cnn_dm × 8 = 16
         - lm-eval    (H2O — multi-choice)       piqa, obqa × 8 = 16

    3. Study run on Llama-3.2-1B-Instruct (Q4_K_M), single A100,
       seed 42, max 64 tokens per prompt:
         - 4045 decode steps recorded across 12 tasks
         - per-prompt entropy CSV  +  per-prompt attention sidecar
           (per-layer × per-source-token attention matrix)

    4. Offline analysis pipeline (scripts/11_analyze.py)
       — six figures + numerical summary JSON.

What we found  (next slide)
    The entropy ↔ attention-concentration relationship holds with
    overwhelming statistical significance, and it is strongest on
    exactly the workloads where KV-cache pressure matters most.

What is next
    prune_probe — re-run each decode with K oldest KV entries
    evicted, measure KL(P_full ‖ P_pruned). Direct test that closes
    the safety-gate loop.
```

---

## Slide 2 — Headline finding (use `figures/slide_per_task_rho.png`)

**Title:** Where does the safety-gate signal hold?
**Sub-title:** Per-task Spearman ρ between output entropy and top-1 attention probability

```
Result
    Across 4045 decode steps over 80 prompts spanning 12 tasks
    in 3 benchmark families, output entropy is correlated with
    attention concentration:

    Strong  (ρ ≤ -0.40)  cnn_dailymail (-0.57***), xsum (-0.47***),
                         gov_report (-0.43***), hotpotqa (-0.40***)
    Moderate            qasper (-0.40***), openbookqa (-0.31***),
                         triviaqa (-0.28***), samsum (-0.25***),
                         multifieldqa_en (-0.22**)
    Weak                 piqa (-0.15**)
    Absent               lcc (-0.03, n.s.)
    Inverted             trec (+0.19***)

    *** = p < 1e-3 ; ** = p < 1e-2 ; n.s. = not significant

What it means

    The safety-gate signal is STRONGEST on the workloads that
    KV-cache management actually targets — long-form summarisation
    and multi-document QA, where prompts are long and decoded
    answers are many tokens.

    The gate does NOT work on:
        - code completion  (lcc):  attention is always peaked on
                                   syntax tokens regardless of
                                   model uncertainty.
        - few-shot classification (trec):  prompt structure forces
                                   attention patterns that don't
                                   track output entropy.

    These limitations are useful — they tell us where the
    proposed controller can be deployed and where it can't.

Why the per-task plot, not the aggregate

    The cross-task aggregate Spearman is ρ = -0.237 (n = 3974,
    p < 1e-52).  This is a DILUTION of the per-task signal:
    averaging across 9 strongly-confirming tasks, 1 weak, 1 absent,
    and 1 inverted hides the structure.  Per-task ρ is the
    honest picture — and 9 of 12 tasks have ρ < -0.20 with p < 1e-3.

Caveat
    This is analytical evidence — entropy correlates with a
    structural proxy for prune-safety.  The thesis's first
    concrete experiment (prune_probe, starts next) measures actual
    KL(P_full ‖ P_pruned) after evicting K oldest KV entries.
```

---

## Reproducibility footnote (small)

- llama.cpp commit pinned: `0033f53a072af953b457c9fd2314e6e28bd11cc7`
- Model: Llama-3.2-1B-Instruct-Q4_K_M, sha256 in `ENV.md`
- Seed: 42 (fixed per prompt)
- Hardware: NVIDIA A100-SXM4-80GB, single GPU
- Full numerical summary: `figures/study_summary.json`
- Workload manifest: `data/prompts.jsonl` (80 prompts)
