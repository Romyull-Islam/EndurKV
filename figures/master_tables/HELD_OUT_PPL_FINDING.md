# Held-out PPL Reveals the True Cost of Attention-Based KV Eviction

## 1. Framing: sampling-NLL is not held-out perplexity

Waves 3 through 9 of this project ranked KV-eviction policies by what we called
"perplexity," but that label was a misnomer. The number being computed was the
mean negative log-likelihood of the model's own greedy decode under a fixed
sampling prompt — a sampling-NLL of self-output, not a teacher-forced
likelihood over held-out text. Because greedy decoding under any reasonably
calibrated language model concentrates probability mass on the token it just
emitted, sampling-NLL behaves like an internal consistency score: a policy that
makes the model repeat itself confidently will score low, regardless of whether
the underlying language model is fluent or has lost long-range context. This is
the well-known reason llama.cpp ships a separate `llama-perplexity` tool that
operates in teacher-forcing mode over a reference corpus.

The Wave-11 protocol was specifically designed to retire this metric and
replace it with WikiText-2 teacher-forced perplexity — the standard intrinsic
benchmark used by H2O (NeurIPS 2023), KVQuant (NeurIPS 2024), KIVI (ICML 2024),
and AdaKV (NeurIPS 2024). Eight non-overlapping 2048-token chunks of
`wiki.test.raw` provide 16,384 scored tokens per cell, with 95 % bootstrap
confidence intervals over chunks. The methodology, as written in
`EVAL_PROTOCOL_WAVE11.md`, is unambiguous: *prefill 2048 tokens, apply the
policy's eviction, teacher-force the NEXT 2048 tokens.* This is the held-out
continuation regime that makes a quality difference between eviction policies
visible, because the tokens being scored are precisely those the policy never
saw during prefill.

## 2. A first-pass smoke run, and a methodology bug it surfaced

The first Wave-11 smoke probe on Llama-3.2-1B chunk 0 produced perplexities
that, at face value, would rewrite this dissertation's narrative:

| Policy           | Wave-11 smoke PPL (Llama-1B, chunk 0) |
|------------------|---------------------------------------|
| vanilla          | 1.02                                  |
| StreamingLLM     | 8.24                                  |
| v1\_fa2\_stack (EndurKV) | 11.44                         |
| H2O              | 158.4                                 |
| TOVA             | 181.5                                 |
| v1 (no FA²)      | 180.5                                 |

A vanilla perplexity of 1.02 on natural text is impossible — even a perfectly
calibrated 1-billion parameter model on WikiText-2 sits comfortably above 6.
That single number, more than any policy comparison, is the diagnostic. An
adversarial methodology audit traced the cause: the launcher script
`scripts/android/phone_wave11_eval.sh` passes the *same* file path to both
`--prompt` and `--eval-text`. The eviction\_bench binary itself is correct, but
because prompt and eval-text reference identical token streams, the runtime is
actually performing a self-recall test, not a held-out continuation. Vanilla
trivially recalls the chunk it just ingested (hence PPL 1.02);
StreamingLLM, whose sink + recency window happens to retain the tail of the
prefill exactly where the eval-text begins, also benefits; attention-based
policies that aggressively evicted prefill positions are then asked to score
the ground-truth tokens for the very positions they dropped, and explode.

These numbers therefore do not refute the published H2O / TOVA claims, do not
demote EndurKV's v1\_FA²-stack, and do not crown StreamingLLM. They measure a
bug — specifically, a different question than the one the protocol intends to
ask. They will not appear in the dissertation's results chapter.

## 3. What the smoke numbers *do* tell us before the rerun

Two structural observations are nonetheless real and survive the bug fix.

First, the *direction* of the smoke ranking is consistent with a known
StreamingLLM-class pathology rather than with random noise. Xiao et al. (2023)
documented that "perplexity skyrockets the moment initial tokens are evicted."
Our smoke run shows exactly that signature: the only non-vanilla policy that
explicitly preserves the first four sink tokens (StreamingLLM) stays in the
single digits, while the three attention-only variants (v1, H2O, TOVA)
explode into the 100s. Even with the prompt-equals-eval bug confounding the
absolute numbers, the *relative* ordering is a free piece of diagnostic
evidence: it suggests that as implemented in this codebase, the v1 / H2O / TOVA
ports may not be reliably retaining attention sinks. The audit (agent D)
flagged this independently: the original H2O paper keeps both a heavy-hitter
ratio *and* a recent-token ratio by construction, and our port may be the
pure-attention variant — algorithmically closer to TOVA's "Window" policy than
to canonical H2O. The Wave-11 rerun will need to verify sink retention as
part of the implementation correctness check, not just rerun the harness.

Second, EndurKV's `v1_fa2_stack` sits between StreamingLLM and the
attention-only variants in the smoke ranking. This is consistent with its
construction: it carries a top-32 attention anchor *and* a recency tier *and*
four sink tokens, so it should behave more like StreamingLLM than like pure
H2O. That position is what the rerun must confirm.

## 4. StreamingLLM may genuinely be a strong PPL baseline — and that is fine

The Wave-11 protocol originally demoted StreamingLLM to a "recency-only floor."
The smoke results plus the literature counter-argument from the audit team
(agent D) suggest this demotion was premature. StreamingLLM's combination of
attention sinks plus a recent-token window is *exactly* the mechanism the
published H2O implementation builds on, and it is the simplest configuration
that avoids the catastrophic Window-policy failure mode reported in TOVA
Table 2 (PPL 4812 at K=64). If, after the methodology fix, StreamingLLM still
beats EndurKV on held-out WikiText-2 PPL, that result will be reported
honestly. EndurKV's contribution does not depend on winning the eviction-quality
benchmark, and treating it as if it does would be exactly the kind of
metric-cherry-picking that the Wave-11 protocol was designed to prevent.

## 5. The residual EndurKV contribution: thermal and endurance control

The eviction-quality story is one of two independent axes EndurKV measures.
The other axis — thermal load and storage endurance under sustained
on-device decode — is unaffected by the PPL methodology bug, because it is
measured from sysfs sensors and kernel counters, not from token logits.

The Wave-9 v1\_FA²-stack measurements stand:

- Peak DDR temperature 64.1 °C vs Wave-8's 72.9 °C, a reduction of 8.8 °C
  under matched mem-gate, cool-down, and DVFS conditions.
- Peak CPU package temperature 66.8 °C vs 78.4 °C, a reduction of 11.6 °C.
- Zero kernel-forced 883 MHz throttle events, compared to one in Wave-8.
- Zero MB UFS swap-out during sustained decode, compared to four-figure
  megabyte swap volumes in Waves 5 and 7.
- Resident set 370 MB below the matched vanilla baseline at peak.

These numbers describe a mobile-LLM execution stack that runs cooler, runs
longer before throttling, and spares the device's flash write endurance. They
are the result of a *composition* — Q8 K-cache, FA²-stack switching,
preemptive DDR-driven CPU watchdog, RSS-aware mem-gate, sustained cool-down
— not of any single eviction-quality trick. None of these mechanisms requires
that v1's attention-based eviction win on PPL; they require only that the
overall stack keep PPL within an honest bound of the vanilla baseline.

The natural reframe, which the dissertation will adopt, is to present EndurKV
as a *thermal and endurance control composition* layered on top of whichever
eviction policy is empirically best on held-out PPL. If StreamingLLM is the
PPL winner after the rerun, the Wave-12 stack is StreamingLLM eviction plus
the EndurKV control loop. The control loop is the contribution; the eviction
choice is a parameter.

## 6. What the literature counter-arguments actually predict

The Phase-1 literature audit (agent D) collected six counter-arguments
against the naive reading of the smoke results, each of which is testable
in the Wave-11 rerun:

1. *Implementation gap.* Published H2O retains a recent-token ratio by
   construction; our v1/H2O port may not. **Test:** instrument the eviction
   trace to verify that positions 0–3 and the last 64 positions are present
   in the post-eviction cache for every policy that claims to keep them.
2. *Sink elision.* Even when sinks are nominally preserved, attention scoring
   may rarely select positions 0–3 because their normalized attention is
   diluted by recent activity. **Test:** log the per-step retained-position
   histogram and check for sink survival across decode.
3. *Evaluation-protocol mismatch.* TOVA's published PPL is over a single
   4096-token chunk where most scored tokens are early-sequence (eviction
   barely matters); our protocol scores only post-eviction tokens.
   **Test:** report PPL split by token position within the held-out chunk.
4. *SnapKV was never PPL-evaluated.* SnapKV's published results are LongBench
   / NIAH only, not perplexity. **Test:** keep NIAH as a parallel benchmark
   so SnapKV-style policies have a fair venue.
5. *Model-size mismatch.* Published baselines used 7B–70B; we use 1B.
   Attention-sink density scales with model size. **Test:** the Wave-11
   Tier-3 plan adds Phi-3-mini (3.8B), which sits between Llama-3.2-1B and
   Llama-2-7B on the sink-density curve.
6. *Aggressive K is not the cause.* K=512 at prompt=2048 (25 %) is not
   extreme by literature standards. **Test:** the Wave-10 K-sweep already
   ran K ∈ {256, 384, 1024}; the Wave-11 K-sweep on Tier-2 will
   cross-validate on the methodologically corrected harness.

## 7. What the corrected Wave-11 will resolve

The methodology fix is mechanical: change the launcher so `--prompt` points to
chunk *i* and `--eval-text` points to chunk *i+1*, giving seven valid
measurements per cell. The interim plotter described in
`eval_pipeline/wave11_interim_plot.py` is already in place and idempotent; it
will render the corrected numbers as soon as the first cells finish.

After that single-line fix, Wave-11 will produce, on Llama-3.2-1B and
Phi-3-mini-128k, across vanilla / v1 / TOVA / H2O / StreamingLLM /
v1\_FA²-stack at K=512 (Tier 1) and K ∈ {256, 384, 1024} (Tier 2), an
honest held-out PPL table with bootstrap 95 % CIs, paired NIAH accuracies,
and matched thermal / endurance traces. The three questions the corrected
table must answer are: (a) Is EndurKV's v1\_FA²-stack within an honest PPL
bound of vanilla? (b) Does StreamingLLM in fact dominate on PPL alone, as
the smoke ranking direction suggests? (c) Does EndurKV retain its thermal
and endurance advantage even when its eviction policy is swapped for the
empirical PPL winner? The dissertation's claims will follow whichever
answers the corrected data return.
