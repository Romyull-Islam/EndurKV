# Supplementary Materials — EndurKV

**Companion document to the main paper.** This appendix collects reproducibility
artifacts, the full per-cell measurement tables, failed-experiment write-ups, the
methodology rationale for the held-out perplexity protocol, the watchdog tier
policy, an honest enumeration of the study's limitations, and a pointer to the
code release. The goal is to make every number in the main paper independently
re-derivable from the raw on-device logs and to document the negative results
that shaped the final design.

---

## S1. Reproducibility

### S1.1 Repository state and MANIFEST.sha256

The authoritative manifest of every artifact required to reproduce Wave-11
results lives at `eval_pipeline/MANIFEST.sha256`. It pins, by sha256:

- **GGUF models on device** (Section 1):
  Llama-3.2-1B-Instruct Q4_K_M (`6f85a640…df83`),
  Phi-3-mini-128k-Instruct Q4_K_M (`dde15f81…6cc3`),
  gemma-2-2b-it Q4_K_M (`e0aee850…7787`). All three are placed at
  `/data/local/tmp/endurkv/models/` on the phone.
- **On-device binaries** built via `scripts/android/build_probe_android.sh`
  (Section 2): `eviction_bench`, `attention_probe`, `controller_probe`,
  `entropy_probe`, `prune_probe`. The `eviction_bench` binary listed in the
  manifest is the *pre-fix* build; the Wave-11 source changes (canonical H2O
  recent+heavy split, cumulative prefill attention accumulator, paired-test
  scoring) require a rebuild and a manifest update before publication freeze.
- **WikiText-2-raw-v1 chunked corpus** (Section 3): nine 2048-token chunks
  derived from the canonical `wikitext-2-raw-v1/test` split (parquet sha256
  `5f1bea06…6d91`), built by `eval_pipeline/data/build_wiki_chunks.py`. The
  prior tokenized `wikitext-2-v1` corpus was replaced on 2026-06-07 because
  its `<unk>` substitutions made PPL non-comparable to literature.
- **NIAH stimuli** (Section 4): 32 deterministic prompts at 4 context lengths
  ×8 depth percentiles. These are **synthetic repeated-paragraph** stimuli,
  *not* the canonical Paul Graham haystack — see Section S6 for the disclosure.
- **Host pipeline scripts** (Section 5): `score_ppl.py`, `score_niah.py`,
  `score_sig.py`, `wave11_cells.json`, `wave11_interim_plot.py`,
  `build_niah_stimuli.py`.

The MANIFEST header records repo HEAD `07dca629…07c1`. Section 6 of the
manifest documents the freeze procedure (commit working-tree modifications,
add llama.cpp as a real submodule pinning an upstream ggerganov SHA, rebuild
binaries, re-checksum data files) that must be executed before the file is
treated as authoritative for an external auditor.

### S1.2 Model versions

| Model | Quantization | Source | Role |
|---|---|---|---|
| Llama-3.2-1B-Instruct | Q4_K_M GGUF | `bartowski/Llama-3.2-1B-Instruct-GGUF` (HF) | Wave-3 long-prompt + Wave-11 smoke |
| Phi-3-mini-128k-Instruct | Q4_K_M GGUF | Microsoft via HF | Primary (Waves 3–11) |
| gemma-2-2b-it | Q4_K_M GGUF | google via HF | Wave-11 tertiary (optional) |

### S1.3 OS, kernel, ROM build

- **Phone:** OnePlus 15 (model `CPH2749`).
- **Android release:** 16, build id `BP2A.250605.015`.
- **Fingerprint:** `OnePlus/CPH2749/OP611FL1:16/BP2A.250605.015/B.R4T3.2da5ed5-a4bd5e-a4bd63:user/release-keys`.
- **Kernel:** `6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k` (built 2026-01-06).
- **DVFS:** `cpu0` governor `performance`, `cpu7` governor `walt` (mixed). Every
  Wave-9+ run pins all cores to `performance` via
  `scripts/android/pin_dvfs.sh` and re-reads the post-pin governor into
  per-run metadata.
- **Sampling:** `--seed 42 --greedy` for all eviction_bench cells; bootstrap
  reseed in `score_ppl.py` is `0 XOR hash((model,policy)) & 0xFFFFFFFF`.
- **Host environment** (analysis/build): Ubuntu 24.04, kernel 6.8.0-101,
  gcc 13.3.0, CUDA 13.1.80, Python 3.12.3. See `ENV.md` for full toolchain.

---

## S2. Full per-cell tables — pointers

The complete per-wave evidence is too large for a single appendix table. The
canonical entry point is
[`ALL_WAVES_MASTER.md`](ALL_WAVES_MASTER.md), which lists every Wave-3..Wave-11
cell on one row with PPL, peak DDR, peak CPU, swap-out, and watchdog state.
Per-wave detail tables live alongside it:

- [`TABLE_PHI3_WAVE3.md`](TABLE_PHI3_WAVE3.md) — Phi-3 narrativeqa, 4 policies.
- [`TABLE_PHI3_WAVE4_LONGDECODE.md`](TABLE_PHI3_WAVE4_LONGDECODE.md) — short-prompt
  + 2048-token decode, the smoking-gun experiment that established cache size
  as the binding thermal lever.
- [`TABLE_4POLICY.md`](TABLE_4POLICY.md) and
  [`TABLE_ALL_POLICIES.md`](TABLE_ALL_POLICIES.md) — cross-policy summary at
  Waves 3–9.
- [`SUBSECTION_KSWEEP.md`](SUBSECTION_KSWEEP.md) and
  [`WAVE10_MASS_RETAINED.md`](WAVE10_MASS_RETAINED.md) — the K∈{256, 384, 512,
  1024} sweep that validated the cache-size-vs-PPL trade.
- [`TABLE_WAVE11_PPL.md`](TABLE_WAVE11_PPL.md),
  [`TABLE_WAVE11_NIAH.md`](TABLE_WAVE11_NIAH.md), and
  [`TABLE_WAVE11_SIGNIFICANCE.md`](TABLE_WAVE11_SIGNIFICANCE.md) — Wave-11
  held-out PPL, NIAH accuracy, and paired-test significance.
- [`LATENCY_BREAKDOWN.md`](LATENCY_BREAKDOWN.md) and
  [`HELD_OUT_PPL_FINDING.md`](HELD_OUT_PPL_FINDING.md) — per-phase latency and
  the Wave-3..10 sampling-NLL post-mortem.

A consolidated chapter narrative is in
[`CHAPTER_RESULTS.md`](CHAPTER_RESULTS.md); the supervisor-facing summary is
[`SUPERVISOR_EVIDENCE_PACKAGE.md`](SUPERVISOR_EVIDENCE_PACKAGE.md).

---

## S3. Failed experiments

### S3.1 Wave-5 file-backed state swap

**Hypothesis.** Replace v1_FA's 3.7 GB heap-resident state buffer with
`llama_state_seq_save_file` to disk, freeing the FA-off context before
allocating the FA-on context. The state file would live in the kernel page
cache and be evictable without anonymous-page swap pressure.

**Result.** UFS swap-out *increased* from 515 MB (Wave-3 heap-buffer
baseline) to 739 MB — a 43 % regression. The kernel was forced to evict
739 MB of *other* processes' anonymous pages to make room for the file's
page cache; the file write itself constituted a 3.7 GB one-shot UFS write
per state-swap event.

**Decision.** Reverted. Documented in
[`MEMORY_SWAP_DISCUSSION.md`](MEMORY_SWAP_DISCUSSION.md) §6c. The correct
fix is per-layer streaming state transfer (`get_layer_k` / `set_layer_k`),
which would drop peak RSS to ~7.1 GB and eliminate the spillover entirely;
this requires roughly 100 lines of llama.cpp patching to expose per-layer
state accessors and is filed as future work.

### S3.2 Wave-7 over-anchoring

**Hypothesis.** A "smart" v1_FA² variant that anchors the full prompt
(all 272 tokens) plus a 256-token recency window would preserve narrative
context and beat the recency-only Wave-6 cell on PPL.

**Result.** Sampling-NLL PPL rose from 3.20 (Wave-6) to 3.92 (Wave-7), and
swap-out exploded from 28 MB to 1552 MB. Generated text repeated
"Executive Summary" three times — direct evidence of context collapse from
anchoring competing with recent decode for the K budget. The 1552 MB
swap-out was traced to a launcher race: only 1 s of cool-down left
MemAvailable at 1.8 GB at iter 1, well below the threshold the snapkv
state-swap requires.

**Decision.** Two changes shipped into Wave-8 and beyond: (a) **selective
anchoring** — after spread-gate prefill eviction, keep only the top-32
prompt tokens by mean attention score, freeing the residual budget for
recency; (b) **memory gate** — the launcher blocks iter 1 until
MemAvailable ≥ 4 GB. Both are documented in
[`OPTIMIZATION_JOURNEY.md`](OPTIMIZATION_JOURNEY.md) Problems #5–#6.

A third reverted attempt (V-cache q8_0 quantization) is noted in
`OPTIMIZATION_JOURNEY.md`'s reverted-experiments table: the FA-off→FA-on
state-swap currently assumes f16-V on both sides, and switching V to q8_0
broke the layout transposition.

---

## S4. Methodology — chunk-pair held-out PPL

The Wave-3..10 figures originally reported "perplexity" computed by
sampling the model's own greedy continuation from a fixed prompt, taking
the mean per-token NLL of that continuation under the model itself, and
exponentiating. This is a *sampling-NLL of self-output*, not a
teacher-forced log-likelihood of held-out text. It is biased low for two
structural reasons. First, a calibrated language model is by construction
unsurprised by its own greedy token: the argmax token has near-1
probability under the very distribution it was drawn from. Second, the
metric measures internal self-consistency of the policy + sampler, not
quality against any reference distribution; a policy that makes the model
repeat itself confidently scores low regardless of fluency or long-range
recall.

The corrected Wave-11 protocol uses **disjoint chunk-pair held-out PPL** on
`wiki.test.raw` (the canonical wikitext-2-raw-v1 split, not the
`<unk>`-substituted v1 variant). The recipe, implemented in
`eviction_bench --eval-mode ppl`:

1. Tokenize the corpus into nine non-overlapping 2048-token chunks.
2. For chunk pair `(i, i+1)`, prefill on chunk `i`, apply the policy's
   eviction, then teacher-force chunk `i+1` and accumulate per-token NLL.
3. Report `exp(mean_nll)` per chunk pair; per-cell mean is the
   token-weighted geometric mean over chunk pairs, with a 1000-sample
   percentile bootstrap 95 % CI over per-chunk NLLs.

This matches the H2O (NeurIPS 2023), KIVI (ICML 2024), StreamingLLM
(ICLR 2024) and TOVA (ACL 2024) convention and is directly comparable to
literature numbers. The full bug post-mortem — including the smoke-run
where vanilla scored a physically impossible PPL of 1.02 because
`--prompt` and `--eval-text` pointed at the same file — is in
[`HELD_OUT_PPL_FINDING.md`](HELD_OUT_PPL_FINDING.md).

Sampling-NLL was rejected for two further reasons specific to this study.
The Wave-3..10 cells used different greedy sampling seeds across policies
when the launcher was re-invoked, introducing an uncontrolled confound;
and the metric provided no defensible way to compare against published
H2O / TOVA / StreamingLLM PPL baselines, foreclosing the head-to-head
comparison the main paper requires.

---

## S5. Watchdog tier policy

The preempt-throttle watchdog (`scripts/android/preempt_throttle_watchdog.sh`)
runs as a root background process at 2 Hz. It reads the DDR thermal zone
(`/sys/class/thermal/thermal_zone47/temp`) and writes
`scaling_max_freq` on cpu6 and cpu7 to one of four tiers. Hysteresis bands
are 5 °C on the up-transition and 3 °C on the down-transition.

| Tier | Label | DDR ↑ threshold (°C) | DDR ↓ threshold (°C) | `scaling_max_freq` (kHz) |
|---:|:--|---:|---:|---:|
| 0 | MAX  | —    | —    | 1 632 000 |
| 1 | HIGH | ≥ 58 | ≤ 55 | 1 497 600 |
| 2 | MED  | ≥ 62 | ≤ 59 | 1 267 200 |
| 3 | LOW  | ≥ 65 | ≤ 62 | 1 017 600 |

The kernel's own thermal mitigation framework drops cpu6/cpu7 to ~883 MHz
once DDR crosses approximately 65 °C; the watchdog's purpose is to
preemptively traverse the tier ladder *before* that drop fires, producing
a smooth glide instead of a throughput cliff. Wave-9 logged 16 HIGH-tier
engagements, 6 MED-tier engagements, and 0 LOW-tier engagements over the
56-minute cell, and zero kernel-forced 883 MHz events. On exit the
watchdog restores cpu6/cpu7 to `F_MAX`.

A known limitation: the watchdog cannot guarantee that Qualcomm RPMh will
not re-override the userspace cpufreq vote; we run a re-read asserter
inside the watchdog loop but cannot fully police the kernel's contested
write path on retail firmware. The applied watchdog state per run is not
yet captured in the MANIFEST and is listed in the manifest's WARNING
section (item 6) as an outstanding logging gap.

---

## S6. Limitations

**Synthetic NIAH haystack.** The Needle-in-a-Haystack stimuli shipped in
`eval_pipeline/data/niah/` are *not* the canonical Kamradt harness. Each
prompt is built by concatenating copies of a single LLM-written 1300-char
`FILLER` paragraph until the target length, then injecting the needle at
the requested depth. This makes the needle trivially detectable for any
policy that retains one copy of a unique token, inflates absolute accuracy
relative to literature NIAH, and biases recency-window policies favorably
where they overlap with the needle. The retraction and recommended fix
(rebuild from Paul Graham essays as the canonical harness does) are in
[`eval_pipeline/data/NIAH_HONESTY.md`](../../eval_pipeline/data/NIAH_HONESTY.md).
Until that rebuild, every NIAH number in the paper is labeled
**synthetic repeated-paragraph retrieval probe**, not "NIAH", and direct
comparison to literature NIAH baselines is avoided. The depth grid also
omits 100 % (the position where recency baselines trivially pass), and
the char→token conversion introduces ±10 % drift on advertised context
lengths.

**Sample size (n = 8 chunks).** The PPL bootstrap is over only eight
chunk pairs per cell, yielding wide 95 % CIs (typically ±0.3 PPL at the
Phi-3 vanilla operating point). Distinguishing policies that differ by
less than the CI half-width requires either more chunks or paired tests
on shared chunks; we report McNemar paired NIAH accuracies and per-chunk
paired PPL deltas in [`TABLE_WAVE11_SIGNIFICANCE.md`](TABLE_WAVE11_SIGNIFICANCE.md)
to extract more signal from the small sample. NIAH uses 32 trials per
cell (4 contexts × 8 depths), and the per-depth heatmap is necessarily
coarse.

**Single-seed runs.** Headline numbers are at seed 1337. The protocol
budgets secondary seeds 2718 and 42 if phone time permits, but the
results chapter is single-seed; we therefore do not claim statistical
control over sampling stochasticity at the per-cell level. Greedy
decoding (`--greedy`) removes most of that stochasticity, but state-swap
timing and watchdog tier-transition latency remain potential per-run
sources of variance.

**Single device, single ROM.** All measurements are on one OnePlus 15
unit (build `BP2A.250605.015`). Cross-device generalization,
firmware-update stability of the kernel thermal trip points, and
silicon-lottery variation are not characterized.

**Vendored llama.cpp.** The `llama.cpp/` directory is a vendored copy
rather than a proper submodule. The recorded build-info
`LLAMA_COMMIT=07dca62` is the parent repo's SHA, *not* an upstream
ggerganov commit. The freeze procedure in MANIFEST §6 adds a real
submodule pinning before publication.

---

## S7. Code availability

All source code, on-device launchers, evaluation pipeline, host plotting
scripts, raw `phone-logs/`, and the figures and tables in this directory
are released in the repository accompanying this paper. The entry points
are:

- `eval_pipeline/MANIFEST.sha256` — pinned artifacts (this appendix S1).
- `scripts/android/` — phone launchers, DVFS pin, watchdog, build helpers.
- `entropy_probe/` — eviction_bench and four diagnostic probes.
- `eval_pipeline/` — PPL/NIAH/significance scoring and interim plotting.
- `figures/master_tables/` — every table referenced in S2 above.
- `phone-logs/wave{3..11}_*/` — raw per-cell `stress.csv`, `sensors.csv`,
  and `iter*/meta.json`.

To reproduce the headline numbers end-to-end on a new device, follow the
freeze procedure in MANIFEST §6, then re-run `scripts/android/run_wave11.sh`
with the watchdog enabled and the memory gate honored.
