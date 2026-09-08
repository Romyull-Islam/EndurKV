# Wave-11 Phi-3-mini-128k — Paired PPL Significance Tests

**Source cells:** `phone-logs/wave11_eval_1780862534/Phi-3-mini-128k/{vanilla,h2o,tova,v1_fa2_stack}/ppl/iter*/meta.json`

**Pairing key:** `prompt_id` (per-chunk identifier shared across policies). Deltas are computed only on chunks where BOTH policies have a finite `mean_nll`.

**Tests per pair (A vs B), Δ = log(PPL_A) − log(PPL_B):**
- Paired Student t on the n deltas (df = n−1), two-sided, exact via regularized incomplete-beta.
- Wilcoxon signed-rank, two-sided. Exact enumeration of the null permutation distribution over the doubled (tie-averaged) ranks for n ≤ 25; otherwise normal approximation with tie + continuity correction.
- Cohen's d_z = mean(Δ) / sd(Δ) on the paired deltas.

**Multiplicity:** Holm–Bonferroni over the 6 pairs (FWER ≤ 0.05). Reported separately for t and Wilcoxon p-values.

## Cell sizes (chunks with finite mean_nll)

| Policy | n_chunks |
|---|---:|
| vanilla | 8 |
| h2o | 7 |
| tova | 7 |
| v1_fa2_stack | 8 |

## Per-pair test statistics

| A | B | n | mean Δlog(PPL) | t | df | t_p | Holm t_p | W+ | W_p | Holm W_p | Cohen d_z |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h2o | vanilla | 7 | +0.0354 | +1.487 | 6 | 0.1877 | 0.1877 | 22.0 | 0.2188 | 0.3125 | +0.562 |
| tova | vanilla | 7 | +0.0625 | +2.016 | 6 | 0.09043 | 0.1809 | 23.0 | 0.1562 | 0.3125 | +0.762 |
| v1_fa2_stack | vanilla | 8 | +0.1201 | +3.928 | 7 | 0.005689 | 0.02275 | 35.0 | 0.01562 | 0.09375 | +1.389 |
| h2o | tova | 7 | -0.0272 | -3.498 | 6 | 0.01285 | 0.03855 | 0.0 | 0.01562 | 0.09375 | -1.322 |
| h2o | v1_fa2_stack | 7 | -0.0871 | -5.702 | 6 | 0.001258 | 0.006288 | 0.0 | 0.01562 | 0.09375 | -2.155 |
| tova | v1_fa2_stack | 7 | -0.0600 | -6.053 | 6 | 0.0009212 | 0.005527 | 0.0 | 0.01562 | 0.09375 | -2.288 |

## Significance verdict (α = 0.05, after Holm correction)

**Power note on Wilcoxon at small n.** With n ≤ 8 chunks and a family of 6 tests, the smallest two-sided Wilcoxon p the data can produce is 2/2^n: 0.0156 at n=7, 0.0078 at n=8. After Holm × 6 the floor is ~0.094 at n=7 and ~0.047 at n=8. Wilcoxon is therefore underpowered to *reject* Holm-corrected at n=7 no matter how strong the signal — the test is reported for completeness but the **paired-t is the primary inferential test** because log(PPL) deltas are approximately normal (n=7-8 cells, central limit support; t is robust to mild deviation).

**Decision rule.** Primary: paired-t Holm-adjusted p-value < 0.05 ⇒ **significant**; otherwise **indistinguishable** at α = 0.05. The Wilcoxon Holm column is shown as a non-parametric sanity check; values flagged `floor` are at or below the n-induced detection floor and should not be interpreted as evidence of no effect.

| A | B | Holm t_p | Holm W_p | Verdict (paired-t) | Wilcoxon sanity |
|---|---|---:|---:|---|---|
| h2o | vanilla | 0.1877 | 0.3125 | **indistinguishable** | ns |
| tova | vanilla | 0.1809 | 0.3125 | **indistinguishable** | ns |
| v1_fa2_stack | vanilla | 0.02275 | 0.09375 | **SIGNIFICANT** | ns |
| h2o | tova | 0.03855 | 0.09375 | **SIGNIFICANT** | floor (n=7, raw p=0.0156) |
| h2o | v1_fa2_stack | 0.006288 | 0.09375 | **SIGNIFICANT** | floor (n=7, raw p=0.0156) |
| tova | v1_fa2_stack | 0.005527 | 0.09375 | **SIGNIFICANT** | floor (n=7, raw p=0.0156) |

## Paired deltas (audit trail)

### h2o vs vanilla  (n = 7)

| prompt_id | log(PPL_A) | log(PPL_B) | Δ |
|---|---:|---:|---:|
| ppl_chunk_0_eval_1 | 2.0155 | 2.0227 | -0.0072 |
| ppl_chunk_1_eval_2 | 2.0520 | 2.0454 | +0.0065 |
| ppl_chunk_2_eval_3 | 1.8547 | 1.8437 | +0.0110 |
| ppl_chunk_3_eval_4 | 1.6379 | 1.5524 | +0.0855 |
| ppl_chunk_4_eval_5 | 1.5064 | 1.5547 | -0.0484 |
| ppl_chunk_5_eval_6 | 1.3364 | 1.1998 | +0.1366 |
| ppl_chunk_6_eval_7 | 1.6570 | 1.5934 | +0.0637 |

### tova vs vanilla  (n = 7)

| prompt_id | log(PPL_A) | log(PPL_B) | Δ |
|---|---:|---:|---:|
| ppl_chunk_0_eval_1 | 2.0214 | 2.0227 | -0.0012 |
| ppl_chunk_1_eval_2 | 2.0798 | 2.0454 | +0.0344 |
| ppl_chunk_2_eval_3 | 1.8702 | 1.8437 | +0.0265 |
| ppl_chunk_3_eval_4 | 1.6896 | 1.5524 | +0.1371 |
| ppl_chunk_4_eval_5 | 1.5064 | 1.5547 | -0.0483 |
| ppl_chunk_5_eval_6 | 1.3828 | 1.1998 | +0.1830 |
| ppl_chunk_6_eval_7 | 1.6997 | 1.5934 | +0.1063 |

### v1_fa2_stack vs vanilla  (n = 8)

| prompt_id | log(PPL_A) | log(PPL_B) | Δ |
|---|---:|---:|---:|
| ppl_chunk_0_eval_1 | 2.0648 | 2.0227 | +0.0421 |
| ppl_chunk_1_eval_2 | 2.1822 | 2.0454 | +0.1367 |
| ppl_chunk_2_eval_3 | 1.9115 | 1.8437 | +0.0678 |
| ppl_chunk_3_eval_4 | 1.7464 | 1.5524 | +0.1940 |
| ppl_chunk_4_eval_5 | 1.5396 | 1.5547 | -0.0151 |
| ppl_chunk_5_eval_6 | 1.4353 | 1.1998 | +0.2355 |
| ppl_chunk_6_eval_7 | 1.7901 | 1.5934 | +0.1967 |
| ppl_chunk_7_eval_8 | 1.9513 | 1.8484 | +0.1028 |

### h2o vs tova  (n = 7)

| prompt_id | log(PPL_A) | log(PPL_B) | Δ |
|---|---:|---:|---:|
| ppl_chunk_0_eval_1 | 2.0155 | 2.0214 | -0.0059 |
| ppl_chunk_1_eval_2 | 2.0520 | 2.0798 | -0.0278 |
| ppl_chunk_2_eval_3 | 1.8547 | 1.8702 | -0.0156 |
| ppl_chunk_3_eval_4 | 1.6379 | 1.6896 | -0.0516 |
| ppl_chunk_4_eval_5 | 1.5064 | 1.5064 | -0.0000 |
| ppl_chunk_5_eval_6 | 1.3364 | 1.3828 | -0.0464 |
| ppl_chunk_6_eval_7 | 1.6570 | 1.6997 | -0.0427 |

### h2o vs v1_fa2_stack  (n = 7)

| prompt_id | log(PPL_A) | log(PPL_B) | Δ |
|---|---:|---:|---:|
| ppl_chunk_0_eval_1 | 2.0155 | 2.0648 | -0.0493 |
| ppl_chunk_1_eval_2 | 2.0520 | 2.1822 | -0.1302 |
| ppl_chunk_2_eval_3 | 1.8547 | 1.9115 | -0.0569 |
| ppl_chunk_3_eval_4 | 1.6379 | 1.7464 | -0.1085 |
| ppl_chunk_4_eval_5 | 1.5064 | 1.5396 | -0.0332 |
| ppl_chunk_5_eval_6 | 1.3364 | 1.4353 | -0.0989 |
| ppl_chunk_6_eval_7 | 1.6570 | 1.7901 | -0.1330 |

### tova vs v1_fa2_stack  (n = 7)

| prompt_id | log(PPL_A) | log(PPL_B) | Δ |
|---|---:|---:|---:|
| ppl_chunk_0_eval_1 | 2.0214 | 2.0648 | -0.0433 |
| ppl_chunk_1_eval_2 | 2.0798 | 2.1822 | -0.1024 |
| ppl_chunk_2_eval_3 | 1.8702 | 1.9115 | -0.0413 |
| ppl_chunk_3_eval_4 | 1.6896 | 1.7464 | -0.0569 |
| ppl_chunk_4_eval_5 | 1.5064 | 1.5396 | -0.0332 |
| ppl_chunk_5_eval_6 | 1.3828 | 1.4353 | -0.0525 |
| ppl_chunk_6_eval_7 | 1.6997 | 1.7901 | -0.0903 |

## Headline

- **Statistically indistinguishable from vanilla (paired-t Holm, α=0.05):** h2o, tova
- **Significantly different from vanilla (paired-t Holm, α=0.05):** v1_fa2_stack (Δlog PPL = +0.1201, d_z = +1.39, Holm t_p = 0.0228)

**Inter-policy verdicts (eviction policies against each other):**
- h2o vs tova: Δlog PPL = -0.0272, d_z = -1.32, Holm t_p = 0.0386 ⇒ **SIGNIFICANT**.
- h2o vs v1_fa2_stack: Δlog PPL = -0.0871, d_z = -2.16, Holm t_p = 0.00629 ⇒ **SIGNIFICANT**.
- tova vs v1_fa2_stack: Δlog PPL = -0.0600, d_z = -2.29, Holm t_p = 0.00553 ⇒ **SIGNIFICANT**.
