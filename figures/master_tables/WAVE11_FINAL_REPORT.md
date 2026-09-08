# Wave-11 FINAL Report

_Generated at **2026-06-08 06:43:15 EDT**._  
Source: `/home/mislam22/EndurKV_workspace/phone-logs/wave11_eval_1780862534`  
Phone-logs root: `/home/mislam22/EndurKV_workspace/phone-logs`

PPL is the **token-weighted geometric mean** per-cell (`exp(sum_i n_tok_i * mean_nll_i / sum_i n_tok_i)`) — the WikiText-2 convention used by H2O / KIVI / StreamingLLM / TOVA. CI is a 1000-sample percentile bootstrap (95%) over per-chunk NLLs, exponentiated for display. `log_std` is the per-chunk std of `log(PPL)` (so its units are nats).

NIAH rule judge: case-insensitive substring `"sandwich at dolores park"` with a negation-window guard (see `score_niah.rule_based_judge`).

**Overall verdict:** PARTIAL (2 complete, 3 partial, 1 models / 5 policies); primary contrast v1_fa2_stack vs tova: v1_fa2_stack lower-PPL, Wilcoxon p=0.018

## 1. Per-(model, policy) PPL aggregate

| Model | Policy | mean_PPL | CI_low | CI_high | log_std | n_chunks | peak_DDR_°C | decode_tps | throttle | evicted_tok | Pareto |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| phi3 | vanilla | 5.4640 | 4.4626 | 6.5116 | 0.2854 | 8 | 64.80 | N/A | 0 | 0 | N/A |
| phi3 | h2o | 5.4744 | 4.6453 | 6.5876 | 0.2643 | 7 | 62.90 | N/A | 0 | 1249188 | N/A |
| phi3 | tova | 5.6267 | 4.7851 | 6.7359 | 0.2573 | 7 | 59.40 | N/A | 0 | 2404470 | N/A |
| phi3 | v1fa2 | 6.0821 | 5.1694 | 7.1292 | 0.2533 | 8 | 66.00 | N/A | 0 | 0 | N/A |
| phi3 | streamingllm | 7.9083 | 7.5999 | 8.2517 | 0.0582 | 2 | 59.00 | N/A | 0 | 3722240 | N/A |

## 2. Paired significance tests on log(PPL)

| Model | A | B | n | mean Δlog(PPL) | t | t_p | W | W_p | Holm-W_p | Cohen d_z |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phi3 | h2o | streamingllm | 2 | -0.0355 | -1.552 | 0.1207 | 0.000 | 0.1797 | 0.8815 | -1.097 |
| phi3 | h2o | tova | 7 | -0.0272 | -3.498 | 0.0004679 | 0.000 | 0.01796 | 0.1729 | -1.322 |
| phi3 | h2o | v1fa2 | 7 | -0.0871 | -5.702 | 1.182e-08 | 0.000 | 0.01796 | 0.1729 | -2.155 |
| phi3 | h2o | vanilla | 7 | 0.0354 | 1.487 | 0.1371 | 22.000 | 0.1763 | 0.8815 | 0.562 |
| phi3 | streamingllm | tova | 2 | 0.0187 | 1.561 | 0.1186 | 3.000 | 0.1797 | 0.8815 | 1.104 |
| phi3 | streamingllm | v1fa2 | 2 | -0.0542 | -3.086 | 0.002029 | 0.000 | 0.1797 | 0.8815 | -2.182 |
| phi3 | streamingllm | vanilla | 2 | 0.0352 | 1.184 | 0.2364 | 3.000 | 0.1797 | 0.8815 | 0.837 |
| phi3 | tova | v1fa2 | 7 | -0.0600 | -6.053 | 1.424e-09 | 0.000 | 0.01796 | 0.1729 | -2.288 |
| phi3 | tova | vanilla | 7 | 0.0625 | 2.016 | 0.04383 | 23.000 | 0.1282 | 0.7691 | 0.762 |
| phi3 | v1fa2 | vanilla | 8 | 0.1201 | 3.928 | 8.558e-05 | 35.000 | 0.01729 | 0.1729 | 1.389 |

## 3. Paired McNemar exact tests on NIAH

_No NIAH pair data._

## 4. Pre-registered primary contrast: `v1_fa2_stack` vs `tova`

- **phi3** PPL (Wilcoxon, paired on chunks): n=7, mean Δlog(PPL)=-0.0600, p=0.01796, Cohen d_z=-2.288

## 5. NIAH pass/fail heatmap (rows = ctx, cols = depth %)

Legend: `O` = correct, `.` = wrong, `?` = missing.

### Overall NIAH accuracy

| Model | Policy | correct | total | accuracy |
|---|---|---:|---:|---:|
| phi3 | vanilla | 7 | 8 | 87.5% |

### phi3 / vanilla

```
ctx \ d%   0  12  25  37  50  62  75  87
    2048    O   ?   ?   ?   O   ?   ?   ?
    4096    ?   ?   .   ?   ?   ?   O   ?
    6144    ?   ?   ?   ?   O   ?   ?   O
    8192    O   ?   ?   ?   O   ?   ?   ?
```

## 6. Health invariants (per WAVE11_FILL_IN_PROTOCOL §5.3)

- **WARN**: `phi3/streamingllm` has only 2 chunks (target ≥ 6)
- INFO: no Holm-corrected paired test p < 0.05 on Phi-3 vs vanilla yet (expected while only the vanilla cell has data).

## 7. Artefacts written by this run

- `figures/master_tables/WAVE11_FINAL_REPORT.md` (this file)
- `figures/eval_plots/wave11_final_ppl_bars.png`
- `figures/eval_plots/wave11_final_pareto.png`
- `figures/eval_plots/wave11_final_niah_heatmap.png`
- `/tmp/wave11_fills.tsv` (TSV for `fill_chapter_results.sh`)
