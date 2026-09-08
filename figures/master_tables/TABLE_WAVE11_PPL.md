# Wave-11 Perplexity Comparison

Per-cell stats over chunks discovered under `phone-logs/wave11_*/<model>/<policy>/ppl/iter*/meta.json` (or legacy `wave11_*/<model>/<policy>/iter*/meta.json`). PPL is the TOKEN-WEIGHTED GEOMETRIC MEAN per-cell (exp(sum_i n_tok_i * mean_nll_i / sum_i n_tok_i)) — the WikiText-2 convention used by H2O/KIVI/StreamingLLM/TOVA. CI is a 1000-sample percentile bootstrap (95%) over per-chunk NLLs, exponentiated for display. std_dev column is in log(PPL) units.

| Model | Policy | mean_PPL | CI_low | CI_high | std_dev | n_chunks |
|---|---|---:|---:|---:|---:|---:|
| Phi-3-mini-128k | vanilla | 5.3499 | 4.3082 | 6.5346 | 0.3021 | 7 |
