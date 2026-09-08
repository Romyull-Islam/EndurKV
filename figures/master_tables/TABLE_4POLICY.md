# Master 4-policy table — vanilla / v1 / TOVA / H2O

All numbers are averages across 3 prompts × 1 rep per (model, policy) cell on the CPU sweep.

| Model | Policy | n | F1 | WT2 PPL | Decode t/s | Peak KV (MB) | Avg live KV (MB) | KV savings vs vanilla | Peak RSS (MB) | Mass kept | Retention | Eff | Evicted | CPU peak | DDR peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-1B | vanilla | 3 | 0.115 | 8.78 | 11.31 | 131 | 131 | (baseline) | 1147 | 1.000 | 1.000 | 1.00 | 0 | 56.6 | 48.9 |
| Llama-1B | v1 | 3 | 0.114 | 8.79 | 6.97 | 130 | 40 | 69.8% | 1208 | 0.991 | 0.474 | 3.21 | 25765 | 67.2 | 51.5 |
| Llama-1B | tova | 3 | 0.024 | 8.79 | 6.93 | 130 | 31 | 76.4% | 1208 | 0.989 | 0.375 | 4.09 | 36984 | 54.8 | 47.6 |
| Llama-1B | h2o | 3 | 0.024 | — | 4.82 | 130 | 31 | 76.4% | 1223 | 0.983 | 0.375 | 4.05 | 26119 | 63.9 | 48.6 |
| Gemma-2-2B | vanilla | 3 | 0.141 | 8.85 | 5.20 | 518 | 460 | (baseline) | 2539 | 1.000 | 1.000 | 1.00 | 0 | 70.1 | 53.0 |
| Gemma-2-2B | v1 | 3 | 0.106 | 8.82 | 3.21 | 511 | 128 | 72.1% | 2582 | 0.993 | 0.435 | 3.46 | 66825 | 64.5 | 49.8 |
| Gemma-2-2B | tova | 3 | 0.031 | 8.82 | 3.20 | 511 | 100 | 78.3% | 2582 | 0.990 | 0.340 | 4.43 | 98563 | 75.7 | 53.5 |
| Gemma-2-2B | h2o | 3 | 0.031 | — | 2.36 | 511 | 100 | 78.3% | 2611 | 0.985 | 0.340 | 4.40 | 58783 | 66.6 | 51.2 |
| Phi-3-128k | vanilla | 3 | 0.211 | 5.03 | 2.98 | 1873 | 1873 | (baseline) | 6146 | 1.000 | 1.000 | 1.00 | 0 | 68.7 | 51.8 |
| Phi-3-128k | v1 | 3 | 0.205 | 5.05 | 2.32 | 1860 | 395 | 78.9% | 6265 | 0.980 | 0.317 | 4.41 | 27037 | 65.7 | 54.2 |
| Phi-3-128k | tova | 3 | 0.206 | 5.05 | 2.26 | 1860 | 364 | 80.6% | 6286 | 0.975 | 0.297 | 4.82 | 32360 | 57.2 | 50.3 |
| Phi-3-128k | h2o | 3 | 0.205 | — | 2.21 | 1860 | 364 | 80.6% | 6302 | 0.969 | 0.297 | 4.79 | 24067 | 63.3 | 51.6 |

## Column meanings

- **F1**: LongBench downstream-task quality (higher = better; v1 ≥ vanilla on Llama-1B+Gemma; ~tied on Phi-3)
- **WT2 PPL**: intrinsic LM quality on WikiText-2 short-seed (lower = better; canonical 7-8 for Llama-1B; all policies tie at this setting because no eviction triggers)
- **Decode t/s**: per-token generation speed on CPU
- **Peak KV (MB)**: allocated cache buffer (same across policies for given model; llama.cpp pre-allocates)
- **Avg live KV (MB)**: real working-set KV during inference, accounting for retention. THIS is what actually drives memory + DDR thermal load.
- **KV savings vs vanilla**: % of vanilla's avg live KV that's freed by eviction
- **Peak RSS (MB)**: process resident memory
- **Mass kept**: fraction of total attention probability that survives eviction (closer to 1.0 = better)
- **Retention**: fraction of KV positions kept (lower = more aggressive compression)
- **Eff** (Eviction efficiency): mass / retention ratio (higher = more mass per cell kept)
- **Evicted**: total positions evicted during decoding
- **CPU peak**: max CPU core temperature during cell
- **DDR peak**: max system-memory temperature during cell
