# Wave-10 K-Sweep: Mass Retained vs K

Source: `/home/mislam22/EndurKV_workspace/phone-logs/wave10_ksweep_1780815847/K{256,384,512,1024}/iter*/meta.json`

Policy: `v1_fa2` (FA2 stack), Phi-3-mini-128k-instruct Q4_K_M, ctx 4096, longgen prompt, greedy sampling.

## Table 1. Mass-Retained, Retention Ratio, Efficiency, Perplexity, Peak KV (mean across iters)

| K     | n_iters | mass_retained | retention_ratio | efficiency | perplexity | peak_kv (cells) | peak_kv (MB) |
|------:|--------:|--------------:|----------------:|-----------:|-----------:|----------------:|-------------:|
|  256  |     12  |        1.000  |          1.000  |     1.000  |    2.0948  |           2 314 |       867.75 |
|  384  |     12  |        1.000  |          1.000  |     1.000  |    2.1227  |           2 316 |       868.50 |
|  512  |      0  |          n/a  |            n/a  |       n/a  |       n/a  |             n/a |          n/a |
| 1024  |     10  |        1.000  |          1.000  |     1.000  |    1.8268  |           2 314 |       867.75 |

## Notes

1. **K=512 absent on disk.** The wave-10 directory contains only
   `K256/`, `K384/`, `K1024/` plus `progress.log` and `watchdog.log`. No
   `K512/` subdirectory exists in
   `phone-logs/wave10_ksweep_1780815847/`. The K=512 row in
   `SUBSECTION_KSWEEP.md` (mean 6.089 tok/s, PPL 2.1686, 10 iters) was sourced
   from a different run and is not represented here.

2. **All mass-retained metrics are trivially 1.000.** Every wave-10 meta.json
   carries `"no_evict_decode": true`, i.e. eviction was disabled during the
   decode phase. The eviction-quality estimators
   (`mean_mass_retained`, `mean_retention_ratio`, `mean_eviction_efficiency`)
   therefore degenerate to 1.0 by construction in every iter and every K. The
   K-sweep in wave-10 does **not** discriminate policies on these axes;
   it discriminates on perplexity, peak KV footprint, and throughput.

3. **Per-iter values are identical within each K.** Greedy sampling on a fixed
   prompt produces deterministic decoding, so PPL / peak_kv / mass_retained /
   etc. are constant across iter0001..iter0012 for a given K. The "mean" row
   is therefore equal to every individual iter value (min = max = mean).

4. **Perplexity ordering (lower = better).** K=1024 (1.8268) < K=256 (2.0948)
   < K=384 (2.1227). The K=1024 cell carries the quality win by a wide margin
   despite using nearly identical peak KV cells / MB to K=256.

5. **Peak KV nearly constant.** Peak cells = 2314-2316 across the sweep,
   peak MB = 867.75-868.50. K is shaping eviction pressure on the
   non-resident tail, not the resident peak, consistent with the
   seq_add-skip caveat documented in `SUBSECTION_KSWEEP.md`.

## Raw per-K values (one iter each, all iters identical)

| K     | mean_mass_retained | mean_retention_ratio | mean_eviction_efficiency | perplexity | peak_kv_cells | peak_kv_mb | evicted_prefill | evicted_total_decode | no_evict_decode |
|------:|-------------------:|---------------------:|-------------------------:|-----------:|--------------:|-----------:|----------------:|---------------------:|:----------------|
|  256  |             1.000  |               1.000  |                   1.000  |   2.094768 |        2 314  |    867.75  |            659  |          1 656 897   | true            |
|  384  |             1.000  |               1.000  |                   1.000  |   2.122720 |        2 316  |    868.50  |            493  |          1 430 516   | true            |
|  512  |               n/a  |                 n/a  |                     n/a  |        n/a |          n/a  |       n/a  |             n/a |                 n/a  | n/a             |
| 1024  |             1.000  |               1.000  |                   1.000  |   1.826802 |        2 314  |    867.75  |            480  |            508 641   | true            |

## Methodological caveat

If a mass-retained sweep that actually discriminates K is required, the
wave-10 runs must be re-issued with `no_evict_decode: false`. As recorded,
the mass-retained channel of this sweep is uninformative.
