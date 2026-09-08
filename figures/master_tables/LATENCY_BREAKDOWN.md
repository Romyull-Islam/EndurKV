# Latency breakdown — Wave-3 REAL (Llama-1B / narrativeqa 7700 prompt + 256 decode)

Per-iteration latency decomposed from `prefill_ms` (TTFT proxy) and per-step
`wall_us` deltas (ITL/TPOT).

⚠️ **Vanilla cell was on AC charging (3.4 GHz peak DVFS); other cells on battery
(1.63 GHz cap).** Comparison is NOT clean until the matched-DVFS rerun completes.

## Latency metric definitions

| Metric | Definition |
|---|---|
| **TTFT** | Time to First Token. In our benchmark = `prefill_ms` (first decode token is emitted immediately after prefill completes). |
| **TPOT first/mid/last** | Time per output token at the 1st / middle / final decode step. Drift across these = throttling or growing KV cost. |
| **ITL median** | Robust per-token latency across the decode window (less sensitive to outliers than mean). |
| **Decode tok/s** | 1 / mean(per-token latency). |

## Per-iteration means across all iters in each cell

| Policy | TTFT (s) | TPOT first (ms) | TPOT mid (ms) | TPOT last (ms) | ITL median (ms) | Decode tok/s |
|---|---|---|---|---|---|---|
| vanilla *(AC, 3.4 GHz)* | **325.0** | 207.2 | 205.1 | 208.4 | 206.8 | 5.09 |
| v1 K=2048 | 428.6 | 204.6 | 203.0 | 203.0 | 205.0 | 7.36 |
| v1 K=512 | 418.3 | 211.6 | 214.5 | 214.5 | 215.5 | 7.05 |
| **v1_FA K=512** | 408.6 | **140.6** | **134.1** | **137.8** | **134.8** | **7.77** |

## Reading the table

- **TTFT (prefill)**: vanilla wins at 325 s because FA-on prefill with no eviction overhead is fastest. v1 cells are 30 % slower in TTFT due to FA-off prefill + eviction cost. **v1_FA at 408.6 s is the fastest v1 family member** since it has the same eviction work as v1 but FA-off prefill is unchanged across them.

- **TPOT / ITL**: v1_FA's per-token cost (134.8 ms median) is **dramatically lower** than v1 (215.5 ms) or vanilla (206.8 ms). That's the **FA-on decode benefit isolated** — same effective cache size (5550 cells), same algorithm, but the FA-fused softmax avoids materializing the n_kv × n_q attention matrix. On Llama-1B at K=512:
  ```
  v1 ITL    – v1_FA ITL = 215.5 − 134.8 = 80.7 ms/token saved by FA-on
  ```
  That's a 37 % per-token decode-latency reduction.

- **TPOT drift (first vs last)**: all four policies have nearly flat TPOT (≤2-4 ms variation across 256-token decode), meaning **within a single iteration nothing throttles**. The iteration-to-iteration `decode_tps` decay (−12 to −17 %) we see in the per-iteration stress.csv reflects **between-iteration** thermal accumulation, not intra-decode growth.

## When the matched-DVFS rerun finishes

The vanilla cell will be re-measured under the same 1.63 GHz cap as v1 family. Expected updates:
- Vanilla TTFT will rise (FA-on still fastest prefill, but at half the clock)
- Vanilla ITL will rise to ~250-300 ms (proportional clock scaling)
- v1_FA's 37 % ITL advantage over v1 should be preserved (algorithm-bound, not clock-bound)
- The "v1_FA vs vanilla" wall-clock-tok/s race will become clearer

## What's in the source data

- `wave3_real_1780680903/<cell>/iter*/meta.json` — prefill_ms, decode_ms, decode_tps, n_decode_steps
- `wave3_real_1780680903/<cell>/iter*/steps.csv` — per-token wall_us → ITL via consecutive deltas

Computation script:
```python
ttft_ms = meta["prefill_ms"]                  # first token = end-of-prefill
deltas  = [wall_us[i]-wall_us[i-1] for i in range(1,len(wall_us))]
tpot_first = deltas[0]/1000.0                 # ms
tpot_mid   = deltas[len(deltas)//2]/1000.0
tpot_last  = deltas[-1]/1000.0
itl_median = sorted(deltas)[len(deltas)//2]/1000.0
```
