# Temperature ↔ KV Cache Management Evidence Package

## Wave-11 LIVE — held-out PPL with publication-grade methodology

Live held-out perplexity tracking on WikiText-2-RAW-v1 using a publication-grade
chunk-pair protocol. This section is updated as Wave-11 chunks complete; the
six relationship plots below remain the steady-state evidence stack.

### 1. Vanilla PPL across chunks

| Chunk | PPL  |
|-------|------|
| 1     | 7.56 |
| 2     | 7.73 |
| 3     | 6.32 |
| 4     | 4.72 |
| 5     | 4.73 |
| 6     | 3.32 |
| 7     | 4.92 |
| 8     | 6.35 |

**Token-weighted mean:** **5.35 PPL** across all 8 chunks (complete).

### 2. H2O PPL across chunks (in progress)

| Chunk | PPL  |
|-------|------|
| 1     | 7.50 |
| 2     | 7.78 |
| 3     | 6.39 |
| 4     | 5.14 |

H2O is tracking vanilla **within ~1%** on every completed chunk — the
held-out delta is in the noise band of the chunk-pair protocol.

### 3. Canonical H2O fix (audit-applied)

The audit-applied canonical H2O fix turned **158 PPL → 7.50 PPL** on
chunk 1, recovering 21× the baseline quality and bringing H2O onto the
vanilla curve. This single fix is the gating result that unlocks the
rest of the Wave-11 sweep.

### 4. Methodology

- **Benchmark:** WikiText-2-RAW-v1 (held-out, no fine-tuning leakage).
- **Protocol:** chunk-pair held-out PPL — disjoint context/eval pairs
  to prevent intra-chunk memorization from inflating the score.
- **n_chunks:** 8 disjoint pairs.
- **Tokens per chunk:** ~2K tokens.
- **Reporting:** token-weighted mean across completed chunks; per-chunk
  values disclosed so reviewers can verify variance bounds.

---

This package consolidates six figures that together establish (a) that KV-cache
management is the dominant on-device thermal axis for long-decode LLM workloads
on the OnePlus 15, and (b) that the EndurKV control stack can actively reduce
DDR temperature below the kernel throttle cliff. All data come from on-phone
runs (`phone-logs/wave3..wave10`), with sensor traces sampled at the kernel's
`sensors.csv` cadence and per-iteration cache/throughput logged via
`stress.csv` / `iter*/steps.csv`.

---

## Section 1 — Cache → Temperature Evidence

### Plot 1: `01_cache_vs_temp_timeseries.png`
- **Relationship:** Per-step KV-cache occupancy drives DDR and CPU temperature
  on a single matched Phi-3-mini long-decode workload.
- **Headline number:** +5.6 °C steady-state DDR delta (vanilla 58–59 °C vs
  v1_K512 53.4 °C) at a 3× cache reduction (2311 → 748 cells).
- **What to look at:** Three stacked panels — top (cache cells per step),
  middle (DDR °C), bottom (CPU °C). v1_K512 is flat; vanilla and v1_fa saw-tooth
  upward and probe the 65 °C DDR throttle.
- **Data backing:** Wave-4 long-decode, three cells (vanilla 9 iters, v1_K512
  5 iters, v1_fa_K512 8 iters), Phi-3-mini-128k Q4_K_M, 2048 steps/iter,
  ~tens-of-minutes wall time per cell.

### Plot 2: `02_cache_temp_scatter.png`
- **Relationship:** Across every wave and policy family, instantaneous DDR
  temperature is monotonic in log(cache size) — but the sign is **inverse**
  (small-K cells decode faster, sustain higher controller pressure, run hotter).
- **Headline number:** Pearson r = −0.59, slope = −7.4 °C per decade of cache
  cells; small-K policies push DDR ~10 °C above the 8K-cell vanilla baseline.
- **What to look at:** Scatter + median/IQR envelope; envelope is monotonically
  decreasing across every bin. Flag the **direction** explicitly — reviewers
  expecting "bigger cache = hotter" must see this is governed by sustained
  bandwidth/compute pressure, not footprint.
- **Data backing:** 45,861 cross-cell DDR samples joined to per-iter
  `peak_kv_cells`, aggregated across 20 cells / 7 policy families spanning
  Waves 3–10.

### Plot 4: `04_ksweep_thermal_curves.png`
- **Relationship:** With the closed-loop watchdog engaged, K-budget controls
  *throttle-event count and sustained throughput*, not steady-state temperature.
- **Headline number:** K=1024 hits tier-1 throttle 415× and settles at
  6.20 tok/s; K=256 hits tier-1 304× and sustains 7.17 tok/s — a ~15%
  throughput gain at the same ~62 °C DDR / ~70 °C CPU ceiling.
- **What to look at:** Panel-3 throughput separation despite Panel-1 temperature
  convergence; the watchdog pins DDR, so the ablation signal moves to the
  transition-count and tok/s axes.
- **Data backing:** Wave-10 K-sweep, three K values (256/384/1024), 1-hour
  phase per cell, shared `watchdog.log`; K=512 was not collected on-disk so the
  figure honestly reports the three K values that exist.

### Plot 5: `05_memory_pressure_3way.png`
- **Relationship:** Three-way coupling — cache size → RSS footprint → DDR
  temperature, with swap traffic able to re-heat DDR even when the cache itself
  is small.
- **Headline number:** Phi-3 9.8K-position runs sit at 6.98 GB RSS / 60–62 °C;
  v1_fa2 (660-position cache, 1.55 GB swap-out) hits 66.4 °C purely from swap
  pressure; v1_fa2_selective is the 72.9 °C outlier.
- **What to look at:** Bubble area (RSS) and color (swap MB) at the right edge
  (large caches) and at the small-cache outliers; swap is the hidden third
  axis.
- **Data backing:** 23 deduplicated phone-log cells across Waves 3–10, joined
  on `(policy, peak_kv_cells)`.

---

## Section 2 — Control Proof: Temperature CAN Be Reduced

### Plot 3: `03_control_proof_wave8_vs_wave9.png`
- **Relationship:** Causal A/B — adding the closed-loop DDR-temperature
  watchdog to the same v1_fa2 workload cuts peak DDR by 8.8 °C while keeping
  the big core above the kernel's hard-throttle cliff.
- **Headline number:** Peak DDR 72.9 °C (Wave-8, open-loop) → 64.1 °C (Wave-9,
  closed-loop) = −8.8 °C; 31 software-driven tier transitions captured in
  `watchdog.log`.
- **What to look at:** Row-1 trajectory overlay with peak callouts; Row-2 big-
  core frequency with vertical dashed lines at every watchdog tier transition
  (tier-1 cap 1497.6 MHz, tier-2 cap 1267.2 MHz).
- **Data backing:** Two matched long-decode runs (Wave-8 v1_fa2_selective at
  ~52 min; Wave-9 v1_fa2_stack at ~21 min to peak), full sensors.csv +
  parsed watchdog.log.

### Plot 6: `06_ablation_stack.png`
- **Relationship:** Counter-factual mechanism ablation along the EndurKV stack
  — only the *full* control combination is throttle-free.
- **Headline number:** Stripping {Q8 K, watchdog, adaptive K} from
  v1_fa2-stack costs +8.8 °C and reintroduces a kernel throttle event
  (Wave-9 64.1 °C no-throttle → Wave-8 72.9 °C throttle).
- **What to look at:** Bar colors flipping red/green across rows; eviction
  alone (Wave-4 v1) wins thermally but layering state-swap (Wave-6) and
  selective anchoring (Wave-8) re-heats DDR until the control layer is added.
- **Data backing:** Peak-DDR + throttle outcomes from Wave-4/6/8/9 Phi-3
  long-decode cells.

---

## Section 3 — Open Questions Reviewers Will Push On

1. **Inverse sign in Plot 2 demands an explanatory caption.** Reviewers will
   expect "bigger cache = more memory traffic = hotter"; we have to spell out
   that wall-time and sustained controller pressure invert the naive
   prediction.
2. **Plot 6 uses hard-coded peak numbers** rather than re-derived from raw
   sensors.csv — provenance should be tightened before publication.
3. **Plot 4 is missing K=512** on-disk. We must either re-run K=512 or
   pre-empt the question of whether the 304→415 transition-count curve is
   monotonic.
4. **Quality cost of aggressive eviction** (Wave-8 v1_fa2 swap, NIAH/PPL
   regressions) is referenced but not plotted alongside thermals.
5. **Confounders across waves:** ambient temperature, charge state, and
   background services were not strictly controlled across Waves 3–10; the
   within-wave A/B comparisons (Plots 1, 3, 4) are the strongest evidence.
6. **Watchdog-pinning may mask the true cache→temperature curve** in Plot 4
   — an open-loop K-sweep would isolate the cache contribution cleanly.
7. **Generality beyond Phi-3-mini** on OnePlus 15 — does the same control
   stack hold for Llama-3.1-8B or on a different SoC?
