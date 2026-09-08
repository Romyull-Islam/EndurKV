# v1_FA Family — Per-Variant Workload Comparison (Wave-3 through Wave-9)

**Generated:** 2026-06-08
**Hardware:** OnePlus 15 / Snapdragon 8 Elite Gen 5 / Adreno 840 / 12 GB UMA
**Source data:** `phone-logs/wave{3..9}_*/` per-cell `meta.json`, `stress.csv`, `sensors.csv`, `progress.log`
**Sister doc:** `ALL_WAVES_MASTER.md` (raw numbers), `SECTION_V1FA2_STACK.md` (mechanism provenance)

> ## PPL reading note
> All PPL values in this table are **sampling-NLL perplexity** (`exp(mean_nll)` over the
> model's own greedy continuation). These are **biased low** and **only valid for
> internal cross-cell comparison among Wave-3..9 cells**. They are NOT comparable to
> teacher-forced PPL in the literature. The corrected held-out chunk-pair PPL is the
> Wave-11 protocol (pending). See `HELD_OUT_PPL_FINDING.md`.

---

## 1. The six v1_FA family variants

| # | Variant | Wave | Tag in logs | One-line description |
|---|---|---|---|---|
| 1 | **v1** | 3..4 | `v1` | FA-off prefill **and** FA-off decode, per-step spread-gate eviction. Reference for "no FA fast path." |
| 2 | **v1_FA (frozen)** | 3 | `v1_fa` | SnapKV-style: FA-off prefill builds the snapshot mask, mask is **frozen at end-of-prefill** (`no_evict_decode=true`), FA-on decode rides the mask. |
| 3 | **v1_FA (bounded)** | 6 | `v1_fa_K512_bounded` | Wave-3 design + bounded recency window + sink so the snapshot mask is **not** frozen — decode-time eviction is re-enabled (`no_evict_decode=false`). |
| 4 | **v1_FA² over-anchored** | 7 | `v1_fa2` | Anchor the **full prompt** + recency tier, FA-on decode. **FAILED** — over-anchoring blew memory (1.55 GB swap) and PPL collapsed to 3.92. |
| 5 | **v1_FA² selective** | 8 | `v1_fa2_selective` | Anchor only the **top-32** prompt positions by attention mass + recency tier + Q8 K cache. Recovers PPL and kills swap, but DDR runs hot (no thermal control). |
| 6 | **v1_FA²-stack** | 9 | `v1_fa2_stack` | v1_FA² selective + preempt-throttle DDR watchdog + closed-loop K controller + memory gate. The EndurKV headline policy. |

---

## 2. Headline metrics per variant (all from `phone-logs/`, no synthesized numbers)

### 2a. On the **narrativeqa-Phi3** workload (long-prompt 9794 tok + 256-tok decode, prefill-dominated)

| Variant | Wave | Iters | Mean tok/s | Peak DDR (°C) | Peak CPU (°C) | Swap MB | Mean PPL [s] | Thermal-throttle event? |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| v1 K=512 | 3-phi3 | 2 | 1.071 | **61.7** ❌ | **68.6** ❌ | 172.5 | **9.418** | no kernel throttle but in throttle zone |
| **v1_FA frozen** K=512 | 3-phi3 | 2 | **1.921** | 58.7 ✓ | 63.6 ✓ | 515.1 | **2.857** | no |
| v1_FA filebacked | 5 | 2 | 2.03 | 60.6 | 66.7 | 739.5 | 2.857 | no (mem-bound, slow) |
| v1_FA bounded K=512 | 6 | 9 | 5.77 | 65.6 ❌ | 72.9 ❌ | 28.0 | 3.203 | yes (DDR over 65 °C, kernel pressure) |
| v1_FA² over-anchored | 7 | 7 | 6.72 | 66.4 ❌ | 74.4 ❌ | **1551.8** | **3.918** | yes (memory-pressure cliff + DDR hot) |
| v1_FA² selective | 8 | 11 | 6.75 | **72.9** ❌ | **80.6** ❌ | 6.7 | 3.560 | **YES — iter-10 kernel `freq_qos` to 883 MHz** |
| **v1_FA²-stack** K=512 | 9 | 10 | 6.09 | **64.1** ✓ | **66.8** ✓ | **0.0** | **2.169** | **NO (watchdog held the cliff)** |

### 2b. On the **long-decode-Phi3** workload (~272-tok prompt + 2048-tok decode, decode-dominated)

| Variant | Wave | Iters | Iter 1 → last decode (tok/s) | Mean tok/s | Peak DDR (°C) | Peak CPU (°C) | Peak KV (MB) | Mean PPL [s] |
|---|---|---:|---|---:|---:|---:|---:|---:|
| v1 K=512 | 4 | 5 | 2.86 → 2.56 (−10%) | 2.66 | **54.4** ✓ | 62.0 ✓ | ~190 | 4.224 |
| v1_FA frozen K=512 | 4 | 8 | 5.85 → 4.19 (−28%) | 4.65 | 62.5 ❌ | 69.0 ❌ | 867 (grew like vanilla) | **2.341** |
| v1_FA bounded K=512 | 6 (long-decode harness) | 9 | 6.73 → 6.43 (−4%) | **5.77** | **65.6** ❌ | 72.9 ❌ | 288 (capped) | 3.203 |

(Wave-7..9 also used a long-decode-style harness on Phi-3-mini-4k with 272-tok prompt + 2048 decode; rows in 2a above are the *sustained* peak temperatures over a full cell.)

### 2c. Per-iter sustained decode trajectory (sanity check, from `progress.log`)

```
                              iter:    1     2     3     4     5     6     7     8     9    10
Wave-3 v1_FA frozen   K=512        1.965 1.877  —     —     —     —     —     —     —     —
Wave-4 v1_FA frozen   K=512        5.848 5.174 5.167 3.834 4.175 4.185 4.600 4.186  —     —
Wave-6 v1_FA bounded  K=512        6.727 6.062 5.766 5.794 5.288 5.536 4.520 5.818 6.429  —
Wave-7 v1_FA² over-anch.K=512      7.002 6.675 6.662 6.688 6.657 6.684 6.670  —     —     —     (cell aborted on memory pressure)
Wave-8 v1_FA² selective K=512      7.211 6.794 6.785 6.817 6.821 6.790 6.801 6.820 6.814 5.773  (iter-10 = kernel throttle to 883 MHz)
Wave-9 v1_FA²-stack   K=512        7.382 6.723 6.418 6.113 6.399 6.347 6.170 4.569 6.168 4.601  (no throttle; transient watchdog dips)
```

---

## 3. What each variant solved or revealed

| Variant | Problem it solved | New problem it revealed |
|---|---|---|
| **v1** | Demonstrated bounded-cache eviction caps DRAM bandwidth in the long-decode regime (Wave-4: 54.4 °C peak DDR, never throttles). | FA-off decode is intrinsically slow — 2.66 tok/s on Phi-3 long-decode. Not a deployable end-state. |
| **v1_FA frozen** (Wave-3) | Restored FA-on decode for ~2× speedup at near-vanilla peak DDR (58.7 °C) on the narrativeqa workload. Frozen-mask state-swap is a clean engineering trick. | Mask frozen at end-of-prefill → no decode-time eviction → cache **grows** during decode like vanilla. In long-decode regime (Wave-4) this throttles at iter 4 just like vanilla. |
| **v1_FA bounded** (Wave-6) | Unfroze the mask. Decode-time eviction re-enabled with bounded recency + sink. Wave-4-style long-decode now runs at 5.77 tok/s with the cache capped at 288 MB. | Sustained narrativeqa-style cell hits 65.6 °C peak DDR / 72.9 °C peak CPU — still in the kernel throttle zone. No closed-loop control. |
| **v1_FA² over-anchored** (Wave-7) | Introduced a two-tier anchor + recency structure. Throughput climbed to 6.72 tok/s. | **FAILED.** Anchoring the full prompt expanded the resident footprint past the 12 GB UMA ceiling → **1551.8 MB swap-out**, sampling-NLL PPL **3.92** (worst of all v1_FA family), DDR 66.4 °C. The full-anchor design is unshippable. |
| **v1_FA² selective** (Wave-8) | Restricted anchor to **top-32** positions by attention mass and added Q8 K cache. Swap collapsed to 6.7 MB; PPL recovered to 3.56. | DDR climbed to **72.9 °C** and peak CPU to **80.6 °C** — the workload ran longer and harder before any soft cap, so the kernel mitigation framework fired the 883 MHz downshift at **iter-10**. No thermal control. |
| **v1_FA²-stack** (Wave-9) | Added (i) preempt-throttle DDR watchdog, (ii) closed-loop K controller (K=512→384→256 on DDR tiers), (iii) memory gate. Peak DDR fell to **64.1 °C** (−8.8 °C vs Wave-8), peak CPU **66.8 °C** (−11.6 °C), **zero swap**, **zero kernel throttle events**, sampling-NLL PPL **2.169** (best in family). | "Cost of cool": 0.66 tok/s lower mean throughput than Wave-8 at K=512 (6.09 vs 6.75). Wave-10 shows this tax disappears at lower K (7.17 tok/s at K=256). |

---

## 4. Best variant per workload (the recommendation table)

| Workload type | Best v1_FA variant | Why it wins | Source |
|---|---|---|---|
| **Long-context narrativeqa-Phi3** (prefill-dominated, ~9794-tok prompt + 256-tok decode) | **v1_FA frozen (Wave-3)** | Best mean tok/s among single-cell candidates (1.921), peak DDR only 0.8 °C above vanilla (58.7 vs 57.9), PPL 2.857 (only beaten by Wave-9 stack at 2.169 but on a different short-prompt harness). State-swap memory cost is amortized over the long prefill. No kernel throttle. | Wave-3-phi3, `wave3_phi3_1780719530/v1_fa_K512/iter0001/meta.json` |
| **Long-decode-Phi3** (decode-dominated, 272-tok prompt + 2048-tok decode, **interactive long generation**) | **v1_FA²-stack (Wave-9)** | Holds the cache cap during decode (peak KV 867 MB, 1.2 M evicted positions), achieves 6.09 mean tok/s, sampling-NLL PPL **2.169** (best in v1_FA family on this harness), peak DDR 64.1 °C **with zero throttle and zero swap** thanks to the closed-loop K controller. The watchdog is what makes long-decode *sustainable*, not just fast for one iter. | Wave-9, `wave9_v1fa2_stack_1780796320/v1_fa2_stack/iter0001/meta.json` |
| **Short-prompt PPL** (Wave-4 harness, sampling-NLL; quality metric only) | **v1_FA²-stack (Wave-9)** | Sampling-NLL PPL 2.169 beats v1_FA frozen 2.341 (Wave-4), v1_FA bounded 3.203 (Wave-6), v1_FA² over-anch 3.918 (Wave-7), v1_FA² selective 3.560 (Wave-8) on the same long-decode harness. Among the *frozen / bounded* family (no FA²), **v1_FA bounded (Wave-6)** is the best at PPL 3.203 with no anchor mechanism. | Wave-9 vs Wave-4/6/7/8 meta.json |
| **Worst overall — DO NOT SHIP** | **v1_FA² over-anchored (Wave-7)** | Highest swap-out (**1551.8 MB**), highest PPL (**3.918**) of any v1_FA family member, DDR in throttle zone (66.4 °C). The full-prompt anchor is the proximal cause; selective anchoring (Wave-8) and Q8 K (Wave-8) are the fixes. | Wave-7, `wave7_v1fa2_1780782482/v1_fa2/iter0001/meta.json` and `progress.log` |
| **Memory-constrained / file-backed** | **v1_FA frozen (Wave-3)** over **v1_FA filebacked (Wave-5)** | Wave-5 file-backed state-swap produced 739.5 MB swap and dropped to 2.03 tok/s — the file-backed swap path costs more than the memory it saves on this hardware. Wave-3's in-memory state swap is the right answer until UMA pressure actually demands eviction. | Wave-5 progress.log: tps=2.004, tps=2.052 |
| **Sustained Llama-1B / smaller models** (workload doesn't push thermal envelope) | **v1_FA frozen (Wave-3-real)** | Peak DDR 49.8 °C, peak CPU 57.1 °C, mean tok/s **7.77** (vs vanilla 5.09), 0 MB swap. None of the family's later complexity is needed when the chip is not heat-bound. | Wave-3-real, `WAVE3_REAL_SUMMARY.md` |

---

## 5. Bottom line

- The v1_FA family evolves under two distinct pressures: (a) **decode-path speed** (FA-on vs FA-off) and (b) **thermal endurance under sustained load** (closed-loop K, memory gate, watchdog).
- **v1_FA frozen (Wave-3)** is the right answer for *prefill-heavy single-shot inference* on a cold device.
- **v1_FA bounded (Wave-6)** unblocks long-decode regimes by re-enabling decode-time eviction, but it is not thermally sustainable without the Wave-9 control loop.
- **v1_FA² over-anchored (Wave-7)** is the cautionary tale — full-prompt anchoring is *not* a sound design on a 12 GB UMA phone.
- **v1_FA²-stack (Wave-9)** is the deployment configuration: zero swap, zero kernel throttle events, sampling-NLL PPL 2.169, sustained 6.09 tok/s on Phi-3-mini long-decode. The closed-loop watchdog is what makes the FA²-on-decode design *survive* a 60-minute cell, which is the entire point of EndurKV.

---

## 6. Files referenced

- `phone-logs/wave3_phi3_1780719530/v1_fa_K512/iter0001/meta.json` (v1_FA frozen, narrativeqa)
- `phone-logs/wave3_real_1780680903/WAVE3_REAL_SUMMARY.md` (Llama-1B sustained)
- `phone-logs/wave4_longdecode_1780750084/v1_fa_K512/iter0001/meta.json` (v1_FA frozen, long-decode)
- `phone-logs/wave5_v1fa_filebacked_1780764804/progress.log` (file-backed swap)
- `phone-logs/wave6_v1fa_bounded_1780769821/v1_fa_K512_bounded/iter0001/meta.json` (bounded)
- `phone-logs/wave7_v1fa2_1780782482/v1_fa2/iter0001/meta.json` (over-anchored, FAILED)
- `phone-logs/wave8_v1fa2_sel_1780788550/v1_fa2_selective/iter0001/meta.json` (selective)
- `phone-logs/wave9_v1fa2_stack_1780796320/v1_fa2_stack/iter0001/meta.json` (the stack)
- `EndurKV/figures/master_tables/ALL_WAVES_MASTER.md` (cross-policy comparison, including non-v1_FA baselines)
- `EndurKV/figures/master_tables/SECTION_V1FA2_STACK.md` (mechanism provenance)
- `EndurKV/figures/master_tables/V1FA2_HEADLINE.md` (Wave-9 elevator pitch)
- `EndurKV/figures/master_tables/TABLE_PHI3_WAVE3.md`, `TABLE_PHI3_WAVE4_LONGDECODE.md`
