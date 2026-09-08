# Wave-11 Phi-3-mini-128k K=512 — Honest Throttle Audit

**Source:** `phone-logs/wave11_eval_1780862534_K512_snapshot/wave11_eval_1780862534/Phi-3-mini-128k/<policy>/ppl/`
**Date:** 2026-06-08
**Audit performed by:** subagent (post-hoc on snapshot logs)

The existing master table marks every Wave-11 cell as "CLEAN" because it only
checks `cpu*_cool_state > 0`. Per the kernel-trip-point sweep in
`PHONE_THROTTLE_THRESHOLDS.md`, that signal *can never trip* until DDR ≥ 100 °C
or big-CPU ≥ 95 °C — neither of which Wave-11 ever approached. So `cool_state`
is a near-useless throttle indicator for this hardware/workload.

This audit looks for **hidden throttle** that is invisible to `cool_state` but
still demonstrably slows the SoC down:

1. **PREFILL_MS DRIFT** across `iter0..iter7` (normalised by `n_prompt_tokens`).
   Same model, same K, same code-path → if `ms/token` grows, the SoC is doing
   the same work more slowly. That is throttle, even if the kernel doesn't
   say so.
2. **CPU6 (prime-core) frequency** distribution from `sensors.csv` (column is
   labelled `cpu6_freq_hz` but the values are in **kHz** — verified by
   inspection: nominal idle states are 1632000 / 1497600 / 1382400 / 1267200
   etc., matching the Snapdragon 8 Elite prime-core OPP table). Anything
   sustained below 1.3 GHz on a prime core during a steady compute-bound
   prefill is a freq cap.
3. **Battery-current pattern.** Wave-11 sensor logs have `usb_online`
   *empty*, so we cannot disambiguate charge / discharge transitions. We
   therefore use the *magnitude* of `bat_current_ma` and look for sustained
   absolute drops as a weak BCL signature.
4. **Cool state** — kept only for completeness. It is 0 everywhere.
5. **Memory pressure.** `mem_avail_kb` start vs end, and `ddr_temp_mc`.
   No `progress.log` was produced in this run, so the per-chunk DDR/free
   numbers are taken from `sensors.csv` instead.

---

## 1. Raw per-iter prefill drift (Phi-3, K=512, PPL)

`ms_per_token = prefill_ms / n_prompt_tokens`. Normalising by token count
removes the effect of varying chunk size and isolates per-token throughput.

### vanilla (8 iters)
| iter | n_tok | prefill_ms |   ms/tok |
|------|-------|------------|---------:|
|   0  | 2175  | 249266     |  114.60  |
|   1  | 2129  | 283272     |  133.05  |
|   2  | 1992  | 263197     |  132.13  |
|   3  | 2268  | 303804     |  133.95  |
|   4  | 2301  | 309757     |  134.62  |
|   5  | 2567  | 350275     |  136.45  |
|   6  | 2701  | 366574     |  135.72  |
|   7  | 2338  | 314148     |  134.37  |

Drift iter0 → iter7: **+17.2 %**. iter0 is the outlier on the *fast* side
(cold-cache + still-cool DDR); the rest of the run sits in a tight 132-136
ms/tok band — i.e. a step-down from ~115 → ~135 ms/tok after the first chunk
and then **flat**. Consistent with a one-time DDR-bandwidth saturation, no
on-going freq cap.

### h2o (7 iters — iter7 missing)
| iter | n_tok | prefill_ms |   ms/tok |
|------|-------|------------|---------:|
|   0  | 2175  | 293159     |  134.79  |
|   1  | 2129  | 334596     |  157.16  |
|   2  | 1992  | 311792     |  156.52  |
|   3  | 2268  | 358644     |  158.13  |
|   4  | 2301  | 363466     |  157.96  |
|   5  | 2567  | 424937     |  165.54  |
|   6  | 2701  | 477543     |  176.80  |

Drift iter0 → iter6: **+31.2 %**. After the iter0 step-down to ~157, the
last two iters climb to 165 → 177 ms/tok — a clean monotonic creep, ~12 %
on top of the steady-state baseline. h2o's per-token attention bookkeeping
is heavier than vanilla, so DDR pressure ramps faster.

### tova (7 iters — iter7 missing)
| iter | n_tok | prefill_ms |   ms/tok |
|------|-------|------------|---------:|
|   0  | 2175  | 293106     |  134.76  |
|   1  | 2129  | 299076     |  140.48  |
|   2  | 1992  | 301122     |  151.17  |
|   3  | 2268  | 349629     |  154.16  |
|   4  | 2301  | 363396     |  157.93  |
|   5  | 2567  | 396205     |  154.35  |
|   6  | 2701  | 436200     |  161.50  |

Drift iter0 → iter6: **+19.8 %**. Slower, gentler ramp than h2o, but still
monotonic in the average — no flat plateau like vanilla.

### v1_fa2_stack (8 iters)
| iter | n_tok | prefill_ms |   ms/tok |
|------|-------|------------|---------:|
|   0  | 2175  | 286785     |  131.85  |
|   1  | 2129  | 328114     |  154.12  |
|   2  | 1992  | 335446     |  168.40  |
|   3  | 2268  | 369709     |  163.01  |
|   4  | 2301  | 373833     |  162.47  |
|   5  | 2567  | 406796     |  158.47  |
|   6  | 2701  | 427897     |  158.42  |
|   7  | 2338  | 363667     |  155.55  |

Drift iter0 → iter7: **+18.0 %**. iter2 is the worst (168 ms/tok), then the
SoC *recovers* monotonically (168 → 163 → 162 → 158 → 158 → 155). This is
the classic "warmed up then the kernel governor settles" curve — bad early,
slowly improving. End-of-cell is only +18 % above iter0; mid-cell was +28 %.

### streamingllm (6 iters — iter6,7 missing)
| iter | n_tok | prefill_ms |   ms/tok |
|------|-------|------------|---------:|
|   0  | 2175  | 292949     |  134.69  |
|   1  | 2129  | 286356     |  134.50  |
|   2  | 1992  | 278615     |  139.87  |
|   3  | 2268  | 358416     |  158.03  |
|   4  | 2301  | 363366     |  157.92  |
|   5  | 2567  | 391038     |  152.33  |

Drift iter0 → iter5: **+13.1 %**. iter0..iter2 are tight (134-140 ms/tok),
then a step-up to 158 at iter3 and back to 152 at iter5. The last data
point we have is *recovering*, not degrading.

---

## 2. CPU6 (prime-core) frequency distribution

Units: kHz in raw CSV (despite the column header `_hz`), converted to GHz.

| policy        | mean | min  | p10  | max  | % < 1.3 GHz | % < 1.0 GHz | % < 0.8 GHz | Q1-mean | Q4-mean | Q1→Q4 Δ |
|---------------|-----:|-----:|-----:|-----:|------------:|------------:|------------:|--------:|--------:|--------:|
| vanilla       | 1.339| 0.883| 1.018| 1.632| **15.4 %**  |  0.3 %      |  0.0 %      | 1.355   | 1.353   |  −0.001 |
| h2o           | 1.389| 0.883| 1.382| 1.632|   3.1 %     |  0.1 %      |  0.0 %      | 1.455   | 1.363   |  −0.092 |
| tova          | 1.447| 1.267| 1.382| 1.632|   0.2 %     |  0.0 %      |  0.0 %      | 1.545   | 1.384   |  −0.161 |
| v1_fa2_stack  | 1.298| 0.883| 1.018| 1.632| **49.5 %**  |  0.4 %      |  0.0 %      | 1.378   | 1.336   |  −0.042 |
| streamingllm  | 1.457| 1.267| 1.382| 1.632|   0.1 %     |  0.0 %      |  0.0 %      | 1.597   | 1.420   |  −0.177 |

Q1-mean / Q4-mean = mean CPU6 freq over the first / last quartile of
sensor samples in the cell.

Notes:
- `v1_fa2_stack` spent **half** of the cell below 1.3 GHz on the prime
  core — a clear sustained freq depression, even though `cool_state` never
  flipped. The Q1→Q4 swing is small (−42 MHz) because the cell was *born*
  in that depressed state.
- `vanilla` had 15 % of samples below 1.3 GHz but its Q1 and Q4 means are
  identical (1.355 vs 1.353 GHz) — the dips were uniformly distributed,
  not a late-cell collapse.
- `tova` and `streamingllm` show large Q1→Q4 freq drops (−161 and −177 MHz
  respectively) but those drops *land at 1.38-1.42 GHz*, still above the
  1.3 GHz "throttle" threshold. This is governor settling, not a cap.
- `h2o` is the cleanest of the heavyweight policies: 97 % of samples ≥
  1.3 GHz, gentle Q1→Q4 drift.

No policy ever sustained sub-1.0 GHz operation. Minimum observed freq is
**883 MHz** (vanilla, v1_fa2_stack, h2o) but that's a small fraction (≤ 0.4 %)
of the samples — likely brief governor undershoots, not a sustained cap.

---

## 3. Battery current pattern (BCL proxy)

`bat_current_ma` is signed and noisy. Per-octile means within the cell:

| policy        | oct1 | oct2 | oct3 | oct4 | oct5 | oct6 | oct7 | oct8 |
|---------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| vanilla       | +288 | +204 | +201 | +271 | +251 | +258 | +230 | +282 |
| h2o           | +145 | +232 | +228 | +229 | +226 | +194 | +219 | +252 |
| tova          | −236 | −206 | −188 | −180 | −131 | +231 |  −51 | −146 |
| v1_fa2_stack  | −259 |  −70 | +207 | +207 | +205 | +224 | +230 | +251 |
| streamingllm  | +249 | +264 |  +91 | −202 | −183 | +261 | +143 | −210 |

Several cells show sign flips that are not explainable by USB plug-in
events (`usb_online` column is empty throughout). Interpreting these as a
classical Qualcomm BCL clamp is *not* reliable here — the magnitude does
not decline monotonically, and current magnitudes stay in the 150-300 mA
band throughout. **No clean BCL signature** in any of the five cells.

---

## 4. Cool-state (existing master-table check)

All `cpu0..cpu7 cool_state` values are **0** for every sample in every cell,
confirming the master table's `Throttle=CLEAN` reading is *consistent* with
this signal — and confirming that this signal is too coarse to detect what
the prefill-drift and freq-distribution data above are showing.

---

## 5. Thermal envelope

| policy        | DDR start | DDR end | DDR max | shell_back max | BCL-L0 max |
|---------------|----------:|--------:|--------:|---------------:|-----------:|
| vanilla       | 49.9 °C   | 62.8 °C | **64.8 °C** | 47.6 °C       | n/a (0)    |
| h2o           | 43.1 °C   | 61.6 °C | 62.9 °C | 47.6 °C       | n/a (0)    |
| tova          | 42.0 °C   | 58.8 °C | 59.4 °C | 44.5 °C       | n/a (0)    |
| v1_fa2_stack  | 49.2 °C   | 62.4 °C | **66.0 °C** | 47.6 °C       | n/a (0)    |
| streamingllm  | 43.9 °C   | 53.3 °C | 59.4 °C | 44.8 °C       | n/a (0)    |

All DDR maxima are far below the 100 °C kernel trip — but consistent with
the *workload-emergent* DDR-saturation effect described in
`PHONE_THROTTLE_THRESHOLDS.md` (DRAM controller arbitration slows down well
before the kernel-visible trip).

The two cells with the worst DDR thermal (vanilla 64.8 °C, v1_fa2_stack
66.0 °C) are also the two with the highest %<1.3 GHz CPU6 dips. **DDR
saturation, not kernel CPU throttle, is the most plausible mechanism** for
the prefill-drift we see.

---

## 6. Honest verdict per cell

Verdict thresholds (per task brief):
- **CLEAN** — no evidence of any throttle.
- **MILD** — prefill_ms drift 5–15 % OR freq dipped to ~1.3 GHz briefly.
- **MODERATE** — prefill_ms drift 15–30 % OR BCL signature.
- **SEVERE** — > 30 % drift, sustained freq < 1.0 GHz, or both BCL + freq drop.

| policy        | drift % | cpu6 < 1.3 GHz | cool_state | BCL evid. | Honest verdict |
|---------------|--------:|---------------:|:----------:|:---------:|:--------------:|
| vanilla       |  +17.2  | 15.4 %         | 0          | none      | **MODERATE**   |
| h2o           |  +31.2  |  3.1 %         | 0          | none      | **SEVERE**     |
| tova          |  +19.8  |  0.2 %         | 0          | none      | **MODERATE**   |
| v1_fa2_stack  |  +18.0  | **49.5 %**     | 0          | none      | **MODERATE**   |
| streamingllm  |  +13.1  |  0.1 %         | 0          | none      | **MILD**       |

Why these:

- **vanilla — MODERATE.** Drift sits at 17 %, just above the MILD ceiling,
  and 15 % of CPU6 samples fell below 1.3 GHz. The per-iter table shows a
  clean step-down after iter0 then a *flat* plateau, so this is most
  consistent with a one-time DDR saturation, not a runaway cap — but the
  iter0 → iter7 number is the throttle indicator the brief asks for.
- **h2o — SEVERE.** +31.2 % drift crosses the SEVERE threshold cleanly,
  and the ramp is monotonic (157 → 158 → 158 → 165 → 177 ms/tok at the
  tail). CPU6 freq stayed high (97 % above 1.3 GHz), so the cap is on the
  memory subsystem, not the prime core. iter7 is missing — quite possibly
  this cell aborted because the prefill budget was busting.
- **tova — MODERATE.** +19.8 % drift, monotonic ramp, but freq stays
  pristine (only 0.2 % of samples below 1.3 GHz). DDR thermal peak is the
  lowest of any policy (59.4 °C), so the slowdown likely tracks growing KV
  cache eviction overhead, not phone-level throttle. Still, by the brief's
  drift rule this is MODERATE. iter7 is missing here too.
- **v1_fa2_stack — MODERATE.** Drift is +18 % (border of MILD/MOD), but
  this is the only cell where the *frequency itself* spent half the cell
  below 1.3 GHz. The iter trace shows iter2 was the worst (168 ms/tok)
  and then the SoC *recovered* down to 155 by iter7. Borderline; we keep
  MODERATE because of the freq signature.
- **streamingllm — MILD.** Drift +13.1 %, freq pristine (99.9 % above
  1.3 GHz), and the last data point we have is *recovering*. The cell is
  the cleanest of the five despite missing iter6/iter7.

---

## 7. What this changes about the master-table reading

The master table marks all five Wave-11 K=512 Phi-3 cells as
**Throttle=CLEAN** because it only checks `cool_state`. By a more honest
definition that also catches workload-emergent DDR saturation and freq
depression that the kernel cdev never registers, only **streamingllm**
deserves a CLEAN-ish ("MILD") tag. The rest are MODERATE → SEVERE.

This does not invalidate the PPL numbers per se — each policy was hit
with the same workload-emergent backpressure — but it does mean the
*latency* numbers (and any latency-derived efficiency claim) should carry
a footnote: **K=512 Phi-3 runs are running into DDR-bandwidth /
DDR-thermal saturation, not into a clean-state CPU.**

---

## Appendix: how to reproduce

```bash
python3 /tmp/audit_throttle.py
# reads each <policy>/ppl/iterNNNN/meta.json
# and each <policy>/ppl/sensors.csv
# emits /tmp/audit_results.json
```

Key fact-checks done during this audit:
- `cpu6_freq_hz` column values were verified against the SM8750-AB prime-core
  OPP table (1632 / 1497.6 / 1382.4 / 1267.2 / 1017.6 / 883.2 MHz). Values in
  the CSV match in **kHz**, so the column name is mislabelled.
- `usb_online`, `usb_current_ua`, `usb_voltage_uv` are *empty* throughout
  every sample in every cell — the device probably wasn't reporting them
  via the path the harness was reading.
- No `progress.log` is present in this snapshot, so the per-chunk DDR /
  free-memory readout the brief asked for comes from `sensors.csv` instead.
