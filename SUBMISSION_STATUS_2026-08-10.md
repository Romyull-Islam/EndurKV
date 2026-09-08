# What is presentable — audit of every result produced 2026-08-05 … 08-10

Three categories. Nothing in READY has a known defect; everything in CAVEATED is usable
if the stated caveat is written into the caption; everything in DEAD must not appear.

---

## READY — use as-is

### R1. Phone GPU, sustained load (the strongest new result)
`/tmp/gpu_sustained` · table via inline script · figure `fig_gpu_sustained_thermal.pdf`

Adreno 840, Vulkan, 17/17 layers offloaded. 8 back-to-back iterations of
[12220-token prompt + 4096 generated], cool gate (DDR≤35 °C, batt≤33 °C) once per arm,
no gate between iterations so the device reaches equilibrium.

| arm | prefill | decode | wall | dec tok/s | cells | evicted | energy | mean W |
|---|---|---|---|---|---|---|---|---|
| vanilla | 32.3 m | 24.4 m | 56.7 m | 22.33 | 12224 (100%) | 0 | 11672 J | 3.44 |
| µKV | 33.4 m | 17.5 m | 51.2 m | 30.05 | 766 (6.3%) | 12486 | 10666 J | 3.45 |
| µKV + v5LOW | 30.7 m | 15.8 m | 46.7 m | 35.71 | 766 (6.3%) | 12486 | 10688 J | 3.83 |
| µKV + v5HIGH | 33.9 m | 17.4 m | 51.5 m | 31.66 | 766 (6.3%) | 12486 | 11021 J | 3.54 |

Why it is solid: v5HIGH fired ZERO times, so it is an accidental replicate of the
no-watchdog arm — the two agree to 0.8% on the mean, which sets the noise floor and puts
v5LOW's +10.8% far outside it. All µKV arms are `compaction_mode=inplace` with an
identical 766-cell keep-set, so the watchdog is provably orthogonal to eviction.

Say: 1.21× wall, 1.60× decode over vanilla; the watchdog is a TIME optimisation, not an
energy one (10% shorter × 11% more power = same joules).
Do NOT say: that the watchdog keeps the phone cool — it runs hotter on purpose.

### R2. Phi-3 at ctx 16384 — a configuration previously reported infeasible
`/tmp/phi3_cooled` (n=2 per arm, interleaved, gate + settle) and `/tmp/phi3_phone`

| arm | prefill | wall | cells | PPL |
|---|---|---|---|---|
| vanilla | 666.0 s | 1685.1 s | 12238 (100%) | 7.2924 |
| µKV + in-place | 703.4 s | 1529.6 s | 874 (7.1%) | 7.9007 |
| µKV + round-trip | — killed — | | | |

Round-trip dies allocating the SECOND 6144 MiB KV cache; reproduced twice with logcat
attached, and Android SIGKILLed six Zygote app processes at the same moment. Same failure
on a 24 GB RTX (`second context alloc failed`), so it is a property of the mechanism, not
of phone RAM. Prefill is 1.056× SLOWER (the scoring cost, matching CUDA's 1.088×) and wall
is 1.10× faster. PPL reproduced to 4 decimals across both run orders.

### R3. FA-off penalty — the strongest systems argument
Phone, 16K, every baseline at ITS OWN published budget:

| policy | prefill | tok/s | vs vanilla | cells kept |
|---|---|---|---|---|
| vanilla | 131.1 s | 28.63 | 1.00× | 9741 (100%) |
| µKV | 133.7 s | 39.47 | 1.38× | 723 (7.4%) |
| SnapKV | 228.6 s | 4.76 | 0.17× | 6592 (67.7%) |
| Ada-KV | 276.1 s | 2.86 | 0.10× | 3519 (36%) |
| H2O | 278.8 s | 3.58 | 0.13× | 6110 (63%) |
| TOVA | 175.9 s | 4.34 | 0.15× | 4125 (42%) |
| StreamingLLM | 175.9 s | 5.55 | 0.19× | 777 (8.0%) |

StreamingLLM keeps 8.0% of cells and is still 5× SLOWER than vanilla; µKV keeps 7.4% and
is 1.38× FASTER. Same retention, opposite outcome — on-device speed is governed by whether
the policy can keep flash-attention ON, not by how much it evicts.

### R4. Realizability gap (retention only — no PPL dependency)
| model / context | SnapKV budget | cells actually retained |
|---|---|---|
| Llama-1B, phone 16K | 2048/head | 6592 = 67.7% |
| Phi-3, phone 16K | 2048/head | 9468 = 84.9% |
| Llama-1B, 64K CUDA | 2048/head | 30634 = 53.6% |
| H2O, 64K CUDA | 20% of N | 57039 = 99.8% |

### R5. In-place vs round-trip compaction — cross-model / cross-policy
`/tmp/verify_others`. Identical keep-sets, PPL within FP reassociation (0.010–0.096%)
across Phi-3, Mistral-7B, StreamingLLM, SnapKV (scattered per-head union), TOVA (frozen).
Two limits belong in the text: in-place DECLINES on sliding-window models (Gemma-2) and
falls back, so the round-trip cannot be deleted; and the two modes are NOT interchangeable
for decode-time evictors, because the round-trip's context swap drops the attention capture
and silently converts TOVA into a freeze-after-prefill policy (1.41% → 0.096% once frozen).

---

## CAVEATED — usable only with the caveat in the caption

### C1. Phone GPU cold-start matrix (watchdog × compaction), `/tmp/phone_wd_matrix`
Speed/energy valid, PPL valid (re-measured against a slice asserted 0/119 disjoint):
vanilla 23.3045, µKV round-trip 22.7150, in-place 22.7160, no-compaction 22.3839.
CAVEATS: n=1 per cell; spread across six identical µKV arms is 3.3% wall / 11.2% energy,
so the −12% energy claim is roughly one noise width and vanilla is n=1. The watchdog rows
are a NULL — both ladders logged zero steps because a cold-start run peaks at 35 °C
battery, below v5LOW's 36 °C trigger. Report that as "untested under this protocol",
never as "the watchdog does not help".

### C2. LongBench retrieval, `/tmp/lb_cuda` (7511 cells)
µKV beats StreamingLLM by +8.8 F1 (Llama-1B) and +11.2 (Phi-3) at matched retention;
on Phi-3 µKV exceeds the full cache at 12.2% retention.
CAVEAT: only hotpotqa and qasper are scored; some Phi-3 cells have n=15.

### C3. Compaction is quality-neutral at 64K
The AGREEMENT between arms survives (same contaminated slice for all, so identical PPL
still proves identical keep-sets). The ABSOLUTE values do not — see D1.

### C4. Vulkan-vs-CUDA numerical penalty
Same model, prompt, config and keep-set: vanilla PPL 13.68 (CUDA) vs 23.30 (Vulkan),
**+70%** on Llama-1B; +19% on Phi-3. Real and worth a footnote, but it means cross-device
PPL comparison is invalid — quote within-device ratios only.

---

## DEAD — must not appear

### D1. EVERY perplexity number measured at ctx 65536
The 57344-token prompt is `wiki.test.raw[0:251502]` and overlaps `wiki_eval_disjoint`
119/119 windows and `wiki_eval_disjoint_long` 70/119. Those slices were verified against
the 12288-token prompt only and the verification was silently reused. So all 64K PPL
scored VERBATIM RECALL: vanilla read 2.98 where the same slice reads 15.70 at 16K.
Kills the PPL column of: the 64K three-way compaction table, the `--k-pct` sweep, and the
no-limit policy comparison. The K-sweep narrative "quality improves 13.39 → 3.89 as K
grows" is really "recall improves as you retain more" and is near-tautological.
SURVIVES from those runs: all speed, wall, cells, evicted, memory and energy columns.
FIX: `benchmarks/ppl/wiki_eval_disjoint_64k.txt` = `raw[252000:276000]`, verified 0/119 and
0/399 against all three prompts. ~10 cells to re-run.

### D2. Any Gemma-2 row at ctx 16384
`n_ctx_train` is 8192; running at 16384 produces word salad (PPL 1262). Re-run at 8192 or
drop.

### D3. The Phi-3 phone timings from the FIRST (uncooled-order) pair
Reported 1.22× faster prefill / 1.37× faster wall. An order-swap control showed the
advantage follows POSITION, not policy: whichever arm ran first won, and vanilla swung
501 → 682 s (36%) between positions. Superseded by R2.

### D4. All GPU clock-distribution percentages from the sustained run's RAW traces
The per-arm samplers were never torn down, so each trace contains its own arm plus all
later arms — vanilla's file spans 287 min instead of 47. The figure repairs this by
time-truncating each trace to its own window; the untruncated numbers (e.g. "v5LOW
eliminates the 422 MHz state", "lower mean cap yet higher throughput") are void.

### D5. "Peak RSS" as a memory-saving metric
On both CUDA and Vulkan the KV lives in device memory that RSS cannot see, and the cache
is allocated at full n_ctx up front regardless of policy — so RSS shows only µKV's scoring
overhead (+20–25 MiB) and none of its saving. Quote retained CELLS.

---

## Figures

| figure | status |
|---|---|
| `fig_gpu_sustained_thermal.pdf` (new) | READY — traces time-truncated to repair D4; skin from NAMED `shell_*` channels, not the stale hardcoded zone IDs that were reading a 95 °C junction sensor |
| `fig_bonsai_thermal.pdf` (existing) | READY — CPU, Bonsai-8B; label it CPU explicitly, since the new one is its GPU counterpart |
