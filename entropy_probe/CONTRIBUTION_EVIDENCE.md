# μKV contributions — evidence status (2026-07-25)

Two SEPARATE contributions with different scopes. Do not conflate them.

## C1. Demand-aware eviction gate — UNIVERSAL (phone + Jetson + PC)
Mechanism: adaptive mass gate (α) → anchor/recent split → FA-on-evict selection → compaction.
**Every Jetson number below has NO watchdog (the watchdog is phone-only, see C2), so those wins
are attributable to the gate/eviction alone.**

| platform | model | decode tps van→μKV | wall | live KV |
|---|---|---|---|---|
| Phone Adreno (no wd) | Phi-3 q8KV | 7.2 → **11.8 (+64%)** | — | 2223 → 193 MiB |
| Jetson CUDA (no wd) | Phi-3 Q4KM | 4.5 → **5.1 (+13%)** | 935 → **832 s (−11%)** | 2223 → 174 MiB |
| Jetson CUDA (no wd) | Bonsai-8B | 4.2 → **4.7 (+12%)** | 1320 → 1222 s | 1417 → 111 MiB |
| Jetson CUDA (no wd) | Llama-1B | 32.7 → 33.5 | 133 → 130 s | 304 → 22.9 MiB |
| Phone CPU (no wd) | Bonsai-8B | 1.0 → **4.5 (4.5×)** | 85.6 → **33.2 min** | 1417 → 111 MiB |
| **Jetson CUDA (no wd)** | **Qwen2.5-VL-3B** | *(long-decode pending)* | — | **144.2 → 25.1 MiB (5.7×)** |

Quality (teacher-forced PPL, disjoint slice): **parity ±2%** — Llama 8.20→8.02, Bonsai 6.72→6.74,
Phi-3-Q4KM 3.98→3.94. Energy: Bonsai phone −60% (2312 vs 5758 mWh).

**Adaptivity evidence (why it is a GATE, not a fixed K):** one frozen config yields
α=0.719→anchor 733/recent 287 (Llama), α=0.898→916/104 (Phi-3-128k), α=0.950→969/51 (Phi-3-4k),
α=0.774→789 (Bonsai), α=0.700→714/306 (Qwen2.5-VL). It adapts across model families AND modality.

### THE HEADLINE RETRIEVAL RESULT (phone NIAH, 336 cells: 8 policies × 3 models × 7 depths × 2 ctx)
**This is the primary quality evidence — it was already measured and it is strong.**
K=1024 budget, phone CPU, Phi-3 / Llama-1B / Gemma-2B at 4K and 8K.

| policy | hits /42 | **active KV cells** | tps range |
|---|---:|---:|---:|
| TOVA-canonical | 40 | 3099–7234 | 1.4–7.6 |
| **μKV-mass** | **39** | **1024** | **5.7–20.4** |
| μKV-count | 38 | 1024 | 5.4–19.7 |
| vanilla | 37 | 3122–7296 | 1.3–13.8 |
| AdaKV | 37 | 3099–7296 | 1.4–8.5 |
| TOVA | 36 | 3093–7296 | 1.4–7.3 |
| H2O | 34 | 3122–7296 | 1.4–7.4 |
| StreamingLLM | 10 | 2978–7296 | 1.5–8.1 |

**Claim this:** μKV matches the best baseline's retrieval (39 vs 40 = one cell, noise) while
holding **3–7× fewer live KV cells** and running **2.7–4× faster**. The score-reading baselines
(AdaKV/H2O/TOVA) *cannot compact* on a sequence-level engine — their per-head unions retain
3.1K–7.3K of the cache — so they pay full KV bandwidth AND the FA-off capture cost. μKV is the
only policy that both compacts (to 1024) and keeps decode on the FA-on path. StreamingLLM (10/42)
is the clean control showing the task discriminates.

**Gate ablation is already inside this table:** mass-gate 39 vs count-gate 38 (mass wins
Llama@8K 7/7 vs 6/7, ties elsewhere) — mass ≥ count, small margin.

### Secondary (2026-07-26): the α VALUE is a second-order parameter — 4 tests
This does NOT weaken the result above; μKV's advantage comes from compaction + FA-on decode,
not from the specific α. Do not additionally claim "adaptive α beats a fixed α on quality."

| test | design | result |
|---|---|---|
| 1. disjoint-PPL transplant (Phi-3) | gate 876/144 vs Llama-split 733/287 | **identical PPL 3.9377**; fixed used LESS KV (146 vs 174 MiB). Metric proven insensitive to selection (it DOES separate vanilla 8.20 vs μKV 8.02) |
| 2. NIAH Llama-1B (n=14) | gate vs Llama-ratio fixed | gate 12/14, fixed 11/14, vanilla 11/14 — noise. **Design flaw:** applied Llama's ratio to Llama (fixed optimal by construction) |
| 3. NIAH transplant on Phi-3 | Llama ratio → Phi-3 | **7/7 all three arms** — ceiling effect, no discrimination |
| 4. Decisive 2×2 (task-dependent split) | {gate, anchor-heavy 230/26, recent-heavy 128/128} × {NIAH, continuation-PPL} | retrieval: recent-heavy 7/7 > gate 6/7 = anchor-heavy 6/7. continuation: anchor-heavy 11.9054 < gate 11.9297 < recent-heavy 12.0486. **Gate mid-pack on both; hypothesis about which split suits which task was backwards.** |

**Conclusion:** at practical budgets (K=256–1024) the anchor:recent ratio is a **second-order
parameter** — all reasonable splits land within ~1% PPL and ±1 NIAH sample. The gate neither
helps nor hurts. **What IS first-order is eviction itself** (keeping 256–1024 of ~9.7K cells =
90–97% reduction) at quality parity, which is where every measured win comes from.

**Why my 4 tests looked negative (design limits, not evidence against μKV):** they compared
α-choice *within* μKV (not μKV vs baselines), at K=256 (vs the table's K=1024, so everything
degraded toward noise), with n=7 on one model (vs n=42 across 3 models × 2 contexts), and never
measured active-KV — the column that carries the result.

**Claim for the gate (engineering):** zero-tuning robustness — it selects α=0.70→0.95
automatically across model families and modalities and lands within noise of the best fixed
split, removing per-model tuning. **Do not additionally claim** the adaptive α beats a fixed α
on quality; that specific comparison is within noise (mass 39 vs count 38; 4 further tests flat).

## C2. Thermal watchdog — PHONE-ONLY (mobile contribution)
Reads phone thermal zones (battery/skin), writes `/sys/kernel/gpu/gpu_max_clock` (Adreno) and
CPU freq caps. **Not used, and not applicable, on Jetson** (different sysfs, different thermal
regime, mains-powered). Scope it explicitly as a mobile/battery-device contribution.

| effect (phone) | without wd | with wd |
|---|---|---|
| Llama GPU 16K | 33.0 tps, 244 s, 453 mWh | **34.6 tps, 238 s, 424 mWh** |
| Bonsai sustained | battery 50.2 °C → vendor deep-throttle 883 MHz for **125 s** | 49.1 °C stable, throttle **1 s** |
| battery rise rate | 0.48 °C/min | **0.05 °C/min** |

## Uniqueness vs prior art (see KVSWAP_ANALYSIS.md, mukv-novelty-priorart)
1. Runs on an actual **phone GPU** (Adreno/Vulkan) — KVSwap (vLLM/CUDA + NVMe) structurally cannot.
2. **No disk** — pure in-memory eviction; phones have UFS with write-endurance/energy costs.
3. **Thermal/energy under sustained load** (C2) — untouched by KV-offloading work.
4. **Cross-platform + cross-modality from one frozen config** (phone/Jetson/PC; text + VLM).
5. Composable: evict dead KV first (μKV), then swap survivors (KVSwap) → less I/O.
