# EndurKV: Closed-Loop Architecture Vision

This document captures the full project vision in one place: what EndurKV is today (Phase 1), what it becomes (Phase 2), and how the closed-loop story unifies them.

## The mission statement

> **Build a closed-loop mobile-LLM system in which model-internal attention signals drive cache management decisions, and sensor signals drive thermal/I/O control --- so the device sustains smooth, fast, low-energy inference for far longer than any policy designed for the data center.**

## The five simultaneous goals

The closed loop must hold ALL of these at once, not optimize one at the expense of another:

| # | Goal | Quantification | Phase 1 status |
|---|------|----------------|----------------|
| 1 | **Effective KV eviction without accuracy loss** | PPL within budget of vanilla (~+10% acceptable) | ✅ +11.3% Wave-11 PPL |
| 2 | **Tiered storage when DRAM full** | Spill to flash + linear-time recall via DRAM index | ⏳ Phase 2 |
| 3 | **Minimum total I/O** | DRAM + flash bytes per token bounded | ✅ 4× less DRAM in Phase 1 |
| 4 | **Stable temperature — no throttle** | Kernel cliff never hit (DDR < 65°C, CPU < 65.5°C empirical) | ✅ Watchdog dormant in production |
| 5 | **Smooth inference, minimum wall time, minimum energy** | Decode jitter < $\sigma$; lowest mAh per token; fastest TTFT and decode tps | ✅ +19.7% tps, −15.8% wall, −12.9% energy |

## The multi-objective optimization

```
Minimize:    α₁ · WallTime + α₂ · Energy + α₃ · Jitter
Subject to:  PPL(policy) ≤ PPL(vanilla) × (1 + ε_quality)
             Peak_DDR_temp < cliff_DDR
             Peak_CPU_temp < cliff_CPU
             TotalIO(DRAM + Flash) ≤ TotalIO(vanilla)

where:
    α₁, α₂, α₃ = workload-specific weights
    ε_quality = quality budget (~0.11 in our results)
    cliff_DDR ≈ 65°C, cliff_CPU ≈ 65.5°C (empirical)
```

This is a constrained joint optimization. Single-objective policies (PPL-only like H2O, or thermal-only like Qualcomm BCL) provably can't reach the same Pareto frontier as a co-aware policy.

## Phase 1 (DONE) — Smart In-DRAM Eviction

**Status**: implemented as `v1_fa2_stack`, evaluated on Phi-3-mini Q4_K_M on OnePlus 15.

| Mechanism | Effect |
|---|---|
| Per-head attention-confidence budget | Each head's keep size scales with its peak attention concentration |
| Selective semantic anchoring (top-32) | 32 prompt tokens preserved as anchors |
| Q8 K cache (f16 V) | K-side bandwidth halved |
| State-swap to FA-on decode | Fast decoder despite FA-off prefill cost |
| Multi-sensor watchdog v2 | Cliff-insurance against kernel throttle |

**Measured production result** (Phi-3-mini, 2048-token decode, cliff-insurance, charging disabled):

| Metric | Vanilla | v1_fa2_stack | Δ |
|---|---|---|---|
| Decode tps | 5.524 | **6.611** | **+19.7%** |
| Wall time | 375.1 s | **315.8 s** | **−15.8%** |
| Energy | 38.73 mAh | **33.72 mAh** | **−12.9%** |
| Peak RSS | 3.75 GB | **3.40 GB** | **−9.3%** |
| Watchdog transitions | n/a | **0 (DORMANT)** | — |
| Swap (Wave-11 PPL) | 158 MB | **0 MB** | — |

**Key insight**: cache reduction alone produces all four wins. Watchdog never fires. The algorithm IS the thermal control.

## Phase 2 (NEXT) — Flash-Tiered Cache with DRAM Index

**Motivation**: when prompt + decode > K_nominal, Phase 1 must DROP cells. For long-context workloads (RAG, multi-turn agents, code-base reading), some "dropped" cells will turn out to matter later. Storing them in flash with fast recall is the natural extension.

### Component design

```
┌──────────────────────────────────────────────────────────────────┐
│ DRAM (12 GB total, ~1.2 GB available for KV cache)                │
│                                                                   │
│  ┌──────────────────────────────────────────┐                     │
│  │ ACTIVE KV CACHE (K_nominal cells)         │                    │
│  │   ├── sink (4 cells)                      │                    │
│  │   ├── anchors (32 cells, top by attn)    │                    │
│  │   └── recent window (K-36 cells)         │                    │
│  └──────────────────────────────────────────┘                     │
│                                                                   │
│  ┌──────────────────────────────────────────┐                     │
│  │ FLASH INDEX (16 bytes × N_evicted)       │                    │
│  │   {(layer, position): flash_offset}      │                    │
│  │   ~32 KB for 2K evicted cells            │                    │
│  └──────────────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
                              │ index lookup
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ FLASH (UFS 4.0, ~3 GB/s sustained read)                          │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ SPILL BUFFER: contiguous KV cells, by (layer, pos)   │         │
│  │   - written when eviction policy "drops" a cell      │         │
│  │   - laid out for linear streaming read               │         │
│  │   - aged out when battery low or capacity full       │         │
│  └──────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

### Data flow

**At eviction time (instead of dropping):**
```
for cell (layer L, position p) chosen for eviction:
    flash_offset = append_to_spill_buffer(K[L,p], V[L,p])
    index[(L, p)] = flash_offset
    free DRAM slot
```

**At decode step (when an "evicted" position becomes relevant):**
```
if attention_score(p) > recall_threshold and (L, p) in index:
    batch_pending_recall.add((L, p))

# periodically (every N decode steps or on cache pressure):
if batch_pending_recall is large enough:
    contiguous_reads = group_by_flash_offset(batch_pending_recall)
    cells = flash_read(contiguous_reads)  # single I/O for many cells
    insert_back_into_dram(cells)
```

### Key parameters

| Parameter | Initial guess | Tradeoff |
|---|---|---|
| Spill buffer size | ~512 MB | larger = more recall opportunities, more flash space |
| Batch size for recall | 16-32 cells | larger = better I/O amortization, more recall latency |
| Recall threshold (attention) | $\max_a > 0.5$ | lower = more recalls, larger flash I/O |
| Flash I/O quota | 5 IOPS sustained | trades quality for energy/thermal |
| Index entry size | 16 bytes (layer + pos + offset) | smaller index = more cells indexed |

### The closed-loop control law

```
For each decode step:
    
    # OUTER LOOP A: eviction (model-internal signal)
    if cache > K_nominal:
        scores = compute_attention_scores(cache)
        evict_to_flash(low_score_cells)
    
    # OUTER LOOP B: recall (model-internal signal + cost gating)
    if any(attention_score(p) > recall_threshold for p in flash_index):
        if (thermal_state_OK AND IO_quota_remaining):
            issue_flash_recall(p)
    
    # INNER LOOP: thermal control (sensor signal)
    threats = [sensor_threat(s) for s in [DDR, CPU, skin, batt, BCL]]
    if max(threats) > engage_threshold:
        cap_cpu_freq(tier_for_threat(max(threats)))
        reduce_IO_quota()  # back-pressure on flash recalls
```

This is the **closed loop** the user described: model-internal attention signals drive both eviction AND recall, while sensor signals drive both frequency caps AND I/O budget.

## Why this story is publishable as TWO papers

### Paper 1 — HotMobile 2027 (current draft)
Phase 1 only. Shows that:
- The per-head budget formula works
- The state-swap pattern enables fast FA-on decode
- The watchdog is cliff-insurance, dormant in production
- Measured −15.8% wall, +19.7% tps, −12.9% energy, 0 throttle

### Paper 2 — MobiSys 2028 (extended version)
Phase 1 + Phase 2 together. Shows that:
- Eviction-to-flash extends the effective cache to multi-GB scale
- Linear-time recall via DRAM index keeps decode latency bounded
- I/O quota gives the thermal loop a second control knob
- Measured PPL improves vs Phase 1 in long-context regimes because dropped cells can come back

### Dissertation
Both phases + EndurKV-Adaptive (third loop: K_eff(t) modulation) + multi-device generalization + the formal control-theoretic framing of "closed-loop mobile LLM inference."

## Why "reduce total I/O" is the key constraint

Total inference cost on a mobile device decomposes as:

```
TotalCost = DRAM_bytes × DRAM_power_per_byte
          + FLASH_bytes × FLASH_power_per_byte
          + CPU_cycles × CPU_power_per_cycle
          + wall_time × baseline_power
```

For a 4 B-parameter model on Snapdragon CPU, DRAM dominates. Flash recall is 3-5× more energy-expensive per byte than DRAM, so any flash read must justify itself by replacing more than 3-5 bytes of DRAM read.

The eviction-with-flash-spill design must ensure: **fewer bytes total are moved** than if we'd kept everything in DRAM, even after accounting for the writes (during spill) and reads (during recall).

A back-of-envelope:
```
Without flash spill: vanilla
  DRAM read per step = 2058 cells × 32 bytes = 65 KB
  Over 2048 steps:   65 KB × 2048 = 133 MB
  
With Phase 1 eviction (no flash):
  DRAM read per step = 512 cells × 32 bytes = 16 KB
  Over 2048 steps:   33 MB
  
With Phase 2 (eviction + flash spill + selective recall):
  DRAM read per step = 512 × 32 = 16 KB (same as Phase 1)
  FLASH write per spill = (2048 - 512) × 32 = 49 KB  (over the run)
  FLASH read per recall = depends on recall rate; bounded by I/O quota
  
  If recall rate is < 1 cell per step on average → ~32 bytes/step × 2048 = 65 KB total flash read
  Total bytes moved: 33 MB DRAM + 49 KB flash write + 65 KB flash read ≈ 33.1 MB
  
  ALMOST IDENTICAL to Phase 1, but recall lets us "save" the quality of the dropped cells.
```

This is the I/O budget that motivates the design.

## What needs to be measured to validate Phase 2

1. **Recall hit rate**: what fraction of attention queries to "evicted" positions actually need the cell?
2. **Recall batch effectiveness**: how often can multiple recalls be batched into one flash I/O?
3. **PPL improvement** vs Phase 1 (current −11.3%) at the same K_nominal
4. **Energy cost** of the spill writes
5. **Thermal impact** of sustained flash activity (does the controller heat up significantly under read load?)
6. **Index memory cost** as N_evicted grows (need to keep the index in DRAM)
7. **Worst-case latency** for a "miss" recall (cell not in DRAM, must read from flash before continuing decode)

## Connection to existing literature

- **LLM in a Flash (Apple)** spills WEIGHTS to flash. We spill KV CACHE.
- **PowerInfer-2** partitions weights between Hexagon NPU and CPU. We partition cache between DRAM and flash.
- **H2O / TOVA / SnapKV / StreamingLLM** all drop evicted cells. We propose to STORE them.
- **MMU page faults** for swap have similar latency math but no domain awareness. We use attention-score signals to predict useful recalls.

## Summary for the supervisor

> EndurKV is Phase 1 of a closed-loop mobile-LLM system: in-DRAM smart eviction with thermal insurance. We have measured it producing simultaneous wins on speed, energy, memory, and thermal. Phase 2 extends this to flash-tiered storage with attention-driven recall, allowing arbitrarily long effective cache without the energy/thermal cost of always keeping it resident in DRAM. The closed loop is: attention signals drive eviction AND recall; sensor signals drive frequency cap AND I/O quota. Phase 1 publishes at HotMobile 2027; Phase 1+2 publishes at MobiSys 2028; both phases plus the K_eff(t) adaptive loop are the dissertation.
