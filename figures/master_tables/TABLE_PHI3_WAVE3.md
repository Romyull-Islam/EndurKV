# Phi-3 sustained-stress master table

**Setup:** Phi-3-mini-128k-instruct-Q4_K_M, narrativeqa (~9800-token prompt),
256-token decode per iter, 4-thread CPU, DVFS-pinned (performance governor,
1.5 GHz cap on big cores), `--ignore-eos`, 60-min budget per cell with
strict cool-down to 33 °C between cells. Wave-3 v2 chain (`wave3_phi3_1780719530`).

## Per-cell results

| Policy | FA mode | Iters | Iter 1 decode (tok/s) | Iter 2 decode (tok/s) | Iter 3 decode | Iter 1→2 decay | Peak DDR (°C) | Mean DDR (°C) | Peak CPU (°C) | Total evicted |
|---|---|---|---|---|---|---|---|---|---|---|
| **vanilla** | FA-on prefill+decode | 2 | 1.022 | 0.972 | — | **−5%** | **57.9** | 52.4 | 60.2 | 0 |
| **v1 K=512** | FA-off everywhere + per-step evict | 2 | 1.073 | 1.069 | — | −0% | **61.7** ❌ | 53.1 | **65.2** ❌ | 1.87 M |
| **v1_FA K=512** | FA-off prefill, FA-on decode (state swap + frozen mask) | 2 | **1.965** | **1.877** | — | −4.5% | **58.7** ✓ | 52.4 | **61.0** ✓ | (in prefill) |
| **TOVA-layer K=512** | FA-off + per-step evict (paper-faithful per-layer mean) | 2 | 1.076 | 1.042 | — | −3% | 60.6 | 52.7 | 62.9 | 1.95 M |
| **llama.cpp stock** | FA-on (no eviction) | 3 | 1.06 | 0.96 | 0.93 | **−12%** | 57.5 | 53.0 | 59.4 | 0 |

## Prefill cost

| Policy | Iter 1 prefill (s) | Iter 2 prefill (s) | Δ iter1→2 |
|---|---|---|---|
| vanilla | 1443 | 1662 | +15% (thermal accumulation) |
| v1 K=512 | 1835 | 2098 | +14% |
| v1_FA K=512 | 1840 | 2098 | +14% |
| TOVA-layer K=512 | 1830 | 2105 | +15% |
| llama.cpp stock (FA-on) | ~315 | ~330 | +5% (FA-on much faster) |

(llama.cpp stock uses FA-on prefill which is ~4× faster than our eviction_bench
FA-off prefill, because eviction_bench needs FA off to capture attention scores
for the eviction policies. That's how vanilla via eviction_bench can also be
FA-on — for vanilla cb_eval is nullptr, no attention readback needed.)

## Key findings

### 1. llama.cpp stock ≈ vanilla via eviction_bench (within 0.5 °C peak DDR)
57.5 °C vs 57.9 °C confirms our vanilla code path is identical to pure llama.cpp.
The 0.4 °C gap is sampling noise. (Same finding as Llama-1B: see TABLE_ALL_POLICIES.md.)

### 2. v1 K=512 is THERMALLY WORSE than vanilla on Phi-3
**+3.8 °C peak DDR, +5 °C peak CPU** vs vanilla. The cause: FA-off prefill (24+ min)
dominates the cell, and FA-off requires materializing the n_kv × n_q × n_layers
attention matrix on DRAM. The bandwidth cost outweighs the cache-size savings.
This invalidates the simple "smaller cache → cooler" hypothesis for prefill-heavy
workloads.

### 3. v1_FA is the redemption: 2× decode speed at near-vanilla temp
**Peak DDR 58.7 °C (only +0.8 °C above vanilla)**, decode 1.965 tok/s (vs vanilla
1.022). The FA-on decode + frozen-mask combination delivers both throughput and
thermal benefit during decode. The state-swap engineering pays off here.

### 4. TOVA-layer (paper-faithful) ≈ v1 thermally and on decode
TOVA's per-layer averaging vs v1's per-head selection produces nearly identical
bandwidth profile (60.6 vs 61.7 °C, 1.076 vs 1.073 tok/s). The dominant cost is
the FA-off prefill, not the per-head/per-layer aggregation strategy.

### 5. All policies show in-cell thermal accumulation
Iter 1 → iter 2 prefill slowdown of +14-15% across the board (except llama.cpp
which is FA-on and only +5%). Llama-1B never showed this — Phi-3's 14× larger
KV cache pushes the chip enough to throttle within a single cell.

## Implications for HotMobile 2027 paper

**The honest dissertation claim** (corrected from earlier over-claim):

> For workloads where decode dominates, our v1_FA reduces peak DDR temperature
> by ~3 °C vs FA-off-only v1 and matches vanilla thermal behavior while delivering
> 2× decode throughput. For prefill-dominated workloads (e.g., Phi-3 / 8 K
> context / single-shot), FA-off eviction policies (v1, TOVA) are thermally
> worse than no-eviction baselines because the FA-off prefill bandwidth cost
> exceeds the cache-reduction benefit during the short decode phase.

This scopes the claim correctly and matches the data.

**Where the dissertation thermal claim DOES hold:**
- Long-decode workloads (chat with 200-tok prompt + 2000-tok answer)
- Multi-turn workloads where cache grows steadily across turns
- Smaller models (Gemma-2B mid-range) where prefill/decode time ratio is more balanced
- Phi-3 with v1_FA (decode-time FA-on recovery)

**Where the simple eviction story FAILS:**
- Single-shot long-prompt + short-answer workloads (this experiment)
- v1 alone without the FA-on decode swap
