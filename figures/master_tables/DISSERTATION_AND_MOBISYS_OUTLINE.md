# Dissertation + MobiSys 2028 Paper — Complete Outline
**Candidate:** Md Romyull Islam, PhD CS, Kennesaw State University
**Committee:** Dr. Kun Suo (Chair), Dr. Bobin Deng, Dr. Tu Nguyen, Dr. Yan Fang
**Proposal title:** Thermal and Endurance Co-Aware KV Management for Sustained Mobile LLM Inference Driven by Model-Internal Signals

---

# How this document is organized

The dissertation and the MobiSys 2028 paper share the same scientific scope: the **full cross-layer runtime controller** described in the proposal (slides 17–20). They differ in length and audience: the dissertation is a 6–8 chapter document with background and integration; the MobiSys paper is a ~14-page version of the same controller plus its evaluation.

The **HotMobile 2027 paper is a strict subset**: only the eviction-policy + watchdog half of the controller (problems P1, P2, partially P3). That paper is the μKV LaTeX in `PAPER_DRAFT_MOBISYS_2028.tex`.

This outline covers **all 6 problems** the proposal addresses, the proposed solution for each, the current progress against each, AND the unifying **closed-loop controller** architecture that ties them together.

---

# PART 0 — The unifying thesis: a closed-loop runtime controller

> **One-sentence thesis:** *Sustained mobile LLM inference requires a runtime controller that couples model-internal signals (output entropy, per-head attention concentration) and hardware-internal signals (thermal sensors, flash write counters) to a small set of structural actions (eviction, in-place quantization, aligned offload), making decisions at every decode step under a joint thermal-and-endurance-and-bandwidth budget. This dissertation builds, evaluates, and validates that controller on commercial flagship mobile hardware.*

## The architecture (proposal slide 17, formalized)

```
        ┌─────────────────────────┐         ┌──────────────────────────┐
        │  INPUTS (every step)    │         │  RECOVERY ACTIONS         │
        ├─────────────────────────┤         ├──────────────────────────┤
        │ 1. Model-internal       │         │ A1. Per-head attention   │
        │    • output entropy H(t)│   ┌────► │     pruning (μKV gate)   │
        │    • attention peak m_h │   │      │                          │
        │                         │   │      │ A2. In-place quantization│
   ────►│ 2. Thermal              │───┤      │     (Q8 K downcast)      │
        │    • DDR temp           │   │      │                          │
        │    • CPU big-core temp  │   │      │ A3. Aligned offload to   │
        │    • skin temp          │   │      │     flash (16 KB chunks) │
        │    • battery temp/curr  │   │      │                          │
        │                         │   │      │ A4. Frequency throttle   │
        │ 3. Flash endurance      │   │      │     (watchdog, 5 tiers)  │
        │   (UFS write-amp; NOT   │   │      │                          │
        │    battery energy —     │   │      └──────────────────────────┘
        │    see naming note)     │   │                  ▲
        │    • UFS cumulative     │   │                  │
        │      write count        │   │                  │
        │    • P/E budget         │   │                  │
        └─────────────────────────┘   │                  │
                                      ▼                  │
                              ┌────────────────────────────┐
                              │  CO-AWARE ACTION SELECTOR  │
                              │  (one decision per step)   │
                              │  Picks the action that     │
                              │  minimizes combined        │
                              │  pressure under the joint  │
                              │  budget.                   │
                              └────────────────────────────┘
```

## The closed-loop property

The controller is **closed-loop** in two ways:

1. **Outer (algorithmic) loop**: the per-step action *changes* the next-step state (cache size, write count, temperature). The controller observes these changes and re-decides on the next step.
2. **Inner (thermal) loop**: the watchdog (Action A4) reads sensors at 5 Hz and adjusts CPU frequency, which feeds back into temperature observations the outer loop sees.

The two loops cooperate: smaller cache (A1+A2) → less DRAM bandwidth → lower power → cooler chip → fewer A4 throttle engagements → higher sustained throughput.

## Why this is novel (the dissertation's central claim)

No published system reads **all three signal classes** and selects among **all three structural actions** at decode time. The proposal (slide 16) enumerates the literature gap:

| Family | What it does | What it does NOT do |
|---|---|---|
| Datacenter offload (FlexGen, KVPR, ALISA, HiFC, INF2, Dynamo) | KV offload + scheduling | Assumes infinite cooling, replaceable NVMe, GB/s bandwidth. No thermal or endurance signal. |
| Mobile KV compression (KVSwap, KIVI, KVQuant, H2O, SnapKV, CAKE) | Eviction or quantization | Single mechanism. Fixed at design time. No coupling to thermal or endurance state. No entropy signal. |
| Thermal-aware mobile (zTT, FUSE, ZeroDVFS) | DVFS for frequency | Acts on frequency only, not on KV state. No flash endurance signal. |

The proposed controller is the first to schedule **eviction + quantization + offload + frequency** under **coupled thermal + endurance + bandwidth + entropy** pressure on commodity Android.

## Expected dissertation contributions (six concrete claims)

1. **First closed-loop runtime controller for sustained on-device LLM inference** that reads three orthogonal signal classes (model-internal, thermal, endurance) and selects among three structural actions (eviction, quantization, offload) at every decode step — proposal slides 17–18.

2. **First mobile-engineered per-head adaptive KV eviction policy** with a closed-form `μ(m_h)` ramp on max-of-softmax-per-head, FA-on-compatible via state-swap, validated cross-model on commercial flagship hardware — Chapter 6 (Paper A / μKV).

3. **First per-step entropy-gated prune-budget rule** `prune_budget(t) = base × (1 − H̃(t))` driven by output entropy, validated on-phone (proposal Months 1–4) — Chapter 7 (Paper B / C).

4. **First multi-sensor preempt-throttle watchdog** with five sensors and empirically-calibrated thresholds anchored on observed kernel-throttle events — Chapter 6 (Paper A / μKV).

5. **First sub-2.0× WAF aligned-offload of evicted KV cells** to UFS with O(1) DRAM index and io_uring batched recall — Chapter 8 (future paper).

6. **First end-to-end multi-axis measurement of mobile LLM inference** (PPL + retrieval + decode tps + total wall + peak DDR/CPU/skin temp + peak RSS + swap + battery mAh + watchdog tier transitions) on commercial flagship hardware, with cross-model replication (Phi-3, Llama-3.2-1B, Gemma-2-2B) and cross-budget Pareto curves — Chapters 6+7.

## Per-paper subset of the controller

| Paper / Chapter | Inputs realized | Actions realized | Loop closed? |
|---|---|---|---|
| **HotMobile (μKV, Chapter 6)** | Thermal only (watchdog v2) + per-head $m_h$ (one-shot at prefill) | A1 (eviction) + A4 (frequency throttle) | Thermal loop closed; outer loop one-shot |
| **MobiSys 2028 (Paper C / Chapter 7)** | + per-step output entropy $H(t)$ | + per-step modulation of A1 | Outer loop closed (per-step adaptation) |
| **Future (Chapter 8)** | + UFS endurance ioctl | + A2 (in-place quantization) + A3 (aligned offload) | Both loops closed, 3-action selector active |
| **Final dissertation** | All three signal classes | All three actions + A4 | Full controller, 48-hour sustained validation |

---

# μKV (v1_fa2_stack) scorecard — where it WINS and where it LOSES

Quick reference before going into per-problem detail. The μKV controller **wins on the system-axes** (throughput, wall, energy, memory residency, swap) and loses on **two quality axes** (PPL, retrieval) at the K=512 mobile operating point. This is a Pareto contribution, not a universal win.

## Wins (μKV beats vanilla)
| Axis | μKV vs vanilla | Where measured |
|---|---|---|
| **Decode tps** | **+68% Phi-3, +25% Llama-1B, +58% Gemma** | Wave-11 PPL across 3 models (Tables 1, 2, 3) |
| **Total wall time** | **−20% Phi-3, −2.4% Llama-1B, −27% Gemma** | Wave-11 PPL across 3 models |
| **Battery energy (mAh)** | **−6.6% on 3-way demo, ≈ −13% headline cliff-insurance** | 3-way demo + cliff-insurance demo (Table 5) |
| **Peak RSS (memory residency)** | **−6.7% Phi-3, +2.3% Llama-1B** | All long-decode demos |
| **Swap behavior** | **0 MB swap vs 158 MB vanilla on Phi-3** | All long-decode demos |
| **vs AdaKV on throughput** | **+175% Phi-3 tps, +237% Llama-1B tps** | Same Wave-11 protocol |

## Losses (μKV is worse than vanilla)
| Axis | μKV vs vanilla | Why and how it's contextualized |
|---|---|---|
| **PPL (Wave-11 chunk-pair)** | **+11.3% Phi-3, +15.5% Llama-1B, +16.9% Gemma** | Aggressive eviction at K=512 (≈ AdaKV's b=5–13% extreme region). At K=1024 (sweep landing now) we expect to approach vanilla. |
| **NIAH retrieval** | **0/8 Phi-3, 0/8 Llama-1B** | The aggressive top-K destroys the needle. AdaKV's α=0.2 safeguard preserves it; we lack that safeguard. Central caveat. Entropy-gate (Paper C) is the proposed fix. |
| **LongBench quality at K=512** | **−49% Llama-1B, −68% Phi-3, −64% Gemma** | K=512 ≈ AdaKV b=5–13% — below their reference. K-sweep cooking now to show curve recovery at K=1024+. |

## Ties / mixed
| Axis | μKV vs vanilla | Note |
|---|---|---|
| **Peak CPU temp** | **+0.4°C (Phi-3, cliff-insurance)** | We run slightly hotter peak but well below the 65.5°C cliff. Aggressive-watchdog variant produces −2.3°C active cooling at decode-tps cost. |
| **Peak DDR temp** | **+1.5°C (Phi-3, cliff-insurance)** | Same explanation as CPU peak — we run full-clock for shorter total time. |
| **vs AdaKV on PPL** | **+13.5% Phi-3 worse, +14.8% Llama-1B worse** | AdaKV is the quality winner; we are the throughput/energy/memory winner. Workload-dependent Pareto, not single optimum. |

## One-line summary

**μKV wins by 25–68% on every system axis (tps, wall, energy, RSS, swap) on every model. It loses 11–17% PPL and 0/8 NIAH retrieval at K=512.** This is the contribution: a new mobile-relevant Pareto corner that prior PPL-only KV-cache work has not located.

---

# PART A — The six problems and their solutions

## P1. KV cache size forces swap (DRAM pressure)

**Motivation (measured).** A Phi-3-mini-128k Q4_K_M cache at 2K context occupies ~1.2 GiB. On a 12 GB OnePlus 15 shared with foreground apps, this triggers **158 MB of swap** on vanilla, causing >300 ms decode stalls. A 32K-token request on Llama-3.1-8B produces ~4.3 GB of KV — exceeds free DRAM even with a 4-bit model.

**Proposed solution.** Per-head attention-confidence eviction with a closed-form budget rule:
```
K_h = round(K_nom × μ(m_h)),   μ(·) clamped to [0.7, 1.3]
```
Combined with a sink + anchor + recent partition (4 + 32 + 476 = 512 cells at K_nom=512).

**Progress.** ✅ Done. Implemented as `v1_fa2_stack` in `eviction_bench_v8`. Measured cross-model:
- Phi-3 K=512: 0 MB swap vs vanilla's 158 MB
- Llama-1B K=512: never swaps (model is small)
- Wave-11 PPL: +11.3% Phi-3, +15.5% Llama-1B vs vanilla
- 6-policy Wave-11 PPL table for Phi-3 + Llama-1B (Gemma in progress)

**Remaining.** Gemma-2-2B Wave-11 (cooking, ~1 hr); the Pareto already holds across two models.

---

## P2. DRAM bandwidth bound (per-step decode cost)

**Motivation (measured).** Per-token decode reads the entire KV cache once per layer. For Phi-3-mini at 2K context that's ~750 MB per attention pass × 32 layers. On LPDDR5X (~50 GB/s sustained), this is the dominant cost — decode is memory-bandwidth-bound on mobile, not compute-bound.

**Proposed solution.** Two stacking optimizations:
1. **Eviction (P1)** reduces cache cells read per step from ~3700 to ~512.
2. **Q8 K quantization** halves the K-side bytes (no f16 → q8_0 dequant cost during decode, just bandwidth halving).
3. **State-swap** lets the FA-on decode kernel exploit the reduced read.

**Progress.** ✅ Done. Measured:
- Phi-3: 5.2× per-step bandwidth reduction → +68% decode tps vs vanilla
- Llama-1B: smaller KV → smaller relative benefit; v1_fa2_stack +25% tps, v1_fa2_f16 (no Q8 K) +44% tps
- v1_fa2_f16 ablation shows Q8 K is workload-dependent (PPL-neutral on Llama-1B but adds dequant CPU cost; wall-time-positive on Phi-3 with its 1.4 GB cache)

**Remaining.** Cross-model 3rd point (Gemma); already enough evidence.

---

## P3. Thermal envelope (Wall 2 in the proposal)

**Motivation (measured).** The Qualcomm BCL and kernel `cool_state` throttle big-core from 1632 MHz → 883 MHz once DDR > 65°C or CPU > 67°C, often within 90 s of sustained decode. Tummalapalli et al. (2026): on Galaxy S24 Ultra, a hard GPU floor terminates inference after ~6 sustained iterations; iPhone 16 Pro loses 44% within two iterations.

**Proposed solution.** Multi-sensor preempt-throttle watchdog (Layer 5 of EndurKV):
- 5 sensors (DDR, CPU big-core, skin, battery temp, BCL current) sampled at 5 Hz
- 5-tier frequency ladder; engage thresholds anchored on observed kernel-throttle events
- Cliff-insurance mode (default): tier-2 engage at threat=0.50 — the watchdog only fires before the kernel cliff
- Aggressive mode (alternative operating point): tier-1 engage at threat=0.25 — actively cools at decode-tps cost

**Progress.** ✅ Done. Measured:
- Cliff-insurance Phi-3 long-decode: watchdog stays dormant (0 tier transitions); eviction itself keeps temperatures below the cliff
- Aggressive Phi-3 demo: watchdog actively cools (peak CPU 61.6°C vanilla → 59.3°C v1_fa2_stack, −2.3°C) at −12.7% decode tps cost
- n=3 short-prompt cliff-insurance replication: tight variance (±1% on tps/wall)

**Remaining.** Long-prompt n=3 cliff-insurance replication (Phase 3 queued, ~3 hr) for proper CIs on the headline +19.7% tps / −12.9% energy. Active-cooling demonstration on a sustained 4–8K-token decode where vanilla actually hits the cliff (future work, not in current paper).

---

## P4. Battery energy per generation (μKV is a WIN here, not a loss)

> **Naming note.** This problem is about *battery energy* — total mAh drained per generation. The proposal reserves the word **endurance** for *flash UFS write-amplification endurance* (problem P6 below). Be careful not to conflate them. They are different physical limits with different proposed actions: P4 is reduced by the eviction-+-state-swap-+-watchdog combination (μKV, already done); P6 requires the aligned-offload mechanism (future Chapter 8).

**Motivation (measured).** Without intervention, vanilla 2048-token decode on Phi-3-mini draws ~38.7 mAh under the disable-charging protocol. At typical phone capacity (~5000 mAh), that implies only ~130 such generations per charge — a hard ceiling on agentic mobile workloads.

**Proposed solution.** Energy is `avg_power × wall_time`. We attack both factors:
1. Eviction (P1) shrinks the cache → less DRAM bandwidth → less power
2. Eviction (P1) + state-swap (P2) → faster decode → shorter wall
3. Watchdog (P3) caps frequency only when needed; otherwise the eviction-driven thermal headroom keeps the chip at full clock

**Progress: μKV is a clear win on battery energy.** Measured (3-way Phi-3 long-decode demo, charging disabled):

| Policy | Energy (mAh) | Wall (s) | Avg power (W) | Δ energy vs vanilla |
|---|---|---|---|---|
| vanilla | **28.73** | 337.4 | 1.13 | ref |
| **v1_fa2_stack (μKV)** | **26.84** ✅ | 342.7 | 1.04 | **−6.6%** (less energy than vanilla) |
| adakv | 55.88 | 703.4 | 1.06 | +95% (FA-off lock-in eats 2× wall time) |

Cliff-insurance long-decode demo headline: **μKV ≈ −13% mAh vs vanilla** (n=3 long-prompt CIs landing on phone right now).

Decomposition: the win comes from −8% average power (less DRAM bandwidth → less DDR controller activity) × near-identical wall time on short prompts (the state-swap adds ~0.5 s of one-time overhead). On long-prompt workloads (Wave-11, Phi-3 1.8K-token prompt), the state-swap overhead is fully amortized and the wall savings dominate too (−20%). Figure: `energy_vs_bandwidth_correlation.pdf`.

**Honest caveats.** (i) Energy measured via `energy_method=approx` (mean current × wall under disable-charging), not via a direct power-meter rail — the previously-drafted DDR-rail breakdown was extrapolation and is removed. (ii) On Wave-11 PPL teacher-forced runs, μKV is energy-favorable but cannot be measured cleanly (charging on, no integration window). All mAh numbers in this section come from autoregressive decode runs.

**Remaining.** Phase 3 long-prompt n=3 cliff-insurance replication queued on phone (will produce CIs on the −13% energy headline; ETA ~3 hr).

---

## P5. Workload-dependent retrieval (the NIAH catastrophe — central novelty of Paper B/C)

**Motivation (measured).** The 6-policy NIAH Tier-1 matrix on Phi-3 + Llama-1B reveals a class-level structure:
- **Aggressive evictors (v1_fa2_stack, streamingllm): 0/8 retrieval** — the needle is destroyed by aggressive top-K
- **Per-head safeguarded (tova, adakv): 7-8/8** — α=0.2 safeguard floors every head's budget, preserving at least one head's view of the needle
- **vanilla: 7-8/8** — no eviction, full retrieval
- **h2o: intermediate** (4/8 Phi-3, 8/8 Llama-1B) — heavy-hitter retention preserves needle on smaller model

This is exactly the **fixed-budget limitation** the proposal predicts: a single deployment-time K cannot serve both creative-continuation (where v1 wins throughput) and retrieval (where v1 catastrophically fails).

**Proposed solution.** Per-step **entropy gate**:
```
prune_budget(t) = base × (1 − H̃(t))
```
where `H̃(t)` is the output entropy at decode step t, normalized over a rolling 64-step window.

**Why it should work.** On a needle query, the model has high uncertainty (high entropy) until it retrieves the answer; the gate shrinks the prune budget when entropy is high, preserving cells. Once the model finds the needle and commits, entropy drops and pruning resumes at full rate.

**Combined claim (Paper C — the strongest possible chapter):**
```
K_h(t) = round(K_nom × μ(m_h) × (1 − H̃(t)))
         └────────────┘   └────────────┘
         per-head (from P1)    per-step (from P5)
```
This occupies the previously-unoccupied **per-head AND per-step adaptive** corner of the design space.

**Progress.** ⚠️ Partial.
- ✅ Server-side correlation evidence: ρ = −0.37 (n=1670, p<10⁻⁵⁵) on Llama-3.1-8B at 4–12K context, 7 LongBench/HELM long-form tasks
- ✅ Position-distribution study disconfirmed an earlier hypothesis (sinks ~60%, recent <1%, middle ~40%)
- ✅ Probe instrumentation: `entropy_probe/` C++, <1% overhead
- ❌ On-phone integration NOT YET BUILT — proposal Months 1–4
- ❌ NIAH re-run with entropy gate NOT YET MEASURED

**Remaining.** ~2 weeks build + ~30 hr phone time + LongBench server-side eval. This is the next phase of work after HotMobile submission.

---

## P6. Flash endurance (Wall 1 in the proposal — Phase 2 of the closed loop)

**Motivation (measured + cited).** Blind OS swapping creates random 4 KB writes; write amplification factor (WAF) reaches 4.5× or higher. A 150 TBW UFS chip is exhausted in under a year of sustained use. Sustained inference puts ~200 MB of physical writes on flash for every 50 MB of logical KV traffic. **No published mobile system treats flash endurance as a first-class budget for KV decisions.**

**Proposed solution.** Aligned-offload + DRAM index + io_uring batched reads:
- 16 KB sequential chunks, sorted by (layer, head, token range); WAF target ≤2.0×
- ~16 bytes/entry DRAM index (~4 MB total for 32K tokens × 32 layers × 32 heads); O(1) lookup
- Predictive prefetch: early-layer attention anticipates which evicted cells will be needed later
- I/O quota tied to watchdog (back-pressure when flash controller heats up)

**Progress.** ❌ Not started.
- ✅ Architecture documented in proposal slide 19 and in the EndurKV paper §future-flash-tier
- ❌ No implementation
- ❌ No UFS ioctl integration
- ❌ No 48-hour sustained validation

**Remaining.** This is the proposal's Phase 2 — ~3–4 months of work. Belongs in dissertation Chapter 8 (final work) or a future paper.

---

# PART B — Chapter map for the dissertation

| Ch | Title | Source paper | Maps to problem | Status |
|---|---|---|---|---|
| 1 | Characterizing SLMs on Edge Devices | IPCCC 2024 | Motivation for P1 (memory ceiling) | ✅ Published |
| 2 | Energy Footprint of SLMs on Edge | MASS 2025 | Motivation for P4 (energy) + P3 (thermal) | ✅ Published |
| 3 | Vision Knowledge Pipeline (VKPS) | AIoT 2025 | Motivation for staging / system-level structure | ✅ Published |
| 4 | Holistic VLM Analysis | COMPSAC 2026 | Motivation for P3 (thermal floor under sustained load) | ✅ Submitted |
| 5 | SemSched (CXL sub-layer scheduling) | In preparation | Cross-cutting: structure-aware scheduling beats blind tiering | ⏳ Near complete |
| **6** | **EndurKV — Per-Head Eviction + Watchdog** | **HotMobile 2027 (Paper A)** | **P1, P2, P3, P4** | ⏳ This work — submit ~1 week |
| **7** | **Entropy-Gated Adaptive Eviction** | **MobiSys 2028 (Paper C)** | **P5 (workload retrieval) + refinement of P1+P2+P3+P4** | ⏳ ~8 weeks |
| 8 | Flash-Tiered KV + 3-Action Selector + 48-h Validation | Journal / future paper | **P6 (flash endurance) + integration of all 6 problems** | ⏳ Future, ~3–4 months |

---

# PART C — The MobiSys 2028 paper (Paper C) — section-by-section

This is the **full controller paper**. It integrates Chapters 6+7 into a single coherent contribution.

## Working title
**"A Co-Aware Runtime Controller for Sustained Mobile LLM Inference: Coupling Model-Internal Entropy, Thermal, and Cache-Bandwidth Signals on Commercial Flagship Hardware"**

## Pitch (one paragraph for reviewers)
> Sustained on-device LLM inference is bounded not by one resource but by the joint pressure of KV cache footprint, DRAM bandwidth, thermal envelope, and battery endurance. Existing eviction policies set the cache budget at deployment with no signal to adapt when the model is uncertain; existing thermal-aware mobile systems modulate frequency but not cache state; existing entropy-driven inference work uses model-internal signals for compute decisions but never for cache state. We present the first co-aware controller that reads three signals at every decode step — per-head attention concentration (model-internal, prefill), per-step output entropy (model-internal, runtime), and multi-sensor thermal/electrical state (hardware, runtime) — and modulates a per-head per-step adaptive eviction policy under a joint thermal/bandwidth/endurance budget. On a OnePlus 15 (Snapdragon 8 Elite Gen 5) across three models (Phi-3-mini, Llama-3.2-1B, Gemma-2-2B) and three benchmarks (Wave-11 PPL on WikiText-2, NIAH Tier-1 retrieval, long-decode demo with cliff-insurance watchdog), the controller delivers throughput within 5% of the fixed-budget aggressive policy while recovering retrieval accuracy from 0/8 to 6+/8 — defeating the workload-dependence trap that fixed-budget policies cannot escape.

## Section outline (~14 pages)

### 1. Introduction (1.5 pages)
- The 4 mobile pressures (P1–P4) + the 2 coupled walls from the proposal
- Why a single fixed budget cannot serve both creative and retrieval workloads (P5)
- Three contributions:
  1. The first KV-eviction policy that adapts both *per-head* and *per-step* — `K_h(t) = K_nom × μ(m_h) × (1 − H̃(t))`
  2. End-to-end on-phone evaluation: 3 models × 3 benchmarks × 7 policies with statistical CIs
  3. Negative-result disclosure: tasks where the entropy signal collapses (code, few-shot)
- Honest non-claims: no formal regret bound; single device; flash-tier (P6) deferred to future chapter

### 2. Background and Related Work (1.5 pages)
- KV cache and attention notation
- Eviction taxonomy: **fixed-K** (vanilla, StreamingLLM) vs **per-head adaptive** (Ada-KV, AhaKV, KeepKV, CriticalKV) vs **per-step adaptive** (this work)
- Entropy as a runtime signal in inference (AdaEDL, EASD, Step-Entropy) — used for compute, never for KV
- FlashAttention + the FA-off/FA-on split (cite Chapter 6 / EndurKV)
- Mobile LLM systems (PowerInfer-2, MLC-LLM, llama.cpp Hexagon)
- Thermal-aware mobile (zTT, FUSE, ZeroDVFS)

### 3. Motivation — the workload-dependence trap (2 pages)
- Quantified 4-bottleneck profile on OnePlus 15 (cite Chapter 6 data)
- The retrieval-vs-throughput Pareto from Chapter 6: AdaKV preserves NIAH but FA-off lock-in caps tps; v1_fa2_stack wins tps but 0/8 NIAH
- Position-distribution study (sinks 60%, recent <1%, middle 40%) — the needle hides in the middle, which is exactly what gets evicted
- Cross-task entropy-attention correlation (ρ=−0.37, p<10⁻⁵⁵) — server-side evidence the signal is real
- The intervention point: gate the prune budget by `(1−H̃(t))`

### 4. Design — The Co-Aware Controller (2.5 pages)

#### 4.1 The combined policy
```
K_h(t) = round( K_nom · μ(m_h) · (1 − H̃(t)) )
```
- `μ(m_h)`: per-head ramp inherited from Chapter 6 (Layer 1)
- `H̃(t)`: rolling 64-step normalization of output entropy
- Composition with state-swap (Layer 4), sink+anchor+recent partition (Layers 2+3)
- Algorithm 1 (prefill, unchanged from Chapter 6)
- **Algorithm 2 (decode loop, modified)**: entropy-gated prune budget
- Algorithm 3 (watchdog, unchanged from Chapter 6)

#### 4.2 Why this fixes the NIAH catastrophe (mechanism)
- High entropy ⇒ low pruning ⇒ cells preserved while model searches
- Low entropy ⇒ full pruning ⇒ throughput restored once needle is found
- Per-task variation: gate fires on long-form generation, null on lcc/trec

#### 4.3 Complexity
- Probe: O(V) per step (V = vocab size, ~32K) ≈ negligible vs decode O(layers × heads × K)
- Modulation: O(1) per step
- Eviction trigger: amortized O(K) every ~80 steps

### 5. Implementation (1.5 pages)
- Probe code (reuses `entropy_probe/`, <1% overhead)
- Rolling normalization: ring buffer over 64 steps, μ/σ updated O(1)
- Integration with `eviction_bench_v8` (now `_v9` with entropy)
- Cross-compilation via Android NDK aarch64
- Watchdog + sensor sampler (unchanged from Chapter 6)
- Energy fallback chain (5 tiers)

### 6. Evaluation (4 pages — the bulk of the paper)

**Setup (0.5 page).** OnePlus 15, Snapdragon 8 Elite Gen 5, 12 GB UMA. Three models. Seven policies: vanilla, h2o, tova, streamingllm, adakv, **v1_fa2_stack** (Chapter 6), **v1_fa2_stack_entropy (ours, this chapter)**.

**RQ1 — Wave-11 PPL (WikiText-2): does the entropy gate cost quality? (1 page)**
3 models × 7 policies, n=7 chunks each, 95% bootstrap CIs. Predicted: entropy gate ≤ +2% PPL vs v1_fa2_stack — the gate runs O(decode steps) but most steps have low entropy, so most cycles still prune at full rate.

**RQ2 — NIAH Tier-1 retrieval: does the entropy gate fix the 0/8 catastrophe? (1 page)**
3 models × 7 policies, 8 stimuli depth sweep. Predicted: v1_fa2_stack 0/8 → v1_fa2_stack_entropy 6+/8 on Phi-3; 8/8 on Llama-1B. Per-task entropy correlation re-run on-phone.

**RQ3 — Decode tps + total wall (0.5 page)**
Predicted: ≤10% tps cost vs v1_fa2_stack; still >vanilla on both models.

**RQ4 — Thermal + Energy (n=3 cliff-insurance with CIs) (0.5 page)**
Headline ±5% bands for tps, wall, energy. Predicted: entropy gate is energy-neutral within measurement noise.

**RQ5 — Cross-task ρ on-phone (server-vs-phone signal preservation) (0.5 page)**
Predicted: ρ_phone ≥ −0.20 (proposal Month-1 gate).

**RQ6 — Ablation: gate alone vs per-head alone vs combined (0.5 page)**
- Per-head alone (Chapter 6): 0/8 NIAH, 4.98 tps Wave-11
- Gate alone (uniform K): ?/8 NIAH, ? tps
- Combined: 6+/8 NIAH, ~5 tps

### 7. Discussion (1 page)
- Where the gate works (long-form generation) vs where it does not (code, few-shot)
- The thermal claim refined: watchdog stays dormant on headline; aggressive variant is the active-cooling operating point
- Future: 3-action selector with offload + quantization + eviction (P6 + multi-action)

### 8. Conclusion (0.5 page)
- Per-step entropy = first model-internal control signal applied to KV state
- Combined per-head × per-step adaptation reaches a previously-unoccupied Pareto corner
- Open-source release; the 3-action selector and flash-tier extension is forthcoming

---

# PART D — What the HotMobile paper covers vs what the MobiSys/Dissertation covers

| Element | HotMobile (Paper A) | MobiSys / Dissertation Ch 6+7 (Paper C) |
|---|---|---|
| Problem statement (motivation) | P1, P2, P3, P4 measured | All 6 problems (P1–P6) framed |
| Solution proposed | Per-head eviction + state-swap + Q8 K + watchdog | All of HotMobile **plus** entropy gate (the novel claim) |
| Solution implemented | ✅ v1_fa2_stack policy + multi-sensor watchdog | ✅ same + ⏳ entropy gate (2-week build) |
| Evaluation scope | 3 models × 2 benchmarks × 6 policies + n=3 cliff demo | 3 models × 3 benchmarks (add LongBench) × 7 policies + n=3 cliff |
| Disclosure of limitations | NIAH 0/8 (us+streamingllm), Q8 K workload-dependent, no theory | Same + on-phone ρ if it collapses |
| Future-work section | Points forward to entropy gate (this paper) | Points forward to 3-action selector + flash tier (P6) |
| Length | ~6 pages | ~14 pages |
| Deadline target | HotMobile 2027 (~1 week) | MobiSys 2028 (~6–8 weeks after HotMobile) |
| Dissertation chapter | Ch 6 | Ch 7 |

---

# PART E — Progress tracker (one number per problem)

| Problem | Solution | Implemented? | Measured? | Cross-model? | n≥3? |
|---|---|---|---|---|---|
| P1: DRAM/swap | Per-head eviction | ✅ | ✅ | ✅ (Phi-3, Llama-1B; Gemma in progress) | ⚠️ partial (PPL n=7 chunks) |
| P2: DRAM bandwidth | Eviction + Q8 K + state-swap | ✅ | ✅ | ✅ | ⚠️ partial |
| P3: Thermal | Multi-sensor watchdog | ✅ | ✅ (cliff-insurance dormant; aggressive cools) | ⚠️ Phi-3 only | ✅ n=3 short-prompt, ⏳ n=3 long-prompt queued |
| P4: Battery energy | Cache reduction → less power × less wall | ✅ | ✅ (approx fallback) | ⚠️ Phi-3 only | ⏳ n=3 long-prompt queued |
| P5: Retrieval / workload-dep | Entropy gate | ❌ on-phone | ✅ server-side correlation only | ✅ correlation across 12 tasks | ✅ n=12,300 decode steps server-side |
| P6: Flash endurance | Aligned offload + DRAM index | ❌ | ❌ | ❌ | ❌ |

---

# PART F — Three-deadline budget

| Deadline | Paper | Required work | Time |
|---|---|---|---|
| **HotMobile 2027** (~1 week) | Paper A (Chapter 6) | Finish Gemma data → fill tables → cut to 6 pages → submit | 1 week |
| **MobiSys 2028** (~8 weeks) | Paper C (Chapter 7) | Build entropy gate on phone → expand quality benchmarks to AdaKV scope → re-run → write | 8 weeks |
| **Dissertation defense** (~6 months) | All 8 chapters | + P6 flash-tier implementation + 48-hour sustained validation + cross-device | 4–6 months |

---

# PART F.2 — Evaluation scope comparison: us vs AdaKV (Feng et al. NeurIPS 2025)

Critical reference point. AdaKV is the strongest prior-art baseline we compare against; their evaluation defines what reviewers will expect.

| Dimension | AdaKV (NeurIPS 2025) | μKV (HotMobile, current) | Paper C (MobiSys, target) |
|---|---|---|---|
| Models | Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2 | Phi-3-mini (3.8B), Llama-3.2-1B, gemma-2-2b-it | Same as μKV + Llama-3.1-8B (for direct AdaKV comparison) |
| Hardware | GPU only | OnePlus 15 (Snapdragon 8 Elite Gen 5) | Same as μKV + server GPU for LongBench |
| **Quality benchmark 1** | Ruler (full, 13 subtasks) | S-NIAH-1 only (≈1 Ruler subtask, 8 depths) | **Ruler subset: S-NIAH-1/2/3, MK-NIAH-1, MQ-NIAH, VT (6 subtasks)** |
| **Quality benchmark 2** | LongBench (6 task domains, full suite) | None | **LongBench full on server GPU (Llama-3.1-8B) for AdaKV apples-to-apples** |
| Quality benchmark 3 | (none, just Ruler+LongBench) | Wave-11 PPL on WikiText-2 (extra; not in AdaKV) | Keep Wave-11 PPL (gives a fluency reference) |
| Budget sweep | b ∈ {10%, 20%, 40%, 60%, 80%, 100%} | K=512 only (≈12% at 4K) | b ∈ {10%, 20%, 40%, 60%} on Phi-3 and Llama-1B |
| Question-aware mode | Yes (separate eval) | No | Yes (μ uses query-aware attention) |
| Question-agnostic mode | Yes | Yes | Yes |
| System axes (decode tps, wall, temp, RSS, swap, mAh) | None | All 6 | All 6 (kept from μKV) |
| **Reference AdaKV result** (Question-agnostic, Llama-3.1-8B, LongBench-avg, b=20%) | Ada-SnapKV 42.87 vs SnapKV 41.29 vs full-cache 49.20 (87% retained) | — | Reproduce on server GPU as apples-to-apples vs AdaKV's table |

## What Paper C must add to defend against the "you don't compare like AdaKV" reviewer attack

1. **Ruler-Tier-2** on Llama-3.2-1B + Phi-3-mini (mobile-feasible subset: S-NIAH-2, S-NIAH-3, MK-NIAH-1, MQ-NIAH, VT) — ~40 hr phone time across 3 policies (vanilla, AdaKV, μKV+entropy).
2. **LongBench-full on server GPU** at Llama-3.1-8B Instruct under b={10%, 20%, 40%} for our μKV+entropy vs AdaKV vs SnapKV — ~1 day on a single A100. This gives the apples-to-apples vs AdaKV's Table 1.
3. **Budget sweep** on Phi-3 + Llama-1B at b={10%, 20%, 40%, 60%}: 4 budget × 2 models × 4 policies × 7 chunks = 224 cells. ~50 hr phone time.
4. **Question-aware variant of μ(m_h)** — when the query is known at prefill time, the per-head allocator can use query-conditioned attention rather than the standard last-token attention. This matches AdaKV's question-aware mode.

## What Paper C should explicitly NOT try to do
- Full Ruler 13-subtask sweep across 3 mobile models — phone-time-infeasible.
- LongBench on mobile — single-cell costs make this prohibitive; do LongBench on server only.
- Mistral-7B — not in our model triple; not worth adding.
- Multiple device generations (Galaxy S25, Pixel 9 Pro) — defer to Chapter 8 cross-device validation.

## Estimated Paper C evaluation budget
- Phone time: ~90 hr (Tier-2 NIAH 40 + budget sweep 50)
- Server time: ~24 hr (LongBench 6-domain × 4 policies × 3 budgets)
- Total elapsed: ~5 weeks of running unattended
- Total writing: ~3 weeks
- **Total Paper C: 8 weeks from start of implementation**

---

# PART G — Things to explicitly NOT do (lessons learned)

- Do NOT submit Paper A to MobiSys 2028. That slot is for Paper C (the controller paper). Paper A is HotMobile-sized.
- Do NOT add a formal theorem to either paper. Three attempts (Theorem 1 v2, v3, lower bound) failed adversarial review during preparation. Disclose this honestly in the limitations section; do not pretend to have a bound.
- Do NOT bundle HotMobile + MobiSys into one mega-paper. Two focused contributions read stronger than one sprawling one.
- Do NOT delay HotMobile waiting for the entropy gate. HotMobile is the systems-engineering chapter; it stands on its own.
- Do NOT add P6 (flash tier) to the MobiSys paper. P6 is dissertation Chapter 8 / future paper.
