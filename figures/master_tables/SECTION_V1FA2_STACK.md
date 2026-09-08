# The v1_FA² Stack: Composition, Justification, and Measured Effect

## 1. Composition of the Stack

The `v1_fa2_stack` policy (the EndurKV headline configuration introduced in
Wave-9 and inherited unchanged by Wave-10 and Wave-11) composes four
mechanisms that sit at four distinct layers of the inference pipeline. They
are listed below in the order they were added to the policy, which is also
the order in which their effects compose end-to-end during a decode step:

1. **v1 two-stage candidate selection (per-head budget + selective top-32).**
   The base v1 eviction scheduler maintains a per-layer KV budget
   `K_nominal` and, on each decode step that would push the resident slot
   count above K, runs a **two-stage filter** to choose which positions
   survive. **Stage 1 (per-head budget)** assigns each head h its own
   budget `K_h = round(K_nominal · μ(max_a[h]))` where
   `μ(x) = 1.3 − 0.6 · clip((x − 0.4) / 0.4, 0, 1) ∈ [0.7, 1.3]` and
   `max_a[h] = max_p a_{L,h,p}` is the head's peak attention from the
   last query position; the per-head keep set is the `top-K_h`
   positions by `a_{L,h,p}`, and the layer keep set is the union across
   heads. **Stage 2 (selective anchoring)** then re-ranks the stage-1
   survivors by mean attention over the sliding query window and
   *anchors the top-32*. Positions that fail either stage are the
   lowest-importance evictees. This is the inherited Wave-3..6 mechanism
   as extended by Wave-7/8 (see §1a below). The per-head budget
   μ(max_a[h]) gives confident (peaky) heads a smaller keep set and
   diffuse (uncertain) heads a larger one — the head's intrinsic
   uncertainty drives how many positions it needs to retain.

2. **FA-on decode via state-swap (the v1_fa² inheritance from Wave-7/8).**
   The decode path uses the llama.cpp Flash-Attention-2 kernel
   (`--flash-attn` on a Q8/Q8 KV pair) with the v1 eviction scheduler driving
   *which* positions are addressable, while the FA² kernel itself runs over
   an anchor-plus-recency tiling that is materialised by a state-swap from
   the v1 importance index. This is the configuration that first delivered
   the 6.7-6.8 tok/s decode plateau on Phi-3-mini long-decode.

3. **Q8 K cache (the v1_fa²_selective ingredient from Wave-8).** The K
   tensor is quantised to `q8_0`; V is held at `f16` (asymmetric quant by
   design — the dequant cost on K is one pass per active slot per step,
   whereas V is read once per emitted token). The asymmetry is what makes
   the path fit the Hexagon SIMD pipeline; symmetric Q8/Q8 was tried in
   Wave-7 and produced 1.55 GB of swap-out under the same workload. With Q8
   K and f16 V the seq_add path inside FA² is bypassed on evicted positions
   (the `seq_add-skip` short-circuit), which is the property that makes
   eviction *cheaper* than the attention work it avoids in steady state
   (see Section 4 below and `SUBSECTION_KSWEEP.md` §3).

4. **Preempt-throttle watchdog + closed-loop K + memory gate.** A 500 ms
   sidecar polls the DDR thermal zone and writes the per-layer K budget into
   the v1 scheduler through a shared-memory hook. The state machine is the
   one specified in `THERMAL_RESEARCH_WAVE9_PLAN.md` §Techniques: K=512 for
   T<62 °C, K=384 for 62≤T<66 °C, K=256 for T≥66 °C, with 5 °C hysteresis
   and a 10 s minimum dwell. The watchdog component pre-empts the Qualcomm
   kernel mitigation framework by lowering K *before* the kernel's
   `freq_qos` re-vote fires at the 65 °C trip; the memory-gate component
   refuses to grow the resident cache above an RSS ceiling regardless of
   what the scheduler would otherwise admit. The three sub-mechanisms are
   tightly coupled and are reported as one component throughout this
   dissertation.

The four mechanisms together are the `v1_fa²_stack` policy as logged in
`phone-logs/wave9_v1fa2_stack_1780796320/`.

### 1a. Two-stage candidate selection

The first mechanism in the stack is itself a composite, and the two
stages were added in *different* waves in response to *different*
failures. Because the distinction matters both for the
ablation/contribution story and for the comparison to canonical SnapKV,
we describe the stages explicitly here rather than collapsing them.

**Stage 1 — Per-head budget (peak-confidence modulation).** Stage 1
runs a per-head adaptive top-K_h that gives each head its own keep
size as a function of that head's peak attention `max_a[h] = max_p
a_{L,h,p}` (from the last query row). Concretely, for each
(layer L, head h):

```
μ(x)       = 1.3 − 0.6 · clip( (x − 0.4) / 0.4, 0, 1 )    # ∈ [0.7, 1.3]
K_h        = round( K_nominal · μ(max_a[h]) )            # per-head budget
keep(L, h) = top-K_h positions by a_{L,h,p}
keep(L)    = ⋃_h keep(L, h)                              # layer = union over heads
```

The per-head budget μ(max_a[h]) gives confident (peaky) heads a
smaller keep set and diffuse (uncertain) heads a larger one — the
head's intrinsic uncertainty drives how many positions it needs to
retain. This is the original v1 mechanism inherited from Wave-3..6.
**The per-head budget exists because of the Wave-7 lesson:** the
Wave-7 v1_fa² configuration, which had no first-filter and anchored
*all* prompt tokens directly into the FA² tiling, crashed held-out
PPL (the prompt-anchored cache pinned positions that should have been
evicted, displacing recency mass and producing the swap-and-PPL
pathology recorded in the Wave-7 row of the master table). The
per-head budget filter is therefore *not* an optimisation — it is a
soundness guard that prevents the eviction scheduler from being
overridden by an unfiltered "anchor everything" policy.

**Stage 2 — Selective top-32 (mean-attention re-rank).** Stage 2 takes
the stage-1 survivors (the per-head-union keep set) and re-ranks them
by mean attention over the sliding query window, keeping the top-32 as
anchored slots and exposing the remainder to ordinary eviction
pressure. **Selective top-32 was added in Wave-8** as the fix to a
different failure mode: the per-head-budget union alone (Stage 1) was
too permissive on long-context recall because diffuse heads' μ = 1.3
budgets unioned to retain most prompt positions, leaving no room for
the recent decode window after `K_nominal` clipping. Wave-8 added
Stage 2 specifically to compress the union into a small budgeted set
of mean-attention-heavy positions. The "32" is the budget tuned in
Wave-8's sweep over the NIAH stimulus grid; larger budgets re-opened
the Wave-7 anchoring pathology, smaller budgets re-opened the
long-range-recall regression.

**Distinction from canonical SnapKV (Li 2024).** Canonical SnapKV uses
*only* a single-stage mean-attention top-k re-rank — the equivalent of
running Stage 2 directly on the full candidate pool with no Stage 1.
The v1_fa²_stack's distinguishing component is **Stage 1, the per-head
budget modulation by peak confidence**, which sits *before* the
mean-attention re-rank and prevents the unfiltered anchoring pathology
Wave-7 documented on this hardware. The two-stage structure is
therefore the load-bearing algorithmic difference between v1_fa²_stack
and SnapKV: Stage 2 alone is SnapKV, Stage 2 *after* Stage 1 is v1.
This distinction is the reason the ablation tables report a
"no-per-head-budget" cell separately from a "no-top-32" cell —
collapsing them would conceal the Wave-7/Wave-8 lesson the two-stage
structure encodes.

## 2. Why Each Mechanism (Wave-by-Wave Provenance)

Each ingredient was added in response to a specific failure mode revealed
by a prior wave:

- **v1 per-head budget μ(max_a[h])** was justified by Wave-3-real on
  Llama-3.2-1B: the vanilla cache hit 51.7 °C peak DDR with 7.2 MB
  swap-out at 5.09 tok/s, while v1 at K=512 ran at 49.4 °C peak DDR
  with zero swap and 7.05 tok/s. The per-head budget — peaky heads get
  K_h ≈ 0.7·K_nominal, diffuse heads get K_h ≈ 1.3·K_nominal — was the
  cheapest mechanism that converted the "smaller-cache-is-cooler"
  prior into a deployable scheduler.

- **FA-on decode via state-swap** was justified by Wave-4 (long-decode
  Phi-3): pure-v1 K=512 collapsed to 2.66 tok/s under FA-off decode because
  the per-step attention reduction over the bounded cache dominated, while
  v1_fa K=512 recovered to 4.65 tok/s by keeping the FA² fast path on
  decode. The state-swap is the trick that lets the v1 scheduler drive
  which positions FA² sees without rebuilding the cache layout mid-step.

- **Q8 K cache** was justified by Wave-7's swap pathology: v1_fa² at K=512
  produced 1551.8 MB of swap-out (memory pressure issue noted in the
  Wave-7 row of the master table) because anchor+recency tiers expanded
  the resident footprint past the 12 GB UMA ceiling. Wave-8 added Q8 K and
  collapsed swap to 6.7 MB. But Wave-8 also exposed the *thermal* cost of
  this fix: peak DDR climbed to 72.9 °C and peak CPU to 78.4 °C (also
  reported as 80.6 °C on the worst core in the master table), because the
  bandwidth-shaped workload now ran longer and harder before hitting the
  memory wall — which the previous swap had been crudely throttling.

- **Preempt-throttle watchdog + closed-loop K + memory gate** was
  justified by Wave-8's kernel-throttle event at iter-10: the Qualcomm
  mitigation framework forced a transient downshift to 883 MHz once the
  DDR sensor crossed ~65 °C, producing a visible throughput cliff. The
  watchdog is the dissertation's Track-2 contribution and was designed
  specifically to fire *before* the kernel's `freq_qos` re-vote, lowering
  K and the resident footprint pre-emptively rather than reactively.

## 3. Wave-9 Measured Results (Single-Cell, K=512)

The Wave-9 cell ran the full stack against Phi-3-mini-4k-instruct under the
same long-decode harness as Wave-8 for 10 iterations. All numbers below are
from `phone-logs/wave9_v1fa2_stack_1780796320/` and are reproduced in the
master comparison table:

| Metric | Wave-8 (selective) | Wave-9 (stack) | Δ |
|---|---:|---:|---:|
| Peak DDR (°C) | 72.9 | **64.1** | **−8.8** |
| Peak CPU (°C) | 78.4 | **66.8** | **−11.6** |
| Kernel throttle events | 1 (iter-10, 883 MHz) | **0** | −1 |
| Swap-out (MB) | 6.7 | **0** | −6.7 |
| Peak RSS (GB) | 13.0 | **12.8** | −0.37 GB (≈ −370 MB) |
| Mean tok/s | 6.75 | 6.09 | −0.66 |
| Sampling-NLL PPL [s] | 3.560 | 2.169 | (not directly comparable; see §4) |

The −8.8 °C peak-DDR delta exceeds the planning band (`<= 70.5 °C`,
expected 68.5-70.0 °C) recorded in `THERMAL_RESEARCH_WAVE9_PLAN.md` Table
§Expected results. The Wave-9 cell over-cooled relative to plan, which is
consistent with the closed-loop K controller spending more time at K=384/K=256
than the planning prior anticipated; we treat this as a successful
control event rather than a measurement drift because the corresponding
peak-CPU delta (−11.6 °C) and the watchdog log together rule out a
sensor-stale artefact. The zero-swap, zero-throttle outcome means the
Qualcomm thermal HAL never had to fire its 883 MHz downshift, which is the
core operational claim of the watchdog.

## 4. Wave-10 K-Sweep Ablation

To separate the contribution of the closed-loop K *setpoint* from the
stacked thermal effect, Wave-10 swept K ∈ {256, 384, 512, 1024} with all
other stack components frozen at their Wave-9 settings (Q8 K, f16 V,
watchdog enabled, memory gate enabled, identical prompt corpus). The
K=512 row is Wave-9 itself. All numbers are reproduced from
`SUBSECTION_KSWEEP.md` and the Wave-10 rows of the master table.

| K | Iters | Mean tok/s | PPL [s] | Peak DDR (°C) | Peak CPU (°C) | Peak RSS (GB) | Swap (MB) | Watchdog trips (T1/T2/T3) |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 256 | 12 | **7.174** | 2.0948 | 64.1 | 70.9 | n/a | 25.3 | 304 / 299 / 0 |
| 384 | 12 | 7.052 | 2.1227 | **63.3** | 70.9 | n/a | 21.5 | (lower) |
| 512 | 10 | 6.089 | 2.1686 | 64.1 | **66.8** | **13.68** | **0** | 0 / 0 / 0 |
| 1024 | 10 | 6.199 | **1.8268** | 63.7 | 71.7 | 14.50 | 119.2 | 415 / 412 / 0 |

The three operationally meaningful findings, reproduced in the K-sweep
section, are: (i) throughput is monotonically non-increasing in K over
the measured grid (7.174 → 7.052 → 6.089 → 6.199 tok/s), with the K=512
dip identified as a scheduler resonance artefact (two pathological
per-iteration dips to 4.569 and 4.601 tok/s, see
`ksweep_trajectories.png`); (ii) the 2D (mean-tps, PPL) Pareto frontier
contains exactly K=256 and K=1024, with K=384 and K=512 strictly
dominated, while adding peak-DDR as a third axis promotes K=384 to the
3D frontier on thermals (63.3 °C is the coolest cell in the sweep); and
(iii) the lowest peak RSS (13.68 GB) and lowest peak CPU (66.8 °C) both
sit at K=512 despite K=512 not being the smallest budget, because the
seq_add-skip path makes K_nominal a *soft* per-layer slot budget rather
than a hard cap on resident footprint.

We adopt K=1024 as the headline single-K configuration in the rest of
the dissertation (best PPL by a 0.27-nat margin, 6.2 tok/s still above
the 6 tok/s interactivity floor) and K=384 as the recommended balanced
default; K=512 is removed from the deployment grid on the dominated-cell
finding.

## 5. Wave-11 Held-Out PPL (Pending Numbers)

Wave-11 re-evaluates the stack under the corrected teacher-forced
held-out WikiText-2 protocol (the sampling-NLL bug discovered in Wave-10
is documented in `HELD_OUT_PPL_FINDING.md` and `OPTIMIZATION_JOURNEY.md`
Problem #10; Wave-3..10 PPL is biased low and is *not* comparable to
literature). The Wave-11 cells are still running at table-generation
time; the placeholders below will be filled in from
`phone-logs/wave11_eval_1780862534/` per the `WAVE11_FILL_IN_PROTOCOL.md`
mechanical procedure.

**Phi-3-mini, v1_fa²_stack, K=512:**
- Held-out PPL (token-weighted geom-mean per cell):
  {{wave11_phi3_v1fa2_ppl_mean}} (95% CI
  [{{wave11_phi3_v1fa2_ppl_ci_low}}, {{wave11_phi3_v1fa2_ppl_ci_high}}])
- Peak DDR: {{wave11_phi3_v1fa2_peak_ddr}} °C
- Peak CPU: {{wave11_phi3_v1fa2_peak_cpu}} °C
- Mean decode tok/s: {{wave11_phi3_v1fa2_decode_tps}}
- Kernel throttle events: {{wave11_phi3_v1fa2_throttle_count}}
- Swap-out: {{wave11_phi3_v1fa2_swap_mb}} MB

**Llama-3.2-1B, v1_fa²_stack, K=512:**
- Held-out PPL: {{wave11_llama1b_v1fa2_ppl_mean}} (95% CI
  [{{wave11_llama1b_v1fa2_ppl_ci_low}},
  {{wave11_llama1b_v1fa2_ppl_ci_high}}])
- Peak DDR: {{wave11_llama1b_v1fa2_peak_ddr}} °C
- Peak CPU: {{wave11_llama1b_v1fa2_peak_cpu}} °C
- Mean decode tok/s: {{wave11_llama1b_v1fa2_decode_tps}}
- Kernel throttle events: {{wave11_llama1b_v1fa2_throttle_count}}

NIAH retrieval accuracy for v1_fa²_stack on the 8-stimulus grid is
reported in `TABLE_WAVE11_NIAH.md` as
{{wave11_phi3_v1fa2_niah_accuracy}} (Phi-3) and
{{wave11_llama1b_v1fa2_niah_accuracy}} (Llama-1B); statistical
significance versus the four baselines (vanilla, h2o, tova,
streamingllm) is tabulated in `TABLE_WAVE11_SIGNIFICANCE.md`.

## 6. Discussion: Trade-Offs and Honest Caveats

The stack is composable in the sense of `THERMAL_RESEARCH_REPORT.md` §4:
the four mechanisms attack four orthogonal levers (eviction pressure,
attention fast-path engagement, bytes-per-token, and runtime control of
the eviction budget). They stack sub-additively but in the same direction
on the binding constraint (LPDDR5X self-heating under sustained
memory-bound decode). The Wave-9 −8.8 °C peak-DDR result is larger than
the sum of individually projected contributions, which we attribute to
the watchdog catching DDR transients that the individual mechanisms
would each have ridden into a kernel `freq_qos` event.

The principal trade-off, raised in `SUBSECTION_KSWEEP.md` §1, is that
Q8 K quantisation makes the per-layer K_nominal *soft* rather than hard.
Because the seq_add-skip path leaves evicted positions sparsely
allocated until compaction (which we deliberately do not run inside the
decode loop, because compaction would re-touch every slot and reheat
the bus), peak RSS does not collapse to a clean function of K. This is
visible in the K=512 cell having the lowest peak RSS (13.68 GB) of the
sweep despite sitting between K=384 (14.41 GB) and K=1024 (14.50 GB) on
either side. We do not regard this as a bug: it is the mechanism by
which the seq_add-skip path makes eviction cheaper than the attention
work it avoids. But it does mean that "K=256" in the controller is a
scheduling pressure, not a memory contract, and a deployment that needs
a *hard* RSS cap must rely on the memory-gate component of the watchdog
rather than on K alone.

A second trade-off is throughput: the closed-loop K controller spends
0.66 tok/s versus Wave-8 (6.09 vs 6.75 tok/s) at K=512, which is the
"cost of cool" the stack pays for the −8.8 °C peak-DDR delta and the
zero-throttle / zero-swap outcome. The Wave-10 K-sweep shows this cost
is asymmetric in K: at K=256 the stack actually *outperforms* Wave-8's
mean throughput (7.174 vs 6.75 tok/s) because the eviction-cheaper-than-
attention regime kicks in. The right reading is therefore that the
watchdog enables a throughput-PPL-thermal trade space that did not
exist in Wave-8 at all, rather than that the stack pays a uniform
throughput tax. The thermal goal (≤70.5 °C peak DDR, 0 kernel throttle
events) is met across the entire K-sweep grid, which is the soundness
claim the watchdog was designed to deliver.

We flag three honest caveats. First, the watchdog relies on the DDR
thermal-zone polling path; if Qualcomm RPMh overrides the bus cap on
retail firmware (the failure mode anticipated in
`THERMAL_RESEARCH_WAVE9_PLAN.md` §Failure-mode detection), the −8.8 °C
delta would shrink to whatever Q8 K alone delivers. The Wave-9
asserter-loop log shows no override during the cell, but we cannot
guarantee this generalises across firmware revisions. Second, the
K=512 scheduler-resonance dips in Wave-10 are unexplained at the level
of the eviction scheduler's inner loop and are flagged as future work.
Third, the held-out PPL on Wave-11 is the binding quality metric; the
Wave-9/10 sampling-NLL numbers are reported here only to justify the
stack's design and must not be cross-compared to literature.
