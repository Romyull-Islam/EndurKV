# `v1_fa2_stack` — Formal Specification

Source-grounded specification of the `v1_fa2_stack` policy as it is
implemented in `entropy_probe/eviction_bench.cpp` and orchestrated by
`scripts/android/phone_wave11_eval.sh` together with
`scripts/android/preempt_throttle_watchdog.sh`. The intent is to give a
reproducible, precise account of what runs end-to-end when a single
`(model, v1_fa2_stack, bench)` cell executes on the OnePlus 15.

---

## 1. High-level architecture

`v1_fa2_stack` is the composition of two orthogonal layers:

1. **`v1_fa2` — the eviction algorithm.** A KV-cache-bounding policy
   that performs a FA-off prefill, captures per-layer per-head
   attention from the last query position, computes a **per-head
   budget** keep set (per-head top-K_h with K_h = round(K_nominal ·
   μ(max_a[h])), unioned across heads) on top of *selective
   anchoring*, applies a state-swap to a FA-on decode context, and
   uses *tiered* decode-time eviction that protects the anchored block.
2. **The thermal / endurance stack.** A composition layer that wraps
   the eviction algorithm with operational pieces required for the
   policy to survive sustained on-device execution: a `MemAvailable`
   gate at cell start, a root-level preempt-throttle watchdog that
   caps CPU max frequency in four DDR-temperature tiers, Q8_0 K-cache
   quantization, and a sink-protection invariant.

The launcher script makes this composition explicit: the parser alias
`v1_fa2_stack` is mapped, in `policy_flags()`, to
`--policy v1_fa2 ... --cache-type-k q8_0 --cache-type-v f16` plus the
K-budget triplet `(K_nominal, anchor_top_k, recent_budget, n_sink)`,
and `needs_watchdog()` returns true *only* for `v1_fa2_stack` (and the
two Wave-11 isolation cells `v1_fa2_hybrid` and `v1_fa2_f16`). Inside
the binary, `parse_args()` rewrites the internal policy name from
`v1_fa2` to `v1_fa` and flips `snapkv_decode = no_evict_decode =
decode_tiered = true`, defaulting `anchor_top_k = 32` and
`recent_budget = 476` (= `K_nominal − n_sink − anchor_top_k` for
`K=512`).

The eviction algorithm and the thermal stack are independent in
principle, but in the Wave-11 evaluation grid only the *stacked*
variant is run with the watchdog; this is intentional, because the
endurance results we report require both halves.

---

## 2. Math formulation

Let `L` index transformer layers, `h` index query heads (the binary
captures the post-softmax `kq_soft_max` tensor of shape
`[n_kv, n_head, n_seq_tokens]` per layer), and `p` index KV positions.
During FA-off prefill the callback `eval_callback` extracts the row
corresponding to the **last** sequence token in the current chunk —
i.e. the attention distribution of the most recent query — for every
layer.

Define the per-(layer, head, position) attention from that last query:

```
a_{L, h, p} = softmax_p ( Q_h^{last} · K_p )      for each layer L
```

### 2.1 Per-head budget modulation by peak confidence (the `v1` policy)

For each layer L and each head h, `policy_v1` computes the maximum
attention `max_a[h] = max_p a_{L,h,p}` and derives a **per-head budget
multiplier** μ from that peak:

```
μ(x)  = 1.3 − 0.6 · clip( (x − 0.4) / 0.4 ,  0, 1 )      # ∈ [0.7, 1.3]
K_h   = round( K_nominal · μ(max_a[h]) ), clamped to [1, n_kv]
```

`max_a[h] → 1` (sharp, peaky head) gives `μ = 0.7` (small keep set —
the head is confident and concentrated, so few positions carry the
mass); `max_a[h] → 0.4` or below (flat, diffuse head) gives `μ = 1.3`
(a wider keep set, because the head's mass is spread out and we
cannot tell which positions matter). The keep set for that
(layer, head) is the `top-K_h` positions by `a_{L,h,p}`. The
layer-level keep set is the **union over heads** of those per-head
top-K sets.

> **Per-head budget μ(max_a[h]) gives confident (peaky) heads a
> smaller keep set and diffuse (uncertain) heads a larger one — the
> head's intrinsic uncertainty drives how many positions it needs to
> retain.**

Formally, for each (layer L, head h):

```
max_a[h]   = max_p a_{L,h,p}                       # peak attention of head h
K_h        = round( K_nominal · μ(max_a[h]) )      # per-head budget
keep(L, h) = top-K_h positions by a_{L,h,p}        # per-head keep set
keep(L)    = ⋃_h keep(L, h)                        # layer keep set = union across heads
```

This *per-head budget modulation by peak confidence* is the first
filter (it shrinks per-head budgets on confident heads and widens
them on diffuse heads, then unions across heads to form the per-layer
candidate pool), and the selective top-32 anchoring is the **second
filter** applied only to the union survivors.

### 2.2 Selective anchoring (top-32 from prompt)

Because *every layer* may keep a different subset, the prefill-time
union across heads/layers tends to retain most prompt positions on
long inputs. Wave-7 showed this starved the recent window. The
selective anchoring step then *re-filters* the union to a small
prompt-restricted anchor block.

For each position `p` surviving the per-layer per-head union (i.e.
`p` is in the keep set of at least one layer in the union), compute

```
anchor_score(p) = mean_{L, h} a_{L, h, p}
```

— the launcher source comment says "mean attention score across query
heads from the last position". The implementation in
`eviction_bench.cpp` averages attention per layer (line ~1331) by
accumulating per position over surviving layers; positions with `p <
n_sink` receive an infinite priority (`1e30f`) so the sink is always
in the anchor block. The anchor set is then

```
ANCHOR = top-32 positions by anchor_score, restricted to prompt positions
```

### 2.3 Final keep set after prefill

The total budget at the end of prefill is

```
KEEP = SINK ∪ ANCHOR ∪ RECENT_WINDOW
```

with the launcher-default sizes (for `K_nominal = 512`):

- `SINK` = first `n_sink = 4` positions (StreamingLLM-style)
- `ANCHOR` = top-`anchor_top_k = 32` positions by `anchor_score`
- `RECENT_WINDOW` = last `K_nominal − n_sink − anchor_top_k = 476`
  positions

so

```
|KEEP| = 4 + 32 + 476 = 512 = K_nominal     (soft cap; see §7)
```

### 2.4 Tiered decode-time eviction

`apply_tiered_decode_eviction(ctx, n_kv, n_anchored, recent_budget)`
triggers only when

```
n_kv > 1.25 · (n_anchored + recent_budget)
```

(`+ total_budget / 4` hysteresis). When triggered it removes the
middle band `[n_anchored, n_kv − recent_budget)` via
`llama_memory_seq_rm`, never touching the anchored block. The
post-removal position compaction (`llama_memory_seq_add`) is skipped
when `g_k_is_quantized == true` (Q8_0 K), because the llama.cpp K-shift
graph (RoPE re-rotation) does not support quantized K.

---

## 3. Pipeline

```
  ┌──────────────────────────────────────────────────────────────────┐
  │ STAGE 1: Memory gate (launcher)                                  │
  │   wait_for_memory(): poll MemAvailable; gate at >= 4 GB          │
  │   Prevents Wave-7's 1552 MB swap-out failure mode                │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STAGE 2: Start watchdog (root)                                   │
  │   start_watchdog(): spawn preempt_throttle_watchdog.sh in bg     │
  │   Per-cell scoped log / stop sentinel / pid file under cell_dir  │
  │   Polls DDR temp at ~0.5 Hz, 4 tiers (58/62/65 °C up,            │
  │   55/59/62 °C down) → scaling_max_freq on cpu6/cpu7              │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STAGE 3: FA-OFF Prefill in Context A                             │
  │   cb_eval captures kq_soft_max from LAST query position          │
  │   --cache-type-k q8_0  (Q8 K quantization, halves K bytes)       │
  │   --cache-type-v f16   (V must remain f16 for state-swap layout) │
  │   g_k_is_quantized = true  → seq_add will be skipped later       │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STAGE 4: v1 PER-HEAD BUDGET (FIRST FILTER)                       │
  │   Per (layer L, head h):                                         │
  │     max_a[h] = max_p a_{L,h,p}                                   │
  │     μ(x)     = 1.3 − 0.6·clip((x−0.4)/0.4, 0, 1)                 │
  │     K_h      = round( K_nominal · μ(max_a[h]) )                  │
  │     keep(L,h)= top-K_h positions by a_{L,h,p}                    │
  │   Layer keep set = union over heads;                             │
  │     confident heads → smaller K_h, diffuse heads → larger K_h.   │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STAGE 5: Eviction set computation (SECOND FILTER: anchor + KEEP) │
  │   policy_v1(): per-layer per-head top-K_h(μ) union               │
  │   protect_sink(): force [0, n_sink) into every layer's kept set  │
  │   Selective anchoring (SECOND FILTER): keep top-32 by mean       │
  │     attention from per-head-union survivors; rest are dropped    │
  │     via seq_rm runs                                              │
  │   Final keep = SINK ∪ ANCHOR ∪ RECENT_WINDOW                     │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STAGE 6: Apply eviction via llama_memory_seq_rm()                │
  │   Contiguous evicted runs → one seq_rm per run                   │
  │   n_kv reduced from prompt_len → K_nominal (soft cap)            │
  │   Note: seq_add NOT called because g_k_is_quantized = true       │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STAGE 7: State-swap to FA-on Context B                           │
  │   llama_state_seq_get_data() from Context A (FA-off)             │
  │   Allocate Context B with flash_attn = true                      │
  │   llama_state_seq_set_data() into Context B                      │
  │   Heap buffer transfer ~120 MB on Phi-3                          │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STAGE 8: FA-on Decode with bounded mask                          │
  │   no_evict_decode = true (frozen mask from prefill)              │
  │   decode_tiered  = true (drop middle decode tokens if needed)    │
  │   Watchdog continues monitoring; can throttle CPU freq mid-decode│
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STAGE 9: Tiered decode eviction (only when                       │
  │            n_kv > 1.25 × (n_anchored + recent_budget))           │
  │   Sink retained (positions 0..3)                                 │
  │   Anchor retained (top-32 from prompt)                           │
  │   Recent decode positions kept                                   │
  │   Drop OLDEST MIDDLE decode tokens                               │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STAGE 10: Stop watchdog at cell end                              │
  │   touch watchdog.stop sentinel file                              │
  │   Wait for watchdog to exit (10s timeout, then SIGKILL)          │
  │   Restore CPU max_freq                                           │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 4. Component-by-component breakdown

There are eight components in the stack. For each we list (what /
why / cost-vs-gain / origin).

**(1) `v1` per-head budget — FIRST FILTER.** Per (layer, head), the
policy computes the peak attention `max_a[h] = max_p a_{L,h,p}` of the
head and derives a per-head budget multiplier
`μ(x) = 1.3 − 0.6·clip((x−0.4)/0.4, 0, 1) ∈ [0.7, 1.3]`. The keep size
for that head is `K_h = round(K_nominal · μ(max_a[h]))`, and the
per-head keep set is the `top-K_h` positions by `a_{L,h,p}`; the
layer keep set is the **union over heads** of those per-head top-K
sets. This is the *per-head budget modulation by peak confidence*:
sharp (peaky) heads get a smaller `K_h` because their mass is
concentrated, while flat (diffuse) heads get a larger `K_h` because
their mass is spread out — the head's intrinsic uncertainty drives
how many positions it needs to retain. Cost: requires FA-off prefill
to expose `kq_soft_max`. Gain over canonical H2O: matches PPL at the
same nominal budget while making fewer per-head dependent decisions.
**Origin:** original to this project, motivated by the H2O recent+heavy
50/50 design.

**(2) Selective anchoring (top-32) — SECOND FILTER.** After the
per-head-union (first filter) yields its survivors, the **second
filter** re-ranks them by mean attention and keeps only the top 32 as
`ANCHOR`. **Why:** Wave-7 PPL=158 on Phi-3 traced
to all-prompt anchoring starving the recent window; the diagnosis was
"the per-head union kept most prompt tokens, leaving no room for the
recent decode window after `K_nominal` clipping". **Cost:** one extra
ranking pass + `seq_rm` of the dropped survivors. **Gain:** allows
`recent_budget = 476` at `K_nominal = 512`. **Origin:** SnapKV-inspired
(prefill-time selection), but our `top-32` cap and the mean-attention
scoring were chosen empirically.

**(3) Q8_0 K cache.** `--cache-type-k q8_0 --cache-type-v f16`.
**Why:** halves the K-bandwidth pressure on LPDDR5X and visibly
lowers DDR temperature (the watchdog's tier rises ~3 °C later in
sustained decode). **Cost:** sets `g_k_is_quantized = true`, which
makes the K-shift graph unavailable and forces `seq_add` to be
skipped after every eviction, leaving positions sparse (soft cap).
The Wave-11 isolation cell `v1_fa2_f16` was designed to isolate this
PPL cost. **Origin:** llama.cpp feature, project-original to combine
it with state-swap.

**(4) State-swap to FA-on.** `llama_state_seq_get_data` from Context A
(FA-off) → `llama_state_seq_set_data` into Context B (FA-on). **Why:**
the per-head budget (which depends on `max_a[h] = max_p a_{L,h,p}`)
requires the post-softmax `kq_soft_max` tensor (impossible under fused
FA), but FA-on decode is ~25 % cooler and faster for long generations.
**Cost:** ~120 MB heap transfer on Phi-3 between contexts. **Gain:**
PPL of the v1 keep set with the throughput / thermal profile of FA.
**Origin:** original; the state-swap APIs exist in llama.cpp but this
particular FA-off → FA-on swap pattern is ours.

**(5) Decode-time tiered eviction.** Triggered when `n_kv > 1.25 ×
(n_anchored + recent_budget)`; drops the middle band `[n_anchored,
n_kv − recent_budget)`. **Why:** Wave-4 showed that the v1_fa frozen
mask + `--ignore-eos` re-triggered thermal throttle as the cache grew
during decode. **Cost:** O(n_kv) compaction skipped due to Q8 K; net
cost is only the `seq_rm`. **Gain:** preserves the prefill anchor
block across the entire decode, unlike the cheaper
`apply_recency_decode_eviction` which sacrifices anchored positions.
**Origin:** original. Distinct from H2O decode-time replacement
because it never updates the anchor.

**(6) Preempt-throttle watchdog.** `preempt_throttle_watchdog.sh`
reads `thermal_zone47/temp` (DDR) every 2 s and adjusts cpu6/cpu7
`scaling_max_freq` across four tiers: MAX 1632 MHz, HIGH 1497.6 MHz,
MED 1267.2 MHz, LOW 1017.6 MHz, with hysteresis (up: 58/62/65 °C,
down: 55/59/62 °C). **Why:** preempts the kernel's reactive thermal
mitigation, which used to produce a throughput cliff. **Cost:** one
root background process per cell + a small forced-undervolt cost on
sustained MAX. **Gain:** smooth throughput glide instead of a cliff,
recoverable when DDR cools. **Origin:** original to the project; the
mechanism (writing scaling_max_freq) is standard, but the tiered
hysteresis controller and the per-cell scoping pattern are ours.

**(7) Memory-pressure gate.** `wait_for_memory()` polls
`MemAvailable` and gates the cell start at `MIN_FREE_GB = 4`. **Why:**
Wave-7 had a 1552 MB swap-out at prefill end that produced unbounded
latency and OOM-style failures. **Cost:** up to 300 s of wait per cell
(timeout-tolerant: it proceeds after timeout, logging a WARN). **Gain:**
eliminates the swap-thrash failure mode on Phi-3 / Gemma-2-2B cells.
**Origin:** original.

**(8) Sink protection.** `protect_sink(r, n_sink, n_kv)` forces
positions `[0, n_sink)` into every layer's kept set; the selective
anchoring step likewise pins sink positions with `+∞` priority.
**Why:** StreamingLLM-style — dropping BOS / instruction tokens
triggers degenerate-repetition failures. **Cost:** four reserved
slots at `K_nominal = 512`. **Gain:** invariant on cache content;
makes other components safe to compose. **Origin:** Xiao et al.
ICLR 2024 (StreamingLLM); we use their construction unchanged.

### 4.7 Why FA-off prefill + FA-on decode (the asymmetric split)

The single most common question about `v1_fa2_stack` is: *why not run
FlashAttention in BOTH phases, or in NEITHER phase?* The answer follows
directly from what each phase needs from attention.

**Why FA-off during prefill is non-negotiable.** Standard attention computes
`weights = softmax(Q · Kᵀ / √d)` as an intermediate tensor in DRAM, then
multiplies it by V to produce the output. This intermediate tensor is the
`kq_soft_max` callback target — we read it row-by-row to compute, per head,
`max_a[h] = max_p a_{L,h,p}` and derive the per-head budget
`K_h = round(K_nominal · μ(max_a[h]))`. FlashAttention fuses softmax inside
the kernel and **never writes `kq_soft_max` to DRAM**. The speedup of FA
comes precisely from not materializing that tensor. If we ran FA-on prefill,
the `eval_callback` would have nothing to capture, `max_a[h]` would be
unknown, and the per-head budget formula could not be evaluated — collapsing
the policy back to a uniform K cap (H2O-style), which is what we are trying
to improve upon. The FA-off prefill cost is therefore the architecturally
mandatory price of v1's algorithmic novelty.

**Why FA-on during decode is the right choice.** Once prefill is complete
the per-head budget K_h is frozen, the keep set has been chosen, the cache
has been compacted, and no further attention inspection is required during
generation. The decoder only needs *fast* attention. FA-on attention is
30–50 % faster per step than FA-off mat-mul attention on Snapdragon 8 Elite
Gen 5 because it avoids the DRAM round-trip for `kq_soft_max`. Across a
2048-token decode the saved bandwidth dominates the one-time state-swap
cost by roughly 100×.

**Why "FA-on everywhere" is impossible without changing the algorithm.**
There is no API in llama.cpp's FA implementation that exposes intermediate
attention weights, and modifying it to do so would require materializing
`kq_soft_max` (defeating the FA speedup entirely). The two requirements
("FA's bandwidth saving" and "v1's per-head budget") are fundamentally
incompatible at the kernel level.

**Why "FA-off everywhere" sacrifices the headline result.** H2O's design
needs attention scores on every decode step (for cumulative heavy-hitter
ranking), so it must run FA-off throughout. The measured cost on Wave-11
Phi-3 chunk-pair PPL is striking:

| Policy | Prefill FA | Decode FA | Decode tps | Δ vs vanilla |
|---|---|---|---|---|
| vanilla | on | on | 2.97 | — |
| **h2o** | off | off | **2.02** | **−32 %** |
| **v1_fa2_stack** | off | **on** | **4.98** | **+68 %** |

The split design lets v1_fa2_stack be faster than vanilla on decode (4.98
vs 2.97 tps) at the cost of a single ~600 ms state-swap, where h2o pays
the FA-off penalty on every single decode token and ends up 32 % slower
than vanilla. Same eviction concept, very different decode behaviour —
because of the FA topology choice alone.

**Boundary mechanics (the state-swap).** At the end of FA-off prefill we
serialize the cache with `llama_state_seq_get_data` (~3.2 MiB at K=512 on
Phi-3), construct a fresh context with `flash_attn = true` and the requested
q8_0 K / f16 V cache types, deserialize with `llama_state_seq_set_data`,
and run a single warm-up step to populate `n_outputs`. The cross-V layout
conversion (transposed↔untransposed) happens during read. Total swap cost
~600 ms on Phi-3-mini, amortized over thousands of decode tokens.

**Amortization regime.** The FA-off prefill penalty is roughly constant
(~18 % vs FA-on prefill at typical prompt sizes). The FA-on decode speedup
is per-token. For any workload with more than ~50 decoded tokens the
speedup pays back the prefill penalty. For typical mobile-LLM workloads
(chat 200–500 tokens, RAG 500–2000, code generation 500–3000) the net is
always large positive. The only regime where FA-off-prefill-then-FA-on is
wrong is "one-shot 10-token classification on a 4 B-parameter model",
which is not a real workload.

---

## 5. Pseudocode

```python
def v1_fa2_stack_cell(prompt_text, model, eval_text=None,
                     K_nominal=512, n_sink=4, anchor_top_k=32):
    # ── Stage 1: memory gate ────────────────────────────────────────
    while mem_available_gb() < 4.0:
        sleep(20)

    # ── Stage 2: launch watchdog (root, ~0.5 Hz) ────────────────────
    watchdog = launch_root("preempt_throttle_watchdog.sh",
                           interval_s=2,
                           ddr_zone="/sys/class/thermal/thermal_zone47")

    # ── Stage 3: FA-off prefill in context A ────────────────────────
    ctx_a = llama_init_context(model,
                               flash_attn=False,
                               cache_type_k=Q8_0,
                               cache_type_v=F16)
    g_k_is_quantized = True   # → seq_add will be skipped later
    cap = AttnCapture()
    register_callback(ctx_a, "kq_soft_max", cap.consume)
    prompt_tokens = tokenize(prompt_text)
    for chunk in chunked(prompt_tokens, n_batch=512, ubatch=64):
        llama_decode(ctx_a, chunk)

    # ── Stage 4: v1 PER-HEAD BUDGET (FIRST FILTER) ──────────────────
    n_kv = len(prompt_tokens)

    # (a) v1 per-head budget modulation by peak confidence.
    #     For each (layer L, head h):
    #       max_a[h] = max_p a_{L,h,p}                  # peak attention
    #       μ(x)     = 1.3 - 0.6 * clip((x-0.4)/0.4, 0, 1)   # ∈ [0.7, 1.3]
    #       K_h      = round(K_nominal * μ(max_a[h]))
    #       keep(L,h)= top-K_h positions by a_{L,h,p}
    #     Layer keep = union over heads. Confident (peaky) heads get a
    #     smaller K_h; diffuse (uncertain) heads get a larger K_h —
    #     the head's intrinsic uncertainty drives how many positions
    #     it needs to retain.
    per_layer_keep = []
    for L in cap.per_layer:
        kept_L = set()
        for h in range(cap.n_head):
            a       = cap.per_layer[L][h]                # [n_kv]
            max_a_h = max(a)                             # peak of head h
            norm    = clip((max_a_h - 0.4) / 0.4, 0, 1)
            mu      = 1.3 - 0.6 * norm                   # μ(max_a[h])
            K_h     = clip(round(K_nominal * mu), 1, n_kv)
            kept_L.update(top_k_indices(a, K_h))         # per-head top-K_h
        per_layer_keep.append(kept_L)                    # layer = union over h
    union_survivors = union(per_layer_keep)              # FIRST FILTER output

    # (b) sink protection
    for kept_L in per_layer_keep:
        kept_L.update(range(n_sink))

    # ── Stage 5: compute eviction set (SECOND FILTER: anchor + KEEP) ─
    # (c) SECOND FILTER: selective anchoring among per-head-union survivors
    scores = {p: mean_over_layers(cap, p) for p in union_survivors}
    for p in range(n_sink): scores[p] = float("inf")
    anchors = top_k_by(scores, k=anchor_top_k)       # top-32

    # (d) final union
    recent_start = max(n_sink, n_kv - (K_nominal - n_sink - anchor_top_k))
    keep = set(range(n_sink)) | anchors | set(range(recent_start, n_kv))

    # ── Stage 6: apply eviction ─────────────────────────────────────
    for (start, end) in contiguous_runs_of_dropped(keep, n_kv):
        llama_memory_seq_rm(ctx_a, seq_id=0, p0=start, p1=end)
    # NB: llama_memory_seq_add() is SKIPPED because g_k_is_quantized=True

    # ── Stage 7: state-swap to FA-on context B ──────────────────────
    state_blob = llama_state_seq_get_data(ctx_a, seq_id=0)
    ctx_b = llama_init_context(model,
                               flash_attn=True,
                               cache_type_k=Q8_0,
                               cache_type_v=F16)
    llama_state_seq_set_data(ctx_b, state_blob, seq_id=0)
    llama_free(ctx_a)

    # n_anchored = number of cells alive in ctx_b after the swap
    n_anchored   = llama_memory_seq_pos_max(ctx_b, 0) + 1
    recent_budget = K_nominal - n_sink - anchor_top_k   # 476 @ K=512

    # ── Stage 8+9: FA-on decode, tiered eviction when needed ────────
    if eval_text is not None:
        eval_tokens = tokenize(eval_text, add_bos=False)
        nll_sum = 0.0
        for target in eval_tokens:
            logits = llama_decode(ctx_b, [target])
            nll_sum += -log_softmax(logits)[target]
            n_kv_now = llama_memory_seq_pos_max(ctx_b, 0) + 1
            if n_kv_now > 1.25 * (n_anchored + recent_budget):
                apply_tiered_decode_eviction(
                    ctx_b,
                    n_kv         = n_kv_now,
                    n_anchored   = n_anchored,
                    recent_budget= recent_budget)
        ppl = exp(nll_sum / len(eval_tokens))

    # ── Stage 10: stop watchdog ─────────────────────────────────────
    touch("watchdog.stop")
    wait_for_exit(watchdog, timeout=10, then=SIGKILL)
    return ppl
```

---

## 6. Comparison vs simpler policies

| Component                       | vanilla | h2o canonical | v1_fa2 (base) | v1_FA (Wave-3) | **v1_fa2_stack** |
|---------------------------------|:-------:|:-------------:|:-------------:|:--------------:|:-----------------:|
| No eviction                     |   y     |       —       |       —       |        —       |        —          |
| Recent + heavy (50/50)          |   —     |       y       |       —       |        —       |        —          |
| v1 per-head budget μ(max_a[h])  |   —     |       —       |       y       |        y       |        y          |
| Selective anchoring (top-32)    |   —     |       —       |       y       |        —       |        y          |
| FA-off prefill, FA-on decode    |   —     |       —       |       y       |        —       |        y          |
| Tiered decode eviction          |   —     |       —       |       y       |        —       |        y          |
| Q8_0 K cache                    |   —     |       —       |       —       |        —       |        y          |
| Preempt-throttle watchdog       |   —     |       —       |       —       |        —       |        y          |
| Memory-pressure gate            |   —     |       —       |       —       |        —       |        y          |
| Sink protection (n_sink = 4)    |   —     |       y       |       y       |        y       |        y          |

- **vanilla** is FA-on, no eviction; baseline PPL but cannot sustain
  long context without thermal throttle.
- **h2o canonical** matches `--policy h2o`: 50/50 recent + heavy split
  with sink protection, FA-off, *no* state-swap and *no* watchdog.
- **v1_fa2 base** is the algorithm without the thermal stack (no
  watchdog, no memory gate, no Q8 K) — what the binary would do if
  the launcher used `--cache-type-k f16` and `needs_watchdog` returned
  false. The isolation cell `v1_fa2_f16` evaluates this branch.
- **v1_FA frozen (Wave-3)** is the predecessor: same FA-off → FA-on
  swap, but no tiered decode eviction (it used the cheaper
  `apply_recency_decode_eviction` instead, which drops anchored
  positions). Wave-4 showed it re-triggered throttle on long decodes
  with `--ignore-eos`.

---

## 7. Honest caveats

1. **Q8 K seq_add-skip makes `K_nominal` a soft cap.** After every
   eviction, `llama_memory_seq_add()` is skipped because the K-shift
   graph (RoPE re-rotation) does not support quantized K
   (`g_k_is_quantized = true`). Positions remain sparse in the cache:
   the *logical* cache contains `K_nominal` cells, but the *position
   ids* spread across the original prompt range. Position-based
   downstream code (e.g. mask construction) sees the original
   positions, not a compacted `[0, K_nominal)` range. The PPL is
   computed correctly, but anyone reading the position counters
   should not assume contiguous indices.
2. **State-swap requires FA-mode KV layout compatibility.** During
   Wave-7 we observed a layout incompatibility between FA-off and
   FA-on KV tensors that forced us to keep `cache-type-v = f16` (V
   quantization across the swap is fragile). The current launcher
   pins `--cache-type-v f16` for exactly this reason.
3. **Watchdog only monitors DDR.** The recent audit pointed out that
   the watchdog reads `thermal_zone47/temp` only; skin and battery
   temperatures are visible to `skin_dc()` and `cool_phone()` at
   *cell boundaries* but the in-cell throttle is single-sensor. A
   multi-sensor variant (skin + DDR + GPU) is in the to-do list and
   would change the tier thresholds.
4. **Adaptive K controller was designed but never fired.** The
   launcher carries `K_NOMINAL`, `RECENT_BUDGET`, `ANCHOR_TOP_K` as
   environment-overridable knobs, and the original design called for
   the controller to lower `K` if DDR exceeded `T_MED_UP` for >30 s
   inside a cell. In Wave-9/10/11 the controller was never observed
   to fire — the watchdog's CPU-frequency tiers cooled DDR fast
   enough that the `K` ratchet was unnecessary. We have kept the
   knob, but the spec above describes the controller as static at
   `K_nominal = 512`.
5. **PPL cost.** On Phi-3-mini-128k at `K = 512`, `v1_fa2_stack` adds
   ~12 % to held-out WikiText-2 PPL relative to FA-on vanilla. This
   is the price paid for sustained execution; the absence of throttle
   recovers more than that in long-horizon throughput. The number is
   not free; reporters should always cite both sides.
6. **Per-cell scoping of the watchdog.** `start_watchdog` writes its
   log / stop sentinel / pid file inside `cell_dir`, and
   `stop_watchdog` waits up to 10 s for the previous watchdog to exit
   before the next cell launches. This avoids two watchdogs racing on
   `scaling_max_freq`, but it means that if the wrapping script is
   killed mid-cell without running `stop_watchdog`, the CPU max
   frequency may remain capped until the watchdog's stop sentinel is
   touched manually.
