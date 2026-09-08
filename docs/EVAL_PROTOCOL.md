# Evaluation Protocol — adapting "Cache Me If You Can" (Bhaskar et al., 2025)

This document records the evaluation protocol we use for EndurKV-Evict's
phone-deployment experiments. We adopt the published methodology of
[Bhaskar et al. (2025), *"Cache Me If You Can: How Many KVs Do You Need for
Effective Long-Context LMs?"*, arXiv:2506.17121](https://arxiv.org/abs/2506.17121)
(Princeton PLI). Code at <https://github.com/princeton-pli/PruLong>.

Adopting their protocol gives us:
1. **Reviewer-defensible methodology** — "we use the published protocol of …"
2. **Direct numeric comparability** with their numbers (and PruLong's) where models match
3. **The KV footprint metric** — a more honest summary than mass-retained / retention-ratio

---

## 1. Benchmarks and tasks

The CMIYC paper evaluates on the following long-context suites:

| Suite | Tasks used | Native metric |
|---|---|---|
| **LongBench** | qasper, hotpotqa, narrativeqa, multiwoz, 2wikimqa, musique, gov_report, trec-covid, passkey | F1 / EM / ROUGE-L / accuracy (per task) |
| **RULER** | retrieval subtasks at multiple context lengths | accuracy |
| **NIAH** (Needle in a Haystack) | needle-recall | percent retrieval success |
| **HELMET** | long-context benchmark suite | per-task |

**What we'll use** (subset matching our current prompts on phone):

| Task | Prompt IDs on phone | Status |
|---|---|---|
| `qasper` | qasper_pub_001 … 005 | ✅ all 5 staged |
| `hotpotqa` | hotpotqa_pub_001 … 005 | ✅ all 5 staged |
| `multifieldqa_en` | multifieldqa_en_pub_001 … 005 | ✅ all 5 staged |
| `gov_report` | gov_report_pub_001 … 005 | ✅ all 5 staged |
| `narrativeqa` | narrativeqa_pub_001 … 005 | ✅ all 5 staged |

That's **25 prompts × 5 LongBench task families**. We follow CMIYC's
selection. Each task has its own primary metric — see §4.

NIAH and RULER are not currently on phone — Wave-6 future work.

---

## 2. Models evaluated by CMIYC vs. our choices

CMIYC evaluates Llama-2/3 (7B, 70B), Gemma (2B, 9B), Mistral (7B). Native
contexts span 4 K – 128 K.

Our hardware (OnePlus 15, Adreno 840) constrains us to <7 B params on GPU.
Our 3-model selection covers three distinct architectures:

| Model | Params | Family | Context | CMIYC overlap? |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1 B | Llama-3.x | 131 K | extends below CMIYC's range |
| Gemma-2-2B-it | 2 B | Gemma-2 | 8 K | extends below CMIYC's 9B |
| Phi-3-mini-128k-instruct | 3.8 B | Phi-3 | 128 K | not in CMIYC — distinct architecture |

We add **Phi-3** as a third arch (CMIYC doesn't cover it) and report
sub-7B results as the mobile-deployment regime (CMIYC's 70B numbers are
out of scope for phones).

---

## 3. K-budget grid — **adopted in fractional form**

CMIYC parameterizes the KV budget as a **fraction of the input sequence length**
(0.01× → 1.0× = full cache). This is more meaningful than absolute K
because attention-head importance scales with seq length.

We adopt the fractional convention. Concretely:

| K-fraction | What it means | Example for 4096-token prompt |
|---|---|---|
| 0.05× | extreme compression | K_total = 205 |
| 0.10× | aggressive | K_total = 410 |
| 0.25× | moderate (TOVA/SnapKV typical) | K_total = 1024 |
| 0.50× | mild | K_total = 2048 |
| 1.00× | full cache (vanilla) | K_total = 4096 |

**For our `K_nominal` flag**, we convert: `K_nominal = round(K_fraction × n_prompt / n_layers)`
(per-layer budget). The Wave-5 sweep will run at K-fractions {0.05, 0.10, 0.25, 0.50}.

---

## 4. Metrics per task — adopted from CMIYC primary metric column

| Task | Primary metric | How we score |
|---|---|---|
| qasper | F1 (token overlap) | `host_score_phone_runs.py f1_score()` ✅ |
| hotpotqa | F1 | same ✅ |
| multifieldqa_en | F1 | same ✅ |
| narrativeqa | F1 | same ✅ |
| gov_report | ROUGE-L | `host_score_phone_runs.py rouge_l()` ✅ |
| 2wikimqa / musique | F1 | same (we'd add these prompts in Wave-6) |
| trec-covid / passkey | EM / accuracy | `em_score()` ✅ |
| NIAH | needle-recall | `needle_recall()` ✅ |

Plus the **KV footprint** metric (§5) as the system-side primary number.

---

## 5. KV footprint — the unified system-side metric

**CMIYC definition** (paraphrased from §3.1 of their paper):

> "KV footprint accounts for both the **amount** of KV entries stored and
> their **lifespan** in memory. We compute it as the integral over decoding
> time of the total bytes occupied by the KV cache."

### Formula we'll compute

```
footprint = Σ_{t=0..T_decode}  n_kv_cells(t) × per_token_bytes
```

where:
- `n_kv_cells(t)` = number of KV positions in cache at step t (from `steps.csv`)
- `per_token_bytes` = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element
  - K + V (×2)
  - per layer (×n_layers)
  - per kv-head (×n_kv_heads)
  - per element of head_dim
  - fp16 = 2 bytes (Q4_K_M models still hold KV in fp16)

For each model on phone:

| Model | n_layers | n_kv_heads | head_dim | bytes_per_token (KB) |
|---|---|---|---|---|
| Llama-3.2-1B  | 16 | 8 | 64 | 2×16×8×64×2/1024 = **32 KB** |
| Gemma-2-2B    | 26 | 4 | 256 | 2×26×4×256×2/1024 = **104 KB** |
| Phi-3-128k    | 32 | 32 | 96 | 2×32×32×96×2/1024 = **384 KB** |

### Footprint contrasted with our existing peak_kv_mb

- `peak_kv_mb` = MAX over time of `n_kv_cells(t) × per_token_bytes / 1MB`
- `footprint`  = INTEGRAL over time of same — penalizes a policy that holds
  the cache HIGH for the whole decode

If a policy peaks early and shrinks quickly, footprint is much smaller than
peak. If a policy holds peak throughout, footprint ≈ peak × time. v1's
"shrinks the cache during prefill, holds it small during decode" pattern
should give a **smaller footprint** than vanilla (which never shrinks).

### Reporting convention

CMIYC reports footprint normalized to the full-cache (vanilla) baseline:
**footprint_ratio = footprint(policy) / footprint(vanilla)**, lower is better.

---

## 5b. Baseline fairness disclosures (audit notes)

This block records **deliberate deviations from each baseline's published spec**
that apply to ALL eviction policies in our comparison, so we can cite the
strengthened-baseline framing honestly in the paper.

### Sink protection — applied uniformly to ALL eviction policies

Every eviction policy in our codebase (v1, TOVA, pyramid, future H2O / AdaKV)
runs with `--n-sink 4`, meaning the first 4 KV positions (BOS + role tokens) are
**never evicted**. This is the StreamingLLM-style "attention-sink" protection
from Xiao et al. ICLR 2024.

**This is a strengthening of TOVA over its original paper.** The original
TOVA (Oren et al. ACL 2024) does **not** explicitly protect sink positions. Pure
TOVA can suffer the "sink-collapse" failure mode where dropping the BOS token
triggers degenerate-repetition during generation.

By adding `--n-sink 4` to TOVA in our comparison, we are giving TOVA an
advantage it does not have in its original paper. This makes the v1 vs TOVA
comparison **more demanding for v1** — we are not comparing v1 against a weakened
TOVA. Specifically:

| Variant | Pure paper-spec TOVA | Our "TOVA + sink" | Our v1 |
|---|---|---|---|
| First 4 positions evictable? | ✅ Yes (paper) | ❌ No (we added protection) | ❌ No (same) |
| Outcome on long-context generation | Often degenerates (sink-collapse) | Doesn't degenerate from this cause | Doesn't degenerate from this cause |

So when our results show v1 ≥ TOVA on F1/quality, this is **against a stronger
TOVA** than the literature reports. If we revert TOVA to no-sink-protection,
the gap would likely widen further in v1's favor — but that would not be a
fair comparison.

### Recent-window protection — NOT applied to any policy

We **do not** explicitly protect the most-recent N positions in any policy.

| Paper | Original spec |
|---|---|
| TOVA (Oren ACL 2024) | No explicit recent-window protection |
| StreamingLLM (Xiao ICLR 2024) | Sink + last-N window |
| H2O (Zhang NeurIPS 2023) | Heavy-hitters + recent window |
| SnapKV (Li NeurIPS 2024) | Observation-window-based |

In practice, recent positions tend to have high attention naturally
(causal-mask diagonal), so they're often preserved by attention-magnitude
selection anyway. But we did not add an explicit `--n-recent K` flag. If we
later add one, it must apply uniformly to v1, TOVA, etc.

### GQA aggregation — Aggregate-OR across query heads

When a model uses GQA (Llama-3.2-1B has 32 Q heads / 8 KV heads, ratio 4:1),
one KV position is shared across multiple query heads. The keep/evict decision
must be one-per-KV-head, not one-per-query-head. We use **Aggregate-OR**:
a KV position is kept if **any** query head in its group wants it.

This is the same convention used in published GQA-aware eviction implementations
(AdaKV, SnapKV, modern TOVA forks). It is more conservative (keeps more) than
Aggregate-MEAN. Both v1 and TOVA share this convention.

### Eviction granularity — sequence-level (llama.cpp constraint)

llama.cpp's `llama_memory_seq_rm` operates per-sequence, not per-layer. We can't
evict different positions in different layers within stock llama.cpp. So we
take the **union** of per-layer keep masks and emit a single seq-level eviction.

| Variant | What it does |
|---|---|
| Ideal (research): per-layer eviction | layer i can keep different positions than layer j |
| Our impl: union across layers | a position is evicted only if **no** layer wants it |

This is **strictly more conservative** than per-layer (keeps more), which is
again a strengthening that applies uniformly to all policies. The relative
comparison is preserved.

### Decoding — greedy (deterministic) for the publication comparison

`--greedy` is used in the CPU sweep so the comparison is RNG-free. Sampling
parameters (`--top-p`, `--top-k`, `--temperature`) are NOT used in greedy mode
even though they're set in the CLI — they're inert under `--greedy`.

---

## 6. Baselines compared in CMIYC

CMIYC compares: TOVA, H2O, SnapKV, PyramidKV, DUO-Attention, plus
random / greedy controls, plus their **PruLong** contribution.

For our phone deployment we currently have: vanilla, TOVA, v1. The CMIYC
protocol implies adding **at least H2O** for reviewer expectation. PruLong
itself would be the strongest 2025 competitor to add, but their method
involves training, not a runtime change. Citing PruLong as concurrent work
and not directly competing-against is defensible.

---

## 7. Post-fill vs pre-fill eviction

CMIYC's main observation: **pre-fill eviction has much smaller footprint**
than post-fill eviction, because evicted KVs never had to be allocated in
the first place. Their PruLong is a pre-fill method.

Our v1/TOVA implementation evicts **after** the prefill `llama_decode` —
that's post-fill. We pay the peak memory for the full prompt prefill, then
shrink. This is the standard llama.cpp pattern.

**Implications**:
- Our `peak_kv_mb` reflects the post-fill peak, which is high for all policies
  except vanilla-no-eviction (where there is no shrink).
- Our **footprint** is lower than vanilla, because we shrink immediately
  after prefill and stay small during decode.
- **Honest disclosure** in the paper: "we report post-fill eviction; the
  pre-fill regime is a known stronger setting (Bhaskar et al. 2025) which
  llama.cpp does not natively support and is left to future work."

---

## 8. Chat templates and decoding

CMIYC uses model-native chat templates + greedy decoding. **We already do
this** (chat templates wrapped via `host_wrap_prompts_chat.py`; greedy via
`--greedy` flag in `eviction_bench`). ✅

---

## 9. What this changes in our pipeline

1. **`host_score_phone_runs.py`** (already exists) — applies per-task
   primary metric from §4. No change needed.

2. **Aggregator** (new) — add `kv_footprint_mb_steps` column computed from
   `steps.csv`. Formula: `Σ n_kv_cells × per_token_bytes_for_model`.
   This becomes the system-side primary number.

3. **K-budget grid sweep** (Wave-5 expansion) — replace fixed `K=1024` with
   a sweep over K-fractions {0.05, 0.10, 0.25, 0.50}.

4. **Reporting** — every quality-vs-budget plot has K-fraction on x-axis
   (not absolute K). Every system-side plot has footprint_ratio on x-axis.

---

## 10. Citation language for the paper

> "We adopt the evaluation protocol of Bhaskar et al. (2025), 'Cache Me If
> You Can: How Many KVs Do You Need for Effective Long-Context LMs?',
> including the KV footprint metric, fractional K-budget sweep, and primary
> per-task metrics from LongBench. We extend the evaluated regime to
> mobile-class models (≤4B params) deployed on a real Snapdragon mobile
> SoC (OnePlus 15, Adreno 840 GPU)."

This pins our methodology to a published reference and isolates our
contribution (mobile-deployment + thermal/endurance signals) cleanly.
