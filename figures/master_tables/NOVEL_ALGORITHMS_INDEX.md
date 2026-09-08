# Novel Algorithms Index — Synthesis Outcome

**Status:** **0 of 5 candidates survived.** No `NOVEL_ALG_<SHORTNAME>.md` proposals were produced.

This document is the honest record of what the speedup-forensics → ideation → novelty-audit → refutation pipeline produced, and why every candidate was rejected before a full proposal was warranted.

---

## 1. What the forensics step told us to attack

The forensics ranking identified **Q8 K-cache quantization** as the lever with by far the most remaining headroom (1.5–2.5x further KV-bandwidth reduction projected), with three concrete sub-axes:

1. Deeper K-quant (Q8 → Q4 → Q2)
2. V-cache quantization (currently V is forced f16 by the FA-off → FA-on state-swap invariant `V_STAYS_F16`)
3. Removing the `seq_add`-skip soft-cap via PagedAttention-style block-sparse FA so `peak_kv_cells` finally drops from ~2047 to K=512

FA-on kernel work was rated MEDIUM headroom (FA-3 warp-spec, ~10–15%), cache-size reduction past K=256 was rated steeply diminishing, and second-order swap-pressure was rated fully saturated at 0 MB.

This is the search space the five candidates were drawn against.

---

## 2. Candidates and verdicts

| # | Candidate | Lever it targeted | Novelty verdict | Reason it was rejected |
|---|-----------|-------------------|-----------------|------------------------|
| 1 | **MF-KV (Mean-Field KV Cache)** | KV compression via low-rank / mean-field summary of evicted tokens | NOT NOVEL | Subsumed by prior art: H2O (Zhang 2023), Scissorhands (Liu 2023), and especially **CaM / KV-Merger** (Zhang 2024) and **DMC** (Nawrot 2024) already merge evicted tokens into running summaries. "Mean-field" framing is a relabel; no new algorithmic primitive vs. weighted-mean merging that DMC and CaM already specify. |
| 2 | **TMQ-KV: Tile-Mixed K Quantization** | Mixed-precision K-cache: high-bits for salient tiles, low-bits for the rest | NOT NOVEL | **KVQuant (Hooper 2024)** already does per-channel + per-token mixed precision on K. **MiKV (Yang 2024)** and **GEAR (Kang 2024)** do mixed-precision residual quantization on KV. Tile-granularity is just a block size choice within the KVQuant design space, not a new mechanism. |
| 3 | **Speculative Attention Skip (SAS)** | Predict low-impact attention steps and skip the full K·Q·V read | NOT NOVEL | **Quest (Tang 2024)** already does query-aware page-level speculative skipping via per-page max/min bounds. **SeerAttention (Gao 2024)** learns gated sparsity. **MagicPIG (Chen 2024)** uses LSH-based speculative selection. SAS as described is a renaming of these. |
| 4 | **K-V Matching Reduction (KVMR)** | Detect K-V pairs whose contribution cancels and drop both | NOT NOVEL | The "cancellation" criterion reduces to a salience score, which is the H2O / Scissorhands / SnapKV objective. The proposal does not specify a cancellation test distinct from low cumulative attention weight, which is the existing eviction objective. No new signal vs. prior art. |
| 5 | **POSITION CLUSTERING ATTENTION (PCA-KV)** | Cluster positions and keep one representative per cluster | NOT NOVEL | This is exactly **CaM**, **KV-Merger**, **PyramidKV** (Zhang 2024) clustering, and **D2O** (Wan 2024) dynamic clustering. The "PCA" naming collides with principal component analysis but the mechanism described is k-means / agglomerative clustering on key vectors, which is published. |

---

## 3. Refutation attempts

The pipeline produced **0 refutation attempts** (the `refutation_attempts` list was empty). Because every candidate failed the novelty audit at step 3, none reached the refutation step. There is therefore no "survived refutation" cohort — the funnel emptied one stage earlier.

This is the correct behaviour: refuting a non-novel idea is wasted effort, since the refutation would amount to re-deriving the prior-art critique already produced by the novelty audit.

---

## 4. Why this outcome is not a failure of the forensics

The forensics correctly identified KV-quant depth, V-quant, and PagedAttention-style compaction as the highest-headroom directions. The ideation step then produced candidates that **landed inside** that high-headroom region — but the region is **densely populated by 2023–2024 prior art** (KIVI, KVQuant, KIVI, KVQuant, Quest, H2O, Scissorhands, SnapKV, PyramidKV, CaM, DMC, GEAR, MiKV, MagicPIG, SeerAttention, D2O). Any candidate that names KV-quantization, eviction merging, or speculative skipping as its primitive will collide with one of these.

Genuine novelty in this region therefore requires one of:

- **A new signal** the prior art does not exploit (e.g., HVX-microarchitecture-aware per-tile scheduling tied to Snapdragon SLC residency — this is **system-level**, not algorithm-level)
- **A new constraint** the prior art does not respect (e.g., the FA-off → FA-on state-swap invariant `V_STAYS_F16`, which forbids quantizing V at swap time — closing this gap would be novel because no prior work addresses it)
- **A composition** that prior work explicitly rules out as incompatible (e.g., KIVI-style INT4 K combined with PagedAttention block-sparse compaction under a fused-softmax FA-2 kernel on HVX — the integration story, not the components)

None of the five proposed candidates formulated their contribution along one of these axes. They were re-labelings of existing primitives.

---

## 5. Recommended next ideation pass

If a follow-up ideation round is run, instruct it to constrain candidates to the following novelty templates so they have a chance of surviving the audit:

1. **"State-swap-safe V quantization"** — an algorithm that lets V be quantized **across** the FA-off prefill → FA-on decode boundary without breaking the FA-2 fused-softmax invariant. This is the V_STAYS_F16 gap from `OPTIMIZATION_JOURNEY.md` / `V1FA2_STACK_FORMAL_SPEC.md §7.2`. No published work addresses this transition; KIVI/KVQuant assume a single regime.
2. **"seq_add-skip compaction under FA-2"** — a PagedAttention variant that uses non-contiguous KV pages **while preserving** the FA-2 fused-softmax kernel's contiguity assumption on HVX/NEON. PagedAttention (vLLM) does this for CUDA tensor cores; no published port exists for Snapdragon HVX with FA-2.
3. **"HVX-SLC-aware per-head budget"** — per-head K budget chosen to fit the Snapdragon 8 Gen 5 system-level cache residency window for the active decode tile. The signal (SLC fill ratio) is not used by any prior eviction policy; all prior policies use attention scores only.
4. **"FA-3 warp-spec port to HVX with Q8 K residency"** — integration novelty: FA-3 (Dao 2024) is CUDA-only; co-designing it with Q8 K so the dequant happens inside the warp-specialized producer would be new.

Each of these is a **gap** that the existing prior-art constellation explicitly leaves open, so a candidate framed against one of them has a defensible novelty claim.

---

## 6. Files produced

- `/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/NOVEL_ALGORITHMS_INDEX.md` (this file)
- No `NOVEL_ALG_<SHORTNAME>.md` files were produced, because no candidate survived.

---

## 7. Bottom line

**0 / 5 survived.** All five candidates were variants of published 2023–2024 KV-cache compression / eviction / speculative-skip ideas. The pipeline behaved correctly by rejecting them at the novelty-audit stage rather than dressing them up as new algorithms. The forensics-identified headroom is real, but capturing it requires either a system-integration contribution (FA-3 on HVX, PagedAttention on HVX, SLC-aware budgets) or closing the specific V_STAYS_F16 / seq_add-skip gaps unique to this codebase — not another general-purpose eviction or quantization heuristic.
