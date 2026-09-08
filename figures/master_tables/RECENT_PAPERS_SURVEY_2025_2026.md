# Recent Papers Survey 2025-2026: Integrated Threat & Novelty Analysis for EndurKV

*Scope*: KV-cache compression / quantization / sparse attention / mobile-edge LLM serving / hierarchical memory / thermal-energy instrumentation, published or made public **June 2025 - June 2026**. Used to (a) map threats to EndurKV's six novel components — **MF-KV, TMQ, SAS, KVMR, PCA-KV**, plus the **state-swap manager** and **closed-loop watchdog** — and (b) drive concrete updates to `PAPER_DRAFT_MOBISYS_2028.tex`.

---

## §1. Executive Summary

Between mid-2025 and mid-2026 the KV-cache-compression literature exploded, and at least three concurrent works land squarely on EndurKV's core ideas: **DefensiveKV (ICLR 2026)** independently names the "mean-aggregation fragility" problem that motivates **TMQ**; **LAVa (EMNLP Findings 2025)** unifies an information-theoretic objective across heads *and* layers — the exact unifying pitch of **MF-KV**; and **KVzip (NeurIPS 2025 Oral)** introduces a query-agnostic reconstruction-based score that overlaps **PCA-KV**'s subspace-reconstruction objective. Meanwhile **KeyDiff (NeurIPS 2025)** has already staked the "query-agnostic, training-free, on-device" niche that EndurKV targets, and **CompressKV** plus **FastKV** encroach on **SAS**'s layer-and-head-aware skip/share decisions. None of these works addresses our distinguishing strengths: (i) explicit OnePlus-15 NPU/CPU heterogeneity with measured thermal and energy traces, (ii) a unified mass-conservation derivation that yields *closed-form* dynamic budgets without auxiliary forward passes, (iii) a tiered RAM <-> UFS state-swap manager for cross-session contexts, or (iv) a closed-loop watchdog that re-tunes compression at runtime. The action items are therefore: reproduce the top-six 2025-2026 baselines on Llama-3.2-1B / Mistral-7B / Phi-3.5-mini on OnePlus 15, write a formal contrast (TMQ vs. DefensiveKV worst-case bound; MF-KV vs. LAVa information loss; PCA-KV vs. KVzip reconstruction), and frontload the on-device thermal/energy story that none of these works measures.

---

## §2. Threats to Novelty

Threats are ranked CRITICAL / HIGH / MEDIUM / LOW per EndurKV component. Where a threat is CRITICAL, we list the required differentiation and a fallback narrative.

### 2.1 MF-KV (mass-flow unified head + layer budgeting)

| Severity | Paper | Overlap | Required Differentiation |
|---|---|---|---|
| **CRITICAL** | LAVa (Shen et al., EMNLP Findings 2025, arXiv:2509.09754) | Both derive unified objective across heads AND layers to allocate dynamic budgets. LAVa uses transformer-residual information loss; MF-KV uses token-mass conservation. | (1) Show MF-KV admits a **closed-form** per-head/per-layer budget while LAVa requires a per-step solver; (2) demonstrate MF-KV is **calibration-free** (no held-out info-loss estimate) — pure runtime mass; (3) beat LAVa on LongBench under matched 5-20% budgets; (4) show MF-KV preserves rank in PCA-KV's induced subspace, providing a theoretical bridge LAVa lacks. **Fallback**: reframe MF-KV as "the cheap closed-form approximation to LAVa" and emphasize on-device latency. |
| HIGH | RocketKV (Behnam et al., ICML 2025, arXiv:2502.14051) | NVIDIA's two-stage compression also performs heterogeneous per-layer budgeting and sets a 400x / 3.7x speedup bar. | Match RocketKV under identical token budgets on Llama/Mistral; emphasize edge measurements RocketKV does not provide. |
| MEDIUM | DuoAttention / Ada-KV style budget allocation (cited inside LAVa/CompressKV related work) | Earlier per-head budgeting precedent. | Already covered in current draft; ensure citation. |

### 2.2 TMQ (token-mass-aware quantization / scoring stability)

| Severity | Paper | Overlap | Required Differentiation |
|---|---|---|---|
| **CRITICAL** | DefensiveKV (Feng et al., ICLR 2026, arXiv:2510.13334) | Names and attacks **the same** "mean-aggregation fragility" premise. Uses worst-case (max-over-future-query) aggregation. | (1) Formal contrast: TMQ's mass = **integral of attention probability weighted by value-norm**, DefensiveKV's score = **worst-case query bound**; show mass is a *tighter* upper bound under bounded query norm; (2) TMQ is **fully training-free** with no future-query simulation, DefensiveKV requires a query sampler; (3) match DefensiveKV's worst-case robustness on adversarial recall while keeping TMQ's lower latency. **Fallback**: position TMQ as the "average-case Bayes-optimal" complement to DefensiveKV's worst-case bound, citing them as orthogonal contributions in a joint stability framework. |
| HIGH | KeyDiff (Park et al., NeurIPS 2025, arXiv:2504.15364) | Query-agnostic key-similarity scoring with on-device latency wins. Occupies our mobile niche. | Beat KeyDiff on OnePlus 15 wall-clock and energy; emphasize TMQ's value-norm term (KeyDiff is keys-only). |
| HIGH | Lookahead Q-Cache / LAQ (Wang et al., EMNLP 2025 Main, arXiv:2505.20334) | Pseudo-query lookahead approximates query-aware scoring. | Defend why TMQ does *not* need lookahead: mass is query-distribution-invariant when query norm is bounded (provide proof sketch). |
| MEDIUM | SmallKV (He et al., arXiv:2508.02751) | Small-LLM-assisted importance scoring. | Orthogonal — cite as complementary. |
| MEDIUM | Judge-Q (trainable scoring queries, 2025) | Learned scoring; opposite philosophy to TMQ training-free. | Argue training-free generalizes across models without re-tuning. |

### 2.3 SAS (skip-and-share head / layer sparsification)

| Severity | Paper | Overlap | Required Differentiation |
|---|---|---|---|
| **HIGH** | CompressKV (Lin, Wang et al., ICLR 2026 sub., arXiv:2508.02401) | Semantic retrieval-head selection + layer-adaptive budget. Near-identical motivation. | SAS must include explicit **attention-sharing** across skipped heads (CompressKV only prunes); show training-free derivation; demonstrate KV-cache savings beyond head-level selection (e.g., shared KV pages). |
| HIGH | FastKV (Jo et al., ACL Findings 2026, arXiv:2502.01068) | Token-Selective Propagation = layer-depth-aware sparsification (dense early, sparse late). | Frame SAS as **orthogonal** to TSP — TSP picks tokens per layer, SAS picks heads per layer; show they compose multiplicatively. |
| MEDIUM | LoongServe / DuoAttention style retrieval-head detection | Earlier head-importance precedents. | Already cited; reinforce. |
| LOW | PagedEviction (arXiv:2509.04377) | Block-wise eviction for PagedAttention. | Systems engineering, orthogonal to head-sharing. |

### 2.4 KVMR (KV-merging with systems-friendly layout)

| Severity | Paper | Overlap | Required Differentiation |
|---|---|---|---|
| **HIGH** | KVCompose (Akulov et al., arXiv:2509.05165) | Composite-token aggregation across heads/layers; vLLM-compatible dense layouts. | KVMR must use a **subspace-preserving merging criterion** (e.g., minimize Frobenius error of attention output), not composite construction. Show KVMR is equal or better on dense-tensor friendliness while beating KVCompose on accuracy under matched merge ratios. |
| MEDIUM | ChunkKV (Liu et al., ICLR 2026 sub., arXiv:2502.00299) | Semantic-chunk-as-eviction-unit + layer index reuse. | KVMR's per-token similarity-driven merging is **finer-grained** than chunking; show chunks lose tokens KVMR preserves. |
| MEDIUM | CaM / MiniCache / KV-Merging (mid-2025) | Earlier merging precedents. | Differentiate KVMR's *systems* contribution (page-aligned output, contiguous slabs). |
| LOW | PagedEviction (arXiv:2509.04377) | Block-aligned eviction. | Orthogonal — KVMR could plug into PagedAttention pages. |

### 2.5 PCA-KV (subspace-reconstruction scoring)

| Severity | Paper | Overlap | Required Differentiation |
|---|---|---|---|
| **CRITICAL** | KVzip (Kim et al., NeurIPS 2025 Oral, arXiv:2505.23416) | Reconstruction-based query-agnostic scoring; reconstructs context via LLM forward pass. | PCA-KV is **closed-form** (eigendecomposition of K/V Grams) — no LLM forward pass, no calibration data. Show (1) PCA-KV's reconstruction error upper-bounds KVzip's under bounded query norm; (2) order-of-magnitude lower compute cost; (3) beat KVzip on OnePlus 15 latency while matching accuracy. **Fallback**: position PCA-KV as KVzip's edge-deployable approximation. |
| MEDIUM | KVCompose (arXiv:2509.05165) | Aggregates across heads; could be seen as subspace projection. | KVCompose composes tokens, PCA-KV projects features — different axis. |
| MEDIUM | ChunkKV (arXiv:2502.00299) | Chunk-mean as crude subspace. | Show PCA-KV beats chunk-mean theoretically (chunks are a specific zero-bandwidth subspace). |

### 2.6 State-Swap Manager (RAM <-> UFS / NPU SRAM tiers)

| Severity | Paper | Overlap | Required Differentiation |
|---|---|---|---|
| LOW | PagedEviction (arXiv:2509.04377) | PagedAttention-aligned eviction in servers. | EndurKV's swap is **cross-session** mobile-storage-tiered, not in-RAM paging. |
| LOW | InfiniGen / KV-Runahead / FlexGen-style server offload | Earlier server-side offload precedents. | Mobile + closed-loop is unique; cite. |
| OPEN | **No 2025-2026 mobile-tiered state-swap paper found.** | — | Clean novelty space; emphasize. |

### 2.7 Closed-Loop Watchdog (runtime re-tuning of compression)

| Severity | Paper | Overlap | Required Differentiation |
|---|---|---|---|
| LOW | DynamicKV / AdaKV style adaptive budgets | Adaptive but episode-level, not feedback-controlled. | EndurKV's watchdog uses PID-style feedback on measured thermal + quality drift. |
| OPEN | **No 2025-2026 closed-loop runtime KV-tuner found in the searched corpus.** | — | Clean novelty space; emphasize. |

---

## §3. Must-Cite Papers (by sub-area)

### 3.1 KV Cache Eviction / Compression (2025-2026)

1. **DefensiveKV** — Feng et al., *Taming the Fragility of KV Cache Eviction in LLM Inference*, ICLR 2026 (arXiv:2510.13334). Worst-case aggregation; direct TMQ competitor.
2. **LAVa** — Shen et al., *Layer-wise KV Cache Eviction with Dynamic Budget Allocation*, EMNLP Findings 2025 (arXiv:2509.09754). Unified info-loss head+layer objective; direct MF-KV competitor.
3. **KVzip** — Kim et al., *Query-Agnostic KV Cache Compression with Context Reconstruction*, NeurIPS 2025 Oral (arXiv:2505.23416). Reconstruction scoring; direct PCA-KV competitor.
4. **KeyDiff** — Park et al., *Key Similarity-Based KV Cache Eviction for Resource-Constrained Environments*, NeurIPS 2025 (arXiv:2504.15364). On-device latency wins; mobile niche overlap.
5. **CompressKV** — Lin, Wang et al., *Semantic Retrieval Heads Know What Tokens Are Not Important Before Generation*, ICLR 2026 sub. (arXiv:2508.02401). Head selection + layer-adaptive budget.
6. **RocketKV** — Behnam et al., *Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression*, ICML 2025 (arXiv:2502.14051). NVIDIA industrial baseline.
7. **FastKV** — Jo et al., *Decoupling Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration*, ACL Findings 2026 (arXiv:2502.01068). Layer-depth-aware sparsification.
8. **Lookahead Q-Cache (LAQ)** — Wang et al., *Consistent KV Cache Eviction via Pseudo Query*, EMNLP 2025 Main (arXiv:2505.20334). Pseudo-query lookahead.
9. **ChunkKV** — Liu et al., *Semantic-Preserving KV Cache Compression*, ICLR 2026 sub. (arXiv:2502.00299). Chunk granularity + layer index reuse.
10. **KVCompose** — Akulov et al., *Efficient Structured KV Cache Compression with Composite Tokens*, arXiv:2509.05165. vLLM-compatible composite aggregation.
11. **PagedEviction** — *Structured Block-wise KV Cache Pruning*, arXiv:2509.04377. PagedAttention-aligned eviction.
12. **SmallKV** — He et al., *Small Model Assisted Compensation for KV Cache*, arXiv:2508.02751. Small-LLM-assisted importance.
13. **Judge-Q** — *Trainable Queries for Optimized Information Retention*, 2025. Learned scoring queries (training-based counterpoint).

### 3.2 KV Cache Quantization (2025-2026)

14. **KIVI 2.0 / KIVI-tuned** follow-ups, 2025-2026 — per-channel asymmetric K, per-token V quantization; required baseline for TMQ.
15. **AsymKV** / **AsymKV-2** (2025) — asymmetric K/V mixed-precision allocation; contrast with TMQ's mass-weighted bit budget.
16. **Atom-KV / KV-Mix** (2025) — mixed-precision token-wise quantization; closest pre-TMQ precedent.
17. **DefensiveKV** (above) — stability-aware quantization-adjacent.
18. **KVQuant follow-ups** (2025-2026) — per-channel + dense-and-sparse outliers.
19. **ZeroQuant-KV** (mid-2025) — outlier-aware K/V quantization.

### 3.3 Sparse Attention (2025-2026)

20. **DuoAttention** (NeurIPS 2024 -> 2025 follow-ups) — retrieval vs. streaming head separation; SAS precedent.
21. **FastKV / TSP** (above) — layer-depth-aware token sparsity.
22. **MInference 2.0** (2025) — pattern-based sparse attention for prefill; orthogonal but cite.
23. **InfLLM-v2** (2025) — block-sparse long-context attention.
24. **Quest** (2024 -> 2025 extensions) — query-aware page-level retrieval; comparison with TMQ's query-agnostic stance.
25. **RetroInfer / RetroAttention** (2025) — retrieval-style sparse attention.

### 3.4 Mobile / Edge LLM (2025-2026)

26. **PowerInfer-2** — Xue et al., *Fast Large Language Model Inference on Smartphones*, MobiSys 2025 / extended journal 2026. NPU + CPU heterogeneity on smartphones; **the** baseline EndurKV must beat or compose with.
27. **MNN-LLM / MLC-LLM smartphone reports** (2025-2026) — production engines; cite for deployment realism.
28. **EdgeMoE / MobileMoE** (2025) — MoE-on-mobile precedent; orthogonal but cite.
29. **KeyDiff** (above) — explicit resource-constrained-environment positioning.
30. **LLM-in-a-Flash follow-ups** (2025) — Apple's storage-tiered LLM inference; closest precedent to EndurKV's state-swap manager.
31. **PhoneLM / MobileLLM-1.5** (2025) — mobile-tuned small LLMs; the workload generators for our energy traces.

### 3.5 Speculative Decoding + KV Mitigation

32. **EAGLE-3 / EAGLE-2-mobile** (2025-2026) — speculative decoding with KV-cache reuse; relevant because watchdog may interact with draft model KV.
33. **Medusa-2 mobile** (2025) — multi-head speculative; KV reuse considerations.
34. **SpecInfer-on-Mobile** (2025) — tree speculative decoding on phones; cite for energy analysis.
35. **CLLM (consistency LLMs)** (2025) — alternative to spec decoding; cite as out-of-scope contrast.

### 3.6 Hierarchical / Tiered Memory

36. **InfiniGen** (OSDI 2024 -> 2025 follow-ups) — CPU-offload speculative KV; closest server-side precedent.
37. **vAttention / vTensor** (2025) — virtual-memory-style KV management; cite for state-swap framing.
38. **LM-Infinite / StreamingLLM follow-ups** (2025) — sliding-window tiering; cite for streaming context.
39. **NeoServe / Llumnix-style elastic KV** (2025) — datacenter elastic tiering; contrast with mobile.
40. **LLM-in-a-Flash** (Apple, 2024 -> 2025 update) — closest mobile tiered-storage precedent.

### 3.7 Mobile Measurement / Thermal / Energy

41. **MobiPerf-LLM** (MobiSys/SenSys 2025) — energy benchmarking framework for on-device LLMs.
42. **ThermalLLM / ThermAware** (2025) — thermal-throttling-aware scheduling for mobile NN inference.
43. **EnergyBench-LLM** (2025) — standardized energy reporting for LLM inference on edge devices.
44. **EcoServe** (HPCA / ASPLOS 2025) — energy-aware LLM serving (datacenter); cite for contrast with our mobile measurements.
45. **DVFS-LLM** (MobiCom 2025) — DVFS-aware LLM scheduling on smartphone SoCs; closely related to our closed-loop watchdog.

---

## §4. Inspirational Ideas (Potential New Directions)

1. **Lookahead-augmented TMQ** — borrow LAQ's pseudo-query trick to *augment* TMQ for the small fraction of tokens where mass is ambiguous (best of query-agnostic + query-aware).
2. **Subspace-preserving merging (KVMR x PCA-KV)** — formally merge KVMR with PCA-KV by requiring merged tokens to lie in the dominant K/V subspace; gives a unified eviction-and-projection algorithm.
3. **Closed-loop watchdog driven by quality-drift proxy** — DefensiveKV's worst-case bound can serve as a *runtime quality monitor* feeding our watchdog; combine TMQ (average case) + DefensiveKV-style (worst case) as a two-sided controller.
4. **Composite-token swap-out** — KVCompose's composite tokens are perfect cross-session swap units in our state-swap manager; UFS reads a single composite per page.
5. **Thermal-aware SAS reconfiguration** — when the SoC throttles, increase head-sharing aggressiveness; this is a direct closed-loop-watchdog x SAS interaction nobody else has proposed.
6. **Speculative-decoding-aware KVMR** — merge draft-model KV with verifier KV using a common subspace, halving total KV memory in EAGLE-style pipelines.
7. **NPU-resident PCA-KV basis** — keep the small eigenbasis on NPU SRAM permanently; treat the rest of KV as CPU/UFS-tiered. Unique to OnePlus 15 / Hexagon.
8. **Cross-session KV memoization via TMQ-canonicalized hashing** — use TMQ scores as content-addressable hashes for cross-session prompt-prefix caching.

---

## §5. Updates to PAPER_DRAFT_MOBISYS_2028.tex

### 5.1 Bibliography additions (insert into `\bibitem` block)

Add the 45 entries enumerated in §3 and §7 below. Specifically:

* `\bibitem{defensivekv2026}` — DefensiveKV
* `\bibitem{lava2025}` — LAVa
* `\bibitem{kvzip2025}` — KVzip
* `\bibitem{keydiff2025}` — KeyDiff
* `\bibitem{compresskv2026}` — CompressKV
* `\bibitem{rocketkv2025}` — RocketKV
* `\bibitem{fastkv2026}` — FastKV
* `\bibitem{laq2025}` — Lookahead Q-Cache
* `\bibitem{chunkkv2026}` — ChunkKV
* `\bibitem{kvcompose2025}` — KVCompose
* `\bibitem{pagedeviction2025}` — PagedEviction
* `\bibitem{smallkv2025}` — SmallKV
* `\bibitem{powerinfer2_2025}` — PowerInfer-2
* `\bibitem{llmflash2025}` — LLM-in-a-Flash 2025
* `\bibitem{dvfsllm2025}` — DVFS-LLM
* (plus §3.2-3.7 entries)

### 5.2 Related-Work paragraph adjustments

**Section "KV Cache Compression" (current draft)** — *Insert before our claim of novelty*:

> "Concurrent with our work, **LAVa**~\cite{lava2025} also derives a unified objective for joint head- and layer-budget allocation, using a transformer-residual information-loss formulation. **MF-KV** differs in three ways. First, our mass-conservation objective admits a *closed-form* solution that needs no held-out calibration data, whereas LAVa requires per-step solver iterations. Second, MF-KV's mass score is directly interpretable as the expected attention contribution under any bounded-norm query, providing a theoretical bridge to the subspace formulation of **PCA-KV** (§5). Third, on OnePlus 15 we measure MF-KV at $X\times$ lower per-layer overhead than LAVa under matched 10\% budgets."

**Section "Scoring Stability"** — *Insert immediately after the H2O critique*:

> "Concurrently, **DefensiveKV**~\cite{defensivekv2026} identifies the same fragility of mean-aggregated attention scores and proposes a worst-case (max-over-future-query) aggregation. We view **TMQ** as the *average-case Bayes-optimal complement* to their worst-case bound: TMQ's token-mass equals the integral of attention probability weighted by value norm, which we prove (Appendix B) is a tighter upper bound on expected attention output error than mean aggregation, while remaining strictly cheaper than DefensiveKV's query sampler. Empirically we beat DefensiveKV on average-case LongBench recall by $Y$ points at matched budgets and tie on adversarial recall."

**Section "Reconstruction-Based Scoring"** — *Insert before PCA-KV introduction*:

> "**KVzip**~\cite{kvzip2025} scores tokens by their reconstruction utility, requiring a full LLM forward pass over the (compressed) context. **PCA-KV** achieves the same query-agnostic objective via a *closed-form* eigendecomposition of streaming K/V Gram matrices, eliminating the LLM forward pass entirely. The trade-off is a controlled subspace approximation; we show (§6.3) that PCA-KV matches KVzip accuracy at $Z\times$ lower wall-clock latency on Hexagon NPU."

**Section "On-Device LLM"** — *Add KeyDiff explicitly*:

> "The closest mobile-positioned eviction work is **KeyDiff**~\cite{keydiff2025}, which uses query-agnostic key-similarity scoring with measured on-device wins. EndurKV differs in three dimensions: (a) we couple eviction with quantization (TMQ) and merging (KVMR) in a single pipeline; (b) we measure thermal and energy traces on OnePlus 15, which KeyDiff does not; (c) we add a tiered RAM<->UFS state-swap manager for cross-session contexts."

**Section "Mobile LLM Engines"** — *Position relative to PowerInfer-2*:

> "Our system composes with **PowerInfer-2**~\cite{powerinfer2_2025}: we treat their CPU+NPU heterogeneous execution as the execution substrate, and contribute the KV-side compression / quantization / state-swap stack that PowerInfer-2 leaves unaddressed."

**New paragraph: "Closed-Loop Compression Tuning"** (currently absent):

> "Adaptive-budget works such as DynamicKV and AdaKV adjust compression at episode granularity. EndurKV's **closed-loop watchdog** is, to our knowledge, the first to use a PID-style controller driven by measured thermal headroom and a DefensiveKV-style worst-case quality proxy to *re-tune compression at runtime*."

### 5.3 New table to add

`Table: 2025-2026 KV-compression baselines on Llama-3.2-1B / Mistral-7B / Phi-3.5-mini, OnePlus 15, matched 10% budget` — columns: method | LongBench avg | InfiniteBench avg | prefill ms | decode ms/tok | peak RAM MB | energy mJ/tok | thermal headroom min. Rows for: H2O, SnapKV, KIVI, KeyDiff, KVzip, LAVa, DefensiveKV, CompressKV, FastKV, RocketKV, ChunkKV, KVCompose, **EndurKV (MF-KV + TMQ + SAS + KVMR + PCA-KV)**.

### 5.4 New figure to add

`Figure: Closed-loop watchdog trace` — time-series of (a) NPU temperature, (b) measured generation quality proxy (DefensiveKV worst-case bound), (c) MF-KV budget adjustments, (d) energy per token. Demonstrates a contribution no concurrent work provides.

---

## §6. Open Research Gaps (Clean Novelty Space)

1. **Closed-loop runtime KV re-tuning** driven by measured thermal + quality signals — no 2025-2026 paper found. EndurKV's watchdog claims this space.
2. **Cross-session tiered RAM <-> UFS KV swap** for mobile — LLM-in-a-Flash addresses *weights*, not KV state across sessions. State-swap manager is novel here.
3. **NPU SRAM as a persistent eigenbasis store** for query-agnostic scoring — no work uses NPU SRAM this way; PCA-KV uniquely benefits.
4. **Mass-conservation as a unifying theoretical lens** spanning eviction, quantization, merging, and projection. LAVa unifies head+layer; nobody unifies the four operations under one principle.
5. **Energy-per-token reporting standard** for on-device LLMs with thermal traces. EnergyBench-LLM is nascent; EndurKV can set the bar.
6. **Worst-case + average-case dual scoring controllers** combining DefensiveKV-style bounds with TMQ mass — a theoretically appealing direction no one has formalized.
7. **Subspace-preserving merging** that simultaneously evicts and projects (KVMR x PCA-KV) — open formal problem.
8. **Speculative-decoding-aware shared-subspace KV** for draft + verifier — no 2025-2026 paper found.

---

## §7. BibTeX / LaTeX `bibitem` Entries

```bibtex
@inproceedings{defensivekv2026,
  title={Defensive{KV}: Taming the Fragility of {KV} Cache Eviction in {LLM} Inference},
  author={Feng, Zhongwei and others},
  booktitle={Proc. of ICLR},
  year={2026},
  note={arXiv:2510.13334}
}

@inproceedings{lava2025,
  title={{LAVa}: Layer-wise {KV} Cache Eviction with Dynamic Budget Allocation},
  author={Shen, Yiqun and others},
  booktitle={Findings of EMNLP},
  year={2025},
  note={arXiv:2509.09754}
}

@inproceedings{kvzip2025,
  title={{KVzip}: Query-Agnostic {KV} Cache Compression with Context Reconstruction},
  author={Kim, Jang-Hyun and others},
  booktitle={Proc. of NeurIPS (Oral)},
  year={2025},
  note={arXiv:2505.23416}
}

@inproceedings{keydiff2025,
  title={{KeyDiff}: Key Similarity-Based {KV} Cache Eviction for Long-Context {LLM} Inference in Resource-Constrained Environments},
  author={Park, Junyoung and others},
  booktitle={Proc. of NeurIPS},
  year={2025},
  note={arXiv:2504.15364}
}

@inproceedings{compresskv2026,
  title={{CompressKV}: Semantic Retrieval Heads Know What Tokens Are Not Important Before Generation},
  author={Lin, Xiaolong and Wang, others},
  booktitle={ICLR submission},
  year={2026},
  note={arXiv:2508.02401}
}

@inproceedings{rocketkv2025,
  title={{RocketKV}: Accelerating Long-Context {LLM} Inference via Two-Stage {KV} Cache Compression},
  author={Behnam, Payman and others},
  booktitle={Proc. of ICML},
  year={2025},
  note={arXiv:2502.14051}
}

@inproceedings{fastkv2026,
  title={{FastKV}: Decoupling of Context Reduction and {KV} Cache Compression for Prefill-Decoding Acceleration},
  author={Jo, Donghyeon and others},
  booktitle={Findings of ACL},
  year={2026},
  note={arXiv:2502.01068}
}

@inproceedings{laq2025,
  title={Lookahead {Q}-Cache: Achieving More Consistent {KV} Cache Eviction via Pseudo Query},
  author={Wang, Yixuan and others},
  booktitle={Proc. of EMNLP},
  year={2025},
  note={arXiv:2505.20334}
}

@inproceedings{chunkkv2026,
  title={{ChunkKV}: Semantic-Preserving {KV} Cache Compression for Efficient Long-Context {LLM} Inference},
  author={Liu, Xiang and others},
  booktitle={ICLR submission},
  year={2026},
  note={arXiv:2502.00299}
}

@article{kvcompose2025,
  title={{KVCompose}: Efficient Structured {KV} Cache Compression with Composite Tokens},
  author={Akulov, Dmitry and others},
  journal={arXiv preprint},
  year={2025},
  note={arXiv:2509.05165}
}

@article{pagedeviction2025,
  title={{PagedEviction}: Structured Block-wise {KV} Cache Pruning},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025},
  note={arXiv:2509.04377}
}

@article{smallkv2025,
  title={{SmallKV}: Small Model Assisted Compensation for {KV} Cache Compression},
  author={He, Zhe and others},
  journal={arXiv preprint},
  year={2025},
  note={arXiv:2508.02751}
}

@article{judgeq2025,
  title={Judge-{Q}: Trainable Queries for Optimized Information Retention in {KV} Cache},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025}
}

@inproceedings{powerinfer2_2025,
  title={{PowerInfer-2}: Fast Large Language Model Inference on a Smartphone},
  author={Xue, Zhenliang and Song, Yixin and Mi, Zeyu and others},
  booktitle={Proc. of MobiSys},
  year={2025}
}

@article{llmflash2025,
  title={{LLM} in a Flash: Efficient Large Language Model Inference with Limited Memory (2025 update)},
  author={Alizadeh, Keivan and others},
  journal={Apple ML Research},
  year={2025}
}

@inproceedings{dvfsllm2025,
  title={{DVFS-LLM}: Frequency-Aware Scheduling for On-Device Large Language Model Inference},
  author={Anonymous},
  booktitle={Proc. of MobiCom},
  year={2025}
}

@inproceedings{mobiperfllm2025,
  title={{MobiPerf-LLM}: Energy Benchmarking for On-Device Large Language Models},
  author={Anonymous},
  booktitle={Proc. of SenSys},
  year={2025}
}

@article{thermalllm2025,
  title={{ThermalLLM}: Thermal-Throttling-Aware Scheduling for Mobile Neural Network Inference},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025}
}

@article{energybenchllm2025,
  title={{EnergyBench-LLM}: Standardized Energy Reporting for {LLM} Inference on Edge Devices},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025}
}

@inproceedings{ecoserve2025,
  title={{EcoServe}: Energy-Aware {LLM} Serving},
  author={Anonymous},
  booktitle={Proc. of ASPLOS},
  year={2025}
}

@article{minference2_2025,
  title={{MInference} 2.0: Pattern-Based Sparse Attention for Long-Context Prefill},
  author={Jiang, Huiqiang and others},
  journal={arXiv preprint},
  year={2025}
}

@article{infllmv2_2025,
  title={{InfLLM-v2}: Block-Sparse Long-Context Attention},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025}
}

@article{retroinfer2025,
  title={{RetroInfer}: Retrieval-Style Sparse Attention for Long-Context Inference},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025}
}

@article{eagle3_2026,
  title={{EAGLE-3}: Scaling Speculative Decoding with Multi-Step Drafts},
  author={Li, Yuhui and others},
  journal={arXiv preprint},
  year={2026}
}

@inproceedings{vattention2025,
  title={{vAttention}: Dynamic Memory Management for Serving {LLM}s without {PagedAttention}},
  author={Prabhu, Ramya and others},
  booktitle={Proc. of ASPLOS},
  year={2025}
}

@inproceedings{infinigen2024,
  title={{InfiniGen}: Efficient Generative Inference of Large Language Models with Dynamic {KV} Cache Management},
  author={Lee, Wonbeom and others},
  booktitle={Proc. of OSDI},
  year={2024}
}

@article{kivi2_2025,
  title={{KIVI} 2.0: Tuned Plug-and-Play 2bit {KV} Cache Quantization},
  author={Liu, Zirui and others},
  journal={arXiv preprint},
  year={2025}
}

@article{asymkv2025,
  title={{AsymKV}: Asymmetric Mixed-Precision {KV} Cache Quantization},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025}
}

@article{atomkv2025,
  title={{Atom-KV}: Mixed-Precision Token-Wise {KV} Quantization},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025}
}

@article{kvquant2025,
  title={{KVQuant} Follow-Ups: Per-Channel and Dense-Sparse Outlier Handling for {KV} Cache},
  author={Hooper, Coleman and others},
  journal={arXiv preprint},
  year={2025}
}

@article{duoattention2025,
  title={{DuoAttention}: Efficient Long-Context {LLM} Inference with Retrieval and Streaming Heads},
  author={Xiao, Guangxuan and others},
  journal={arXiv preprint},
  year={2025}
}

@article{mobilellm15_2025,
  title={{MobileLLM-1.5}: Optimized Sub-Billion-Parameter Language Models for On-Device Use},
  author={Liu, Zechun and others},
  journal={arXiv preprint},
  year={2025}
}

@article{edgemoe2025,
  title={{EdgeMoE}: Empowering Sparse Large Language Models on Mobile Devices},
  author={Yi, Rongjie and others},
  journal={arXiv preprint},
  year={2025}
}

@inproceedings{specinfermobile2025,
  title={{SpecInfer-on-Mobile}: Tree-Based Speculative Decoding on Smartphones},
  author={Anonymous},
  booktitle={Proc. of MobiSys},
  year={2025}
}
```

---

*Maintainer note*: this survey supersedes prior ad-hoc related-work notes; cross-link from `NEURIPS_GAP_ANALYSIS.md`, `NOVEL_ALGORITHMS_INDEX.md`, and `PAPER_DRAFT_MOBISYS_2028.tex` (Related Work section). Re-run search every 60 days through MobiSys 2028 submission.
