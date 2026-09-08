# Ada-KV / AhaKV Differentiation Audit and Edit Plan

**Target file:** `/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/PAPER_DRAFT_MOBISYS_2028.tex`
**Date:** 2026-06-12
**Audit scope:** Ada-KV (NeurIPS 2025), CriticalKV (ICML 2026), DefensiveKV / Layer-DefensiveKV (ICLR 2026), AhaKV (arXiv:2506.03762).

---

## 1. Headline verdict

**No preempt of any *measured systems* claim.** AhaKV, CriticalKV, and DefensiveKV/Layer-DefensiveKV are all quality-only GPU papers. None report on-device throughput, thermals, DRAM bandwidth, RSS, swap, or battery energy. The EndurKV systems contributions (FA-off→FA-on state-swap, Q8 K + f16 V composition, 5-Hz multi-sensor watchdog, OnePlus 15 measurement protocol, disable-charging energy methodology) remain untouched.

**Two *algorithmic* claims need to be tightened**, but neither requires retracting a result:

- **(a)** Our Layer 1 description currently calls Ada-KV's allocator a "frequency-of-selection $f_i$" rule. That is accurate but incomplete — we omit the **$\alpha=0.2$ safeguard** (Algorithm 2 line 8) and the **SnapKV max-pool kernel-7 / window-32 plumbing** that `policy_adakv` already replicates verbatim in our codebase. Section §2 (line 141) and §3.1 (line 228) should both be updated to name these two pieces. Honesty cost: zero — we already implement them in `policy_adakv`.

- **(b)** The "per-head allocation" family has now grown to a *line of work* by Feng et al. (Ada-KV → CriticalKV → DefensiveKV / Layer-DefensiveKV). We should acknowledge the lineage in one sentence so reviewers don't think we are unaware of the 2026 follow-ups. None of them does what we do (on-device measurement), so the acknowledgment is cheap.

**AhaKV is *not* critical overlap.** AhaKV is conceptually parallel to our Layer 1 — both replace the raw accumulated-attention score with a cheap, training-free correction — but the corrections operate on different axes:

| Axis | AhaKV | EndurKV Layer 1 |
|---|---|---|
| Per-token signal | $\|V_i\|^2$ value-norm prior | (none) |
| Softmax temperature | adaptive $\lambda = \sqrt{2 \log(i/k)/d}$ | model native |
| Per-head budget | uniform $k$ per layer | per-head $K_h = K_{\text{nom}} \cdot \mu(m_h)$ |
| Anchor / recent | 32-token *recent* window | 32-token *prompt anchor* (Layer 2) + sink + recent |
| Theory | bias-of-expectation argument (not a bound) | none (we defer to Ada-KV) |

AhaKV is *complementary* to our stack and should be cited as such. It does not displace any EndurKV contribution and we do not displace AhaKV — its SG-softmax and value-norm prior could be dropped into our Layer 1 scoring step as future work.

---

## 2. Specific edits to PAPER_DRAFT_MOBISYS_2028.tex

### Edit A — §2.1 line 141 (Ada-KV paragraph): expand to name the three pieces and the lineage

**Current text (line 141):**

> \textbf{Ada-KV}~\cite{feng2025adakv} (NeurIPS 2025, concurrent with this work) is the closest prior art for the per-head allocation idea used in our Layer~1. Feng et al.\ define an $L_1$ eviction-loss metric, prove that Top-$K$ eviction minimizes a tight upper bound for any given allocation, and prove that an adaptive per-head allocation strategy (Algorithm~1 in their paper) further minimizes that bound. Their evaluation is quality-only on GPUs (Ruler, LongBench). Our Layer~1 uses the same per-head idea with a closed-form heuristic ($\mu(\cdot)$) instead of their attention-frequency allocator, trading provable optimality for $O(H)$ per-layer cost. We integrate it into a mobile systems stack (Q8 K, state-swap, watchdog), measure end-to-end on a real phone, and adopt their fixed-budget grid $\{128, 256, 512, 1024, 2048\}$ in our evaluation (\S\ref{sec:eval}). Their theory is the formal foundation for our Layer~1; our contribution is the integration, not the allocation rule.

**Replacement text:**

> \textbf{Ada-KV and its successors}~\cite{feng2025adakv} (NeurIPS 2025, concurrent with this work) is the closest prior art for the per-head allocation idea used in our Layer~1. Feng et al.\ contribute three pieces: (i) an $L_1$ eviction-loss bound $\varepsilon = 2hC - 2C\sum_{i,j}\mathbb{I}_i^j A_i^j$ with $C = \max_i \|V_i W_i^O\|_\infty$, with Theorems~3.1--3.3 proving that per-head Top-$K$ achieves the minimal $\varepsilon^\ast$ under any allocation and that their Algorithm~1 allocation achieves $\varepsilon^{\ast\ast} = \min_{\{B_i\}} \varepsilon^\ast$; (ii) the allocation rule itself, which concatenates post-softmax attention across heads of a layer, takes the global Top-$B$, and reads off head budgets $B_i = f_i$ as the per-head selection count; and (iii) a safeguard $B_i^\ast = \alpha B_i + (1-\alpha) B/H$ with $\alpha = 0.2$ that linearly mixes the adaptive allocation toward uniform $B/H$ to prevent very sparse heads from being starved (Algorithm~2 line~8). Ranking uses SnapKV's window-32 observation and a max-pool kernel of size~7. On the systems side they pair a flattened cache layout with FlashAttention-2's \texttt{flash\_attn\_varlen\_func} variable-length kernel so heterogeneous head budgets run at parity with the uniform SnapKV/PyramidKV baselines on GPU; on our mobile CPU+NPU stack that variable-length kernel is not available, which is one reason we trade Ada-KV's allocation rule for a closed-form $\mu(m_h)$ that needs only one reduction per head. Evaluation is quality-only on GPUs across the 13 RULER and 16 LongBench tasks. The same group has since released CriticalKV~\cite{feng2025criticalkv} (ICML~2026) and DefensiveKV / Layer-DefensiveKV~\cite{feng2025defensivekv} (ICLR~2026); Layer-DefensiveKV explicitly re-uses Ada-KV-style layer-wise adaptive allocation, cementing this idea as Feng et al.'s line of work. Our Layer~1 uses the same per-head idea with a closed-form heuristic ($\mu(\cdot)$) instead of the global Top-$B$ + $\alpha$-blend pair, trading the provable $\varepsilon^{\ast\ast}$ for $O(H)$ per-layer cost. Our \texttt{policy\_adakv} reference implementation faithfully replicates Ada-SnapKV (kernel-7 max-pool, $\alpha{=}0.2$ safeguard) and is the apples-to-apples baseline for Table~\ref{tab:three-way}. None of Ada-KV, CriticalKV, or DefensiveKV reports thermal, power, memory-residency, or kernel-throttle measurements on a mobile device.

### Edit B — §2.1 add new paragraph immediately after the Ada-KV paragraph (insert before line 143 "None of the GPU-evaluated works above..." sentence)

**New paragraph to insert:**

> \textbf{AhaKV}~\cite{gu2025ahakv} identifies that the accumulated-attention score used by H2O, SnapKV, and TOVA is mathematically biased in expectation: under softmax normalization, the expected accumulated score of a token monotonically decreases with its position, so retained tokens cluster toward the start of the prompt regardless of semantic relevance. They propose two corrections layered on a standard eviction pipeline: a Step-Gain softmax that adaptively sharpens the score with temperature $\lambda = \sqrt{2\log(i/k)/d}$ tied to the compression ratio, and a Value-Prior Enhancement that multiplies the score by the normalized squared $L_2$ norm of the value vector, $\bar\gamma_i = \|V_i\|^2 / \max_j \|V_j\|^2$. Evaluation is GPU-only on LongBench plus four short-text reasoning suites across LLaMA-2/3, Qwen2, and Gemma at a single budget per model with a fixed 32-token recent window; no comparison against Ada-KV or any per-head allocator is reported. AhaKV is conceptually parallel to our Layer~1 in that both replace the raw accumulated-attention score with a cheap, training-free correction, but the corrections operate on disjoint axes: AhaKV corrects the per-token \emph{score} (via $\|V_i\|^2$ and SG-softmax) under a uniform per-layer budget, while our Layer~1 corrects the per-head \emph{budget} via $\mu(m_h)$. AhaKV's score corrections are compatible with our per-head allocator and could be substituted into our Layer~1 scoring step at the cost of one extra reduction over $V$ per token; we leave that combination to future work and note that it does not displace any of our systems contributions (state-swap, Q8 K, watchdog, on-phone measurement).

### Edit C — §2.1 line 143 (the existing "None of the GPU-evaluated works above..." sentence)

**Current:**

> None of the GPU-evaluated works above (including Ada-KV) report any thermal, power, or memory residency measurements on a mobile device.

**Replacement:**

> None of the GPU-evaluated works above (including Ada-KV, CriticalKV, DefensiveKV, and AhaKV) reports any thermal, power, kernel-throttle, or memory-residency measurement on a mobile device.

### Edit D — §3.1 line 228 (Difference from Ada-KV paragraph): name the safeguard explicitly

**Current text (line 228):**

> \paragraph{Difference from Ada-KV.} Ada-KV's Algorithm~1 takes the top-$B$ attention weights across the concatenated heads and reads off head budgets as the frequency-of-selection $f_i$. Our heuristic $\mu(m_i)$ is a closed-form approximation: it requires only one reduction per head (the max) and one piecewise-linear mapping, which is cheaper at very low cache sizes and well-suited to the on-device setting where we cannot afford the global top-$B$ scan over $H \cdot n$ attention weights per layer. We have not proven that our heuristic minimizes the same bound as Ada-KV's allocation; we view it as a low-overhead approximation justified empirically (\S\ref{sec:eval}).

**Replacement text:**

> \paragraph{Difference from Ada-KV.} Ada-KV's allocation has three pieces: a SnapKV pre-pass (window~32, max-pool kernel~7), a global top-$B$ over the flattened $H \cdot n$ post-softmax attention with head budgets $B_i = f_i$ from the selection count (Algorithm~1), and a convex safeguard $B_i^\ast = \alpha B_i + (1-\alpha) B/H$ with $\alpha = 0.2$ that interpolates the adaptive allocation toward uniform $B/H$ (Algorithm~2 line~8). Our heuristic replaces the global top-$B$ + $\alpha$-blend pair with a single closed-form ramp $K_h = \mathrm{round}(K_{\text{nominal}} \cdot \mu(m_h))$ that requires only one reduction per head (the max) and one piecewise-linear mapping --- $O(H)$ per layer versus $O(H n \log(Hn))$ for the top-$B$ pass --- and is the only path available when the FlashAttention-2 variable-length batched kernel that Ada-KV relies on for parity decoding latency is not available, as is the case on our mobile CPU+NPU stack. The $[0.7, 1.3]$ clamp range in $\mu(\cdot)$ plays a role analogous to (but is not equivalent to) the $\alpha$-safeguard: it prevents any head from being starved of budget and any head from running away with the layer budget. We have not proven that our heuristic achieves Ada-KV's $\varepsilon^{\ast\ast}$ minimum; we view it as a low-overhead approximation justified empirically (\S\ref{sec:eval}). The faithful Ada-SnapKV replication (kernel-7 max-pool, $\alpha{=}0.2$ safeguard) lives in our codebase as \texttt{policy\_adakv} and is the baseline used in Table~\ref{tab:three-way}.

### Edit E — Table 1 (Wave-11 PPL, line 498): expand the adakv row + add an ahakv row

**Current (line 498):**

```latex
adakv~\cite{feng2025adakv} & 512 & \nodata & \nodata & \nodata & \nodata & \nodata & \nodata & \nodata & \nodata & \nodata \\
```

**Replacement (two rows):**

```latex
adakv~\cite{feng2025adakv} & 512 & \nodata & \nodata & \nodata & \nodata & \nodata & \nodata & \nodata & \nodata & \nodata \\
ahakv~\cite{gu2025ahakv}   & 512 & \nodata & \nodata & \nodata & \nodata & \nodata & \nodata & \nodata & \nodata & \nodata \\
```

**Caption addendum (append to the existing Table~\ref{tab:wave11} caption):**

> The \texttt{adakv} row will be filled in from our \texttt{policy\_adakv} run (SnapKV pre-pass with kernel-7 max-pool, global top-$B$ allocation, $\alpha{=}0.2$ safeguard, FA-off throughout per the GPU baseline). The \texttt{ahakv} row will be filled from a port of \texttt{policy\_snapkv} that substitutes Step-Gain softmax ($\lambda = \sqrt{2 \log(i/k)/d}$) and Value-Prior Enhancement ($\bar\gamma_i = \|V_i\|^2 / \max_j \|V_j\|^2$) into the eviction-score step, keeping the layer budget uniform; FA-off throughout (the value-norm read requires $V$ visibility).

### Edit F — §1 Contributions line 118 (already disclaims theory): tighten the disclaimer

**Current (last clause):**

> ...or (iii) novelty of state-swap APIs themselves (they are available in \texttt{llama.cpp}; the FA-off$\to$FA-on use of them is our pattern).

**Add a fourth disclaimer clause:**

> ...or (iii) novelty of state-swap APIs themselves (they are available in \texttt{llama.cpp}; the FA-off$\to$FA-on use of them is our pattern), or (iv) novelty of correcting the eviction-score signal --- AhaKV~\cite{gu2025ahakv} contributes orthogonal score corrections (Step-Gain softmax and value-norm prior) that operate on a uniform per-layer budget and are compatible with, but not equivalent to, our per-head budget rule.

### Edit G — Bibliography (after line 740, the existing `feng2025adakv` entry): add three new bibitem entries

```latex
\bibitem{feng2025criticalkv}
Y. Feng, J. Lv, Y. Cao, X. Xie, S. K. Zhou,
``CriticalKV: A Perturbation Perspective on KV Cache Eviction,''
in \emph{Proc. ICML}, 2026.

\bibitem{feng2025defensivekv}
Y. Feng, J. Lv, Y. Cao, X. Xie, S. K. Zhou,
``DefensiveKV: Worst-Case Risk Control for KV Cache Eviction (with Layer-DefensiveKV),''
in \emph{Proc. ICLR}, 2026.

\bibitem{gu2025ahakv}
Y. Gu et al.,
``AhaKV: Adaptive Holistic Attention-Driven KV Cache Eviction for Efficient Inference of Large Language Models,''
arXiv preprint arXiv:2506.03762, 2025.
```

### Edit H — Bibliography: add KeepKV bibitem (currently cited at line 683 but missing)

> Note: `tian2026keepkv` is cited at line 683 of the .tex but no bibitem exists. This is independent of the Ada-KV/AhaKV audit but should be fixed in the same pass to avoid a dangling cite. Suggested entry (placeholder; verify against the actual paper):

```latex
\bibitem{tian2026keepkv}
J. Tian et al.,
``KeepKV: Cumulative Regret Analysis of KV Cache Eviction,''
arXiv preprint, 2026.
```

---

## 3. Honest verdict: do AhaKV / CriticalKV / DefensiveKV preempt anything we claim?

**No.** Concretely:

1. **AhaKV does not preempt our per-head budget allocation.** AhaKV uses a uniform per-layer budget. Its corrections (SG-softmax temperature + $\|V_i\|^2$ prior) are *complementary* to per-head allocation and could be combined with it. AhaKV also does not compare against Ada-KV or any per-head allocator, so the per-head allocation lane remains owned by the Feng et al. family.

2. **CriticalKV / DefensiveKV continue the Feng et al. line.** They extend Ada-KV's framework (perturbation worst-case in CriticalKV, worst-case risk control in DefensiveKV, layer-wise allocation in Layer-DefensiveKV). They reinforce — they do not displace — our position that "per-head/per-layer adaptive allocation" is Feng et al.'s family, which we already credit. We adopt their notation and disclaim the eviction-loss bound, so we lose no ground here.

3. **None of the four works reports any of our measured systems quantities.** On-device decode tps, peak DDR/CPU temperature, peak RSS, swap, mAh under disable-charging protocol, kernel-throttle headroom — all remain ours alone.

4. **The one cosmetic risk:** if a reviewer reads our current §3.1 line 228 and notices the $\alpha=0.2$ safeguard is unmentioned, they may charge us with under-acknowledgment. Edit D fixes this preemptively. The $\mu$ clamp into $[0.7, 1.3]$ is a *different* fallback (multiplicative clamp on individual heads, not a convex blend toward uniform), but it is not stronger and we do not need to claim it is — we just need to name both rules and let the reader see the trade.

5. **One small win we should make explicit:** the FlashAttention-2 `flash_attn_varlen_func` variable-length batched kernel is the *systems* mechanism that makes Ada-KV's heterogeneous head budgets free on GPU. That kernel is not available on our mobile CPU+NPU stack, which is a clean systems reason for the closed-form $\mu(\cdot)$ choice. Edit A names this explicitly.

---

## 4. Updated Table 1 (Wave-11 PPL) — add AhaKV alongside AdaKV?

**Yes — add as a placeholder row now, fill from a `policy_ahakv` port later.** Rationale:

- We are already showing `\nodata` for `adakv` — adding `ahakv` is symmetric and tells reviewers we are aware of both prior-art axes (per-head budget *and* score correction).
- The implementation cost for `policy_ahakv` is modest: SG-softmax (one scalar multiply pre-softmax in the prefill-window scoring) + Value-Prior multiplier (one reduction over $V$ per token), on top of `policy_snapkv`. No state-swap is needed because the budget is uniform per layer.
- Adding the row commits us to running it; if Wave-12 cannot fit a `policy_ahakv` cell, downgrade to a single sentence in the caption noting that AhaKV is in the same uniform-budget family as SnapKV and the SnapKV row is a reasonable upper bound on AhaKV's performance on this hardware. Either way, the citation in §2.1 plus the placeholder row is enough to discharge the "did you know about AhaKV?" reviewer question.

---

## 5. Edit summary (TL;DR for the editor)

| # | Where | What | Risk if skipped |
|---|---|---|---|
| A | §2.1 line 141 | Replace Ada-KV paragraph with expanded version naming the 3 pieces + 2026 follow-ups | Reviewer: "you under-acknowledge Ada-KV" |
| B | §2.1 insert before line 143 | Add new AhaKV paragraph | Reviewer: "do you know about AhaKV?" |
| C | §2.1 line 143 | Expand sentinel sentence to name CriticalKV, DefensiveKV, AhaKV | Cheap consistency |
| D | §3.1 line 228 | Expand "Difference from Ada-KV" to name $\alpha=0.2$ safeguard and the variable-length kernel | Reviewer: "you skipped Algorithm 2 line 8" |
| E | Table 1 line 498 | Add `ahakv` row alongside `adakv` (both `\nodata` placeholders) + caption addendum | Symmetric framing |
| F | §1 line 118 | Add 4th disclaimer clause about score corrections | Cheap honesty |
| G | Bibliography after line 740 | Add `feng2025criticalkv`, `feng2025defensivekv`, `gu2025ahakv` bibitems | Required by edits A, B, F |
| H | Bibliography | Add `tian2026keepkv` bibitem (dangling cite at line 683) | Compilation warning |

**Net assessment:** all eight edits are *additive and honest*. None retracts a measured result. None weakens a contribution claim. The paper is *stronger* after these edits because reviewers will see we have audited the 2025–2026 KV-cache literature and located our contribution precisely on the systems axis where the prior art is silent.
