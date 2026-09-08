# Chapter X.5 - Synthesis: Picking the Right Eviction Policy for On-Device Decode

## X.5.1 Methodology: A Weighted Efficiency Score

To collapse four competing axes -- throughput, thermal headroom, endurance, and output quality -- into a single dissertation-defensible ranking, we define:

```
efficiency = 0.35 * (tps / max_tps)
           + 0.25 * thermal_score * (0.7 if throttle else 1.0)
           + 0.20 * memory_endurance_score
           + 0.20 * accuracy_score
```

where `thermal_score = 0.5*(1 - DDR/105) + 0.5*(1 - CPU/105)`, `memory_endurance_score = 0.5*(1 - swap_MB/1000) + 0.5*(1 - RSS_GB/8)`, and `accuracy_score = 1/max(1, PPL)`. The weights are justified by user-visibility hierarchy: throughput is the primary KPI a human sees (35%); thermal headroom is the dominant constraint that, once exceeded, collapses throughput by 25-35% via kernel-side mitigation (25%); endurance (UFS swap is the chief wear-out vector on a handset) and on-device PPL (the only quality proxy we can collect in-situ) take 20% each. The score is restricted to the **decode-dominated regime (Wave-4/6, long-decode-Phi3)** because that is the regime in which an eviction *mechanism* is actually exercised every step -- comparing against narrativeqa Wave-3 cells would conflate prefill-cost amortization with decode-cache management.

## X.5.2 Per-Objective Winners

* **Throughput**: v1_FA bounded K=512 at 5.77 tok/s mean (1.15x vanilla and 25% above the next-best non-vanilla policy).
* **Thermal**: v1 (FA-off) K=512 -- the only Wave-4 policy that did not trigger a throttle event, with peak DDR 54.4 C and peak CPU 57.9 C.
* **Memory / Endurance**: v1 (FA-off) K=512 and v1_FA frozen K=512 tied at 0 MB swap-out, but v1_FA bounded's 28 MB swap is also far below vanilla-Phi3-narrativeqa's 220 MB and Phi3-narrativeqa-v1's 172 MB.
* **Accuracy (PPL)**: v1_FA frozen K=512 at PPL 2.34, narrowly beating v1_FA bounded at 3.20 -- but at the cost of full vanilla-like thermal collapse.
* **Overall**: v1_FA bounded K=512 with efficiency = 0.625.

## X.5.3 Headline Recommendation

**v1_FA bounded (K=512) is the on-device winner**: 5.77 mean tok/s (1.15x vanilla), 4.4% decode decay over nine iterations (versus vanilla's 32.5%), zero throughput collapse despite an observed throttle event, and the highest efficiency score (0.625) of any policy in the decode-dominated regime.

## X.5.4 Discussion

The result is consistent across the four sub-scores. v1_FA bounded keeps FA-off only at prefill -- a one-time cost amortized over the entire decode -- and then runs FA-on with a 512-token sliding cache during decode, which is exactly the path the Snapdragon 8 Elite Gen 5's DDR controller is happiest with. The throttle event still fires (peak DDR hits 65.6 C, peak CPU 69.1 C), but unlike vanilla and v1_FA frozen, the bounded cache means the post-throttle 0.88 GHz floor is sufficient to keep up with the steady stream of 512-key dot products, so decay stays at 4.4% instead of cratering to 28-32%.

The runner-ups each fail in an instructive way. **Vanilla** is fast at the start (6.05 tok/s iter-1) but the unbounded cache pulls 2311 KV cells into DDR by iter-9 and the resulting bandwidth pressure forces the kernel to cut frequency, costing 32.5% throughput. **v1_FA frozen** inherits exactly this failure mode in long-decode because its eviction only ran at prefill -- the paper's "bounded fix" was added precisely to close this hole. **v1 (FA-off)** is the thermal champion (54.4 C, no throttle, 0 swap, the cleanest possible decode footprint) but pays for it by running FA-off through every decode step, halving throughput to 2.66 tok/s; its PPL also drifts to 4.22 because the autoregressive-on-own-generation evaluator amplifies the per-step entropy that FA-off plus aggressive recency eviction introduces.

The published baselines were designed for a fundamentally different deployment regime -- batched continuous serving on A100/T4/H100 GPUs -- and we cite them fairly on those terms. **TOVA** (EMNLP 2024) reports 4.8x throughput, but that gain comes from raising the maximum batch from 8 to 70 on a V100, which is irrelevant for batch-1 mobile decoding. **H2O** (NeurIPS 2023) reports 29x vs DeepSpeed Zero-Inference on A100; the comparison policy on a phone is not Zero-Inference but vanilla llama.cpp, against which the FA-off penalty is decisive. **SnapKV** (NeurIPS 2024) is the most promising literature candidate to port next because its selection runs only at end-of-prefill, leaving decode FA-compatible. **StreamingLLM** (ICLR 2024) is the only published policy that needs no softmax materialization at all, making it the natural FA-on candidate for a future Wave; it was not measured here because our eviction_bench tooling targets attention-score-driven policies.

**Limitations.** Our on-device PPL is autoregressive-on-own-generation -- closer to a self-consistency probe than a true accuracy benchmark -- and we did not run PG-19, LongBench, or QASPER on the phone. The thermal regime is OnePlus 15 / Snapdragon 8 Elite Gen 5 specific; the 1.63 GHz -> 0.88 GHz throttle floor is a kernel policy, not a hardware floor, and future Android builds may move it. Finally, the throttle event in v1_FA bounded fired but did not collapse throughput, which we attribute to the bounded cache -- but a longer-decode wave (4096 tokens) would be needed to confirm that the policy remains stable under deeper thermal soak.
