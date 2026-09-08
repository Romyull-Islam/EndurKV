# Phone (Adreno 840, OnePlus 15) — Phi-3-mini-128k Q2_K @16K, GPU full offload
Measured 2026-07-25. Same 12K WikiText prompt (9737→11158 Phi-3 tokens) + decode, ctx 16384,
n-ubatch 64 (Adreno TDR cap), threads 4, native DVFS from cold. μKV = frozen config.

| KV type | policy | KV alloc | KV live | prefill | decode tps |
|---|---|---:|---:|---:|---:|
| f16 | vanilla | **6144 MiB (allocated OK)** | 4184 MiB | 583.3 s | 2.0 |
| q8_0 | vanilla | 3264 MiB | 2223 MiB | 730.8 s | 7.2 |
| q8_0 | **μKV** | 3264 MiB | **193 MiB** | 751.6 s | **11.8 (+64%)** |

## Findings
1. **The Adreno has NO ~4 GiB single-alloc cap** (unlike Jetson Tegra): the 6144 MiB f16 KV
   buffer allocated and ran. Corrects the earlier prediction that the phone would fail like
   Tegra. So Phi-3 runs on the phone GPU at f16 OR q8 KV.
2. **μKV's largest decode win measured anywhere: +64%** (11.8 vs 7.2 tps at q8), from cutting
   live KV 11.5× (2223→193 MiB). Phi-3's no-GQA fat KV makes it the most bandwidth-bound →
   biggest μKV benefit — same pattern as Jetson, amplified.
3. Prefill is 580–750 s: Adreno TDR forces n-ubatch=64, and Phi-3 (no GQA, 3.8B) is
   compute-heavy. It RUNS on GPU (the question asked), but is slow; a TDR-safe larger ubatch
   would help and is a phone-tuning item, not a μKV issue.
4. Quality: Phi-3 **Q2_K is quantization-destroyed** (PPL ≥272 even short-context) — these
   phone numbers are a speed/memory/allocation demonstration only. The quality-valid Phi-3
   is **Q4_K_M** (μKV PPL 3.94 on Jetson).
