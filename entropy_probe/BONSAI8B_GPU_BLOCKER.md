# CORRECTION (2026-07-26): THIS DOCUMENT'S CENTRAL CLAIM WAS WRONG

I claimed "Q1_0 has no GPU kernel on any backend." **False.** PrismML-Eng/llama.cpp
(vendored at `workspace/prismml/llama.cpp`) implements Q1_0 for **CUDA (13 files),
VULKAN (10 files), Metal (5), SYCL (3)** — including `vulkan-shaders/dequant_q1_0.comp`.
Vulkan is exactly our phone backend, so **Bonsai-8B CAN run on the Adreno GPU**; our
EndurKV fork simply never had those kernels (different llama.cpp lineage).

The real situation: Bonsai always RAN (it was never blocked) but its 1-bit weights fell
back to CPU on our builds, so prefill was CPU-bound (346 s Jetson / phone likewise).
Fix = port Prism's kernels. CUDA port done 2026-07-26; Vulkan port is the phone path.
Everything below this line is the older, partly-incorrect analysis, kept for history.

---

# Why Bonsai-8B does not run on the Adreno GPU (and how it *could*)

**Date:** 2026-07-24. **TL;DR:** the blocker is the **1-bit weight format, not μKV.**

## Root cause (definitive, from the load log)
Bonsai-8B is `Q1_0` (1-bit). llama.cpp's **Vulkan backend has no Q1_0 kernel**, so
the weights cannot live in a Vulkan buffer:
```
llama_model_loader: - type q1_0:  254 tensors          (file type = Q1_0, arch = qwen3, 36 layers)
done_getting_tensors: tensor 'token_embd.weight' (q1_0) (and 253 others)
  cannot be used with preferred buffer type Vulkan_Host, using CPU instead
sched_reserve: graph splits = 291                       (CPU<->GPU round-trip per 1-bit matmul)
```
Even with `--n-gpu-layers 99`, all 254 one-bit tensors are pinned to **CPU**; only the
non-quantized ops run on Vulkan. Every token therefore round-trips CPU↔GPU 291 times →
slow decode + Adreno `vk::DeviceLostError`. The Prism 1-bit matmul kernels
(i8mm/repack, see [[bonsai-q1-kernel-port]]) are **ARM-CPU only**; there is no GLSL/SPIR-V
equivalent. The "fused Gated Delta Net enabled" line is a generic capability probe, NOT
model-specific (arch is plain `qwen3`) — a red herring.

**This is orthogonal to μKV.** μKV runs correctly on GPU on the 1B (Q4_K_M, which *does*
have Vulkan kernels): FA-on prefill parity + logical eviction, validated.

## Options to run 8B, ranked
1. **CPU (WORKS — this is the paper's 8B result).** 1-bit CPU kernels exist; μKV gives
   4.5× decode tps, 111 vs 1417 MiB retained KV, −60% energy. Done, in the draft.
2. **Write a Vulkan Q1_0 compute shader** (port Prism's 1-bit matmul to SPIR-V). The only
   true path to full-GPU 8B. Real systems work; **out of scope for HotMobile** — this is
   exactly the "Bonsai GPU counterpart left to future work" the draft already scopes.
3. **Run 8B in a Vulkan-supported quant (Q4_K_M/f16) on GPU.** Loses the 1-bit memory win
   and is no longer *Bonsai*; the 1B GPU row already demonstrates μKV-on-GPU with a
   Vulkan-native quant, so this adds little.
4. **NPU (Hexagon) — not viable near-term:**
   - PowerInfer-2's QNN pipeline (`powerinfer2/qnn`, QAIRT 2.42): per-model ONNX→HTP
     compilation + calibration; model-specific, enormous, and QNN has **no 1-bit path**.
     (Also: do not modify that directory.)
   - llama.cpp `ggml-hexagon`/QNN backend (PR #12326 / #12063): experimental, unmaintained
     since Jul-2025, upstream-unmerged; **quantized matmul falls back to CPU**, so Q1_0
     would execute on CPU anyway. No μKV integration.

## Conclusion for the paper
8B-on-GPU full offload is blocked by a **missing Vulkan 1-bit kernel** (backend gap), not
by μKV or the KV cache. The honest, already-correct framing: 8B on **CPU** (strong μKV
result, shipped), μKV-on-**GPU** demonstrated on the **1B** (Vulkan-native quant), and
full-GPU 1-bit 8B = **future work** (needs a Vulkan Q1_0 kernel). The NPU is not a
near-term substitute.
