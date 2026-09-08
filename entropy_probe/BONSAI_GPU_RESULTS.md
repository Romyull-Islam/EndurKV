# ⚠️ RETRACTION (2026-07-26): ALL PHONE-GPU BONSAI NUMBERS ARE INVALID

**Prism's Vulkan Q1_0 kernels are numerically BROKEN on Adreno 840.** The model loads
(990 MiB on Vulkan0), runs fast, and emits tokens — but the logits are non-finite.

Evidence (same binary, same prompt, only `-ngl` differs):
| backend | generated text |
|---|---|
| GPU `-ngl 36` | `" fine!!!!!!!!!!!!!!!!!!!!!!!"` — degenerate |
| CPU `-ngl 0`  | `"2006 's Best Actor in a Play award at the National Theatre Festival . In 2006"` |

Teacher-forced PPL returned **`mean_nll=nan  ppl=nan` for BOTH vanilla and μKV**, which was
the tell. With greedy decoding, NaN logits still emit tokens at full speed, so throughput
looked healthy while the computation was wrong.

**Therefore RETRACTED** (everything below that concerns the phone GPU):
- "μKV +29–38% decode on phone GPU" (the K=64/128/256 matrix)
- the `n_gpu_layers` sweep and its "**decode 8.8× on GPU**" conclusion
- "Bonsai-8B runs on the Adreno GPU"
- the "~730-token Adreno ceiling" (those runs produced garbage before crashing)

**CUDA is NOT affected — verified.** On Jetson, `-ngl 99` and `-ngl 0` produce byte-identical,
coherent text, so the CUDA Q1_0 port is numerically correct and its numbers stand
(prefill 346.7→40.3 s, decode 4.2→9.9 tps).

**Root-cause split:** the same Prism Q1_0 source is correct on CUDA and broken on Vulkan/Adreno.
Fixing it would mean debugging the GLSL shader (`dequant_q1_0.comp` + the mul_mm path) against
the Adreno driver — real work, not attempted here.

**Process lesson (now mandatory):** validating mechanism (weights on device, no DeviceLost,
plausible tps) does NOT validate correctness. Every new backend/kernel must pass an
OUTPUT-VALIDITY check — generate N tokens and diff against a known-good CPU reference —
BEFORE any performance number is recorded.

---

# Bonsai-8B (PrismML 1-bit) on GPU — measured results (2026-07-26)

Supersedes `BONSAI8B_GPU_BLOCKER.md`, whose central claim ("Q1_0 has no GPU kernel on
any backend") was **wrong**. PrismML-Eng/llama.cpp implements Q1_0 for CUDA, **Vulkan**,
Metal and SYCL. Bonsai was never "blocked" — it always ran, but on our fork the 1-bit
weights fell back to CPU because our llama.cpp lineage lacks those kernels.

## Jetson Orin NX (CUDA) — port the kernels, GPU wins big
Ported Prism's CUDA Q1_0 (dequantize.cuh, vecdotq.cuh, common.cuh, mmvq.cu, convert.cu,
ggml-cuda.cu). Same 12K prompt, 64-token decode, vanilla policy:

| | before (weights on CPU) | after (Q1_0 CUDA kernels) |
|---|---:|---:|
| weights on GPU | 1.18 MiB | **1015.99 MiB** |
| prefill | 346.7 s | **40.3 s (8.6× faster)** |
| decode | 4.2 tps | **9.9 tps (2.4× faster)** |

## Phone, Adreno 840 (Vulkan) — GPU/CPU split, and the split is NOT what I assumed
Prism's Vulkan build runs on Adreno (control: Llama-1B Q4_K_M ngl=99 → pp 59.6 / tg 44.4).
Bonsai-8B `n_gpu_layers` sweep (llama-bench, -p 64 -n 16, -b 512 -ub 64, cooled between cells):

| ngl | pp64 (prefill) | tg16 (decode) |
|---:|---:|---:|
| 0 (CPU only) | **11.70** | 0.57 |
| 8 | 10.63 | 0.64 |
| 16 | 9.34 | 0.84 |
| 24 | 8.44 | 1.27 |
| 32 | 7.63 | 2.49 |
| **36 (all layers)** | 7.30 | **5.02** |

**Two opposite monotonic trends:**
- **Prefill: CPU is 1.6× faster** than full GPU (11.70 vs 7.30). Prism's ARM **i8mm/NEON
  1-bit repack kernels** (the `CPU_REPACK` buffer) beat the generic Vulkan Q1_0 shader.
- **Decode: GPU is 8.8× faster** (5.02 vs 0.57). Decode is bandwidth-bound GEMV — Adreno's
  strength. For a 4096-token generation: ~816 s on GPU vs ~7186 s on CPU.

`ngl=99` aborts with `vk::DeviceLostError`; `ngl=36` (every transformer layer) is fine —
so the fault is offloading the **output/embedding** tensor beyond the 36 layers, not capacity.

## The resulting hybrid design (measured, not assumed)
1. **Prefill on CPU** — i8mm 1-bit kernels, 1.6× faster than the GPU.
2. **μKV evicts at the prefill→decode boundary** — ONE backend crossing, no per-token
   ping-pong (the 291-graph-split pathology that made naive offload slow).
3. **Decode on GPU** — 8.8× faster, over the small post-eviction cache.

μKV's one-shot eviction already sits exactly at that boundary, so the split is natural.

## Platform-dependent conclusion (do not generalize either way)
| platform | best engine for 1-bit weights | why |
|---|---|---|
| Jetson Orin (CUDA) | **GPU everywhere** | CUDA cores massively outclass the Orin CPU |
| Phone (Adreno) | **CPU prefill + GPU decode** | i8mm 1-bit kernels beat the Vulkan shader on compute; Adreno wins on bandwidth |

NPU remains out: QNN has no 1-bit path and llama.cpp's Hexagon backend falls back to CPU
for quantized ops (arxiv 2605.27435 also measures NPU prefill up to 1.6× *slower* than CPU).

## Provenance / gotchas for reproduction
- Prism Android build needs: explicit `Vulkan_INCLUDE_DIR`/`Vulkan_LIBRARY`/`Vulkan_GLSLC_EXECUTABLE`
  from the NDK, **SPIRV-Headers** installed (`-DCMAKE_PREFIX_PATH`), and `-DCMAKE_CXX_FLAGS=-I<spirv>/include`.
- Use **`llama-bench`**, not `llama-cli`: the CLI's loading spinner busy-spins on a non-TTY
  and wrote 8.5 GB of stdout in 48 min, which looks exactly like a hang.
- Push **all** `.so` files incl. `libllama-bench-impl.so`; stage via `/data/local/tmp` then
  `su cp` (a root-owned target dir rejects direct `adb push`).

## FINAL (2026-07-26): the Adreno hard limit for Bonsai-8B — 512-token prompts

μKV was ported INTO PrismML's fork (6 files, ~18 lines: the `kq_evict` side node in
`build_attn_mha`, the two C APIs, cell-index eviction) so μKV could use Prism's Vulkan
Q1_0 kernels. **Validated bit-exact** on the phone GPU: Llama-1B f16 reproduced
`anchor=723 recent=297, retained_kv=22.60 MiB` — identical to our frozen reference — so
numbers from that build are comparable. Our own fork was never modified (zero regression risk).

**Prompt-length boundary sweep** (Bonsai-8B Q1_0, ngl=36, b512/ub64, llama-bench):
| prompt tokens | prefill tps | result |
|---:|---:|---|
| 128 | 7.32 | OK |
| **512** | **7.29** | **OK — last working size** |
| 1024 | — | **vk::DeviceLostError** |
| 2048 / 4096 | — | DeviceLostError |

The 10 074-token workload aborts for BOTH vanilla and μKV, and `-b 128 -ub 32` does NOT
help — so it is not batch granularity, not capacity (weights load fine: 990 MiB on Vulkan0),
and not μKV. It is the Adreno driver's watchdog on sustained GPU work during long prefill.

### Deployment conclusion (phone, 8B 1-bit)
| context | best engine |
|---|---|
| ≤512-token prompts | **GPU** — decode 5.02 vs 0.57 tps (8.8×) |
| long context (4K–16K) | **CPU** — the GPU cannot complete the prefill at all |

So for the paper's long-context workload the shipped **CPU + μKV** result (4.5× decode,
85.6→33.2 min) remains the correct 8B phone deployment — now backed by a measured GPU
limit rather than an assumption. The Jetson keeps the full GPU win (8.6× prefill, 2.4× decode),
because CUDA has no equivalent watchdog.
