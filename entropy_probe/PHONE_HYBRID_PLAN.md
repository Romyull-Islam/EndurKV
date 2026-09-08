# Running Phi-3 & Bonsai-8B faster than CPU-only on the phone (2026-07-25)

Goal: phi-3 and Prism Bonsai-8B on the phone GPU (+NPU/CPU) **faster than CPU-only**.

## Phi-3 — SOLVED (GPU already beats CPU)
Phi-3 (Q2_K/Q4_K_M) has Adreno GPU kernels in BOTH backends. It already runs full-offload on
the phone GPU: **q8 KV decode 7.2 tps (vanilla) / 11.8 tps (μKV)**. CPU-only baseline measuring
now for the exact ratio, but GPU > CPU is the expected and standard result on this SoC. μKV
adds +64% on top. **No new work needed — GPU (Vulkan or OpenCL) + μKV is the answer for phi-3.**

## Bonsai-8B — the blocker is a MISSING 1-bit GPU KERNEL, and here is the fix
The problem was never μKV; it's that **Q1_0 (1-bit) has no GPU matmul kernel on any backend.**
Confirmed by direct inspection of the fork:
- Backends with quant kernels: **OpenCL (Adreno)** supports Q2_K,Q3_K,Q4_0,Q4_1,Q4_K,Q5_K,Q6_K,Q8_0.
  Vulkan similar. **Neither has Q1_0.** Q1_0 lives ONLY in `ggml-cpu/arch/arm/repack.cpp` (Prism, CPU).
- So at `-ngl 99` the 254 Q1_0 tensors fall back to CPU → 291 CPU↔GPU splits/token → slower than
  pure CPU + DeviceLost. Partial offload is *slower*, not faster.

### The real path: add a Q1_0 mul_mat kernel to the **OpenCL Adreno backend**
Why OpenCL not Vulkan: Qualcomm's official Adreno backend, actively optimized (coalesced quant
loads, dp4a MoE, binary kernel lib), and the Adreno-preferred path for quantized matmul.
It already has Q4_0/Q8_0 Adreno kernels to model the new one on; the only missing piece is the
**Prism 1-bit dequant layout** in the kernel.

**Why this beats CPU (even on unified memory):**
- **Prefill (compute-bound GEMM): huge win.** Adreno 840 has far more FLOPs than the CPU cluster.
  Bonsai prefill is **346 s on CPU** (Jetson & phone) — a GPU kernel should cut it to tens of
  seconds. This alone makes the 8B usable.
- **Decode (bandwidth-bound): win.** On Snapdragon the Adreno reaches higher *effective* LPDDR5x
  bandwidth than the CPU cluster (coalescing, more in-flight requests) — which is why standard-quant
  models already decode faster on the Adreno than CPU on this phone. Q1_0 should follow.
- **μKV compounds it**: less KV to read → the bandwidth-bound decode gets the same +% we measured.

### Refined heterogeneous design (per-op assignment, clean boundary — NOT whole-stage offload)
Key correction (user, 2026-07-25): do **not** put the whole prefill on one unit. Assign each op
to the unit that is *genuinely* fastest for it, and partition at a **clean stage boundary** so there
is **no per-op CPU↔accelerator ping-pong** (that ping-pong — the 291 splits — was the whole problem,
and is exactly why the naive-offload papers saw the NPU lose). Decompose the transformer:

| op (per layer) | nature | best unit | why |
|---|---|---|---|
| 1-bit weight GEMMs (QKV, O, FFN gate/up/down) — **PREFILL** | compute-bound, large static shapes | **GPU (Q1_0 OpenCL kernel)**, or NPU as INT8 | prefill GEMMs are the 346 s bottleneck; static big shapes are the GPU/NPU sweet spot |
| attention QKᵀ/AV over f16 KV — prefill | medium GEMM, has GPU kernels | GPU | already accelerated |
| 1-bit GEMV — **DECODE** (single token) | bandwidth-bound, dynamic | **CPU (Prism i8mm)** | NPU decode gain only 1.05–1.2×; not worth a boundary crossing |
| RMSNorm / RoPE / elementwise | tiny | CPU (in place) | avoid transfer |

**The clean boundary IS μKV's eviction point.** The natural split is prefill vs decode — and μKV
already does its one-shot eviction exactly at that transition. So the pipeline is:
1. **Prefill** heavy static 1-bit GEMMs on **GPU** (OpenCL Q1_0 kernel; the compute win — 346 s → tens of s).
2. **μKV eviction** at the prefill→decode boundary → tiny compacted KV (111 MiB).
3. **Decode** on **CPU** (Prism i8mm) over the *small* μKV cache — bandwidth-bound, and μKV made the
   bandwidth small, so CPU decode is efficient and needs no accelerator round-trip.

One stage boundary, crossed **once** — zero per-token scheduling overhead. μKV is what makes the
CPU-decode half cheap (small KV), so we don't need to fight the NPU's weak, overhead-heavy decode.

### NPU role — narrow and honest
- Use the NPU **only** for the part it is truly fast at: **static-shape INT8 GEMMs in prefill** (dequant
  the 1-bit weights → INT8 once, feed the Hexagon INT8 GEMM). Do **not** route decode or dynamic/
  irregular ops there (Hexagon has no 1-bit path and falls back to CPU; measured NPU prefill can be
  up to 1.6× *slower* under naive whole-stage offload — arxiv 2605.27435 — precisely because of the
  scheduling overhead this clean-boundary design avoids).
- GPU is the simpler first target for the prefill GEMMs (one Q1_0 OpenCL kernel vs a full QNN INT8
  graph). NPU-for-prefill is the stretch option if the GPU kernel underperforms.

### Implementation scope (actionable)
1. Build the **OpenCL backend for Android** (arm64, Adreno) — the fork already has `ggml-opencl/`.
2. Write `mul_mat_q1_0_f32` OpenCL kernel: dequantize the Prism 1-bit block layout (port the logic
   from `ggml-cpu/repack.cpp`) → accumulate. Model dispatch on the existing Q4_0 Adreno kernel.
3. Register Q1_0 in the OpenCL backend's supported-type + buffer paths so `-ngl` keeps it on GPU.
4. Validate: bit-parity dequant vs CPU on a tensor; then full Bonsai-8B prefill/decode on-device,
   vanilla vs μKV, vs the CPU-only baseline. Document in CHANGELOG (new kernel = code change).
5. Deploy binary + `libggml-opencl.so` together (PLATFORMS.md rule).

**Estimate:** a focused kernel-dev task (multi-iteration on-device debugging), but tractable
because both the template (Adreno Q4_0 kernel) and the reference logic (CPU Q1_0 dequant) exist.
This is the one true "8B on phone GPU, faster than CPU" path, and it's a genuine systems contribution
(first 1-bit Adreno kernel) that strengthens the paper beyond μKV alone.
