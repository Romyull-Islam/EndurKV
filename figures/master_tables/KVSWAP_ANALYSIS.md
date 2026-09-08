# KVSwap vs μKV — competitive analysis (2026-07-25)
Source: github.com/hwhwz23/KVSWAP-CODE — "KVSwap: Disk-aware KV Cache Offloading for
Long-Context On-device Inference" (MobiSys submission / artifact).

## What KVSwap is
A **swap-to-disk** KV system: keeps the **FULL** KV cache on disk (eMMC + NVMe), holds a
small `token_budget` in unified memory + a larger `reuse_budget` of predicted-reuse blocks,
uses compact block metadata to **predict + preload** entries, overlaps compute with
io_uring disk reads, and orchestrates read patterns to the storage device. Also does
group-wise KV quantization. **Lossless** (all KV retained → quality preserved).

- **Framework:** vLLM + PyTorch 2.7 (CUDA). io_uring (Liburing) for disk. → **CUDA-only.**
- **Throughput platform:** Jetson Orin **AGX** + NVMe SSD (≥256 GB) + eMMC (≥64 GB), JetPack 6.2.
- **Quality platform:** A100 80 GB server.
- **Baselines:** Quest, ShadowKV, InfiniGen, vLLM/PagedAttention, StreamingLLM.
- **Models:** LLM = Llama-3.1/3.2, Qwen2/2.5/3, Gemma, Phi-3. VLM = InternVL3 (+14B), LLaVA
  (1.6/OneVision/Next), **Qwen2.5-VL-3B/7B**.
- **Benchmarks:** RULER, LongBench, NIAH. Workload: 32K context, batch up to 16.

## KVSwap vs μKV — the core distinction
| axis | KVSwap | μKV (ours) |
|---|---|---|
| strategy | **swap** full KV to disk (+ block select + quant) | **evict** KV permanently |
| quality | lossless (all KV kept) | near-parity (PPL ±2%); loses evicted-verbatim recall |
| **disk** | **REQUIRES** NVMe SSD + eMMC | **none** — pure in-memory |
| framework | vLLM/PyTorch (**CUDA only**) | llama.cpp (CUDA **+ phone Adreno Vulkan**) |
| **phone** | **cannot run** (no vLLM/CUDA/SSD on a phone) | **runs on the phone** (OnePlus 15) |
| edge target | Jetson Orin **AGX** (32/64 GB) + SSD | **phone (16 GB, no SSD)** + Orin **NX** 16 GB |
| batch | up to 16 (server-like) | single-stream (true consumer mobile) |
| thermal/energy | not addressed | **core contribution** (watchdog, energy, 85→33 min) |

## μKV's defensible moat (honest)
1. **The phone.** KVSwap's "mobile" is a dev board with an NVMe SSD; it is vLLM/CUDA and
   **physically cannot run on a phone** (same wall as vLLM — no Android/Adreno backend, no
   fast SSD to swap 10+ GB of KV to). μKV runs on the actual Adreno GPU. This is the moat.
2. **No disk.** Phones have UFS flash — limited write bandwidth and **write-endurance**
   concerns; repeatedly swapping GBs of KV per inference wears the flash and **burns energy**
   (I/O power) — a battery/thermal negative. μKV needs zero disk.
3. **Thermal + energy under sustained load** — the mobile-sustained-inference story KVSwap
   omits; swap-to-disk is itself an energy cost, whereas eviction *lowers* DDR traffic/heat.
4. **Composable, not just competing.** Evict the clearly-dead KV first (μKV, free), then swap
   only the survivors (KVSwap) → smaller swap volume, less I/O. "μKV complements KVSwap."

## KVSwap's advantages over μKV (be honest)
- **Lossless** — no verbatim-recall loss (μKV's known boundary; what NIAH probes).
- Handles **arbitrarily long** context (disk-bounded, not memory-bounded), batch-16 throughput.
- More mature eval: **RULER + LongBench + NIAH**, on SOTA baselines (Quest/ShadowKV/InfiniGen).

## Implications for OUR MobiSys full work
1. **Lead with the phone** — the one thing KVSwap structurally cannot do. Position μKV as
   *training-free eviction for true phone-class devices with no disk and thermal limits*.
2. **Adopt their quality benchmarks**: RULER, LongBench, NIAH (not just disjoint-PPL). NIAH
   directly measures μKV's recall boundary — report it honestly, own the trade.
3. **Same-model Jetson comparison**: our Orin NX runs Qwen/Llama + Qwen2.5-VL (matches their
   set) — show the eviction-vs-swap operating-point trade (μKV: no-disk, faster, lossy;
   KVSwap: disk, lossless). Note AGX(32/64GB)≠NX(16GB), so frame as operating points not a race.
4. **VLM**: our μKV×VL integration (Qwen2.5-VL-3B) overlaps their VLM eval → run InternVL3 +
   Qwen2.5-VL under μKV for a matched multimodal row.
5. **Cite as complementary** in related work; cite their baselines (Quest/ShadowKV/InfiniGen).
