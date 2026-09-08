# Phone (OnePlus 15 / Snapdragon 8 Elite Gen 5) — all runs, all models, all policies

Prompt: 12K WikiText (`prompt_12k.txt`) → **9737** Llama / **10074** Bonsai / **11158** Phi-3 tokens.
Protocol: native DVFS, cool gate before each timed cell (DDR ≤ 35 °C, battery ≤ 34 °C, charging OFF).
μKV = frozen config `--policy v1_fa2 --fa-on-evict --adaptive-anchor --gate-alpha-floor 0.70`.

---

## A. CPU (n_gpu_layers = 0)

### A1. Llama-3.2-1B Q4_K_M — 4096-token decode (paper master table)
| policy | tps | prefill (s) | wall (s) | live KV | α | RSS (MB) | DDR °C | CPU °C | energy (mWh) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 5.0 | 229 | 1055 | 9737 | — | 2102 | 59.8 | 65.8 | 1033 |
| **μKV-mass-full** | **24.2** | 226 | **406** | **721** | 0.71 | 2154 | 54.4 | 63.5 | **391** |
| μKV (state-swap) | 22.7 | 250 | 438 | 733 | 0.72 | 2235 | 58.3 | 68.1 | 546 |
| SnapKV | 6.8 | 287 | 902 | 5929 | — | 2236 | 56.3 | 63.1 | 818 |
| AdaKV | 6.7 | 252 | 875 | 3484 | — | 2236 | 57.1 | 63.5 | 1002 |
| StreamingLLM | 6.6 | 250 | 877 | 777 | — | 2236 | 56.7 | 63.5 | 1052 |
| H2O | 5.8 | 292 | 1006 | 6110 | — | 2312 | 54.8 | 60.7 | 1094 |
| TOVA | 6.0 | 253 | 947 | 4091 | — | 2235 | 54.4 | 60.7 | 1069 |

### A2. Bonsai-8B Q1_0 (1-bit) — 4096-token decode (paper master table)
| policy | tps | prefill (s) | wall (s) | live KV | α | RSS (MB) | DDR °C | CPU °C | energy (mWh) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **μKV-mass-full** | **4.5** | 1073 | **2012** (33.2 min) | **789** | 0.70 | 4517 | **62.5** | 79.5 | **2312** |
| vanilla | 1.0 | 1058 | 5137 (85.6 min) | 10074 | — | 4393 | 72.2 | 79.8 | 5758 |
| SnapKV | 1.1 | 1103 | 4713 | 10015 | — | 4475 | 67.9 | 81.4 | 5476 |
| AdaKV | 1.6 | 1191 | 3718 | 6514 | — | 4580 | 65.2 | 76.0 | 3971 |
| StreamingLLM | 1.6 | 1161 | 3777 | 858 | — | 4580 | 63.7 | 76.4 | 4012 |
| H2O | 1.5 | 1214 | 3900 | 9587 | — | 4748 | 63.3 | 76.8 | 4136 |

### A3. Phi-3-mini-128k Q2_K — 16K ctx, q8_0 KV, 128-token decode (measured 2026-07-26)
| policy | tps | prefill (s) | live KV | notes |
|---|---:|---:|---:|---|
| vanilla | 1.2 | 2253.3 | 2223 MiB | |
| **μKV** (K=1024) | **5.7 (4.75×)** | 2206.8 | **255 MiB (8.7×)** | prefill parity |

### A4. Phi-3-mini-4k Q4_K_M — earlier campaign (paper prose)
| policy | tps | wall (s) | swap | PPL |
|---|---:|---:|---:|---:|
| vanilla | 2.97 | 1036 | 158 MB | 5.46 |
| **μKV** (`v1_fa2_stack`, K=512) | **4.98 (+68%)** | **832 (−20%)** | **0 MB** | 6.08 |
| AdaKV | 1.80 | 1674 | — | **5.345** |

### A5. Bonsai-8B Q1_0 — llama-bench control (pp64/tg16, 2026-07-26)
| policy | prefill tps | decode tps |
|---|---:|---:|
| vanilla, ngl 0 | **11.70** | 0.57 |

---

## B. GPU (Adreno 840, Vulkan, n_gpu_layers = 99)

### B1. Llama-3.2-1B Q4_K_M — 4096-token decode (paper GPU table, n=3)
| policy | tps | prefill (s) | wall (s) | energy (mWh) | GPU °C | DDR °C |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 30.7 | **116** | 252 | 492 | 90.3 | 83.8 |
| SnapKV (seq) | 33.6 | 117 | 241 | 437 | 86.9 | **80.7** |
| μKV (no watchdog) | 33.0 | 117 | 244 | 453 | 87.6 | 81.1 |
| **μKV + watchdog v4** | **34.6** | 117 | **238** | **424** | 88.4 | 82.2 |

### B2. Llama-3.2-1B Q4_K_M — strict-settle reproduction (n=3, 2026-07-25)
| policy | tps | prefill (s) | wall (s) |
|---|---:|---:|---:|
| vanilla | 29.3 | 114.6 | 255.7 |
| **μKV** | **33.4** | 115.8 | **239.7** |

### B3. Phi-3-mini-128k Q2_K — 16K ctx, 128-token decode (2026-07-25)
| policy | KV type | KV allocated | live KV | prefill (s) | tps |
|---|---|---:|---:|---:|---:|
| vanilla | f16 | 6144 MiB | 4184 MiB | 583.3 | 2.0 |
| vanilla | q8_0 | 3264 MiB | 2223 MiB | 730.8 | 7.2 |
| **μKV** | q8_0 | 3264 MiB | **193 MiB (11.5×)** | 751.6 | **11.8 (+64%)** |

### B4. Bonsai-8B Q1_0 — ❌ **NO VALID RESULTS**
Prism's Vulkan Q1_0 kernels are **numerically broken on Adreno**: teacher-forced PPL returns
`nan` for every policy, and generation is degenerate (`" fine!!!!!!!!"`) where the CPU produces
coherent text from the same binary/prompt. All previously reported phone-GPU Bonsai numbers
are **retracted** (see `BONSAI_GPU_RESULTS.md`). CUDA is unaffected and verified correct.

---

## C. CPU vs GPU, same model/policy (phone)
| model | policy | CPU tps | GPU tps | GPU speedup |
|---|---|---:|---:|---:|
| Llama-1B Q4_K_M | vanilla | 5.0 | 30.7 | **6.1×** |
| Llama-1B Q4_K_M | μKV | 24.2 | 33.0–34.6 | 1.4× |
| Phi-3-mini Q2_K | vanilla | 1.2 | 7.2 | **6.0×** |
| Phi-3-mini Q2_K | μKV | 5.7 | **11.8** | 2.1× |
| Bonsai-8B Q1_0 | vanilla | 1.0 | — (broken) | — |
| Bonsai-8B Q1_0 | μKV | 4.5 | — (broken) | — |

**Reading:** the GPU gives ~6× on vanilla, but μKV already recovers most of that on the CPU
(4.5–4.8×), so GPU+μKV compounds to a smaller additional factor. For the 8B the GPU is
unusable, so **CPU + μKV is the deployment**.

## D. μKV gain by model (phone)
| model | CPU gain | GPU gain | live-KV cut |
|---|---:|---:|---:|
| Llama-1B Q4_K_M | **4.8×** | +14% (n=3 strict) | 13.5× |
| Phi-3-mini Q2_K | **4.75×** | **+64%** | 8.7× (CPU) / 11.5× (GPU) |
| Phi-3-mini-4k Q4_K_M | +68% | — | — |
| Bonsai-8B Q1_0 | **4.5×** | n/a | 12.8× |

μKV's decode gain is largest where the KV is fattest per token (Phi-3 has no GQA) and where
the baseline is most bandwidth-starved (CPU).
