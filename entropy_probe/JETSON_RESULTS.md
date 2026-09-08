# μKV on Jetson Orin NX 16 GB — measured results (2026-07-24/25)

Platform: JetPack R36.4.7, CUDA 12.6, SM_87, 16 GB unified. Binary: `build-jetson-cuda/eviction_bench`
SHA `39d640c4…` (post V-gate fix; f16 regression EXACT vs `f33634a6…`). Profile B of `PLATFORMS.md`.
Workload: 12K WikiText prompt (`prompt_12k.txt`, SHA `50c4497f…`) + 4096-token greedy decode,
ctx 16384, seed 42. Prompt tokens per tokenizer: Llama 9737 / Phi-3 11158 / Bonsai 10074.
μKV = frozen config (`MUKV_GPU_VERSION.md`): logical eviction, no defrag, FA-on end-to-end.

## Memory table
| Model | params | bit | file | KV alloc @16K | KV live van→μKV | peak RSS van/μKV |
|---|---:|---:|---:|---:|---:|---:|
| Bonsai (Prism) | 8.19 B | 1 (Q1_0, trained) | **1.08 GiB** | 2304 MiB f16 | 1417 → **111 MiB** (12.8×) | 4.20/4.28 GB |
| Llama 3.2 1B | 1.24 B | 8 (Q8_0) | 1.23 GiB | 512 MiB f16 | 304 → **22.9 MiB** (13.3×) | 2.71/2.73 GB |
| Phi-3-mini-128k | 3.82 B | 4 (Q4_K_M) | 2.23 GiB | 3264 MiB q8_0¹ | 2223 → **174.5 MiB** (12.7×) | 5.92/6.00 GB² |
| Phi-3-mini-128k | 3.82 B | 2 (Q2_K PTQ) | 1.35 GiB | 3264 MiB q8_0¹ | 2223 → **182.5 MiB** (12.2×) | 5.32/5.35 GB |

² Q4_K_M peak RSS from PPL-mode cells (includes eval-text pass); KV alloc/live valid.
The two Phi-3 rows share KV geometry (KV = arch×ctx×quant, weight-independent); Q4_K_M is
the quality-valid variant (PPL 3.94), Q2_K is quantization-destroyed (PPL ≥272).

¹ f16 KV = 6144 MiB in ONE buffer → **unallocatable ON JETSON** (Tegra ~4 GiB single-alloc
limit; idle-confirmed). Even the 11K prompt alone (~4.1 GiB f16) exceeds it → both policies
run q8_0 KV here. NOTE (2026-07-25): this limit is **Tegra-specific** — the phone's Adreno 840
allocated the same 6144 MiB f16 buffer fine (see `PHONE_PHI3_RESULTS.md`); do not generalize it.
Current μKV allocates full-ctx then evicts logically → fails identically in f16 on Jetson; the
evict-during-prefill (allocate-small) variant would cap the buffer at ~B_max and run where
vanilla cannot start. Measured motivation for that future work.

## Performance (vanilla → μKV)
| Model | prefill | decode tps | wall |
|---|---:|---:|---:|
| Llama 1B Q8_0 | 4.83 → 5.14 s | 32.7 → 33.5 | 133 → 130 s |
| **Phi-3-128k Q4_K_M** (q8 KV) | 25.7 → 27.8 s | 4.5 → **5.1 (+13%)** | 935 → **832 s (−11%)** |
| Phi-3-128k Q2_K (q8 KV) | 33.9 → 36.9 s | 4.3 → **4.9 (+14%)** | ~986 → ~873 s |
| Phi-3-4k Q2_K (q8 KV)² | 33.6 → 36.0 s | 4.4 → **5.9 (+34%)** | ~964 → 737 s |
| Bonsai 8B Q1_0 | 346.7 → 350.4 s | 4.2 → **4.7 (+12%)** | 1320 → ~1222 s |

² 4k variant beyond trained window — speed/memory valid, quality invalid; superseded by 128k row.
Bonsai Q1_0 weights CPU-fallback (no CUDA 1-bit kernel; 1.18 MiB on GPU) — same blocker class as
Vulkan on the phone (`BONSAI8B_GPU_BLOCKER.md`); prefill/decode CPU-bound.
μKV decode gain scales with KV fatness: Phi-3 (no GQA, fattest) > Bonsai > Llama-1B (skinny GQA).


## μKV × VLM (Qwen2.5-VL-3B-Instruct Q4_K_M, single image, 1024-token decode)
GPU vision encoder (llama.cpp default `use_gpu=true`; applied identically to BOTH policies —
a platform setting, never an μKV mechanism). Prompt: "Describe everything you see…".
| | vanilla | μKV |
|---|---:|---:|
| prefill (incl. 21.5 s GPU image encode) | 26.83 s | **26.97 s (+0.5%)** |
| decode | 20.2 tps | 20.3 tps |
| retained KV post-prefill | 144.40 MiB | **22.90 MiB (6.3×)** |
| output | correct description | correct description |

- **Prefill parity holds on multimodal** (+0.5%, tighter than text's +4–6%): the last-chunk
  `kq_evict` side node fires only on the final text chunk (which carries the question).
- **No speed win at single-image scale, and that is expected**: 4105 image+text cells (144 MiB)
  vs ~1.8 GB of weights → KV is not the decode bottleneck. The throughput case for VL requires
  multi-image/video context where KV reaches GB scale (KVSwap's target workload).
- Required fork work for VL (see CHANGELOG): cell-index eviction API
  (`llama_endurkv_seq_rm_cells`) because M-RoPE gives every image token ONE dim-0 position
  (4105 cells packed into 110 positions), plus skipping `seq_add` position-compaction
  (illegal on M-RoPE). Text path verified bit-identical after both changes.
- Measurement caveat: the CPU vision encoder (earlier default) took 184–242 s with ~30%
  run-to-run variance and buried the ~5.4 s LLM prefill; always run the encoder on GPU.
- Transient OOM note: back-to-back cells can hit a CUDA compute-buffer OOM on the 16 GB Orin;
  rerun the affected cell alone (the isolated μKV run completed normally). This is memory
  pressure, NOT an μKV requirement — μKV never needs more memory than vanilla (PLATFORMS.md
  robustness invariant).

## Quality — teacher-forced PPL over a VERIFIED-DISJOINT WT2 slice
(`wiki_eval_disjoint_4k.txt`, 0/159 sliding 200-char probes present in the prompt)
| Model | vanilla | μKV | ratio |
|---|---:|---:|---:|
| Llama 1B Q8_0 | 8.198 | 8.023 | 0.979 |
| Bonsai 8B Q1_0 | 6.717 | 6.736 | 1.003 |
| **Phi-3-128k Q4_K_M** | **3.983** | **3.938** | **0.989** |
| Phi-3-128k Q2_K | 771.9³ | 475.4³ | — |

**μKV = quality parity** (±2%) on ALL THREE healthy configs (Llama-Q8, Bonsai-Q1, Phi-3-Q4KM)
after evicting ~92% of prompt KV. Phi-3-Q4KM confirms the fat-KV model keeps quality under
eviction just like the others — the win there is decode bandwidth, not a quality trade.

³ **Phi-3 Q2_K is quantization-destroyed, independent of policy**: short-context diag
(36-byte prompt, no eviction) PPL = **271.8** → the 2-bit PTQ alone is unusable; long context
under Q2_K worsens it to 772; the 4k variant beyond its window scored 24,757. μKV's lower number
(475) = removing degraded-context noise from a broken model, NOT a quality win.
**Paper-ready contrast: trained 1-bit (Bonsai 8.2B: PPL 6.7, 1.08 GiB) vs post-hoc 2-bit PTQ
(Phi-3 3.8B: PPL ≥272, 1.35 GiB)** — at 2× params and a smaller file, trained low-bit is
production-quality while 2-bit PTQ has no usable operating point at ~4B scale.

## Diagnostics kept (not headline numbers)
- **Recall-stress artifact** (first PPL pass, eval slice verbatim inside prompt): vanilla PPL 1.16
  (copies retained text) vs μKV 11.7 (evicted the verbatim source → falls back to LM knowledge).
  Honest limitation: eviction sacrifices verbatim recall of dropped spans (what NIAH probes).
- Adaptive gate adapts per architecture (same frozen config): Llama α=0.72→anchor 733/recent 287;
  Phi-3-128k α=0.90→916/104; Phi-3-4k α=0.95→969/51; Bonsai→789 anchors.

## Provenance
Results dirs: `orin-nx:~/ukv/results/*` (gen.err + meta.json + mem.csv per cell).
Code change log: `CHANGELOG.md` (07-24 V-gate fix). Build: `build_jetson_cuda.sh`.
