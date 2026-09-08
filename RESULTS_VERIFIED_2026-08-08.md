# Verified results as of 2026-08-08 — what is safe to put in the paper

Every number below was re-measured after the harness bugs listed at the bottom were
fixed. Numbers NOT in this file should be treated as stale.

## 1. Phone GPU (Adreno 840), Llama-3.2-1B, 12K prompt + 4096 gen, ctx 16384, f16
Cool gate before every cell (DDR<=35C, batt<=33C, charging off while cooling).

| arm | prefill | decode | wall | tok/s | dec x | cells | peak RSS | energy | mJ/tok |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | 130.1s | 166.3s | 296.4s | 24.63 | 1.00x | 9741 | 517 MiB | 1352 J | 330.0 |
| muKV (round-trip) | 127.3s | 130.8s | 259.9s | 31.30 | 1.27x | 723 | 537 MiB | 1191 J | 290.7 |
| muKV (in-place) | 131.7s | 132.4s | 266.2s | 30.95 | 1.26x | 723 | 537 MiB | 1243 J | 303.6 |

Quality, teacher-forced on a slice VERIFIED disjoint (0/119 windows shared):
vanilla 23.3045 | muKV round-trip 22.7150 | muKV in-place 22.7160  (muKV <= vanilla)

Run-to-run spread across six muKV arms is +/-4% on tok/s and energy. Round-trip vs
in-place sits INSIDE that: they are indistinguishable on speed at this scale.

## 2. Watchdog: NULL RESULT, and the reason is measurable
Both v5 ladders started correctly (zones by name) and logged ZERO clock steps.
Peak battery 34.2-35.4C, peak skin 39.1-39.5C.
  v5LOW  trips bat 36.0 / skin 39.5  -> missed by 1.8 C
  v5HIGH trips bat 47.0 / skin 50.0  -> missed by 12.8 C
A cold-start 4096-token generation never reaches either ladder. This is a
workload-too-cool result, NOT "the watchdog does not help". Testing it needs the
natural-equilibrium protocol (no cool gate, back-to-back generations).

## 3. Phi-3-mini at ctx 16384 — previously reported INFEASIBLE, now runs
| device | round-trip | in-place |
|---|---|---|
| RTX 4500 (24 GB) | "second context alloc failed" | OK, PPL 6.4013 |
| OnePlus 15 | KILLED mid-allocation of the 2nd 6144 MiB cache | OK, PPL 7.9007, 874 cells, compacted 6.2 s |

Phone kill evidence (reproduced twice, logcat attached): MemTotal 15.47 GB unified;
exactly 1 of 2 KV allocations succeeds; Android killed 6 Zygote app processes with
signal 9. The draft's caption attributes this to the DEVICE; it belongs to the
round-trip compaction mode.

## 4. CROSS-DEVICE PPL IS NOT COMPARABLE — quote within-device ratios only
| device | backend | vanilla | muKV | ratio |
|---|---|---|---|---|
| RTX | CUDA | 6.1269 | 6.4013 | 1.0448 |
| OnePlus 15 | Vulkan | 7.2914 | 7.9007 | 1.0836 |

Vanilla ALONE is +19.0% worse on Vulkan with identical model bytes, identical prompt
and eval MD5s, identical token count, identical 874-cell keep-set, and batch size
proven irrelevant (CUDA PPL identical at 2048/512 and 512/64). The backend accounts
for most of the gap; muKV's own cost is +4.5% (CUDA) vs +8.4% (Vulkan).

## 5. The FA-off penalty — the strongest on-device systems result
Attention-based evictors must run FA-OFF to read attention weights, and llama.cpp
requires flash-attention for a quantized V (SnapKV core-dumps at q8_0).

Phone, 16K, WITH each method's own budget:
| policy | prefill | tok/s | vs vanilla | cells kept |
|---|---|---|---|---|
| vanilla | 131.1s | 28.63 | 1.00x | 9741 (100%) |
| muKV | 133.7s | 39.47 | 1.38x | 723 (7.4%) |
| SnapKV | 228.6s | 4.76 | 0.17x | 6592 (67.7%) |
| Ada-KV | 276.1s | 2.86 | 0.10x | 3519 (36%) |
| H2O | 278.8s | 3.58 | 0.13x | 6110 (63%) |
| TOVA | 175.9s | 4.34 | 0.15x | 4125 (42%) |
| StreamingLLM | 175.9s | 5.55 | 0.19x | 777 (8.0%) |

StreamingLLM keeps 8.0% of cells and is still 5x SLOWER than vanilla. muKV keeps 7.4%
and is 1.38x FASTER. Same retention, opposite outcome -> on-device speed is governed by
whether the policy can keep flash-attention ON, not by how much it evicts.

## 6. Realizability gap (per-head budget -> sequence-level cache)
| model | prompt | SnapKV budget | cells actually retained |
|---|---|---|---|
| Llama-3.2-1B (phone) | 9737 | 2048/head | 6592 = 67.7% |
| Phi-3-mini (phone) | 11157 | 2048/head | 9468 = 84.9% |
| Llama-3.2-1B (64K, CUDA) | 57118 | 2048/head | 30634 = 53.6% |
| H2O (64K, 20% of N) | 57118 | 11424 | 57039 = 99.8% |

Severity scales with selector count (layers x heads), because a sequence-level cell
array can only free a cell that EVERY selector dropped.

## 7. LongBench retrieval, matched retention (CUDA)
| model | policy | kept | hotpotqa | qasper | avg |
|---|---|---|---|---|---|
| Llama-1B | vanilla | 100% | 40.00 | 20.05 | 30.02 |
| Llama-1B | muKV | 13.0% | 43.13 | 12.29 | 27.71 |
| Llama-1B | StreamingLLM | 14.7% | 23.57 | 14.35 | 18.96 |
| Phi-3 | vanilla | 100% | 60.00 | 34.00 | 47.00 |
| Phi-3 | muKV | 12.2% | 53.45 | 42.76 | 48.11 |
| Phi-3 | StreamingLLM | 12.7% | 47.40 | 26.40 | 36.90 |

muKV beats StreamingLLM by +8.8 / +11.2 F1 at the same retention. Only 2 of 5 tasks
scored in this matrix; some phi3 cells have n=15.

## 8. Compaction is a MECHANISM, not muKV's property — state this explicitly
Giving StreamingLLM the same in-place compaction takes it from 1.01x to 2.61x at 64K
with an identical keep-set. On WikiText at MATCHED retention (3.6%) StreamingLLM beats
muKV on quality (PPL 6.32 vs 11.75). muKV's selection earns its keep on RETRIEVAL
(section 7), not on next-token prediction. Say this rather than let a reviewer find it.

## 9. In-place compaction: verified scope and limits
Verified identical keep-sets + PPL within FP reassociation across:
  models   Phi-3-mini, Mistral-7B, Llama-3.2-1B
  policies muKV, StreamingLLM, SnapKV (scattered per-head union), TOVA (frozen mask)
LIMITS:
  - DECLINES on sliding-window models (Gemma-2) and falls back to round-trip. The
    round-trip cannot be removed from the codebase.
  - NOT interchangeable with the round-trip for DECODE-TIME evictors: the round-trip's
    context swap drops the attention capture, silently converting TOVA from per-step
    eviction into freeze-after-prefill (1.41% PPL divergence; 0.096% once frozen).

## HARNESS BUGS FIXED (numbers predating these are void)
1. Destination context inherited n_batch=2048; state_seq_set_data places all cells in
   ONE find_slot call -> compaction silently capped at ~2048 live cells. Every K>1024
   run at 64K reported compaction_applied=false and no speedup.
2. --force-defrag was never generalized to baselines the way --compact-inplace was, so
   for StreamingLLM/TOVA/SnapKV it silently did NOTHING. Produced a spurious 1.41%
   TOVA "mode mismatch" that was really no-compaction vs in-place.
3. StreamingLLM was forced FA-OFF although it reads no attention values. Worth 4.4x:
   0.23x -> 1.01x at 64K.
4. The phone PPL eval slice was a stale scratchpad copy overlapping the prompt in 98.5%
   of 60-char windows -> measured RECALL, not prediction. Inverted the conclusion
   (vanilla 1.07 / muKV 13.35 became vanilla 23.30 / muKV 22.72). The runner now
   ASSERTS disjointness and refuses to run.
5. Gemma-2 cells at ctx 16384 exceed n_ctx_train=8192 -> PPL 1262, word salad. Any
   Gemma row must be re-run at 8192 or dropped.
