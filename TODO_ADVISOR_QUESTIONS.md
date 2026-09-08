# Advisor questions — MUST be answered before submission

Raised 2026-08-10. These are not optional; both are framing-level objections that a
reviewer will also raise.

---

## Q1. "Your design is not energy-aware. Based on the phone's existing charge, how does
##      performance adapt? There is no loop for energy in your system design."

### Why the objection is correct as the system stands
The frozen config has **no battery-state input anywhere**. Verified in the code:
- `--k-nominal` is a CLI constant. The adaptive machinery (`alpha_a`, the mass gate) only
  splits K between anchor and recent; it never changes K's magnitude. So the cache budget
  cannot respond to anything, let alone charge level.
- The watchdog reads battery **temperature** (thermal_zone93) but never
  `capacity` / `charge_counter` / `current_now`. It is a thermal controller, not an
  energy one.
- TDAK (cache-as-thermal-actuator) exists in the binary but is OFF in the frozen config,
  and the draft reports it as a NEGATIVE result: coupling cache size to temperature does
  not cap heat.
So "closed-loop" in the current title refers to a loop that is not deployed, and no loop
of any kind closes on energy.

### What the measurements can already support
- Energy per token IS measured (USB-rail integration) for every phone cell.
- muKV cuts mJ/token 302.9 -> 293.4 (Llama-1B) and 1440 -> 1059 (Phi-3) — but ONLY by
  finishing sooner. Mean power is identical (3.45 W vs 3.44 W). Eviction does not lower
  draw; it shortens the run.
- The watchdog is energy-NEUTRAL by measurement: 10% shorter x 11% more power = same
  joules (10688 J vs 10666 J).
- `--k-pct` (added 2026-08-07) makes the budget a percentage of prompt length, so a
  budget-setting hook now EXISTS even though nothing drives it.

### The honest answer + the minimum credible fix
Two options, in increasing cost:
1. **Reframe (cheap, defensible today).** Drop "closed-loop" and any energy-adaptive
   claim. State plainly: muKV reduces energy per request by finishing sooner, at constant
   power; adapting to state-of-charge is future work. Costs nothing, removes the attack.
2. **Close the loop for real (one evening of work).** Read
   `/sys/class/power_supply/battery/capacity` and drive `--k-pct` from it, e.g.
   K = 20% above 50% charge, 10% between 20-50%, 5% below 20%. The K-vs-quality curve is
   ALREADY MEASURED at 64K (2%/5%/10%/20%/30%/50% -> PPL and speedup), so the policy can
   be specified from existing data rather than guessed. This turns the objection into a
   contribution: a cache budget that trades quality for battery life on a measured curve.
   NOTE: the 64K PPL column of that sweep is recall-contaminated (see below) and must be
   re-measured on `benchmarks/ppl/wiki_eval_disjoint_64k.txt` before being used to
   justify thresholds.

---

## Q2. "Is our design CUDA vs our design on phone? What is unique in the mechanism?
##      Why doesn't the original CUDA design work there, and we had to be more accurate?
##      What are the selling points?"

### The short answer
Nothing about muKV's *selection rule* is phone-specific. What is phone-specific is that
THREE constraints, all absent on a server GPU, force design choices that would look
unnecessary in a CUDA paper — and every competing method violates at least one of them.

### The three constraints, each measured on this hardware
1. **A sequence-level cell array.** llama.cpp (and on-device engines generally) share ONE
   cell array across all heads and layers; a cell frees only if EVERY selector dropped it.
   Per-head budgets therefore do not materialize:
       SnapKV 2048/head -> 6592 cells (67.7%) on Llama-1B, 9468 (84.9%) on Phi-3,
       30634 (53.6%) at 64K; H2O's 20%-of-N ratio -> 99.8% retained.
   muKV selects ONE shared keep-set, which is why 1024 -> 723 cells actually happens.
2. **Flash-attention is not optional.** A per-head evictor must read attention weights,
   which forces FA-off. On-device that is catastrophic and reproduces on three backends:
       SnapKV 0.20x (phone GPU), 0.24x (Jetson), 0.17x (64K CUDA) vs vanilla.
   muKV scores INSIDE the FA graph via the kq_evict side node, emitted only on the last
   prefill chunk, so it never turns FA off. StreamingLLM also keeps FA — but only by
   having no attention signal at all, which costs it 1-2/7 on NIAH and -6 to -11 F1.
   *** muKV is the only policy with BOTH the attention signal AND flash-attention. ***
3. **Memory is unified and small.** Compaction via the state round-trip needs a SECOND
   full context: 2 x 6144 MiB + 2.4 GB weights > 15.47 GB, so Phi-3 at 16K is KILLED on
   the phone (reproduced twice; Android SIGKILLs six app processes). The same call fails
   on a 24 GB RTX with "second context alloc failed", so this is a property of the
   MECHANISM, not of phone RAM. In-place chunked compaction (added 2026-08-07) needs one
   cache and runs the same configuration at 874 cells.

### Why "we had to be more accurate" — concrete backend hazards found on the phone only
- q8_0 KV is NUMERICALLY BROKEN on this Adreno build (random tokens; NLL > ln(vocab)).
- FA-off + the cb_eval attention capture corrupts output on Vulkan: 1-13% degenerate
  tokens for five baselines, while vanilla FA-off is clean. Our bug, still open.
- Vulkan f16 attention costs +19% PPL (Phi-3) to +70% (Llama-1B) vs CUDA on identical
  inputs, so cross-device PPL comparison is invalid; quote within-device ratios only.
- `n_ubatch` cannot exceed 64: 128+ triggers vk::DeviceLostError (driver GPU-hang
  watchdog). This makes prefill ~86x slower than CUDA where compute specs predict 25-50x,
  which in turn makes the workload PREFILL-dominated (65%) and caps what any cache-side
  optimization can deliver end-to-end (1.35x decode -> 1.11x wall).
- The watchdog's sign FLIPS with thermal regime: -14% prefill on a cold-start cell (it
  concedes clock the vendor had not taken) but +10.8% under sustained load (its glide
  holds a HIGHER clock than the vendor's reactive drop to 726/422 MHz).

### Selling points, ranked by how well the data supports them
1. muKV is the ONLY policy in this set that is FASTER than not evicting at all on a phone
   (1.23x / 2.06x) — every FA-off baseline is 0.10-0.30x. Retention does not predict
   speed; keeping flash-attention does. (fig_retention_vs_speedup)
2. First measurement of the realizability gap on real on-device engines, across 4 models.
3. In-place compaction makes a configuration feasible that was previously impossible
   (Phi-3 @16K), on both phone and RTX.
4. Quality is not the trade: LongBench +0.1 / -0.0 at 12-13% retention, NIAH 7/7.

### What must NOT be claimed
- Not "closed-loop" (eviction is one decision at end of prefill; only the recent window
  slides during decode).
- Not "cache-as-thermal-actuator" (TDAK is off and is a published negative result).
- Not a memory saving in peak footprint (the KV buffer is allocated at full n_ctx up
  front; muKV's peak RSS is HIGHER than vanilla's because of scoring state). Quote
  retained CELLS.

---

## Blocking data issues that feed both answers
- [ ] **IN-PLACE COMPACTION IS WRONG FOR StreamingLLM AT 64K** (found 2026-08-13). Three-way
      controlled comparison, same policy, same 2049-cell keep-set, all FA-on, only the
      compaction mode differing:
          no compaction  PPL 10.438
          round-trip     PPL 10.432   (agrees with uncompacted to 0.06%)
          in-place       PPL 13.474   (+29%)
      The uncompacted control and the round-trip agree, so IN-PLACE is the one that moved.
      This is our own contribution, so it must be resolved, not caveated. Notes:
      * muKV is NOT affected at 64K: none/round-trip/in-place = 10.436/10.427/10.420,
        a 0.15% spread. The three-way muKV check at K=1024, K=8192 and K=free all agree.
      * R5 verified StreamingLLM + in-place at ctx 16384 to 0.010-0.096%, so this is
        context- or keep-set-shape dependent, not a blanket failure.
      * Suspect the keep-set SHAPE rather than the size: StreamingLLM at 64K keeps two runs
        separated by a 55066-cell gap (4 sinks at 0-3, then the window at 55074-57118),
        whereas muKV's survivors are many short runs spread through the cache. The move
        loop's dst<=src invariant holds in both cases, so the fault is more likely in the
        metadata gather/scatter (cells.cp / rm / set at llama-kv-cache.cpp:501-509) than in
        the tensor slide.
      * NEXT STEP: sweep ctx for StreamingLLM + in-place (8K/16K/32K/64K) to find where it
        starts diverging, then bisect on keep-set shape with a synthetic src[].
      Until resolved, no table may present a StreamingLLM + in-place row as valid, and the
      in-place contribution must be stated as verified for freeze-after-prefill selection
      with muKV-shaped keep-sets.
- [x] ALL 64K PPL was recall-contaminated -- RE-RUN AND FIXED 2026-08-13. 27 cells
      re-measured against benchmarks/ppl/wiki_eval_disjoint_64k.txt (asserted 0/119 and
      0/399 disjoint from the prompt at run time by rerun_64k_ppl_clean.sh). Void cells
      preserved under /tmp/rtx64k_*_VOID_contaminated_ppl. Vanilla now reads 11.045
      (it read 1.03 under contamination). Original note follows:
- [ ] ALL 64K PPL is recall-contaminated: the 57344-token prompt overlaps
      wiki_eval_disjoint 119/119 and wiki_eval_disjoint_long 70/119. Clean slice created
      (`benchmarks/ppl/wiki_eval_disjoint_64k.txt`, 0/119 and 0/399 vs all three
      prompts); ~10 cells to re-run. Needed before the K-sweep can justify Q1's thresholds.
- [x] cb_eval Vulkan corruption -- **ROOT-CAUSED 2026-08-13**, fix not yet implemented.
      It is NOT a read bug and NOT the policies. Adreno and CUDA return kq_soft_max in a
      byte-identical layout (f32, nb=[4,1024,65536], packed), so the capture was reading
      correctly all along. Isolated with two new diagnostic flags on vanilla FA-off, same
      seed, same prompt, 200 tokens:
          A  no callback                       0/978 degenerate '!'    0.0%
          B  callback: split + readback       76/609                  12.5%
          C  callback: split, NO readback     68/679                  10.0%
      C is decisive: the readback is innocent. Returning true from the ask phase makes
      ggml_backend_sched end a SPLIT at kq_soft_max, and on this Vulkan backend a split
      there corrupts the rest of the attention. Vanilla only ever looked clean because it
      is the one policy that never installs the callback.
      * PAPER CONSEQUENCE (stronger than the bug): the mid-graph attention intercept, which
        is how every score-reading evictor is implemented, is numerically unsafe on this
        mobile GPU regardless of the policy on top. muKV is immune because kq_evict is a
        real side NODE, not an intercept -- so the mechanism muKV needed to keep FA on is
        also the only safe way to read attention here. This belongs in the design section.
      * FIX: emit a side node copying the soft_max output for FA-off policies instead of
        splitting the live graph. Until then, phone-GPU FA-off cells are TIMING-ONLY and
        their perplexity must not be reported.
- [ ] Phi-3 in-place row confounds compaction with the watchdog (v5LOW is ON, the
      comparison rows have no watchdog). Needs an in-place + no-watchdog cell.
- [ ] Gemma-2 at ctx 16384 exceeds n_ctx_train=8192; re-run at 8192 or drop.
- [ ] Phi-3/TOVA LongBench does not reproduce (-6.7 and -14.0 F1 vs the draft).
