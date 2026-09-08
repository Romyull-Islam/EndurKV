
## 2026-07-26 — layout overflow fixes + PPL reframing on measured data

### Layout (155 overfull boxes -> 0)
WHY: text was bleeding up to 118pt past the column edge, and the running head
overlapped the venue string on every odd page.
- `\title[short]{long}`: the full title overflowed the running head. ROOT CAUSE of
  the "overlapping text" the reader saw.
- `\slb` (breakable slash) + `\ub` (breakable underscore) macros: TeX cannot break a
  text-mode "/" or hyphenate a `\texttt` identifier, so the watchdog frequency ladders
  (47.0/48.0/... , 1497/1382/...) and names like `llama_state_seq_get_size` were single
  unbreakable boxes.
- Eq. (2) `K_init` split into two aligned lines; the `mu(x)`/`K_h` display split into a
  `gathered` — both were wider than the column.
- Contribution (i): removed the inline `\mathrm{clip}(...)` and `\mathrm{round}(...)`
  groups. TeX has NO legal breakpoint inside them ("/" and "," are not binops), so they
  could only overflow. Prose now states the same thing; formulas stay in Sec. 3.
- `\emergencystretch=2em` for the residual sub-25pt cases.

### PPL reframing (content, not cosmetic)
WHY: the PPL_tf column was measured on an eval slice that OVERLAPS the prompt, so it
scored verbatim recall, not language modeling — it ranked policies by how much prompt
they retained, i.e. by the thing under test. Vanilla's 1.06 was recall, not quality.
Re-measured on a VERIFIED-DISJOINT slice (0/159 sliding 200-char probes present):
  Llama-3.2-1B Q8_0      8.198 -> 8.023  (0.98)
  Bonsai-8B Q1_0 (1-bit) 6.717 -> 6.736  (1.00)
  Phi-3-mini-128k Q4_K_M 3.983 -> 3.938  (0.99)
=> muKV holds full-cache perplexity after evicting ~92% of the prompt cache.
Source: entropy_probe/JETSON_RESULTS.md (CUDA; selection bit-identical across backends).
Updated to match: abstract, intro, CPU-results, GPU-results, discussion, conclusion.
The honest cost is now stated correctly: eviction costs VERBATIM RECALL of dropped
spans (what NIAH probes), not prediction quality.

### GPU / Bonsai-8B additions (reader asked for these explicitly)
- GPU results now carry the measured disjoint-slice PPL (Bonsai 6.717 vs 6.736,
  Llama-1B 8.198 vs 8.023) instead of only asserting hardware-independence.
- Added the negative result: PrismML's Vulkan Q1_0 returns non-finite logits on Adreno
  (PPL = nan both policies, degenerate generation) while the same binary is correct on
  CPU. Greedy decode emits tokens at full speed from NaN logits, so throughput and
  thermal metrics all looked healthy. This is why the 8B phone numbers are CPU numbers,
  and why an output-validity check is now mandatory before any new-backend measurement.
