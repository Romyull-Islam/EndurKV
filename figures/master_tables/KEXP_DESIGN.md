# Adaptive-K proof: K depends on (prompt × output) NATURE, not prompt size

**Claim to prove.** The KV budget K a request needs is set by the *effective attended span +
generation demand*, not by input prompt length N. Decisive prediction: a **short prompt with a
long output needs the LARGEST K**, while a **long prompt with a short output needs a small K** —
the inverse of what a prompt-length ladder assigns.

## Experiment (runs as Stage E, after LongBench n=3; `/tmp/phone_kexp.sh`)

A vault code (`8642`) is embedded **near the end** of each prompt (so prefill retention is easy and
the *generation* is what applies eviction pressure). 2×2 grid × fixed-window budget sweep:

- **prompt** ∈ {`pshort` ~65 tok, `plong` ~3600 tok}
- **output** ∈ {`o64` short, `o2048` long}, greedy, `--ignore-eos`
- **budget** ∈ {streamingllm K = 64/128/256/512/1024, vanilla = full-cache oracle}
- **model** ∈ {Llama-3.2-1B, Phi-3-mini}
- \+ `mukv_k256` on the decisive (pshort, o2048) case (content-aware retention vs fixed-window)
- = 2×2×6×2 + 2 = **50 cells**. Same 170-col sensor capture as every other stage.

**Metric (auto, `/tmp/kexp_analyze.py`).** `code_last` = vault code present in the **last 15%** of the
generation (retention-through-generation); `code_any`; 4-gram `rep`. **Break-even K** per
(model, prompt, output) = smallest streamingllm K with `code_last=1`.

**Expected 2×2 (the proof):**
| | short output (o64) | long output (o2048) |
|---|---|---|
| **short prompt** | small K | **LARGEST K** (code evicted after ~K gen tokens) |
| **long prompt** | small K | large K |

So break-even K rises with **output** and is ~flat in **prompt length** → prompt length
mispredicts, output/nature predicts. `mukv_k256` should keep the code where `stream_k256`
loses it (content-aware > fixed-window). Figure: `fig_kexp_proof.png` (break-even K vs output
length, split by prompt size).

## Rest — complete later (deferred)
1. **Live adaptive selector in the binary:** compute `r_eff` (positions holding ~90% of column-mean
   attention mass — μKV already has that tensor at the L6 state-swap) and emit a *recommended K*;
   show it matches the measured break-even K across cases. (Small addition to eviction_bench: cumsum
   over sorted mass; output to meta.)
2. **Mid-depth retrieval case** (needle at 50% depth) to test *placement* retention (where μKV's
   mass-based selection beats fixed-window) rather than recency.
3. **Nature-signal capture:** log per-prompt α_a + `r_eff` + attention entropy, and show they
   separate the cases (retrieval = low entropy/small span; reasoning/long-gen = high span).
4. Draft: turn L1 from the length ladder into the measured-span + demand controller (Eq. kdemand
   already added; add the algorithm block + signal table).
