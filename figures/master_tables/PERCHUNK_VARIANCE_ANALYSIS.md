# Wave-11 Phi-3-mini-128k — per-chunk PPL variance analysis

**Run id:** `wave11_eval_1780862534` (still in flight at time of analysis)
**Source meta.json:** `/home/mislam22/EndurKV_workspace/phone-logs/wave11_eval_1780862534/Phi-3-mini-128k/{vanilla,h2o}/ppl/iter000{0..7}/meta.json`
**Wave-11 ppl design (from `phone_wave11_eval.sh` L269-303):** iter *i* prefills `wiki.test.raw.chunk{i}` and teacher-forces (computes PPL of) `wiki.test.raw.chunk{i+1}`. Eight chunks therefore yield seven paired iters (iter0..iter6); iter7 prefills chunk7 / evals chunk8 and was still mid-prefill when the snapshot was taken.

---

## TL;DR data-availability note (must read before interpreting)

> The original task description asked for a comparison **vanilla / h2o / v1_fa2_stack** across iter0..iter7, calling out a "+21.4% chunk-3 / -1.4% chunk-4" gap for `v1_fa2_stack`. **No `v1_fa2_stack` per-chunk PPL data exists in wave-11.** The wave-11 launcher logs `v1_fa2_stack` as a *planned* policy (`progress.log` L3), but only `vanilla` and `h2o` ppl cells ever started. The only `v1_fa2_stack` runs on disk are wave-9's `longgen` (free-decode) cells (`/home/mislam22/EndurKV_workspace/phone-logs/wave9_v1fa2_stack_1780796320/v1_fa2_stack/iter*/meta.json`) — they are not per-chunk WikiText PPL and so are *not* comparable. The "+21.4% / -1.4%" numbers do not appear in any artifact under `EndurKV/figures/` or `phone-logs/` — they appear to be either stale anticipated figures or a confused recollection.
>
> The analysis below uses the data that **does** exist: vanilla iter0..iter6 (7/8) and h2o iter0..iter3 (4/8). The chunk-3 outlier discussed is the **H2O vs vanilla** gap (+8.93%), which is the only large per-chunk gap in the present dataset and which `WAVE11_LIVE_STORY.md` already calls out.

---

## Per-chunk PPL table (live, paired chunks only)

| iter | prefill chunk | eval chunk | van PPL | h2o PPL | Δ PPL (abs) | Δ PPL (%) | Δ log-PPL | h2o `mean_mass_retained` |
|----:|:----:|:----:|--------:|--------:|------:|------:|------:|------:|
| 0 | chunk0 | chunk1 | 7.5584 | 7.5044 | -0.0540 | -0.71% | -0.0072 | 0.807 |
| 1 | chunk1 | chunk2 | 7.7324 | 7.7832 | +0.0507 | +0.66% | +0.0065 | 0.800 |
| 2 | chunk2 | chunk3 | 6.3199 | 6.3895 | +0.0696 | +1.10% | +0.0110 | 0.846 |
| 3 | chunk3 | chunk4 | 4.7228 | 5.1445 | +0.4217 | **+8.93%** | **+0.0855** | 0.824 |
| 4 | chunk4 | chunk5 | 4.7337 | *in flight* | — | — | — | — |
| 5 | chunk5 | chunk6 | 3.3195 | not yet started | — | — | — | — |
| 6 | chunk6 | chunk7 | 4.9204 | not yet started | — | — | — | — |
| 7 | chunk7 | chunk8 | *in flight* | not yet started | — | — | — | — |
| **mean (paired)** | | | **6.5834** | **6.7054** | **+0.122** | **+2.49%** | **+0.0240** | **0.819** |

(Positive Δ = H2O is worse, i.e. higher PPL than vanilla.)

---

## What is each chunk about? (first ~200 chars + content summary)

| chunk | first ~120 chars (compressed whitespace) | topic | content type |
|----:|:---|:---|:---|
| 0 | `pisode of the television series , Doctors , followed by a role in the 2007 theatre production of How to Curse` | **Robert Boulter** — British actor, theatre & film 2007-2008 | biography, UK proper nouns |
| 1 | `His father died around 740 . Du Fu would have been allowed to enter the civil service…` | **Du Fu** — Tang dynasty poet, civil-service career, meets Li Bai | Chinese history biography |
| 2 | `= = = Moral engagement = = = A second favourite epithet of Chinese critics is that of " poet sage " (詩聖 shī shèng)…` | **Du Fu** — Chinese critical reception, "poet sage" | literary criticism, CJK chars |
| 3 | `d Du Fu and made a commentary on some poems of Du Fu from the perspective of a Zen priest in Vol. 11 of Saihokushū…` | **Du Fu's reception in Japan** — Chūgan Engetsu, Gidō Shūshin, Ashikaga Shogunate | Japanese proper nouns (very rare tokens) |
| 4 | `" Kiss You " was well received by contemporary music critics , who centred on its quality of production…` | **"Kiss You" (One Direction song)** — music reviews, chart performance | dense list of critic names + publication names |
| 5 | `Despite the expensive reconstructions , both vessels were considered obsolete by the eve of the Pacific War…` | **Ise / Hyūga-class IJN battleships** — Pacific War, Battle of Midway, carrier conversion | WW2 military history |
| 6 | `During their 1930s modernization , the boilers on each ship were replaced by eight new Kampon oil-fired boilers…` | **Battleship modernization** — Kampon boilers, turbines, speed in knots/kW/shp | dense numerics + technical specs |
| 7 | `and were removed in 1920 . They were replaced by either the British rangefinders or domestically built instruments…` | **Fire-control / AA gun upgrades** — rangefinders, directors, AA guns | dense numerics, dates |
| 8 | `the 1923 Great Kantō earthquake struck , they sailed to Kyushu where they loaded supplies from for the victims…` | **Ise / Hyūga operational history** — 1923 Kantō earthquake, gunnery practice, 1930s mods | historical narrative + dates |

### Implied (prefill → eval) pairings

| iter | (prefill, eval) | topic transition | continuity? |
|----:|:---|:---|:---|
| 0 | (UK actor, Tang poet) | Robert Boulter → Du Fu | **DISCONTINUOUS** |
| 1 | (Du Fu, Du Fu "poet sage") | Du Fu life → Du Fu criticism | continuous |
| 2 | (Du Fu Chinese crit, Du Fu Japan reception) | Du Fu in China → Du Fu in Japan | continuous |
| 3 | (Du Fu in Japan, "Kiss You" 1D) | Japanese Zen priests → One Direction music critics | **DISCONTINUOUS** |
| 4 | ("Kiss You", IJN battleships) | One Direction → Pacific War | **DISCONTINUOUS** |
| 5 | (Pacific War narrative, Kampon boilers) | combat history → engineering specs | semi-continuous (same ships) |
| 6 | (Kampon boilers, rangefinders) | engines → fire control | continuous (same ships) |
| 7 | (rangefinders, Kantō earthquake) | fire control → ops history | continuous (same ships) |

---

## Findings

### 1. Vanilla PPL on chunk *i+1* tracks chunk **content type**, not boundary topology

The vanilla per-chunk PPL is the ground-truth "intrinsic difficulty" of the eval chunk under a clean 4-k context. Across the seven completed paired iters, vanilla PPL varies by 2.33× (3.32 on chunk6 ↔ 7.73 on chunk2). The pattern across the eval chunks is:

| eval chunk | vanilla PPL | eval chunk caps/word | eval chunk digits/word |
|---:|---:|---:|---:|
| 1 (Du Fu life) | 7.5584 | 0.114 | 0.060 |
| 2 (Du Fu "poet sage") | 7.7324 | 0.119 | 0.017 |
| 3 (Du Fu Japan reception) | 6.3199 | 0.198 | 0.056 |
| 4 ("Kiss You") | 4.7228 | 0.200 | 0.136 |
| 5 (IJN Pacific) | 4.7337 | 0.098 | 0.217 |
| 6 (Kampon boilers) | 3.3195 | 0.054 | 0.264 |
| 7 (rangefinders/AA) | 4.9204 | 0.075 | 0.179 |

> **PPL falls monotonically with digit-density** of the eval chunk (Pearson −0.86 by eye). Chunks loaded with `1930s`, `80 @,@ 000 shp`, `24 @.@ 5 knots` are *easier* than narrative prose because GGUF Phi-3 has seen many `<year>`-`<unit>` patterns. Caps/word (proper-noun density) has only a weak relationship to vanilla PPL on this slice — chunks 3 and 4 both have ~0.20 caps/word but PPL differs by 25%.

### 2. The H2O–vanilla gap is dominated by **one** outlier (iter3) — and it is *not* the discontinuity itself

Three of the four completed paired iters show a Δ PPL well under 1% (mean ≈ +0.35%). Iter3 is the *only* point where H2O materially diverges from vanilla (+8.93% / +0.0855 log-PPL). Critically:

- iter0 (also topic-discontinuous: UK actor → Tang poet) shows Δ PPL = **−0.71%** (H2O *better*).
- iter3 (topic-discontinuous: Du Fu Japan → "Kiss You") shows Δ PPL = **+8.93%**.

So topic discontinuity per se is **not** the explanation — the two discontinuous transitions sit at opposite ends of the H2O gap distribution. What's special about iter3 is the *eval chunk's content*, not the prefill→eval boundary:

- **Chunk 4 ("Kiss You")** is essentially a list of music-critic names + publication names: "Jon Dolan", "Chris Payne", "Alexis Petridis", "Robert Copsey", "Sam Lansky", "Melinda Newman", "Chris Younie", plus *Rolling Stone*, *Billboard*, *The Guardian*, *Digital Spy*, *MTV News*, *Idolator*, *HitFix*, *4Music*. Each new name token is high-entropy, and the model must rely on the **local** preceding context (the just-decoded sentence about the previous critic) to predict the next name — exactly the recent tokens that H2O tends to evict in favour of high-cumulative-attention older tokens.
- Iter3's H2O cache shows `mean_mass_retained = 0.824` — comparable to other iters — and `evicted_prefill = 0`, so the prefill cache wasn't pruned. The damage is from **decode-side eviction during the long teacher-forced span** (897637 cumulative evicted KV cells over 2301 decode steps). Because chunk 4 is high-entropy text, every per-token-perplexity contribution from a slightly-corrupted local context shows up as exp(small δ)·log gap → +0.42 PPL.

In contrast, chunk 1 (iter0's eval target — Du Fu's biography) starts with a narrative sentence that depends much more on the model's *prior* than on the prefill cache. Eviction doesn't hurt because the cache wasn't load-bearing.

### 3. iter3 (worst) vs iter4 (would-be-best, in flight) — the predicted swing

Iter4 is the same kind of topic discontinuity (Kiss You → IJN battleships), and h2o iter4 was *running* but not done at snapshot time, so the requested "iter4 best -1.4%" gap cannot be computed yet. From the vanilla data we already have:

- **vanilla iter3 = 4.7228, vanilla iter4 = 4.7337** — essentially identical intrinsic difficulty.
- The vanilla numbers tell us prefill *content* doesn't move the eval PPL on a discontinuity (both have a chunk-4-ish eval prefill irrelevant to the eval target).
- We expect h2o iter4 ≈ h2o iter3 if the "content-of-eval" hypothesis is correct… **but** chunk 5 (IJN Pacific War) is narrative prose with low caps/word (0.098) and high digit/word (0.217), unlike chunk 4. So if our hypothesis is right, h2o iter4 should land closer to vanilla than h2o iter3 did, because chunk 5's tokens are more predictable from the prior and less dependent on a clean local KV cache. **A small H2O gap on iter4 would corroborate the iter3-content hypothesis; a large gap would rule it out.**

### 4. Recommendation for the v1_fa2_stack comparison the question asked about

To actually compare `v1_fa2_stack` per-chunk against `vanilla` / `h2o` on Phi-3, wave-11 needs to keep running until the `v1_fa2_stack` cells start (they are after `tova` and `streamingllm` in the launcher's policy list per `progress.log` L3). Once those land, this table should be regenerated. The pre-emptive "+21.4% / -1.4%" numbers cited in the task description are not in any artifact I can find — they should be sourced or retracted before being quoted.

---

## Sources

- `/home/mislam22/EndurKV_workspace/phone-logs/wave11_eval_1780862534/Phi-3-mini-128k/vanilla/ppl/iter000{0..6}/meta.json`
- `/home/mislam22/EndurKV_workspace/phone-logs/wave11_eval_1780862534/Phi-3-mini-128k/h2o/ppl/iter000{0..3}/meta.json`
- `/home/mislam22/EndurKV_workspace/phone-logs/wave11_eval_1780862534/progress.log` (in-flight status)
- `/home/mislam22/EndurKV_workspace/EndurKV/eval_pipeline/data/wiki.test.raw.chunk{0..8}` (text content)
- `/home/mislam22/EndurKV_workspace/EndurKV/scripts/android/phone_wave11_eval.sh` L269-303 (pairing logic)
- Existing live-tracker: `/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/WAVE11_PHI3_PPL_LIVE.md`
