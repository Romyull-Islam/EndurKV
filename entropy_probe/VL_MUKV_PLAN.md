# μKV × vision-language on Jetson — status & integration plan (2026-07-25)

## Baseline established (no code changes)
- **Qwen2.5-VL-3B-Instruct Q4_K_M + f16 mmproj runs on the Orin NX** via `llama-mtmd-cli`
  (build-jetson-cuda): LLM 37/37 layers on CUDA0, correct image description, greedy.
- Vision encoder currently on **CPU** (`--no-mmproj-offload`, 15.8 s per 896-px image):
  clip-on-GPU hit NvMap ENOMEM under page-cache pressure right after 3.1 GB of downloads —
  retry after reboot/cache drop; not a hard blocker (encoder is a one-shot cost per image).
- Files: `~/ukv/models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` (1.8 G) + `mmproj-…-f16.gguf` (1.3 G).

## Why μKV applies directly
mtmd feeds image patches into the SAME llama KV cache as text tokens (arch `qwen2vl`,
GQA: 2 KV heads × 36 layers — skinny KV). At the cache level, cells are cells: the frozen
μKV mechanism (FA-on capture → end-window scoring → logical `seq_rm` eviction) needs no
policy change. The work is plumbing, not mechanism.

## Integration steps (eviction_bench, documented per CHANGELOG discipline)
1. Link `libmtmd` into eviction_bench; add `--image PATH[,PATH…]` + `--mmproj PATH`.
2. Prefill = mtmd chunks (text → image embeddings → text), reusing the existing chunked
   loop; keep the last-chunk-only `kq_evict` side node exactly as today (the observation
   window is the text QUESTION after the image(s) — the natural SnapKV setup).
3. Eviction as-is post-prefill; FA-on decode over the compacted cache.
4. Workload for the table: N high-res images (e.g., 4–8 pages/photos ≈ 6–12K image tokens)
   + a question; decode 512–1024 tokens. Cells: vanilla vs μKV; metrics: prefill, tps,
   retained_kv, peak RSS, and answer-correctness spot checks (+ optional PPL-style
   teacher-forced scoring on a reference answer).

## Research questions this answers
- Does the adaptive mass gate treat image-token spans as "distant mass" (evictable) or
  does the question's observation window anchor onto the relevant image regions?
  (Expect: keeps the attended image patches — visual needle-in-haystack.)
- KV-per-token is small for this arch (2 KV heads), so the VL win may be more about
  long multi-image sessions than single images — multi-image chat is the target workload.

## Decision needed
The integration is a real (but bounded) eviction_bench change — proceed, or keep the
Jetson deliverable at the current text-model table + this validated VL baseline?
