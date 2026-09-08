# OnePlus 15 / Adreno 840 — Hardware Stress Limits

This document records every failure mode we observed on the OnePlus 15
(Snapdragon 8 Elite Gen 5, Adreno 840, 12 GB RAM) while running llama.cpp
inference. Newest entries first.

Hardware:
- Phone: OnePlus 15 (SM8850 "Canoe")
- GPU: Adreno 840 (Qualcomm Vulkan driver, fp16=1, bf16=0, shared mem = 32 KB)
- CPU: 8-core
- RAM: ~15 GB total (with swap)
- Vulkan llama.cpp build: build-android-vulkan/ (NDK r27c, GGML_VULKAN=ON)

---

## 1. Phone-reboot conditions (CRITICAL — do not exceed)

| Trigger | Reproduces? | Recovery |
|---|---|---|
| Multiple consecutive `vk::DeviceLostError` events in <5 min | Yes | Force-shutdown, cool 10 min, power-on |
| Phi-3-mini-128k + GPU + long prompt + repeated runs | Yes (~3 sessions in) | Same |
| Long prefill (pp>2000) on 7B+ models with FA-on, repeated | Yes | Same |

**Operational rules now in force:**
- **Hard cap: 1 GPU cell every 2 minutes.** No back-to-back GPU runs.
- **Stop immediately on any DeviceLost.** Cool 10 min before resuming.
- **Never run GPU stress while phone is in deep cool-down state from a prior crash.** Wait until skin ≤ 35°C.

---

## 2. Vulkan `DeviceLost` — single-kernel TDR ceiling

The Adreno watchdog kills any compute kernel that runs longer than ~1–2 seconds.
This crashes `eviction_bench` (and `llama-bench`) with `vk::DeviceLostError`.

| Model | Params | Crashes at | Works at |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 8 B | pp≥1300, all `ngl` settings | n/a — does not fit on GPU |
| Mistral-7B-Instruct-v0.3 | 7 B | pp≥1300, FA-on/off | n/a |
| Qwen2-7B-Instruct | 7 B | pp≥1300 | n/a |
| DeepSeek-R1-Distill-Llama-8B | 8 B | (assumed, same arch) | n/a |
| Phi-3-mini-128k-instruct | 3.8 B | pp≥8000 (LongBench narrativeqa fits at 8500) | up through pp7700 |
| **Gemma-2-2B-it** | 2 B | **pp≥7000** (hotpotqa 7693 crashes; qasper 3500 works) | pp ≤ ~3500 |
| Llama-3.2-1B-Instruct | 1 B | never (in tested range up to pp9000) | all prompts |

**Cause** — the per-Vulkan-dispatch attention compute size grows with prompt
length × head_dim × n_heads. Gemma-2's 256-dim heads + global-attention layers
make it crash at smaller prompts than Phi-3 (96-dim heads). Llama-1B's smaller
hidden dim keeps it under the watchdog.

**Mitigations tried (none restored 7B+ functionality):**
- Smaller `n_ubatch` (64 → 32 → 16): didn't help
- Smaller `n_batch` with chunked prefill: didn't help
- Partial offload (`-ngl 16` instead of `-ngl 32`): didn't help
- FA-off instead of FA-on: didn't help

---

## 3. Vulkan output-degeneration bug (model-dependent)

Even when GPU inference completes without crashing, the **output is degenerate**
on some models. Reproduced with **stock `llama-completion`** (not our binary),
confirming this is an upstream llama.cpp / Vulkan / Adreno driver issue, not
an `eviction_bench` bug.

| Model | GPU FA-on | GPU FA-off | CPU (any FA) |
|---|---|---|---|
| Llama-3.2-1B | ❌ "Paris.@@@@@@@@..." (1 correct token, then loop) | ❌ same loop | ✅ "Paris." correct |
| Gemma-2-2B-it | ⚠ truncates early ("The capital of") | (?) | ✅ "Paris." correct |
| Phi-3-mini-128k | ❌ "FC dst局enci..." gibberish from token 1 | (?) | ✅ correct |
| Phi-3-mini-4k | (not tested on long prompts due to 4 K ctx) | (?) | (?) |

Pattern: the GPU produces **1–3 correct tokens**, then collapses into:
- Repeated punctuation (`@`, `!`, `8`)
- Random tokens (Phi-3)
- Premature truncation (Gemma)

This is a known class of bug — attention-sink collapse during autoregressive
decoding when fp16 accumulation goes unstable at scale.

**Workaround:** run all generation on CPU. Use GPU only for prefill-only
workloads where output text isn't needed (e.g. eviction policies that read
attention but discard generated tokens).

---

## 4. eviction_bench startup hang (FIXED 2026-05-31)

Symptom: binary hangs at startup with zero stderr output. Cause: missing
NEEDED libs (`libggml-cpu.so`, `libggml-vulkan.so`) — runtime
`ggml_backend_load_all()` deadlocked on Adreno. Fix: link backends as NEEDED
via CMake. See README "eviction_bench hung silently on Vulkan: missing
NEEDED libs".

---

## 5. KV-state cross-context restore (UNVERIFIED — for Strategy A)

`llama_state_seq_get_data` / `_set_data` API exists. Whether KV state from a
FA-off context can be restored into a FA-on context (Strategy A's swap)
hasn't been validated yet. The Vulkan output-degeneration bug above blocks
FA-on validation entirely; Strategy A is on hold.

---

## 6. Safe-operating envelope (current)

For sustained data collection without crashing the phone:

| Pillar | Recommended | Hard limit |
|---|---|---|
| Model size on GPU | ≤ 3.8 B params | 7 B+ crashes |
| Prompt length on GPU (Phi-3) | ≤ 6 K tokens | ~8 K crashes |
| Prompt length on GPU (Gemma) | ≤ 3 K tokens | 7 K crashes |
| Prompt length on GPU (Llama-1B) | ≤ 9 K tokens | none observed |
| GPU output text | **avoid — use CPU** | output is degenerate on Llama-1B / Phi-3 |
| GPU back-to-back runs | 1 per 2 min | unlimited triggered phone reboot |
| Skin temperature before next cell | ≤ 38°C | beyond 40°C triggers throttle/crash |
| Battery temperature ceiling | ≤ 42°C | beyond 42°C device throttles aggressively |

---

## Recovery procedures observed

- **DeviceLost crash**: process exits with signal 134 / EXIT 134. ADB stays alive.
  No phone reboot needed. Wait 60 s, retry with smaller cell.
- **Phone reboot**: ADB disconnects, `adb devices` shows empty list. Phone is
  bootlooping or off. **Physical action required:** unplug, hold power, let cool,
  power on. Wait until home screen stable for 2 min before reconnecting USB.
- **Bootloop persistence**: rare but happened once after extended GPU stress.
  Full power off + 5 min cool + power on resolved it.

---

## What this means for the dissertation

1. **Eviction policies must do their reads on GPU but their output sampling on CPU.**
   The cleanest pattern is: prefill on GPU (with eviction policy callback reading
   attention), then transfer KV state for decode on CPU.
2. **Or:** do everything on CPU. Slower (~3–5 t/s on 1B, ~1 t/s on 8B) but
   correct across the board.
3. **Or:** use Llama-3.2-1B only for GPU work (it's the only model that doesn't
   produce GPU output degeneration in our tests).
4. **The "Adreno 840 + Vulkan + Llama-Q4" stack is not production-ready** for
   long-context output generation. This is itself a reportable finding for the
   dissertation: mobile GPU inference is a real but constrained regime, and the
   constraints are stricter than the marketing materials suggest.
