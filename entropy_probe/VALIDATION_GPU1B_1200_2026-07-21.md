# Llama-1B GPU held-1200 validation — 2026-07-21

## Purpose

Separate the observed 191 s μKV prefill result from a suspected stale-binary
failure. This is a validation record, not a source-code change.

## Controls

- Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`
- Prompt/decode: 9,737 / 4,096 tokens; context 16,384; seed 42; greedy.
- Policy: `v1_fa2`, mass adaptive-anchor, `K=1024`, `n_sink=4`, `R_min=32`,
  `gate_alpha_floor=0.70`, `snapkv_pool=7`, `fa_on_evict`.
- Start temperatures by named zones: GPUSS/DDR about 35 C, shell 31–32 C,
  battery 30.8 C.
- A source-mtime guard passed before launch. Host and device executable hashes
  matched: `e97a23f8f293a0ec29e12cb06e9ec5a7b976c79d1eb6d1e69bbbfac988e002a2`.
- The device script reasserted the 1200 MHz GPU cap every two seconds.

## Result

| Run | Active-prefill GPU clock | Prefill | Decode | Total | Notes |
|---|---:|---:|---:|---:|---|
| 2026-07-15 reference (`/tmp/gpu_sweep/mukv_mass`) | 1200 MHz median | 117.0 s | 31.9 tok/s | 245.4 s | historical artifact |
| Earlier capped run | 726 MHz median | 191 s | ~32.8 tok/s | — | 117 × 1200/726 = 193 s, so prefill is clock-explained |
| 2026-07-21 held-1200 run (`/tmp/gpu1b_1200`) | **1200 MHz for every active prefill sample** | **123.4 s** | **44.0 tok/s** | **216.5 s** | peak GPUSS/DDR 97.3/88.8 C |
| 2026-07-21 v3 protocol replica (`artifacts/reproductions/repro0715_gpu1b_v3`) | **1200 MHz for every active prefill sample** | **131.1 s** | **27.0 tok/s** | **283.3 s** | watchdog v3 started but took no action; vendor governor reduced decode clocks |

The held-1200 result validates the last-chunk prefill fix and the 726 MHz
clock explanation for the 191 s observation. The remaining 5% prefill spread
is ordinary run/build variation until repeated.

## Important non-conclusion

This is **not** a whole-row reproduction of the July 15 result: current decode
is 44.0 tok/s rather than 31.9 tok/s. The historical Vulkan executable was
overwritten and the source edit after that run was not tracked, so the precise
code/toolchain cause cannot be reconstructed. Do not combine the old decode,
energy, or thermal figures with this binary. Future comparisons require the
binary SHA-256 and a dated source-change entry.

The v3 protocol replica further shows that, even with matching visible policy
flags and prompt hash, current vendor-GPU governor behavior can dominate decode:
it delivered 27.0 tok/s while prefill still ran fully at 1200 MHz. This is a
captured environmental result, not evidence that the policy changed.
