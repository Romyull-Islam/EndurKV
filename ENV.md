# Environment — EndurKV server-side measurement study

Captured: 2026-04-26.

## Host

| | |
|---|---|
| Hostname | `a100` |
| OS | Ubuntu 24.04 |
| Kernel | `6.8.0-101-generic` |
| CPU | AMD EPYC 7742 64-Core, 256 logical cores |
| Memory | 2.0 TiB |
| GPU | 8× NVIDIA A100-SXM4-80GB |
| FS / free | NFS `172.16.3.1:/mnt/bst/a100/ksuo`, 3.9 TB free |

## Toolchain

| | |
|---|---|
| gcc / g++ | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| cmake | 3.28.3 |
| make | GNU Make 4.3 |
| git | 2.43.0 |
| pkg-config | 1.8.1 |
| nvcc | CUDA 13.1.80 |

## Python

System interpreter at `/usr/bin/python3`, Python 3.12.3, pip 24.0. Project-local
venv (with `--system-site-packages`) at `./.venv/`.

Packages required (managed in venv):

| Package | Purpose |
|---|---|
| numpy, pandas, scipy | analysis |
| matplotlib, seaborn | plotting |
| tqdm | progress bars |
| datasets, huggingface_hub | TriviaQA fetch + model download |

## llama.cpp

| | |
|---|---|
| Path | `./llama.cpp/` |
| Remote | `https://github.com/ggerganov/llama.cpp` (origin) |
| Pinned commit | `0033f53a072af953b457c9fd2314e6e28bd11cc7` |
| Build tag (from `llama-cli`) | `b8684-0033f53a0` |
| Branch | `master` |
| Subject | `docs: fix typo in build.md (emdawbwebgpu -> emdawnwebgpu) (#21518)` |
| Build dir | `./llama.cpp/build/` |
| Build flavor | **CUDA-enabled** (`libggml-cuda.so` linked into `llama-cli`); generation observed at ~199 t/s on Llama-3.2-1B-Q4_K_M |
| CMake export | `./llama.cpp/build/llama-config.cmake` (consumed via `find_package(Llama CONFIG)`) |
| Library ABI | `libllama.so.0.0.8684`, `libggml.so.0.9.11` |

**Local modifications carried by the user (NOT vanilla upstream):**

```
M  tools/cli/cli.cpp           (+27 lines, user's thermokv-related changes)
?? tools/cli/thermokv_mock.h
?? exact_prompt.txt
?? heavy_textbook.txt
?? huge_textbook.txt
```

The entropy-probe added in Phase 1 lives in `llama.cpp/examples/entropy-probe/`
(new directory, no edits to existing source files). To reproduce exactly,
re-clone llama.cpp, check out `0033f53a072af953b457c9fd2314e6e28bd11cc7`, apply
the user's local diff to `tools/cli/cli.cpp`, and copy `examples/entropy-probe/`
in.

## Model

| | |
|---|---|
| Name | Llama 3.2 1B Instruct, Q4_K_M GGUF |
| Path | `./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` |
| Source | `bartowski/Llama-3.2-1B-Instruct-GGUF` on Hugging Face |
| Size | 771 MB (807,694,464 bytes) |
| sha256 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` |

## Reproducibility

- Random seeds are fixed per-prompt in `scripts/run_study.sh` (`prompt_id` → seed).
- Sampling temperature is set to `1.0` for the study (raw distribution).
- All study runs append to `./logs/` with a timestamp; full study output is
  concatenated into `./logs/study_full.csv` and metadata is recorded in
  `./logs/study_meta.json`.
