# μKV platform profiles — build & run settings per device class

**Rule: the μKV POLICY config is frozen and identical on every platform**
(see `MUKV_GPU_VERSION.md`). Only the PLATFORM profile below changes.
Never mix profiles: an Adreno workaround on CUDA (or vice versa) silently
distorts results (measured: wrong n-ubatch alone is a >20× prefill difference).

## Frozen μKV policy flags (ALL platforms, verbatim)
```
--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 \
--obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --k-nominal 1024 \
--cache-type-k f16 --cache-type-v f16
```
Correct-run tells (stderr): `[fa-on-evict] capture frozen`, `[cache] retained_kv=…
(TRUE compacted)`, and NO `[snapkv] swap done` when n_gpu_layers != 0.

## Profile A — Android phone, Adreno GPU (Vulkan)
| setting | value | why |
|---|---|---|
| build | `bash entropy_probe/build_android_vulkan.sh` | NDK r27c, `-march=armv8.7-a` (dotprod+i8mm), GGML_VULKAN=ON |
| deploy | push **binary AND .so libs**, never binary alone | stale libs = silently different backend |
| GPU binary | **`bin_vulkan_new/eviction_bench`** + libs from **`bin_vulkan/`** | CORRECTED 2026-07-30: `bin_vulkan/eviction_bench` is a 2026-05-31 build that rejects every current flag (`unknown arg: --ignore-eos`); its usage banner still lists `--policy {vanilla|tova|pyramid|v1}`. Following the old wording here silently produced 12 empty cells. `bin_vulkan_new` ships no `.so` of its own, hence the split. |
| `--n-ubatch` | **64** | Adreno TDR/DeviceLost workaround — larger ubatches hang the driver |
| `--n-batch` | 512 | |
| `--threads` | 4 | big cores only |
| protocol | native DVFS, NO periodic clock writes; **energy/time cool gate below** | periodic `gpu_max_clock` writes cost ~10% prefill (measured 07-24) |

### Energy/time cooling gate (MANDATORY before every timed or energy-measured phone run)
Cool the phone **before each cell** until **DDR ≤ 36 °C** (and battery ≤ 34 °C). **Keep
charging OFF during cooling** — the charge current is itself a heat source; if DDR is not
dropping, charging-off is the lever that makes it fall. Restore charging only after the run.
This is required so every cell starts from the same thermal state (energy and wall are
otherwise contaminated by residual heat). Canonical helper: `scripts/android/cool_gate.sh`
(sourced by all phone run scripts).
```sh
cool_ddr36() {   # POSIX sh; run under su
  echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable   # charging OFF (heat source)
  DZ=; BZ=
  for z in /sys/class/thermal/thermal_zone*; do t=$(cat $z/type)
    [ "$t" = ddr ] && DZ=$z/temp; [ "$t" = battery ] && BZ=$z/temp; done
  while :; do dd=$(( $(cat $DZ)/1000 )); b=$(( $(cat $BZ)/1000 ))
    [ $dd -le 36 ] && [ $b -le 34 ] && { echo "cool ddr=$dd batt=$b"; break; }
    sleep 10   # charging stays OFF the whole time it is not cool
  done
}                # caller restores: echo 1 > .../mmi_charging_enable  AFTER the timed run
| verify | `scripts/android/assert_binary_current.sh` (SHA-256 host↔device) | |

## Profile B — NVIDIA Jetson Orin (CUDA)
| setting | value | why |
|---|---|---|
| build | `bash entropy_probe/build_jetson_cuda.sh` | CUDA 12.6, SM_87, `-march=native` |
| deploy | build natively on device (`~/ukv/code`), no push step | |
| `--n-ubatch` | **512** | no TDR issue; large ubatch = full GPU utilization (5.1 s vs 117 s prefill on 1B/12K) |
| `--n-batch` | 512 | |
| `--threads` | 8 | all Orin cores (CPU-fallback models need them) |
| protocol | default clocks (optionally `jetson_clocks` for pinned runs — note it in the results dir) | |
| memory | unified: read `peak_rss` (meta.json) AND global used-RAM delta (`free -m` sampling) | CUDA buffers may not appear in RSS |

## Shared limits (both GPUs)
- **Q1_0 (1-bit) has no GPU kernel on either backend** — those tensors fall back to
  CPU regardless of `-ngl` (Vulkan: `cannot be used with preferred buffer type`;
  CUDA: `CPU_Mapped model buffer`). Memory columns stay valid; decode is CPU-bound.
- GPU defrag is intentionally absent: with `n_gpu_layers != 0` the defrag gate
  (eviction_bench.cpp ~line 2630) skips the state-swap → logical eviction only.
- **Single-allocation limit differs by platform (measured 2026-07-25, NOT universal):**
  Jetson Tegra rejects any single GPU buffer > ~4 GiB (`cudaMalloc OOM`) — so Phi-3-mini
  f16 KV @16K (6144 MiB) cannot init; must use q8_0 KV (3264 MiB). The **Adreno 840
  (OnePlus 15) has no such cap** — it allocated the full 6144 MiB f16 KV buffer and ran
  (prefill 583 s, 2.0 tps). Do not assume the phone inherits the Tegra limit.


## GPU PROFILE MATRIX (2026-08-01) — the settings that MUST differ per backend

Post-eviction **compaction** (a `llama_state_seq_get/set` round-trip that closes the holes
`seq_rm` leaves) is the setting that differs most, and it is now chosen automatically by
`detect_gpu_profile_wants_defrag()` from the active backend name.

| setting | Adreno / Vulkan / Metal | CUDA / ROCm / SYCL | CPU |
|---|---|---|---|
| compaction (defrag) | **OFF** | **ON** | ON |
| `--n-ubatch` | **64** (Adreno TDR workaround) | 512 | 512 |
| `--threads` | 4 (big cores) | 8–16 | 6 |
| binary | `bin_vulkan_new` + libs from `bin_vulkan` | `build-pc-cuda` / `build-jetson-cuda` | `bin_cpu_v87` |

**Why compaction differs — measured, not assumed.** On an RTX 4500 Ada (Phi-3-mini Q4_K_M,
11K prompt, batch 1, idle GPU):

| policy | tps | vs vanilla |
|---|---:|---:|
| vanilla (full cache) | 50.13 | — |
| muKV, compaction OFF | 51.49 | 1.03x |
| **muKV, compaction ON** | **121.54** | **2.42x** |
| SnapKV (paper NiaH cfg: window 16, kernel 5) | 15.22 | 0.30x |

Without compaction the cache is *logically* evicted but *physically* sparse, so decode still
walks the full prompt span: a 12.7x smaller cache buys nothing. The gate used to be
`n_gpu_layers == 0` (CPU only), which silently suppressed this 2.42x on every discrete GPU.
It was written from an Adreno measurement and generalised to all GPUs.

On Vulkan/Adreno the round-trip crosses a driver-managed buffer and was measured prohibitive,
so it stays OFF there **pending re-measurement** — the same generalisation that cost us the
CUDA result, so treat the Adreno "prohibitive" finding as provisional too.

Overrides: `--force-defrag` / `--no-defrag`. Unknown backends fall back to the conservative
pre-2026-08 behaviour (compaction only when `n_gpu_layers == 0`).

**SnapKV baseline configurability (2026-08-01).** `--obs-window` was never passed to
`policy_snapkv()`; the call took the C++ defaults (window 64, kernel 5 = FasterDecoding repo
default), so the flag was inert. Now wired, plus `--snapkv-kernel`. The SnapKV paper retunes
per experiment: **NiaH window 16 / kernel 5**, LongBench window 32 / kernel 7, Command-R
window 64 / kernel 13. Run the baseline at the setting matching the benchmark.

## GitHub release checklist (so anyone can install on either device)
1. `git init` / track `entropy_probe/` (source is currently untracked — see CHANGELOG).
2. Ship both build scripts + this file + `MUKV_GPU_VERSION.md` (frozen policy) + `CHANGELOG.md`.
3. Tag the release with the eviction_bench source SHA; publish per-platform binary SHA-256s.
4. README quickstart = Profile A block for phones, Profile B block for Jetson, one
   copy-paste run command each (the templates in `MUKV_GPU_VERSION.md`).

## Robustness invariant (standing rule, user-stated 2026-07-25)
**If vanilla runs a configuration, μKV MUST run it too.** μKV allocates the same KV
buffer as vanilla and then *removes* cells — it can never legitimately require more
memory or more capability. Therefore any config where vanilla succeeds and μKV fails
is a **μKV bug**, not a resource limit, and must be root-caused rather than worked around.

Violations found by this rule (both were real bugs, both fixed — see CHANGELOG):
| symptom | root cause | fix |
|---|---|---|
| Phi-3 q8 KV: vanilla allocated 3264 MiB, μKV died needing 4704 MiB | stale force of V to f16 for all non-vanilla policies, obsolete under fa-on-evict (FA-on supports quantized V) | gate the force on `!fa_on_evict` (2026-07-24) |
| VL long decode: vanilla completed, μKV aborted | `llama_memory_seq_add()` asserts `n_pos_per_embd()==1`; M-RoPE has 4 | skip position compaction on M-RoPE caches (2026-07-25) |

Run this as a smoke check whenever a new model/quant/modality is added: same command,
`--policy vanilla` then the frozen μKV flags. μKV failing alone = stop and debug.

## MANDATORY: output-validity check before any new-backend measurement
Mechanism checks (weights on the device, no DeviceLost, plausible tps) do **NOT** prove the
computation is correct. Greedy decoding emits tokens at full speed even from NaN logits, so a
broken kernel looks like a fast one. Before recording ANY performance number on a new
backend/kernel/quant:
```sh
# same binary, same prompt, only -ngl differs; CPU is the reference
eviction_bench ... --n-gpu-layers 99 --max-tokens 24 --greedy --out-gen /tmp/g.txt
eviction_bench ... --n-gpu-layers 0  --max-tokens 24 --greedy --out-gen /tmp/c.txt
diff /tmp/g.txt /tmp/c.txt   # must be identical (greedy ⇒ deterministic)
```
Also run one teacher-forced PPL cell: `mean_nll=nan` / `ppl=nan` means the logits are
non-finite and every timing from that backend is meaningless.
Caught 2026-07-26: Vulkan Q1_0 on Adreno passed every mechanism check and was numerically
broken; CUDA Q1_0 passed both mechanism and validity.

## Watchdog versions — use the FIXED ones only
| script | date | zone resolution | use |
|---|---|---|---|
| `preempt_throttle_watchdog_v2.sh` | 07-19 | **by name (fixed)** | **CPU runs — current** |
| `gpu_watchdog_v5_real.sh` | 07-20 | **by name (fixed)** | **GPU runs — current** |
| `gpu_watchdog_v4_surface_aware.sh` | 07-16 | hardcoded zone ids (BUGGY) | do not use |
Both use the SAME battery ladder (47.0/48.0/48.5/49.0/49.5 C); only the clock domain differs
(CPU 1497/1382/1267/1132/1017 MHz, GPU 1050/967/902/826 MHz). The watchdog runs ONLY under
muKV. NOTE: v2 header comment (35/35.5/36 C) is STALE - the active code at line 280 uses the
47 C ladder, which is what the paper describes.
The 07-18 bug: v4 hardcodes BATZ=93 (line 22). On the CURRENT boot zone93 IS the battery
(verified: type=battery 34 C; zone94=usb 3 C), so the index happens to be right today. But
zone numbering is not stable across boots/kernels, and it WAS wrong when the bug was found,
leaving the
watchdog inert. Any result produced with a hardcoded-zone watchdog is therefore unverifiable after the fact - it is an
efficiency result, not a watchdog result. The paper's GPU table row "μKV + wd v4" must be
re-measured with v5 before it is described as a watchdog effect.
