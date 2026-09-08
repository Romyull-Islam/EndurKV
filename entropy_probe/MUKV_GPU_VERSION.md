# Canonical μKV GPU version — frozen spec (for exact reproduction)

**Do not run a μKV GPU experiment whose config/binary does not match this.**
Verify with: `bash scripts/android/assert_binary_current.sh entropy_probe/build-android-vulkan <device-bin>`

## Version label
**μKV-mass-sol2(FA-on) + logical eviction, NO GPU defrag**  (GPU, full offload)
— FA-on prefill+decode; last-chunk scoring (prefill ≈ vanilla); `seq_rm` logical
eviction; the state-round-trip defrag is **skipped** on GPU (`n_gpu_layers != 0`).
(Contrast: the CPU version, `n_gpu_layers=0`, additionally runs the CPU defrag.)

## Exact command (Llama-1B & Bonsai-8B GPU, full offload)
```
LD_LIBRARY_PATH=<bin_vulkan_libs> eviction_bench \
  --policy v1_fa2 --fa-on-evict \
  --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 \
  --snapkv-pool 7 --gate-alpha-floor 0.70 \
  --k-nominal 1024 --cache-type-k f16 --cache-type-v f16 \
  --n-gpu-layers 99 --threads 4 --ctx-size 16384 --n-batch 512 --n-ubatch 64 \
  --eval-mode gen --max-tokens 4096 --ignore-eos --seed 42 --greedy
```
Effective knobs (from run log): `K=1024, n_sink=4, R_min=32, obs_window=16,
snapkv_pool=7, gate α-floor=0.70, gate=mass+floor (α_a≈0.71), FA=ON`.

## Binary provenance
- Path: `entropy_probe/build-android-vulkan/eviction_bench`
- **Current phone SHA-256 (07-25, post V-gate fix, f16-identical): `e60671184f901a0e…`**
- Reference (07-21) SHA-256: `e97a23f8f293a0ec29e12cb06e9ec5a7b976c79d1eb6d1e69bbbfac988e002a2`
- **Definitive n=3 reproduction (07-25, strict settle-to-stability, native DVFS):**
  vanilla wall 256/255/256 s (mean 255.7), prefill 114.6 s, 29.3 tps;
  **μKV wall 240/240/239 s (mean 239.7), prefill 115.8 s, 33.4 tps** — matches the
  published table within ~2%, μKV wins wall+decode in every rep, spread <1%.
- Built: **2026-07-21 07:50**  from source `eviction_bench.cpp` (mtime **2026-07-17 20:58**)
- Build: `cmake --build entropy_probe/build-android-vulkan -j8 --target eviction_bench` (NDK r27c, arm64-v8a, GGML_VULKAN=ON)

## Reference result (Llama-1B, held 1200 MHz, cold, same-run vanilla control)
| | Stock llama.cpp | μKV |
|---|---:|---:|
| Prefill | 114.91 s | 117.27 s (parity) |
| Decode | 32.55 tok/s | 33.94 tok/s |
| Total | 242.83 s | 238.05 s |
Decode is DVFS-variable (~27–44 tok/s across runs); μKV-vs-vanilla **same-run**
comparison is the robust invariant. retained_kv ≈ 22.6 MiB.

## MEASUREMENT PROTOCOL — required to reproduce the table numbers
The version is **binary + config + protocol**. The table (`tab:gpu-wikitext`) is
**native DVFS**: the GPU governor and charging state are NOT touched during the run.
1. **NO clock-keeper.** Do not write `gpu_max_clock` periodically during a run —
   each write forces a devfreq transition and costs **~10–13% prefill**
   (measured A/B 2026-07-24, same binary/workload/day: μKV prefill 133.16 s with
   a 2-s keeper vs **117.38 s** hands-off). A single write to release a stale
   726-cap *before* cooling is fine; then hands off.
2. **Cool gate before each cell:** battery < 34 °C AND ddr < 36 °C (charging held
   OFF while cooling); the original sweeps used GPUSS ≤ 37/DDR ≤ 37 settle-to-stability.
3. Decode tps must be measured over the full 4096-token decode. Short decodes
   read high (256-token early decode measured 45 tok/s; the table's ~33 is the
   4096-token mean). Table's 34.6 tps row is μKV **+ watchdog v4**; without the
   watchdog the correct comparison row is 33.0.

## Re-confirmation runs (2026-07-24, same binary SHA e97a23f8…)
Two A/B passes, same workload (9737-token WikiText prompt, identical flags):
| protocol | vanilla prefill | μKV prefill | note |
|---|---:|---:|---|
| 2-s clock-keeper (WRONG protocol) | 128.25 s | 133.16 s | keeper's devfreq writes = ~10% tax on both |
| **native DVFS (table protocol)** | 121.67 s | **117.38 s** | **μKV = table's 117 ✓ reproduced** |
- **retained_kv = 22.60 MiB — EXACT match** in every pass (logical eviction intact).
- Config echo exact every pass: `anchor=723 recent=297, K=1024, n_sink=4, R_min=32,
  gate=mass+floor α=0.709, FA=ON`, `[fa-on-evict] capture frozen`, no `[snapkv] swap done`.
- 4096-token decode (keeper pass): vanilla 33.25 / μKV 33.71 tok/s — matches the
  table's no-watchdog row (33.0). Version = INTACT; the earlier 128/133 readings
  were a **protocol** deviation (keeper), not a version change.
