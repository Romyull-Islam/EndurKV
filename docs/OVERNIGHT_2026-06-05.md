# Overnight progress — June 5 2026

Working overnight (~02:00 → 09:00 EDT) while user sleeps. This file is the
hand-off summary; everything described here happened with continuous on-phone
work.

## Headline result

**v1_FA implemented and end-to-end working.** v1's per-head spread-gate
eviction now runs with Flash Attention enabled during decode (the thermal-
dominant phase), via a state-swap from an FA-off prefill context to an FA-on
decode context. This required:

1. **A llama.cpp patch** to allow cross-`v_trans` state loading (FA-off writes
   transposed V; FA-on reads untransposed V — they were incompatible and the
   load was hard-rejected). Patch transposes V element-wise on read.
2. **A `v1_fa` policy alias** in `entropy_probe/eviction_bench.cpp` that auto-
   enables `--snapkv-decode` + `--no-evict-decode`.
3. **A warm-up re-decode** of the last prompt token after the swap, so the
   FA-on context's output buffer is populated and `llama_get_logits_ith` doesn't
   crash with `n_outputs=0`.

End-to-end test on Gemma-2B, 3143-token prompt, `--max-tokens 16`:
- `state_read_data: cross-v_trans load (src=transposed, dst=untransposed); transposing V on read` (twice — one per stream)
- `[snapkv] swap done in 803.3ms`
- `[snapkv] warm-up decode ok; n_outputs populated`
- `policy=v1_fa K=512 prompt=gov_report_pub_001 n_prompt=3143 steps=16 decode_tps=6.0`
- Generation: `"This report from the Government Accountability Office (GAO) examines the Coast Guard'..."`

Quality metrics from `meta.json`:
- `mean_nll` = 0.324, `perplexity` = 1.383 (low → coherent)
- `peak_kv_cells` = 3158 = 3143 prompt + 15 decode (231 evicted from prompt during prefill phase)
- `evicted_prefill` = 231 (v1's spread-gate eviction applied successfully)
- `evicted_total_decode` = 0 (FA-on decode runs with frozen mask, as expected)
- `prefill_ms` = 240 s (FA-off prefill, dominated by the 3K-token prompt processing)
- `decode_tps` = 6.0 tok/s (FA-on decode)

## Code changes

### `EndurKV/llama.cpp/src/llama-kv-cache.cpp` (lines ~2193–2249)

Replaced the hard `incompatible V transposition` rejection with:
```cpp
const bool cross_v_trans = (this->v_trans != (bool) v_trans);
if (cross_v_trans && (this->v_trans || !(bool) v_trans)) {
    // only FA-off (src trans) -> FA-on (dst untrans) supported
    LLAMA_LOG_ERROR(...);
    return false;
}
```
plus a new `if (cross_v_trans) { ... }` branch in the V read loop that:
- reads the transposed source format (per-embedding-dim slabs of `cell_count` elements)
- transposes element-by-element into a host `std::vector<uint8_t>` in cell-major layout
- calls `ggml_backend_tensor_set` on the untransposed V cache with the assembled buffer

### `EndurKV/entropy_probe/eviction_bench.cpp`

Three edits:
- **arg parse** (line ~160): added `"v1_fa"` to valid policies; auto-sets
  `snapkv_decode = true`, `no_evict_decode = true`.
- **prefill eviction** (line ~706): `v1_fa` shares the v1 spread-gate path
  (`policy_v1(cap, args.k_nominal, n_kv_heads)`).
- **post-swap warm-up** (line ~770): after `[snapkv] swap done`, removes the
  last prompt position from KV and re-decodes that one token so the FA-on
  context has fresh logits before the sampling loop starts.

### Build + push

```
ANDROID_NDK=/home/mislam22/tools/ndk/android-ndk-r27c
cd EndurKV/llama.cpp/build-android   && cmake --build . --target llama
cd EndurKV/entropy_probe/build-android && cmake --build . --target eviction_bench
adb push libllama.so eviction_bench   /data/local/tmp/endurkv/bin_cpu/
```

## Phone state on root

Phone was wiped during the Magisk root (this is normal). Re-pushed:

| Asset | Source on host |
|---|---|
| `bin_cpu/` (post-patch binaries) | rebuilt + pushed |
| `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` | re-downloaded from HuggingFace (was on phone only, never on host — I missed this in pre-root reconfirmation) |
| `models/gemma-2-2b-it-Q4_K_M.gguf` | host `models/` |
| `models/Phi-3-mini-128k-instruct-Q4_K_M.gguf` | host `models/` |
| `prompts/`, `prompts_chat/`, `corpora/` | from `phone-logs/prompts_from_phone` + workspace |
| `scripts/`, `sensor_map.txt` | repo + pull backup |

Verified rooted access:
- `su -c id` → `uid=0(root) gid=0(root) groups=0(root) context=u:r:magisk:s0`
- `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` writable as root
- DDR temp `thermal_zone34/temp` readable

## Experiments launched

### KV-growth thermal study (foundational)

`/tmp/run_kvgrow_v4.sh`, output `phone-logs/kvgrow_$TS/`.

4 conditions × Llama-3.2-1B × `narrativeqa_pub_001` (~7700 prompt + 512 decode):
- **A**: vanilla — KV grows unbounded → upper bound for thermal vs KV
- **B**: v1 K=2048 → mid plateau
- **C**: v1 K=1024 → lower plateau
- **D**: v1 K=256 → floor

Sensor sampler 5 Hz throughout each condition; cool-down between.

Analysis script: `EndurKV/scripts/android/host_kv_growth_plot.py <kvgrow_dir>`
→ `kv_growth_temperature.png` (DDR temp vs time, KV cells overlay).

### v1 vs v1_fa thermal A/B (script ready, awaits KV-growth completion)

`EndurKV/scripts/android/phone_v1_vs_v1fa_thermal.sh`.

2 cells × 3 iterations of narrativeqa prefill + 256-token decode:
- **v1, K=512** (FA-off throughout)
- **v1_fa, K=512** (FA-off prefill + FA-on decode)

Run when KV-growth completes:
```
ITERS=3 ./EndurKV/scripts/android/phone_v1_vs_v1fa_thermal.sh
```

## Known issues / things to verify in the morning

1. **v1_fa generation quality vs v1**: end-to-end test produced coherent
   English but not directly compared. Recommended sanity check: run both
   policies with identical seed/prompt and `--out-gen`, eyeball the outputs.
2. **v1_fa prefill is still slow** (~240 s for 3K tokens on Gemma-2B CPU). FA-off
   prefill dominates; FA-on decode is fast (~6 tok/s after warm-up). The
   thermal claim is about the decode phase — that's the heat-dominant phase
   in real workloads.
3. **Phi-3-128k pushed but not tested** with v1_fa yet (Phi-3 has different
   model arch; should "just work" but worth verifying).
4. **The llama.cpp patch is one-way** (FA-off → FA-on). The reverse direction
   is rejected. We never need FA-on → FA-off in this codebase but if you did,
   you'd need a symmetric branch.

## Reference: the actual files I touched

| Path | Change |
|---|---|
| `EndurKV/llama.cpp/src/llama-kv-cache.cpp` | added cross-v_trans branch in `state_read_data`, added `#include <vector>` |
| `EndurKV/entropy_probe/eviction_bench.cpp` | v1_fa policy + warm-up decode |
| `EndurKV/scripts/android/host_kv_growth_plot.py` | **new** — KV growth analysis |
| `EndurKV/scripts/android/phone_v1_vs_v1fa_thermal.sh` | **new** — thermal A/B |
| `EndurKV/docs/OVERNIGHT_2026-06-05.md` | **this file** |
| `/tmp/run_kvgrow_v4.sh` | KV-growth launcher (running tonight) |

Phone-side binaries that produced all data tonight:
- `bin_cpu/eviction_bench` (rebuilt with v1_fa)
- `bin_cpu/libllama.so` (rebuilt with v_trans patch)

Snapshot of the pre-patch binaries is at `phone-logs/bin_cpu_snapshot/` — kept
for bit-exact reproducibility of the canonical 4-policy Wave-1-redux results.
