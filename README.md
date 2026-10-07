# μKV: energy-aware KV cache eviction on a phone

Research code behind μKV, a KV cache eviction policy and an energy-aware controller for LLM
inference on a OnePlus 15 (Snapdragon 8 Elite Gen 5, Adreno 840 GPU).
Md Romyull Islam, Kennesaw State University.

On-device engines store the KV cache as one array of cells shared by every head and layer,
and they run attention in a fused kernel. Eviction policies built for servers clash with both:
a per-head budget frees a cell only when every head drops it, and a policy that reads attention
scores must leave the fused kernel. μKV scores inside the fused prefill graph, keeps one set of
positions for every head and layer, and compacts the survivors in place. Around it, a controller
measures what each GPU clock, cache budget and answer length costs, lets the battery level
choose among them, and corrects itself with two feedback loops. A reduce-only watchdog lowers
clock caps before the vendor's thermal throttle does.

## HotMobile 2027 paper

"μKV: Bridging Logical KV Eviction and Physical Efficiency on Mobile LLMs" covers the eviction
policy only: prefill scoring, one keep set for all heads and layers, and in-place compaction,
measured against six published eviction policies. It does not use the controller, the watchdog
or `energy_rl/`. The parts of this repository it uses are:

| Paper | Code |
|---|---|
| Stage 1, prefill scoring | the `kq_evict` side node in `llama.cpp/src/llama-graph.cpp` |
| Stage 2, one keep set | policy `v1_fa2` in `entropy_probe/eviction_bench.cpp` |
| Stage 3, in-place compaction | `endurkv_compact_seq()` in `llama.cpp/src/llama-kv-cache.cpp` |
| Baselines | H2O, TOVA, SnapKV, Ada-KV, KeyDiff and StreamingLLM in `entropy_probe/eviction_bench.cpp` |
| Needle test | `scripts/android/run_niah_current_build.sh`, scored by `eval_pipeline/score_niah_strict.py` |
| Phone LongBench | `scripts/android/run_longbench_native_budgets.sh`, scored by `scripts/longbench_score.py` |
| Budget sweep (Figure 1) | `figures/master_tables/figures/fig_why_col.py` |

The exact μKV flags are under "μKV as run in the paper" below.

## Repository layout

| Path | What it holds |
|---|---|
| `llama.cpp/` | llama.cpp fork. The `kq_evict` side node in the prefill graph (`src/llama-graph.cpp`) and in-place compaction, `endurkv_compact_seq()` (`src/llama-kv-cache.cpp`, `include/llama.h`). |
| `entropy_probe/eviction_bench.cpp` | Benchmark driver. μKV, H2O, TOVA, SnapKV, Ada-KV, StreamingLLM and KeyDiff behind one interface, with decode-time trimming and cache accounting. |
| `scripts/android/ukv_sched.sh` | Per-request scheduler: battery tier, lever, cost-table walk and the two feedback loops. The cost table is `ukv_sched_table.txt`. |
| `scripts/android/preempt_throttle_watchdog_v2.sh`, `gpu_watchdog.sh` | Reduce-only CPU and GPU clock ladders. |
| `scripts/android/sample_sensors.sh`, `cool_gate.sh` | Sensor and power sampling at 2 to 5 Hz, and the cooling gate run before every timed request. |
| `scripts/android/run_*` | The measurement campaigns behind the tables and figures. |
| `scripts/longbench_score.py`, `eval_pipeline/score_niah_strict.py` | LongBench token-F1 and the needle-in-a-haystack scorer. |
| `energy_rl/` | Contextual bandit and simulator experiments for the learner. |
| `figures/master_tables/` | `claims_data.py` and `energy_perf_data.py` collect the measured values behind the paper's tables; `figures/fig_why_col.py`, `fig_three_relations_col.py` and `fig_thermal_col.py` draw its data figures. Each script names its data sources in its header. |
| `figures/fig_discharge_timeline.py` | Rebuilds the energy of every request in the battery discharge from its raw sensor trace. |

Further notes: `scripts/android/README.md` (the on-phone setup), `entropy_probe/PLATFORMS.md`
(build and run settings per device) and `docs/HARDWARE_STRESS_LIMITS.md`.

## Build for Android (arm64)

Host: Android NDK r27c, CMake and Ninja on the path, and `adb` to a rooted phone.

```
scripts/android/build_llama_android.sh          # llama.cpp, CPU
scripts/android/build_llama_android_vulkan.sh   # llama.cpp, Adreno GPU through Vulkan
scripts/android/build_probe_android.sh          # eviction_bench and the probes
```

Build the CPU libraries with `-march=armv8.7-a` in the C and C++ flags; without it the
dot-product and int8 matrix-multiply kernels are left out.

Models are not in the repository. The paper uses Llama-3.2-1B-Instruct, Phi-3-mini-128k-instruct
and gemma-2-2b-it in Q4_K_M, and the 1-bit Bonsai-8B.

## μKV as run in the paper

The policy's internal name in `eviction_bench` is `v1_fa2`. The flags the scheduler passes are:

```
--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 \
--obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024
```

That is 4 sink tokens, scores from the last 16 prompt queries max-pooled over 7 positions,
an anchor share floored at 0.70, and compaction in place at a budget of 1024 cells.
`scripts/android/ukv_sched.sh` builds the full command line for each request, including the
model, context size, clocks and answer cap.

## Data

Raw sensor traces and per-run outputs are not in this repository.

`llama.cpp/` is upstream llama.cpp under its MIT license (`llama.cpp/LICENSE`) with the changes
listed above.
