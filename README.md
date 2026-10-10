# μKV: KV cache eviction that pays off on a phone

Research code for μKV, a KV cache eviction policy for LLM inference on a phone
(OnePlus 15: Snapdragon 8 Elite Gen 5, Adreno 840 GPU), built on a fork of llama.cpp.
Md Romyull Islam, Kennesaw State University.

On-device engines keep the KV cache as one array of cells, one cell per token position, shared by
every head and layer, and they run attention in a fused kernel. Eviction policies built for servers
clash with both: a per-head choice frees a cell only when every head drops it, and a policy that
reads attention scores must leave the fused kernel. μKV scores the prompt once during prefill
beside the fused kernel, keeps one set of positions for every head and layer, and compacts the
kept cells in place so decode stays fused over a dense block.

## The HotMobile 2027 paper

"μKV: Bridging Logical KV Eviction and Physical Efficiency on Mobile LLMs" uses only μKV and six
published eviction policies. **The energy-aware controller (`scripts/android/ukv_sched.sh`), the
thermal watchdogs and `energy_rl/` are separate, ongoing work and are not part of this paper.**

### Where μKV lives

| μKV stage (paper Section 3) | Code |
|---|---|
| Stage 1, prefill-only scoring | the `kq_evict` side node in `llama.cpp/src/llama-graph.cpp`, switched on with `llama_endurkv_set_evict_obs_window()` |
| Stage 2, one keep set for all heads and layers | policy `v1_fa2` in `entropy_probe/eviction_bench.cpp` |
| Stage 3, in-place compaction | `endurkv_compact_seq()` in `llama.cpp/src/llama-kv-cache.cpp` |

`entropy_probe/eviction_bench.cpp` is the benchmark driver. It runs μKV and every baseline behind
one interface and reports speed, prefill time and the cells left live after prefill.

### Running μKV and the baselines

All runs use `eviction_bench` with `--eval-mode gen --ctx-size 16384 --seed 42 --greedy`.
The policy flags are:

| Policy | Flags |
|---|---|
| μKV (K = 1024) | `--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024` |
| full cache | `--policy vanilla` |
| StreamingLLM, as published | `--policy streamingllm --n-sink 4 --k-nominal 2004 --compact-inplace --sllm-window` |
| SnapKV | `--policy snapkv` with `--k-nominal` per head; add `--fa-on-evict --no-evict-decode --compact-inplace` for the side-node variant that keeps fused attention |
| Ada-KV | `--policy adakv`; same side-node flags as SnapKV |
| H2O, TOVA | `--policy h2o`, `--policy tova` |
| KeyDiff | `--policy keydiff --n-sink 0 --k-nominal 2048 --compact-inplace --keydiff-decode-block 128` |

`--sllm-window` gives StreamingLLM's published behaviour: 4 sink tokens and a rolling window of
2000 recent tokens, trimmed at every decode step, with positions assigned inside the cache, so
every cached key is re-rotated at each step. The flag also makes that re-rotation a pure
rotation (`llama_endurkv_set_pure_kshift()`): by default llama.cpp's K-shift applies the RoPE
magnitude factor again (YaRN for Bonsai, the LongRoPE attention factor for Phi-3), which compounds
at every step and corrupts the output. Without it, `--policy streamingllm` keeps the window
chosen after prefill and does not trim during decode. The paper uses that untrimmed version only
as a control for fused attention ("Window 4+2000, kernel off").

### Where each result comes from

Scripts are in `scripts/android/` unless a path says otherwise. Each campaign script runs on a
rooted phone over adb and writes its results to the host.

| Paper item | Campaign script(s) | Scored or drawn by |
|---|---|---|
| Table 1, Llama: full cache, μKV | `run_streamingllm_faithful.sh` | `figures/master_tables/claims_data.py` |
| Table 1, Llama: SnapKV, H2O, TOVA, Ada-KV | `run_phone_gpu_16k_wikitext.sh`, `run_phone_gpu_complete.sh` (round 1); `phone_queue/queue_gpu.sh` (rounds 2 to 4) | `claims_data.py` |
| Table 1, μKV not compacted | `run_phone_gpu_16k_wikitext.sh`, `run_phone_gpu_repeats.sh` | `claims_data.py` |
| Table 1, Window 4+2000 kernel off | `phone_queue/queue_gpu.sh` | `claims_data.py` |
| Table 1, StreamingLLM (published) | `run_sllm_window_gpu.sh`, `phone_queue_runner.sh` | ratio to the Table 1 full cache |
| Table 1, Phi-3-mini block | `run_phi3_gpu_complete.sh` (round 1); `phone_queue/phi3_chain.sh`, `run_keydiff_phi3_gpu_n3.sh` (rounds 2 and 3) | `claims_data.py` |
| Table 1 text, Llama side-node SnapKV and Ada-KV | `run_llama_sidenode_gpu.sh` | ratio to its own full cache |
| Table 2, Llama | `run_natural_cpu.sh`; `phone_queue/cpu_sllm_published.sh` (rows marked †); `run_n3_headline_v2.sh` (KeyDiff) | ratio to the campaign's full cache |
| Table 2, Bonsai-8B | `run_natural_bonsai.sh`; `phone_queue/cpu_sllm_published.sh` (†); `run_gap_closure.sh` (KeyDiff) | same |
| Table 2, Phi-3-mini | `run_phi3_cpu_complete.sh` | same |
| Table 2, Gemma-2B | `run_gap_closure.sh` (section F) | same |
| Table 2, StreamingLLM (published) | `phone_queue_runner.sh` (cells `cpu_*_sllmw`) | ratio to each column's original full cache |
| Table 3, needle test | `run_niah_tableC.sh` (full cache, μKV, SnapKV); `run_niah_published_budgets.sh` (Ada-KV, TOVA, H2O); `run_niah_keydiff.sh`; `run_niah_sllmw.sh` (StreamingLLM) | `eval_pipeline/score_niah_strict.py` |
| Table 3, phone LongBench | `run_longbench_native_budgets.sh`, `run_longbench_wide.sh`, `run_lb_keydiff.sh`; `run_longbench_sllmw.sh` (StreamingLLM) | `scripts/longbench_score.py` |
| Table 3, CUDA LongBench | `scripts/run_longbench.sh` | `scripts/make_longbench_table.py` |
| Figure 1, cells live versus budget | `phone_queue/queue_gpu.sh` (cells `*_k256` to `*_k2048`) | `claims_data.py`, then `figures/master_tables/figures/fig_why_col.py` |

Phone LongBench scores 103 prompts from four tasks (hotpotqa, qasper, 2wikimqa, triviaqa): the
prompts on which every policy gives a non-empty answer. F1 is the mean of the four task means.

### Measurement protocol

- **Cooling gate.** Every timed run starts at DDR ≤ 35 °C and battery ≤ 33 °C, with charging off
  (`cool_gate.sh`). Sensors are logged at 2 to 5 Hz (`sample_sensors.sh`).
- **Ratios.** Speeds are reported as ratios to the full cache of the same campaign, as medians over
  rounds where there are several.
- **Phone-side queue.** Long campaigns run on the phone itself with `phone_queue_runner.sh`, which
  reads a list of cells (`phone_queue_cells.txt`), holds a wake lock, cools before each timed cell,
  skips finished cells on restart, stops if storage runs low and charges if the battery runs low.
  `pull_phone_queue.sh` copies results to the host whenever the adb link is up, so a lost link does
  not stop a campaign.

## Repository layout

| Path | What it holds |
|---|---|
| `llama.cpp/` | upstream llama.cpp (MIT license, `llama.cpp/LICENSE`) with the μKV changes listed above. Our additions are marked `EndurKV` in the source. |
| `entropy_probe/` | `eviction_bench.cpp` and the probes, with their CMake and build scripts |
| `scripts/android/` | phone campaigns, cooling gate, sensor sampling, phone-side queue |
| `eval_pipeline/`, `scripts/longbench_score.py` | needle and LongBench scoring |
| `eval_corpora/` | evaluation text files |
| `figures/master_tables/` | scripts that collect measured values and draw the figures |
| `energy_rl/`, `scripts/android/ukv_sched.sh`, the watchdog scripts | the energy-aware controller and thermal control, not used in the HotMobile paper |

Further notes: `scripts/android/README.md` (phone setup), `entropy_probe/PLATFORMS.md` (build and
run settings per device), `docs/HARDWARE_STRESS_LIMITS.md`.

## Build for Android (arm64)

Host: Android NDK r27c, CMake and Ninja, and `adb` to a rooted phone.

```
scripts/android/build_llama_android.sh          # llama.cpp, CPU
scripts/android/build_llama_android_vulkan.sh   # llama.cpp, Adreno GPU through Vulkan
scripts/android/build_probe_android.sh          # eviction_bench and the probes
```

Build the CPU libraries for `armv8.7-a`. Without it the dot-product and int8 matrix-multiply
kernels are left out. Push the shared libraries together with `eviction_bench`, never the binary
alone.

## Not in this repository

- **Models.** The paper uses Llama-3.2-1B-Instruct, Phi-3-mini-128k-instruct and gemma-2-2b-it in
  Q4_K_M, and the 1-bit Bonsai-8B.
- **Raw results.** Per-run outputs and sensor traces are not included. Some collection scripts
  still read these from local paths.
