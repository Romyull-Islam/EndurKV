# EndurKV — On-Phone Port (Android / OnePlus 15)

This directory ports the EndurKV measurement infrastructure from a Linux server
to a phone (initially a OnePlus 15, Snapdragon 8 Elite Gen 5, Android 16). It
adds three on-phone shell scripts and seven host-side Python orchestrators so
the existing `entropy_probe` / `attention_probe` / `prune_probe` binaries can
run on the phone while a parallel sensor sampler captures thermal, memory, and
endurance signals at decode-step granularity.

The Month-1 measurement gate of the proposal — *does the entropy↔attention
relationship survive 4-bit KV + mobile CPU + thermal throttling?* — is answered
end-to-end here.

## Headline result reproduced on phone

On Llama-3.1-8B Q4_K_M on the OnePlus 15, `cnn_dailymail` ρ = **−0.42**
(paper baseline −0.43, within 0.01). On Llama-3.2-1B Q4_K_M, long-form pooled
ρ = **−0.404** across 7 long-form generation tasks (n=2 161, p ≈ 10⁻⁸⁵). The
paper's per-task pattern reproduces task-by-task. See the README at the repo
root for the server-side reference numbers.

## Prerequisites on the host PC

| Tool | Version used | Where it goes |
|---|---|---|
| Android NDK | r27c | `$WORKSPACE/toolchain/android-ndk-r27c/` (or set `$ANDROID_NDK`) |
| CMake | ≥ 3.14 | on PATH |
| Ninja | any | on PATH |
| Android Platform-Tools (`adb`) | r37 | on PATH |
| Python | 3.11 | `pandas`, `scipy`, `matplotlib`, `huggingface_hub`, `datasets` |

The recommended workspace layout (parent of this repo):

```
<workspace>/
├── EndurKV/                ← this repo
├── toolchain/
│   └── android-ndk-r27c/
├── models/                 ← GGUF files (gitignored)
├── prompts/                ← prompts.jsonl (gitignored)
├── phone-deploy/           ← staged bin/+scripts/ to adb-push (gitignored)
└── logs/                   ← per-prompt CSVs + plots (gitignored)
```

Set `WORKSPACE=<workspace>` in your shell, or every script will walk up from
its own location to infer it.

## On the phone

- Enable Developer Options → USB debugging.
- Connect via USB-C, accept the "Allow USB debugging" prompt.
- No root required for the Month-1 gate. `/sys/block/sda/stat` and `/proc/diskstats`
  are PERM-DENIED on Android 16 non-root, so the sampler falls back to
  `/proc/vmstat` (`pswpout`, `pgmajfault`, `pgpgout`, `pswpin`) and
  `/proc/<probe_pid>/io` for the endurance signal. See "Without root" below.

## Pipeline

```
host: build_probe_android.sh        →  aarch64 ELF binaries
       │
host: host_phone_check.py           →  adb identity + sensor map
       │
host: download GGUF + run script 09 →  prompts/prompts.jsonl
       │
host: host_run_study.py             →  pushes bin/lib/model/prompts,
       │                                runs the prompt loop, pulls results
       │   ↓ adb shell
       │
phone: run_one_prompt.sh            →  starts sample_sensors.sh,
       │                                runs the probe, stops sampler
       │                                emits .entropy.csv + .attn.bin
       │                                       + .sensors.csv + .run.json
       │
host: host_join_and_rho.py          →  joins entropy and sensors,
       │                                computes per-task Spearman ρ
       │
host: host_make_plots.py            →  7 PNG figures
host: host_make_progress_plots.py   →  cross-regime panel (1B + 8B)
host: host_make_side_by_side.py     →  paired 1B/8B figures
host: host_make_summary_slide.py    →  one-pager summary PNG
```

## Files in this directory

### On-phone (pushed to `/data/local/tmp/endurkv/scripts/`)

| File | Role |
|---|---|
| `discover_sensors.sh` | Runs once per device. Dumps the thermal-zone map, block-device list, UFS controller paths, mount points, PSI availability, and memory state to `sensor_map.txt`. Identifies which world-readable paths the sampler can use. |
| `sample_sensors.sh` | Runs alongside each probe invocation. Samples 98 thermal zones + `/proc/vmstat` swap counters + `/proc/<probe_pid>/io` + (v3) per-core CPU freq + cpufreq-cpuN cooling-device state + Adreno `gpubusy` + battery via `dumpsys battery`. Default 10 Hz; 5 Hz recommended for 8B (lower sampler overhead). |
| `run_one_prompt.sh` | Orchestrates a single prompt: starts sampler in background, snapshots wall + monotonic + thermal at start, runs the probe binary with the right args (auto-adds `--output-attn` if the probe is `attention_probe`), stops sampler, writes `run.json` join metadata. |

### Host-side (Python)

| File | Role |
|---|---|
| `build_probe_android.sh` | Cross-compiles `entropy_probe` + `attention_probe` + `prune_probe` for `aarch64-Android-28`. Requires the llama.cpp Android build to already exist at `EndurKV/llama.cpp/build-android/`. |
| `host_phone_check.py` | Verifies adb sees the device, reports identity (model, SoC, Android version, ABI, root status), pushes and runs `discover_sensors.sh`, pulls the sensor map back to `logs/sensor_map_<device>.txt`. |
| `host_run_study.py` | Master orchestrator. Pushes the deploy bundle, then for each prompt: pushes prompt text, runs `run_one_prompt.sh` over `adb shell`, pulls four output files. Configurable via env vars: `PROBE` (default `attention_probe`), `LOG_SUBDIR`, `N_TOKENS`, `SEED`, `SENSORS_HZ`, `MAX_PROMPTS`, `MAX_PER_TASK`. |
| `host_join_and_rho.py` | Parses the `.attn.bin` sidecar (magic ATTN, uint32 n_steps/n_layers/n_head, then per-step per-layer uint32 n_kv + n_kv float32). Joins entropy.csv and sensors.csv by wall-clock. Filters trip-point thermal zones. Computes per-task Spearman ρ and pooled ρ. Prints the gate decision against the −0.20 fallback threshold. |
| `host_make_plots.py` | Seven per-model figures: thermal trace, thermal aggregate, KV cache growth, decode latency, entropy↔attention scatter, per-task ρ, memory + endurance. |
| `host_make_progress_plots.py` | Cross-regime defense panel: 1B vs 8B comparison + paper-baseline overlay + thermal+endurance contrast + sustained-run latency drift + one-page composite. Reads BOTH joined CSVs. |
| `host_make_side_by_side.py` | Paired 1B / 8B rendering of the same seven metrics for direct visual comparison. |
| `host_make_summary_slide.py` | 1920×1080 PNG slide with the session steps, data table, and headline result. |

## Build

```bash
export WORKSPACE=/abs/path/to/<workspace>     # parent of this repo
export ANDROID_NDK=$WORKSPACE/toolchain/android-ndk-r27c

# 1) build the llama.cpp Android libs (one-time; ~5 min)
cd $WORKSPACE/EndurKV/llama.cpp
mkdir -p build-android && cd build-android
cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_C_FLAGS="-march=armv8.7-a" \
  -DCMAKE_CXX_FLAGS="-march=armv8.7-a" \
  -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release
ninja -j8 llama ggml ggml-base ggml-cpu llama-bench

# 2) build the probes (~30 s)
bash $WORKSPACE/EndurKV/scripts/android/build_probe_android.sh
```

The `entropy_probe/CMakeLists.txt` was patched in this repo to (a) replace
`-march=native` with `-march=armv8.7-a` when `ANDROID` is set, (b) set
`BUILD_RPATH=$ORIGIN` so the binary finds its `.so` files next to itself, and
(c) override `CMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH` so `find_library` can see
the cross-built `libllama.so` outside the NDK sysroot.

## Stage everything to push

```bash
DEPLOY=$WORKSPACE/phone-deploy
mkdir -p $DEPLOY/bin $DEPLOY/scripts
cp $WORKSPACE/EndurKV/entropy_probe/build-android/{entropy,attention,prune}_probe   $DEPLOY/bin/
cp $WORKSPACE/EndurKV/llama.cpp/build-android/bin/libllama.so                       $DEPLOY/bin/
cp $WORKSPACE/EndurKV/llama.cpp/build-android/bin/libggml*.so                       $DEPLOY/bin/
cp $WORKSPACE/EndurKV/llama.cpp/build-android/bin/llama-bench                       $DEPLOY/bin/   # optional
cp $WORKSPACE/EndurKV/scripts/android/*.sh                                          $DEPLOY/scripts/
# strip CRLF if any (Windows tooling can introduce them)
for f in $DEPLOY/scripts/*.sh; do
  awk 'BEGIN{RS="\r\n"; ORS="\n"} {print}' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
```

## Run

```bash
# generate the workload (~5 min, fetches LongBench + HELM + lm-eval from HF)
PROMPTS_OUT=$WORKSPACE/prompts/prompts.jsonl \
  python $WORKSPACE/EndurKV/scripts/09_generate_workload.py

# verify the phone is reachable and dump its sensor map
python $WORKSPACE/EndurKV/scripts/android/host_phone_check.py

# run the study (~15 min on 1B, ~95 min on 8B)
export PROBE=attention_probe
export LOG_SUBDIR=study_phone_1b
export N_TOKENS=64
export SENSORS_HZ=10                  # 5 recommended for 8B
python $WORKSPACE/EndurKV/scripts/android/host_run_study.py \
    --model  $WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --prompts $WORKSPACE/prompts/prompts.jsonl

# analyze
python $WORKSPACE/EndurKV/scripts/android/host_join_and_rho.py \
    --log-dir $WORKSPACE/logs/$LOG_SUBDIR

# plot
python $WORKSPACE/EndurKV/scripts/android/host_make_plots.py \
    --log-dir $WORKSPACE/logs/$LOG_SUBDIR
```

For the cross-regime panels (after running both 1B and 8B), the progress and
side-by-side plotters read the two joined CSVs directly:

```bash
python $WORKSPACE/EndurKV/scripts/android/host_make_progress_plots.py
python $WORKSPACE/EndurKV/scripts/android/host_make_side_by_side.py
```

## Without root

These signals stop working on a non-rooted Android 16 device. The current
pipeline degrades gracefully; corresponding CSV columns are blank.

| Signal | Path | Workaround used |
|---|---|---|
| Block-device write counters | `/sys/block/sda/stat` | `pswpout × 4 KB` from `/proc/vmstat` — counts kernel-swap-out bytes only, not other UFS writes |
| Per-partition I/O | `/proc/diskstats` | none |
| UFS Health Descriptor (P/E budget) | SCSI IOCTL via `/dev/sg*` | none — controller's full WAF measurement requires root |
| Adreno GPU busy % | `/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage` | `gpubusy` legacy file (cumulative `busy_us total_us`), joiner takes delta |
| PSI | `/proc/pressure/{cpu,memory,io}` | none |

The proposal's slide 25 ("Risk and mitigation") already names this fallback as
acceptable for the Month-1 measurement gate. The rooted ablation is left for
follow-up work on a dedicated research device.

## On-phone deploy paths

```
/data/local/tmp/endurkv/
├── bin/                  ← probe binaries + .so libs + llama-bench
├── models/               ← GGUF files (push once, ~5 GB for both)
├── scripts/              ← discover + sampler + run_one_prompt
├── prompts/              ← per-prompt text files (transient)
└── logs/<LOG_SUBDIR>/    ← per-prompt outputs
```

Files persist across reboots — Android does not wipe `/data/local/tmp/`. To
clean up entirely:

```bash
adb shell rm -rf /data/local/tmp/endurkv
```
