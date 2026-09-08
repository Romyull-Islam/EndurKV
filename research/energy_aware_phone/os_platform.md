# Energy-aware phone usage at the OS and platform level: survey (2024 to 2026)

Date of survey: 2026-09-04.
Method: about 54 web searches and about 90 page fetches. Every item below carries a URL that was fetched in this session. Venue and year were read from the fetched page, from a Crossref record, or from a dblp record. Items where the primary page was blocked and only an index record or a secondary page could be read are marked "partially verified". Items that could not be confirmed are marked "unverified".

Relevance target used throughout: a per-request energy-aware scheduler for on-device LLM inference that can cap the GPU clock, shrink the KV cache and cap the output length by battery state. Each item ends with one "Relevance" sentence about that target.

## 0. Summary of what the survey found

1. The kernel already has a full energy-aware CPU stack: EAS picks CPUs by predicted energy from the Energy Model, schedutil sets frequency from PELT with a 25% margin, and uclamp gives user space a per-task performance floor and ceiling. None of this covers the GPU or the NPU, and EAS switches itself off above 80% CPU utilization, which is exactly where LLM decode runs.
2. Android's app-facing levers are ADPF hint sessions (target and actual work duration, power-efficiency mode since Android 15), the thermal headroom forecast (0.0 to 1.0, with OEM thresholds since Android 15), and CPU and GPU headroom estimates (Android 16). Android 17 (June 2026) adds no per-app energy budget API. Google Play now enforces a battery quality bar (excessive wake locks) as of March 2026.
3. Battery-state policy on Android is still threshold based (Battery Saver at a user percentage, off at 90%, Extreme Battery Saver pauses apps and slows processing), with one routine-based hook (Routine Battery Saver). Apple's iOS 26 Adaptive Power is the first mainstream policy that is triggered by predicted deficit against a 7-day learned routine, that "makes performance adjustments", and that exempts camera and Game Mode workloads.
4. On Snapdragon the GPU clock cap is a documented sysfs lever (`/sys/class/kgsl/kgsl-3d0/max_pwrlevel`, `max_clock_mhz`, `thermal_pwrlevel`), and the driver composes user and thermal caps with a max() so they cannot conflict. Vendors (Qualcomm, MediaTek, Samsung, Google) publish efficiency percentages and marketing names for schedulers but expose no public per-request energy API.
5. Energy accounting moved from batterystats heuristics to measured rails: Power Stats HAL rails, ODPM on Pixel 6 and later, Perfetto `android.power`, Android Studio Power Profiler, and trace-based Wattson (1 point average error). Battery Historian is deprecated. On non-Pixel phones only the coulomb counter is available to an app.
6. Academic work in 2025 and 2026 converged on the same levers the target scheduler uses. Frequency coordination across CPU, GPU and DDR (FUSE, CORE at MLSys 2026) cuts decode latency 25% to 40% at equal energy. Modest NPU and DDR down-clocking (EnerInfer) saves up to 65% energy on phones without hurting QoE. Millisecond GPU frequency scaling during stalls (TurboInfer, MobiSys 2026) targets decode power. Output length dominates energy at the prompt level (EMNLP 2026, "Keyword Matters"). Sustained-load studies show phones lose half their throughput within two runs and that the OS will hard-floor the GPU. Nothing found ties these levers to battery state per request; that composition is open.
7. Users respond to battery state early and strongly: the average American starts to worry at 38% (Talker Research, 2025), a low battery measurably changes choices (Journal of Consumer Behaviour, 2026), and accurate per-app energy forecasts reduce anxiety (SERENUS, UIST 2024). Human reading speed is about 4.8 tokens/s and listening 3.3 tokens/s (Andes), which bounds the useful decode rate.

## 1. Linux and Android kernel and platform

### 1.1 Energy Aware Scheduling (EAS)
- Title: "Energy Aware Scheduling" (kernel scheduler documentation)
- Organization: Linux kernel community
- Venue and year: docs.kernel.org, current mainline documentation (fetched 2026-09-04)
- URL: https://docs.kernel.org/scheduler/sched-energy.html
- Summary: On task wake-up `find_energy_efficient_cpu()` picks the CPU with most spare capacity in each performance domain and calls `compute_energy()` with the platform Energy Model to place the task where predicted energy is lowest without hurting throughput. EAS disables itself and hands back to the load balancer when any CPU is over-utilized.
- Key numbers: over-utilization threshold is 80% of compute capacity; EAS will not start if EM complexity exceeds EM_MAX_COMPLEXITY (2048 at the time of writing); it requires asymmetric CPU capacities and "the only sane governor to use together with EAS is schedutil".
- Relevance: heavy decode pushes CPUs past 80% and turns EAS off, so the kernel stops saving CPU energy precisely while inference runs; a per-request scheduler must therefore manage CPU-side energy itself (uclamp or core selection) rather than rely on EAS.

### 1.2 Energy Model of devices (EM framework)
- Title: "Energy Model of devices"
- Organization: Linux kernel community
- Venue and year: docs.kernel.org, kernel 7.3.0-rc1 documentation (fetched 2026-09-04)
- URL: https://docs.kernel.org/power/energy-model.html
- Summary: Each performance domain holds a table of states with frequency (kHz), power (uW), a derived cost (10 x power x max_frequency / frequency) and a performance value. Registration is by driver callback (`em_dev_register_perf_domain()`), by device tree `opp-microwatt`, or by the simple `Power = C x V^2 x f` model.
- Key numbers: runtime-modifiable tables (`em_table_alloc()`, `em_dev_compute_costs()`, `em_dev_update_perf_domain()`) and `em_update_performance_limits()` to change minimum and maximum after registration.
- Relevance: the EM is the only first-class energy cost table in the kernel; a scheduler could hold an EM-like table for the GPU (clock to power to tokens/s) and choose the cap per battery tier from it instead of guessing.

### 1.3 Utilization clamping (uclamp)
- Title: "Utilization Clamping"
- Organization: Linux kernel community
- Venue and year: docs.kernel.org mainline documentation (fetched 2026-09-04); Android common kernels ship the same document
- URL: https://docs.kernel.org/scheduler/sched-util-clamp.html
- Summary: Every task carries UCLAMP_MIN and UCLAMP_MAX in the 0 to 1024 performance range, set by `sched_setattr()`, by cgroup `cpu.uclamp.min` and `cpu.uclamp.max`, or by sysctls. Schedutil reads the runqueue effective clamp to choose frequency and EAS uses it in placement.
- Key caveats from the page: the runqueue clamp is the max over attached tasks, so a capped task loses its cap when an uncapped task shares the CPU; hard UCLAMP_MAX caps remove idle time and saturate PELT at 1024, causing frequency spikes; schedutil rate limiting delays reaching the requested point.
- Relevance: `cpu.uclamp.max` is the sanctioned CPU clock cap for the inference thread group on Android; the max-aggregation rule means the cap only holds if decode threads are isolated from uncapped work.

### 1.4 Schedutil
- Title: "Schedutil"
- Organization: Linux kernel community
- Venue and year: docs.kernel.org (fetched 2026-09-04)
- URL: https://docs.kernel.org/scheduler/schedutil.html
- Summary: The governor maps frequency-invariant PELT utilization (CFS with UTIL_EST, plus RT, IRQ and deadline contributions) through uclamp bounds and a 1.25 x u x f_max margin into a DVFS request, with iowait boost and rate limiting.
- Key numbers: 25% headroom margin.
- Relevance: a busy decode thread always requests f_max under schedutil, so without a uclamp ceiling or a PowerHAL cap the CPU side of inference runs at the top OPP by design.

### 1.5 Android Dynamic Performance Framework (ADPF) and the Android 15 additions
- Title: "Optimize thermal and CPU performance with the Android Dynamic Performance Framework" and "Android 15 features and APIs"
- Organization: Google
- Venue and year: developer.android.com; Android 15 (2024)
- URLs: https://developer.android.com/games/optimize/adpf and https://developer.android.com/about/versions/15/features
- Summary: ADPF bundles the Thermal API, Performance Hint sessions (`reportActualWorkDuration()`, `updateTargetWorkDuration()`), Game Mode and Game State APIs, Fixed Performance Mode and the CPU/GPU headroom APIs. Android 15 added `Session.setPreferPowerEfficiency(boolean)` for long-running work, joint CPU and GPU work-duration reporting via `reportActualWorkDuration(WorkDuration)` so the platform moves CPU and GPU frequency together, and `PowerManager.getThermalHeadroomThresholds()`.
- Key numbers: power-efficiency mode and GPU durations are Android 15 and later; the Android 15 page links the Google I/O 2024 talk on background battery efficiency.
- Relevance: an LLM runtime can open a hint session per decode loop, report per-token duration and a target token time, and set `setPreferPowerEfficiency(true)` on low battery; this is the sanctioned Android lever before touching sysfs.

### 1.6 ADPF Thermal API semantics
- Title: "Thermal API" (ADPF guide)
- Organization: Google
- Venue and year: developer.android.com (fetched 2026-09-04)
- URL: https://developer.android.com/games/optimize/adpf/thermal
- Summary: `PowerManager.getThermalHeadroom(forecastSeconds)` returns a 0.0 to 1.0 forecast where 1.0 means THERMAL_STATUS_SEVERE; status levels are NONE, LIGHT, MODERATE, SEVERE, CRITICAL, EMERGENCY, SHUTDOWN. The guide tells apps to reduce frame rate, fidelity or move work to smaller cores as headroom rises.
- Key numbers: call at most once per 10 seconds or it returns NaN; guidance: below 0.85 safe, above 0.85 light, above 0.95 moderate, above 1.0 severe; example forecast horizon 30 s.
- Relevance: an OEM-calibrated thermal forecast lets the scheduler shrink KV or cap tokens at 0.85 to 0.95 before throttling instead of after a trip point; the 10 s polling floor matches per-request granularity.

### 1.7 Android 16 CPU and GPU headroom APIs
- Title: "Android 16 features and APIs"
- Organization: Google
- Venue and year: developer.android.com, Android 16 (API level 36, 2025)
- URL: https://developer.android.com/about/versions/16/features
- Summary: `SystemHealthManager.getCpuHeadroom(CpuHeadroomParams)` and `getGpuHeadroom(GpuHeadroomParams)` estimate the CPU or GPU capacity still available over a chosen window using average or minimum, and are meant to be combined with ADPF thermal signals so heavy apps cut work before throttling.
- Key numbers: the NDK reference (https://developer.android.com/ndk/reference/group/system-health) states that each call does at least one synchronous binder transaction that can exceed 1 ms; supported devices only.
- Relevance: GPU headroom is the first platform signal that says how much GPU is left after other apps and thermal caps, which the scheduler can use to decide whether to run on GPU at all and at what clock.

### 1.8 Game Mode API and OEM interventions
- Titles: "Game Mode API" and "Game Mode interventions"
- Organization: Google
- Venue and year: developer.android.com; select Android 12 devices, full support from Android 13
- URLs: https://developer.android.com/games/optimize/adpf/gamemode/gamemode-api and https://developer.android.com/games/optimize/adpf/gamemode/gamemode-interventions
- Summary: A user picks STANDARD, PERFORMANCE or BATTERY per game; a game opts in via `game_mode_config.xml`. In BATTERY mode the system may lower refresh rate (30 or 60 Hz) and frame targets, and OEMs can apply interventions without developer involvement: WindowManager backbuffer downscaling and FPS override (Android 13 and later).
- Key numbers: backbuffer downscaling can cut GPU use by up to 30% and system power by 10%; adb test hooks `cmd game mode battery` and `device_config put game_overlay ... downscaleFactor=0.5`.
- Relevance: Game Mode is the closest shipped precedent for a user-selected battery tier that trades output fidelity for energy; the analogue for LLMs is a user-visible "battery answer" tier with shorter output and a smaller KV.

### 1.9 App Standby Buckets and the power-management restriction table (Android 9 to 16)
- Titles: "App Standby Buckets", "Power management restrictions", "Power management (Android 9)"
- Organization: Google
- Venue and year: developer.android.com; buckets since Android 9, restricted bucket since Android 12, quota changes in Android 16
- URLs: https://developer.android.com/topic/performance/appstandby , https://developer.android.com/topic/performance/power/power-details , https://developer.android.com/about/versions/pie/power
- Summary: The system places each app in Active, Working Set, Frequent, Rare, Restricted or Never, using an OEM machine-learning app if present, and limits jobs, alarms and network per bucket. Doze defers jobs and alarms to maintenance windows; charging lifts most limits; Android 16 adjusts JobScheduler quota by bucket and by whether the app is top or has a foreground service.
- Key numbers: regular-job quota Active 20 min per 60 min, Working Set 10 min per 4 h, Frequent 10 min per 12 h, Rare 10 min per 24 h, Restricted 10 min per day; alarms Working Set 10 per hour, Frequent 2 per hour, Rare 1 per hour, Restricted 1 per day; Rare and Restricted lose network; apps enter Restricted after 8 days without interaction on Android 13 and later (45 days on Android 12).
- Relevance: this is the only per-app "budget" Android has and it budgets time, not energy; a foreground LLM request is unconstrained by it, so the energy budget for inference must come from the app itself.

### 1.10 Battery Saver, Extreme Battery Saver and Routine Battery Saver
- Titles: "Make your Pixel phone's battery last longer" (Pixel Help), "Routine Battery Saver" (AOSP)
- Organization: Google
- Venue and year: support.google.com (fetched 2026-09-04); source.android.com, Routine Battery Saver since Android 10
- URLs: https://support.google.com/pixelphone/answer/6187458 and https://source.android.com/docs/core/power/routine-battery-saver
- Summary: Battery Saver refreshes app content only when opened, stops background activity and location with the screen off, forces Dark theme, turns off smooth display and may drop 5G to 4G. Extreme Battery Saver pauses most apps, "slows processing speed", sets screen timeout to 30 s and stops hotspot and scanning. Routine Battery Saver lets an OEM app call `setDynamicPowerSaveHint(true, disableThreshold)` from a learned routine rather than a percentage.
- Key numbers: user-set on threshold; auto-off at 90% (Pixel); routine mode default disable threshold 80% (`config_dynamicPowerSavingsDefaultDisableThreshold`).
- Relevance: the platform already lowers processing speed in its most aggressive tier and gives `PowerManager.isPowerSaveMode()` as a signal; a scheduler should treat Battery Saver as a tier boundary and copy the on/off hysteresis so requests near the threshold do not flap.

### 1.11 Android 16 battery health, capacity estimate and charge limit (Pixel)
- Organization: 9to5Google quoting Google settings text (secondary; partially verified)
- Venue and year: 9to5google.com, June 11, 2025
- URL: https://9to5google.com/2025/06/11/android-16-pixel-battery-health-2/
- Summary: Android 16 adds a Battery health page with "battery health assistance" that steps down maximum charge voltage from 200 to 1000 cycles (on by default and not switchable on Pixel 9a), a capacity estimate flagged "Reduced" below 80% (Pixel 8a and newer), and moves Charging optimization (Off, Adaptive Charging, Limit to 80%) under it.
- Key numbers: 200 to 1000 cycles; 80% limit; 80% capacity flag.
- Relevance: percentage is not joules; a scheduler that keys on state of charge should also read health and the 80% cap, since the same percentage on an aged or capped battery buys fewer tokens.

### 1.12 Android 17 (2026): no new energy budget API, charging fixes
- Titles: "Android 17 is here" (developer blog), "Android 17 features and APIs", "Everything new in Android 17 Beta 3" (secondary)
- Organization: Google; 9to5Google for Beta 3 notes
- Venue and year: android-developers.googleblog.com June 16, 2026; developer.android.com; 9to5google.com March 26, 2026
- URLs: https://android-developers.googleblog.com/2026/06/Android-17.html , https://developer.android.com/about/versions/17/features , https://9to5google.com/2026/03/26/android-17-beta-3-everything-new/
- Summary: Android 17's power-related work is indirect: generational GC in ART, strict per-app memory limits with termination, lock-free MessageQueue, background audio limits, and ProfilingManager triggers such as TRIGGER_TYPE_KILL_EXCESSIVE_CPU_USAGE. Beta 3 fixed charging stalling at 77% under the 80% limit and fixed Battery Saver staying on when the 80% limit was enabled. "Priority Charging" was seen in code but is not user-visible (unverified feature).
- Key numbers: 77% stall, 80% cap.
- Relevance: through Android 17 there is still no per-app energy budget or battery-tier API, so the scheduler must build its own from BatteryManager, thermal headroom, CPU/GPU headroom and Battery Saver state.

### 1.13 Google Play battery technical quality enforcement (excessive wake locks)
- Titles: "Raising the bar on battery performance: excessive partial wake locks metric is now out of beta" and "Battery Technical Quality Enforcement is Here"
- Organization: Google, Android Developers Blog
- Venue and year: November 10, 2025 and March 4, 2026
- URLs: https://android-developers.googleblog.com/2025/11/raising-bar-on-battery-performance.html and https://android-developers.googleblog.com/2026/03/battery-technical-quality-enforcement.html
- Summary: Android vitals reports the share of user sessions with more than 2 cumulative hours of non-exempt partial wake locks in 24 hours; from March 1, 2026 apps above the bad-behaviour bar may get a store-listing warning and be excluded from recommendations. Audio playback, location and user-initiated transfers are exempt.
- Key numbers: 2 hours per session; bad if more than 5% of sessions over 28 days.
- Relevance: this is the first store-level energy budget for apps; long on-device generations that hold a wake lock with the screen off will count, so a scheduler that caps output length under low battery also protects the app's Play standing.

### 1.14 Energy accounting stack on Android (fuel gauge, Power Stats HAL, Perfetto, ODPM, Wattson, Battery Historian)
- Titles and URLs:
  - "Measure device power": https://source.android.com/docs/core/power/device
  - "Power Stats HAL": https://source.android.com/docs/core/power/power-stats-hal
  - "Power data sources" (Perfetto): https://perfetto.dev/docs/data-sources/battery-counters
  - "Power Profiler" (Android Studio): https://developer.android.com/studio/profile/power-profiler
  - "Collect and view traces" (Wattson): https://source.android.com/docs/core/power/wattson/how-to-wattson
  - "Battery Historian": https://developer.android.com/topic/performance/power/battery-historian
  - Developer story, Kuro Games: https://developer.android.com/stories/games/kuro-powerprofiler
- Organization: Google, AOSP, Perfetto project
- Venue and year: AOSP and developer docs, fetched 2026-09-04; IPowerStats since Android 10; ODPM on Pixel 6 and later; Wattson support tied to 2025 builds
- Summary: Apps read the fuel gauge through `BatteryManager` (CHARGE_COUNTER in uAh, CURRENT_NOW in uA, ENERGY_COUNTER in nWh). The Power Stats HAL exposes energy rails, power entities and state residencies and is read opportunistically on each 1% drop; batterystats uses it instead of `power_profile.xml`. Perfetto `android.power` samples battery counters (example 250 ms) and ODPM rails, which are measured downstream of the battery. The Power Profiler shows rails (CPU big/mid/little, GPU, display, memory, UFS, radios). Wattson estimates CPU and GPU power from a trace and attributes it per thread, process or package. Battery Historian carries a "no longer actively maintained" warning.
- Key numbers: fuel-gauge accuracy good above about 20 mA; Wattson average error 1 percentage point, standard deviation 1.5 points on 10 s to 4 h workloads; Kuro Games cut total power from 3,233 mW to 2,920 mW (9.68%) at equal FPS using rails.
- Relevance: on Pixel the GPU and memory rails give a per-request energy measurement good enough to calibrate a clock-to-energy table; on other phones only CHARGE_COUNTER and CURRENT_NOW exist, charging must be off, and request-level attribution is bounded by fuel-gauge resolution and update period.

## 2. Apple

### 2.1 Adaptive Power (iOS 26)
- Title: "Use Adaptive Power to extend the battery life of your iPhone"
- Organization: Apple Support
- Venue and year: support.apple.com article 123707, iOS 26 (2025); settings text quoted by MacRumors, August 21, 2025 (secondary)
- URLs: https://support.apple.com/en-us/123707 and https://www.macrumors.com/how-to/ios-extend-iphone-battery-life-adaptive-power-mode/
- Summary: On days when usage runs above the learned norm, on-device intelligence predicts a shortfall and the phone "makes performance adjustments", lowers brightness "by a small amount", limits background activity and turns on Low Power Mode at 20%. The in-settings text says some tasks may "take a little longer". It does not act during camera use or in games with Game Mode.
- Key numbers: needs at least 7 days of charging history; on by default on iPhone 17, 17 Pro, 17 Pro Max and iPhone Air; off by default on iPhone 15 Pro and all iPhone 16 models; requires Apple Intelligence-capable hardware. Apple publishes no size for the performance adjustment.
- Relevance: Apple's policy is predicted deficit versus routine, with a foreground exemption; a per-request scheduler can copy both rules (predict remaining need, then exempt interactive requests and throttle only background or long ones).

### 2.2 Low Power Mode
- Title: "Use Low Power Mode to save battery life on your iPhone or iPad"
- Organization: Apple Support
- Venue and year: support.apple.com article 101604 (fetched 2026-09-04)
- URL: https://support.apple.com/en-us/101604
- Summary: Low Power Mode turns off 5G on most models, sets auto-lock to 30 s, reduces brightness, caps ProMotion at 60 Hz, disables some visual effects, pauses iCloud Photos and stops automatic downloads, mail fetch and background refresh. It turns off when charge reaches 80%.
- Key numbers: 60 Hz, 30 s, 80% auto-off, 20% trigger when Adaptive Power is on. The page states no CPU or GPU clock figure.
- Relevance: a documented two-threshold hysteresis (on at 20%, off at 80%) and a fixed list of cuts; the LLM analogue is a fixed low-battery profile (clock cap, KV budget, max tokens) with the same hysteresis.

### 2.3 Apple Intelligence on-device foundation model (2025)
- Titles: "Updates to Apple's On-Device and Server Foundation Language Models" and "Apple Intelligence Foundation Language Models: Tech Report 2025"
- Organization: Apple
- Venue and year: machinelearning.apple.com June 9, 2025 (updated July 17, 2025); arXiv 2507.13575, July 2025 (revised August 2025)
- URLs: https://machinelearning.apple.com/research/apple-foundation-models-2025-updates and https://arxiv.org/abs/2507.13575
- Summary: The about 3B on-device model uses two blocks with a 5:3 depth ratio; all KV caches of block 2 are shared with the final layer of block 1, cutting KV memory by 37.5% and improving time-to-first-token. Decoder weights are 2-bit (quantization-aware training), embeddings 4-bit, KV cache 8-bit, with low-rank adapters for recovery.
- Key numbers: 37.5% KV memory reduction; no power, battery or thermal numbers are published.
- Relevance: Apple ships KV reduction as a fixed architectural choice; a per-request KV shrink is the dynamic counterpart and stacks on top of it.

### 2.4 Foundation Models framework availability reasons (no battery or thermal gate)
- Title: `SystemLanguageModel.Availability.UnavailableReason`
- Organization: Apple Developer Documentation
- Venue and year: developer.apple.com, iOS 26 SDK (fetched 2026-09-04 via the documentation JSON endpoint)
- URL: https://developer.apple.com/documentation/foundationmodels/systemlanguagemodel/availability-swift.enum/unavailablereason
- Summary: The only unavailable reasons are `appleIntelligenceNotEnabled`, `deviceNotEligible` and `modelNotReady`. There is no case for Low Power Mode, battery level or thermal state.
- Key numbers: none.
- Relevance: Apple exposes no battery-state signal to third-party on-device LLM use; an app on iOS must read `ProcessInfo.isLowPowerModeEnabled` and `thermalState` itself and has no GPU clock lever, so output length and model or KV size are the only per-request levers there.

## 3. OEM and SoC vendors

### 3.1 Qualcomm KGSL GPU power control (primary source)
- Title: `kgsl_pwrctrl.c` in qualcomm-linux/kgsl (branch gfx-kernel.le.0.0)
- Organization: Qualcomm
- Venue and year: GitHub source, current Qualcomm Linux GPU driver (fetched 2026-09-04)
- URL: https://raw.githubusercontent.com/qualcomm-linux/kgsl/gfx-kernel.le.0.0/kgsl_pwrctrl.c (repository https://github.com/qualcomm-linux/kgsl)
- Summary: `/sys/class/kgsl/kgsl-3d0/` exposes writable `max_pwrlevel`, `min_pwrlevel`, `thermal_pwrlevel`, `default_pwrlevel`, `max_gpuclk`, `gpuclk`, `max_clock_mhz`, `min_clock_mhz`, `idle_timer`, `force_clk_on`, `force_bus_on`, `force_rail_on`, `pwrscale`, and read-only `temp`, `gpubusy`, `gpu_busy_percentage`, `gpu_available_frequencies`, `freq_table_mhz`, `clock_mhz`, `num_pwrlevels`, `thermal_time`, `gpu_clock_stats`. `thermal_pwrlevel` is applied as a PM QoS max-frequency request (`dev_pm_qos_update_request(&pwr->sysfs_thermal_req, gpu_freq/1000)`). DCVS runs in GMU firmware on new parts and in host `kgsl_pwrscale` on old ones.
- Key mechanism (quoted): `max_pwrlevel = max_t(u32, pwr->max_pwrlevel, thermal_pwrlevel)` and `min_pwrlevel = max_t(u32, pwr->min_pwrlevel, thermal_pwrlevel)` in `_adjust_pwrlevel()`; a higher index is a lower clock, so the thermal level always wins. `max_pwrlevel` store clamps to `min_pwrlevel`.
- Relevance: this is the exact lever for "cap the GPU clock by battery state" on Snapdragon; write a level index to `max_pwrlevel` or MHz to `max_clock_mhz`, and the kernel composes it with the thermal cap using max() so battery and thermal policies cannot race.

### 3.2 Adreno power levels (device tree binding)
- Title: "adreno-pwrlevels" device tree binding
- Organization: Qualcomm (msm kernel documentation in AOSP)
- Venue and year: android.googlesource.com kernel/msm documentation (older tag; the structure is unchanged in current kgsl)
- URL: https://android.googlesource.com/kernel/msm/+/android-wear-5.0.2_r0.1/Documentation/devicetree/bindings/gpu/adreno-pwrlevels.txt
- Summary: The GPU carries a table of power levels with `qcom,gpu-freq` (Hz), `qcom,bus-freq` (bus scaling use-case index) and `qcom,io-fraction`; lower `reg` index means higher performance.
- Key numbers: structure only.
- Relevance: a GPU clock cap on Snapdragon is a power-level index, and each level also fixes a bus (DDR) vote, so capping the GPU level also caps memory bandwidth, which matters for memory-bound decode.

### 3.3 Snapdragon 8 Elite Gen 5 (2025) power claims and measured behaviour
- Titles: product page; launch press release; Counterpoint analysis; Notebookcheck measurements; press release for Galaxy (Feb 2026, from search index only)
- Organization: Qualcomm; Counterpoint Research; Notebookcheck
- Venue and year: qualcomm.com product page (fetched 2026-09-04); press release September 24, 2025; Counterpoint October 2, 2025; Notebookcheck device page 2026
- URLs: https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5 , https://www.qualcomm.com/news/releases/2025/09/snapdragon-8-elite-gen-5--the-world-s-fastest-mobile-system-on-a , https://counterpointresearch.com/en/insights/qualcomm-snapdragon-8-elite-gen-5-redefines-mobile-performance-and-ai-efficiency , https://www.notebookcheck.net/Qualcomm-Snapdragon-8-Elite-Gen-5-for-Galaxy-Processor-Benchmarks-and-Specs.1271123.0.html
- Summary: Qualcomm claims 20% more CPU performance with 35% better CPU power efficiency, 23% more GPU performance with 20% better GPU efficiency, 37% faster Hexagon NPU with 16% better performance per watt, a dedicated NPU power delivery system, 18 MB of GPU "High Performance Memory", and "up to 16% overall SoC power savings, equating to 1 hour and 48 minutes of additional playtime". Notebookcheck lists 2 x 4.7 GHz prime plus 6 x 3.6 GHz Oryon Gen 3 cores, Adreno 840 up to 1300 MHz, TSMC N3P, and measured package power of 5.92 W to 9.66 W in Geekbench with about 0.85 to 1.05 W idle.
- Key numbers: as above. Third-party sustained-load throttling reports (PhoneArena, Android Headlines) were blocked (403) and are unverified; a February 2026 report that the next generation may adopt Samsung's Heat Pass Block is a rumour (https://www.igorslab.de/en/qualcomm-breaks-new-ground-in-cooling-with-the-snapdragon-8-elite-gen-6/, marked as such by the article).
- Relevance: Qualcomm publishes efficiency deltas and DCVS lives in GMU firmware, but the only public per-app control path is ADPF hints plus the kgsl sysfs caps; no vendor API accepts a battery-state input.

### 3.4 Samsung: Light performance profile and Galaxy AI on-device-only processing
- Titles: "Extend battery life with the Light performance profile on your Samsung Galaxy"; "Data processing for Galaxy AI" (Knox Service Plugin)
- Organization: Samsung
- Venue and year: samsung.com support (fetched 2026-09-04; profiles arrived with One UI 6 in 2024 per search index, unverified on the page); docs.samsungknox.com, One UI 6.1 and later
- URLs: https://www.samsung.com/ca/support/mobile-devices/light-performance-profile-on-your-samsung-galaxy/ and https://docs.samsungknox.com/admin/knox-platform-for-enterprise/knox-service-plugin/configure-advanced-policies/data-processing-for-galaxy-ai/
- Summary: Settings > Device care > Performance profile offers Standard and Light; Light "reduces processing power slightly to prioritize battery life and cooling efficiency" and "does not affect game performance". Separately, "Process data only on device" (user toggle under Settings > Galaxy AI, and a Knox admin policy) forces Galaxy AI features (Notes, Voice Recorder, Web Assist Summary, translation, Interpreter, Generative Edit, Call Assist, Chat Assist) to run locally; some cloud-only features become unavailable.
- Key numbers: Samsung publishes none; user reports of 5% to 15% longer battery life on Light are anecdotal (unverified).
- Relevance: Samsung already ships a user-selected "slightly slower for cooler and longer" tier that exempts games, and a switch that moves AI work on-device; a per-request LLM scheduler is the missing piece that makes the on-device switch battery-aware.

### 3.5 Google Tensor G5 (Pixel 10)
- Organization: Android Authority (secondary; partially verified)
- Venue and year: androidauthority.com, August 27, 2025
- URL: https://www.androidauthority.com/google-tensor-g5-benchmarks-3590355/
- Summary: Tensor G5 moves to TSMC 3 nm with 1 x Cortex-X4 at 3.78 GHz, 5 x Cortex-A725 at 3.05 GHz and 2 x Cortex-A520 at 2.25 GHz; Google claims "30+ hours" battery versus "24+" for Pixel 9 and a 34% faster CPU on average.
- Key numbers: stress-test surface 46.1 C (Pixel 10) versus 44.7 C (Pixel 9); performance gains disappear after 6 to 7 minutes of sustained load.
- Relevance: even a new efficiency-focused SoC throttles within minutes, so long generations on Pixel need pre-emptive caps rather than reactive throttling.

### 3.6 MediaTek Dimensity 9500
- Organization: MediaTek
- Venue and year: mediatek.com press release, September 22, 2025
- URL: https://www.mediatek.com/press-room/mediatek-dimensity-9500-unleashes-best-in-class-performance-ai-experiences-and-power-efficiency-for-the-next-generation-of-mobile-devices
- Summary: A second-generation Dimensity Scheduling Engine is marketed as the source of "sustained efficiency"; the NPU 990 with Generative AI Engine 2.0 supports BitNet 1.58-bit processing.
- Key numbers: up to 55% lower power at peak on the ultra core; GPU 42% better power efficiency; BitNet 1.58-bit up to 33% lower power; 56% lower power at peak for AI tasks; "100% faster 3B LLM output".
- Relevance: vendors now advertise 1-bit LLM support and scheduler-driven sustained efficiency but expose no public per-request energy control, which leaves room for an app-level scheduler.

## 4. Academic work, 2024 to 2026

### 4.1 Coordinated DVFS for mobile LLM inference: FUSE (arXiv 2025) and CORE (MLSys 2026)
- Authors: Zongpu Zhang, Pranab Dash, Y. Charlie Hu, Qiang Xu, Jian Li, Haibing Guan
- Venue and year: arXiv 2507.02135 (July 2025) "Dissecting the Impact of Mobile DVFS Governors on LLM Inference Performance and Energy Efficiency"; "Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE", Proceedings of MLSys 2026 (Bellevue, May 2026)
- URLs: https://arxiv.org/abs/2507.02135 and https://proceedings.mlsys.org/paper_files/paper/2026/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html
- Summary: Independent CPU, GPU and memory governors on phones make uncoordinated frequency choices for LLM prefill and decode; a unified governor picks the three frequencies jointly per phase.
- Key numbers: default governors are 23.0% to 40.4% slower or use 5.0% to 16.6% more energy than the coordinated optimum; CORE cuts TTFT 8.5% to 17.7% and time per token 27.8% to 39.6% at no increase in energy per token (FUSE preprint: TTFT 7.0% to 16.9%, TPOT 25.4% to 36.8%).
- Relevance: a GPU clock cap alone is a partial lever; DDR frequency matters as much for decode, so the per-request scheduler should treat GPU and memory clocks together (on Snapdragon the pwrlevel table already couples them).

### 4.2 EnerInfer: Energy-Aware On-Device LLM Inference
- Authors: Bohua Zou, Nian Liu, Binqi Sun, Matteo Mascherin, Debayan Roy, Yutao Liu, Yu Peng, Ning Jia, Haibo Chen
- Venue and year: arXiv 2606.23001, June 2026 (v2 June 24, 2026); no venue stated
- URL: https://arxiv.org/abs/2606.23001
- Summary: Modestly lowering NPU and DDR frequencies keeps quality of experience while cutting energy and heat; the system predicts throughput and power for unseen models across NPU/DDR settings, picks QoE-satisfying efficient points under interference, and uses short-horizon thermal prediction to switch between energy-optimized and thermally constrained modes.
- Key numbers: energy efficiency gains of 65% on phones, 12% on laptops, 24% on development boards with no QoE violations.
- Relevance: direct evidence that the frequency-to-QoE curve is flat near the top, so a battery-tier accelerator clock cap costs little in user experience.

### 4.3 Act Before It's Too Late: Power-Efficient LLM Inference on Mobile Device (TurboInfer)
- Authors: Haolin Chu, Jinxiao Fan, Jiabin Deng, Bensong Yu, Liguang Xie, Liang Liu, Huadong Ma, Xiaolong Zheng
- Venue and year: Proceedings of the 24th Annual International Conference on Mobile Systems, Applications and Services (MobiSys 2026), ACM, June 20, 2026 (Crossref record for DOI 10.1145/3745756.3809208; title also on the MobiSys 2026 accepted list)
- URL: https://dl.acm.org/doi/10.1145/3745756.3809208 (ACM page blocked in this session; metadata from https://api.crossref.org/works/10.1145/3745756.3809208 and https://www.sigmobile.org/mobisys/2026/accepted_papers/)
- Summary (from the ACM index abstract, partially verified): GPU stalls waiting on resources are a large share of token-generation latency; TurboInfer does millisecond-level GPU frequency scaling that lowers the clock during stalls and raises it during active compute, detecting stalls with PMU instruction counts.
- Key numbers: not obtained (abstract blocked).
- Relevance: the same GPU clock lever the target scheduler uses, driven at millisecond scale by stall detection; a battery-state cap sits above it as the maximum level TurboInfer may choose.

### 4.4 Energy-Efficient Small Language Model Inference for Mobile Agents via DVFS
- Authors: Jiesong Chen, Lixiang Han, Jiani Cao, Zhenjiang Li (City University of Hong Kong)
- Venue and year: Proceedings of MobiSys 2026 Workshops, ACM, June 20, 2026 (Crossref record for DOI 10.1145/3812836.3814753)
- URL: https://doi.org/10.1145/3812836.3814753 (ACM page blocked; metadata from https://api.crossref.org/works/10.1145/3812836.3814753)
- Summary: DVFS applied to small language model inference for agent workloads on phones (abstract not retrievable; partially verified).
- Key numbers: not obtained.
- Relevance: confirms DVFS for SLM agents is an active 2026 topic; the per-request battery-tier angle is not in the title.

### 4.5 MNN-AECS: energy optimization for LLM decoding via adaptive core selection
- Authors: Zhengxiang Huang, Chaoyue Niu, Zhaode Wang, Jiarui Xue, Hanming Zhang, Yugang Wang, Zewei Xin, Xiaotang Jiang, Chengfei Lv, Fan Wu, Guihai Chen
- Venue and year: arXiv 2506.19884, June 2025 (cs.OS)
- URL: https://arxiv.org/abs/2506.19884
- Summary: Decode is memory bound, so running it on low-power cores costs little speed; the engine selects cores at runtime without root or OS changes.
- Key numbers: 23% less energy than stock MNN over 7 devices and 4 datasets; 39% to 78% less than llama.cpp, executorch, mllm and MediaPipe; 12% to 363% faster than those engines; 5 Android and 2 iOS devices.
- Relevance: a no-root CPU-side energy lever that works on both platforms; pair it with output caps where GPU clock control is unavailable (iOS, unrooted Android).

### 4.6 Is Your NPU Ready for LLMs? (PowerBench)
- Authors: Guanyu Cai, Ruiming Tian, Lang Yang, Zhouhong Ren, Jinliang Yuan, Lingkun Li, Jiliang Wang
- Venue and year: arXiv 2607.05475, July 2026 (cs.AR)
- URL: https://arxiv.org/abs/2607.05475
- Summary: PowerBench attributes energy per backend (CPU, GPU, NPU) across five frameworks; NPUs win compute-bound prefill, CPUs win memory-bound decode; framework gaps grow up to 10x on NPUs.
- Key numbers: up to 40% energy wasted by poor scheduling configurations; up to 54.8% energy reduction on the NPU backend with tuned configurations.
- Relevance: backend choice per phase is a large lever; a battery-aware scheduler could route prefill and decode to different backends by tier.

### 4.7 Elastic On-Device LLM Service (ELMS)
- Authors: Wangsong Yin, Rongjie Yi, Daliang Xu, Gang Huang, Mengwei Xu, Xuanzhe Liu
- Venue and year: ACM MobiCom 2025 (arXiv comment "MobiCom'25"; DOI 10.1145/3680207.3765259)
- URL: https://arxiv.org/abs/2409.09071
- Summary: An on-device LLM service elasticizes the model (one-time neuron reordering yields sub-models) and the prompt (a dual-head compact model shortens prompts) to hit per-app latency SLOs.
- Key numbers: up to 14.83% absolute accuracy gain over baselines (10.45% average), under 1% TTFT switching overhead, under 100 offline GPU hours; energy not evaluated.
- Relevance: per-request quality scaling with sub-1% switch cost is practical on COTS phones; a battery tier could pick a smaller sub-model or shorter prompt alongside a smaller KV.

### 4.8 KVSwap: disk-aware KV cache offloading for long-context on-device inference
- Authors: Huawei Zhang, Chunwei Xia, Zheng Wang
- Venue and year: arXiv 2511.11907, November 2025 (v2 December 2025); on the MobiSys 2026 accepted list
- URLs: https://arxiv.org/abs/2511.11907 and https://www.sigmobile.org/mobisys/2026/accepted_papers/
- Summary: Only a small, changing subset of KV entries matters for generation; the full cache lives on disk, compact in-memory metadata predicts what to preload, and reads are shaped to the storage device.
- Key numbers: "higher throughput under tight memory budgets" (no figures in the abstract).
- Relevance: the closest published competitor to KV shrinking on phones; it trades UFS energy and latency for memory rather than dropping entries, and it is not battery-state aware.

### 4.9 LLM Inference at the Edge: Mobile, NPU and GPU trade-offs under sustained load
- Authors: Pranay Tummalapalli, Sahil Arayakandy, Ritam Pal, Kautuk Kundan
- Venue and year: arXiv 2603.23640, March 2026 (revised June 2026)
- URL: https://arxiv.org/abs/2603.23640
- Summary: Qwen2.5 1.5B under repeated inference on Raspberry Pi 5 with Hailo-10H, Galaxy S24 Ultra, iPhone 16 Pro and a laptop RTX 4050, focusing on sustainable rather than peak throughput.
- Key numbers: iPhone 16 Pro loses nearly half its throughput within two iterations; S24 Ultra hits a hard OS-enforced GPU frequency floor that terminates inference; RTX 4050 131.7 tok/s at 34.1 W; Hailo under 2 W at 6.9 tok/s.
- Relevance: phones cannot sustain peak decode and the OS will intervene; capping GPU clock up front avoids the cliff and the termination.

### 4.10 Are We There Yet? Efficiency of LLM applications on mobile devices
- Authors: Xiao Yan, Yi Ding (University of Texas at Dallas)
- Venue and year: Proceedings of the 2nd International Workshop on Foundation Models for Cyber-Physical Systems and Internet of Things (FMSys 2025), ACM, May 6, 2025 (Crossref); arXiv 2504.00002
- URLs: https://arxiv.org/abs/2504.00002 and https://dl.acm.org/doi/10.1145/3722565.3727192
- Summary: AutoLife-Lite, an LLM app inferring location and activity from phone sensors, measured on mobile, edge and cloud.
- Key numbers: only models under about 4B run on strong phones; mobile latency above 30 s versus under 10 s in the cloud; compression degrades output quality.
- Relevance: user-facing LLM apps already sit at tens of seconds per request on phones, so an output-length cap under low battery has a large absolute energy effect.

### 4.11 Prompt-level energy: EMNLP 2026 and "Keyword Matters"
- Authors: Wei Hu, Xiaolong Tu, Dawei Chen, Yitao Chen, Kyungtae Han, Haoxin Wang ("How Do Prompt Variations Affect Energy Consumption in On-Device LLMs?"); Ruiyi Tao, Xiaolong Tu, Haoxin Wang ("Keyword Matters: Unveiling the Energy Sensitivity of On-Device LLM Prompting")
- Venue and year: EMNLP 2026 main conference (camera-ready, arXiv 2609.01798, September 2026); arXiv 2607.22568, May 2026
- URLs: https://arxiv.org/abs/2609.01798 and https://arxiv.org/abs/2607.22568
- Summary: Phase-level profiling separates prefill and decode energy; cognitive load changes energy per token while phrasing changes energy mainly through token count; imperative keywords shift output length and total energy consistently across tasks.
- Key numbers: not in the abstracts.
- Relevance: token count is the dominant prompt-level energy term, which supports max-token capping as the first lever in a battery-aware scheduler.

### 4.12 Understanding LLMs in Your Pockets (COTS measurement study)
- Authors: Jie Xiao, Qianyi Huang, Xu Chen, Chen Tian
- Venue and year: arXiv 2410.03613, October 2024 (v5 February 2026)
- URL: https://arxiv.org/abs/2410.03613
- Summary: Measurement of lightweight LLMs on SoCs from major vendors covering throughput, latency, quality, resource use, OS strategies, battery drain and launch time.
- Key numbers: not in the abstract.
- Relevance: an early baseline of how OS governors and thermal policy shape on-device LLM behaviour.

### 4.13 Human token consumption speed and QoE-aware streaming: Andes, DiSCo, Streaming Fast and Slow
- Authors and venues: Andes (Jiachen Liu, Jae-Won Chung, Zhiyu Wu, Fan Lai, Myungjin Lee, Mosharaf Chowdhury; arXiv 2404.16283, April 2024, revised December 2024); DiSCo (Ting Sun, Penghan Wang, Fan Lai; ACL 2025, arXiv 2502.11417); Streaming, Fast and Slow (Chang Xiao, Brenda Yang; UIST 2025, arXiv 2504.17999)
- URLs: https://arxiv.org/html/2404.16283 , https://arxiv.org/abs/2502.11417 , https://arxiv.org/abs/2504.17999
- Summary: Andes defines text-streaming QoE over the whole interaction and states "user reading/listening speeds are 4.8 and 3.3 tokens/s" (its Figure 2, one word counted as 1.3 tokens); DiSCo repeats that 52% of users aged 25 to 44 read 4 to 5 tokens/s and audio is 3 to 4 tokens/s, and routes requests between device and server; Streaming Fast and Slow paces output to inferred cognitive load to save compute.
- Key numbers: 4.8 tokens/s reading, 3.3 tokens/s listening; Andes up to 4.7x average QoE or 61% GPU saving; DiSCo tail TTFT down 11% to 52% and cost down up to 84%.
- Relevance: a decode rate above about 5 tokens/s buys no perceived value, so a battery-tier GPU clock cap that holds 5 to 8 tokens/s is invisible to the reader while cutting power.

### 4.14 Energy-Based Fair Queuing scheduler for mobile systems
- Author: Deshpande Mandar Anil
- Venue and year: arXiv 2606.20602, May 2026 (cs.OS)
- URL: https://arxiv.org/abs/2606.20602
- Summary: Extends fair queuing to energy: the OS limits discharge rate to reach a target battery lifetime and shares the power budget proportionally across tasks, built on CFS.
- Key numbers: qualitative claims only in the abstract.
- Relevance: the OS-level analogue of a per-app energy budget; the per-request LLM scheduler is the app-level version that fits a request into its share.

### 4.15 Energy-Aware Process Scheduling in Linux (Wattmeter)
- Authors: Feitong Qiao, Yiming Fang, Asaf Cidon (Columbia University)
- Venue and year: ACM SIGEnergy Energy Informatics Review, Vol. 4, Issue 5, pp. 91 to 97 (issue dated December 2024 in Crossref; online 2025)
- URL: https://dl.acm.org/doi/10.1145/3727200.3727214 (ACM page blocked; metadata from https://api.crossref.org/works/10.1145/3727200.3727214; partially verified)
- Summary: Wattmeter uses eBPF to attribute energy per process at millisecond granularity without kernel changes, and demonstrates two policies: equalize energy across processes and cap a process's energy.
- Key numbers: not obtained from the blocked page.
- Relevance: a per-process energy cap in Linux is the primitive a phone OS would need to enforce a per-request energy budget; today it exists as research, not as an Android feature.

### 4.16 SERENUS: alleviating low-battery anxiety with app-level energy prediction
- Authors: Sera Lee, Dae R. Jeong, Junyoung Choi, Jaeheon Kwak, Seoyun Son, Jean Y. Song, Insik Shin
- Venue and year: ACM UIST 2024 (DOI 10.1145/3654777.3676437; dblp https://dblp.org/rec/conf/uist/LeeJCKSSS24.html)
- URL: https://dl.acm.org/doi/10.1145/3654777.3676437 (PDF https://daeryong.me/lee-serenus.pdf, binary only in this session)
- Summary: A phone framework predicts per-app energy in real time and shows it as remaining minutes; user studies show accurate predictions let users plan usage and reduce low-battery anxiety.
- Key numbers: in the PDF (not text-extractable here).
- Relevance: users cope with low battery by rationing; an LLM scheduler that states the cost of an answer and offers a shorter one matches that behaviour.

### 4.17 Running Low, Diversifying Less: battery depletion and consumer choice
- Authors: Yangyi Tang (Wuhan University), Xing Gao (Renmin University of China)
- Venue and year: Journal of Consumer Behaviour, published June 12, 2026 (Crossref for DOI 10.1002/cb.70189; Wiley page blocked)
- URL: https://onlinelibrary.wiley.com/doi/10.1002/cb.70189
- Summary: Six experiments (N above 2000) show a low battery reduces variety seeking through diminished perceived control; the effect weakens while charging, for users less attached to the device, or when choices restore control.
- Key numbers: N above 2000, six experiments.
- Relevance: low battery changes what users choose, not only how long they use the phone; a scheduler that visibly gives control back (a chosen shorter answer) should be better accepted than a silent quality cut.

### 4.18 Talker Research survey: battery worry starts at 38%
- Organization: Talker Research (poll for a commercial client; secondary but primary for the survey)
- Venue and year: talkerresearch.com, survey run January 31 to February 3, 2025
- URL: https://talkerresearch.com/poll-phone-at-38-thats-when-most-americans-start-to-panic/
- Summary: 2,000 Americans (1,000 men, 1,000 women) surveyed online; worry begins at 38% on average and earlier for younger cohorts.
- Key numbers: Gen Z 44%, Millennials 43%, Gen X 38%, Boomers 34%; 61% show the exact percentage.
- Relevance: the useful "low battery" tier for a scheduler starts near 40%, well above the OS's 20% Low Power trigger.

### 4.19 Understanding the Effects of Smartphone Battery Level Awareness on In-App Behavior (unverified venue)
- Organization: authors not confirmed; ResearchGate listing only
- Venue and year: unverified (dblp search returned no record); the study analyses battery logs of 137,311 iOS users from May and June 2021 with difference-in-differences and finds that low-battery alerts change in-app behaviour on a social platform
- URL: https://www.researchgate.net/publication/374943896_Understanding_the_Effects_of_Smartphone_Battery_Level_Awareness_on_In-App_Behavior
- Relevance: large-scale evidence that the 20% alert itself changes behaviour; keep as unverified until a venue is confirmed.

### 4.20 Related DVFS and SLM energy papers outside phones (brief)
- ZeroDVFS (Mohammad Pivezhandi, Mahdi Banisharif, Abusayeed Saifullah, Ali Jannesari; arXiv 2601.08166, January 2026): model-based RL plus LLM-extracted program features for core and frequency allocation on Jetson TX2, Orin NX, RubikPi and Core i7; 7.09x better energy efficiency, decisions in 358 ms. https://arxiv.org/abs/2601.08166
- Energy-Efficient GPU DVFS for Fine-Tuning of SLMs on embedded devices (Jurn-Gyu Park et al.; arXiv 2607.05933, July 2026): ML-selected DVFS points on Jetson AGX Orin save 13.11% on average (up to 26.73%). https://arxiv.org/abs/2607.05933
- Characterizing Energy Footprint and Efficiency of SLMs on Edges (Md Romyull Islam et al.; IEEE MASS 2025 per arXiv comment; arXiv 2511.11624): five SLMs on Raspberry Pi 5, Jetson Nano and Orin Nano. https://arxiv.org/abs/2511.11624
- Relevance: these confirm that learned or ML-selected DVFS points recover 10% to 25% energy on edge GPUs, which is the size of saving a battery-tier GPU cap can expect before output-length effects are added.

### 4.21 Venue scans (what was and was not found)
- MobiSys 2025 accepted papers (https://www.sigmobile.org/mobisys/2025/accepted_papers/): energy-related titles are EffVR (mobile VR energy), EdgeLoRA (multi-tenant LLM serving on edge), "Never Start from Scratch" (on-device LLM personalization), DAF (on-device training) and Shepherd Nova (energy-harvesting testbed). No battery-state-aware LLM scheduling.
- MobiSys 2026 accepted papers (https://www.sigmobile.org/mobisys/2026/accepted_papers/): "Act Before It's Too Late" (4.3), KVSwap (4.8), Agent-X, FBLayout, "A Greener Edge", BIONIC.
- MobiCom 2025 (dblp https://dblp.org/db/conf/mobicom/mobicom2025.html, excerpt only): ELMS (4.7) is the relevant on-device LLM paper found through its own record; the dblp excerpt shown was mostly sensing and wireless.
- SenSys 2025 (dblp https://dblp.org/db/conf/sensys/sensys2025.html, excerpt): "CheckMate: LLM-Powered Approximate Intermittent Computing" (Sayyid-Ali et al., DOI 10.1145/3715014.3722056) is the only LLM plus energy title seen; not smartphone.
- HotMobile 2025 program (https://hotmobile.org/2025/index.php?id=program): TinyMem (multi-DNN on tiny accelerators), GreenAuto (sustainable model design on edge devices), RheelPower, BioPulse. Nothing on phone battery policy.
- HotMobile 2026 program (https://www.hotmobile.org/2026/index.php?id=program, February 25 to 26, 2026): SAPE (cellular power equalization), NAS for latency-constrained DNNs, EarCalo. Nothing on phone battery policy.
- EuroSys 2026 (https://2026.eurosys.org/papers.html): "Scaling LLM Test-Time Compute with Mobile NPU on Smartphones", "TZ-LLM", "viNPU"; none is battery-state aware by title.
- IMWUT 2024 and 2025: the UbiComp paper lists (https://www.ubicomp.org/ubicomp-iswc-2025/imwut_papers/) do not enumerate titles on the fetched page; no 2024 to 2026 IMWUT battery-behaviour paper was confirmed.
- Conclusion: no paper found in 2024 to 2026 at these venues composes GPU clock cap, KV size and output length as a per-request function of battery state.

## 5. Unverified or partially verified items
- TurboInfer abstract and numbers (ACM page blocked; title, authors, venue and date verified via Crossref and the accepted list).
- "Energy-Efficient SLM Inference for Mobile Agents via DVFS" abstract (blocked; metadata via Crossref).
- Wattmeter numbers (blocked; metadata via Crossref).
- Journal of Consumer Behaviour article body (blocked; abstract via Crossref).
- Snapdragon 8 Elite Gen 5 sustained-load throttling percentages from PhoneArena and Android Headlines (blocked).
- Snapdragon 8 Elite Gen 6 cooling (rumour per igor'sLAB, February 7, 2026).
- Android 17 "Priority Charging" (seen in Beta 3 code per secondary reports; not user-visible; not in official notes).
- Samsung Light profile battery gains (user reports only).
- "Understanding the Effects of Smartphone Battery Level Awareness on In-App Behavior" venue (no dblp record).
- Apple Intelligence behaviour under Low Power Mode or thermal pressure: no official statement found; the Foundation Models framework has no battery or thermal unavailability reason (verified), so any such behaviour is undocumented.
- "A review of smartphone energy management: Power distribution, battery technologies, fast charging, thermal management and challenges" (ScienceDirect, March 11, 2026; page blocked, authors and journal not confirmed): https://www.sciencedirect.com/science/article/pii/S2590174526002187

## 6. Implications for a per-request energy-aware LLM scheduler

1. Signals to read on Android: `BatteryManager` CHARGE_COUNTER and CURRENT_NOW (per request), `isPowerSaveMode()`, `getThermalHeadroom(30)` at most every 10 s with `getThermalHeadroomThresholds()`, `getGpuHeadroom()` and `getCpuHeadroom()` on Android 16 and later, battery health and the 80% cap on Pixel. On iOS: `isLowPowerModeEnabled`, `thermalState`, and Adaptive Power state is not exposed.
2. Levers and where they live: GPU clock cap through ADPF hint sessions with `setPreferPowerEfficiency(true)` (unprivileged) or `/sys/class/kgsl/kgsl-3d0/max_pwrlevel` and `max_clock_mhz` (privileged, composes with the thermal cap by max()); CPU ceiling through `cpu.uclamp.max` with isolated decode threads; core selection without root (MNN-AECS); KV budget and max tokens inside the runtime.
3. Ordering by evidence: cap output length first (dominant energy term at prompt level, and the only lever on iOS), then cap accelerator and DDR clock (flat QoE curve down to about 5 to 8 tokens/s, given 4.8 tokens/s reading speed), then shrink KV (memory bandwidth and thermal, and it is the lever no platform provides).
4. Policy shape: tiers with hysteresis (OS precedent 20% on, 80% or 90% off), a "worry" tier near 40% (survey), prediction against a learned routine for background or long requests (Adaptive Power, Routine Battery Saver), and an exemption for short interactive requests (Adaptive Power exempts camera and Game Mode).
5. Measurement: Perfetto `android.power` with ODPM rails and Wattson on Pixel; coulomb counter with charging off elsewhere; report energy per request and per token, and log wake-lock time because Play now scores it.

## 7. All URLs fetched for this report
- https://docs.kernel.org/scheduler/sched-energy.html
- https://docs.kernel.org/power/energy-model.html
- https://docs.kernel.org/scheduler/sched-util-clamp.html
- https://docs.kernel.org/scheduler/schedutil.html
- https://developer.android.com/games/optimize/adpf
- https://developer.android.com/games/optimize/adpf/thermal
- https://developer.android.com/about/versions/15/features
- https://developer.android.com/about/versions/16/features
- https://developer.android.com/about/versions/17/features
- https://android-developers.googleblog.com/2026/06/Android-17.html
- https://developer.android.com/games/optimize/adpf/gamemode/gamemode-api
- https://developer.android.com/games/optimize/adpf/gamemode/gamemode-interventions
- https://developer.android.com/topic/performance/appstandby
- https://developer.android.com/topic/performance/power/power-details
- https://developer.android.com/about/versions/pie/power
- https://support.google.com/pixelphone/answer/6187458
- https://support.google.com/android/answer/7664692
- https://source.android.com/docs/core/power/routine-battery-saver
- https://source.android.com/docs/core/power/mgmt
- https://source.android.com/docs/core/power/device
- https://source.android.com/docs/core/power/power-stats-hal
- https://source.android.com/docs/core/power/wattson/how-to-wattson
- https://perfetto.dev/docs/data-sources/battery-counters
- https://developer.android.com/studio/profile/power-profiler
- https://developer.android.com/topic/performance/power/battery-historian
- https://developer.android.com/stories/games/kuro-powerprofiler
- https://android-developers.googleblog.com/2025/11/raising-bar-on-battery-performance.html
- https://android-developers.googleblog.com/2026/03/battery-technical-quality-enforcement.html
- https://9to5google.com/2025/06/11/android-16-pixel-battery-health-2/
- https://9to5google.com/2026/03/26/android-17-beta-3-everything-new/
- https://support.apple.com/en-us/123707
- https://support.apple.com/en-us/101604
- https://www.macrumors.com/how-to/ios-extend-iphone-battery-life-adaptive-power-mode/
- https://machinelearning.apple.com/research/apple-foundation-models-2025-updates
- https://arxiv.org/abs/2507.13575
- https://developer.apple.com/tutorials/data/documentation/foundationmodels/systemlanguagemodel/availability-swift.enum/unavailablereason.json
- https://raw.githubusercontent.com/qualcomm-linux/kgsl/gfx-kernel.le.0.0/kgsl_pwrctrl.c
- https://android.googlesource.com/kernel/msm/+/android-wear-5.0.2_r0.1/Documentation/devicetree/bindings/gpu/adreno-pwrlevels.txt
- https://deepwiki.com/qualcomm-linux/kgsl
- https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5
- https://www.qualcomm.com/news/releases/2025/09/snapdragon-8-elite-gen-5--the-world-s-fastest-mobile-system-on-a
- https://counterpointresearch.com/en/insights/qualcomm-snapdragon-8-elite-gen-5-redefines-mobile-performance-and-ai-efficiency
- https://www.notebookcheck.net/Qualcomm-Snapdragon-8-Elite-Gen-5-for-Galaxy-Processor-Benchmarks-and-Specs.1271123.0.html
- https://www.igorslab.de/en/qualcomm-breaks-new-ground-in-cooling-with-the-snapdragon-8-elite-gen-6/
- https://www.samsung.com/ca/support/mobile-devices/light-performance-profile-on-your-samsung-galaxy/
- https://docs.samsungknox.com/admin/knox-platform-for-enterprise/knox-service-plugin/configure-advanced-policies/data-processing-for-galaxy-ai/
- https://www.androidauthority.com/google-tensor-g5-benchmarks-3590355/
- https://www.mediatek.com/press-room/mediatek-dimensity-9500-unleashes-best-in-class-performance-ai-experiences-and-power-efficiency-for-the-next-generation-of-mobile-devices
- https://arxiv.org/abs/2507.02135
- https://proceedings.mlsys.org/paper_files/paper/2026/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html
- https://arxiv.org/abs/2606.23001
- https://api.crossref.org/works/10.1145/3745756.3809208
- https://api.crossref.org/works/10.1145/3812836.3814753
- https://www.sigmobile.org/mobisys/2025/accepted_papers/
- https://www.sigmobile.org/mobisys/2026/accepted_papers/
- https://arxiv.org/abs/2506.19884
- https://arxiv.org/abs/2607.05475
- https://arxiv.org/abs/2409.09071
- https://arxiv.org/abs/2511.11907
- https://arxiv.org/abs/2603.23640
- https://arxiv.org/abs/2504.00002
- https://api.crossref.org/works/10.1145/3722565.3727192
- https://arxiv.org/abs/2609.01798
- https://arxiv.org/abs/2607.22568
- https://arxiv.org/abs/2410.03613
- https://arxiv.org/html/2404.16283
- https://arxiv.org/html/2502.11417
- https://arxiv.org/abs/2502.11417
- https://arxiv.org/abs/2504.17999
- https://arxiv.org/abs/2502.16721
- https://arxiv.org/abs/2606.20602
- https://arxiv.org/abs/2605.24569
- https://api.crossref.org/works/10.1145/3727200.3727214
- https://dblp.org/rec/conf/uist/LeeJCKSSS24.html
- https://api.crossref.org/works/10.1002/cb.70189
- https://talkerresearch.com/poll-phone-at-38-thats-when-most-americans-start-to-panic/
- https://arxiv.org/abs/2601.08166
- https://arxiv.org/abs/2607.05933
- https://arxiv.org/abs/2511.11624
- https://dblp.org/db/conf/mobisys/mobisys2026.html
- https://dblp.org/db/conf/mobicom/mobicom2025.html
- https://dblp.org/db/conf/sensys/sensys2025.html
- https://hotmobile.org/2025/index.php?id=program
- https://www.hotmobile.org/2026/index.php?id=program
- https://2026.eurosys.org/papers.html
- https://www.ubicomp.org/ubicomp-iswc-2025/imwut_papers/
