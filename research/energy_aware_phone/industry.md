# Industry survey: energy-aware on-device AI and battery-aware behaviour (2025 to 2026)

Survey date: 2026-09-04. Method: 200 web searches plus about 70 page fetches. Every item below carries a URL that was fetched and a date confirmed on that page, unless it is marked "unverified". Items marked "partly verified" have one claim that only appears in a secondary source.

Target prototype for the relevance notes: an LLM running on a OnePlus 15 (Snapdragon 8 Elite Gen 5, Adreno 840, 7,300 mAh) that caps the GPU clock, shrinks the KV cache and caps output length by battery state.

## Headline findings

1. No shipped phone, SoC or OS feature changes on-device LLM behaviour by battery level. Apple's Adaptive Power (iOS 26) and Samsung's Network Battery Saver (One UI 8.5) are the closest: both are on-device predictors that slow tasks or limit radios, but neither touches AI models. Apple's Foundation Models availability API has three reasons for refusing the model and none is battery or thermal.
2. The developer-facing knobs that do exist on Snapdragon are NPU performance profiles (nine htp_performance_mode values, mirrored in ExecuTorch) and the kgsl GPU sysfs nodes (max_gpuclk, thermal_pwrlevel, max_pwrlevel). No LLM framework (llama.cpp, MLC-LLM, ExecuTorch, MediaPipe, LiteRT-LM) exposes a battery-aware mode.
3. The best published energy data for the exact target device is the PowerBench study (arXiv 2607.05475, July 2026). On a OnePlus 15, CPU decode of Qwen 1.5B costs about 97 mJ per token while the Hexagon NPU costs about 320 mJ per token. The study reports 40 percent energy waste from thread and frequency misconfiguration and up to 54.8 percent savings from an energy-oriented configuration.
4. OnePlus 15 thermal behaviour is documented: surface 47 to 52.7 C under sustained GPU load, an OS lockout that closes apps for about 15 minutes, and after the December 2025 update a fall to about 60 percent of peak by the end of a stress test.
5. None of the mainstream AI benchmarks (MLPerf Mobile v6.0, Geekbench AI 1.3 to 1.7, Qualcomm AI Hub profiling) report power on phones.
6. The research literature of 2025 to 2026 covers each prototype lever separately: frequency scaling (EnerInfer, Camel), core selection (MNN-AECS), output length (Brevity, Caveman), and KV cache capacity on phones (KVSwap, DynaKV). Nothing found combines them under a battery-state policy.

## Section 1: SoCs and vendor AI platforms

### 1.1 Qualcomm Snapdragon 8 Elite Gen 5 platform claims (verified)

- What: flagship mobile SoC with 3rd-gen Oryon CPU, Adreno 840 GPU, Hexagon NPU. Ships in the OnePlus 15.
- Who: Qualcomm. Date: press release 2025-09-24 (Snapdragon Summit, Maui).
- URLs: https://www.qualcomm.com/news/releases/2025/09/snapdragon-8-elite-gen-5--the-world-s-fastest-mobile-system-on-a ; https://m.gsmarena.com/hz5/qualcomm_snapdragon_8_elite_gen_5_soc_features_specs-news-69656.php (2025-09-25) ; https://counterpointresearch.com/en/insights/qualcomm-snapdragon-8-elite-gen-5-redefines-mobile-performance-and-ai-efficiency (2025-10-02)
- Description: Qualcomm positions the chip for "agentic AI" with an upgraded Sensing Hub and a Hexagon NPU that supports INT2 mixed precision and GenAI model encryption. The press release lists OnePlus among 15 launch OEMs.
- Numbers: CPU +20 percent performance and 35 percent better CPU power efficiency; GPU +23 percent performance, 20 percent better GPU efficiency, +25 percent ray tracing; NPU +37 percent; "up to 16 percent" overall power efficiency, which Qualcomm equates to 1 h 48 min extra gaming time (GSMArena). Press reports of an 18 MB "Adreno High Performance Memory" cache saving up to 10 percent GPU power were not found on any fetched Qualcomm page (partly verified).
- Relevance: the 20 percent GPU efficiency claim is the vendor baseline the prototype's GPU clock cap is measured against; Qualcomm exposes no user-facing power mode for AI on this chip.

### 1.2 Snapdragon 8 Elite Gen 5 measured power (verified)

- What: independent power measurements of the SoC.
- Who: Geekerwan, reported by Notebookcheck. Date: 2025-09-25.
- URL: https://www.notebookcheck.net/Qualcomm-Snapdragon-8-Elite-Gen-5-vs-Dimensity-9500-and-Apple-A19-Pro-in-efficiency-analysis.1123883.0.html
- Description: Geekerwan measured package power under full CPU load and under 3DMark Steel Nomad Light. The Snapdragon wins GPU efficiency but draws far more than the A19 Pro on CPU.
- Numbers: 19 W at full CPU load (8 Elite Gen 5) vs 17 W (8 Elite) vs 12 W (A19 Pro); about 12 W on the GPU test vs 13 to 15 W for A19 Pro and Dimensity 9500.
- Relevance: a 12 W GPU envelope in a passively cooled phone explains why OnePlus 15 hits lockouts; the prototype's GPU clock cap works inside this envelope.

### 1.3 Hexagon NPU performance profiles (verified)

- What: the Qualcomm AI Engine Direct (QNN) HTP backend exposes named performance profiles that trade NPU speed for power.
- Who: Qualcomm, surfaced through ONNX Runtime, ExecuTorch and Genie. Dates: ONNX Runtime doc (current), ExecuTorch schema (current), Genie walkthrough 2026-03-23.
- URLs: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html ; https://github.com/pytorch/executorch/blob/main/backends/qualcomm/serialization/qc_schema.py ; https://grapeup.com/blog/running-llms-on-device-with-qualcomm-snapdragon-8-elite
- Description: htp_performance_mode accepts burst, balanced, default, high_performance, high_power_saver, low_balanced, low_power_saver, power_saver and sustained_high_performance. ExecuTorch mirrors these as kHtpBurst, kHtpSustainedHighPerformance, kHtpPowerSaver and so on; Genie's htp_backend_ext_config.json picks burst or sustained_high_performance for thermal stability.
- Numbers: Genie on Snapdragon 8 Elite NPU: about 10 tok/s for Llama 3.2 3B and about 5 tok/s for Llama 3.1 8B (grapeup). No vendor power figures per mode.
- Relevance: this is the only vendor-documented power knob for AI on the target SoC, and it applies to the NPU only; the prototype's GPU path must use kgsl nodes instead.

### 1.4 Adreno GPU clock controls in kgsl sysfs (verified)

- What: kernel sysfs nodes that cap Adreno GPU frequency.
- Who: Qualcomm kgsl driver (public tree); third-party tool Pulse uses them without root on gaming handhelds.
- URLs: https://raw.githubusercontent.com/qualcomm-linux/kgsl/gfx-kernel.le.0.0/kgsl_pwrctrl.c ; https://github.com/keiretrogaming/pulse
- Description: kgsl_pwrctrl.c defines gpuclk, max_gpuclk, max_pwrlevel, min_pwrlevel, thermal_pwrlevel, default_pwrlevel, gpu_available_frequencies, force_clk_on and related nodes under /sys/class/kgsl/kgsl-3d0/. The max_gpuclk store converts a frequency to the nearest power level and applies it as a thermal limit; thermal_pwrlevel sets a thermal QoS frequency constraint.
- Numbers: Pulse holds a target FPS "at the lowest power possible" by writing max_pwrlevel and cpufreq scaling_max_freq; no absolute watts published. No date on the repo page.
- Relevance: these are exactly the nodes the prototype writes; the source confirms max_gpuclk is treated as a thermal limit, so the battery policy and the OEM thermal governor share one mechanism.

### 1.5 llama.cpp OpenCL backend for Adreno, Adreno 840 listed (verified)

- What: Qualcomm-contributed GPU backend in llama.cpp.
- Who: Qualcomm and ggml-org. Dates: blog December 2024, updated 2025-02-17; docs current.
- URLs: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md ; https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu
- Description: the backend is verified on Adreno 750, 810, 830, 840 (Snapdragon 8 Elite Gen 5), X1-85 and X2-90, with Q4_0 as the primary optimised format and Q1_0, Q4_K, Q6_K, MXFP4 also supported. It contains no power, battery or frequency options.
- Numbers: none published on the fetched pages.
- Relevance: confirms the prototype's engine path (llama.cpp on Adreno 840) is vendor-supported and that any battery-aware behaviour must be added above the engine.

### 1.6 Apple A19 Pro and iPhone 17 Pro (verified)

- What: Apple's 2025 flagship SoC with Neural Accelerators inside each GPU core.
- Who: Apple. Date: 2025-09-09. Independent tests: Argmax 2025-09-21, Draw Things 2025-09-19.
- URLs: https://www.apple.com/newsroom/2025/09/apple-unveils-iphone-17-pro-and-iphone-17-pro-max/ ; https://www.argmaxinc.com/blog/iphone-17-on-device-inference-benchmarks ; https://releases.drawthings.ai/p/iphone-17-pro-doubles-ai-performance
- Description: Apple says the 6-core GPU with Neural Accelerators and the 16-core Neural Engine deliver "up to 40 percent better sustained performance" with a new vapor chamber, and it names "running large local language models" as a use case. Argmax measured 2.5x to 3.1x GPU speedups over iPhone 16 Pro on Parakeet v3 while the Neural Engine gained only 1x to 1.15x.
- Numbers: Draw Things reports about 2x on FLUX and Qwen Image generation and notes iPhone 16 Pro throttles after about one minute at 1024x1024 while iPhone 17 Pro sustains; iPhone 17 Pro Max up to 39 h video.
- Relevance: Apple's answer to sustained on-device AI is hardware cooling, not a battery policy, which contrasts with the prototype's software-side approach.

### 1.7 Apple Foundation Models framework and the on-device 3B model (verified)

- What: system API giving apps the Apple Intelligence on-device LLM, plus the model's efficiency design.
- Who: Apple. Dates: announced 2025-06-09; model report 2025-06-09 (updated 2025-07-17); WWDC26 update June 2026.
- URLs: https://www.apple.com/newsroom/2025/06/apple-intelligence-gets-even-more-powerful-with-new-capabilities-across-apple-devices/ ; https://machinelearning.apple.com/research/apple-foundation-models-2025-updates ; https://developer.apple.com/videos/play/wwdc2026/241/ ; https://developer.apple.com/documentation/foundationmodels/systemlanguagemodel/availability-swift.enum/unavailablereason/devicenoteligible
- Description: the on-device model is about 3B parameters, quantised to 2 bits per weight with QAT, and shares block-2 KV caches with block 1 to cut KV memory by 37.5 percent. The WWDC26 update gives an 8,192-token context, token usage counters (input, cached, output, reasoning), a Private Cloud Compute model with 32K context and reasoning levels, and a LanguageModel protocol for swapping local or server models.
- Numbers: 37.5 percent KV cache memory reduction; 8,192-token on-device context; 32K cloud context. Availability reasons are deviceNotEligible, appleIntelligenceNotEnabled and modelNotReady; none is battery or thermal.
- Relevance: Apple already shrinks the KV cache structurally and counts output tokens, but exposes no battery-state control; the prototype's runtime KV shrink and output cap sit one level above what Apple ships.

### 1.8 Google Tensor G5, Gemini Nano and AICore (verified)

- What: Pixel 10 SoC and the on-device Gemini Nano stack.
- Who: Google. Dates: 2025-08-20 (chip), 2025-08-22 (ML Kit GenAI APIs).
- URLs: https://blog.google/products-and-platforms/devices/pixel/tensor-g5-pixel-10/ ; https://android-developers.googleblog.com/2025/08/the-latest-gemini-nano-with-on-device-ml-kit-genai-apis.html ; https://developer.android.com/ai/gemini-nano
- Description: Tensor G5 on TSMC 3 nm has a TPU up to 60 percent more powerful and runs Gemini Nano 2.6x faster and "2x more efficiently" for Pixel Screenshots and Recorder. AICore is the system service that runs Gemini Nano; its page states no battery or charging conditions for inference.
- Numbers: prefix speed 510 tok/s (Pixel 9 Pro) to 940 tok/s (Pixel 10 Pro); every Pixel 10 "over 30 hours" battery. A 32K token window claim appears only in secondary press (unverified).
- Relevance: Google quotes energy per feature as a vendor claim but gives developers no battery-state hook, so an app cannot ask AICore to run smaller or shorter when the battery is low.

### 1.9 Samsung Exynos 2600 and Galaxy AI on-device settings (verified, one number unverified)

- What: first 2 nm GAA phone SoC, in European Galaxy S26; plus Samsung's on-device-only AI toggle.
- Who: Samsung. Dates: 2025-12-18 (chip coverage), 2025-02-20 (toggle), 2026-03-12 (review), 2026-03-30 (battery test), 2026-06-15 (MLPerf).
- URLs: https://www.androidauthority.com/samsung-exynos-2600-announced-3626595/ ; https://9to5google.com/2025/02/20/how-to-turn-on-galaxy-ai-on-device-processing/ ; https://www.gsmarena.com/samsung_galaxy_s26-review-2942p4.php ; https://www.notebookcheck.net/Samsung-Galaxy-S26-with-Exynos-2600-fails-in-battery-test-compared-to-Snapdragon.1261969.0.html ; https://www.digitimes.com/news/a20260615PD207/samsung-exynos-performance-on-device-ai-mobile.html
- Description: Samsung claims a 113 percent NPU gain, SME2 on the CPU, a Heat Path Block that cuts thermal resistance up to 16 percent, and an ISP up to 50 percent more efficient. Settings > Galaxy AI > "Process data only on device" disables the cloud parts of Writing, Note, Browsing and Photo assist.
- Numbers: GSMArena stress test shows about 40 percent CPU and over 50 percent GPU loss under sustained load; a mixed-use battery test gave 6 h 48 min (Exynos 2600) vs 9 h 26 min (Snapdragon 8 Elite Gen 5), 28 percent longer for Snapdragon. DigiTimes reports "more than double" on-device AI performance vs Exynos 2500 in June 2026 MLPerf results; the 2.1x Mobile-BERT and 2.4x Stable Diffusion figures come from a page that could not be fetched (unverified).
- Relevance: Samsung gives users a privacy toggle but no energy toggle for AI, and the same SoC shows how large the throttling loss is on sustained load, which is the regime the prototype's clock cap targets.

### 1.10 MediaTek Dimensity 9500 NPU 990 (verified)

- What: MediaTek's 2025 flagship SoC with a ninth-generation NPU.
- Who: MediaTek. Date: 2025-09-22.
- URL: https://www.mediatek.com/press-room/mediatek-dimensity-9500-unleashes-best-in-class-performance-ai-experiences-and-power-efficiency-for-the-next-generation-of-mobile-devices
- Description: NPU 990 doubles compute, adds BitNet 1.58-bit model support and a "Super Efficient NPU" with compute-in-memory. UFS 4.1 four-channel storage is claimed to load large models 40 percent faster.
- Numbers: 1.58-bit processing cuts power up to 33 percent; 3B LLM output 100 percent faster; 128K token processing; 56 percent lower NPU power at peak; CPU ultra core up to 55 percent lower power at peak; GPU 42 percent better efficiency.
- Relevance: MediaTek's route to LLM energy is 1-bit weights on the NPU, which complements a KV-side and clock-side policy like the prototype's.

## Section 2: OS features that change behaviour by battery state or temperature

### 2.1 iOS 26 Adaptive Power (verified)

- What: on-device model that predicts heavy-use days and trims performance.
- Who: Apple. Dates: iOS 26 released 2025-09-15; support article 2025-12-04; MacRumors 2025-08-21 and 2025-09-05.
- URLs: https://support.apple.com/en-us/123707 ; https://www.macrumors.com/how-to/ios-extend-iphone-battery-life-adaptive-power-mode/ ; https://www.macrumors.com/guide/ios-26-battery-improvements/ ; https://www.apple.com/newsroom/2025/09/new-versions-of-apples-software-platforms-are-available-today/
- Description: Adaptive Power "makes performance adjustments", lowers brightness slightly, limits background activity and turns on Low Power Mode at 20 percent. It needs seven days of learning, is on by default on iPhone 17, 17 Pro, 17 Pro Max and iPhone Air, off by default on iPhone 15 Pro and 16 models, and pauses during camera use and Game Mode.
- Numbers: 20 percent Low Power trigger; 7-day learning window. iOS 26 also adds a charging-time estimate and per-app weekly battery views.
- Relevance: Adaptive Power is the closest shipped analogue to a battery-state policy, but it is a whole-device throttle that does not know about AI workloads; the prototype's per-inference levers are finer-grained.

### 2.2 Android 16 CPU/GPU headroom APIs and ADPF thermal headroom (verified)

- What: SystemHealthManager.getCpuHeadroom / getGpuHeadroom and PowerManager.getThermalHeadroom.
- Who: Google. Dates: Android 16 released 2025-06-10; thermal API since Android 11, thresholds API since Android 15.
- URLs: https://android-developers.googleblog.com/2025/06/android-16-is-here.html ; https://developer.android.com/about/versions/16/features ; https://developer.android.com/games/optimize/adpf/thermal
- Description: the headroom calls let an app read remaining CPU and GPU capacity over a chosen window (average or minimum), and Google says using them can "improve battery life". getThermalHeadroom(forecastSeconds) returns 0 to 1 and the guidance is to reduce workload immediately above 0.95 and watch above 0.85.
- Numbers: headroom thresholds 0.85 and 0.95; no per-app power budget API exists in Android 16 or 17.
- Relevance: these are the sanctioned Android signals the prototype can read to trigger its GPU cap and KV shrink without root, alongside battery level.

### 2.3 Android 16 Pixel battery health and charge limits (verified)

- What: user-facing battery controls in Android 16 on Pixel.
- Who: Google. Date: 2025-06-11.
- URL: https://9to5google.com/2025/06/11/android-16-pixel-battery-health-2/
- Description: a Battery health menu shows estimated capacity on Pixel 8a and newer and offers charging optimisation as off, Adaptive Charging or Limit to 80 percent. Battery health assistance steps down maximum voltage from 200 to 1,000 cycles and cannot be disabled on Pixel 9a and later.
- Numbers: 80 percent limit; 200 to 1,000 cycle voltage schedule.
- Relevance: the OS already exposes battery state and health to users, but none of it modulates app workloads, so an LLM app must implement its own policy.

### 2.4 Android 17: no new battery APIs, but new AI controls and thermal fixes (verified)

- What: Android 17 stable and QPR2 betas.
- Who: Google. Dates: Beta 1 2026-02-13; stable 2026-06-16; QPR2 Beta 4 2026-08-28.
- URLs: https://android-developers.googleblog.com/2026/02/the-first-beta-of-android-17.html ; https://www.androidauthority.com/android-17-3561251/ ; https://developer.android.com/about/versions/17/qpr2/release-notes ; https://9to5google.com/2026/08/31/android-17-qpr2-beta-4-everything-new/
- Description: the Beta 1 post lists generational garbage collection and a lock-free MessageQueue but no battery, thermal or ADPF changes, and the August 2026 feature roundup has none either. QPR2 Beta 4 fixed a thermal issue that caused overheating reboots and an 80 percent charge-limit bug, and added a Private Compute Core log, a Notification Intelligence menu with cooldown controls and an Agents dashboard.
- Numbers: none for power.
- Relevance: Android 17 gives users visibility and controls over AI agents but still no energy budget, so the prototype's policy remains app-level.

### 2.5 Samsung One UI 8.5 Network Battery Saver (verified)

- What: predictive radio limiter.
- Who: Samsung. Date: 2025-10-14 (shipping with Galaxy S26, February 2026).
- URL: https://www.sammobile.com/news/one-ui-8-5-may-offer-longer-battery-life-by-limiting-network-usage/
- Description: uses Personal Data Intelligence to predict idle periods such as sleep and limits network performance then. Samsung's text: "Save battery by limiting network performance when your phone isn't likely to be used."
- Numbers: none.
- Relevance: another example of an on-device predictor driving a power policy, applied to radios rather than AI.

### 2.6 OnePlus 15 and OxygenOS 16 thermal behaviour (verified)

- What: the target phone's OS, cooling and observed throttling.
- Who: OnePlus. Dates: OxygenOS 16 2025-10-16; reviews 2025-11-13 and 2025-11-14; fix test 2025-12-05; spec sheet 2026-08-13.
- URLs: https://www.notebookcheck.net/OnePlus-just-released-OxygenOS-16-with-new-AI-features-and-surprising-Apple-ecosystem-integration.1140255.0.html ; https://hothardware.com/reviews/oneplus-15-review?page=2 ; https://www.androidauthority.com/oneplus-15-benchmarks-3614813/ ; https://www.androidpolice.com/we-tried-to-make-the-oneplus-15-overheat-heres-what-happened/ ; https://www.androidauthority.com/oneplus-15-update-benchmarks-3621249/ ; https://www.techrepublic.com/article/news-oneplus-15-cheat-sheet/
- Description: OxygenOS 16 ships Plus Mind with Gemini integration, AI Writer and AI VoiceScribe; the article does not say which parts run on device, and no AI-specific power setting is described. Reviews found the phone reached 52.7 C, shut down a stress benchmark two-thirds through and blocked all apps except calls and messages until cool; Android Police saw 47 to 50 C with one 15-minute lockout, and HotHardware saw the same test fail at 72 F ambient but pass at 68 F.
- Numbers: 7,300 mAh silicon-carbon battery, 80 W wired (US), 50 W wireless, 12 or 16 GB RAM, dual-layer vapor chamber; 3DMark stability 79.8 percent (Steel Nomad Light), 66.4 percent (Solar Bay), 76.9 percent (Wild Life); PCMark battery 26 h 16 min; after the update, peak internal 47 C and about 60 percent of peak score by test end.
- Relevance: the OEM's response to heat is a hard lockout and a flatter thermal curve, so a prototype that caps its own GPU clock before 47 C can avoid the OS killing the inference session.

### 2.7 HarmonyOS 6 (verified)

- What: Huawei's OS release with system-level AI and battery claims.
- Who: Huawei. Dates: 2025-10-22 and 2025-10-23.
- URLs: https://technode.com/2025/10/23/huawei-rolls-out-harmonyos-6-ahead-of-next-months-mate-80-launch/ ; https://www.fonearena.com/blog/467247/huawei-harmonyos-6-features.html
- Description: Huawei claims 15 percent better performance than HarmonyOS 5 and battery life gains of 35 to 51 minutes from system optimisation, with the Xiaoyi assistant upgraded to multi-step agent tasks. The Mate 80 pairs it with a 6,000 mAh battery and, for the first time, an active cooling fan.
- Numbers: 35 to 51 min battery gain; 15 to 40 percent performance gain by prior version.
- Relevance: Huawei, like Apple, adds cooling hardware for sustained AI rather than a battery-aware AI policy.

### 2.8 Google Private AI Compute: hybrid routing by capability, not battery (verified)

- What: cloud extension for Pixel AI features.
- Who: Google. Date: 2025-11-11.
- URL: https://9to5google.com/2025/11/11/google-private-ai-compute-pixel/
- Description: Magic Cue and Recorder summaries now use Gemini cloud models in a sealed environment "for more timely suggestions", while on-device Gemini Nano remains for other tasks. Google says the routing exists because some tasks exceed on-device capacity.
- Numbers: none.
- Relevance: the shipped hybrid routers pick cloud by task capability and privacy, never by battery level, which leaves the prototype's battery-driven degradation path unoccupied.

## Section 3: Measured battery cost of on-device AI

### 3.1 Gemma 3 270M battery claim (verified)

- Who: Google. Date: 2025-08-14. URL: https://developers.googleblog.com/en/introducing-gemma-3-270m/
- Description: Google's internal test of the INT4 QAT model on a Pixel 9 Pro. It is the only vendor figure found that states battery percent per conversation.
- Numbers: 0.75 percent of battery for 25 conversations.
- Relevance: a per-conversation battery budget is a natural unit for the prototype's output-length cap.

### 3.2 PowerBench study on OnePlus 15 and Xiaomi 17 (verified)

- What: "Is Your NPU Ready for LLMs? Dissecting the Hidden Efficiency Bottlenecks in Mobile LLM Inference".
- Who: Cai, Tian, Yang, Ren, Yuan, Li, Wang. Date: 2026-07-06. URL: https://arxiv.org/abs/2607.05475 (HTML: https://arxiv.org/html/2607.05475)
- Description: measures llama.cpp, MNN, MLC-LLM, MLLM and Qualcomm Genie on CPU, GPU and NPU across four Snapdragon phones (OnePlus 15 and Xiaomi 17 on SM8850, Xiaomi 15, Xiaomi 14) using per-rail Qualcomm power telemetry. NPUs win prefill, CPUs win decode, and framework choice changes results up to 10x on the NPU.
- Numbers: Genie prefill 1,463.7 tok/s vs llama.cpp 115.1 tok/s on Qwen 1.5B; CPU decode 51.6 to 72.2 tok/s vs NPU about 23 tok/s; decode energy on OnePlus 15 about 97,000 uJ per token (CPU) vs about 320,000 uJ per token (NPU); up to 40 percent energy wasted by thread misconfiguration; energy-oriented config saves up to 54.8 percent.
- Relevance: this is the only peer-style energy dataset on the exact target phone and it supplies a joules-per-token baseline and a warning that the NPU is not the low-energy decode path.

### 3.3 EnerInfer (verified)

- What: "EnerInfer: Energy-Aware On-Device LLM Inference".
- Who: Zou, Liu, Sun, Mascherin, Roy, Liu, Peng, Jia, Haibo Chen. Date: 2026-06-22 (v2 2026-06-24). URLs: https://arxiv.org/abs/2606.23001 ; https://arxiv.org/html/2606.23001v2
- Description: lowers NPU and DDR frequencies with a model-structure-aware predictor and online feedback that also predicts skin temperature. It keeps back-shell temperature under 42 C and holds a QoE target.
- Numbers: up to 65 percent better energy efficiency on phones, 12 percent on a laptop, 24 percent on an Orange Pi 5 Pro; up to 11 percent whole-device energy vs default governors; 27.9 percent more tokens and 32.1 percent longer runs under thermal limits; 44 percent NPU plus memory power cut on a high-end phone with Qwen2-1.5B Q4.
- Relevance: the closest research match to the prototype's clock cap, but it scales NPU and DDR rather than the GPU and does not use battery level as an input.

### 3.4 LLM inference at the edge under sustained load (verified)

- Who: Tummalapalli, Arayakandy, Pal, Kundan. Date: 2026-03-24 (v2 2026-06-07). URL: https://arxiv.org/abs/2603.23640
- Description: sustained-load tests on Galaxy S24 Ultra, iPhone 16 Pro, Raspberry Pi 5 with Hailo-10H and an RTX 4050 laptop. The authors conclude that thermal management, not peak compute, is the binding constraint on phones.
- Numbers: iPhone 16 Pro loses nearly half its throughput within two iterations; S24 Ultra hits a hard OS-enforced GPU frequency floor that terminates inference; RTX 4050 34.1 W at 131.7 tok/s; Hailo under 2 W at 6.9 tok/s.
- Relevance: documents an OEM GPU floor killing inference outright, the failure mode the prototype's own cap is meant to pre-empt.

### 3.5 MNN-AECS adaptive core selection (verified)

- Who: Alibaba MNN team (Huang, Niu, Wang et al.). Date: 2025-06-24. URL: https://arxiv.org/abs/2506.19884
- Description: picks CPU cores for the memory-bound decode phase to cut energy without slowing generation. Merged into MNN; needs no root.
- Numbers: 23 percent energy cut vs stock MNN with no slowdown; 39 to 78 percent vs llama.cpp, ExecuTorch, mllm and MediaPipe on 5 Android and 2 iOS phones. The paper also notes MNN's Power_Low single-thread mode is too slow for LLMs.
- Relevance: a shipped engine-level energy feature, but it is static with respect to battery state.

### 3.6 Sustainability Is Not Linear (verified)

- Who: Ehsani, Giamattei, Malavolta, Pietrantuono. Date: 2026-03-27 (v2 2026-05-20). URL: https://arxiv.org/abs/2603.26603
- Description: eight LLMs from 0.5B to 9B on a flagship Android phone. Finds that importance-aware quantisation cuts memory but not energy, and that MoE models give 7B capacity at 1B to 2B energy.
- Numbers: Qwen2.5-3B named as the quality-per-joule sweet spot.
- Relevance: supports a small-model routing tier below the prototype's main model when battery is low.

### 3.7 Press measurements of AI battery cost (partly verified)

- URLs and dates: https://itsfoss.com/android-on-device-ai/ (2025-09-15) ; https://www.androidpolice.com/ai-battery-killers/ (2026-03-23) ; https://www.notebookcheck.net/Samsung-Galaxy-S26-with-Exynos-2600-fails-in-battery-test-compared-to-Snapdragon.1261969.0.html (2026-03-30)
- Description: It's FOSS ran MLC Chat, SmolChat and Google AI Edge Gallery on a Snapdragon 8 Gen 2 phone and reported "50 percent battery gone in under 90 minutes" in a first session, with Llama 3 class 3B to 4B models at 8 to 10 tok/s. Android Police lists Gemini Nano, Pixel Screenshots analysis, Live Translate, Now Playing and Smart Reply as drains and recommends disabling each individually; it cites a survey where only 11 percent of US buyers upgrade for AI and 54 percent rank battery first.
- Apple Intelligence: ZDNET's claim that it made an iPhone "impossible to make it through a day" and forum figures of about 20 to 25 percent better endurance with it off, or 6 h to over 9 h screen time on an iPhone 15 Pro Max, could not be fetched from a dated primary page (unverified; https://ios.gadgethacks.com/how-to/the-ai-that-wont-stay-off-mastering-your-iphones-most-persistent-feature/ is undated and anecdotal).
- Copilot on phones: no phone-specific battery measurement found; coverage is limited to Copilot+ PCs (not applicable).
- Relevance: press reporting frames AI as a battery cost users want to switch off, which is the user need a battery-state policy serves.

## Section 4: Frameworks and benchmarks

### 4.1 ExecuTorch 1.0 (verified)

- Who: PyTorch. Date: 2025-10-24. URLs: https://pytorch.org/blog/introducing-executorch-1-0/ ; https://docs.pytorch.org/executorch/main/backends-qualcomm.html
- Description: production status for the Qualcomm Hexagon NPU delegate, Core ML, XNNPACK with Arm Kleidi, Vulkan and Arm Ethos-U; new Samsung Exynos NPU/GPU and MediaTek backends. The Qualcomm backend schema exposes the nine HTP performance modes but the docs contain no DCVS, thermal or battery option.
- Numbers: none on power.
- Relevance: ExecuTorch is the only mainstream framework that exposes a vendor power profile, and only for the NPU.

### 4.2 LiteRT-LM and the MediaPipe LLM API sunset (verified)

- Who: Google. Dates: LiteRT-LM post 2026-05-19; Qualcomm NPU post 2025-11-24; MediaPipe page current.
- URLs: https://developers.googleblog.com/blazing-fast-on-device-genai-with-litert-lm/ ; https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/ ; https://developers.google.com/edge/mediapipe/solutions/genai/llm_inference/android
- Description: LiteRT-LM is the runtime behind Gemini Nano in Chrome, ChromeOS and Pixel Watch, with CPU, GPU and Android NPU backends, session save and restore of the KV cache, and multi-token prediction. The MediaPipe LLM Inference API is in maintenance mode; its only length control is maxTokens (default 512, input plus output) and it has no power option.
- Numbers: 52 tok/s decode on Android GPU, 56 tok/s on iOS, 2.2x with multi-token prediction, 607 MB footprint for Gemma 4 E2B; on Snapdragon 8 Elite Gen 5 the NPU is "up to 100x" faster than CPU and 10x faster than GPU on LiteRT models, FastVLM-0.5B decodes over 100 tok/s.
- Relevance: Google's runtime manages KV cache persistence and token limits but offers no battery hook; the maxTokens default of 512 is a precedent for capping output length.

### 4.3 MLC-LLM configuration (verified)

- Who: MLC AI. Date: docs current. URL: https://llm.mlc.ai/docs/deploy/mlc_chat_config.html
- Description: mlc-chat-config.json carries compile-time context_window_size, sliding_window_size and prefill_chunk_size, which fix KV cache capacity. No power, battery or thermal option exists.
- Numbers: example values 4096, -1, 4096.
- Relevance: KV capacity in MLC is a compile-time constant, so the prototype's runtime KV shrink has no equivalent there.

### 4.4 Qualcomm AI Hub profiling and Genie (verified)

- Who: Qualcomm. Dates: docs current (copyright 2026); Genie tutorial current.
- URLs: https://workbench.aihub.qualcomm.com/docs/hub/howitworks.html ; https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie
- Description: AI Hub profile jobs report compile time, load times, inference time and peak memory, and do not report power or energy. The Genie LLM tutorial supports Snapdragon 8 Elite Gen 5 and configures context length, threads, core affinity and the HTP performance profile.
- Numbers: 16 GB memory needed for 7B or 4096-context models.
- Relevance: Qualcomm's own cloud profiler cannot tell a developer the energy cost of a configuration, so on-device rail measurement (as in PowerBench) remains necessary.

### 4.5 Benchmarks: MLPerf Mobile v6.0, Geekbench AI, MLPerf Power (verified, one detail unverified)

- URLs and dates: https://mlcommons.org/2026/06/mlperf-mobile-v6/ (2026-06-15) ; https://www.primatelabs.com/blog/ (Geekbench AI 1.3 2025-03-17, 1.4 2025-06-30, 1.5 2025-09-05, 1.7 2026-02-11) ; https://arxiv.org/abs/2410.12032 (MLPerf Power, v2 2025-02-06)
- Description: MLPerf Mobile v6.0 adds Llama 3.2 1B and 3B and Llama 3.1 8B, with the 8B model NPU-accelerated on Snapdragon 8 Elite Gen 5, and adds Dimensity 9500 and Exynos 2600 support; it reports no power. None of the Geekbench AI releases mention power or battery.
- Numbers: none for power. The claim that MLPerf Power excludes phones because they cannot be externally powered without altering behaviour comes from a search summary, not the fetched abstract (unverified).
- Relevance: no standard benchmark reports joules per token on phones, so the prototype's own energy numbers cannot be compared to a public leaderboard.

### 4.6 Apple MLX and Neural Accelerators (partly verified)

- URLs and dates: https://machinelearning.apple.com/research/exploring-llms-mlx-m5 (2025-11-19) ; Argmax 2025-09-21 as above.
- Description: Apple shows MLX using M5 Neural Accelerators for up to 4x faster time-to-first-token vs M4 and 19 to 27 percent faster decode from 153 GB/s bandwidth. The page covers M5 only; Argmax reports MLX maintainers would add A19 support "in the coming weeks", which was not confirmed on an Apple page (unverified for iOS).
- Relevance: confirms prefill, not decode, is where GPU matrix accelerators help; decode stays bandwidth-bound, which is the phase a KV shrink attacks.

## Section 5: New ideas under discussion (2025 to 2026)

### 5.1 Output length as an energy lever (verified)

- Papers: "Brevity is the soul of sustainability" (Poddar et al., 2025-06-10, ACL 2025 Findings) https://arxiv.org/abs/2506.08686 ; "From Caveman to Expert Analyst" (Manya et al., 2026-07-02) https://arxiv.org/abs/2608.12350
- Description: Brevity benchmarks 12 decoder models on 5 datasets, finds responses far longer than needed, and cuts energy 25 to 60 percent by prompting for shorter answers. Caveman evaluates user-side behaviours and finds non-reasoning models use about one-twentieth the energy of reasoning models.
- Numbers: 25 to 60 percent (Brevity); up to 65 percent from prompt changes and 4 to 35 percent from low-intrusion practices (Caveman).
- Relevance: direct evidence that the prototype's output cap is a first-order energy lever, though both papers measure servers, not phones.

### 5.2 KV cache capacity on phones (verified, venue year unverified)

- Papers: KVSwap (Zhang, Xia, Wang; arXiv 2025-11-14, v2 2025-12-11) https://arxiv.org/abs/2511.11907 with lab page https://issl-uk.com/publication/mobisys-26/ ; DynaKV (Wang et al., 2025-10-20) https://arxiv.org/abs/2511.07427
- Description: KVSwap keeps the full KV cache on flash and predicts which entries to preload so long-context inference fits in phone memory. DynaKV adapts clustered KV retrieval with flash tiering for long-sequence decoding on smartphones.
- Numbers: DynaKV 1.38x retrieval accuracy and 1.47x latency reduction. KVSwap's lab page lists the 24th MobiSys; the fetched summary said 2025 while the page path says 2026 (venue year unverified).
- Relevance: both treat KV capacity as a memory problem; neither ties KV size to battery or thermal state, which is the prototype's angle.

### 5.3 Device-cloud routing without a battery signal (verified)

- Papers: "Bridging On-Device and Cloud LLMs" (Fang et al., 2025-09-28, v4 2026-05-23) https://arxiv.org/abs/2509.24050 ; HybridFlow (Dong et al., 2025-12-11) https://arxiv.org/abs/2512.22137
- Description: the first trains the on-device model to decide itself when to offload; the second routes DAG subtasks by a benefit-cost model over accuracy, token cost and latency. Neither uses battery or energy as a routing input.
- Numbers: none on energy.
- Relevance: battery-state routing is an open slot in the 2025 to 2026 routing literature.

### 5.4 Frequency and batch tuning for edge LLMs (verified)

- Paper: Camel (Xu et al., 2025-08-07) https://arxiv.org/abs/2508.09173
- Description: searches GPU frequency and batch size on an NVIDIA Jetson AGX Orin to balance latency and energy.
- Numbers: energy-delay product cut 12.4 to 29.9 percent for Llama3.2-1B and Qwen2.5-3B.
- Relevance: same lever as the prototype's GPU cap, on a board with a fan rather than a phone.

### 5.5 Small-model and profile switching in Apple's framework (verified)

- Source: WWDC26 session 241 (June 2026) https://developer.apple.com/videos/play/wwdc2026/241/
- Description: Dynamic Profiles let one session switch instructions, tools and model per task, and the cloud model's reasoning level is chosen per request via contextOptions. Token usage including reasoning tokens is reported back to the app.
- Numbers: 8,192 on-device vs 32,000 cloud context.
- Relevance: Apple ships the plumbing for cost-tiered model choice; a battery-aware profile is a small step an app could add, but Apple has not.

### 5.6 User controls over AI activity (verified) and per-feature "AI minutes" budgets (unverified)

- Sources: Android 17 QPR2 Beta 4 (2026-08-31) https://9to5google.com/2026/08/31/android-17-qpr2-beta-4-everything-new/ ; Samsung toggle (2025-02-20) as above; TechTimes 2026-02-18 https://www.techtimes.com/articles/314690/20260218/ai-smartphones-2026-ultimate-smartphone-performance-battery-life-guide.htm
- Description: Android 17 adds a Private Compute Core log, notification-summary cooldown controls and an Agents dashboard; Samsung offers on-device-only processing. A TechTimes guide describes "per-app AI performance budgets" of AI minutes per day and scheduling AI maintenance to charging time, but the page returned 403 and names no shipping product (unverified).
- Relevance: user controls today are about privacy and attention, not energy; a per-feature battery budget remains a proposal.

## Gaps this survey found

- No phone maker, SoC vendor or OS in 2025 to 2026 ships a feature that changes LLM model size, context size, output length or accelerator clock as a function of battery level.
- No LLM framework has a battery-aware or thermal-aware mode; the only power knobs are Qualcomm NPU profiles and kernel sysfs nodes.
- No public benchmark reports phone power for LLM inference; the only per-rail dataset on the OnePlus 15 is the July 2026 PowerBench paper.
- The research levers exist separately (frequency, cores, KV capacity, output length) but no work found combines them under a battery-state controller on a phone.

## Summary table

| # | Item | Ships or discussed by | Date | Status |
|---|------|----------------------|------|--------|
| 1.1 | Snapdragon 8 Elite Gen 5 claims | Qualcomm | 2025-09-24 | verified (HPM detail partly) |
| 1.2 | 8 Elite Gen 5 measured watts | Geekerwan / Notebookcheck | 2025-09-25 | verified |
| 1.3 | Hexagon NPU performance profiles | Qualcomm via ONNX RT, ExecuTorch, Genie | current, 2026-03-23 | verified |
| 1.4 | kgsl GPU sysfs nodes | Qualcomm kgsl, Pulse | current | verified |
| 1.5 | llama.cpp OpenCL Adreno 840 | Qualcomm, ggml | 2025-02-17, current | verified |
| 1.6 | Apple A19 Pro sustained AI | Apple, Argmax, Draw Things | 2025-09 | verified |
| 1.7 | Apple Foundation Models 3B | Apple | 2025-06-09, 2026-06 | verified |
| 1.8 | Tensor G5, Gemini Nano, AICore | Google | 2025-08-20/22 | verified (32K unverified) |
| 1.9 | Exynos 2600, Galaxy AI toggle | Samsung | 2025-02 to 2026-06 | verified (2.1x/2.4x unverified) |
| 1.10 | Dimensity 9500 NPU 990 | MediaTek | 2025-09-22 | verified |
| 2.1 | iOS 26 Adaptive Power | Apple | 2025-09-15, 2025-12-04 | verified |
| 2.2 | Android 16 headroom, ADPF thermal | Google | 2025-06-10 | verified |
| 2.3 | Android 16 Pixel battery health | Google | 2025-06-11 | verified |
| 2.4 | Android 17 AI controls, thermal fixes | Google | 2026-02 to 2026-08 | verified |
| 2.5 | One UI 8.5 Network Battery Saver | Samsung | 2025-10-14 | verified |
| 2.6 | OnePlus 15 thermal lockouts | OnePlus, reviewers | 2025-10 to 2026-08 | verified |
| 2.7 | HarmonyOS 6 | Huawei | 2025-10-22/23 | verified |
| 2.8 | Private AI Compute routing | Google | 2025-11-11 | verified |
| 3.1 | Gemma 3 270M 0.75 percent | Google | 2025-08-14 | verified |
| 3.2 | PowerBench on OnePlus 15 | arXiv 2607.05475 | 2026-07-06 | verified |
| 3.3 | EnerInfer | arXiv 2606.23001 | 2026-06-22 | verified |
| 3.4 | Edge sustained-load study | arXiv 2603.23640 | 2026-03-24 | verified |
| 3.5 | MNN-AECS | arXiv 2506.19884 | 2025-06-24 | verified |
| 3.6 | Sustainability Is Not Linear | arXiv 2603.26603 | 2026-03-27 | verified |
| 3.7 | Press battery-cost reports | It's FOSS, Android Police, Notebookcheck | 2025-09 to 2026-03 | partly verified (Apple Intelligence figures unverified) |
| 4.1 | ExecuTorch 1.0 | PyTorch | 2025-10-24 | verified |
| 4.2 | LiteRT-LM, MediaPipe sunset | Google | 2025-11-24, 2026-05-19 | verified |
| 4.3 | MLC-LLM KV config | MLC AI | current | verified |
| 4.4 | AI Hub profiling, Genie | Qualcomm | current | verified |
| 4.5 | MLPerf Mobile v6.0, Geekbench AI | MLCommons, Primate Labs | 2025-03 to 2026-06 | verified (MLPerf Power detail unverified) |
| 4.6 | MLX Neural Accelerators | Apple | 2025-11-19 | partly verified for iOS |
| 5.1 | Output-length energy papers | ACL 2025, arXiv 2608.12350 | 2025-06, 2026-07 | verified |
| 5.2 | KVSwap, DynaKV | arXiv | 2025-10, 2025-11 | verified (venue year unverified) |
| 5.3 | Device-cloud routing papers | arXiv | 2025-09, 2025-12 | verified |
| 5.4 | Camel | arXiv 2508.09173 | 2025-08-07 | verified |
| 5.5 | Apple dynamic profiles | Apple WWDC26 | 2026-06 | verified |
| 5.6 | AI activity controls, AI-minutes budgets | Google, Samsung, TechTimes | 2025 to 2026 | controls verified, budgets unverified |

## Verification log

- Fetched and confirmed: 68 pages (vendor press releases, developer docs, kernel source, arXiv abstracts and HTML, review sites).
- Could not fetch: Qualcomm product brief PDF (binary only), Qualcomm HTP and DCVS doc pages (JavaScript shell), Apple UnavailableReason page (JavaScript shell, confirmed via secondary page), ZDNET (blocked), Gizmochina and Sammy Fans (403), TechTimes (403), OnePlus specs page (404).
- Search budget exhausted at 200 searches; all final claims rest on fetched pages or are marked unverified.
