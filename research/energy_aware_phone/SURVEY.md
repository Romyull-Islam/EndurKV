# How energy-aware phone usage works, and where μKV sits

Survey date: 4 September 2026. Built from four verified sub-surveys in this directory: `os_platform.md` (kernel, Android, iOS, vendors), `industry.md` (what shipped in 2025 and 2026), `llm_energy.md` (measured LLM energy on phones), `control_rl.md` (learning and control methods). Every number below carries a bracketed reference to the list at the end. Items the sub-surveys could not verify from a primary page are marked as such there and are not used here.

## 1. The short answer

A phone manages energy in four layers, and none of them knows about an LLM request.

1. **Kernel.** Energy Aware Scheduling places tasks by a per-CPU energy model, schedutil sets the clock from utilization with a 25 percent margin, and uclamp gives user space a per-task clock floor and ceiling [1, 2, 3, 4]. EAS switches itself off above 80 percent utilization, which is where LLM decode runs [1]. The GPU is outside all of this.
2. **Platform.** Android exposes hint sessions, a thermal headroom forecast, CPU and GPU headroom estimates, a Game Mode battery tier, and Battery Saver tiers at fixed percentages [5, 6, 7, 8, 9]. iOS 26 adds Adaptive Power, the first mainstream policy that acts on a predicted shortfall against a learned seven-day routine [13]. Android 17 (June 2026) adds no per-app energy budget [11].
3. **Vendor firmware.** On Snapdragon the GPU clock is governed by DCVS inside the GMU firmware; the only external levers are the kgsl sysfs nodes, and the driver composes a user cap with the thermal cap by max(), so the two cannot race [16]. Each GPU power level also fixes a DDR bus vote, so a GPU cap caps memory bandwidth too [17]. NPUs expose nine named performance profiles [18].
4. **Application.** No shipped phone, SoC or OS feature changes LLM model size, context, output length or accelerator clock by battery level, and no LLM framework has a battery mode [20, 21, 22, 23]. Apple's on-device model has no battery or thermal unavailability reason [15].

Research supplies each lever on its own: accelerator and DDR clock under a QoE or thermal target [31, 32], core selection [33], KV capacity on phones [36, 37], output length [40, 41]. Nothing found composes them as a per-request function of battery state. EnerInfer, the closest system, says it does not consider battery level or charging [31]. That composition, with measured exchange rates, is what μKV's scheduler does.

## 2. How the phone decides today

**Battery-state policy is a threshold with hysteresis.** Android Battery Saver turns on at a user percentage and off at 90 percent; Extreme Battery Saver pauses apps and "slows processing speed" [8]. iOS Low Power Mode turns on at 20 percent and off at 80 percent [14]. Routine Battery Saver lets an OEM app set a dynamic hint with a default disable threshold of 80 percent [8]. Adaptive Power is the only predictor: it needs seven days of charging history, "makes performance adjustments", and is switched off during camera use and Game Mode [13].

**Users act earlier than the OS.** The average American starts to worry at 38 percent [26]. Low battery measurably reduces variety seeking in six experiments with over two thousand people, and the effect weakens while charging [27]. Per-app energy forecasts shown as remaining minutes reduce that anxiety [28]. So the useful "low" tier starts near 40 percent, not 20, and a visible choice such as a shorter answer is better accepted than a silent quality cut.

**Human token rate bounds the useful decode rate.** Reading is about 4.8 tokens per second and listening 3.3 [29]. Decode above about 5 tokens per second buys nothing perceptible in a streaming reply, which is why a decode clock cap can be invisible.

**Energy accounting.** Pixel phones expose measured rails through the Power Stats HAL, Perfetto and Wattson (1 point average error) [10]. Every other phone offers the fuel-gauge coulomb counter and instantaneous current, valid only with charging off [10]. Our method, USB rail plus pack coulomb delta with charging off, is the non-Pixel method the platform itself documents.

**Sanctioned, no-root signals and levers on Android 15 and 16.** `getThermalHeadroom(forecastSeconds)` returns 0 to 1 with OEM thresholds, at most once per 10 seconds; guidance is to act above 0.85 [6]. `getGpuHeadroom()` and `getCpuHeadroom()` estimate remaining capacity, at a cost of at least one binder call over 1 ms [7]. A hint session can report per-token work duration and set `setPreferPowerEfficiency(true)` [5]. `cpu.uclamp.max` caps the CPU clock for a thread group, but only if decode threads do not share a CPU with uncapped work [3]. These are the deployment path for the μKV scheduler without root; the sysfs GPU cap stays the research path.

**Store-level pressure.** Since March 2026 Google Play flags apps whose sessions hold more than two hours of non-exempt wake locks in a day [12]. A long screen-off generation counts. An output cap at low battery also protects Play standing.

## 3. What shipped for AI energy in 2025 and 2026

| Item | Claim or measurement | Source |
|---|---|---|
| Snapdragon 8 Elite Gen 5 | CPU +20% at 35% better efficiency, GPU +23% at 20% better, NPU +37%, "up to 16% SoC power saving" | [19] |
| Same, measured | 19 W at full CPU load; about 12 W on a GPU benchmark | [19] |
| OnePlus 15 under sustained GPU load | surface 47 to 52.7 C, OS lockout of about 15 min, about 60% of peak by test end after the December 2025 update | [24] |
| Apple A19 Pro, iPhone 17 Pro | vapor chamber, "up to 40% better sustained performance"; iPhone 16 Pro throttles after about one minute of image generation | [25] |
| Gemini Nano 4 on Pixel 10 Pro XL | claimed 4x faster and 60% less battery; a tester saw 2x because the model writes 50 to 100% more text | [42] |
| Gemma 3 270M on Pixel 9 Pro | 0.75% battery for 25 conversations | [43] |
| Frameworks (llama.cpp, MLC-LLM, ExecuTorch, LiteRT-LM, QNN) | no battery-aware mode; NPU profiles and kernel sysfs are the only knobs | [20, 21, 22, 23] |
| Benchmarks (MLPerf Mobile v6.0, Geekbench AI) | no phone power reported | [44] |
| Shipped hybrid routing (Google Private AI Compute) | routes by capability and privacy, never by battery | [45] |

Two of these matter directly. The OnePlus 15 lockout at 47 C is the failure our 902 MHz cap avoids by keeping DDR at 65 C. The Gemini Nano 4 result shows output length erasing a hardware efficiency gain, which is the data lever in our scheduler measured by someone else.

## 4. Measured LLM energy on phones, against ours

Decode energy per token in the literature, for 1B to 3B 4-bit models, brackets what we measured on the OnePlus 15.

| Study | Device, model | Decode energy per token | Note |
|---|---|---|---|
| MELTing point, MobiCom 2024 | iPhone 14 Pro, Zephyr-3B | 0.72 J | 13.8 W sustained, 47.9 C, throttled after 20 to 32 prompts [30] |
| PowerInfer-2, 2024 | OnePlus 12, Mixtral-47B sparse | 0.257 J | 5.1 W peak [34] |
| LLMs in Your Pockets, TMC 2026 | Snapdragon phones | GPU 0.058 mAh (about 0.8 J) | GPU cuts prefill energy 88% and decode 60% vs CPU; CPU clock falls 50% in 9 rounds [35] |
| FUSE, 2025 | Pixel 7, TinyLlama-1.1B GPU | 0.397 J | default governors up to 40% slower at equal energy [32] |
| Sustained load, 2026 | Galaxy S24 Ultra, Qwen2.5-1.5B | 0.146 J | Adreno 750 boosts to 1000 MHz then settles at 720 to 770 MHz; GPU 68.5 C [38] |
| PowerBench, July 2026 | OnePlus 15 and Xiaomi 17, Qwen2.5-1.5B | CPU 0.10 to 0.12 J, NPU 0.32 to 0.43 J | 54.8% decode energy cut for 13.4% latency by polling, sleep and a lower CPU clock; cooled below 28 C first [39] |
| μKV, this project | OnePlus 15, Llama-3.2-1B, K=1024 | 0.19 to 0.23 J on GPU and CPU | prefill about 60 mJ per prompt token on both; cooled to DDR 35 C, charging off |

Four points of agreement with our measurements:

- **The Adreno plateau.** The S24 Ultra settles at 720 to 770 MHz under heat [38]. Our 726 MHz level is that plateau, and our 902 cap holds the phone above the point where the vendor would force it there.
- **The CPU clock is not an energy lever.** PowerBench finds mid CPU clock levels beat high levels for decode energy [39]. We measured energy flat from 883 to 1632 MHz with power close to linear in frequency, which is the voltage-pinned regime the older measurement papers describe [50, 51, 52]. The control survey notes that no 2024 to 2026 paper reports a measured power exponent on a current Snapdragon at the voltage floor [53]; ours is such a measurement.
- **Prefill belongs to the accelerator.** llm.npu reaches over 1000 tokens per second of prefill on the NPU with 30x less energy than CPU [46], and PowerBench confirms NPUs win prefill while CPUs win decode [39]. Prefill is 39 percent of our request energy. Moving it to the Hexagon NPU is the largest lever we have not pulled, and it composes with the KV and clock policy rather than replacing it.
- **A fixed profiled setting beats the governors.** FUSE pins a profiled CPU, GPU and memory frequency triple per phase with no online learning and cuts time per token 25 to 37 percent at equal energy [32]. That is the same conclusion as our table against the bandit.

One point of difference to keep in view: FUSE, PolyThrottle and EnerInfer all find memory frequency to be a first-order decode knob [32, 31, 47]. Our GPU power level carries a DDR bus vote with it [17], so our cap moves DDR implicitly, but we have not measured DDR frequency as its own axis.

## 5. Learned policies against rules, and the two loops

**What learned governors achieve on phones.** zTT holds a target frame rate in a hot room with 23.9 percent less power [54]. GearDVFS gains 6 to 9 percent performance per watt on a Redmi Note 9 after 35 minutes of training plus 500 to 650 seconds of on-device adaptation [55]. MetaDVFS reaches up to 17 percent on five Pixels with 3.5 minutes of adaptation per new device and app pair [56]. MobiRL, deployed in commercial phones, reports 42.8 percent less power than the vendor scheduler [57]. None of these is compared against a tuned static policy; MetaDVFS says so [56]. Where a static profiled policy was measured, FUSE, it beat the governors by as much as the learned ones do, at zero online cost [32]. No paper reports the energy spent on learning [53]. Ours does: 24 real requests, 162 minutes, 15.7 kJ, to reach a policy the table already held.

**Learning cost in the literature.** PolyThrottle's Bayesian optimizer needs about 15 samples under a latency constraint [47]. PaRMIS needs 300 to 500 policy evaluations to learn a Pareto front once [58]. JouleGuard's bandit costs 249 microseconds per iteration on a mobile board [60]. Our bandit's decision costs 11 microseconds and the whole on-phone scheduler step 0.3 seconds.

**How two loops are kept from fighting.** The control literature has six answers, and our design uses three of them [53]:

| Pattern | Where it is published | Our use |
|---|---|---|
| One loop is the constraint, the other translates it into the cheapest configuration | POET, RTAS 2015; CALOREE, ASPLOS 2018 [59, 61] | the performance loop holds the time budget; the energy loop minimizes energy inside it |
| Disjoint knobs by priority | Filieri, Hoffmann, Maggio, FSE 2015; E4, AAAI 2025 [62, 63] | measured assignment: decode clock and output cap to E, CPU clock to P, prefill clock shared under the lever |
| A scalar passed from the slow loop to the fast loops | Odyssey's battery-lifetime goal, SOSP 1999; JouleGuard's energy goal factor; Autothrottle's throttle ratio, NSDI 2024 [64, 60, 65] | the lever, set by battery tier |
| Couple through model error | JouleGuard's adaptive pole; WASL, ICPE 2026 [60, 66] | the loop nudges key on the relative budget error; not yet exercised on the phone |
| Supervisory switch of objective on a thermal event | SOSA, MICRO 2019; SPECTR, ASPLOS 2018 [67, 68] | the hot-phone rule excludes uncapped 1200 MHz plans |
| Hierarchy in time scale | GRACE-1, TMC 2006 [69] | tier per request, lever bias across requests |

**The split-clock plan has a theorem behind it.** Kim, Imes and Hoffmann prove that the energy-optimal schedule under a deadline uses at most two configurations, and measure on a big.LITTLE board that race-to-idle uses 3.34 times the optimal energy while pace-to-idle uses 1.12 times [49]. Our best plan, prefill at 1200 MHz and decode at 902, is a two-configuration schedule: race the compute-bound phase, pace the bandwidth-bound one.

**Battery as an input to LLM knobs.** The precedents for a battery-driven scalar lever are Odyssey (a user-set battery lifetime lowers application fidelity, 1999), GRACE-1 (2006) and Budget RNNs (discrete energy budget levels, RTAS 2021) [64, 69, 70]. In the LLM era, EnerInfer switches modes on shell temperature [31], DVFSLM minimizes energy under a token deadline with analytic estimators [71], PELM varies speculative verification depth with DVFS [72], and adaptive KV quantization picks per-token precision by importance [73]. None takes battery level as an input [53].

## 6. The gap, stated by the surveys

- No phone maker, SoC vendor or OS ships a feature that changes LLM model size, context, output length or accelerator clock by battery level [20, 24].
- No LLM framework has a battery- or thermal-aware mode; the only knobs are NPU profiles and kernel sysfs nodes [21, 22, 23].
- No public benchmark reports phone power for LLM inference; the only per-rail OnePlus 15 dataset is PowerBench, July 2026 [39, 44].
- No 2024 to 2026 paper at MobiSys, MobiCom, HotMobile, SenSys, EuroSys or ASPLOS composes GPU clock, KV size and output length as a per-request function of battery state [24, 53].
- No contextual-bandit DVFS or thermal policy on a phone was found; the nearest run on an ODROID and in a datacenter [53].
- No paper reports the energy spent learning a policy on a phone [53].

μKV's scheduler answers each: a measured cost table, a ladder walk with a stopping rule, a lever by tier, measured exchange rates per knob, a bandit on real requests as a check, and the learning cost measured in joules.

## 7. What to take into the design and the paper

1. **Frame the split-clock plan with the two-configuration result** [49]. It turns a measured curiosity into a predicted optimum.
2. **Justify the table over the RL with FUSE and MetaDVFS** [32, 56]: static profiled settings match learned governors on known workloads, and LLM decode on a known engine is a known workload.
3. **Key the lever nudge on prediction error**, as JouleGuard and WASL do [60, 66]. Our table predicts within 3.7 percent on average; the nudge should fire on a miss larger than that, not on any miss.
4. **Offer a no-root deployment path**: thermal headroom above 0.85 and a hint session with power-efficiency preference as the trigger and the CPU lever [5, 6]; the sysfs GPU cap remains the research prototype's lever, and the driver's max() composition means it never fights the vendor thermal cap [16].
5. **Move the low tier to about 40 percent** with hysteresis, and exempt short interactive requests, following the user studies and Adaptive Power [26, 13].
6. **Name the NPU prefill composition as future work** [46, 39]: prefill is 39 percent of our energy and the NPU cuts it by an order of magnitude.
7. **Report DDR frequency as an unmeasured axis** [32, 47], since the GPU cap moves it implicitly [17].
8. **Cite the output-length evidence** for the data lever: 25 to 60 percent on servers [40], 15 to 20 percent from stopping failing agent runs [41], and Gemini Nano 4's verbosity erasing a 4x hardware gain [42].
9. **State the cooling protocol against PowerBench's** (below 28 C before runs) [39]; ours is DDR at or below 35 C with charging off.

## 8. References

### Kernel and Android

1. Energy Aware Scheduling, Linux kernel documentation. https://docs.kernel.org/scheduler/sched-energy.html
2. Energy Model of devices, Linux kernel documentation. https://docs.kernel.org/power/energy-model.html
3. Utilization clamping, Linux kernel documentation. https://docs.kernel.org/scheduler/sched-util-clamp.html
4. Schedutil, Linux kernel documentation. https://docs.kernel.org/scheduler/schedutil.html
5. Android Dynamic Performance Framework, and Android 15 additions. https://developer.android.com/games/optimize/adpf and https://developer.android.com/about/versions/15/features
6. ADPF Thermal API. https://developer.android.com/games/optimize/adpf/thermal
7. Android 16 CPU and GPU headroom. https://developer.android.com/about/versions/16/features
8. Battery Saver, Extreme Battery Saver, Routine Battery Saver. https://support.google.com/pixelphone/answer/6187458 and https://source.android.com/docs/core/power/routine-battery-saver
9. Game Mode API and interventions. https://developer.android.com/games/optimize/adpf/gamemode/gamemode-api
10. Power Stats HAL, Perfetto battery counters, Wattson, Battery Historian status. https://source.android.com/docs/core/power/power-stats-hal , https://perfetto.dev/docs/data-sources/battery-counters , https://source.android.com/docs/core/power/wattson/how-to-wattson
11. Android 17 release, June 2026. https://android-developers.googleblog.com/2026/06/Android-17.html
12. Google Play battery technical quality enforcement, March 2026. https://android-developers.googleblog.com/2026/03/battery-technical-quality-enforcement.html

### Apple

13. Adaptive Power, iOS 26. https://support.apple.com/en-us/123707
14. Low Power Mode. https://support.apple.com/en-us/101604
15. Foundation Models framework, UnavailableReason; Apple Foundation Models 2025 tech report. https://developer.apple.com/documentation/foundationmodels/systemlanguagemodel/availability-swift.enum/unavailablereason and https://arxiv.org/abs/2507.13575

### Vendors

16. Qualcomm kgsl GPU power control source (max_gpuclk, thermal_pwrlevel, composition by max). https://raw.githubusercontent.com/qualcomm-linux/kgsl/gfx-kernel.le.0.0/kgsl_pwrctrl.c
17. Adreno power levels device tree binding (GPU frequency with bus frequency per level). https://android.googlesource.com/kernel/msm/+/android-wear-5.0.2_r0.1/Documentation/devicetree/bindings/gpu/adreno-pwrlevels.txt
18. QNN HTP performance modes. https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
19. Snapdragon 8 Elite Gen 5 announcement, September 2025, and measured power. https://www.qualcomm.com/news/releases/2025/09/snapdragon-8-elite-gen-5--the-world-s-fastest-mobile-system-on-a and https://www.notebookcheck.net/Qualcomm-Snapdragon-8-Elite-Gen-5-vs-Dimensity-9500-and-Apple-A19-Pro-in-efficiency-analysis.1123883.0.html
20. Android AICore and Gemini Nano documentation (no battery condition). https://developer.android.com/ai/gemini-nano
21. llama.cpp OpenCL backend for Adreno (no power option). https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md
22. MLC-LLM chat config (compile-time context and window, no power option). https://llm.mlc.ai/docs/deploy/mlc_chat_config.html
23. ExecuTorch 1.0, October 2025. https://pytorch.org/blog/introducing-executorch-1-0/
24. OnePlus 15 thermal behaviour and update. https://www.androidpolice.com/we-tried-to-make-the-oneplus-15-overheat-heres-what-happened/ and https://www.androidauthority.com/oneplus-15-update-benchmarks-3621249/
25. Apple iPhone 17 Pro announcement and Draw Things sustained test. https://www.apple.com/newsroom/2025/09/apple-unveils-iphone-17-pro-and-iphone-17-pro-max/ and https://releases.drawthings.ai/p/iphone-17-pro-doubles-ai-performance

### Users

26. Talker Research, phone battery worry starts at 38 percent, 2025. https://talkerresearch.com/poll-phone-at-38-thats-when-most-americans-start-to-panic/
27. Tang and Gao, Running Low, Diversifying Less, Journal of Consumer Behaviour, 2026. https://onlinelibrary.wiley.com/doi/10.1002/cb.70189
28. Lee et al., SERENUS, UIST 2024. https://dl.acm.org/doi/10.1145/3654777.3676437
29. Liu et al., Andes (reading 4.8 and listening 3.3 tokens per second), 2024. https://arxiv.org/html/2404.16283

### LLM energy on phones

30. Laskaridis et al., MELTing point, MobiCom 2024. https://arxiv.org/abs/2403.12844
31. Zou et al., EnerInfer, June 2026. https://arxiv.org/abs/2606.23001
32. Zhang et al., FUSE, Dissecting Mobile DVFS Governors for LLM Inference, 2025; CORE, MLSys 2026. https://arxiv.org/abs/2507.02135
33. Huang et al., MNN-AECS, June 2025. https://arxiv.org/abs/2506.19884
34. Xue et al., PowerInfer-2, 2024. https://arxiv.org/abs/2406.06282
35. Xiao et al., Understanding LLMs in Your Pockets, IEEE TMC 2026. https://arxiv.org/abs/2410.03613
36. Zhang, Xia, Wang, KVSwap, MobiSys 2026. https://arxiv.org/abs/2511.11907
37. Wang et al., DynaKV (OnePlus phones, 1.57x lower energy), 2025. https://arxiv.org/abs/2511.07427
38. Tummalapalli et al., LLM Inference at the Edge under Sustained Load, March 2026. https://arxiv.org/abs/2603.23640
39. Cai et al., PowerBench, Is Your NPU Ready for LLMs, July 2026. https://arxiv.org/abs/2607.05475
40. Poddar et al., Brevity is the soul of sustainability, ACL 2025 Findings. https://arxiv.org/abs/2506.08686
41. Pham et al., AgentStop, ACM CAIS 2026. https://arxiv.org/abs/2605.15206
42. Gemini Nano 4 tested on Pixel 10 Pro XL, April 2026. https://www.androidauthority.com/gemini-nano-4-benchmarks-3655763/
43. Gemma 3 270M, August 2025. https://developers.googleblog.com/en/introducing-gemma-3-270m/
44. MLPerf Mobile v6.0, June 2026. https://mlcommons.org/2026/06/mlperf-mobile-v6/
45. Google Private AI Compute, November 2025. https://9to5google.com/2025/11/11/google-private-ai-compute-pixel/
46. Xu et al., llm.npu, ASPLOS 2025. https://arxiv.org/abs/2407.05858
47. Yan, Wang, Venkataraman, PolyThrottle, 2023. https://arxiv.org/abs/2310.19991
48. Yin et al., Elastic On-Device LLM Service, MobiCom 2025. https://arxiv.org/abs/2409.09071

### Control and learning

49. Kim, Imes, Hoffmann, Racing and Pacing to Idle, CPSNA 2015. https://people.cs.uchicago.edu/~hankhoffmann/kim-cpsna2015.pdf
50. Le Sueur and Heiser, DVFS: The Laws of Diminishing Returns, HotPower 2010. https://www.usenix.org/legacy/events/hotpower/tech/full_papers/LeSueur.pdf
51. Miyoshi et al., Critical Power Slope, ICS 2002. https://dl.acm.org/doi/10.1145/514191.514200
52. De Vogeleer et al., The Energy/Frequency Convexity Rule, 2014. https://arxiv.org/abs/1401.4655
53. Sub-survey `control_rl.md` in this directory, sections 5 and 6 (learning cost table, gaps).
54. Kim et al., zTT, MobiSys 2021. https://dl.acm.org/doi/10.1145/3458864.3468161
55. Lin et al., GearDVFS, MobiCom 2023. https://dl.acm.org/doi/10.1145/3570361.3592524
56. Yan et al., MetaDVFS, September 2025. https://arxiv.org/abs/2509.22707
57. Dou, Liu, Xiao, MobiRL, ACM TACO 2024. https://dl.acm.org/doi/10.1145/3674910
58. Deshwal et al., PaRMIS, DAC 2021. https://arxiv.org/abs/2105.09282
59. Imes et al., POET, RTAS 2015. https://people.cs.uchicago.edu/~ckimes/poet/
60. Hoffmann, JouleGuard, SOSP 2015. https://doi.org/10.1145/2815400.2815403
61. Mishra et al., CALOREE, ASPLOS 2018. https://dl.acm.org/doi/10.1145/3173162.3173184
62. Filieri, Hoffmann, Maggio, Automated Multi-objective Control, FSE 2015. https://dl.acm.org/doi/10.1145/2786805.2786833
63. Zhang et al., E4, AAAI 2025. https://arxiv.org/abs/2503.04865
64. Flinn and Satyanarayanan, Odyssey, SOSP 1999. https://dl.acm.org/doi/10.1145/319151.319155
65. Wang et al., Autothrottle, NSDI 2024. https://arxiv.org/abs/2212.12180
66. Pervaiz et al., WASL, ICPE 2026. https://doi.org/10.1145/3777884.3797009
67. Donyanavard et al., SOSA, MICRO 2019. https://dl.acm.org/doi/10.1145/3352460.3358312
68. Rahmani et al., SPECTR, ASPLOS 2018. https://dl.acm.org/doi/10.1145/3296957.3173199
69. Yuan et al., GRACE-1, IEEE TMC 2006. https://doi.org/10.1109/TMC.2006.98
70. Kannan and Hoffmann, Budget RNNs, RTAS 2021. https://ieeexplore.ieee.org/document/9470487/
71. Chen et al., DVFSLM, MobiSys 2026 Workshops. https://doi.org/10.1145/3812836.3814753
72. Yang and Xia, PELM, 2026. https://dl.acm.org/doi/10.1145/3774906.3802783
73. Boroujeni et al., Adaptive KV-Cache Quantization, 2026. https://arxiv.org/abs/2604.04722
