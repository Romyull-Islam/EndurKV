# Energy, power and thermal behaviour of on-device LLM inference on smartphones: literature survey (2024 to 2026)

Survey date: 2026-09-04.
Method: 40 web searches and about 60 page fetches (arXiv abstract and HTML pages, dblp, ACM and SIGOPS program pages, vendor and framework pages). Every item lists the URL that was actually opened. Venue and year were read from the fetched page unless marked "unverified".
Target use: a per-request scheduler on a Snapdragon phone that picks GPU clock, KV-cache size and an output-length cap from battery state (level, temperature, charging).

Reading guide for each item: title; authors or organization; venue and year; URL; idea in two sentences; key measured numbers; one sentence on relevance to the scheduler.

---

## 0. Summary of what the literature says

1. Decode is memory-bound on phones. Energy per token is set mostly by DRAM traffic and by how many cores or accelerators are kept awake, not by peak compute. (MELT, Pockets, Is Your NPU Ready, FUSE, MNN-AECS.)
2. Measured decode energy on flagship phones ranges from about 0.1 J/token to 0.5 J/token for 1B to 3B 4-bit models, and about 0.26 J/token for a 47B sparse model in PowerInfer-2. Phone-level power during inference is 5 W to 14 W peak and 3 W to 8.5 W sustained.
3. Thermal throttling starts within 1 to 3 minutes of sustained decode. Reported shell or SoC temperatures reach 42 C to 48 C on phones, and CPU clocks fall by about 50 percent within ten rounds. GPU clocks on Adreno drop from 1000 MHz to 720 to 770 MHz and then stay flat.
4. NPUs win prefill by 20x to 40x in speed and energy, but for decode the CPU is often as efficient or better, and host CPU polling can waste 30 to 40 percent of system energy during NPU runs.
5. Lowering NPU, DDR, CPU or GPU frequency during decode saves 25 to 65 percent energy at little or no throughput cost, because decode is bandwidth-bound (EnerInfer, FUSE, Is Your NPU Ready).
6. Output length is the strongest single lever on energy per request. Response-length control saves 25 to 60 percent on servers, and early termination of failing agent runs saves 15 to 20 percent on consumer devices.
7. Gap: no paper found in 2024 to 2026 changes LLM inference settings by battery level or charging state. EnerInfer states explicitly that it does not consider battery level or charging. Claims that AICore routes by battery state come only from third-party blogs.

---

## 1. Measurement studies of LLM inference energy on phones

### 1.1 MELTing point: Mobile Evaluation of Language Transformers
- Authors: Stefanos Laskaridis, Kleomenis Katevas, Lorenzo Minto, Hamed Haddadi (Brave Software, Imperial College London).
- Venue and year: ACM MobiCom 2024 (arXiv comments field: "Accepted at MobiCom 2024"; ACM DOI 10.1145/3636534.3690668).
- URL: https://arxiv.org/abs/2403.12844 (numbers from https://arxiv.org/html/2403.12844v4)
- Idea: MELT is a headless benchmarking harness that runs LLMs on Android, iOS and Jetson with llama.cpp and MLC-LLM and records accuracy, latency, energy and memory. The paper shows that inference is memory-bound and that sustained use is limited by energy and heat.
- Numbers: Galaxy S23 (Snapdragon 8 Gen 2), Pixel 6a, iPhone 14 Pro, iPhone SE. iPhone 14 Pro with Zephyr-3B 4-bit: 0.20 mWh per token (0.72 J/token) on MLC and 0.16 mWh per token on LLMFarm. iPhone 14 Pro drew 13.8 W sustained and over 18 W peak; Galaxy S23 drew 14 W peak and under 8.5 W sustained. About 490 to 590 prompts per full battery. iPhone 14 Pro reached 47.9 C after a full conversation, with throttling after roughly 20 to 32 prompts. Dequantize plus matmul took 97 percent of prefill and 95.7 percent of decode time for Llama-7B 3-bit.
- Scheduler relevance: gives the baseline prompts-per-charge and thermal onset curve that a battery-aware scheduler must improve on, and shows that memory traffic, not compute, sets energy per token.

### 1.2 Understanding Large Language Models in Your Pockets: Performance Study on COTS Mobile Devices
- Authors: Jie Xiao, Qianyi Huang, Xu Chen, Chen Tian.
- Venue and year: IEEE Transactions on Mobile Computing 25(8):13077-13094, 2026 (dblp; DOI 10.1109/TMC.2026.3674827). arXiv v1 Oct 2024, v5 Feb 2026.
- URL: https://arxiv.org/abs/2410.03613 (numbers from https://arxiv.org/html/2410.03613v5)
- Idea: A measurement study of llama.cpp (CPU) and MLC-LLM (GPU) on seven devices from Qualcomm, MediaTek, HiSilicon and Apple, using Perfetto and Snapdragon Profiler. It covers throughput, battery, DVFS, launch time and OS behaviour.
- Numbers: CPU prefill 0.031 to 0.144 mAh per token, CPU decode 0.042 to 0.143 mAh per token, GPU prefill 0.013 mAh per token, GPU decode 0.058 mAh per token (Table V). GPU cut prefill energy by 87.79 percent and decode energy by 59.75 percent versus CPU. About 20 mAh per round of 64 input and 128 output tokens. Snapdragon CPU clocks fell about 50 percent within 9 rounds; temperature rose from 34 C to about 42 C and stabilised. GPU frequency stayed more stable than CPU frequency. Adreno 750 prefill about 46 tok/s versus Mali-G720 about 5 tok/s.
- Scheduler relevance: shows that on Snapdragon the GPU is both the more efficient decode engine and the more stable one under heat, which supports a GPU-clock knob rather than a CPU-core knob.

### 1.3 PalmBench: A Comprehensive Benchmark of Compressed Large Language Models on Mobile Platforms
- Authors: Yilong Li, Jingyu Liu, Hao Zhang, M Badri Narayanan, Utkarsh Sharma, Shuai Zhang, Pan Hu, Yijing Zeng, Jayaram Raghuram, Suman Banerjee.
- Venue and year: arXiv preprint, v1 Oct 2024, v2 Jan 2025 (no venue listed).
- URL: https://arxiv.org/abs/2410.05315 (numbers from https://arxiv.org/html/2410.05315v2)
- Idea: An automated benchmark of 2-bit to 6-bit quantized LLMs on eight devices with MLC-LLM and llama.cpp. It reports throughput, energy, memory, temperature and also hallucination and toxicity of compressed models.
- Numbers: iPhone 12 Pro, iPhone 15 Pro, Pixel 4, Pixel 5a, Pixel 7, Galaxy S22 Ultra, Orange Pi 5, Jetson Orin Nano. Llama-3.2-3B 3-bit used 11.21 mWh per run on iPhone 12 and 25.4 mWh on Orange Pi (Table 3). 4-bit models used 25.2 percent more power than 3-bit. Peak temperatures from 45.3 C (iPhone 15 Pro, 3-bit) to 75.4 C (Orange Pi 5).
- Scheduler relevance: quantization level is a coarse energy knob with a quality cost, which a scheduler could pair with output-length caps at low battery.

### 1.4 LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load
- Authors: Pranay Tummalapalli, Sahil Arayakandy, Ritam Pal, Kautuk Kundan.
- Venue and year: arXiv preprint, v1 24 Mar 2026, v2 7 Jun 2026.
- URL: https://arxiv.org/abs/2603.23640 (numbers from https://arxiv.org/html/2603.23640v2)
- Idea: Runs Qwen2.5-1.5B 4-bit for 20 back-to-back iterations on iPhone 16 Pro (MLX), Galaxy S24 Ultra (MLC-LLM, Adreno 750), Raspberry Pi 5 with Hailo-10H, and an RTX 4050 laptop. It argues that thermal management, not peak compute, bounds sustained phone inference.
- Numbers: iPhone 16 Pro 40.49 tok/s peak, 24.96 warm, 23.67 hot (about 40 percent loss within three iterations); 5 percent battery for 20 iterations of 819 output tokens, about 400 inferences per charge. S24 Ultra 12.21 tok/s peak during a 1000 MHz GPU boost, then 10.38 tok/s on a 720 to 770 MHz plateau; 7 percent battery for 20 iterations of 646 tokens; GPU 68.5 C peak and 63 to 65 C steady; reported 146 plus or minus 11 mJ per token. Hailo-10H 6.9 tok/s under 2 W. RTX 4050 131.7 tok/s at 34.1 W.
- Scheduler relevance: gives a direct Adreno GPU-clock versus throughput trace under heat, which is the plant model a GPU-clock scheduler needs.

### 1.5 Is Your NPU Ready for LLMs? Dissecting the Hidden Efficiency Bottlenecks in Mobile LLM Inference
- Authors: Guanyu Cai, Ruiming Tian, Lang Yang, Zhouhong Ren, Jinliang Yuan, Lingkun Li, Jiliang Wang.
- Venue and year: arXiv preprint, 6 Jul 2026.
- URL: https://arxiv.org/abs/2607.05475 (numbers from https://arxiv.org/html/2607.05475v1)
- Idea: Measures five frameworks (llama.cpp, MNN, MLC-LLM, MLLM, GENIE) on CPU, GPU and NPU of Xiaomi 14, Xiaomi 15, Xiaomi 17 and OnePlus 15 (SM8650, SM8750, SM8850). It finds up to 10x framework gaps on the NPU and up to 40 percent energy waste from scheduling.
- Numbers: NPU decode of Qwen2.5-1.5B on Xiaomi 17 with GENIE used 0.43 J/token versus 0.12 J/token on CPU (Table 5). W4A16 cut NPU energy by 53.5 to 66.9 percent. Host CPU polling took up to 30 percent of system energy during NPU runs. Mid CPU frequency levels beat high levels for GPU and NPU runs in both throughput and energy. A coordinated setting (RPC polling 20 us, NPU sleep latency 65535 us, fixed low CPU clock) cut decode energy 54.8 percent for 13.4 percent more latency. Devices were cooled below 28 C before each run.
- Scheduler relevance: the strongest recent evidence that per-request frequency and polling settings on Snapdragon change decode energy by half, and that the CPU should be clocked down when GPU or NPU decode.

### 1.6 Dissecting the Impact of Mobile DVFS Governors on LLM Inference Performance and Energy Efficiency (FUSE)
- Authors: Zongpu Zhang, Pranab Dash, Y. Charlie Hu, Qiang Xu, Jian Li, Haibing Guan.
- Venue and year: arXiv preprint, 2 Jul 2025.
- URL: https://arxiv.org/abs/2507.02135 (numbers from https://arxiv.org/html/2507.02135v1)
- Idea: Measures how the uncoordinated CPU, GPU and memory governors on Android hurt LLM inference, using Pixel 7 and 7 Pro (Tensor G2, Mali-G710) with battery bypassed and a Monsoon monitor at 0.2 ms. It proposes FUSE, a unified governor for the three domains.
- Numbers: 18 CPU levels (500 to 2850 MHz), 12 GPU levels (151 to 848 MHz), 13 memory levels (421 to 3172 MHz). Default governors gave up to 40.4 percent longer decode latency than the best frequency mix at equal energy. Example: GPU governor decode 215.1 ms per token at 402.7 mJ/token versus optimal 126.9 ms at 396.5 mJ/token (TinyLlama); EAS decode 955.1 mJ/token (StableLM 2.7B). FUSE cut TTFT 7.0 to 16.9 percent and per-token time 25.4 to 36.8 percent at equal energy per token; alternatively energy per token can fall up to 14.9 percent (prefill) and 5.0 percent (decode).
- Scheduler relevance: the closest prior work to a GPU-clock knob for LLM decode on a phone, but it targets Mali and optimises latency at fixed energy rather than energy at a battery-set target.

### 1.7 Edge-board measurement studies (not phones, listed for energy-per-token reference)
- Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency. Erik Johannes Husom et al. (SINTEF). arXiv 2504.03360, Apr 2025. Listed by ACM DL search as ACM Transactions on Internet of Things, DOI 10.1145/3767742 (ACM page returned 403, venue unverified). URL: https://arxiv.org/abs/2504.03360. Raspberry Pi 4, 28 quantized models via Ollama; a search snippet reports qwen2.5-0.5b at 2.61 J/token, llama3.2-1b 8.40 J/token, gemma2-2b 9.35 J/token (numbers not confirmed from the page).
- Characterizing and Understanding Energy Footprint and Efficiency of Small Language Model on Edges. Md Romyull Islam, Bobin Deng, Nobel Dhar, Tu N. Nguyen, Selena He, Yong Shi, Kun Suo. IEEE MASS 2025 (arXiv comments). URL: https://arxiv.org/abs/2511.11624. Raspberry Pi 5, Jetson Nano, Jetson Orin Nano; Llama 3.2, Phi-3 Mini, TinyLlama, Gemma 2; Orin Nano GPU had the best energy-to-performance ratio.
- Scheduler relevance: these give order-of-magnitude J/token on boards but do not model phone thermal or battery limits.

---

## 2. Systems that make mobile LLM inference cheaper or battery-aware

### 2.1 PowerInfer-2: Fast Large Language Model Inference on a Smartphone
- Authors: Zhenliang Xue, Yixin Song, Zeyu Mi, Xinrui Zheng, Yubin Xia, Haibo Chen (SJTU IPADS).
- Venue and year: arXiv preprint, v1 Jun 2024, v3 Dec 2024 (no venue in comments).
- URL: https://arxiv.org/abs/2406.06282 (numbers from https://arxiv.org/html/2406.06282v3)
- Idea: Decomposes matmuls into neuron clusters, runs dense clusters on the NPU and sparse ones on the CPU, and pipelines UFS reads with compute through a segmented neuron cache. It serves models larger than DRAM.
- Numbers: OnePlus 12 (Snapdragon 8 Gen 3, 24 GB, UFS 4.0) and OnePlus Ace 2 (8+ Gen 1). TurboSparse-Mixtral-47B at 11.68 tok/s; up to 27.8x over llama.cpp. Table 8: 0.257 J/token, 31.1 percent less energy than QNN and 61.8 percent less than llama.cpp, peak power 5.095 W.
- Scheduler relevance: a phone-level J/token and peak-watt reference for a Snapdragon 8 Gen 3 that a scheduler can use as an upper envelope for large models.

### 2.2 Fast On-device LLM Inference with NPUs (llm.npu, earlier mllm-NPU)
- Authors: Daliang Xu, Hao Zhang, Liming Yang, Ruiqi Liu, Gang Huang, Mengwei Xu, Xuanzhe Liu (PKU, BUPT).
- Venue and year: ASPLOS 2025, pages 445-462, DOI 10.1145/3669940.3707239 (confirmed on dblp; ACM page returned 403).
- URL: https://arxiv.org/abs/2407.05858 (numbers from https://arxiv.org/html/2407.05858v2)
- Idea: Offloads prefill to the Hexagon NPU with fixed-size prompt chunks, outlier extraction to CPU or GPU, and out-of-order block scheduling. It is the first system to exceed 1000 tok/s prefill for a billion-parameter model on a phone.
- Numbers: Redmi K70 Pro (8 Gen 3) and K60 Pro (8 Gen 2); energy from /sys/class/power_supply every 100 ms. At 1024-token prompts: 18.2 to 38.4x faster than llama.cpp CPU, 32.5 to 43.6x faster than MLC GPU, 3.3 to 5.3x faster than PowerInfer-v2 NPU; energy 35.6 to 59.5x lower than llama.cpp CPU and 1.85 to 4.32x lower than TFLite GPU. Abstract: 22.4x prefill speedup and 30.7x energy savings on average.
- Scheduler relevance: prefill should go to the NPU regardless of battery state, so the scheduler's knobs matter mainly in decode.

### 2.3 Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference (HeteroInfer, earlier HeteroLLM)
- Authors: Le Chen, Dahu Feng, Erhu Feng, Yingrui Wang, Rong Zhao, Yubin Xia, Pinjie Xu, Haibo Chen (SJTU, Tsinghua, SenseTime).
- Venue and year: SOSP 2025 (confirmed on the SIGOPS accepted-papers page; ACM DOI 10.1145/3731569.3764808).
- URL: https://arxiv.org/abs/2501.14794 and https://sigops.org/s/conferences/sosp/2025/accepted.html
- Idea: Characterises GPU, NPU and memory bandwidth on Snapdragon 8 Gen 3 and runs GPU and NPU together with tensor partition strategies that differ for prefill and decode. A fast sync path uses unified memory.
- Numbers: 1.34x to 6.02x end-to-end speedup over GPU-only and NPU-only engines; up to 9.99x prefill over MLC. No energy numbers on the pages fetched.
- Scheduler relevance: documents the GPU and NPU bandwidth ceilings on the same SoC family the scheduler targets.

### 2.4 EdgeMoE: Empowering Sparse Large Language Models on Mobile Devices
- Authors: Rongjie Yi, Liwei Guo, Shiyun Wei, Ao Zhou, Shangguang Wang, Mengwei Xu.
- Venue and year: IEEE Transactions on Mobile Computing 24(8):7059-7073, 2025, DOI 10.1109/TMC.2025.3546466 (dblp). arXiv v1 Aug 2023, v2 Mar 2025.
- URL: https://arxiv.org/abs/2308.14352
- Idea: Keeps non-expert weights in DRAM and experts in flash, with expert-wise bit-width adaptation and expert preloading. An expert-buffer eviction policy uses activation frequency and position.
- Numbers: abstract reports "significant memory savings and speedup" on Mixtral-class MoE models; no energy figures on the abstract page.
- Scheduler relevance: shows that flash-to-DRAM traffic is an energy and latency term a scheduler must budget when models exceed DRAM.

### 2.5 ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference
- Authors: Wangsong Yin, Daliang Xu, Mengwei Xu, Gang Huang, Xuanzhe Liu.
- Venue and year: ACM MobiSys 2026 (arXiv comments; ACM DOI 10.1145/3745756.3809205).
- URL: https://arxiv.org/abs/2508.16703 (numbers from https://arxiv.org/html/2508.16703v4)
- Idea: Keeps attention on the NPU with shadowAttn, a sparse attention that uses a cheap NPU pilot pass to pick important tokens. It adds graph bucketing, head-wise NPU-CPU pipelining and per-head sparsity ratios.
- Numbers: Xiaomi 14 (8 Gen 3, Hexagon V75) and Redmi K60 Champion (8 Gen 2, V73); Qwen2-0.5B/1.5B, PhoneLM-0.5B/1.5B. Attention kernel up to 6.9x faster (3.5x average), end-to-end up to 4.5x, up to 7.66x lower energy than CPU or GPU attention (Table 8), 0.4 point accuracy loss.
- Scheduler relevance: per-head sparsity is a runtime quality knob that trades attention work for energy, similar in spirit to a KV-size cap.

### 2.6 MNN-AECS: Energy Optimization for LLM Decoding on Mobile Devices via Adaptive Core Selection
- Authors: Zhengxiang Huang, Chaoyue Niu, Zhaode Wang, Jiarui Xue, Hanming Zhang, Yugang Wang, Zewei Xin, Xiaotang Jiang, Chengfei Lv, Fan Wu, Guihai Chen (SJTU, Alibaba MNN).
- Venue and year: arXiv preprint, 24 Jun 2025 (no venue found).
- URL: https://arxiv.org/abs/2506.19884 (numbers from https://arxiv.org/html/2506.19884v1)
- Idea: Because decode is memory-bound, it picks a small set of CPU cores at runtime, without root, to cut energy at almost no speed loss. It is shipped in the MNN engine.
- Numbers: 5 Android and 2 iOS phones including Meizu 21 (8 Gen 3) and Xiaomi 15 Pro (8 Elite); energy from BatteryManager current times voltage. Example: 389 to 298 mJ/token on Mate 40 Pro with Qwen 1.5B (23 percent) while decode went 20.7 to 20.3 tok/s. 39 to 78 percent less energy than llama.cpp, ExecuTorch, mllm and MediaPipe. Tuned settings usually use two performance cores.
- Scheduler relevance: a no-root, per-request core-count knob that composes with a GPU-clock knob and gives the energy model for CPU fallback.

### 2.7 EnerInfer: Energy-Aware On-Device LLM Inference
- Authors: Bohua Zou, Nian Liu, Binqi Sun, Matteo Mascherin, Debayan Roy, Yutao Liu, Yu Peng, Ning Jia, Haibo Chen.
- Venue and year: arXiv preprint, v1 22 Jun 2026, v2 24 Jun 2026.
- URL: https://arxiv.org/abs/2606.23001 (numbers from https://arxiv.org/html/2606.23001v2)
- Idea: Predicts throughput and power of an unseen LLM across NPU and DDR frequency settings from model structure, then picks the most efficient setting that meets a tokens-per-second target and a shell-temperature limit. It uses ranking-driven online feedback under interference and a short-horizon thermal predictor.
- Numbers: high-end and mid-tier phones (models not named), a laptop and Orange Pi 5 Pro; LLaMA2-1.3B/7B, LLaMA3-3B, Qwen2-1.5B, Gemma2-2B. Up to 65 percent energy-efficiency gain on phones (50 to 65 percent at a 5 tok/s target), 12 percent laptop, 24 percent board; 4.2 to 11 percent end-to-end energy in real scenarios. Shell temperature held below 42 C with 27.9 percent more tokens and 32.1 percent longer runs before the threshold. Prefill runs at max frequency; only decode is scaled. The paper states it does not consider battery level or charging state.
- Scheduler relevance: the nearest prior work to the target scheduler (frequency choice per request under a QoE and thermal target), and it leaves battery state, GPU clock, KV size and output length as open knobs.

### 2.8 Accelerating Mobile Language Model via Speculative Decoding and NPU-Coordinated Execution (sd.npu)
- Authors: Zhiyang Chen, Daliang Xu, Haiyang Shen, Chiheng Lou, Mengwei Xu, Shangguang Wang, Xin Jin, Yun Ma.
- Venue and year: arXiv preprint, v1 Oct 2025, v4 Dec 2025.
- URL: https://arxiv.org/abs/2510.15312 (numbers from https://arxiv.org/html/2510.15312v4)
- Idea: For on-device RAG, it pipelines NPU graph reloads with chunked context compute and runs NPU-centric speculative decoding so idle NPU cycles become tokens.
- Numbers: Redmi K60 Pro (8 Gen 2), K70 Pro (8 Gen 3), OnePlus 13 (8 Elite); Qwen2.5-0.5B/1.5B, LLaMA3.2-3B; energy via /sys/class/power_supply. 1.06 to 3.81x speedup and 1.07 to 4.71x less energy; 15.08 versus 57.41 ms per token example; 1.35 to 4.18x less energy than NPU-only decode and 1.11 to 2.50x less than CPU decode.
- Scheduler relevance: speculative decoding shifts energy per token, so a scheduler's J/token model must be per execution mode.

### 2.9 Lever: Speculative LLM Inference on Smartphones
- Authors: Tuowei Wang, Fengzu Li, Yanfan Sun, Wei Gao, Ju Ren (Tsinghua).
- Venue and year: arXiv preprint, 16 May 2026.
- URL: https://arxiv.org/abs/2605.16786
- Idea: Keeps a draft model in DRAM and a large target model in flash so that each flash-resident verification pass checks many tokens. It re-plans drafting, verification and execution for long I/O latency and low parallelism on phones.
- Numbers: 2.93x average speedup over flash-offloaded inference and 1.50x over plain speculative decoding; no energy figures on the abstract page.
- Scheduler relevance: flash-backed decode has a different energy profile than DRAM decode, which a battery-aware policy could switch off at low charge.

### 2.10 AHASD: Asynchronous Heterogeneous Architecture for LLM Adaptive Drafting Speculative Decoding on Mobile Devices
- Authors: Ma Zirui, Fan Zhihua, Li Wenxing, Wu Haibin, Zhang Fulin, Ye Xiaochun, Li Wenming.
- Venue and year: DAC 2026 (arXiv comments). arXiv Apr 2026.
- URL: https://arxiv.org/abs/2604.25326
- Idea: A simulated mobile NPU plus LPDDR5 processing-in-memory design that drafts and verifies in parallel with entropy-aware drafting control. It is a hardware proposal, not a COTS phone result.
- Numbers: up to 4.2x throughput and 5.6x energy efficiency over a GPU baseline; 1.5x and 1.24x over GPU plus PIM; under 3 percent DRAM area.
- Scheduler relevance: peripheral; it shows where future phone hardware may move the decode energy floor.

### 2.11 PrismML 1-bit Bonsai (8B, 4B, 1.7B, 27B)
- Organization: PrismML (Caltech spin-out).
- Venue and year: company announcement, 31 Mar 2026.
- URL: https://prismml.com/news/bonsai-8b
- Idea: 1-bit weight LLMs that fit 8B parameters in 1.15 GB, run through MLX on Apple devices and llama.cpp CUDA on NVIDIA. They claim quality close to 16-bit peers at 14x smaller size.
- Numbers: iPhone 17 Pro about 40 tok/s, iPhone 17 Pro Max about 44 tok/s, M4 Pro 131 tok/s, RTX 4090 368 tok/s. Energy 0.068 mWh per token (0.245 J/token) on iPhone 17 Pro Max and 0.074 mWh per token on M4 Pro; 4 to 5x better energy than 16-bit. No Android numbers on the page.
- Scheduler relevance: 1-bit weights cut DRAM traffic per token, so the same GPU clock gives more tokens per joule; a scheduler can treat weight precision as a static term in its energy model.

### 2.12 bitnet.cpp: 1-bit AI Infra, Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs
- Authors: Jinheng Wang, Hansong Zhou, Ting Song, Shaoguang Mao, Shuming Ma, Hongyu Wang, Yan Xia, Furu Wei (Microsoft).
- Venue and year: arXiv technical report, Oct 2024.
- URL: https://arxiv.org/abs/2410.16144 and https://github.com/microsoft/BitNet (README fetched raw)
- Idea: Ternary-weight kernels for CPUs that avoid dequantisation. The README reports energy on ARM (Apple M2) and x86, not on phones.
- Numbers: ARM CPUs 1.37x to 5.07x faster with 55.4 to 70.0 percent less energy; x86 2.37x to 6.17x with 71.9 to 82.2 percent less energy. No smartphone results.
- Scheduler relevance: peripheral; phone ports of ternary kernels would lower the CPU-fallback energy floor.

### 2.13 MobileQuant: Mobile-friendly Quantization for On-device Language Models
- Authors: Fuwen Tan, Royson Lee, Lukasz Dudziak, Shell Xu Hu, Sourav Bhattacharya, Timothy Hospedales, Georgios Tzimiropoulos, Brais Martinez (Samsung AI Center Cambridge).
- Venue and year: EMNLP 2024 Findings (arXiv comments).
- URL: https://arxiv.org/abs/2408.13933
- Idea: Integer-only post-training quantization that jointly learns weight transforms and activation ranges so W4A8 or W8A8 models run fully on the Hexagon NPU.
- Numbers: 20 to 50 percent lower latency and energy than prior on-device quantization; search results say TinyLlama-1.1B W8A8 on Galaxy S24 HTP (device not confirmed from the abstract page).
- Scheduler relevance: activation quantization decides whether the NPU path is usable, which changes the energy per token the scheduler sees.

### 2.14 Quant.npu: Enabling Efficient Mobile NPU Inference for on-device LLMs via Fully Static Quantization
- Authors: Jinghe Zhang, Daliang Xu, Chenghua Wang, Weikai Xie, Tao Qi, Yun Ma, Mengwei Xu, Gang Huang.
- Venue and year: arXiv preprint, 19 May 2026.
- URL: https://arxiv.org/abs/2605.20295
- Idea: Fully static integer quantization with learnable rotations so no quantization parameters are recomputed at runtime on the NPU.
- Numbers: up to 15.1 percent lower inference latency on real mobile NPUs; no energy figures on the abstract page.
- Scheduler relevance: minor; removes runtime overhead on the NPU path.

### 2.15 Elastic On-Device LLM Service
- Authors: Wangsong Yin, Rongjie Yi, Daliang Xu, Gang Huang, Mengwei Xu, Xuanzhe Liu.
- Venue and year: ACM MobiCom 2025 (arXiv comments).
- URL: https://arxiv.org/abs/2409.09071
- Idea: Makes one on-device model elastic in two dimensions, model width through one-shot neuron reordering and prompt length through a dual-head tiny LM, so the service can meet varying memory and latency limits per request.
- Numbers: up to 14.83 point absolute accuracy gain over 7 baselines, under 1 percent TTFT switching overhead, tested on COTS smartphones; no energy figures on the abstract page.
- Scheduler relevance: the closest published mechanism for a per-request quality knob that a battery-aware policy could drive.

### 2.16 EdgeFlow: Fast Cold Starts for LLMs on Mobile Devices
- Authors: Yongsheng Yan, Jiacheng Shen, Xuchuan Luo, Yangfan Zhou.
- Venue and year: arXiv preprint, 10 Apr 2026.
- URL: https://arxiv.org/abs/2604.09083
- Idea: Adaptive weight quantization, SIMD-friendly packing and a CPU-NPU pipeline to cut model cold start.
- Numbers: up to 4.07x lower cold-start latency than llama.cpp, MNN and llm.npu; no energy figures.
- Scheduler relevance: minor; cold-start energy is a fixed per-session cost the scheduler can amortise.

### 2.17 Battery-state-aware inference: gap
- No 2024 to 2026 paper was found that changes LLM inference settings by battery level, charging state or battery temperature. EnerInfer (2.7) states it does not consider battery level or charging. PowerLens (Xingyu Feng et al., arXiv 2603.19584, Mar 2026, https://arxiv.org/abs/2603.19584) uses an LLM agent to tune 18 Android power settings and saves 38.8 percent energy versus stock Android, but it does not control LLM inference itself. Device-cloud offloading work such as TMO (Liangqi Yuan, Dong-Jun Han, Shiqiang Wang, Christopher Brinton, ACM MobiHoc 2025 best paper runner-up, extended in IEEE/ACM ToN, https://arxiv.org/abs/2502.11007) picks device versus cloud with resource-constrained RL, but its abstract does not mention battery level.
- Scheduler relevance: a per-request policy keyed on battery level, temperature and charging state is unclaimed territory.

---

## 3. KV-cache work aimed at phones or edge devices (2025 to 2026)

### 3.1 KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference
- Authors: Huawei Zhang, Chunwei Xia, Zheng Wang (University of Leeds).
- Venue and year: ACM MobiSys 2026, DOI 10.1145/3745756.3809234 (confirmed on the White Rose eprint page).
- URL: https://arxiv.org/abs/2511.11907 and https://eprints.whiterose.ac.uk/id/eprint/240121/ (numbers from https://arxiv.org/html/2511.11907v2)
- Idea: Stores the whole KV cache on disk, keeps compact metadata in memory to predict which entries to preload, and groups reads to match disk characteristics. Only a small changing subset of KV entries is needed per step.
- Numbers: evaluated on Jetson Orin AGX (64 GB) with NVMe (1.8 GB/s) and eMMC (250 MB/s), not on a phone. LLaMA-3.1-8B, 3.2-3B, Qwen3 4B to 14B; 4K to 32K context. At 32K context and batch 8: 35.6 tok/s versus ShadowKV 21.5 on NVMe, 15.7 versus 4.5 tok/s on eMMC; 11.0x less KV memory than vLLM; accuracy loss at most 2.6 percent (NVMe) and 4.4 percent (eMMC) on RULER. No energy numbers.
- Scheduler relevance: a swap-based alternative to eviction whose disk traffic is an energy term; a battery-aware scheduler would prefer eviction over swapping at low charge.

### 3.2 DynaKV: Enabling Accurate and Efficient Long-Sequence LLM Decoding on Smartphones
- Authors: Tuowei Wang, Minxing Huang, Fengzu Li, Ligeng Chen, Jinrui Zhang, Ju Ren (Tsinghua).
- Venue and year: arXiv preprint, Oct 2025.
- URL: https://arxiv.org/abs/2511.07427 (numbers from https://arxiv.org/html/2511.07427v1)
- Idea: Splits the KV cache between DRAM and UFS flash with adaptive clusters, a flash-friendly layout and a cache replacement policy tuned to retrieval accuracy.
- Numbers: OnePlus Ace5 Pro (8 Elite, UFS 4.0), OnePlus 12 (8 Gen 3), Ace3 (8 Gen 2), Ace2 (8+ Gen 1, UFS 3.1); five models 1.2B to 7.8B with FP8 KV. Llama3.2-1B at 64K tokens has a 2.1 GB KV cache versus 1.2 GB weights. 1.38x retrieval accuracy and 1.47x speedup; 1.56 to 2.10x effective bandwidth; Figure 18: 1.57x lower energy on average with similar power.
- Scheduler relevance: the only phone KV paper found with a measured energy result, and it shows KV placement changes energy per request on the same Snapdragon phones the scheduler targets.

### 3.3 LLM as a System Service on Mobile Devices (LLMS)
- Authors: Wangsong Yin, Mengwei Xu, Yuanchun Li, Xuanzhe Liu.
- Venue and year: arXiv technical report, Mar 2024.
- URL: https://arxiv.org/abs/2403.11805
- Idea: Treats the LLM as an OS service shared by apps and manages per-app KV caches in chunks with tolerance-aware compression, IO-recompute pipelined loading and an LCTRU eviction queue.
- Numbers: context-switch latency reduced by up to two orders of magnitude on edge devices; no energy figures.
- Scheduler relevance: shows a system-level owner of KV memory that could expose a KV budget knob to a battery-aware policy.

### 3.4 KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments
- Authors: Junyoung Park, Dalton Jones, Matthew J Morse, Raghavv Goel, Mingu Lee, Chris Lott.
- Venue and year: NeurIPS 2025 (arXiv comments; v4 Jan 2026).
- URL: https://arxiv.org/abs/2504.15364
- Idea: Training-free eviction that keeps geometrically distinctive keys, since those tend to draw high attention. It works with fixed budgets and no attention-score access.
- Numbers: at an 8K budget, about 23 percent KV reduction with under 0.04 percent LongBench gap on Llama 3.1-8B and 3.2-3B; up to 30 percent lower latency than other eviction methods. A search snippet says scoring latency was measured on a Samsung Android phone (not confirmed from the abstract page). No energy figures.
- Scheduler relevance: an eviction score that is cheap on phone GPUs, which is what a KV-size cap needs at runtime.

### 3.5 Kelle: Co-design KV Caching and eDRAM for Efficient LLM Serving in Edge Computing
- Authors: Tianhua Xia, Sai Qian Zhang.
- Venue and year: MICRO 2025 (ACM DOI 10.1145/3725843.3756071 in search listing; venue unverified from the arXiv page). arXiv Oct 2025.
- URL: https://arxiv.org/abs/2510.16040
- Idea: Uses eDRAM for the KV cache on an edge accelerator and co-designs eviction, recomputation and refresh control to cut refresh energy.
- Numbers: 3.9x speedup and 4.5x energy savings over baselines (simulated hardware).
- Scheduler relevance: peripheral; it confirms KV refresh and movement, not compute, dominate KV energy.

### 3.6 QKVShare: Quantized KV-Cache Handoff for Multi-Agent On-Device LLMs
- Authors: Pratik Honavar, Tejpratap GVSL.
- Venue and year: arXiv preprint, 5 May 2026.
- URL: https://arxiv.org/abs/2605.03884
- Idea: Hands a token-level mixed-precision KV cache between on-device agents in a self-contained "CacheCard" instead of re-prefilling.
- Numbers: Llama-3.1-8B; 397.1 ms versus 1029.7 ms re-prefill at 8K context; no energy or device named.
- Scheduler relevance: minor; avoids repeated prefill energy in multi-agent flows.

### 3.7 Apple on-device KV-cache sharing
- Apple's 2025 tech report (see 4.5) states the 3B on-device model shares Block 2 KV caches with Block 1, cutting KV memory by 37.5 percent, alongside 2-bit quantization-aware training.
- Scheduler relevance: architectural KV reduction lowers the memory the scheduler must cap.

---

## 4. Vendor and framework material

### 4.1 Qualcomm, Unlocking on-device generative AI with an NPU and heterogeneous computing (whitepaper)
- Organization: Qualcomm Technologies.
- Venue and year: vendor whitepaper (PDF text extracted locally; date not printed in the extracted text, content covers Snapdragon 8 Gen 3, late 2023 to 2024).
- URL: https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Unlocking-on-device-generative-AI-with-an-NPU-and-heterogeneous-computing.pdf
- Idea: Argues that the Hexagon NPU is the right engine for sustained AI at low power and that the AI Engine should route on-demand tasks by latency and sustained tasks by power efficiency.
- Numbers and claims: Hexagon NPU in Snapdragon 8 Gen 3 gives "98% faster performance and 40% improved performance-per-watt" for sustained generative AI; dedicated NPU power rail since 8 Gen 2; INT4 support; "for sustained and pervasive use cases, in which battery life is vital and power efficiency is the critical factor, the NPU is the best option".
- Scheduler relevance: vendor guidance already frames engine choice by power budget, which a battery-aware scheduler can formalise.

### 4.2 Qualcomm Snapdragon 8 Elite NPU claims (via Android Authority deep dive)
- Organization: Qualcomm claims, reported by Android Authority.
- Venue and year: press article, page dated 27 Apr 2025 (chip launched Oct 2024).
- URL: https://www.androidauthority.com/snapdragon-8-elite-deep-dive-3491526/
- Claims: Hexagon NPU up to 45 percent faster and 45 percent better AI performance per watt "depending on what's being run"; up to 70 tok/s on certain small models. No thermal statements.
- Scheduler relevance: marketing numbers only; measured NPU decode on SM8850 in item 1.5 is 0.43 J/token, above CPU.

### 4.3 Qualcomm AI Engine Direct (QNN) HTP performance modes (via ONNX Runtime QNN EP docs)
- Organization: Qualcomm QNN, documented by ONNX Runtime.
- Venue and year: framework documentation, current page fetched 2026-09-04.
- URL: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- Content: htp_performance_mode accepts burst, balanced, default, high_performance, high_power_saver, low_balanced, low_power_saver, power_saver, sustained_high_performance; rpc_control_latency in microseconds is a separate option. The page documents these as session-level provider options. The underlying QNN_HTP_PERF_PROFILE enum mapping and DCVS voltage-corner settings could not be fetched (Qualcomm docs page rendered empty; ONNX Runtime and ExecuTorch source paths returned 404), so that mapping is unverified here.
- Scheduler relevance: these modes are the public NPU-side power knob; item 1.5 shows that RPC polling and NPU sleep latency settings alone swing decode energy by about half.

### 4.4 Apple, Introducing Apple's On-Device and Server Foundation Models
- Organization: Apple Machine Learning Research.
- Venue and year: Apple research post, 10 Jun 2024.
- URL: https://machinelearning.apple.com/research/introducing-apple-foundation-models
- Content: about 3B parameter on-device model with mixed 2-bit and 4-bit palettization averaging 3.7 bits per weight; on iPhone 15 Pro about 0.6 ms per prompt token time-to-first-token and 30 tok/s generation before token speculation. No power numbers.
- Scheduler relevance: sets the throughput a first-party 3B model achieves on a 2023 flagship, which is a QoE floor for third-party runtimes.

### 4.5 Apple Intelligence Foundation Language Models Tech Report 2025
- Organization: Apple.
- Venue and year: Apple tech report, Jul 2025 (arXiv 2507.13575).
- URL: https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025 and https://arxiv.org/abs/2507.13575
- Content: 3B on-device model with KV-cache sharing (37.5 percent less KV memory) and 2-bit quantization-aware training; a Swift Foundation Models framework with guided generation, tool calling and LoRA adapters. No latency, power or thermal numbers on the fetched pages.
- Scheduler relevance: Apple exposes no power mode to developers on this page; third-party thermal measurements of the iPhone 16 Pro are in item 1.4.

### 4.6 Google, Gemini Nano on Android (AICore)
- Organization: Google, Android Developers.
- Venue and year: developer documentation, fetched 2026-09-04.
- URL: https://developer.android.com/ai/gemini-nano
- Content: Gemini Nano runs inside the AICore system service; "AICore leverages on-device hardware to accelerate inference"; ML Kit GenAI APIs (prompt, summarization, proofreading, rewriting, image description) sit on top. The page has no battery, thermal or power statements and names no accelerator. Third-party blog claims that AICore routes between GPU and NPU by thermal state and battery level were not found in Google documentation (unverified).
- Scheduler relevance: the platform hides its power policy, so a research scheduler must work below AICore, in an app-level runtime.

### 4.7 Google Research, Accelerating Gemini Nano models on Pixel with frozen Multi-Token Prediction
- Organization: Google Platforms and Devices (Eden Cohen, Michelle Ramanovich).
- Venue and year: Google Research blog, 26 Jun 2026.
- URL: https://research.google/blog/accelerating-gemini-nano-models-on-pixel-with-frozen-multi-token-prediction/
- Content: a frozen multi-token-prediction head on Gemini Nano v3 for Pixel 9 and 10 gives 50 percent or more faster generation, nearly two extra accepted tokens per pass in Notification Summaries and Proofread, 130 MB less memory than a separate drafter, and "less time waking heavy processors, reducing energy consumption and improving battery life" (qualitative).
- Scheduler relevance: fewer forward passes per token is a production energy lever on the same class of device.

### 4.8 Android Authority, Gemini Nano 4 tested on Pixel 10 Pro XL
- Organization: Android Authority (independent test) reporting Google claims.
- Venue and year: press article, 11 Apr 2026.
- URL: https://www.androidauthority.com/gemini-nano-4-benchmarks-3655763/
- Numbers: on Tensor G5 TPU, Gemini Nano 3 averaged 9.6 tok/s, Nano 4 Fast 19.14 tok/s, Nano 4 Full 5.3 tok/s. Google claims up to 4x faster and up to 60 percent less battery on the TPU. The tester saw about 2x because Nano 4 is more verbose, often writing 50 to 100 percent more text for the same query.
- Scheduler relevance: a direct example of output length erasing a hardware efficiency gain, which motivates the output-length cap.

### 4.9 PyTorch and Arm, Unleashing the Power of AI on Mobile: Llama 3.2 quantized models with ExecuTorch and KleidiAI
- Organization: Arm and Meta (PyTorch blog).
- Venue and year: PyTorch blog, posted late 2024, page last updated 7 May 2025.
- URL: https://pytorch.org/blog/unleashing-ai-mobile/
- Numbers: Samsung S24+, Llama 3.2 1B SpinQuant and QLoRA 4-bit per-block, 6 of 8 CPU cores, prompt 64, sequence 128: over 350 tok/s prefill and over 40 tok/s decode; more than 2x decode and 5x prefill over BF16; peak RSS 1.9 GiB versus 3.1 GiB; KleidiAI adds about 20 percent prefill. No power or thermal numbers.
- Scheduler relevance: shows CPU-only decode above 40 tok/s on a Snapdragon 8 Gen 3 class phone, so a scheduler has real headroom to trade speed for energy.

### 4.10 llama.cpp Snapdragon backend README
- Organization: ggml-org (llama.cpp maintainers).
- Venue and year: repository documentation, fetched 2026-09-04.
- URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md
- Content: three backends on Snapdragon: CPU, Adreno GPU (OpenCL) and Hexagon NPU (experimental, HTP0 to HTPn virtual sessions, Hexagon v73, v75, v79, v81). Sample: Llama 1B Q4_0 at 169.42 tok/s prompt and 51.54 tok/s generation. No power, thermal or battery guidance. Community discussion #14356 (Jun to Jul 2025, https://github.com/ggml-org/llama.cpp/discussions/14356) compares CPU, QNN-GPU, QNN-NPU and cDSP matmul on 8 Gen 3 and 8 Elite but has no power data.
- Scheduler relevance: llama.cpp exposes backend and layer-offload choice but no per-request power control; a scheduler must add clock and KV control on top.

### 4.11 MLCommons, MLPerf Inference: Mobile (v6.0) and MLPerf Power
- Organization: MLCommons.
- Venue and year: benchmark pages, fetched 2026-09-04; MLPerf Power paper at IEEE HPCA 2025 (MLCommons blog, 4 Mar 2025).
- URL: https://mlcommons.org/benchmarks/inference-mobile/ and https://mlcommons.org/2025/03/ml-commons-power-hpca/
- Content: MLPerf Mobile v6.0 includes Llama 3.1 8B and Llama 3.2 3B (TinyMMLU, IFEval), Stable Diffusion 1.5, MobileNetV4, MobileDETS, MOSAIC, MobileBERT and EDSR; submissions may include optional power columns including "Energy Per Stream". MLPerf Power measures performance per unit power across datacenter, edge, mobile and tiny divisions.
- Scheduler relevance: an emerging standard energy-per-stream metric for phone LLMs, but it reports peak, not sustained or throttled, behaviour.

### 4.12 Geekbench AI
- URL attempted: https://www.geekbench.com/ai/ (HTTP 403). Not verified. Search results indicate it scores CPU, GPU and NPU on vision and language workloads and does not report power. Marked unverified.

---

## 5. Scheduling or control of LLM inference on a phone by energy budget, deadline or quality target

### 5.1 EnerInfer (see 2.7)
- Per-model, per-request choice of NPU and DDR frequency to meet a tokens-per-second target and a 42 C shell limit, with 50 to 65 percent efficiency gain at a 5 tok/s target. It is the most direct precedent for a QoE-constrained frequency scheduler, but it does not use battery state and does not touch GPU clock, KV size or output length.

### 5.2 FUSE (see 1.6)
- A unified CPU, GPU and memory governor for LLM phases on Pixel 7. Shows that default governors waste up to 40.4 percent latency at equal energy and that decode energy per token on Mali ranges from about 400 mJ to 955 mJ across models.

### 5.3 Is Your NPU Ready for LLMs? (see 1.5)
- Shows a coordinated static setting (RPC polling, NPU sleep latency, fixed low CPU clock) cuts Snapdragon decode energy by 54.8 percent for 13.4 percent more latency. This is the energy-delay trade the target scheduler would move along per request.

### 5.4 MNN-AECS (see 2.6)
- Runtime core selection without root, 23 percent decode energy cut at under 2 percent speed loss. An orthogonal knob to GPU clock.

### 5.5 AgentStop: Terminating Local AI Agents Early to Save Energy in Consumer Devices
- Authors: Dzung Pham, Kleomenis Katevas, Ali Shahin Shamsabadi, Hamed Haddadi (Brave, Imperial).
- Venue and year: ACM CAIS 2026 (arXiv comments). arXiv 1 May 2026.
- URL: https://arxiv.org/abs/2605.15206
- Idea: Watches token log-probabilities during local agent runs and stops trajectories that are likely to fail. It targets the long, iterative generations that dominate agent energy.
- Numbers: 15 to 20 percent less wasted energy with under 5 percent utility loss on web QA and coding tasks on consumer devices (devices not named on the abstract page).
- Scheduler relevance: an output-length control driven by a cheap runtime signal, which is the same mechanism a battery-aware output cap would use.

### 5.6 Brevity is the soul of sustainability: Characterizing LLM response lengths
- Authors: Soham Poddar, Paramita Koley, Janardan Misra, Sanjay Podder, Navveen Balani, Niloy Ganguly, Saptarshi Ghosh.
- Venue and year: ACL 2025 Findings (arXiv comments).
- URL: https://arxiv.org/abs/2506.08686
- Idea: Benchmarks 12 decoder-only models on 5 datasets and finds responses are often much longer than needed. Prompt-level length control cuts energy while keeping quality.
- Numbers: 25 to 60 percent energy savings from shorter responses; server GPUs, not phones.
- Scheduler relevance: quantifies how much an output-length cap can save before any hardware knob is touched.

### 5.7 The Diminishing Returns of Early-Exit Decoding in Modern LLMs
- Authors: Rui Wei, Rui Du, Hanfei Yu, Devesh Tiwari, Jian Li, Zhaozhuo Xu, Hao Wang.
- Venue and year: arXiv preprint, 24 Mar 2026.
- URL: https://arxiv.org/abs/2603.23701
- Idea: Layer-wise early exit gets weaker on newer models because better pretraining reduces layer redundancy; dense models keep more early-exit potential than MoE or state-space models.
- Numbers: no mobile or energy measurements.
- Scheduler relevance: argues against layer early-exit as the quality knob and in favour of length and KV knobs.

### 5.8 Scaling LLM Test-Time Compute with Mobile NPU on Smartphones
- Authors: Zixu Hao, Jianyu Wei, Tuowei Wang, Minxing Huang, Huiqiang Jiang, Shiqi Jiang, Ting Cao, Ju Ren.
- Venue and year: arXiv preprint, 27 Sep 2025.
- URL: https://arxiv.org/abs/2509.23324
- Idea: Uses idle Snapdragon NPU capacity for parallel test-time scaling so a small model matches a larger one, with tile quantization and LUT operators.
- Numbers: up to 19.0x mixed-precision GEMM and 2.2x softmax speedups; a new accuracy-cost Pareto frontier; no energy figures on the abstract page.
- Scheduler relevance: the opposite direction of the same knob: spend more tokens for quality when charging, fewer when on battery.

### 5.9 Elastic On-Device LLM Service (see 2.15)
- Per-request model-width and prompt-length elasticity with under 1 percent switch overhead, published at MobiCom 2025. The natural quality knob for a battery-aware policy.

### 5.10 Device-cloud routing (TMO, see 2.17)
- RL-based choice of device versus cloud per turn under resource constraints (MobiHoc 2025). Battery level is not part of the abstract's decision inputs.

---

## 6. Numbers table (phone-only, as reported)

| Item | Device | Model | Engine | Decode tok/s | Energy per token | Power | Temperature |
|---|---|---|---|---|---|---|---|
| MELT 2024 | iPhone 14 Pro | Zephyr-3B 4-bit | MLC | n/a | 0.20 mWh (0.72 J) | 13.8 W sustained, 18 W peak | 47.9 C |
| MELT 2024 | Galaxy S23 | Zephyr-3B 4-bit | MLC | n/a | n/a | 8.5 W sustained, 14 W peak | throttles after 20 to 32 prompts |
| PowerInfer-2 2024 | OnePlus 12 | Mixtral-47B sparse | PowerInfer-2 | 11.68 | 0.257 J | 5.095 W peak | n/a |
| Pockets, TMC 2026 | Xiaomi 14 Pro and others | 4-bit 1B to 7B | llama.cpp, MLC | 2 to 10 (CPU) | GPU decode 0.058 mAh (about 0.8 J at 3.85 V) | n/a | 34 to 42 C |
| MNN-AECS 2025 | Mate 40 Pro | Qwen2.5-1.5B | MNN | 20.3 | 0.298 J (from 0.389) | n/a | n/a |
| FUSE 2025 | Pixel 7 | TinyLlama 1.1B 4-bit | llama.cpp GPU | 7.9 (126.9 ms) | 0.397 J | n/a | n/a |
| Edge trade-offs 2026 | Galaxy S24 Ultra | Qwen2.5-1.5B 4-bit | MLC | 10.38 plateau, 12.21 peak | 0.146 J (as reported) | n/a | GPU 68.5 C peak |
| Edge trade-offs 2026 | iPhone 16 Pro | Qwen2.5-1.5B 4-bit | MLX | 23.67 hot, 40.49 peak | n/a (5 percent battery per 20 runs) | n/a | n/a |
| Is Your NPU Ready 2026 | Xiaomi 17 (SM8850) | Qwen2.5-1.5B 4-bit | GENIE NPU vs CPU | n/a | 0.43 J NPU, 0.12 J CPU | n/a | cooled below 28 C first |
| Bonsai 2026 | iPhone 17 Pro Max | Bonsai-8B 1-bit | MLX | 44 | 0.068 mWh (0.245 J) | n/a | n/a |
| Gemini Nano 4 2026 | Pixel 10 Pro XL | Nano 4 Fast | AICore TPU | 19.14 | n/a (claim: 60 percent less battery) | n/a | n/a |

---

## 7. Implications for a battery-aware per-request scheduler on Snapdragon

1. GPU clock. Adreno DVFS already boosts to 1000 MHz and then settles at 720 to 770 MHz under heat (1.4). Decode is bandwidth-bound, so a capped clock costs little throughput (1.5, 1.6, 2.7). No paper sets the Adreno clock per request from battery state.
2. KV-cache size. Phone KV work uses flash swapping (3.1, 3.2) or chunked compression (3.3). Only DynaKV reports energy (1.57x). No paper ties the KV budget to battery or temperature.
3. Output-length cap. Length is the largest energy lever (5.5, 5.6, 4.8). No phone paper caps length by battery state.
4. Battery state. EnerInfer explicitly excludes battery level and charging (2.7). The field has thermal-aware and QoE-aware control, but not battery-aware control.
5. Measurement practice worth copying: cool below 28 C before runs (1.5); bypass the battery with a Monsoon monitor for rail-level power (1.6); or read /sys/class/power_supply at 100 ms for whole-phone energy (2.2, 2.8); report J/token separately for prefill and decode (1.2, 1.5).

---

## 8. Search log

Queries run (40): MELTing point MobiCom 2024; on-device LLM energy measurement 2025 J/token; PowerInfer-2; llm.npu ASPLOS 2025; HeteroLLM; KVSwap MobiSys; thermal throttling LLM smartphone 2025; battery-aware LLM inference smartphone; EdgeMoE; speculative decoding smartphone energy; Bonsai PrismML; BitNet mobile ARM energy; Qualcomm QNN HTP performance modes; Apple Foundation Models on-device 2025; MNN-AECS venue; KV cache eviction on-device 2025 2026; Android AICore Gemini Nano power; ExecuTorch Llama 3.2 Android; MLC LLM Android energy; llama.cpp Android power discussion; MLPerf Mobile Geekbench AI power; energy budget deadline scheduler on-device LLM; early exit anytime LLM mobile; output length control energy; GPU DVFS LLM mobile Adreno; Snapdragon 8 Elite NPU per watt; thermal-aware on-device LLM 2026; battery charging adaptive on-device LLM; KV cache quantization flash UFS smartphone 2026; energy-delay product LLM mobile; SLM smartphone energy Perfetto; MLPerf Mobile v5 LLM power; Mengwei Xu 2026 NPU energy; LLM smartphone battery level local cloud routing; Kelle MICRO 2025; Apple Foundation Models measured iPhone 16 power; Gemini Nano Pixel measured power; on-device KV eviction measured energy 2026; MobileQuant; LLM as a System Service; battery saver low power mode on-device LLM.

Pages that could not be fetched (403, 404 or empty): ACM DL pages for llm.npu, Sustainable LLM Inference (TIoT) and Kelle; Qualcomm HTP backend documentation page (JS-only); Geekbench AI page; ONNX Runtime and ExecuTorch source files for the QNN perf-profile mapping. Items that depend on them are marked unverified above.
