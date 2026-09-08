# Learning-based and control-theoretic performance versus energy trade-offs on mobile devices and SoCs

Survey date: 2026-09-04. Method: 20 web searches plus about 90 page fetches (arXiv abstracts and HTML bodies, Semantic Scholar API, dblp, USENIX, GitHub, author pages), plus the text extracts already in this directory (jouleguard, racepace, geardvfs, dora, lesueur, sosa, fse2015, poet.html, hoffmann.html, imes.html). Every item lists a URL that was opened or that appeared in a search result page. "Verified" means venue and year were confirmed from a fetched page (Semantic Scholar API, dblp, arXiv, USENIX, or the local extract). Items whose numbers come only from a search snippet say so. Items that could not be confirmed are marked "unverified".

Target use: a two-loop on-device scheduler. A performance loop enforces a time budget. An energy loop minimizes energy inside that budget. A scalar lever set by battery tier fixes the exchange rate between the two.

Numbers use "to" for ranges. No em dashes are used.

## 0. Short answers to the five questions

1. RL and bandit governors on phones. zTT (MobiSys 2021), GearDVFS (MobiCom 2023), MetaDVFS (arXiv 2025), MobiRL (TACO 2024) and the DATE 2020 user-interaction agent all learn a Q-function or policy over CPU and GPU frequency from on-device state (clocks, utilization, power, temperature, fps). Reported learning cost: GearDVFS needs about 35 minutes to train from scratch and 500 to 650 s to adapt on device; MetaDVFS needs 3.5 minutes (plus or minus 1.1) per new device-app pair; PolyThrottle's Bayesian optimizer needs about 15 samples; PaRMIS needs 300 to 500 policy evaluations. All of them beat the stock governor (schedutil plus simple_ondemand) by 4 to 24 percent in performance per watt on phones. None of the phone RL papers reports a comparison against a per-application tuned fixed frequency. FUSE (2025) shows that a fixed, offline-profiled frequency triple beats the default governors by up to 40 percent latency at equal energy, so "beats the stock governor" is a weak bar. No paper was found that tunes governor parameters with Bayesian optimization on a phone, and no contextual-bandit DVFS paper on a phone was found; the closest are JouleGuard's bandit over system configurations (SOSP 2015) and Autothrottle's contextual bandit for a datacenter throttle target (NSDI 2024).

2. Two coupled loops. The literature contains six distinct ways to keep a performance loop and an energy loop from fighting: (a) split by role and pass a summary signal (JouleGuard: a bandit finds the most efficient system configuration, a PI controller adds application speedup, and the controller pole is set from the learner's model error so control slows when learning is uncertain); (b) one controller computes a required speedup and an optimizer translates it into the minimum-energy pair of configurations (POET, CALOREE, MEANTIME); (c) cascade by priority with disjoint knob sets (Filieri, Hoffmann, Maggio FSE 2015; CoAdapt); (d) supervisory control that sets references and priorities of low-level loops and switches objective on a thermal event (SOSA, SPECTR); (e) hierarchy in time scale, global slow adaptation sets quality and budget, local fast loops enforce (GRACE-1, Autothrottle, E4, ALERT); (f) rate limiting of uncoordinated loops (WASL). A scalar exchange-rate lever appears explicitly in Odyssey (user battery-lifetime goal), GRACE-1 (desired battery lifetime), JouleGuard (energy goal factor f), CoAdapt (which two of three dimensions are fixed), and Budget RNNs (discrete energy budget levels).

3. EDP, Pareto and race versus pace. Kim, Imes, Hoffmann (CPSNA 2015) prove that the energy-optimal schedule uses at most two configurations, that pace-to-idle always beats race-to-idle, and measure on an Exynos 5 big.LITTLE board that race-to-idle uses 3.34x the optimal energy while pace-to-idle uses 1.12x. Le Sueur and Heiser (HotPower 2010) show DVFS savings shrink to zero as the voltage range narrows; EDP was minimized at the highest frequency on their 2009 platform. De Vogeleer et al. (2014) measure a convex energy-versus-frequency curve on a smartphone with an interior minimum. Miyoshi et al. (ICS 2002) show the answer depends on the "critical power slope", that is on idle power. FUSE (2025) and PolyThrottle (2023) show memory frequency is a first-order knob for memory-bound inference. On linear versus cubic scaling: the RL papers assume E proportional to f cubed (GearDVFS) or quadratic power (FUSE); the measurement papers show that once voltage is pinned the dynamic term is linear in f, so lowering the clock does not save energy per operation for compute-bound work and can cost energy through static power and longer runtime; it still lowers peak power and heat, and it saves energy when the work is memory-bound.

4. Battery- and thermal-aware quality scaling of on-device ML. The foundational lever is Odyssey (SOSP 1999): the user sets a battery-lifetime goal and the OS lowers application fidelity to meet it. NestDNN (MobiCom 2018), ALERT (ATC 2020), adaptive model selection (TECS 2020), Budget RNNs (RTAS 2021), SLEXNet (TECS 2024) and E4 (AAAI 2025) select model capacity, exit point or model variant under latency, energy or budget constraints. For LLMs in 2025 to 2026: EnerInfer switches between an energy-optimized and a thermally constrained mode from a short-horizon thermal prediction and lowers NPU and DDR clocks; DVFSLM meets token-generation deadlines at minimum energy; PELM varies speculative verification depth jointly with DVFS; adaptive KV-cache quantization (CVPR 2026) picks per-token precision; an output-length study (2025) shows the best model size changes with the token budget; a 2026 study finds early-exit gains are shrinking in newer LLMs. No paper was found that uses battery level or battery tier as an explicit input to LLM knobs (output length, KV budget, quantization); temperature is used (EnerInfer), deadlines are used (DVFSLM, FUSE), battery is not.

5. Learned versus simple rules on real phones with learning cost (2024 to 2026). MetaDVFS (five Pixels, 3.5 min adaptation, up to 17 percent PPW over the best of schedutil, zTT, GearDVFS, Orthrus) and MobiRL (TACO 2024, deployed in commercial phones, 42.8 percent less power than the commercial scheduler) are the strongest phone results. GearDVFS on a Redmi Note 9 gains only 6 to 9 percent over the stock governor (zTT 4 to 7 percent) despite 35 minutes of training. FUSE, DVFSLM and MNN-AECS get 23 to 40 percent gains on LLM workloads with no online learning at all, using offline profiles or analytic estimators. PowerLens (2026) learns user preferences in 3 to 5 days. The pattern: model-based or profiled rules match or beat learned governors on phones when the workload is known (LLM decode is such a workload), and learned governors pay off mainly when workloads are mixed and unknown.

## 1. Reinforcement learning, bandit and Bayesian DVFS or thermal management on phones and SoCs

### 1.1 zTT: Learning-based DVFS with Zero Thermal Throttling for Mobile Devices
Seyeon Kim, Kyungmin Bin, Sangtae Ha, Kyunghan Lee, Song Chong. MobiSys 2021. Verified (Semantic Scholar API, GitHub README). URLs: https://dl.acm.org/doi/10.1145/3458864.3468161 and https://github.com/ztt-21/zTT
Summary: A DQN jointly sets CPU and GPU frequency from a state of clocks, power, temperature and application fps, with a reward that pays for meeting the target fps, penalizes power, and penalizes approaching the thermal threshold. It was implemented on a Google Pixel 3a and a Jetson TX2.
Numbers: in a hot environment where the default DVFS cannot hold the target frame rate, zTT holds it with 23.9 percent less average power. Training length is not stated in the accessible sources; GearDVFS reports that zTT explores a subset of nine actions per step and took about 700 s to adapt to a QoE change.
Relevance: zTT folds performance, energy and thermal into one scalar reward, so the exchange rate is hidden in reward weights; a two-loop design makes that rate explicit and settable by battery tier.

### 1.2 GearDVFS: A Workload-Aware DVFS Robust to Concurrent Tasks for Mobile Devices
Chengdong Lin, Kun Wang, Zhenjiang Li, Yu Pu. MobiCom 2023. Verified (local extract, ACM DOI). URLs: https://dl.acm.org/doi/10.1145/3570361.3592524 and https://www.cs.cityu.edu.hk/~zhenjili/2023-MobiCom-GearDVFS.pdf
Summary: An LSTM autoencoder turns raw counters (active versus stalled CPU cycles, GPU utilization, clocks, temperatures) into "meta-states"; a branched DQN then picks a frequency per power domain, which shrinks the action space from 1872 combinations to 37 outputs on a TX2. The reward keeps utilization near 80 percent and reuses zTT's thermal penalty at 50 C.
Numbers: PPW gains of 15.0 to 23.9 percent over schedutil plus simple_ondemand and 6.2 to 26.9 percent over zTT on Jetson NX under concurrent tasks; on a Redmi Note 9 phone the gains are 6 to 9 percent (zTT 4 to 7 percent), up to 16.92 percent after fine-tuning on six user traces. Training from scratch takes about 35 minutes on a desktop CPU; on-device adaptation takes about 500 s for a QoE change and about 650 s for an unseen task; runtime cost about 170 mW and one decision per 100 ms.
Relevance: the branch-per-domain Q-network is a learned analog of decoupled per-actuator loops, and the 100 ms decision period sets a realistic bound on how fast an energy loop can move a phone clock.

### 1.3 MetaDVFS: Metadata-Guided Adaptable Frequency Scaling across Heterogeneous Applications and Devices
Jinqi Yan, Fang He, Qianlong Sang, Bifeng Tong, Peng Sun, Yili Gong, Chuang Hu, Dazhao Cheng. arXiv 2509.22707, 23 Sep 2025, preprint. Verified (arXiv abstract and HTML body). URL: https://arxiv.org/abs/2509.22707
Summary: DVFS is cast as multi-task RL over device and application pairs, with device and application metadata used to transfer a DQN (liquid neural network backbone) across Pixel 3, 4, 6, 8 and 9 and across TikTok, Kwai, Bilibili, Weibo, Taobao and 3DMark. State is IPC, CPU and GPU utilization and frequency, and power; actions are per-cluster and GPU frequencies.
Numbers: up to 17 percent PPW and up to 26 percent QoE over the best of schedutil, zTT, GearDVFS and Orthrus; adaptation to a new device-app pair takes 3.5 plus or minus 1.1 minutes versus 11.8 plus or minus 5.2 minutes for incremental training (70.8 percent faster); 5.8 to 27.6 percent better than device-application-specific training. No fixed-frequency oracle or tuned static policy is reported.
Relevance: it gives the best 2025 estimate of online learning cost on real phones (minutes, not hours), and it confirms that learned governors are still benchmarked against other learned governors rather than against tuned static policies.

### 1.4 MobiRL: An Intelligent Scheduling Approach on Mobile OS for Optimizing UI Smoothness and Power
Xinglei Dou, Lei Liu, Limin Xiao. ACM TACO 2024. Verified (Semantic Scholar API). URL: https://dl.acm.org/doi/10.1145/3674910
Summary: An RL scheduler adjusts CPU and GPU frequency from device status to trade frame drops against power on current smartphones. The authors state it is implemented in commercial mobile devices.
Numbers: 4.1 percent lower frame-drop rate and 42.8 percent lower power than the commercial scheduler; versus a Q-learning scheduler up to 2.5 percent lower frame drops and 32.6 percent less power.
Relevance: a rare deployed learned governor on phones; its reward again mixes performance and power with fixed weights.

### 1.5 User Interaction Aware Reinforcement Learning for Power and Thermal Efficiency of CPU-GPU Mobile MPSoCs
Somdip Dey, Amit Singh, Xiaohang Wang, Klaus McDonald-Maier. DATE 2020. Verified (Essex repository page). URL: https://repository.essex.ac.uk/27546/
Summary: A QoS agent watches user behaviour to infer the frame rate the user actually needs, then an RL DVFS agent drives a CPU-GPU Exynos platform to that frame rate. The performance target is learned from the user rather than fixed.
Numbers: up to 50 percent power saving and 29 percent lower peak temperature versus stock Android power saving; 41 percent and 19 percent better than prior power and thermal schemes.
Relevance: it separates "what performance is enough" (a target-setting loop) from "how to reach it cheaply" (the DVFS loop), which is the same split as a time-budget loop above an energy loop.

### 1.6 An Interpretable Machine Learning Model Enhanced Integrated CPU-GPU DVFS Governor
Jurn-Gyu Park, Nikil Dutt, Sung-Soo Lim. ACM TECS 2021. Verified (Semantic Scholar API). URL: https://dl.acm.org/doi/10.1145/3470974
Summary: Model trees (piecewise linear) are trained offline on mobile games and used online inside an integrated CPU-GPU governor. Interpretability is treated as a design constraint.
Numbers: over 10 percent (up to 38 percent) lower energy per frame with 3 percent higher fps over a linear-regression governor, on 20 mobile games.
Relevance: offline-learned models plus a simple online rule is the cheap end of the learning-cost spectrum and a natural baseline for a learned energy loop.

### 1.7 DORA: Optimizing Smartphone Energy Efficiency and Web Browser Performance under Interference
Davesh Shingari, Akhil Arunkumar, Benjamin Gaudette, Sarma Vrudhula, Carole-Jean Wu. ISPASS 2018. Verified (local extract). URL: https://doi.org/10.1109/ISPASS.2018.00015
Summary: Regression models predict page load time and power per frequency on a Nexus 5, including memory interference from co-scheduled apps. The governor picks fopt = fE when the PPW-optimal frequency fE already meets the deadline, else the minimum deadline-meeting frequency fD.
Numbers: model accuracy 97.5 percent (time) and 96 percent (power); average 16 percent and up to 35 percent higher energy efficiency than the interactive governor; deadline met 82 percent of the time.
Relevance: fopt = max(fE, fD) is exactly a performance floor wrapped around an energy optimum, the simplest possible two-loop rule, and it worked on a real phone without online learning.

### 1.8 Learning-Directed DVFS with Adjustable Performance for Single-Core and Multi-Core Embedded and Mobile Systems
Yen-Lin Chen, Ming-Feng Chang, Chao-Wei Yu, Xiu-Zhi Chen, Wen-Yew Liang. Sensors 2018. Verified (PMC page). URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6163884/
Summary: A counter-propagation network classifies task behaviour and predicts a voltage-frequency setting, with a user-set acceptable performance level (70 or 90 percent). Platforms are a PXA270 board and a Jetson TK1.
Numbers: versus ondemand, 4.88 to 42.63 percent energy reduction at 70 percent performance and 0.54 to 20.93 percent at 90 percent on the single core; 3.9 to 15.4 percent and 2.3 to 8.8 percent on the quad core; overhead 0.02 to 0.12 percent.
Relevance: the "acceptable performance loss" knob is a scalar exchange-rate lever set by the user, the same role as a battery tier.

### 1.9 PolyThrottle: Energy-efficient Neural Network Inference on Edge Devices
Minghao Yan, Hongyi Wang, Shivaram Venkataraman. arXiv 2310.19991, 2023 (v2 Jan 2024), preprint. Verified (arXiv abstract and ar5iv body). URL: https://arxiv.org/abs/2310.19991
Summary: Constrained Bayesian optimization searches CPU, GPU and memory frequency and batch size on Jetson TX2 and Orin to minimize energy per query subject to a latency SLO. The search space has 5005 (TX2) and 1820 (Orin) points; exhaustive search would take 14 and 5 hours.
Numbers: about 15 samples reach a near-optimal setting in a few minutes; up to 36 percent energy per query saved; memory frequency alone saves 12 to 25 percent; a lower GPU floor saves up to 47.6 percent for small models.
Relevance: the clearest published sample count for Bayesian tuning of hardware knobs under a latency constraint; a battery-tier lever could reweight the same acquisition function.

### 1.10 PaRMIS: Learning Pareto-Frontier Resource Management Policies for Heterogeneous SoCs
Aryan Deshwal, Syrine Belakaria, Ganapati Bhat, Janardhan Rao Doppa, Partha Pratim Pande. DAC 2021. Verified (arXiv abstract and ar5iv body). URL: https://arxiv.org/abs/2105.09282
Summary: An information-theoretic Bayesian search proposes parametric policies (a small MLP over counters that sets big and little core counts and frequencies) and evaluates them on an Odroid-XU3 (Exynos 5422) to maximize information about the true Pareto front of time, energy and PPW.
Numbers: at most 500 policy evaluations, usually converging within 300; 13 percent higher Pareto hypervolume than scalarized RL and 23 percent higher than imitation learning; 16 and 21 percent better PPW; global policies within 2 percent of application-specific ones; also compared with ondemand, interactive, performance, powersave and DyPO on 12 MiBench and CortexSuite benchmarks.
Relevance: it learns the whole Pareto front once, so a battery-tier lever can be applied afterwards as a choice of point on the front rather than as a retrained policy.

### 1.11 DyPO: Dynamic Pareto-Optimal Configuration Selection for Heterogeneous MpSoCs
Ujjwal Gupta, Chetan Arvind Patil, Ganapati Bhat, Prabhat Mishra, Umit Y. Ogras. ACM TECS 16(5s), 2017. Verified (ASU page, ACM listing). URLs: https://dl.acm.org/doi/10.1145/3126530 and https://asu.elsevierpure.com/en/publications/dypo-dynamic-pareto-optimal-configuration-selection-for-heterogen/
Summary: Offline characterization finds Pareto-optimal voltage, frequency and core configurations and trains classifiers from performance counters to those configurations; at runtime the classifier picks the configuration for the current phase.
Numbers: PPW improves 93 percent over interactive, 81 percent over ondemand and 6 percent over powersave across 18 applications.
Relevance: shows how far a stock governor sits from the Pareto front on a big.LITTLE SoC, which is the headroom any energy loop is competing for.

### 1.12 Temporal-encoding DRL DVFS and its multi-task successor (Jetson Nano)
Ti Zhou, Man Lin. "CPU frequency scheduling of real-time applications on embedded devices with temporal encoding-based deep reinforcement learning", Journal of Systems Architecture 2023. Verified (arXiv abstract). URLs: https://arxiv.org/abs/2309.03779 and https://www.sciencedirect.com/science/article/abs/pii/S1383762123001340
Xinyi Li, Ti Zhou, Haoyu Wang, Man Lin. "Energy-Efficient Computation with DVFS using Deep Reinforcement Learning for Multi-Task Systems in Edge Computing", arXiv 2409.19434, 2024, preprint. Verified (arXiv abstract). URL: https://arxiv.org/abs/2409.19434
Summary: An in-kernel quantized network sets CPU frequency for periodic tasks from temporally encoded utilization, with the deadline slack as the performance budget; the 2024 version handles 3, 5 and 8 concurrent tasks.
Numbers: 3 to 11 percent more energy saving than ondemand on MiBench and 5 to 14 percent on AudioReg and FaceReg (2023); 3 to 10 percent power saving over Linux governors (2024).
Relevance: single-digit gains over ondemand once a deadline is fixed, which bounds what an RL energy loop adds over a rule when the time budget is already known.

### 1.13 HiDVFS: Hierarchical Multi-Agent DVFS for Real-Time OpenMP DAG Workloads
Mohammad Pivezhandi, Abusayeed Saifullah, Ali Jannesari. arXiv 2601.06425, Jan 2026, preprint under review at IEEE TPDS. Verified (arXiv abstract). URL: https://arxiv.org/abs/2601.06425
Summary: Three agents (profiler for cores and frequency, thermal for grouping cores by temperature, priority for ordering tasks) act under a schedulability gate and a conformal shield that bounds predicted response times. Platforms are Jetson TX2, Orin NX and RubikPi.
Numbers: 2.83x speedup and 32.9 percent energy reduction versus GearDVFS; 15 to 18 percent energy reduction versus pinning the maximum frequency.
Relevance: it keeps the deadline agent and the thermal agent separate and resolves conflict by hard gates, a multi-agent version of two loops with a feasibility guard.

### 1.14 Autothrottle: A Practical Bi-Level Approach to Resource Management for SLO-Targeted Microservices
Zibo Wang, Pinghe Li, Chieh-Jan Mike Liang, Feng Wu, Francis Y. Yan. USENIX NSDI 2024. Verified (arXiv page). URLs: https://arxiv.org/abs/2212.12180 and https://www.usenix.org/conference/nsdi24/presentation/wang-zibo
Summary: Not mobile, but the cleanest published bi-level bandit design. An application-level contextual bandit sets a CPU throttle-ratio target from SLO feedback; per-service heuristic loops meet the target on their own. The throttle ratio is the only interface between the levels.
Numbers: up to 26.21 percent CPU savings over the best baseline and up to 93.84 percent over all baselines.
Relevance: the throttle-ratio interface is the analog of a scalar lever passed from a slow performance-budget loop to fast energy loops; contextual bandits are suitable because one step's action has little long-term effect.

### 1.15 Taming and Controlling Performance and Energy Trade-offs Automatically in Network Applications
Han Dong, Yara Awad, Sanjay Arora, Orran Krieger, Jonathan Appavoo. arXiv 2502.14987, Feb 2025, preprint. Verified (arXiv abstract). URL: https://arxiv.org/abs/2502.14987
Summary: Not mobile. A Bayesian optimizer tries a few settings of interrupt coalescing and DVFS on a live server and finds a configuration that meets a tail-latency SLA with much less energy.
Numbers: up to 60 percent energy reduction; specialized OSes reach more than 2x the energy efficiency of general-purpose OSes.
Relevance: an example of BO on a live system with a latency constraint, the pattern one would use to tune the energy loop's set points per battery tier.

Briefly noted for section 1: Orthrus (PPO-based governor, used as a baseline in MetaDVFS; original paper not retrieved, unverified); DVFO (Zhang et al., IEEE TMC 2024, arXiv 2306.01811, DRL over CPU, GPU and memory frequency plus offloading on three edge devices, 33 percent energy and 28.6 to 59.1 percent latency reduction, verified from arXiv; https://arxiv.org/abs/2306.01811); SparseDVFS (arXiv 2603.21908, Mar 2026, offline sparsity-to-frequency map, 78.17 percent energy-efficiency claim, verified abstract only; https://arxiv.org/abs/2603.21908); AGFT (arXiv 2508.01744, Aug 2025, online RL GPU frequency tuner for cloud LLM serving, 44.3 percent GPU energy saved with under 10 percent latency overhead, not mobile; https://arxiv.org/abs/2508.01744).

## 2. Control-theoretic coordination of multiple knobs and multiple loops

### 2.1 Odyssey: Energy-aware adaptation for mobile applications
Jason Flinn, Mahadev Satyanarayanan. SOSP 1999. Verified venue (Semantic Scholar API). URL: https://dl.acm.org/doi/10.1145/319151.319155
Summary: The user states a desired battery lifetime; the OS measures supply and demand and asks applications to lower data fidelity (video quality, speech recognition vocabulary, map detail) when the goal is at risk, and to raise it when there is slack. This is the first system in which a single scalar goal set by the user drives application-level quality.
Numbers (from search snippets; the PDF could not be parsed): energy reductions of 7 to 72 percent, mean 36 percent; 31 to 76 percent combined with hardware power management; battery-life goals that varied by as much as 30 percent were met.
Relevance: the battery-lifetime goal is the original battery-tier lever, and the OS loop that turns it into fidelity settings is the ancestor of the proposed energy loop.

### 2.2 GRACE-1: Cross-Layer Adaptation for Multimedia Quality and Battery Energy
Wanghong Yuan, Klara Nahrstedt, Sarita V. Adve, Douglas L. Jones, Robin H. Kravets. IEEE TMC 5(7), 2006. Verified venue (Semantic Scholar API); abstract from the search listing. URLs: https://doi.org/10.1109/TMC.2006.98 and http://rsim.cs.illinois.edu/Pubs/06MOBILE.pdf
Summary: Global adaptation runs when a task joins or leaves and decides each task's quality level, CPU allocation and the device's average power to maximize overall quality for a desired battery lifetime; speed-aware real-time scheduling and per-task internal adaptation then enforce those decisions at fine grain. Coordination is by hierarchy in time scale.
Numbers: not verified from a parsed page.
Relevance: the global-versus-internal split is a two-loop design where the slow loop owns the battery lever and the fast loops own the clock and the per-task quality.

### 2.3 POET: A Portable Approach to Minimizing Energy Under Soft Real-time Constraints
Connor Imes, David H. K. Kim, Martina Maggio, Henry Hoffmann. RTAS 2015. Verified (local poet.html with DOI; Lund publication listing). URLs: https://people.cs.uchicago.edu/~ckimes/poet/ and https://doi.org/10.1109/RTAS.2015.7108419
Summary: A single feedback controller computes the speedup needed to hit a latency target; an optimizer then converts that speedup into a schedule of at most two system configurations that minimizes energy, using a platform-supplied table of configurations. The controller is platform independent; only the table changes.
Numbers (Lund listing): 1.3 percent more energy than the dynamic optimal oracle on a mobile Haswell and 2.9 percent on an ARM big.LITTLE board.
Relevance: performance is the controlled variable and energy is the objective of the translator, so the two never compete for the actuator; the same structure fits a token-rate budget with an energy-minimal clock choice.

### 2.4 LEO: A Probabilistic Graphical Model-based Approach for Minimizing Energy Under Performance Constraints
Nikita Mishra, Huazhe Zhang, John D. Lafferty, Henry Hoffmann. ASPLOS 2015. Verified (Semantic Scholar API). URL: https://dl.acm.org/doi/10.1145/2775054.2694373
Summary: A hierarchical Bayesian model estimates an application's power and performance in every system configuration online from a few samples, borrowing strength from previously seen applications, so that Pareto-optimal trade-offs can be found without exhaustive profiling.
Numbers: compared with offline learning, online learning, a heuristic and the true optimum, LEO gives the most accurate estimates and near-optimal energy savings (abstract does not give percentages).
Relevance: LEO is the learner that feeds a POET-style controller, showing how to learn the energy loop's model from a handful of on-device samples.

### 2.5 JouleGuard: Energy Guarantees for Approximate Applications
Henry Hoffmann. SOSP 2015. Verified (local extract with DOI). URL: https://doi.org/10.1145/2815400.2815403
Summary: The energy-budget problem is split into a System Energy Optimizer (a multi-armed bandit with value-difference exploration that finds the most energy-efficient system configuration) and an Application Accuracy Optimizer (a PI controller on speedup that picks the most accurate application configuration delivering the required speedup). The two are coupled through the learner's model error: the controller pole is set to 1 minus 2 over delta when the learner's multiplicative error delta exceeds 2, so control slows while the bandit explores and speeds up once the model is accurate.
Numbers: on ODROID-XU3, a tablet and a server with eight applications, energy stays within a few percent of the goal and accuracy within a few percent of an oracle; uncoordinated application plus system adaptation oscillated (2080 qps, 147 W, 81 percent fewer results) while the coordinated version met the goal with 24 percent fewer results; runtime cost 249 microseconds per iteration on the mobile board.
Relevance: this is the closest published design to a two-loop scheduler with a bandit energy loop and a control performance loop, and the adaptive pole is a concrete decoupling mechanism; its energy-goal factor f is a scalar lever.

### 2.6 CoAdapt: Predictable Behavior for Accuracy-Aware Applications Running on Power-Aware Systems
Henry Hoffmann. ECRTS 2014. Verified venue (Semantic Scholar API); description from the IEEE listing snippet. URL: https://ieeexplore.ieee.org/document/6932604/
Summary: A runtime that coordinates accuracy-aware applications with power-aware systems, guaranteeing any two of performance, power and accuracy while optimizing the third, because decisions at one level otherwise counteract decisions at the other. Implemented on Linux/x86.
Numbers: not extracted.
Relevance: choosing which two dimensions are constrained and which one floats is a discrete form of the exchange-rate lever.

### 2.7 Automated Multi-objective Control for Self-Adaptive Software Design
Antonio Filieri, Henry Hoffmann, Martina Maggio. ESEC/FSE 2015. Verified (local extract, dblp DOI). URL: https://dl.acm.org/doi/10.1145/2786805.2786833
Summary: Goals are ranked; the lead dimension is controlled first with the smallest set of knobs that affect it (a deadbeat integral controller), and each subordinate dimension is controlled only with knobs that do not affect higher-ranked dimensions (Kvalid = (K minus Klead) intersect Ksub). The translator schedules at most two configurations and re-estimates the subordinate baseline from the lead decision each period.
Numbers: dual-problem pruning cut a radar case study from over 2 million configurations to under 2 thousand; case study 1 manages performance, security and energy of encrypted communication on a mobile device.
Relevance: cascade by priority with disjoint actuator sets is the textbook answer to two loops fighting over one knob; if the clock is the shared actuator, one loop must own it and the other must use a different knob (quality, output size).

### 2.8 MEANTIME: Achieving Both Minimal Energy and Timeliness with Approximate Computing
Anne Farrell, Henry Hoffmann. USENIX ATC 2016. Verified venue (USENIX listing, dblp). URL: https://www.usenix.org/conference/atc16/technical-sessions/presentation/farrell
Summary: Hard timing needs conservative resource allocation while energy minimization needs aggressive release of resources; MEANTIME resolves this by using approximation to absorb timing risk and resource allocation to cut energy. Evaluated with six applications on Linux/ARM.
Numbers: not extracted (PDF blocked).
Relevance: an explicit split where application-level approximation guards the deadline and the system-level knob minimizes energy, matching a design where output size guards the time budget and the clock is chosen for energy.

### 2.9 CALOREE: Learning Control for Predictable Latency and Low Energy
Nikita Mishra, Connor Imes, John Lafferty, Henry Hoffmann. ASPLOS 2018. Verified (Semantic Scholar API). URL: https://dl.acm.org/doi/10.1145/3173162.3173184
Summary: Resource allocation is split into learning how interacting resources affect speedup (a LEO-style learner, run off device) and controlling speedup to meet latency with minimal energy (a general controller on device whose parameters, including the pole, are set by the learner). Formal guarantees on meeting latency are retained.
Numbers: on ARM big.LITTLE in single- and multi-application scenarios, 60 percent fewer deadline misses and 13 percent less energy than the best prior learning and control solutions.
Relevance: the learner-to-controller interface (a small control model plus an uncertainty that sets the pole) is a proven way to let a learning energy loop feed a control performance loop without instability.

### 2.10 SOSA: Self-Optimizing Learning with Self-Adaptive Control for Hierarchical System-on-Chip Management
Bryan Donyanavard, Tiago Muck, Amir M. Rahmani, Nikil Dutt, Armin Sadighi, Florian Maurer, Andreas Herkersdorf. MICRO 2019. Verified (local extract with DOI). URL: https://dl.acm.org/doi/10.1145/3352460.3358312
Summary: A supervisory controller (Supervisory Control Theory) sets objective functions, references and constraints for per-core rule-based RL controllers (Learning Classifier Tables) that learn the model from scratch; on a thermal event the supervisor switches the low-level objective from "meet IPS, minimize power" to "cap power, approach IPS target".
Numbers: target performance met with under 1 percent error from an untrained model; under disturbance a classical SISO controller degraded up to 14 percent while the LCT settled below 5 percent then below 1 percent; hardware LCTs run every 5 ms at 9.62 percent FPGA slice overhead; supervisor every 100 ms in simulation.
Relevance: the supervisor is where a battery tier would live; it rewrites the reward and constraint of the fast loops instead of touching the actuator.

### 2.11 SPECTR: Formal Supervisory Control and Coordination for Many-core Systems Resource Management
Amir M. Rahmani, Bryan Donyanavard, Tiago Muck, Kasra Moazzemi, Axel Jantsch, Onur Mutlu, Nikil Dutt. ASPLOS 2018. Verified venue (ACM SIGPLAN Notices listing); details from search snippets. URLs: https://dl.acm.org/doi/10.1145/3296957.3173199 and https://research.ece.cmu.edu/safari/pubs/SPECTR-formal-supervisory-control-for-many-core-resource-management_asplos18.pdf
Summary: A synthesized supervisory controller coordinates classical low-level controllers on an Exynos big.LITTLE platform, using gain scheduling and dynamic priorities so each low-level controller keeps autonomy while system goals change at runtime.
Numbers: not extracted (PDF did not parse).
Relevance: a formal way to change which loop has priority as conditions change, the same job as a battery-tier lever.

### 2.12 ALERT: Accurate Learning for Energy and Timeliness
Chengcheng Wan, Muhammad Santriaji, Eri Rogers, Henry Hoffmann, Michael Maire, Shan Lu. USENIX ATC 2020. Verified (arXiv page). URLs: https://arxiv.org/abs/1911.00119 and https://www.usenix.org/conference/atc20/presentation/wan
Summary: For interactive DNN inference, ALERT jointly picks the DNN variant (latency versus accuracy) and the system configuration (latency versus energy), using a probabilistic estimate of a global slowdown factor to track environment volatility so both levels adapt together.
Numbers: over 13 percent energy saving and 27 percent lower error than application-only or system-only adaptation; within 3 percent energy and 2 percent error of an oracle; CPU and GPU platforms, image and speech tasks.
Relevance: ALERT is a two-knob (model, clock) controller under a latency budget with an energy objective, the closest DNN-era match to the proposed design.

### 2.13 WASL: Harmonizing Uncoordinated Adaptive Modules in Multi-Tenant Cloud Systems
Ahsan Pervaiz, Anwesha Das, Vedant Kodagi, Muhammad Santriaji, Henry Hoffmann. ICPE 2026. Verified (Semantic Scholar API). URL: https://doi.org/10.1145/3777884.3797009
Summary: Each adaptive module watches the gap between the effect it expected and the effect it observed; a large gap signals interference from another module, and the module slows its own adaptation rate. No shared state or protocol is needed.
Numbers: tail latency reduced by up to 84 percent versus uncoordinated operation, comparable to centralized coordination, across five TailBench applications and three adaptation strategies.
Relevance: it is the 2026 statement of the JouleGuard idea that model error is the right coupling signal between loops, and it works when the loops cannot be redesigned together.

### 2.14 E4: Energy-Efficient DNN Inference for Edge Video Analytics via Early-Exit and DVFS
Ziyang Zhang, Yang Zhao, Ming-Ching Chang, Changyao Lin, Jie Liu. AAAI 2025. Verified (arXiv abstract and HTML body). URL: https://arxiv.org/abs/2503.04865
Summary: An attention-based cascade picks the early-exit point per frame first; a just-in-time profiler then uses coordinate descent to set CPU and GPU frequency for the layers before that exit. Ordering removes conflict: the quality decision precedes the clock decision. Evaluated on five Jetson boards (Nano, TX2, Xavier NX, Orin Nano, AGX Orin) with EfficientNet-B0 and MobileNet-v2 against EENet, zTT and Ring-DVFS.
Numbers: up to 2.8x speedup and 26 percent average energy saving; 20 to 37 percent energy for EfficientNet-B0 and 18 to 30 percent for MobileNet-v2.
Relevance: a worked example of "application knob first, system knob second" with a coordinate-descent energy loop, and it beats zTT with no RL.

Briefly noted for section 2: Bard (Imes and Hoffmann, SAMOS 2016, unified handling of soft timing and power constraints; listed on the POET page, not fetched); Adapt&Cap (IEEE Design and Test 2016, listed on Hoffmann's page, not fetched); Automated Control of Multiple Software Goals using Multiple Actuators (FSE 2017, listed on Hoffmann's page, not fetched).

## 3. Energy-delay product, Pareto fronts, race-to-idle versus pace-to-idle, linear versus cubic

### 3.1 Racing and Pacing to Idle: Theoretical and Empirical Analysis of Energy Optimization Heuristics
David H. K. Kim, Connor Imes, Henry Hoffmann. CPSNA 2015 (extends the HotPower 2013 paper, DOI 10.1145/2525526.2525854). Verified (local extract; dblp). URL: https://people.cs.uchicago.edu/~hankhoffmann/kim-cpsna2015.pdf
Summary: Minimizing energy under a deadline is a linear program whose dual is two-dimensional, so the optimal schedule uses at most two configurations, and race-to-idle, pace-to-idle and no-idle are special cases. An idling heuristic is optimal only if the power function is a straight line from the idle point, so race-to-idle is near optimal only on machines with a nearly linear power curve.
Numbers: on the Exynos 5 big.LITTLE mobile board, normalized energy (1 = optimal) was race 3.34, pace 1.12, no-idle 1.11, DVFS plus DPM 2.21; pace beats race by up to 20 percent on x86 and 3x on ARM; pace is never more than 12 percent above optimal; the mobile board idles at 0.12 W versus 0.17 W lowest active power.
Relevance: the energy loop should hold a pace configuration (highest performance per watt that still meets the budget) rather than race, and on big.LITTLE the biggest lever is which core type runs, not the clock alone.

### 3.2 Dynamic Voltage and Frequency Scaling: The Laws of Diminishing Returns
Etienne Le Sueur, Gernot Heiser. HotPower 2010. Verified (local extract; ACM listing). URLs: https://www.usenix.org/legacy/events/hotpower/tech/full_papers/LeSueur.pdf and https://dl.acm.org/doi/10.5555/1924920.1924921
Summary: Across three AMD Opteron generations (130 nm to 45 nm), DVFS saved up to 34 percent on the 2003 platform but increased energy on the 2009 platform even for the memory-bound mcf, because the voltage window shrank toward the 0.7 V threshold, leakage grew, memory got faster and idle states improved. When idle energy is padded in, the lowest frequency wins, which shows the metric matters.
Numbers: EDP on the 2009 platform was minimized at the maximum 2.7 GHz for one or two mcf instances and at 0.8 GHz only with four memory-bound instances.
Relevance: this is the published basis for expecting near-linear power in frequency once voltage is pinned; a clock cap then buys thermal headroom and lower peak power, not lower energy per token, unless the work is memory-bound.

### 3.3 Slow Down or Sleep, That Is the Question
Etienne Le Sueur, Gernot Heiser. USENIX ATC 2011. Verified venue (USENIX and dblp listings); PDF mirror fetched. URLs: https://www.usenix.org/conference/usenixatc11/slow-down-or-sleep-question and https://trustworthy.systems/publications/papers/LeSueur_Heiser_11.pdf
Summary: Compares slowing down with DVFS against completing work fast and using deep idle states on Core i7, Atom and OMAP platforms with MPEG playback, Apache and SPECjbb. Finds that on modern parts with good idle states, racing to a deep sleep is often more energy efficient than sustained slow-down, especially for interactive work.
Numbers: not extracted from the parsed text.
Relevance: the idle-state depth available between requests decides whether the energy loop should pace or race; on a phone with an inference session that idles between requests, the answer depends on measured idle power.

### 3.4 Critical Power Slope: Understanding the Runtime Effects of Frequency Scaling
Akihiko Miyoshi, Charles Lefurgy, Eric Van Hensbergen, Ram Rajamony, Raj Rajkumar. ICS 2002. Venue confirmed from the ACM listing and from the Le Sueur 2010 text; findings from search snippets. URL: https://dl.acm.org/doi/10.1145/514191.514200
Summary: Defines the critical power slope, the slope of power versus performance above which a lower operating point stops being energy efficient once idle power is counted. On a Pentium system the highest frequency was always most efficient; on a PowerPC system the lowest was.
Numbers: three systems; results consistent across register, cache, memory, disk and web workloads.
Relevance: the sign of the energy loop's gradient with respect to the clock is platform specific and must be measured, which is why a table-driven translator (POET) or a learned front (PaRMIS) is safer than a fixed cube-law assumption.

### 3.5 The Energy/Frequency Convexity Rule: Modeling and Experimental Validation on Mobile Devices
Karel De Vogeleer, Gerard Memmi, Pierre Jouvelot, Fabien Coelho. arXiv 1401.4655, 2014. Verified (arXiv abstract). URL: https://arxiv.org/abs/1401.4655
Summary: One week of high-resolution power traces on a smartphone show that energy per input element versus CPU frequency is convex with a clear interior minimum, so both the lowest and the highest clock waste energy.
Numbers: minimum inside the 0.2 to 1.6 GHz window.
Relevance: gives the energy loop a unimodal objective in the clock, which makes a small bandit or a one-dimensional search sufficient.

### 3.6 Mobile Multicores: Use Them or Waste Them
Aaron Carroll, Gernot Heiser. HotPower 2013. Verified venue (dblp, DOI 10.1145/2525526.2525850); content from the medusa governor repository. URLs: https://dl.acm.org/doi/10.1145/2525526.2525850 and https://github.com/xaaronc/medusa
Summary: Measures smartphone SoCs (Snapdragon APQ8064, Exynos in the Galaxy S4 family) and argues that spreading work over more cores at lower frequency is often more energy efficient than fewer cores at high frequency, with a per-platform frequency threshold below which extra cores are not worth it (the medusa governor's fthresh, 0 kHz on Cortex parts and about 800 MHz on Snapdragon 600).
Numbers: not verified from the paper text.
Relevance: core count is a second actuator for the energy loop that does not conflict with a clock cap owned by the performance loop.

### 3.7 FUSE: Dissecting the Impact of Mobile DVFS Governors on LLM Inference Performance and Energy Efficiency
Zongpu Zhang, Pranab Dash, Y. Charlie Hu, Qiang Xu, Jian Li, Haibing Guan. arXiv 2507.02135, Jul 2025, preprint. Verified (arXiv abstract and HTML body). URL: https://arxiv.org/abs/2507.02135
Summary: On Pixel 7 and 7 Pro (Tensor G2, Mali-G710) with llama.cpp and TinyLlama-1.1B, StableLM-Zephyr-3B, Llama-2-7B and DeepSeek-R1-Distill-Qwen-1.5B, the independent CPU, GPU and memory governors leave large latency on the table. FUSE profiles the CPU, GPU and memory frequency triple at install time for five prefill-length ranges and pins a fixed triple per phase; there is no online learning.
Numbers: default governors give up to 40.4 percent longer prefill and 31.8 percent longer decode than the best fixed triple at equal energy; FUSE cuts TTFT 7.0 to 16.9 percent and TPOT 25.4 to 36.8 percent at equal energy per token.
Relevance: memory frequency is a first-order knob for decode, and a fixed profiled triple is the baseline any learned energy loop must beat on LLM workloads.

### 3.8 Joint Optimization of Memory and Computing Frequency for Energy-Efficient DNN Inference
Yunchu Han, Zhaojun Nan, Sheng Zhou, Zhisheng Niu. arXiv 2608.13863, Aug 2026, preprint. Verified (arXiv abstract). URL: https://arxiv.org/abs/2608.13863
Summary: Formulates energy minimization under a deadline over memory and compute frequency and gives a near-optimal closed form via convex optimization for local inference, plus an optimal transmit-power solution for edge offloading.
Numbers: within 2.5 percent of optimal under strict deadlines; up to 10.4 percent lower device energy than alternatives.
Relevance: a closed-form energy loop for two clocks under one deadline, which could replace search inside the performance budget.

Briefly noted for section 3: DORA's fopt rule (section 1.7) and PolyThrottle's memory-frequency result (section 1.9) are the phone and Jetson evidence that the energy-optimal point is interior and workload dependent. On linear versus cubic scaling: GearDVFS assumes E proportional to f cubed, JouleGuard initializes its bandit with cubic power in clock, and FUSE describes power as scaling quadratically; the measurement papers (Le Sueur 2010, Miyoshi 2002, De Vogeleer 2014) show the actual exponent is platform and voltage-window dependent and can be close to linear once voltage is pinned. No 2024 to 2026 paper was found that reports a measured power-versus-frequency exponent on a current Snapdragon with voltage pinned; that measurement remains project specific.

## 4. Battery-state-aware and thermal-aware quality scaling of on-device ML, including LLMs

### 4.1 NestDNN: Resource-Aware Multi-Tenant On-Device Deep Learning for Continuous Mobile Vision
Biyi Fang, Xiao Zeng, Mi Zhang. MobiCom 2018. Verified (arXiv page). URLs: https://arxiv.org/abs/1810.10090 and https://dl.acm.org/doi/10.1145/3241539.3241559
Summary: Each model is trained as a nested multi-capacity model; a runtime scheduler picks a capacity per application to fit the currently available resources and to maximize joint accuracy and frame rate.
Numbers: 4.2 percent higher accuracy, 2.0x frame rate, 1.7x lower energy versus fixed models.
Relevance: an elastic model is the application-level actuator that lets the performance loop meet a time budget without touching the clock.

### 4.2 Optimizing Deep Learning Inference on Embedded Systems Through Adaptive Model Selection
Vicent Sanz Marco, Ben Taylor, Zheng Wang, Yehia Elkhatib. ACM TECS 2020. Verified (arXiv page). URLs: https://arxiv.org/abs/1911.04946 and https://dl.acm.org/doi/abs/10.1145/3371154
Summary: An offline-trained predictor chooses which pre-trained DNN to run for each input under a target accuracy and inference time on a Jetson TX2.
Numbers: 1.8x faster with 7.52 percent higher accuracy than the most capable single model for image classification; 1.34x faster for machine translation.
Relevance: per-input model choice is a cheap quality knob whose set point can be moved by a battery tier.

### 4.3 Budget RNNs: Multi-Capacity Neural Networks to Improve In-Sensor Inference Under Energy Budgets
Tejas Kannan, Henry Hoffmann. RTAS 2021, Outstanding Paper. Verified venue (dblp); mechanism from the author repository. URLs: https://ieeexplore.ieee.org/document/9470487/ and https://github.com/tejaskannan/budget-rnn
Summary: A single RNN with several capacity levels changes its subsampling and halting behaviour at runtime; thresholds on the halting signal are fitted per discrete energy budget level so the network spends exactly the allowed energy. Hardware is a TI MSP430 FR5994 with a BLE radio and supercapacitors.
Numbers: not extracted from a parsed page.
Relevance: the discrete budget level is a direct precedent for a battery-tier lever that selects a quality-versus-energy operating point.

### 4.4 SLEXNet: Adaptive Inference Using Slimmable Early Exit Neural Networks
Basar Kutukcu, Sabur Baidya, Sujit Dey. ACM TECS 2024. Verified (Semantic Scholar API). URL: https://dl.acm.org/doi/10.1145/3689632
Summary: Combines dynamic width and dynamic depth in one network and adds a runtime scheduler that estimates inference time and power of each variant to meet varying time and power conditions on a Jetson Orin.
Numbers: outperforms depth-only and width-only baselines across time and power constraints (no headline percentage in the abstract).
Relevance: a two-dimensional quality knob with a runtime cost model is what an energy loop needs to trade quality for energy inside a fixed time budget.

### 4.5 EnerInfer: Energy-Aware On-Device LLM Inference
Bohua Zou, Nian Liu, Binqi Sun, Matteo Mascherin, Debayan Roy, Yutao Liu, Yu Peng, Ning Jia, Haibo Chen. arXiv 2606.23001, Jun 2026, preprint. Verified (arXiv abstract). URL: https://arxiv.org/abs/2606.23001
Summary: Exploits configuration slack by lowering NPU and DDR frequency while holding a QoE target; a model-structure-aware predictor estimates throughput and power for unseen LLMs without per-model profiling, ranking-driven online feedback corrects it under interference, and a limited-horizon thermal predictor switches between an energy-optimized mode and a thermally constrained mode.
Numbers: up to 65 percent energy-efficiency gain on phones, 12 percent on a laptop, 24 percent on a dev board, with no QoE violations.
Relevance: the closest LLM-era match to a two-mode design; its mode switch is driven by shell temperature, not battery level, and the exchange rate between modes is fixed.

### 4.6 DVFSLM: Energy-Efficient Small Language Model Inference for Mobile Agents via DVFS
Jiesong Chen, Lixiang Han, Jiani Cao, Zhenjiang Li. MobiSys 2026 Workshops. Verified (Semantic Scholar API). URL: https://doi.org/10.1145/3812836.3814753
Summary: Workload-aware estimators map core matrix-operation counts and hardware metadata to power and latency per processor frequency; a runtime governor then minimizes energy subject to a token-generation deadline across processors.
Numbers: reported as outperforming state-of-the-art methods on a rich set of SLMs (abstract gives no percentage).
Relevance: a token deadline is the performance loop and the governor is the energy loop; the design matches the target architecture with an analytic rather than learned energy model.

### 4.7 PELM: Power Efficient On-Device LLM Inference with Speculative Decoding and Dynamic Voltage Frequency Scaling
Weisi Yang, Stephen Xia. 2026 ACM/IEEE International Conference on Embedded AI and Sensing Systems (ACM DL listing; Semantic Scholar files it under the SenSys family). Verified (Semantic Scholar API). URL: https://dl.acm.org/doi/10.1145/3774906.3802783
Summary: Combines DVFS with speculative decoding and a variable verification depth, on the observation that not all tokens need full-depth inference.
Numbers: up to 23.1 percent speedup and 52.4 percent energy reduction with comparable output quality across platforms and datasets.
Relevance: verification depth is an application-level knob that trades energy for quality independently of the clock, so it can be given to the energy loop while the clock stays with the performance loop.

### 4.8 MNN-AECS: Energy Optimization for LLM Decoding on Mobile Devices via Adaptive Core Selection
Zhengxiang Huang, Chaoyue Niu, Zhaode Wang, Jiarui Xue, Hanming Zhang, Yugang Wang, Zewei Xin, Xiaotang Jiang, Chengfei Lv, Fan Wu, Guihai Chen. arXiv 2506.19884, Jun 2025, preprint. Verified (arXiv abstract). URL: https://arxiv.org/abs/2506.19884
Summary: For the memory-bound decode phase, an adaptive heuristic moves decoding onto low-power cores without root access or OS changes, integrated into the MNN engine; tested on 5 Android and 2 iOS devices with 5 LLMs.
Numbers: 23 percent energy reduction versus MNN with no speed loss; 39 to 78 percent versus llama.cpp, ExecuTorch, mllm and MediaPipe with 12 to 363 percent speedups.
Relevance: core selection is a user-space energy knob that a phone app can pull without privileges, unlike the clock.

### 4.9 Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs
Sayed Pedram Haeri Boroujeni, Niloufar Mehrabi, Patrick Woods, Gabriel Hillesheim, Abolfazl Razi. arXiv 2604.04722, Apr 2026; the abstract page lists CVPR 2026. Verified (arXiv abstract). URL: https://arxiv.org/abs/2604.04722
Summary: A learned controller assigns 2, 4, 8 bit or FP16 precision per token during decoding from token frequency, quality score, attention variance and entropy, in the spirit of Huffman coding.
Numbers: 17.75 percent lower decode latency than static quantization on SmolLM-360M; 7.60 points higher on HellaSwag; within 0.30 points of FP16; SmolLM 135M, 360M and 1.7B.
Relevance: KV precision is an adaptive KV-budget knob that a battery tier could bias, but the paper adapts to token importance, not to battery or temperature.

### 4.10 An Empirical Study of LLM Reasoning Ability Under Strict Output Length Constraint
Yi Sun, Han Wang, Jiaqiang Li, Jiacheng Liu, Xiangyu Li, Hao Wen, Yizhen Yuan, Huiwen Zheng, Yan Liang, Yuanchun Li, Yunxin Liu. arXiv 2504.14350, Apr 2025, preprint. Verified (arXiv abstract). URL: https://arxiv.org/abs/2504.14350
Summary: Evaluates 30 models under token budgets and ties the budgets to on-device latency; the best model size and prompt style change with the budget.
Numbers: 30 models; no single headline percentage in the abstract.
Relevance: an output-length cap changes which model is best, so a battery-tier lever that caps output length should also be allowed to switch model size.

### 4.11 The Diminishing Returns of Early-Exit Decoding in Modern LLMs
Rui Wei, Rui Du, Hanfei Yu, Devesh Tiwari, Jian Li, Zhaozhuo Xu, Hao Wang. arXiv 2603.23701, Mar 2026, preprint. Verified (arXiv abstract). URL: https://arxiv.org/abs/2603.23701
Summary: Newer pretraining recipes reduce layer redundancy, so early-exit benefit shrinks across model generations; base and larger dense models keep the most, fine-tuned and non-dense models the least.
Numbers: qualitative trend across generations.
Relevance: early exit is a weakening application-level energy knob for current LLMs; output length, KV precision and verification depth are safer choices.

### 4.12 LLM Inference at the Edge: Mobile, NPU and GPU Performance Efficiency Trade-offs Under Sustained Load
Pranay Tummalapalli, Sahil Arayakandy, Ritam Pal, Kautuk Kundan. arXiv 2603.23640, Mar 2026, preprint. Verified (arXiv abstract). URL: https://arxiv.org/abs/2603.23640
Summary: Qwen2.5-1.5B 4-bit under 20 warm iterations on iPhone 16 Pro, Galaxy S24 Ultra, RTX 4050 and a Hailo-10H NPU; thermal management, not peak compute, is the binding constraint on phones.
Numbers: iPhone 16 Pro loses about 50 percent throughput within two iterations; the S24 Ultra hits an OS GPU frequency floor that halts inference; RTX 4050 sustains 131.7 tok/s at 34.1 W; Hailo-10H 6.9 tok/s under 2 W.
Relevance: motivates a performance loop that budgets time under a sustained thermal state rather than at peak, and a thermal tier as a second lever beside battery.

Briefly noted for section 4: ALERT (section 2.12) and E4 (section 2.14) belong here as well. Also verified from arXiv abstracts: "Energy-Efficient GPU DVFS for Fine-Tuning of SLMs on Resource-constrained Embedded Devices" (Park et al., arXiv 2607.05933, Jul 2026, Jetson AGX Orin, 13.11 percent average and up to 26.73 percent energy saving over MAXN; https://arxiv.org/abs/2607.05933) and PowerLens (arXiv 2603.19584, Mar 2026, LLM agents managing 18 Android settings, 38.8 percent energy saving over stock Android, preferences learned in 3 to 5 days; https://arxiv.org/abs/2603.19584).

## 5. Do learned policies beat simple rules on real phones, and what does learning cost (2024 to 2026)

| Work | Device | Learned online? | Learning or profiling cost | Baseline beaten | Gain |
|---|---|---|---|---|---|
| MetaDVFS (arXiv 2025) | Pixel 3, 4, 6, 8, 9 | Yes, DQN with metadata transfer | 3.5 plus or minus 1.1 min per new device-app pair | schedutil, zTT, GearDVFS, Orthrus | up to 17 percent PPW, 26 percent QoE |
| MobiRL (TACO 2024) | commercial smartphones | Yes, RL | not stated | commercial scheduler, Q-learning | 42.8 percent less power, 4.1 percent fewer frame drops |
| GearDVFS (MobiCom 2023) | Redmi Note 9 (phone), Jetsons | Yes, DQN, optional fine-tune | 35 min training, 500 to 650 s adaptation | schedutil plus simple_ondemand, zTT | 6 to 9 percent PPW on the phone, up to 16.92 percent after six traces |
| zTT (MobiSys 2021) | Pixel 3a | Yes, DQN | not stated in accessible sources | default DVFS | 23.9 percent less power at target fps in a hot room |
| FUSE (arXiv 2025) | Pixel 7, 7 Pro | No, install-time profile | one profiling pass per prefill-length range | CPU, GPU, memory governors | TTFT 7.0 to 16.9 percent, TPOT 25.4 to 36.8 percent at equal energy |
| EnerInfer (arXiv 2026) | phones, laptop, board | Online ranking feedback on a structural predictor | no per-model profiling | default configuration | up to 65 percent energy efficiency on phones |
| MNN-AECS (arXiv 2025) | 5 Android, 2 iOS | Adaptive heuristic | none | MNN and four engines | 23 percent, 39 to 78 percent |
| DVFSLM (MobiSys 2026 workshop) | mobile SoC | Analytic estimators | none | prior DVFS schemes | not quantified in abstract |
| PowerLens (arXiv 2026) | rooted Android | LLM agent with preference memory | 3 to 5 days of implicit feedback | stock Android | 38.8 percent energy |
| PolyThrottle (2023) | Jetson TX2, Orin | Bayesian optimization | about 15 samples, a few minutes | grid search, random search | up to 36 percent energy |
| PaRMIS (DAC 2021) | Odroid-XU3 | Bayesian policy search | 300 to 500 evaluations | RL, imitation learning, governors, DyPO | 13 to 23 percent hypervolume |
| HiDVFS (arXiv 2026) | Jetson TX2, Orin NX, RubikPi | Yes, multi-agent RL | not stated | GearDVFS, max-frequency pinning | 32.9 percent energy vs GearDVFS, 15 to 18 percent vs max pin |
| E4 (AAAI 2025) | five Jetsons | No, cascade plus coordinate descent | just-in-time profiling | EENet, zTT, Ring-DVFS | 26 percent energy, up to 2.8x |

Observations. (1) On phones the learned governors report gains against stock governors and earlier learned governors, never against a tuned static per-application policy; MetaDVFS says so explicitly. (2) Where a static profiled policy was measured (FUSE), it beat the governors by as much as the learned governors do, at zero online cost. (3) On Jetson-class boards, rule-based or search-based methods (E4, HiDVFS's gated design, PolyThrottle) beat zTT and GearDVFS. (4) The only learning costs reported on phones are minutes (MetaDVFS) to tens of minutes (GearDVFS) of interaction plus a one-time desktop training run; no paper reports the energy spent on learning. (5) Learned policies pay off when the workload mix is unknown; LLM decode on a known engine is a known workload, which favours a profiled or analytic energy loop with a small online correction (EnerInfer's ranking feedback, JouleGuard's bandit) over a full RL governor.

## 6. Gaps found

- No paper was found that tunes Android governor parameters (schedutil rate limits, EAS margins) with Bayesian optimization on a phone. The nearest are PolyThrottle (Jetson, CBO over clocks), PaRMIS (Odroid, Bayesian policy search) and Dong et al. 2025 (servers).
- No contextual-bandit DVFS or thermal policy on a phone was found. JouleGuard's bandit runs on an ODROID; Autothrottle's contextual bandit runs in a datacenter.
- No 2024 to 2026 LLM paper uses battery level or battery tier as an explicit input to output length, KV budget or quantization. Temperature (EnerInfer) and token deadlines (DVFSLM, FUSE) are used instead. Odyssey (1999), GRACE-1 (2006) and Budget RNNs (2021) are the precedents for a battery-driven scalar lever.
- No 2024 to 2026 paper reports a measured power-versus-frequency exponent on a current Snapdragon with the voltage floor reached; the claim that clock reduction saves energy still rests on the 2002 to 2014 measurement papers.
- zTT's training length and MEANTIME's, SPECTR's, GRACE-1's and Odyssey's headline numbers could not be read from a fetched page (ACM, IEEE and USENIX PDF endpoints returned 403 or binary content); their venues are verified, their numbers are marked accordingly above.

## 7. What this implies for a two-loop design with a battery-tier lever

- Give the clock to one loop only. Every stable design in section 2 either assigns disjoint actuators (FSE 2015, MEANTIME, E4) or passes a scalar (speedup in POET and JouleGuard, throttle ratio in Autothrottle, references in SOSA) from the slow loop to the fast loop. A performance loop that owns the time budget through output length or model choice, and an energy loop that owns clock, core selection and memory frequency, matches the successful pattern.
- Couple through model error, not through the actuator. JouleGuard's adaptive pole and WASL's rate limiting are the two published ways to keep a learning loop from destabilizing a control loop; both key on the gap between predicted and observed behaviour.
- Pace, do not race, but measure idle power first (Kim et al. 2015; Le Sueur and Heiser 2011). On big.LITTLE the core type dominates the clock.
- Expect linear-ish power in clock near the voltage floor (Le Sueur 2010), so treat a clock cap as a thermal and peak-power lever and let memory frequency and core selection carry the energy savings for decode (FUSE, PolyThrottle, MNN-AECS).
- Learning cost that is defensible on a phone is minutes (MetaDVFS) or about 15 samples (PolyThrottle); anything longer should be done offline and shipped as a table, with a bandit for online correction.
- The battery tier maps naturally onto Odyssey's lifetime goal, JouleGuard's f, CoAdapt's choice of fixed dimensions and Budget RNNs' budget levels: it should change the energy loop's objective weight or set point, not its actuator.

## Appendix A. Raw notes, batch 1 (local extracts and first fetches)

- JouleGuard (Hoffmann, SOSP 2015). Local extract jouleguard.txt. DOI in file: 10.1145/2815400.2815403. Two sub-problems: System Energy Optimizer (bandit over system configs, VDBE exploration, reward = energy efficiency) and Application Accuracy Optimizer (PI controller on speedup). Adaptive pole: pole(t) = 1 - 2/delta(t) when model error delta > 2, else 0. Controller slows down when the learner is uncertain. Platforms: ODROID-XU3 (mobile), Sony Vaio tablet, dual Xeon server. Eight approximate apps. Energy within a few percent of goal, accuracy within a few percent of oracle. Runtime overhead 249 us per iteration on the mobile board.
- Racing and Pacing to Idle (Kim, Imes, Hoffmann, CPSNA 2015). Local extract racepace.txt. URL from search: https://people.cs.uchicago.edu/~hankhoffmann/kim-cpsna2015.pdf. Optimal allocation uses at most two configurations. Pace-to-idle always beats race-to-idle. Measured normalized energy (1 = optimal) on the Exynos 5 big.LITTLE mobile board: race 3.34, pace 1.12, no-idle 1.11, DVFS+DPM 2.21. On x86 pace saves up to 20 percent over race. Pace-to-idle is never more than 12 percent worse than optimal.
- GearDVFS (Lin, Wang, Li, Pu, MobiCom 2023). Local extract geardvfs.txt. DOI 10.1145/3570361.3592524. DQN with learned meta-states (LSTM autoencoder over CPU active/stalled utilization, GPU utilization, frequencies, temperatures). Branch Q-network reduces action space from product (1872 on Jetson TX2) to sum (37). Reward: utilization near 80 percent target plus the zTT thermal penalty (50 C threshold). Devices: Jetson NX, Nano, Odroid-XU3, Raspberry Pi 4B, Redmi Note 9. Up to 23.9 percent and 26.9 percent energy-efficiency (PPW) improvement over stock governor and zTT respectively; up to 16.92 percent with low concurrency. Governs every 100 ms.
- Le Sueur and Heiser, "Dynamic Voltage and Frequency Scaling: The Laws of Diminishing Returns" (HotPower 2010). Local extract lesueur.txt. ACM DL: https://dl.acm.org/doi/10.5555/1924920.1924921. Three AMD Opteron generations. DVFS saved up to 34 percent on the 2003 platform, but increased energy on the 2009 platform even for memory-bound mcf. EDP minimized at the highest frequency except for four memory-bound instances at once. Reasons: smaller voltage range, leakage, faster memory, better idle states.
- SOSA (Donyanavard et al., MICRO 2019). Local extract sosa.txt. DOI 10.1145/3352460.3358312. Hierarchical: supervisory controller (Supervisory Control Theory) sets references and priorities for low-level rule-based RL controllers (Learning Classifier Tables in hardware). Meets target performance with less than 1 percent error starting from an untrained model. Explicit discussion of goal switching (performance to power on a thermal event) and of CALOREE needing off-device learning.
- DORA (Shingari, Arunkumar, Gaudette, Vrudhula, Wu, ISPASS 2018). Local extract dora.txt. DOI 10.1109/ISPASS.2018.00015. Google Nexus 5. Regression models of page load time (97.5 percent accuracy) and power (96 percent). Rule: fopt = fE if fD <= fE else fD, where fE maximizes PPW and fD is the minimum deadline-meeting frequency. Average 16 percent (up to 35 percent) energy-efficiency gain over the interactive governor; deadline met 82 percent of the time.
- Automated Multi-objective Control for Self-Adaptive Software Design (Filieri, Hoffmann, Maggio, FSE 2015). Local extract fse2015.txt. Cascade control by goal rank. Lead dimension controlled with the fewest knobs; subordinate dimension uses only knobs that do not affect the lead dimension (Kvalid = (K minus Klead) intersect Ksub). Optimal schedule uses at most two configurations. Case study on a mobile device (performance, security, energy for encrypted communication).
- POET (Imes, Kim, Maggio, Hoffmann, RTAS 2015). Local poet.html: https://people.cs.uchicago.edu/~ckimes/poet/ with DOI 10.1109/RTAS.2015.7108419. Search hit (Lund publications) reports 1.3 percent more energy than the dynamic optimal oracle on Haswell and 2.9 percent on ARM big.LITTLE.
- Hoffmann publication page (local hoffmann.html): confirms CoAdapt (ECRTS 2014), CALOREE (ASPLOS 2018, preprint caloree.pdf), FSE 2017 multi-actuator control, MEANTIME (ATC 2016), ALERT (ATC 2020), Budget RNNs (RTAS 2021), Gambler (EWSN 2025), WASL (ICPE 2026), Orthogonalized SGD for anytime networks (ICML 2020).
- zTT (MobiSys 2021). GitHub README fetched: https://github.com/ztt-21/zTT/blob/main/README.md. DQN over CPU and GPU frequency, state includes clocks, power, temperature, fps. Pixel 3a and Jetson TX2. 23.9 percent less average power while holding target fps in a hot environment. ACM page returned 403.
- MetaDVFS (Yan et al., arXiv 2509.22707, 23 Sep 2025, preprint). Fetched abs. Multi-task RL with device and application metadata. Five Google Pixel devices. Up to 17 percent PPW, up to 26 percent QoE, 70.8 percent faster adaptation than standalone training, 5.8 to 27.6 percent better than device-application-specific training.
- FUSE / Dissecting the Impact of Mobile DVFS Governors on LLM Inference (Zhang, Dash, Hu, Xu, Li, Guan, arXiv 2507.02135, 2 Jul 2025). Fetched abs. Default independent CPU, GPU and memory governors give up to 40.4 percent longer prefill and decode latency than the best frequency combination at the same energy. FUSE cuts TTFT by 7.0 to 16.9 percent and TPOT by 25.4 to 36.8 percent at equal energy per token.
- EnerInfer (Zou et al., arXiv 2606.23001, 22 Jun 2026, preprint). Fetched abs. Lowers NPU and DDR frequency under a QoE target, model-structure-aware prediction plus ranking-driven online feedback, limited-horizon thermal prediction. Up to 65 percent energy-efficiency gain on phones, 12 percent laptop, 24 percent dev board, no QoE violations.
- MNN-AECS (Huang et al., arXiv 2506.19884, 24 Jun 2025). Fetched abs. Adaptive core selection for decode. 5 Android and 2 iOS devices. 23 percent energy cut vs MNN with no speed loss; 39 to 78 percent vs llama.cpp, ExecuTorch, mllm, MediaPipe.
- LLM Inference at the Edge under sustained load (Tummalapalli et al., arXiv 2603.23640, Mar 2026). Fetched abs. iPhone 16 Pro loses about 50 percent throughput within two iterations; Galaxy S24 Ultra hits an OS GPU floor that halts inference; RTX 4050 131.7 tok/s at 34.1 W; Hailo-10H 6.9 tok/s under 2 W.
- Autothrottle (Wang, Li, Liang, Wu, Yan, NSDI 2024). Fetched arXiv 2212.12180. Bi-level: application-level learned controller (contextual bandit) sets CPU throttle-ratio targets from SLO feedback; per-service heuristic controllers meet the targets. Up to 26.21 percent CPU savings over the best baseline.
- Taming and Controlling Performance and Energy Trade-offs in Network Applications (Dong, Awad, Arora, Krieger, Appavoo, arXiv 2502.14987, Feb 2025). Fetched abs. Bayesian optimizer tries a few settings of interrupt coalescing and DVFS on the live server; up to 60 percent energy reduction while meeting tail-latency SLA.
- Odyssey (Flinn, Satyanarayanan, SOSP 1999). Search hit: https://dl.acm.org/doi/pdf/10.1145/319151.319155. Fidelity adaptation driven by user-specified battery lifetime; energy reductions 7 to 72 percent, mean 36 percent; goals varying by 30 percent met. Venue later verified by the Semantic Scholar API; PDF pages did not parse.
- GRACE-1 (Yuan, Nahrstedt, Adve, Jones, Kravets, IEEE TMC 2006). Search hit: http://rsim.cs.illinois.edu/Pubs/06MOBILE.pdf. Global adaptation sets quality level, CPU allocation and average power; local adaptation enforces. Venue later verified by the Semantic Scholar API; PDF did not parse.
- CoAdapt (Hoffmann, ECRTS 2014). Search hit: https://ieeexplore.ieee.org/document/6932604/. Guarantees any two of performance, power, accuracy while optimizing the third. Venue later verified by the Semantic Scholar API.

## Appendix B. Raw notes, batch 2 (second fetch round)

- JouleGuard remainder (jouleguard.txt lines 783 to 1199): coordinated approach gives uniformly higher accuracy than application-only for the same energy; accuracy only drops once system-level knobs are exhausted; phase change at frame 200 gives a short energy spike then higher accuracy. Related-work text: prior cross-layer work (GRACE, Agilos, CoAdapt) splits the problem into two linear problems; JouleGuard keeps the dependence and makes the controller robust to the learner through the model-error interface.
- GearDVFS remainder (geardvfs.txt lines 634 to 1199): baselines Deft (schedutil plus simple_ondemand), Q-learning, zTT. PPW gains 15.0 to 23.9 percent over Deft, 6.0 to 19.6 percent over Q-learning, 6.2 to 15.2 percent over zTT on self-driving tasks; 7.0 to 26.9 percent over zTT on the robot tasks. Training from scratch takes about 35 minutes (2100 s) on an i7-8700K PC. Adaptation to a QoE change takes about 500 s (zTT about 700 s); a new unseen task takes about 650 s. Redmi Note 9 phone: zTT improves PPW 4 to 7 percent, Gear 6 to 9 percent over the stock governor, up to 16.92 percent after six user traces. Runtime cost about 170 mW, 0.4 MB, decision in about 100 ms. Deep learning tasks on Jetson NX: 9.06 W with schedutil versus 7.01 W with Gear (22.6 percent).
- SOSA remainder (sosa.txt lines 450 to 950): objective functions delta_IPS = Power/maxPower subject to IPS >= constraint, delta_Power = |IPS - refIPS|/maxIPS subject to Power <= budget; reward = 1 - delta, zero on constraint violation. Supervisor invoked every 100 ms, LCTs every 10 ms in gem5; hardware LCTs every 5 ms on a Virtex-7 FPGA with 9.62 percent slice overhead. Under disturbance the classical SISO controller degrades up to 14 percent while the LCT settles below 5 percent then below 1 percent error. Learning exploration lasts about 3.5 s per core.
- zTT (Semantic Scholar API, DOI 10.1145/3458864.3468161): title, MobiSys 2021, authors Seyeon Kim, Kyungmin Bin, Sangtae Ha, Kyunghan Lee, Song Chong, 77 citations, abstract confirms Pixel 3a and Jetson TX2 and the 23.9 percent number. README does not give training length. No open-access PDF listed.
- CALOREE (Semantic Scholar API, DOI 10.1145/3173162.3173184): ASPLOS 2018, Mishra, Imes, Lafferty, Hoffmann. Two sub-tasks: learn how interacting resources affect speedup; control speedup to meet latency with minimal energy. ARM big.LITTLE, single and multi-application. Versus best prior learning and control solutions: 60 percent fewer deadline misses and 13 percent less energy.
- LEO (Semantic Scholar API, DOI 10.1145/2694344.2694373): ASPLOS 2015, Mishra, Zhang, Lafferty, Hoffmann. Hierarchical Bayesian model gives online estimates of power and performance per configuration; compared to offline learning, online learning, a heuristic and the true optimum; most accurate estimates and near-optimal energy.
- NestDNN (arXiv 1810.10090, MobiCom 2018, Fang, Zeng, Zhang): multi-capacity model with nested descendant models; runtime picks a resource-accuracy point per app; 4.2 percent accuracy gain, 2.0x frame rate, 1.7x energy reduction.
- PELM (Semantic Scholar API, DOI 10.1145/3774906.3802783): Weisi Yang, Stephen Xia, 2026. DVFS plus speculative decoding with variable verification depth; up to 23.1 percent speedup and 52.4 percent energy reduction.
- DVFSLM (Semantic Scholar API, DOI 10.1145/3812836.3814753): Jiesong Chen, Lixiang Han, Jiani Cao, Zhenjiang Li, MobiSys 2026 Workshops. Workload-aware power and latency estimators from matrix-operation counts and hardware metadata; runtime governor minimizes energy while meeting token-generation deadlines.
- Interpretable ML CPU-GPU governor (Semantic Scholar API, DOI 10.1145/3470974): Park, Dutt, Lim, ACM TECS 2021. Offline model trees deployed online; 20 mobile games; over 10 percent (up to 38 percent) energy-per-frame improvement with 3 percent higher fps versus a linear-regression governor.
- DVFO (arXiv 2306.01811, journal version IEEE TMC 2024 per search hit): DRL co-optimizes CPU, GPU, memory frequency and offloading; three edge devices; 33 percent energy reduction, 28.6 to 59.1 percent latency reduction.
- Adaptive model selection (arXiv 1911.04946, ACM TECS 2020, Marco, Taylor, Wang, Elkhatib): Jetson TX2; offline-trained predictor picks a DNN per input for a target accuracy and time; 1.8x faster with 7.52 percent higher accuracy than the most capable single model.
- E4 (arXiv 2503.04865, AAAI 2025, Zhang, Zhao, Chang, Lin, Liu): attention-based cascade picks the exit point per frame; a just-in-time profiler with coordinate descent sets CPU and GPU frequency per layer; up to 2.8x speedup and 26 percent average energy saving; five Jetsons; baselines EENet, zTT, Ring-DVFS.
- PaRMIS (arXiv 2105.09282, DAC 2021, Deshwal, Belakaria, Bhat, Doppa, Pande): information-theoretic Bayesian search for Pareto-optimal parametric resource-management policies on Odroid-XU3; at most 500 iterations, converging by about 300; 13 and 23 percent higher hypervolume than RL and IL.
- Energy/Frequency Convexity Rule (arXiv 1401.4655, 2014, De Vogeleer, Memmi, Jouvelot, Coelho): one week of high-resolution power traces on a smartphone; energy per input element versus CPU frequency is convex with a clear minimum inside 0.2 to 1.6 GHz.
- Temporal-encoding DRL DVFS (arXiv 2309.03779, Journal of Systems Architecture 2023, Zhou, Lin): Jetson Nano; 3 to 11 percent more energy saving than ondemand on MiBench, 5 to 14 percent on AudioReg and FaceReg; in-kernel quantized network engine.
- Multi-task DRL DVFS (arXiv 2409.19434, Li, Zhou, Wang, Lin, 2024, preprint): Jetson Nano 2GB; 3 to 10 percent power saving over Linux governors on 3, 5, 8 task sets.
- HiDVFS (arXiv 2601.06425, Jan 2026, under review at IEEE TPDS, Pivezhandi, Saifullah, Jannesari): three agents (profiler, thermal, priority) with a schedulability gate and a conformal shield; Jetson TX2, Orin NX, RubikPi; 2.83x speedup and 32.9 percent energy reduction versus GearDVFS; 15 to 18 percent energy reduction versus max-frequency pinning.
- SparseDVFS (arXiv 2603.21908, Mar 2026, Zhang, Wu, Liu, Mottola): offline map from operator sparsity to CPU/GPU/EMC frequency triplets; 78.17 percent energy-efficiency gain claim.
- PolyThrottle (arXiv 2310.19991, 2023, Yan, Wang, Venkataraman): Constrained Bayesian Optimization over GPU, memory and CPU frequency and batch size on Jetson TX2 and Orin under latency constraints; about 15 samples; up to 36 percent energy saving; memory frequency 12 to 25 percent; grid 5005 and 1820 points, exhaustive 14 and 5 hours.
- GPU DVFS for SLM fine-tuning (arXiv 2607.05933, Jul 2026, Park et al.): Jetson AGX Orin; ML model selects energy-optimal GPU frequency; 13.11 percent average (up to 26.73 percent) energy saving versus MAXN.
- Joint memory and compute frequency (arXiv 2608.13863, Aug 2026, Han, Nan, Zhou, Niu): convex closed-form near-optimal solution within 2.5 percent of optimal under deadlines; up to 10.4 percent device energy reduction.
- WASL (Semantic Scholar API, DOI 10.1145/3777884.3797009): Pervaiz, Das, Kodagi, Santriaji, Hoffmann, ICPE 2026. Detects interference between colocated adaptive modules from expected-versus-observed deviations and slows each module's adaptation rate; tail latency reduced up to 84 percent, comparable to centralized coordination.
- CoAdapt (Semantic Scholar API, DOI 10.1109/ECRTS.2014.32): ECRTS 2014, Hoffmann; abstract not exposed by the API.
- Odyssey (Semantic Scholar API, DOI 10.1145/319151.319155): SOSP 1999, Flinn and Satyanarayanan; abstract not exposed by the API; TOCS 2004 version (DOI 10.1145/986533.986534) likewise; numbers remain from search snippets.
- MobiRL (Semantic Scholar API, DOI 10.1145/3674910): Dou, Liu, Xiao, ACM TACO 2024. RL scheduler for CPU/GPU frequency on commercial smartphones; 4.1 percent lower frame-drop rate and 42.8 percent lower power than commercial schedulers; versus Q-learning up to 2.5 percent lower frame drops and 32.6 percent less power; deployed in commercial devices.
- SLEXNet (Semantic Scholar API, DOI 10.1145/3689632): Kutukcu, Baidya, Dey, ACM TECS 2024. Slimmable plus early-exit network with a runtime scheduler that estimates time and power of each variant; Jetson Orin.
- Adaptive KV-cache quantization (arXiv 2604.04722, CVPR 2026 per abs page, Boroujeni et al.): learned controller picks 2, 4, 8 bit or FP16 per token; 17.75 percent lower decode latency than static quantization on SmolLM-360M; within 0.30 points of FP16.
- Diminishing returns of early-exit decoding (arXiv 2603.23701, Mar 2026, Wei et al.): newer LLM generations have less layer redundancy; early-exit benefit shrinks; dense base models keep the most.
- LLM reasoning under strict output length constraint (arXiv 2504.14350, Apr 2025, Sun et al.): 30 models; the best model size and prompt style change with the token budget; budgets are tied to on-device latency.
- MetaDVFS body (arXiv html 2509.22707): Pixel 3, 4, 6, 8, 9; apps TikTok, Kwai, Bilibili, Weibo, Taobao, 3DMark; baselines schedutil, zTT, GearDVFS, Orthrus (PPO); DQN with a liquid neural network; state IPC, CPU util, CPU freq, GPU util, GPU freq, power; adaptation 3.5 plus or minus 1.1 minutes for a new device-app pair versus 11.8 plus or minus 5.2 minutes incremental training; PPW 1.00 to 1.17 versus zTT 0.52 to 0.98; no fixed-frequency oracle or tuned static policy reported.
- FUSE body (arXiv html 2507.02135): Pixel 7 and 7 Pro (Tensor G2, Mali-G710), Android 13, llama.cpp with OpenCL; TinyLlama-1.1B, StableLM-Zephyr-3B, Llama-2-7B, DeepSeek-R1-Distill-Qwen-1.5B. FUSE profiles offline at install time for five prefill-length ranges and pins CPU, GPU and memory frequencies per phase; no online learning. Fixed best combinations cut TTFT up to 40.4 percent and TPOT up to 31.8 percent at the same energy as the governors.
- PowerLens (arXiv 2603.19584, Mar 2026): LLM agents manage 18 Android settings on rooted devices; 81.7 percent action accuracy, 38.8 percent energy saving over stock Android, 0.5 percent of daily battery overhead, preferences converge in 3 to 5 days. Peripheral.

## Appendix C. Raw notes, batch 3 (last verification round)

- ALERT (arXiv 1911.00119; USENIX ATC 2020; Wan, Santriaji, Rogers, Hoffmann, Maire, Lu): selects DNN variant and system configuration jointly; probabilistic global-slowdown estimate; over 13 percent energy and 27 percent error reduction over single-level adaptation; within 3 percent energy and 2 percent error of an oracle; CPU and GPU; image and speech.
- FSE 2015 DOI from dblp: 10.1145/2786805.2786833, ESEC/FSE 2015 pages 13 to 24.
- Learning-directed CPN DVFS (PMC6163884; Sensors 2018; Chen, Chang, Yu, Chen, Liang): PXA270 and Jetson TK1; counter-propagation network; performance levels 70 and 90 percent; versus ondemand 4.88 to 42.63 percent (single core, 70 percent) and 3.9 to 15.4 percent (multicore, 70 percent) energy reduction; overhead 0.02 to 0.12 percent.
- DATE 2020 user-interaction-aware RL (Essex repository 27546; Dey, Singh, Wang, McDonald-Maier): Exynos platform; up to 50 percent power saving and 29 percent lower peak temperature versus stock Android; 41 and 19 percent over prior power and thermal schemes.
- DyPO (ASU listing; ACM TECS 16(5s) 2017; Gupta, Patil, Bhat, Mishra, Ogras): offline classifiers from counters to Pareto-optimal configurations; PPW 93 percent over interactive, 81 percent over ondemand, 6 percent over powersave on 18 applications.
- SPECTR (ACM SIGPLAN Notices listing; ASPLOS 2018; Rahmani, Donyanavard, Muck, Moazzemi, Jantsch, Mutlu, Dutt): supervisory control theory with gain scheduling on Exynos big.LITTLE; CMU PDF did not parse.
- MEANTIME (USENIX listing; ATC 2016 pages 421 to 435; Farrell, Hoffmann): approximation for hard timing, resource allocation for energy; six applications on Linux/ARM; PDF blocked.
- Budget RNNs (dblp; RTAS 2021 pages 143 to 156, Outstanding Paper; Kannan, Hoffmann; GitHub tejaskannan/budget-rnn): MSP430 FR5994 plus HM-10 BLE plus supercapacitors; halting thresholds fitted per energy budget level.
- Slow Down or Sleep (USENIX ATC 2011; Le Sueur, Heiser): mirror https://trustworthy.systems/publications/papers/LeSueur_Heiser_11.pdf parsed partially: Core i7, Atom, OMAP; MPEG, Apache, SPECjbb; racing to deep idle often beats sustained slow-down on modern parts.
- Carroll and Heiser HotPower 2013 (dblp DOI 10.1145/2525526.2525850; medusa governor README): APQ8064, I9500, I9505; per-platform frequency threshold for adding cores.
- Critical Power Slope (ACM listing, ICS 2002; Miyoshi, Lefurgy, Van Hensbergen, Rajamony, Rajkumar): Pentium highest frequency always efficient, PowerPC lowest always efficient; idle power decides.
- AGFT (arXiv 2508.01744, Aug 2025; Ye, Zhang, Tang): online RL GPU frequency tuner for cloud LLM serving; 44.3 percent GPU energy reduction with under 10 percent latency overhead; not mobile.
- Odyssey slides (TU Dresden PDF) and CMU PDF did not parse; Semantic Scholar web page returned empty; numbers stay marked as search-snippet only.
- Search budget (200 searches) was exhausted at the end of batch 3; Orthrus and Gambler could not be verified and are marked unverified.
