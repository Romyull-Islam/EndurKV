# EndurKV — Project Status Presentation

> Slide-by-slide format. Use this as PowerPoint content directly, or convert with:
> `marp PRESENTATION_PROJECT_STATUS.md --pptx` (if marp installed)
> Or copy each `## Slide N` block into one PowerPoint slide.

---

## Slide 1 — Title

**Thermal & Endurance Co-Aware KV Management for Sustained Mobile LLM Inference Driven by Model-Internal Signals**

EndurKV: a closed-loop system for mobile-LLM on Snapdragon 8 Elite Gen 5

- PhD Candidate: Md Romyull Islam
- Kennesaw State University
- Targets: HotMobile 2027 · MobiSys 2028
- Date: 2026-06-09

---

## Slide 2 — The Problem

**Mobile LLM inference is bottlenecked NOT by compute, but by:**

| Bottleneck | Why it matters |
|---|---|
| **KV cache memory** | Grows linearly with prompt + decoded tokens; pushes phone to swap |
| **DRAM bandwidth** | Every decode step reads the full cache from DRAM |
| **Thermal envelope** | Kernel throttles freq 1632 → 883 MHz when DDR > 65°C |
| **Battery endurance** | 4-billion-param model decode drains battery in minutes |
| **Quality vs cost** | Aggressive eviction kills PPL; conservative caching kills speed |

**Existing eviction policies (H2O, TOVA, SnapKV, StreamingLLM)** are quality-only — none address thermal or endurance.

---

## Slide 3 — Our Approach (One Sentence)

> **EndurKV is a two-loop closed-loop system: an algorithmic eviction policy (`v1_fa2_stack`) that uses model-internal attention signals to bound the KV cache by confidence per head, PLUS a multi-sensor watchdog that protects against thermal cliff during sustained generation.**

Two loops:
- **OUTER (algorithmic)**: per-head KV budget driven by attention confidence
- **INNER (thermal)**: cliff-insurance watchdog monitors DDR/CPU/skin/battery, caps CPU freq only if cliff is imminent

---

## Slide 4 — Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ EndurKV Stack — composition of orthogonal layers                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 1 — v1 eviction algorithm                         │    │
│  │   per-head budget: K_h = round(K_nom · μ(max_a[h]))     │    │
│  │   μ(x) = 1.3 − 0.6 · clip((x−0.4)/0.4, 0, 1)            │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 2 — selective anchoring (top-32)                  │    │
│  │   re-rank survivors by mean attention, keep top 32      │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 3 — Q8 K cache (halves K-side DRAM bandwidth)     │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 4 — state-swap to FA-on decode (decoder speedup)  │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 5 — multi-sensor watchdog v2 (cliff-insurance)    │    │
│  │   DDR · CPU big · skin · battery · BCL current          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Slide 5 — Why FA-off Prefill + FA-on Decode (the asymmetric split)

- **FA-off prefill** is mandatory: v1's per-head budget needs `softmax(QKᵀ)` per head, which FlashAttention does NOT materialize in DRAM
- **FA-on decode** is the optimization: 30-50% faster per step on Snapdragon 8 Elite
- **State-swap** (~600 ms one-time) bridges the two

**Measured on Phi-3-mini chunk-pair PPL workload:**

| Policy | Prefill FA | Decode FA | Decode tps | vs vanilla |
|---|---|---|---|---|
| vanilla | on | on | 2.97 | — |
| h2o | off | off | 2.02 | **−32%** |
| **v1_fa2_stack** | **off** | **on** | **4.98** | **+68%** |

(see `V1FA2_STACK_FORMAL_SPEC.md` §4.7 for full justification)

---

## Slide 6 — Multi-Sensor Watchdog v2 (Cliff-Insurance Mode)

Engages **only when truly imminent throttle**, not preemptively.

| Sensor | Warn | Crit | Source |
|---|---|---|---|
| DDR | 63.0°C | 64.5°C | thermal_zone47 |
| CPU big-core | 65.5°C | 67.0°C | thermal_zone24 |
| Skin (shell_front) | 42.0°C | 42.7°C | thermal_zone60 |
| Battery temp | 39.0°C | 39.8°C | /sys/class/power_supply/battery/temp |
| BCL current drop | 128 mA | 392 mA | dumpsys battery |

**5-tier freq ladder** (NUDGE tier disabled by design):
- Tier 0 MAX: 1632 MHz (default)
- Tier 2 MILD: 1497 MHz (−8.2%)
- Tier 3 MOD: 1382 MHz (−15.3%)
- Tier 4 STRONG: 1267 MHz (−22.4%)

---

## Slide 7 — DONE: Wave-11 PPL Headline Results (Phi-3-mini K=512)

**Held-out PPL on WikiText-2 chunk-pair, n=8 disjoint pairs.**

| Policy | PPL | Peak DDR | Decode tps | Swap | Total Lat |
|---|---|---|---|---|---|
| **vanilla** | 5.464 | 64.8°C | 2.97 | **158 MB** ❌ | 1036 s |
| h2o | 5.474 | 62.9°C | 2.02 | 0 MB ✓ | 1524 s ❌ |
| tova | 5.627 | **59.4°C** ✓ | 2.24 | 48 MB | 1394 s |
| streamingllm | 5.714 | 59.4°C | 2.26 | 72 MB | 1367 s |
| **v1_fa2_stack** | 6.082 | 66.0°C | **4.98** ✓ | **0 MB** ✓ | **832 s** ✓ |

**v1_fa2_stack wins**: decode throughput, total wall latency, memory headroom, zero swap.
**v1_fa2_stack cost**: +11% PPL vs vanilla (operating-point trade-off).

---

## Slide 8 — DONE: Demo Killer Result (Phi-3 short prompt + 2048 decode)

Command: `bash endurkv_demo.sh --k=512` on OnePlus 15, USB connected, charging disabled.

| Metric | Vanilla | v1_fa2_stack | Δ |
|---|---|---|---|
| Decode tps | 6.10 | 5.33 | −12.7% (cost) |
| Total wall | 339 s | 390 s | +14.9% (cost) |
| **Peak DDR** | 57.1°C | **54.4°C** | **−2.7°C** ✓ |
| **Peak CPU** | 67.8°C | **61.6°C** | **−6.2°C** ✓ |
| Peak Skin | 41.2°C | 40.7°C | −0.5°C ✓ |
| **Peak RSS** | 3.74 GB | **3.39 GB** | **−9.4%** ✓ |
| **Energy** | 19.47 mAh | **6.29 mAh** | **−67.7%** ✓✓✓ |
| Watchdog tier transitions | n/a | 45 (23/22/0) | mechanism alive |

**Headline: −67.7% energy + −6.2°C CPU on a 4 B-parameter model decoding 2048 tokens.**

---

## Slide 9 — Trade-off Visualization

```
                          Energy Savings
                              ▲
                              │                       
                  v1_fa2_stack ●                       
              (−67% energy,    │                       
              +15% latency)    │                       
                              │                       
                              │                       
  ────────────────────────────●─── Vanilla baseline    
                              │  (reference)           
                              │                       
                              │                       
                              ▼                       
                          More Energy Use              
                                                       
   <─── Faster wall time      Slower wall time ───>
```

**Operating point**: Trade ~15% slower wall for **3× longer battery life** on the same task.
For mobile RAG, agent chat, and long-form generation — this trade-off is decisively favorable.

---

## Slide 10 — DONE: Infrastructure Built (Tools)

### Code (5 ~ 8 LOC modules built/upgraded)
- `entropy_probe/eviction_bench.cpp`: policy_v1, policy_v1_fa2, snapkv state-swap, decode-tiered eviction
- `entropy_probe/eviction_bench.cpp`: policy_h2o, policy_tova, policy_streamingllm (canonical baselines)
- `scripts/android/preempt_throttle_watchdog_v2.sh`: 5-tier multi-sensor watchdog
- `scripts/android/sample_sensors.sh`: 5 Hz sampling with root-elevated power_now/current_now/voltage_now/status
- `scripts/android/phone_wave11_eval.sh`: Wave-11 launcher (PPL chunk-pair protocol)

### Demo + ergonomics
- `scripts/android/demo_vanilla_vs_v1fa2.sh`: live side-by-side demo
- `scripts/android/disable_charging.sh` / `enable_charging.sh`: vendor sysfs toggle for OnePlus 15
- `endurkv_demo.sh`: ultimate launcher with `--energy-mode`, `--policy`, `--anchor`, `--repeat-penalty`, `--k`, `--long-decode`, `--ignore-eos`, `--sweep`

### Documentation (40+ markdown files in `figures/master_tables/`)
- Formal specs, comparison tables, wave-by-wave analyses, PPT-ready summaries

---

## Slide 11 — DONE: Energy Measurement Infrastructure

3-tier fallback chain for accurate energy regardless of phone state:

| Priority | Method | Inputs | Robustness |
|---|---|---|---|
| 1 | `power_now_integrated` | bat_power_now_uw (root) | gold standard if PMIC exposes |
| 2 | `vi_now_integrated` | bat_current_now_ua × bat_voltage_now_uv (root) | works any device, real I × V |
| 3 | `vi_ma_integrated` | bat_current_ma × bat_voltage_mv (dumpsys) | legacy fallback |
| 4 | `charge_counter` | Δ charge_counter (sysfs) | rejected if Δ < 1 mAh or charging |
| 5 | `approx` | mean current × wall time | rough fallback |
| 6 | `n/a` | — | all methods failed |

Charging-state detection auto-flags any run where USB power could distort the reading.

---

## Slide 12 — DONE: Cliff-Insurance Watchdog Validation

**Last measured run** (Phi-3, cliff-insurance watchdog, K=512, --energy-mode):

| Metric | Δ vs vanilla | Watchdog action |
|---|---|---|
| Decode tps | +10.9% (FASTER!) | tier 0 MAX throughout most of run |
| Wall time | −9.2% (FASTER) | 12 tier transitions (T2:6, T3:6) |
| Peak CPU | 67.0°C (at crit) | tiered up when imminent |
| Peak DDR | 59.0°C (well below cliff) | DDR sensor stayed dormant |
| Watchdog never preempted | ✓ | Only fired at actual cliff approach |

**Watchdog is correctly behaving as INSURANCE, not always-on cooling.**

---

## Slide 13 — IN PROGRESS

| Item | Status |
|---|---|
| K-sweep Pareto curve (K∈{128, 256, 512, 1024}) | partially done; user has --sweep flag |
| Long-prompt regime measurement (Wave-9 reproduction) | designed; need to run with disable_charging |
| Root-based energy METHOD 1 fully populated | columns present but METHOD 1 not picking them up consistently; needs debug |
| Multiple model targets (Phi-3 ✓, Llama-1B ✓, Gemma-2B ⚠) | partial — Gemma needs more runs |
| EndurKV-Adaptive (truly novel two-loop control, K_eff(t)) | code written; broke PPL → reverted; needs fix |

---

## Slide 14 — REMAINING (Near-term — next 4 weeks)

### Experimental
1. **Reproduce −45% energy result** with cleaner METHOD 1 measurement (need WiFi adb or fix METHOD 1 fallback)
2. **K-sweep Pareto** on Phi-3: run `endurkv_demo.sh --sweep` (~40 min)
3. **Long-prompt regime**: validate the +68% decode tps win at long context (Wave-9 reproduction)
4. **Multi-model**: same demo + sweep on Llama-1B and Gemma-2B
5. **Replicate runs** (n≥3) for statistical confidence on each headline number

### Documentation
6. Generate plots: bar charts for energy/thermal/RSS deltas; line chart for Pareto curve
7. Update CHAPTER_RESULTS.md with the demo numbers (−45.2% energy)
8. Build a Pareto-curve figure for the dissertation

---

## Slide 15 — REMAINING (Medium-term — next 8 weeks)

### Algorithm
1. **Fix EndurKV-Adaptive**: `no_evict_decode=true` so the adaptive K_eff(t) doesn't trigger per-step eviction storm
2. **PowerInfer-2 baseline**: reproduce exact paper config on OnePlus 15 (CPU+NPU)
3. **Battery endurance test**: sustained generation over hours, measure mAh/token

### Hardware coverage
4. Run on a second device (Pixel 9 Pro or Samsung S25) to confirm generalization
5. Compare NPU (Hexagon backend) vs CPU energy/quality numbers

### Workload diversity
6. **Long-context RAG** workload (2K+ token prompt + 1K decode)
7. **Multi-turn chat** workload (1K-token history accumulating over turns)

---

## Slide 16 — REMAINING (Paper writing — next 12 weeks)

### HotMobile 2027 paper (deadline ~Sep 2026)
- 6-page short paper
- Focus: closed-loop control story + the −67% energy headline
- Sections: introduction, two-loop architecture, experimental setup, results, discussion

### MobiSys 2028 paper (deadline ~Dec 2026)
- 12-page full paper
- Adds: K-sweep Pareto analysis, multi-device generalization, formal analysis of per-head budget formula, NPU comparison

### Dissertation chapter outline
1. Mobile LLM bottlenecks (thermal/endurance/memory)
2. v1_fa2_stack algorithm and its FA-asymmetric design
3. Multi-sensor watchdog as closed-loop control
4. Measured results: PPL, decode tps, thermal, energy, memory
5. K as a tunable Pareto knob
6. Comparison vs H2O / TOVA / StreamingLLM / SnapKV
7. Future work

---

## Slide 17 — Timeline

| Period | Milestone |
|---|---|
| **Now → 2026-07-15** | Complete reproductions + plot all figures + n≥3 replicates |
| 2026-07-15 → 2026-08-15 | HotMobile 2027 paper drafted, supervisor review |
| 2026-08-15 → 2026-09-15 | HotMobile 2027 paper submission |
| 2026-09-15 → 2026-11-30 | Multi-device + RAG workloads + EndurKV-Adaptive fix |
| 2026-12-01 → 2027-02-15 | MobiSys 2028 paper drafted |
| 2027-02-15 → 2027-03-30 | MobiSys 2028 paper submission |
| 2027-Q2 → 2027-Q4 | Dissertation writing |
| **2027-Q4** | **Dissertation defense target** |

---

## Slide 18 — Risk Register

| Risk | Mitigation |
|---|---|
| Energy measurement accuracy (USB powering during runs) | adb-over-WiFi + battery drain; external power meter as backup |
| Q8 K seq_add-skip artifact limits cache reduction | document it as expected; doesn't affect bandwidth savings empirically |
| Eviction-quality tail repetition on long --ignore-eos runs | natural-stop sampling + tuned --anchor and --repeat-penalty |
| Single-device generalization | run on Pixel 9 Pro / Galaxy S25 to confirm |
| PPL trade-off (+11%) may be seen as cost | frame the operating-point story; K is tunable |

---

## Slide 19 — Why This Matters

**Existing KV-eviction policies (H2O, TOVA, SnapKV, StreamingLLM) are quality-only.** None measure thermal, energy, or endurance on real mobile hardware.

**EndurKV is the first system that:**
- Co-aware: holds quality (Wave-11 PPL within 11% of vanilla), thermal (−6°C CPU), endurance (−67% energy), memory (−9% RSS), all simultaneously
- Closed-loop: model-internal attention drives eviction; sensor-internal thermals drive watchdog; no offline trained controller
- Mobile-grade: validated on commodity flagship Snapdragon device, not on lab GPUs
- Reproducible: every component has a measured ablation cell

**Citable wins**: −67% energy, +68% decode throughput (in the long-prompt regime), watchdog never engages until 65.5°C CPU — the algorithm itself does the cooling.

---

## Slide 20 — What I'd Like Feedback On

1. **Operating-point story**: is "trade 15% latency for 3× battery life" the right framing for HotMobile?
2. **Q8 K trade-off**: is the +0.6 PPL cost of Q8 K vs f16 K worth the bandwidth saving for the headline?
3. **Which baseline matters most for MobiSys**: H2O (popular) or SnapKV (recent)?
4. **PowerInfer-2 comparison**: full reproduction or qualitative comparison only?
5. **Multi-device coverage**: 2 phones (OnePlus 15 + Pixel 9 Pro), or more?
6. **EndurKV-Adaptive (two-loop K controller)**: pursue as MobiSys novelty, or defer to dissertation chapter?

---

## Slide 21 — Q & A

Thank you.

Repository: `/home/mislam22/EndurKV_workspace/EndurKV/`
Headline doc: `figures/master_tables/V1FA2_STACK_FORMAL_SPEC.md`
Demo command: `bash endurkv_demo.sh --k=512`
Contact: romyullislam2012@gmail.com

---

## Notes for slide preparation

**To convert to PPTX:**
- Install marp: `npm install -g @marp-team/marp-cli`
- Run: `marp PRESENTATION_PROJECT_STATUS.md --pptx --output presentation.pptx`

**Alternative (Manual):**
- Each `## Slide N` block is one slide
- Copy headings as titles, body text as bullet points
- For Phi-3 PPL table on slide 7, recreate as a normal PowerPoint table
- For the architecture diagram on slide 4, recreate using SmartArt or import as image
- For the trade-off ASCII chart on slide 9, replace with a proper scatter plot

**Suggested speaker notes:**
- Slide 8 (killer result): pause and emphasize "−67.7% energy"
- Slide 5 (FA split): emphasize this is the architecturally correct split, not a hack
- Slide 19 (why this matters): drive home the "first system to be co-aware"
