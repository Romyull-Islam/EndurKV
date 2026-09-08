# Figure provenance (PAPER_MUKV_6PG / PAPER_MUKV_10PG), 2026-09-07

Every figure regenerates from the listed script and data on this date. Diagrams carry no plotted data; their numbers are checked against the scripts named.

| Figure | Script | Data | Notes |
|---|---|---|---|
| fig_architecture_v9 | master_tables/figures/fig_architecture_v9.py | scripts/android/ukv_sched.sh (tiers, lever, exchange rates, bias +-0.1, clamp 0.3, EMA clip 10%), preempt_throttle_watchdog_v2.sh (battery 47.0..49.5 C, skin 50.0..52.5 C -> 1497.6..1017.6 MHz), gpu_watchdog.sh (DDR 60/62/63.5 -> 1050/967/902) | cache strip proportional to the measured 4/721/299 split of K=1024 (Table 1, /tmp/nat_cpu/mukv_faon); same split drawn in fig_mechanism |
| fig_control_plane_v2 | master_tables/figures/fig_control_plane_v2.py | same scripts | diagram |
| fig_mechanism | master_tables/figures/fig_mechanism.py | master_tables/figures/data/attn_llama1b_layer8_16q_bins64.npy, mech_joint_keep_16bins.npy, mech_perhead_keep_layer8_16bins.npy, mech_keepsets.json; source capture: attention_probe (entropy_probe/build-host) on prompts_chat/Llama-3.2-1B/gov_report_pub_001.txt, Llama-3.2-1B-Instruct-Q4_K_M, 16 decode steps, seed 42 | heatmap and both keep-set grids are measured; union over 32 heads keeps 13 of 16 bins, the joint set 6 |
| fig_cache_trajectory | master_tables/figures/make_pub_figures.py (ROOT=/tmp/nat_cpu) | /tmp/nat_cpu/{vanilla,snapkv,adakv,mukv_faon}/gen.json, gen_prefill.csv, gen_steps.csv | retained cells 9737/5929/3484/721 = Table 1 |
| fig_bonsai_thermal, _2p | master_tables/figures/fig_bonsai_thermal.py, fig_bonsai_thermal_2p.py | /tmp/nat_bonsai/vanilla, /tmp/nat_bonsai/mukv_faon, /tmp/wd_vanilla (sensors.csv, meta) | wall 85.6/33.2/85.7 min; battery peaks 50.2/44.2/49.1 C; vanilla 17 dips, longest 123 s, 923 s at 883 MHz; watchdog run 0 s at 883 |
| fig_loop_proof | figures/fig_loop_proof.py | /tmp/loop_proof/*/sched_log.txt (13 cells) | misses: burn cells 1.31/1.30 of energy budget; cap cells 1.10/1.11/1.12 of time budget |
| fig_discharge_timeline, _col | figures/fig_discharge_timeline.py | /tmp/discharge_final/discharge (123 requests, sensors.csv per request, timeline.csv) | tiers 885/748/506 J, 280/210/177 s, n=15/21/10 |

## Update 2026-09-07 (evening)

Figures 1 to 3 are now TikZ, compiled with tectonic from `figures/fig_architecture_tikz.tex`, `fig_control_plane_tikz.tex`, `fig_mechanism_tikz.tex` (the mechanism figure inputs `heat.pdf`, the measured log-scale attention slice, and `mech_grids.tex`, generated from `data/mech_*_16bins.npy`). Content and numbers are unchanged from the matplotlib versions listed above.

Table 1 lower blocks: `/tmp/phi3_cpu_complete` (Phi-3-mini, 7542-token prompt, 2048 generated, run_phi3_cpu_complete.sh, 2026-08-25) and `/tmp/gap_cpu_gemma` (gemma-2-2b-it, 6382 tokens, run_gap_closure.sh, 2026-08-26); baselines at published budgets (SnapKV 1024/head window 32 pool 7, Ada-KV 2048, TOVA 2048, StreamingLLM 4+2000, H2O 20% of N); energy from scripts/clock_cell_report.py (USB rail + coulomb pack), J/3.6; live cells = retained_kv_bytes / (vanilla retained bytes / N); DDR peak from sensors.csv over the request.

Mechanism figure left column corrected 2026-09-07: the per-head budget gate box is gone; the drawn steps are aggregate (mean over layers and heads), max-pool kernel 7, alpha-gate (floor 0.70), joint keep set. Evidence: '[v1_fa2] selective anchoring: spread-gate kept 9409, top-721' in /tmp/nat_cpu/mukv_faon/gen.err and the matching lines in the GPU, Phi-3 and gemma cells (gate pool 96.5 to 99.8% of the prompt).

## Gate removal A/B (2026-09-07)

Host build (entropy_probe/build-host, libllama rebuilt), Llama-3.2-1B-Instruct-Q4_K_M, the CPU table's 9737-token prompt (pulled from the phone), frozen muKV arguments, 32 greedy tokens, seed 42. Arms: gate on, and `--no-perhead-gate` (pool = every prompt position). Keep sets dumped with EVICT_DUMP_KEEP: 721 of 721 positions identical, retained_kv_bytes identical (23634780), generated text identical. Files under artifacts/gate_ab/. The bench flag and the dump were added to eviction_bench.cpp on this date.

## Architecture figure, wide layout (2026-09-08)

fig_architecture_tikz.tex/.pdf is now the wide two-row layout (486 x 212 pt, aspect 2.3; the three-row version is kept as fig_architecture_tikz_tall.*). Same content and numbers: pass with the KV strip on top; thermal, energy and learning columns in two rows below; steps 1 to 9. Used by both drafts (Figure 1), the main talk deck slide 10 and the energy deck slide 5.
