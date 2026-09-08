# EndurKV / μKV — Work Plan (updated 2026-07-20)

## The 3 canonical μKV variants (all share `--gate-alpha-floor 0.70`)
`MU = --policy v1_fa2 --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70`
1. **μKV-sol2(fa-on)+defrag** = `$MU --fa-on-evict`  (Solution 2 fa-on prefill+decode + CPU state-roundtrip defrag)
2. **μKV-mass (state-swap)**  = `$MU`                (FA-off prefill → state-swap → FA-on decode; mass α_a)
3. **μKV-count (state-swap)** = `$MU --gate-count`   (same, count-based α_a; count ≈0.79–0.95 so 0.70 floor rarely binds)

Anything other than these three = old/broken → removed (72 MB cleaned 2026-07-20; Prism fallback `nat_bonsai/mukv_faon_oldwd_*` kept until new data lands).

## RUNNING NOW — Bonsai/Prism WikiText (rest2, new-build armv8.7-a, new watchdog wd=1 on μKV)
Order: `mukv_faon (sol2+defrag, wd=1)` → adakv → streamingllm → h2o → tova
- snapkv, vanilla already complete (valid).
- Priority: **mukv_faon (sol2+defrag)** first — the μKV re-run with the NEW watchdog — then the 4 baselines.
- The other μKV *versions* are deferred (below); baselines run in this pass.
- Resilient: nohup benches on-device + host poll; tunnel-drop alert.

## DEFERRED — the other μKV *versions*, run later for Prism (append after the pass above)
- [ ] **μKV-mass (state-swap)** on Bonsai: `run mukv_swap 1 $MU` (wd=1)
- [ ] **μKV-count (state-swap)** on Bonsai: `run mukv_count 1 $MU --gate-count` (wd=1)  ← LAST
- [ ] **μKV-count** on Llama WikiText too — `nat_cpu` has only mukv_faon + mukv_swap (no count cell).
(rest2 is resume-safe: rerunning it later skips completed cells and does these; or run the two lines directly.)

## THEN (pipeline)
- [3] NIAH phase-A: μKV / SnapKV / vanilla × Phi/Llama/Gemma (new build; tova, canonical dropped)
- [4] GPU WikiText campaign (Adreno)
- [5] NIAH rest (adakv/h2o/tova/streamingllm) + Bonsai
- [6] Final 8-page consolidation; update Prism/Bonsai table incrementally as cells land

## Draft (figs/results consistency)
- Fig 6 = "where the heat comes from" (Llama μKV vs vanilla: CPU clk/CPU temp/DDR).
- Fig 7 = "two levers vs the throttle" (Bonsai: μKV cache lever + watchdog clock lever). Fig 8 (separate watchdog) removed/consolidated.
- Update the Bonsai/Prism table rows once the new-watchdog μKV + baselines complete.
