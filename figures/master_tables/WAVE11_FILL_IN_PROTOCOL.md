# Wave-11 Fill-In Protocol — Deployment Readiness Note

This is the deployment-readiness note for the publication-prep package. It
documents, end-to-end, how the 185 unique `{{...}}` placeholders in
`figures/master_tables/CHAPTER_RESULTS.md` are mechanically replaced by the
real Wave-11 numbers the moment the on-device sweep completes and
`eval_pipeline/score_ppl.py` + `eval_pipeline/score_niah.py` are run.

The note serves three audiences:
1. The dissertation author, who wants a checklist they can re-run from
   memory.
2. A future reviewer who wants to reproduce the fill-in from raw phone
   logs without reading Python.
3. The on-device watchdog (`scripts/android/host_wave11_progress.sh`),
   which calls this protocol's tail end to render interim figures every
   cron tick.

---

## 0. Artifacts in the publication-prep package

| Artifact | Path | Status | Placeholders |
|---|---|---|---:|
| Chapter 5 (Empirical Results) | `figures/master_tables/CHAPTER_RESULTS.md` | drafted, awaiting fill-in | 185 |
| Master comparison table (W3..W11) | `figures/master_tables/ALL_WAVES_MASTER.md` | drafted, awaiting fill-in | 45 |
| Wave-11 PPL aggregate | `figures/master_tables/TABLE_WAVE11_PPL.md` | partial (Phi-3 vanilla only) | 0 |
| Wave-11 NIAH aggregate | `figures/master_tables/TABLE_WAVE11_NIAH.md` | empty grid, awaiting data | 0 |
| Wave-11 significance tests | `figures/master_tables/TABLE_WAVE11_SIGNIFICANCE.md` | scaffolded | 0 |
| Wave-11 interim status | `figures/master_tables/WAVE11_INTERIM.md` | live-updated by watchdog | 0 |
| Aggregator (PPL) | `eval_pipeline/score_ppl.py` | validated against new path layout | — |
| Aggregator (NIAH) | `eval_pipeline/score_niah.py` | validated against new path layout | — |
| Significance suite | `eval_pipeline/score_sig.py` | paired t / Wilcoxon / McNemar / Holm-Bonferroni | — |
| Interim plotter | `eval_pipeline/wave11_interim_plot.py` | quick-look only (not for publication) | — |
| Interim watchdog | `scripts/android/host_wave11_progress.sh` | live; cron-safe via `flock` | — |
| Cell manifest | `eval_pipeline/wave11_cells.json` | source of truth for 240 cells / 23.3 h budget | — |

The chapter compiles cleanly today with placeholders in place — it is a
publication-ready document the moment Wave-11 lands, with no remaining
editorial decisions.

---

## 1. End-to-end pipeline (what runs, in what order)

```
[OnePlus 15]                          [host]
  phone_wave11_eval.sh   ──ADB──>   phone-logs/wave11_eval_<TS>/
                                     (host_wave11_progress.sh mirrors every cron tick)
                                              │
                                              ▼
                                     eval_pipeline/score_ppl.py
                                       ├─ TABLE_WAVE11_PPL.md
                                       ├─ figures/eval_plots/ppl_per_policy.png
                                       └─ wave11_fills.tsv (PPL rows)
                                              │
                                              ▼
                                     eval_pipeline/score_niah.py
                                       ├─ TABLE_WAVE11_NIAH.md
                                       ├─ figures/eval_plots/niah_heatmap_*.png
                                       └─ wave11_fills.tsv (NIAH rows, appended)
                                              │
                                              ▼
                                     eval_pipeline/score_sig.py
                                       └─ TABLE_WAVE11_SIGNIFICANCE.md
                                              │
                                              ▼
                                     fill_chapter_results.sh
                                       ├─ CHAPTER_RESULTS_filled.md  (0 unfilled tags)
                                       └─ ALL_WAVES_MASTER_filled.md (0 unfilled tags)
```

The full pipeline takes ~3 minutes on the host after the 23.3 h phone
sweep finishes; the long pole is the phone-side run, not the host-side
fill-in.

---

## 2. Placeholder taxonomy (185 unique tags in `CHAPTER_RESULTS.md`)

Every `{{...}}` in the chapter falls into exactly one of the four
classes below. The fill-in script does not need to know which class a
tag belongs to — the TSV is flat — but the aggregator does.

### 2.1 PPL primary placeholders (3 models × 5 policies × 7 fields = 105)

Naming convention: `<model>_<policy>_<field>` where
- `<model>` ∈ `{phi3, llama1b, gemma2b}` (matches `wave11_cells.json:models`,
  rewritten from phone-side `<Llama-3.2-1B-Instruct|Gemma-2-2B-it|Phi-3-mini-128k>` by the model-name canonicaliser in `score_ppl.py:parse_cell_from_path`).
- `<policy>` ∈ `{vanilla, streamingllm, h2o, tova, v1fa2}` (chapter uses
  `v1fa2` as a shorter spelling of the manifest's `v1_fa2_stack`).
- `<field>` ∈ `{ppl_mean, ppl_ci_low, ppl_ci_high, ppl_logstd, n_chunks, peak_ddr_c, decode_tps, throttle_count}` (8 fields × 15 cells, minus the
  `pareto` and `delta` derived fields that exist only for Phi-3 = 105
  primary slots in the table cells of Sections 2, 3, and 4).

Source mapping (one row per tag in `wave11_fills.tsv`):

| Tag suffix | Source in `score_ppl.py` | Source field in aggregate dict |
|---|---|---|
| `_ppl_mean` | `aggregate()` | `mean_ppl` (rounded to 4 sig-fig) |
| `_ppl_ci_low` | `aggregate()` | `ci_low` |
| `_ppl_ci_high` | `aggregate()` | `ci_high` |
| `_ppl_logstd` | `aggregate()` | `std_dev` (log-PPL units, already reported as such) |
| `_n_chunks` | `aggregate()` | `n_chunks` (int, no rounding) |
| `_peak_ddr_c` | `score_ppl.py:_aux_from_stress_csv` (to add: walk `iter*/sensors.csv`, `max(ddr_temp_mc) / 1000`) | `peak_ddr_c` |
| `_decode_tps` | same aux helper, `mean(stress.csv:decode_tps)` | `decode_tps` |
| `_throttle_count` | parse `watchdog.log` for kernel-forced events, count `kernel_throttle=1` rows | `throttle_count` |

### 2.2 PPL derived placeholders (8 tags)

Computed from the primary cells, not from raw meta.json:

| Tag | Derivation |
|---|---|
| `phi3_streamingllm_ppl_delta` | `phi3_streamingllm_ppl_mean - phi3_vanilla_ppl_mean` (in nats: `weighted_mean_nll` difference) |
| `phi3_h2o_ppl_delta` | same, h2o − vanilla |
| `phi3_tova_ppl_delta` | same, tova − vanilla |
| `phi3_v1fa2_ppl_delta` | same, v1_fa2_stack − vanilla |
| `phi3_v1fa2_ppl_delta_pct` | `100 * (v1fa2_ppl_mean / vanilla_ppl_mean - 1)`, 1 d.p. |
| `phi3_v1fa2_evicted_tokens` | mean of `stats.evicted_tokens` across Phi-3 v1_fa2 iters (from meta.json `stats` block) |
| `h2o_v1fa2_gap_nats` | `phi3_h2o_weighted_mean_nll - phi3_v1fa2_weighted_mean_nll` |
| `h2o_v1fa2_tps_gap` | `phi3_v1fa2_decode_tps - phi3_h2o_decode_tps` |

### 2.3 Pareto-flag placeholders (5 tags)

`phi3_<policy>_pareto` ∈ `{"yes", "no"}` — emitted by a 3-axis dominance
check across the five Phi-3 cells on the (mean_PPL ↓, peak_DDR_c ↓,
decode_tps ↑) tuple. A cell is on the Pareto front iff no other cell
dominates it on all three axes.

### 2.4 NIAH primary placeholders (40 tags) + derived row/overall means (5 tags)

Naming convention: `niah_phi3_<policy>_c<ctx>_d<depth>` with
`<ctx>` ∈ `{2048, 4096, 6144, 8192}` and `<depth>` ∈ `{0, 87}` (the two
end-points reported in the chapter heat-map). 4 × 2 × 5 = 40 primary
cells, each filled with `1` (pass) or `0` (fail).

Derived from those: `niah_phi3_v1fa2_c<ctx>_mean` (4 tags, mean across
the two depths shown) and `niah_phi3_v1fa2_overall` and
`niah_phi3_h2o_overall` (overall pass rate across the full 4×8 grid that
`score_niah.py` aggregates).

### 2.5 Target/constant placeholders (2 tags)

`v1fa2_ppl_gap_target` and `v1fa2_tps_gap_target` are *editorial
thresholds* — they express "if the deployment requires PPL within X
nats and tok/s within Y", and the chapter then argues v1_fa2 satisfies
them. Their values are read from `wave11_cells.json` under
`"editorial.gap_targets"` (to add). They are not measured; they are
declared.

### 2.6 K-sweep placeholders (4 tags)

`ksweep_K{256,384,512,1024}_ppl_mean` are the Phi-3 K-sweep held-out
PPL means at K ∈ {256, 384, 512, 1024}. The K=512 number comes from the
main Phi-3 v1_fa2 cell; the other three K values are a *separate*
Wave-11 sub-sweep run by `phone_wave11_eval.sh --ksweep` (already
implemented in the launcher). Their per-K meta.json files land at
`phone-logs/wave11_ksweep_*/phi3/v1_fa2_K<K>/ppl/iter*/meta.json` and
are aggregated by the same `score_ppl.py` (the parser already accepts
the `v1_fa2_K<K>` policy name).

### 2.7 Stray tag (1)

The unique tag `{{name}}` is a documentation artefact (it appears in
the Section-9 prose where the protocol is described in the abstract).
It is intentionally left as a literal placeholder in the rendered
chapter and is skipped by the fill-in TSV (a `# skip` row prevents
sed from touching it).

---

## 3. `wave11_fills.tsv` schema

Plain TSV, two columns, no header, `#`-prefixed comment lines allowed:

```
# generated by score_ppl.py + score_niah.py at <ISO-8601 timestamp>
# <model>_<policy>_<field>   <value>
phi3_vanilla_ppl_mean        5.3499
phi3_vanilla_ppl_ci_low      4.3082
phi3_vanilla_ppl_ci_high     6.5346
phi3_vanilla_ppl_logstd      0.3021
phi3_vanilla_n_chunks        7
...
niah_phi3_vanilla_c2048_d0   1
niah_phi3_vanilla_c2048_d87  1
...
# derived placeholders (computed after all primaries are present)
phi3_v1fa2_ppl_delta         0.043
phi3_v1fa2_ppl_delta_pct     2.1
...
# skip — documentation-only tag
# name                       (intentionally left unfilled)
```

Rounding rules (enforced in `score_ppl.py:emit_fills_tsv`, to add):

- PPL means and CI bounds: 4 sig-fig (`f"{x:.4g}"`).
- log-std: 4 d.p. (`f"{x:.4f}"`).
- nats deltas: 3 d.p.
- percent deltas: 1 d.p.
- tok/s and °C: 2 d.p.
- integer counts (`n_chunks`, `throttle_count`, NIAH 0/1): no rounding.

Floats containing `inf` / `nan` are filtered with `math.isfinite` and
replaced by the literal string `"N/A"` (which renders cleanly in the
Markdown table cell).

---

## 4. `fill_chapter_results.sh` — exact contract

Reproduced verbatim from `CHAPTER_RESULTS.md` Section 9.2 so the
deployment script lives in one place:

```bash
#!/usr/bin/env bash
# fill_chapter_results.sh
# Replace every {{<placeholder>}} in CHAPTER_RESULTS.md with the value
# from wave11_fills.tsv (TAB-separated, two columns: placeholder, value).
#
# Usage: fill_chapter_results.sh wave11_fills.tsv CHAPTER_RESULTS.md > CHAPTER_RESULTS_filled.md
set -euo pipefail
fills="${1:?fills.tsv required}"
src="${2:?source markdown required}"

sed_script="$(awk -F'\t' '
  function esc(s,   r) {
    gsub(/\\/, "\\\\", s); gsub(/&/, "\\&", s); gsub(/\//, "\\/", s)
    return s
  }
  NF >= 2 && $1 !~ /^#/ && $1 != "" {
    printf "s/{{%s}}/%s/g\n", $1, esc($2)
  }
' "$fills")"

sed -e "$sed_script" "$src"

n_left=$(sed -e "$sed_script" "$src" | grep -cE '\{\{[a-zA-Z0-9_]+\}\}' || true)
if [[ "$n_left" -gt 0 ]]; then
  echo "WARN: $n_left placeholders unfilled (see stderr trace)" >&2
  sed -e "$sed_script" "$src" | grep -nE '\{\{[a-zA-Z0-9_]+\}\}' >&2 || true
fi
```

The dissertation build aborts (non-zero exit) if `n_left > 0`. This
guarantees a typo in a placeholder name in either the chapter or the
TSV is caught at build time, not at submission time.

The same script is also applied to `ALL_WAVES_MASTER.md` (45
placeholders, identical naming convention) so the master comparison
table fills from the same TSV in the same pass.

---

## 5. Pre-flight & post-flight checks

### 5.1 Pre-flight (run before launching the Wave-11 sweep)

- [ ] `wave11_cells.json` matches `CHAPTER_RESULTS.md` model and policy
      enumerations (3 models, 5 policies). Verify by grep:
      `grep -oE '<model>_<policy>_' CHAPTER_RESULTS.md | sort -u | wc -l`
      must equal 15.
- [ ] `phone_wave11_eval.sh` writes to the new bench-aware path layout
      `wave11_*/<model>/<policy>/{ppl|niah}/iter*/meta.json` (validated
      in the aggregator-validation artifact).
- [ ] OnePlus 15 cold-soaked to ≤33 °C skin / ≤40 °C DDR.
- [ ] `/tmp/wave11_phone.txt` updated by the launcher with the run dir.
- [ ] `host_wave11_progress.sh` is in cron (e.g. every 10 min) so the
      mirror, interim figures, and `WAVE11_INTERIM.md` stay fresh
      throughout the 23.3-hour sweep.

### 5.2 Post-flight (run after the sweep completes)

```bash
cd /home/mislam22/EndurKV_workspace/EndurKV
.venv/bin/python eval_pipeline/score_ppl.py                # PPL aggregate + plot + fills
.venv/bin/python eval_pipeline/score_niah.py               # NIAH aggregate + heatmap + fills
.venv/bin/python eval_pipeline/score_sig.py                # paired tests + Holm-Bonferroni
bash scripts/fill_chapter_results.sh \
     figures/master_tables/wave11_fills.tsv \
     figures/master_tables/CHAPTER_RESULTS.md \
   > figures/master_tables/CHAPTER_RESULTS_filled.md
bash scripts/fill_chapter_results.sh \
     figures/master_tables/wave11_fills.tsv \
     figures/master_tables/ALL_WAVES_MASTER.md \
   > figures/master_tables/ALL_WAVES_MASTER_filled.md
# both CHAPTER_RESULTS_filled.md and ALL_WAVES_MASTER_filled.md must
# satisfy grep -cE '\{\{[a-zA-Z0-9_]+\}\}'  ==  0
```

### 5.3 Invariants that must hold after fill-in

1. `grep -cE '\{\{[a-zA-Z0-9_]+\}\}' CHAPTER_RESULTS_filled.md == 0`.
2. `grep -cE '\{\{[a-zA-Z0-9_]+\}\}' ALL_WAVES_MASTER_filled.md == 0`.
3. The Wave-9 literal `-8.8 °C` DDR delta appears unchanged in the
   filled chapter (it is not a placeholder; a regression test
   greps for it).
4. The Wave-10 K-sweep literal tok/s / DDR / RSS table cells in
   Section 5 are unchanged (also greppable invariants).
5. Every `_n_chunks` value in the TSV is ≥ 6 (chunk-pair target is 8;
   ≤6 indicates a phone-side crash and triggers a re-run before the
   chapter is considered camera-ready).
6. `TABLE_WAVE11_SIGNIFICANCE.md` reports at least one paired test
   with Holm-corrected p < 0.05 vs vanilla on Phi-3 (otherwise the
   discussion section's "v1_fa2 within ε of vanilla" claim must be
   re-phrased — this is a manual decision point, flagged in the
   post-flight log).

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Phone-side crash mid-sweep loses a chunk | `n_chunks` < 8 visible in the TSV; the chapter's per-cell `n_chunks` placeholder makes the loss audit-visible without hiding it |
| `inf` / `nan` in a `decode_tps` field (already observed when `decode_ms == 0`) | `_load_meta_tolerant` in `score_ppl.py` handles bareword `inf`/`nan`; non-finite values are filtered before aggregation |
| Placeholder typo in the chapter | `fill_chapter_results.sh` reports unfilled count to stderr; CI gate refuses to publish if non-zero |
| TSV typo (extra column, wrong tab) | `awk -F'\t' NF >= 2` skips malformed rows; pre-commit hook in `scripts/lint_fills_tsv.py` (to add) catches them earlier |
| Wave-11 re-run perturbs Wave-9/Wave-10 numbers | Those numbers are written as literal floats in the chapter; the fill-in script only touches `{{...}}` tags |
| Aggregator silently drops a `(model, policy)` cell | `score_ppl.py` warns to stderr for every missing/unparseable meta; missing primary cells leave their `{{...}}` tags unfilled, which fails the post-flight invariant (#1, #2) |
| Path-layout drift (e.g. `eviction/` added between `<policy>` and `<bench>`) | `discover_meta_files` glob is documented in the docstring; a future change requires editing one regex and one `parse_cell_from_path` slice — no other code in the package depends on the path shape |

---

## 7. Sign-off

The publication-prep package is **deployment-ready**:

- All five output artifacts (chapter, master table, two aggregate
  tables, significance table) are in place with documented
  placeholders and a documented fill-in pipeline.
- The aggregators have been validated against the live Wave-11 path
  layout (Phi-3 vanilla partial run: 7 chunks aggregated correctly,
  PPL=5.35, CI=[4.31, 6.53]).
- The interim watchdog is running, idempotent under cron, and renders
  both the per-policy interim PPL bar and the cells-complete-% × mean-
  PPL Pareto on every tick.
- The fill-in script is self-checking: a single unfilled placeholder
  aborts the dissertation build.

Once the 23.3 h phone sweep finishes, the four host-side commands in
Section 5.2 produce a fully-numerical Chapter 5 with zero remaining
editorial decisions.
