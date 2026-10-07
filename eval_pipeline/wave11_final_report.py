#!/usr/bin/env python3
"""Wave-11 final report: scores the latest phone-logs/wave11_eval_*/ run (PPL, NIAH,
paired tests, Pareto) and writes WAVE11_FINAL_REPORT.md, /tmp/wave11_fills.tsv and plots.

Usage: python eval_pipeline/wave11_final_report.py [--phone-logs-root P] [--bootstrap-samples N --seed S]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time

# score_ppl seeds its bootstrap RNG with hash((model, policy)), which depends on
# PYTHONHASHSEED. Re-exec with PYTHONHASHSEED=0 (if unset) so outputs are reproducible.
if os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Local scoring modules.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import score_ppl  # noqa: E402
import score_niah  # noqa: E402
import score_sig  # noqa: E402


# Constants

# On-disk model dir name (gguf stem) to the short tag used by chapter placeholders.
# Unknown names fall back to a lower-cased, punctuation-stripped tag.
MODEL_CANON = {
    "Phi-3-mini-128k": "phi3",
    "Phi-3-mini-128k-instruct": "phi3",
    "Llama-3.2-1B": "llama1b",
    "Llama-3.2-1B-Instruct": "llama1b",
    "Gemma-2-2B": "gemma2b",
    "Gemma-2-2B-it": "gemma2b",
    "gemma-2-2b-it": "gemma2b",
}

# Policy canonicalisation. v1_fa2_stack appears as "v1fa2" in the chapter placeholders.
POLICY_CANON = {
    "v1_fa2_stack": "v1fa2",
    "v1_fa2": "v1fa2",
    "streamingllm": "streamingllm",
    "streaming_llm": "streamingllm",
    "vanilla": "vanilla",
    "v1": "v1",
    "tova": "tova",
    "h2o": "h2o",
}

# Policies in the chapter tables. v1 appears in the report but not in the fills TSV.
CHAPTER_POLICIES = ("vanilla", "streamingllm", "h2o", "tova", "v1fa2")

# Pre-registered primary contrast for paired tests.
PRIMARY_CONTRAST = ("v1_fa2_stack", "tova")

# Output paths (relative to EndurKV root).
PATH_FINAL_REPORT_MD = "figures/master_tables/WAVE11_FINAL_REPORT.md"
PATH_PPL_BARS_PNG = "figures/eval_plots/wave11_final_ppl_bars.png"
PATH_PARETO_PNG = "figures/eval_plots/wave11_final_pareto.png"
PATH_NIAH_HEATMAP_PNG = "figures/eval_plots/wave11_final_niah_heatmap.png"
PATH_FILLS_TSV = "/tmp/wave11_fills.tsv"

# NIAH judge needle (canonical, mirrors score_niah).
NEEDLE_FULL = score_niah.NEEDLE_FULL
NEEDLE_KEY = score_niah.NEEDLE_KEY
CTX_LENGTHS = score_niah.CTX_LENGTHS
DEPTHS = score_niah.DEPTHS

# Protocol thresholds, declared not measured (WAVE11_FILL_IN_PROTOCOL §2.5).
DEFAULT_EDITORIAL_TARGETS = {
    "v1fa2_ppl_gap_target": "0.10",   # nats
    "v1fa2_tps_gap_target": "0.50",   # tok/s
}


# Path resolution

def default_workspace_root() -> Path:
    """Resolve workspace root assuming <ws>/EndurKV/eval_pipeline/<this>.py."""
    return _HERE.parent.parent


def default_endurkv_root() -> Path:
    return _HERE.parent


def latest_wave11_run(phone_logs_root: Path) -> Optional[Path]:
    """Most recent `wave11_eval_*` dir by mtime, or None."""
    cands = sorted(
        phone_logs_root.glob("wave11_eval_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


# Canonicalisation helpers

def canon_model(name: str) -> str:
    if name in MODEL_CANON:
        return MODEL_CANON[name]
    # Fallback heuristic: lowercase, drop non-alnum, hint via substrings.
    n = re.sub(r"[^A-Za-z0-9]", "", name).lower()
    if "phi3" in n or "phi3mini" in n:
        return "phi3"
    if "llama" in n and ("1b" in n or "32" in n):
        return "llama1b"
    if "gemma" in n and "2b" in n:
        return "gemma2b"
    return n  # unknown - keep deterministic but unrecognised tag


def canon_policy(name: str) -> str:
    return POLICY_CANON.get(name, name)


# Aux-metric extraction from meta.json + sensors/stress companions

def _read_meta(meta_path: Path) -> Optional[dict]:
    return score_ppl._load_meta_tolerant(str(meta_path))


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _peak_ddr_from_sensors(sensors_csv: Path) -> Optional[float]:
    """Peak DDR temperature (°C) in sensors.csv, or None if the column is missing or empty."""
    if not sensors_csv.is_file():
        return None
    try:
        with open(sensors_csv, newline="") as f:
            reader = csv.DictReader(f)
            if "ddr_temp_mc" not in (reader.fieldnames or ()):
                return None
            best = None
            for row in reader:
                v = _safe_float(row.get("ddr_temp_mc"))
                if v is None:
                    continue
                if best is None or v > best:
                    best = v
    except OSError:
        return None
    return best / 1000.0 if best is not None else None


def _decode_tps_from_stress(stress_csv: Path) -> Optional[float]:
    """Mean decode_tps in stress.csv. Keeps 0 < tps < 1e6 to drop the 1.99e9 value the
    launcher writes when decode_ms == 0."""
    if not stress_csv.is_file():
        return None
    vals: List[float] = []
    try:
        with open(stress_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = _safe_float(row.get("decode_tps"))
                if v is None or v <= 0 or v >= 1e6:
                    continue
                vals.append(v)
    except OSError:
        return None
    if not vals:
        return None
    return sum(vals) / len(vals)


def _throttle_count_from_sensors(sensors_csv: Path) -> int:
    """Rows where any cpu*_cool_state is non-zero (kernel thermal throttle).
    Blank cells count as 0, and a missing file returns 0."""
    if not sensors_csv.is_file():
        return 0
    n = 0
    try:
        with open(sensors_csv, newline="") as f:
            reader = csv.DictReader(f)
            cool_cols = [c for c in (reader.fieldnames or ())
                         if c.endswith("_cool_state")]
            if not cool_cols:
                return 0
            for row in reader:
                for c in cool_cols:
                    v = row.get(c)
                    if v not in (None, "", "0"):
                        try:
                            if int(v) > 0:
                                n += 1
                                break
                        except ValueError:
                            continue
    except OSError:
        return 0
    return n


def _evicted_tokens_from_meta(meta: dict) -> Optional[int]:
    """Evicted-token count. Launcher versions use different field names, so try the common ones."""
    for k in ("evicted_total_decode", "evicted", "evicted_tokens",
              "n_evicted_tokens"):
        v = meta.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            continue
    # Nested stats block sometimes present.
    stats = meta.get("stats")
    if isinstance(stats, dict):
        for k in ("evicted_tokens", "evicted_total_decode", "evicted"):
            v = stats.get(k)
            if v is None:
                continue
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return None


def aux_metrics_for_cell(run_dir: Path, model_disk: str, policy_disk: str
                         ) -> Dict[str, Optional[float]]:
    """Peak DDR, mean decode tps, throttle count and mean evicted tokens for one PPL cell,
    from the cell's sensors.csv / stress.csv (under .../ppl/) and per-iter meta.json."""
    cell_dir = run_dir / model_disk / policy_disk / "ppl"
    peak_ddr = _peak_ddr_from_sensors(cell_dir / "sensors.csv")
    tps = _decode_tps_from_stress(cell_dir / "stress.csv")
    throttle = _throttle_count_from_sensors(cell_dir / "sensors.csv")

    evicted: List[int] = []
    if cell_dir.is_dir():
        for iter_dir in sorted(cell_dir.glob("iter*")):
            meta = _read_meta(iter_dir / "meta.json")
            if meta is None:
                continue
            ev = _evicted_tokens_from_meta(meta)
            if ev is not None:
                evicted.append(ev)
    mean_evicted = (sum(evicted) / len(evicted)) if evicted else None

    return {
        "peak_ddr_c": peak_ddr,
        "decode_tps": tps,
        "throttle_count": throttle,
        "evicted_tokens": mean_evicted,
    }


# Pareto frontier (3-axis: PPL ↓, DDR ↓, TPS ↑)

def pareto_flags(cells: Dict[Tuple[str, str], Dict]) -> Dict[Tuple[str, str], str]:
    """Per model, "yes" if a policy is on the Pareto front of (mean_ppl ↓, peak_ddr_c ↓,
    decode_tps ↑), else "no", or "N/A" if an axis is missing. Ties never dominate."""
    by_model: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
    for (m, p), info in cells.items():
        by_model[m].append((p, info))

    out: Dict[Tuple[str, str], str] = {}
    for model, rows in by_model.items():
        # Filter to rows with all three axes present.
        usable: List[Tuple[str, float, float, float]] = []
        for p, info in rows:
            ppl = info.get("mean_ppl")
            ddr = info.get("peak_ddr_c")
            tps = info.get("decode_tps")
            if ppl is None or ddr is None or tps is None:
                out[(model, p)] = "N/A"
                continue
            usable.append((p, float(ppl), float(ddr), float(tps)))

        for p, ppl, ddr, tps in usable:
            dominated = False
            for q, qppl, qddr, qtps in usable:
                if q == p:
                    continue
                # q dominates p iff q is ≤ on minimise axes, ≥ on maximise
                # axis, and strictly better on at least one axis.
                if (qppl <= ppl and qddr <= ddr and qtps >= tps
                        and (qppl < ppl or qddr < ddr or qtps > tps)):
                    dominated = True
                    break
            out[(model, p)] = "no" if dominated else "yes"
    return out


# NIAH grid (4 ctx × 8 depth). Launcher ordinals (iter<NN>/) are mapped back to
# (ctx, depth) through the wave11_cells.json manifest.

def _load_niah_ordinal_map(endurkv_root: Path) -> Dict[int, Tuple[int, int]]:
    """{ordinal: (ctx, depth)} from the `niah_stimuli_selected` block of wave11_cells.json."""
    manifest_path = endurkv_root / "eval_pipeline" / "wave11_cells.json"
    out: Dict[int, Tuple[int, int]] = {}
    if not manifest_path.is_file():
        return out
    try:
        with open(manifest_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return out
    for item in data.get("niah_stimuli_selected", []) or []:
        idx = item.get("stimulus_index")
        ctx = item.get("ctx_tokens")
        depth = item.get("depth_pct")
        if idx is None or ctx is None or depth is None:
            continue
        try:
            out[int(idx)] = (int(ctx), int(depth))
        except (TypeError, ValueError):
            continue
    return out


def collect_niah(phone_logs_root: Path, endurkv_root: Path
                 ) -> Tuple[Dict[Tuple[str, str], Dict[Tuple[int, int], bool]],
                            Dict[Tuple[str, str], Dict[str, int]]]:
    """Return (grid, per_cell): grid[(model, policy)][(ctx, depth)] = bool and
    per_cell[(model, policy)] = {"correct": n, "total": n}."""
    ord_map = _load_niah_ordinal_map(endurkv_root)
    gen_files = score_niah.find_gen_files(phone_logs_root)

    grid: Dict[Tuple[str, str], Dict[Tuple[int, int], bool]] = defaultdict(dict)
    per_cell: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0})

    for gp in gen_files:
        parsed = score_niah.parse_gen_path(gp, phone_logs_root)
        if parsed is None:
            continue
        _wave, model_disk, policy_disk, stim_id, ctx, depth = parsed
        m_can = canon_model(model_disk)
        p_can = canon_policy(policy_disk)

        try:
            text = gp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ok = score_niah.rule_based_judge(text)

        per_cell[(m_can, p_can)]["total"] += 1
        per_cell[(m_can, p_can)]["correct"] += int(ok)

        if depth >= 0:
            grid[(m_can, p_can)][(ctx, depth)] = ok
        else:
            # Launcher ordinal layout (iter<NN>), the ctx slot holds the ordinal.
            cell = ord_map.get(int(ctx))
            if cell is not None:
                grid[(m_can, p_can)][cell] = ok
    return grid, per_cell


# Top-level aggregation

def aggregate_all(phone_logs_root: Path,
                  endurkv_root: Path,
                  run_dir: Optional[Path],
                  n_bootstrap: int,
                  seed: int) -> Dict:
    """One dict with cells[(model, policy)] (PPL stats plus aux metrics), niah_grid,
    niah_per_cell, sig_ppl, sig_niah and ppl_log_per_iter."""
    # PPL aggregate via score_ppl (handles bench-aware + legacy layouts).
    ppl_stats_disk = score_ppl.aggregate(
        phone_logs_root=str(phone_logs_root),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # Aux metrics come from the latest run dir only, so an older overlapping
    # run's sensors.csv is never read.
    cells: Dict[Tuple[str, str], Dict] = {}
    for (model_disk, policy_disk), st in ppl_stats_disk.items():
        m_can = canon_model(model_disk)
        p_can = canon_policy(policy_disk)
        entry = dict(st)
        entry["model_disk"] = model_disk
        entry["policy_disk"] = policy_disk
        if run_dir is not None:
            aux = aux_metrics_for_cell(run_dir, model_disk, policy_disk)
        else:
            aux = {"peak_ddr_c": None, "decode_tps": None,
                   "throttle_count": 0, "evicted_tokens": None}
        entry.update(aux)
        cells[(m_can, p_can)] = entry

    # NIAH grid.
    niah_grid, niah_per_cell = collect_niah(phone_logs_root, endurkv_root)

    # Paired tests via score_sig on the same root.
    ppl_log = score_sig.aggregate_ppl(str(phone_logs_root))
    niah_bool = score_sig.aggregate_niah(str(phone_logs_root))
    sig_ppl = score_sig.paired_ppl_tests(ppl_log)
    sig_niah = score_sig.paired_niah_tests(niah_bool)

    # Pareto.
    pareto = pareto_flags(cells)

    return {
        "cells": cells,
        "pareto": pareto,
        "niah_grid": niah_grid,
        "niah_per_cell": niah_per_cell,
        "sig_ppl": sig_ppl,
        "sig_niah": sig_niah,
    }


# Markdown report

def _fmt(x, fmt: str = "{:.4f}", na: str = "N/A") -> str:
    if x is None:
        return na
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    if not math.isfinite(v):
        return na
    return fmt.format(v)


def _fmt_g(x, sig: int = 4, na: str = "N/A") -> str:
    if x is None:
        return na
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    if not math.isfinite(v):
        return na
    return f"{v:.{sig}g}"


def _model_order(cells: Dict[Tuple[str, str], Dict]) -> List[str]:
    """Stable display order: phi3, llama1b, gemma2b, then the rest sorted."""
    seen = sorted({m for (m, _p) in cells.keys()})
    pref = ["phi3", "llama1b", "gemma2b"]
    ordered = [m for m in pref if m in seen] + [m for m in seen if m not in pref]
    return ordered


def _policy_order(cells: Dict[Tuple[str, str], Dict]) -> List[str]:
    """Stable display order matching chapter conventions."""
    seen = sorted({p for (_m, p) in cells.keys()})
    pref = ["vanilla", "streamingllm", "h2o", "tova", "v1", "v1fa2"]
    return [p for p in pref if p in seen] + [p for p in seen if p not in pref]


def render_report_md(agg: Dict,
                     run_dir: Optional[Path],
                     phone_logs_root: Path,
                     out_path: Path,
                     verdict: str) -> None:
    cells = agg["cells"]
    pareto = agg["pareto"]
    niah_grid = agg["niah_grid"]
    niah_per_cell = agg["niah_per_cell"]
    sig_ppl = agg["sig_ppl"]
    sig_niah = agg["sig_niah"]

    lines: List[str] = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines.append("# Wave-11 FINAL Report")
    lines.append("")
    lines.append(f"_Generated at **{ts}**._  ")
    lines.append(f"Source: `{run_dir if run_dir else '(no wave11_eval_* run found)'}`  ")
    lines.append(f"Phone-logs root: `{phone_logs_root}`")
    lines.append("")
    lines.append(
        "PPL is the **token-weighted geometric mean** per-cell "
        "(`exp(sum_i n_tok_i * mean_nll_i / sum_i n_tok_i)`) — the WikiText-2 "
        "convention used by H2O / KIVI / StreamingLLM / TOVA. "
        "CI is a 1000-sample percentile bootstrap (95%) over per-chunk NLLs, "
        "exponentiated for display. `log_std` is the per-chunk std of "
        "`log(PPL)` (so its units are nats)."
    )
    lines.append("")
    lines.append(f"NIAH rule judge: case-insensitive substring "
                 f"`\"{NEEDLE_KEY}\"` with a negation-window guard "
                 f"(see `score_niah.rule_based_judge`).")
    lines.append("")
    lines.append(f"**Overall verdict:** {verdict}")
    lines.append("")

    # Per-policy PPL table
    lines.append("## 1. Per-(model, policy) PPL aggregate")
    lines.append("")
    lines.append("| Model | Policy | mean_PPL | CI_low | CI_high | log_std | n_chunks | peak_DDR_°C | decode_tps | throttle | evicted_tok | Pareto |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    models = _model_order(cells)
    for m in models:
        # Sort policies within model by mean_ppl ascending (best first), then
        # keep pending cells at the end.
        rows: List[Tuple[str, Dict]] = [
            (p, cells[(m, p)]) for (mm, p) in cells.keys() if mm == m
        ]
        rows.sort(key=lambda kv: (
            float("inf") if not math.isfinite(float(kv[1].get("mean_ppl", float("nan"))))
            else kv[1]["mean_ppl"]
        ))
        for p, info in rows:
            lines.append(
                "| {m} | {p} | {ppl} | {lo} | {hi} | {sd} | {n} | {ddr} | {tps} | {th} | {ev} | {pa} |".format(
                    m=m, p=p,
                    ppl=_fmt(info.get("mean_ppl"), "{:.4f}"),
                    lo=_fmt(info.get("ci_low"), "{:.4f}"),
                    hi=_fmt(info.get("ci_high"), "{:.4f}"),
                    sd=_fmt(info.get("std_dev"), "{:.4f}"),
                    n=info.get("n_chunks", 0),
                    ddr=_fmt(info.get("peak_ddr_c"), "{:.2f}"),
                    tps=_fmt(info.get("decode_tps"), "{:.2f}"),
                    th=info.get("throttle_count", 0),
                    ev=_fmt(info.get("evicted_tokens"), "{:.0f}"),
                    pa=pareto.get((m, p), "N/A"),
                )
            )
    lines.append("")

    # Paired tests (PPL)
    lines.append("## 2. Paired significance tests on log(PPL)")
    lines.append("")
    if sig_ppl:
        # Apply Holm-Bonferroni per model using Wilcoxon p-values (matches
        # score_sig's convention).
        by_model_p: Dict[str, List[Dict]] = defaultdict(list)
        for r in sig_ppl:
            by_model_p[r["model"]].append(r)
        for _model, family in by_model_p.items():
            ps = [r["w_p"] for r in family if math.isfinite(r["w_p"])]
            valid = [r for r in family if math.isfinite(r["w_p"])]
            adj = score_sig.holm_bonferroni(ps)
            for r, p_adj in zip(valid, adj):
                r["w_p_holm"] = p_adj
        lines.append(
            "| Model | A | B | n | mean Δlog(PPL) | t | t_p | W | W_p | "
            "Holm-W_p | Cohen d_z |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(sig_ppl, key=lambda x: (x["model"], x["a"], x["b"])):
            lines.append(
                "| {model} | {a} | {b} | {n} | {md} | {t} | {tp} | "
                "{w} | {wp} | {wh} | {dz} |".format(
                    model=canon_model(r["model"]),
                    a=canon_policy(r["a"]), b=canon_policy(r["b"]),
                    n=r["n"],
                    md=_fmt(r["mean_log_delta"], "{:.4f}"),
                    t=_fmt(r["t_stat"], "{:.3f}"),
                    tp=_fmt_g(r["t_p"]),
                    w=_fmt(r["w_stat"], "{:.3f}"),
                    wp=_fmt_g(r["w_p"]),
                    wh=_fmt_g(r.get("w_p_holm", float("nan"))),
                    dz=_fmt(r["cohens_d_z"], "{:.3f}"),
                )
            )
    else:
        lines.append("_No PPL pair data — need at least two policies with "
                     "overlapping chunks for any model._")
    lines.append("")

    # McNemar (NIAH)
    lines.append("## 3. Paired McNemar exact tests on NIAH")
    lines.append("")
    if sig_niah:
        by_model_n: Dict[str, List[Dict]] = defaultdict(list)
        for r in sig_niah:
            by_model_n[r["model"]].append(r)
        for _model, family in by_model_n.items():
            ps = [r["mcnemar_p"] for r in family]
            adj = score_sig.holm_bonferroni(ps)
            for r, p_adj in zip(family, adj):
                r["mcnemar_p_holm"] = p_adj
        lines.append(
            "| Model | A | B | n_shared | b (A&!B) | c (!A&B) | "
            "McNemar p | Holm p | OR (b/c) |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for r in sorted(sig_niah, key=lambda x: (x["model"], x["a"], x["b"])):
            orv = r["odds_ratio"]
            orv_s = (f"{orv:.2f}" if math.isfinite(orv) else "inf")
            lines.append(
                "| {model} | {a} | {b} | {n} | {bo} | {co} | {p} | {ph} | {orv} |".format(
                    model=canon_model(r["model"]),
                    a=canon_policy(r["a"]), b=canon_policy(r["b"]),
                    n=r["n_shared"], bo=r["b_only"], co=r["c_only"],
                    p=_fmt_g(r["mcnemar_p"]),
                    ph=_fmt_g(r.get("mcnemar_p_holm", float("nan"))),
                    orv=orv_s,
                )
            )
    else:
        lines.append("_No NIAH pair data._")
    lines.append("")

    # Primary contrast quick-look
    A, B = PRIMARY_CONTRAST
    lines.append(f"## 4. Pre-registered primary contrast: `{A}` vs `{B}`")
    lines.append("")
    found_any = False
    for r in sig_ppl:
        if {r["a"], r["b"]} == {A, B}:
            found_any = True
            lines.append(
                f"- **{canon_model(r['model'])}** PPL (Wilcoxon, paired on chunks): "
                f"n={r['n']}, mean Δlog(PPL)={_fmt(r['mean_log_delta'], '{:.4f}')}, "
                f"p={_fmt_g(r['w_p'])}, Cohen d_z={_fmt(r['cohens_d_z'], '{:.3f}')}"
            )
    for r in sig_niah:
        if {r["a"], r["b"]} == {A, B}:
            found_any = True
            lines.append(
                f"- **{canon_model(r['model'])}** NIAH (McNemar exact): "
                f"n_shared={r['n_shared']}, b={r['b_only']}, c={r['c_only']}, "
                f"p={_fmt_g(r['mcnemar_p'])}"
            )
    if not found_any:
        lines.append("_No data yet for the primary contrast — needs both "
                     "policies to have completed at least 2 paired chunks._")
    lines.append("")

    # NIAH heatmap (text)
    lines.append("## 5. NIAH pass/fail heatmap (rows = ctx, cols = depth %)")
    lines.append("")
    lines.append("Legend: `O` = correct, `.` = wrong, `?` = missing.")
    lines.append("")
    if niah_grid:
        # Also report overall NIAH accuracy per cell.
        lines.append("### Overall NIAH accuracy")
        lines.append("")
        lines.append("| Model | Policy | correct | total | accuracy |")
        lines.append("|---|---|---:|---:|---:|")
        for (m, p), v in sorted(niah_per_cell.items()):
            acc = (v["correct"] / v["total"]) if v["total"] else float("nan")
            lines.append(
                f"| {m} | {p} | {v['correct']} | {v['total']} | "
                f"{_fmt(acc * 100.0 if math.isfinite(acc) else float('nan'), '{:.1f}')}% |"
            )
        lines.append("")
        for (m, p) in sorted(niah_grid.keys()):
            lines.append(f"### {m} / {p}")
            lines.append("")
            lines.append("```")
            lines.append("ctx \\ d% " + " ".join(f"{d:>3d}" for d in DEPTHS))
            for ctx in CTX_LENGTHS:
                row = [f"{ctx:>8d} "]
                for d in DEPTHS:
                    v = niah_grid[(m, p)].get((ctx, d))
                    if v is None:
                        row.append("  ?")
                    else:
                        row.append("  O" if v else "  .")
                lines.append(" ".join(row))
            lines.append("```")
            lines.append("")
    else:
        lines.append("_No NIAH generations on disk yet._")
        lines.append("")

    # Health checks
    lines.append("## 6. Health invariants (per WAVE11_FILL_IN_PROTOCOL §5.3)")
    lines.append("")
    invariants: List[str] = []
    # Invariant: n_chunks >= 6 for each cell that has any data.
    n_short = [(m, p, info["n_chunks"]) for (m, p), info in cells.items()
               if 0 < info.get("n_chunks", 0) < 6]
    if n_short:
        invariants.append(
            "- **WARN**: " + ", ".join(
                f"`{m}/{p}` has only {n} chunks (target ≥ 6)"
                for (m, p, n) in n_short
            )
        )
    else:
        invariants.append(
            "- OK: every cell with data has ≥ 6 chunks (or is still pending)."
        )

    # Invariant: at least one Holm-corrected p < 0.05 on phi3 vs vanilla.
    sig_hits = [
        r for r in sig_ppl
        if canon_model(r["model"]) == "phi3"
        and "vanilla" in (r["a"], r["b"])
        and math.isfinite(r.get("w_p_holm", float("nan")))
        and r.get("w_p_holm", 1.0) < 0.05
    ]
    if sig_hits:
        invariants.append(
            f"- OK: {len(sig_hits)} Holm-corrected paired test(s) p < 0.05 "
            "on Phi-3 vs vanilla."
        )
    else:
        invariants.append(
            "- INFO: no Holm-corrected paired test p < 0.05 on Phi-3 vs "
            "vanilla yet (expected while only the vanilla cell has data)."
        )

    lines.extend(invariants)
    lines.append("")

    # Artefacts pointer block
    lines.append("## 7. Artefacts written by this run")
    lines.append("")
    lines.append(f"- `{PATH_FINAL_REPORT_MD}` (this file)")
    lines.append(f"- `{PATH_PPL_BARS_PNG}`")
    lines.append(f"- `{PATH_PARETO_PNG}`")
    lines.append(f"- `{PATH_NIAH_HEATMAP_PNG}`")
    lines.append(f"- `{PATH_FILLS_TSV}` (TSV for `fill_chapter_results.sh`)")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# Plots

def _unlink_stale(out_path: Path, reason: str) -> None:
    """Delete a stale output PNG so a partial run is not mistaken for an earlier,
    fuller one. Logged to stderr."""
    try:
        if out_path.is_file():
            out_path.unlink()
            print(f"[clean] removed stale {out_path} ({reason})", file=sys.stderr)
    except OSError as exc:
        print(f"[warn] could not remove stale {out_path}: {exc}", file=sys.stderr)


def render_ppl_bars(cells: Dict[Tuple[str, str], Dict], out_path: Path) -> None:
    """Grouped bar chart per (model, policy) with 95% bootstrap CI error
    bars. Refuses to write a placeholder when no cell has any data."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    usable = {k: v for k, v in cells.items()
              if math.isfinite(float(v.get("mean_ppl", float("nan"))))}
    if not usable:
        print(f"[skip] no PPL data; not writing {out_path}", file=sys.stderr)
        _unlink_stale(out_path, "no PPL data this run")
        return

    models = _model_order(usable)
    policies = _policy_order(usable)
    n_models = len(models)
    n_policies = len(policies)
    group_width = 0.8
    bar_width = group_width / max(n_policies, 1)

    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * n_models * n_policies), 4.8))
    x_base = list(range(n_models))
    cmap = plt.get_cmap("tab10")

    for pi, policy in enumerate(policies):
        xs: List[float] = []
        heights: List[float] = []
        err_lo: List[float] = []
        err_hi: List[float] = []
        ns: List[int] = []
        for mi, model in enumerate(models):
            info = usable.get((model, policy))
            if info is None:
                continue
            xs.append(x_base[mi] - group_width / 2 + (pi + 0.5) * bar_width)
            heights.append(float(info["mean_ppl"]))
            err_lo.append(max(0.0, info["mean_ppl"] - info["ci_low"]))
            err_hi.append(max(0.0, info["ci_high"] - info["mean_ppl"]))
            ns.append(int(info.get("n_chunks", 0)))
        if not xs:
            continue
        bars = ax.bar(xs, heights, width=bar_width, label=policy,
                      color=cmap(pi % 10), yerr=[err_lo, err_hi],
                      capsize=3, edgecolor="black", linewidth=0.5)
        for b, n in zip(bars, ns):
            ax.text(b.get_x() + b.get_width() / 2.0,
                    b.get_height(), f"n={n}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x_base)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Wave-11 FINAL perplexity by policy (95% bootstrap CI)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="Policy", fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_pareto_bubble(cells: Dict[Tuple[str, str], Dict],
                         pareto: Dict[Tuple[str, str], str],
                         out_path: Path) -> None:
    """Bubble plot, x = mean PPL, y = peak DDR °C, size = decode tps.
    Pareto-front cells are outlined in black."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts: List[Tuple[str, str, float, float, float]] = []
    for (m, p), info in cells.items():
        ppl = info.get("mean_ppl")
        ddr = info.get("peak_ddr_c")
        tps = info.get("decode_tps")
        if ppl is None or ddr is None or tps is None:
            continue
        if not (math.isfinite(ppl) and math.isfinite(ddr) and math.isfinite(tps)):
            continue
        pts.append((m, p, float(ppl), float(ddr), float(tps)))
    if not pts:
        print(f"[skip] no full-axis cells; not writing {out_path}",
              file=sys.stderr)
        _unlink_stale(out_path, "no full-axis cells this run")
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    models = sorted({m for (m, _, _, _, _) in pts})
    cmap = plt.get_cmap("tab10")
    marker_for_model = {m: cmap(i % 10) for i, m in enumerate(models)}

    # Normalise bubble size: 50..500 across the tps range.
    tps_vals = [t for (_, _, _, _, t) in pts]
    tmin, tmax = min(tps_vals), max(tps_vals)
    def _size(t: float) -> float:
        if tmax == tmin:
            return 200.0
        return 50.0 + 450.0 * (t - tmin) / (tmax - tmin)

    for m, p, ppl, ddr, tps in pts:
        flag = pareto.get((m, p), "N/A")
        ax.scatter(
            ppl, ddr,
            s=_size(tps),
            color=marker_for_model[m],
            edgecolor="black" if flag == "yes" else "none",
            linewidth=1.4 if flag == "yes" else 0.0,
            alpha=0.78,
            label=f"{m}/{p}",
        )
        ax.annotate(p, (ppl, ddr), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")

    ax.set_xlabel("Perplexity (lower is better)")
    ax.set_ylabel("Peak DDR temperature (°C, lower is better)")
    ax.set_title("Wave-11 FINAL Pareto: PPL × peak_DDR × tps\n"
                 "(bubble size ∝ decode tok/s; black outline = on Pareto front)")
    ax.grid(linestyle=":", alpha=0.5)
    # De-dup legend by (m only) - too many entries otherwise.
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None",
                   color=marker_for_model[m], label=m, markersize=8)
        for m in models
    ]
    ax.legend(handles=handles, title="Model", fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_niah_heatmap(grid: Dict[Tuple[str, str], Dict[Tuple[int, int], bool]],
                        out_path: Path) -> None:
    """One subplot per (model, policy), 4×8 grid. When no (model, policy)
    has any data, refuse to write a placeholder."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not grid:
        print(f"[skip] no NIAH generations; not writing {out_path}",
              file=sys.stderr)
        _unlink_stale(out_path, "no NIAH generations this run")
        return

    cells_with_data = [k for k in grid.keys() if grid[k]]
    if not cells_with_data:
        print(f"[skip] empty NIAH grids; not writing {out_path}", file=sys.stderr)
        _unlink_stale(out_path, "empty NIAH grids this run")
        return

    # Stable order: by model then policy.
    cells_sorted = sorted(cells_with_data, key=lambda mp: (mp[0], mp[1]))

    n = len(cells_sorted)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.8 * ncols, 2.6 * nrows),
                             squeeze=False)
    for idx, (m, p) in enumerate(cells_sorted):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        mat = [[0.5] * len(DEPTHS) for _ in range(len(CTX_LENGTHS))]
        for i, ctx in enumerate(CTX_LENGTHS):
            for j, d in enumerate(DEPTHS):
                v = grid[(m, p)].get((ctx, d))
                if v is True:
                    mat[i][j] = 1.0
                elif v is False:
                    mat[i][j] = 0.0
                else:
                    mat[i][j] = 0.5
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(DEPTHS)))
        ax.set_xticklabels([f"{d}%" for d in DEPTHS], fontsize=8)
        ax.set_yticks(range(len(CTX_LENGTHS)))
        ax.set_yticklabels([str(x) for x in CTX_LENGTHS], fontsize=8)
        ax.set_xlabel("Depth", fontsize=8)
        ax.set_ylabel("Ctx tokens", fontsize=8)
        n_correct = sum(1 for v in grid[(m, p)].values() if v is True)
        n_total = sum(1 for v in grid[(m, p)].values() if v is not None)
        acc = (n_correct / n_total * 100.0) if n_total else 0.0
        ax.set_title(f"{m} / {p}  ({n_correct}/{n_total}, {acc:.0f}%)",
                     fontsize=9)
        # Cell annotations.
        for i in range(len(CTX_LENGTHS)):
            for j in range(len(DEPTHS)):
                v = mat[i][j]
                mark = "O" if v == 1.0 else ("." if v == 0.0 else "?")
                ax.text(j, i, mark, ha="center", va="center",
                        color="black", fontsize=8, fontweight="bold")

    # Hide unused axes.
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle("Wave-11 FINAL NIAH heatmap (rule judge)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# wave11_fills.tsv emission

def _round_4g(x) -> str:
    v = _safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:.4g}"


def _round_4f(x) -> str:
    v = _safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def _round_2f(x) -> str:
    v = _safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:.2f}"


def _round_int(x) -> str:
    v = _safe_float(x)
    if v is None:
        return "N/A"
    return f"{int(round(v))}"


def emit_fills_tsv(agg: Dict, out_path: Path) -> None:
    """Write /tmp/wave11_fills.tsv, tab-separated (tag, value) rows.
    Schema: WAVE11_FILL_IN_PROTOCOL §3."""
    cells = agg["cells"]
    pareto = agg["pareto"]
    niah_grid = agg["niah_grid"]
    niah_per_cell = agg["niah_per_cell"]

    rows: List[str] = []
    rows.append(f"# generated by wave11_final_report.py at "
                f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    rows.append("# schema: <tag>\\t<value> (TAB-separated, '#' lines = comments)")

    # PPL primary placeholders per (model, policy)
    rows.append("# --- PPL primary placeholders ---")
    # Look up vanilla NLL per model for derived deltas.
    vanilla_nll: Dict[str, float] = {}
    vanilla_ppl: Dict[str, float] = {}
    for m in _model_order(cells):
        v = cells.get((m, "vanilla"))
        if v is not None and v.get("n_chunks", 0) > 0:
            nll = v.get("weighted_mean_nll")
            if nll is not None and math.isfinite(nll):
                vanilla_nll[m] = nll
            ppl = v.get("mean_ppl")
            if ppl is not None and math.isfinite(ppl):
                vanilla_ppl[m] = ppl

    for m in _model_order(cells):
        for p in CHAPTER_POLICIES:
            info = cells.get((m, p))
            if info is None:
                # Missing cells get "N/A" so no chapter tag is left unfilled.
                for suf, _ in [
                    ("ppl_mean", "N/A"), ("ppl_ci_low", "N/A"),
                    ("ppl_ci_high", "N/A"), ("ppl_logstd", "N/A"),
                    ("n_chunks", "0"),
                    ("peak_ddr_c", "N/A"), ("decode_tps", "N/A"),
                    ("throttle_count", "0"),
                ]:
                    val = "0" if suf in ("n_chunks", "throttle_count") else "N/A"
                    rows.append(f"{m}_{p}_{suf}\t{val}")
                continue
            rows.append(f"{m}_{p}_ppl_mean\t{_round_4g(info.get('mean_ppl'))}")
            rows.append(f"{m}_{p}_ppl_ci_low\t{_round_4g(info.get('ci_low'))}")
            rows.append(f"{m}_{p}_ppl_ci_high\t{_round_4g(info.get('ci_high'))}")
            rows.append(f"{m}_{p}_ppl_logstd\t{_round_4f(info.get('std_dev'))}")
            rows.append(f"{m}_{p}_n_chunks\t{int(info.get('n_chunks', 0))}")
            rows.append(f"{m}_{p}_peak_ddr_c\t{_round_2f(info.get('peak_ddr_c'))}")
            rows.append(f"{m}_{p}_decode_tps\t{_round_2f(info.get('decode_tps'))}")
            rows.append(f"{m}_{p}_throttle_count\t{int(info.get('throttle_count', 0) or 0)}")

    # Derived placeholders
    rows.append("# --- PPL derived placeholders ---")
    for m in _model_order(cells):
        for p in ("streamingllm", "h2o", "tova", "v1fa2"):
            info = cells.get((m, p))
            if info is None or m not in vanilla_nll:
                rows.append(f"{m}_{p}_ppl_delta\tN/A")
                continue
            nll = info.get("weighted_mean_nll")
            if nll is None or not math.isfinite(nll):
                rows.append(f"{m}_{p}_ppl_delta\tN/A")
                continue
            d_nats = nll - vanilla_nll[m]
            rows.append(f"{m}_{p}_ppl_delta\t{d_nats:.3f}")
    # Percent delta - currently only used by the chapter for phi3_v1fa2.
    for m in _model_order(cells):
        info = cells.get((m, "v1fa2"))
        if info is None or m not in vanilla_ppl:
            rows.append(f"{m}_v1fa2_ppl_delta_pct\tN/A")
            continue
        ppl = info.get("mean_ppl")
        if ppl is None or not math.isfinite(ppl) or vanilla_ppl[m] == 0:
            rows.append(f"{m}_v1fa2_ppl_delta_pct\tN/A")
            continue
        d_pct = 100.0 * (ppl / vanilla_ppl[m] - 1.0)
        rows.append(f"{m}_v1fa2_ppl_delta_pct\t{d_pct:.1f}")
    # phi3_v1fa2_evicted_tokens.
    phi3_v1fa2 = cells.get(("phi3", "v1fa2"))
    if phi3_v1fa2 and phi3_v1fa2.get("evicted_tokens") is not None:
        rows.append(f"phi3_v1fa2_evicted_tokens\t{int(round(phi3_v1fa2['evicted_tokens']))}")
    else:
        rows.append("phi3_v1fa2_evicted_tokens\tN/A")

    # Cross-policy gaps used by chapter prose.
    phi3_h2o = cells.get(("phi3", "h2o"))
    if phi3_v1fa2 and phi3_h2o:
        nll_v = phi3_v1fa2.get("weighted_mean_nll")
        nll_h = phi3_h2o.get("weighted_mean_nll")
        if (nll_v is not None and math.isfinite(nll_v)
                and nll_h is not None and math.isfinite(nll_h)):
            rows.append(f"h2o_v1fa2_gap_nats\t{(nll_h - nll_v):.3f}")
        else:
            rows.append("h2o_v1fa2_gap_nats\tN/A")
        tps_v = phi3_v1fa2.get("decode_tps")
        tps_h = phi3_h2o.get("decode_tps")
        if tps_v is not None and tps_h is not None:
            rows.append(f"h2o_v1fa2_tps_gap\t{(tps_v - tps_h):.2f}")
        else:
            rows.append("h2o_v1fa2_tps_gap\tN/A")
    else:
        rows.append("h2o_v1fa2_gap_nats\tN/A")
        rows.append("h2o_v1fa2_tps_gap\tN/A")

    # Pareto flags
    rows.append("# --- Pareto flags (phi3) ---")
    for p in CHAPTER_POLICIES:
        flag = pareto.get(("phi3", p), "N/A")
        rows.append(f"phi3_{p}_pareto\t{flag}")

    # NIAH primary placeholders
    rows.append("# --- NIAH primary placeholders ---")
    niah_depths_for_chapter = (0, 87)  # the two depths the chapter shows
    for m in _model_order(cells):
        for p in CHAPTER_POLICIES:
            for ctx in CTX_LENGTHS:
                for d in niah_depths_for_chapter:
                    v = niah_grid.get((m, p), {}).get((ctx, d))
                    if v is None:
                        rows.append(f"niah_{m}_{p}_c{ctx}_d{d}\tN/A")
                    else:
                        rows.append(f"niah_{m}_{p}_c{ctx}_d{d}\t{int(bool(v))}")
    # Derived row means (phi3 v1fa2 only - the chapter only shows that one).
    for ctx in CTX_LENGTHS:
        vals = []
        for d in niah_depths_for_chapter:
            v = niah_grid.get(("phi3", "v1fa2"), {}).get((ctx, d))
            if v is None:
                continue
            vals.append(int(bool(v)))
        if not vals:
            rows.append(f"niah_phi3_v1fa2_c{ctx}_mean\tN/A")
        else:
            rows.append(f"niah_phi3_v1fa2_c{ctx}_mean\t"
                        f"{sum(vals) / len(vals):.2f}")
    # Overall per-cell NIAH accuracy.
    for tag_policy in ("v1fa2", "h2o"):
        v = niah_per_cell.get(("phi3", tag_policy))
        if v is None or not v.get("total"):
            rows.append(f"niah_phi3_{tag_policy}_overall\tN/A")
        else:
            rows.append(f"niah_phi3_{tag_policy}_overall\t"
                        f"{v['correct'] / v['total']:.2f}")

    # Editorial / target constants
    rows.append("# --- editorial targets (declared, not measured) ---")
    for k, v in DEFAULT_EDITORIAL_TARGETS.items():
        rows.append(f"{k}\t{v}")

    # K-sweep placeholders (read from any wave11_ksweep_* run)
    rows.append("# --- K-sweep (best-effort; left as N/A if no ksweep run) ---")
    for K in (256, 384, 512, 1024):
        rows.append(f"ksweep_K{K}_ppl_mean\tN/A")

    # Stray docs-only tag
    rows.append("# skip — documentation-only tag (kept literal in chapter)")
    rows.append("# name\t(intentionally left unfilled)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


# Overall verdict

def overall_verdict(agg: Dict) -> str:
    """One-line summary of the current data. The chapter's conclusions come from the
    paired tests and Pareto front, not from this line."""
    cells = agg["cells"]
    if not cells:
        return ("NO DATA — no Wave-11 PPL cells found under "
                "phone-logs/wave11_eval_*")
    n_cells_complete = sum(1 for v in cells.values()
                           if v.get("n_chunks", 0) >= 8)
    n_cells_partial = sum(1 for v in cells.values()
                          if 0 < v.get("n_chunks", 0) < 8)
    n_models_present = len({m for (m, _p) in cells.keys()})
    n_policies_present = len({p for (_m, p) in cells.keys()})

    # Only report a verdict once at least one primary-pair cell is complete.
    A, B = PRIMARY_CONTRAST
    prim_hits = [r for r in agg["sig_ppl"]
                 if {r["a"], r["b"]} == {A, B}]
    if prim_hits:
        best = min(prim_hits, key=lambda r: r["w_p"]
                   if math.isfinite(r["w_p"]) else float("inf"))
        if math.isfinite(best["w_p"]) and best["w_p"] < 0.05:
            dir_word = ("v1_fa2_stack lower-PPL" if best["mean_log_delta"] < 0
                        else "tova lower-PPL")
            return (f"PARTIAL ({n_cells_complete} complete, {n_cells_partial} "
                    f"partial, {n_models_present} models / {n_policies_present} "
                    f"policies); primary contrast {A} vs {B}: "
                    f"{dir_word}, Wilcoxon p={best['w_p']:.3g}")

    if n_cells_complete == 0 and n_cells_partial > 0:
        return (f"IN-PROGRESS — {n_cells_partial} cell(s) partial, "
                f"0 complete; primary contrast not yet evaluable")
    return (f"INTERIM — {n_cells_complete} complete, "
            f"{n_cells_partial} partial cell(s); "
            f"{n_models_present} model(s), {n_policies_present} policy(ies)")


# CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace-root", default=str(default_workspace_root()),
                    help="EndurKV_workspace root (default: %(default)s)")
    ap.add_argument("--phone-logs-root", default=None,
                    help="Override phone-logs directory "
                         "(default: <workspace>/phone-logs)")
    ap.add_argument("--bootstrap-samples", type=int, default=1000,
                    help="Bootstrap resample count (default: 1000)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for bootstrap reproducibility (default: 0)")
    ap.add_argument("--fills-tsv", default=PATH_FILLS_TSV,
                    help="Override the fills TSV path (default: %(default)s)")
    args = ap.parse_args(argv)

    workspace_root = Path(args.workspace_root).resolve()
    endurkv_root = workspace_root / "EndurKV"
    phone_logs_root = Path(
        args.phone_logs_root or (workspace_root / "phone-logs")
    ).resolve()
    phone_logs_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] workspace_root  = {workspace_root}")
    print(f"[info] phone_logs_root = {phone_logs_root}")

    run_dir = latest_wave11_run(phone_logs_root)
    if run_dir is None:
        print("[warn] no wave11_eval_* run found locally; emitting empty report")
    else:
        print(f"[info] latest run      = {run_dir}")

    agg = aggregate_all(
        phone_logs_root=phone_logs_root,
        endurkv_root=endurkv_root,
        run_dir=run_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )

    verdict = overall_verdict(agg)

    # Render outputs.
    final_md = endurkv_root / PATH_FINAL_REPORT_MD
    ppl_png = endurkv_root / PATH_PPL_BARS_PNG
    pareto_png = endurkv_root / PATH_PARETO_PNG
    niah_png = endurkv_root / PATH_NIAH_HEATMAP_PNG
    fills_tsv = Path(args.fills_tsv)

    render_report_md(agg, run_dir, phone_logs_root, final_md, verdict)
    print(f"[ok] wrote {final_md}")

    render_ppl_bars(agg["cells"], ppl_png)
    if ppl_png.is_file():
        print(f"[ok] wrote {ppl_png}")

    render_pareto_bubble(agg["cells"], agg["pareto"], pareto_png)
    if pareto_png.is_file():
        print(f"[ok] wrote {pareto_png}")

    render_niah_heatmap(agg["niah_grid"], niah_png)
    if niah_png.is_file():
        print(f"[ok] wrote {niah_png}")

    emit_fills_tsv(agg, fills_tsv)
    print(f"[ok] wrote {fills_tsv}")

    print(f"wave11_final_report: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
