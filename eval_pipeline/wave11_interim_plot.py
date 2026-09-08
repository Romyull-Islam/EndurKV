#!/usr/bin/env python3
"""
wave11_interim_plot.py - Host-side INTERIM plotter for Wave-11 evals.

While the phone is grinding through the 30 (model x policy x bench) cells of
the Wave-11 Tier-1 manifest, this script gives you an at-a-glance status
snapshot WITHOUT waiting for the full run to finish.

What it does
------------
1. Pull /data/local/tmp/endurkv/logs/wave11_eval_*/ from the phone to
   <workspace>/phone-logs/, but only if the on-device tree is newer than the
   local copy (mtime comparison via `adb_resilient.sh`'s adb_safe_shell).
   If adb is unavailable, we silently fall back to whatever is already local
   so the script remains useful offline.

2. Walk every (model, policy, benchmark) cell directory under the most-recent
   wave11_eval_*/ run and classify it as complete / partial / pending using
   the wave11_cells.json manifest as ground truth for per-cell-group counts
   (8 PPL chunks + 8 NIAH stimuli per (model, policy)).

3. For PPL cell-groups: aggregate completed chunks, compute mean PPL and a
   1000-sample percentile bootstrap 95% CI.

4. For NIAH cell-groups: count gen.txt files and rule-grade them by
   case-insensitive substring "sandwich at dolores park".

5. Emit:
     figures/eval_plots/wave11_interim_ppl.png
         bar chart per (model, policy) of mean PPL +/- 95% CI.
     figures/eval_plots/wave11_interim_niah.png
         bar chart per (model, policy) of rule-based NIAH accuracy.
     figures/master_tables/WAVE11_INTERIM.md
         markdown table:  model | policy | bench |
             cells_complete | cells_pending | mean_ppl | niah_acc

6. Print a progress summary line:
     "X of 30 cells complete, ETA Yh"

Idempotent: safe to call repeatedly. Each invocation overwrites its outputs
in place. If the phone is offline / unauthorised / not connected the script
still works on the local mirror.

Stdlib + matplotlib only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PHONE_LOG_ROOT = "/data/local/tmp/endurkv/logs"
NEEDLE_KEY = "sandwich at dolores park"   # case-insensitive substring

# How long each cell takes (used for ETA estimate). The wave11 manifest has
# a per-cell estimate but for the interim ETA we just use a flat average
# (total_expected_minutes / total_cells) so we don't have to re-key into the
# manifest for every (model, policy, bench, idx) lookup.
DEFAULT_TOTAL_CELLS = 30           # 3 models x 5 policies x 2 benches
DEFAULT_CHUNKS_PER_CELL = 8        # 8 PPL chunks / 8 NIAH stimuli per cell

# Output paths (relative to EndurKV repo root)
PPL_PNG = "figures/eval_plots/wave11_interim_ppl.png"
NIAH_PNG = "figures/eval_plots/wave11_interim_niah.png"
MD_TABLE = "figures/master_tables/WAVE11_INTERIM.md"


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #

def default_endurkv_root() -> Path:
    """<workspace>/EndurKV, derived from this file's location."""
    return Path(__file__).resolve().parent.parent


def default_workspace_root() -> Path:
    return default_endurkv_root().parent


# --------------------------------------------------------------------------- #
# adb_resilient.sh wrappers
# --------------------------------------------------------------------------- #

def _adb_script_path(endurkv_root: Path) -> Path:
    return endurkv_root / "scripts" / "android" / "adb_resilient.sh"


def _run_adb_helper(endurkv_root: Path, helper: str, *args: str,
                    timeout: int = 600) -> Tuple[int, str, str]:
    """
    Source adb_resilient.sh and invoke one of its helpers. Returns
    (returncode, stdout, stderr). All errors are caught and reported as a
    non-zero rc; we never raise from here so the caller can fall back to the
    existing local mirror.
    """
    script = _adb_script_path(endurkv_root)
    if not script.is_file():
        return 1, "", f"adb_resilient.sh not found at {script}"
    quoted = " ".join(_shquote(a) for a in args)
    cmd = f". {_shquote(str(script))} && {helper} {quoted}"
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return 1, "", f"{type(exc).__name__}: {exc}"
    return proc.returncode, proc.stdout, proc.stderr


def _shquote(s: str) -> str:
    if not s or re.search(r"[^A-Za-z0-9_./:=@%+\-]", s):
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s


def adb_device_online(endurkv_root: Path) -> bool:
    """True if at least one device is in 'device' state. Non-blocking."""
    try:
        proc = subprocess.run(
            ["adb", "get-state"], capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "device"


def adb_remote_mtime(endurkv_root: Path, remote_path: str) -> Optional[int]:
    """
    Return the mtime (epoch seconds) of remote_path on the phone, or None
    if it doesn't exist / we can't query it.
    """
    rc, out, _ = _run_adb_helper(
        endurkv_root,
        "adb_safe_shell",
        f"stat -c %Y {_shquote(remote_path)} 2>/dev/null || echo MISSING",
    )
    if rc != 0:
        return None
    s = out.strip().splitlines()[-1] if out.strip() else ""
    if not s or s == "MISSING":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def adb_list_wave11_runs(endurkv_root: Path) -> List[str]:
    """
    List wave11_eval_* directory names under PHONE_LOG_ROOT on the phone.
    Returns [] if the phone is offline or no runs exist.
    """
    rc, out, _ = _run_adb_helper(
        endurkv_root,
        "adb_safe_shell",
        f"ls -1 {_shquote(PHONE_LOG_ROOT)} 2>/dev/null | grep '^wave11_eval_' || true",
    )
    if rc != 0:
        return []
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def adb_pull_run(endurkv_root: Path, run_name: str, local_dest: Path) -> bool:
    """
    Pull /data/local/tmp/endurkv/logs/<run_name> into local_dest/<run_name>.
    Returns True on success. Idempotent: the resilient pull will retry under
    USB drops and overwrite local files with the newer phone copies.
    """
    local_dest.mkdir(parents=True, exist_ok=True)
    remote = f"{PHONE_LOG_ROOT}/{run_name}"
    rc, _out, _err = _run_adb_helper(
        endurkv_root,
        "adb_safe_pull",
        remote,
        str(local_dest),
        timeout=1800,
    )
    return rc == 0


def maybe_sync_from_phone(endurkv_root: Path, phone_logs_root: Path) -> None:
    """
    For each wave11_eval_* run on the phone, pull it locally IF the phone-side
    mtime is newer than the local copy (or the local copy is missing).
    Silently skips everything if no device is online.
    """
    if not adb_device_online(endurkv_root):
        print("[sync] no adb device online; skipping phone pull")
        return
    phone_runs = adb_list_wave11_runs(endurkv_root)
    if not phone_runs:
        print("[sync] no wave11_eval_* runs on phone; nothing to pull")
        return
    print(f"[sync] phone runs: {len(phone_runs)} ({', '.join(phone_runs)})")
    for run in phone_runs:
        remote = f"{PHONE_LOG_ROOT}/{run}"
        local = phone_logs_root / run
        remote_mtime = adb_remote_mtime(endurkv_root, remote)
        local_mtime = int(local.stat().st_mtime) if local.exists() else 0
        if remote_mtime is None:
            # Phone glitch; pull anyway to be safe.
            print(f"[sync] mtime unknown for {run}; pulling")
            ok = adb_pull_run(endurkv_root, run, phone_logs_root)
        elif remote_mtime > local_mtime:
            age = remote_mtime - local_mtime
            print(f"[sync] {run}: phone newer by {age}s -> pulling")
            ok = adb_pull_run(endurkv_root, run, phone_logs_root)
        else:
            print(f"[sync] {run}: local up-to-date (mtime={local_mtime})")
            ok = True
        if not ok:
            print(f"[sync] warning: failed to pull {run}; using stale local copy")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def load_manifest(endurkv_root: Path) -> Dict:
    """
    Load wave11_cells.json. The interim plotter only needs the high-level
    counts and the per-cell expected runtime; everything else is informational.
    """
    p = endurkv_root / "eval_pipeline" / "wave11_cells.json"
    if not p.is_file():
        print(f"[warn] manifest missing: {p}; using defaults")
        return {
            "models": [],
            "policies": [],
            "benchmarks": ["ppl", "niah"],
            "n_ppl_chunks_per_cell_group": DEFAULT_CHUNKS_PER_CELL,
            "n_niah_stimuli_per_cell_group": DEFAULT_CHUNKS_PER_CELL,
            "total_cells": DEFAULT_TOTAL_CELLS * DEFAULT_CHUNKS_PER_CELL,
            "total_expected_minutes": 1395.6,
            "cells": [],
        }
    with open(p) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Wave-11 layout discovery
# --------------------------------------------------------------------------- #

def latest_wave11_run(phone_logs_root: Path) -> Optional[Path]:
    """
    Pick the most-recent local wave11_eval_*/ directory (by mtime). Falls back
    to wave11_smoke_* if there's no _eval_ run yet (useful early in the cycle).
    """
    candidates = sorted(
        phone_logs_root.glob("wave11_eval_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # Fallback: smoke test, so the script still produces something to look at.
    smoke = sorted(
        phone_logs_root.glob("wave11_smoke_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return smoke[0] if smoke else None


def walk_cells(run_dir: Path) -> Dict[Tuple[str, str, str], Path]:
    """
    Walk run_dir/<model>/<policy>/<bench>/ and return a map
    (model, policy, bench) -> Path to that bench directory.
    """
    out: Dict[Tuple[str, str, str], Path] = {}
    if not run_dir or not run_dir.is_dir():
        return out
    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for policy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for bench_dir in sorted(p for p in policy_dir.iterdir() if p.is_dir()):
                bench = bench_dir.name.lower()
                if bench not in ("ppl", "niah"):
                    continue
                out[(model_dir.name, policy_dir.name, bench)] = bench_dir
    return out


# --------------------------------------------------------------------------- #
# PPL aggregation
# --------------------------------------------------------------------------- #

def _read_ppl(meta_path: Path) -> Optional[float]:
    # eviction_bench writes bareword `inf` for non-finite decode_tps; tolerate
    # by patching to the json-permissive token before parsing.
    try:
        with open(meta_path) as f:
            raw = f.read()
    except OSError:
        return None
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        patched = re.sub(
            r'(?<![A-Za-z0-9_."])(-?inf|nan)(?![A-Za-z0-9_])',
            lambda m: {"inf": "Infinity", "-inf": "-Infinity",
                       "nan": "NaN"}[m.group(0).lower()],
            raw,
            flags=re.IGNORECASE,
        )
        try:
            meta = json.loads(patched)
        except json.JSONDecodeError:
            return None
    raw = meta.get("perplexity")
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def aggregate_ppl_cell(bench_dir: Path) -> Tuple[List[float], int]:
    """
    Returns (ppl_values, n_iter_dirs). n_iter_dirs counts every iter*/
    directory even if its meta.json hasn't been written yet (which is the
    "in-flight" signal).
    """
    if not bench_dir.is_dir():
        return [], 0
    iter_dirs = sorted(p for p in bench_dir.iterdir()
                       if p.is_dir() and p.name.startswith("iter"))
    ppls: List[float] = []
    for it in iter_dirs:
        v = _read_ppl(it / "meta.json")
        if v is not None:
            ppls.append(v)
    return ppls, len(iter_dirs)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs: List[float], n_samples: int = 1000,
                 alpha: float = 0.05,
                 rng: Optional[random.Random] = None
                 ) -> Tuple[float, float]:
    if rng is None:
        rng = random.Random(0)
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (xs[0], xs[0])
    samples = []
    for _ in range(n_samples):
        resample = [xs[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(resample) / n)
    samples.sort()
    lo = max(0, min(n_samples - 1,
                    int(math.floor((alpha / 2.0) * n_samples))))
    hi = max(0, min(n_samples - 1,
                    int(math.ceil((1.0 - alpha / 2.0) * n_samples)) - 1))
    return samples[lo], samples[hi]


# --------------------------------------------------------------------------- #
# NIAH aggregation
# --------------------------------------------------------------------------- #

def aggregate_niah_cell(bench_dir: Path) -> Tuple[int, int]:
    """
    Returns (n_correct, n_gen_files). A trial is "correct" iff the
    rule-based substring is present in gen.txt.
    """
    if not bench_dir.is_dir():
        return 0, 0
    gen_files = sorted(bench_dir.glob("**/gen.txt"))
    n_total = len(gen_files)
    n_correct = 0
    for gf in gen_files:
        try:
            text = gf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if NEEDLE_KEY in text.lower():
            n_correct += 1
    return n_correct, n_total


# --------------------------------------------------------------------------- #
# Cell-level summary (one row per (model, policy, bench))
# --------------------------------------------------------------------------- #

def summarise_cells(run_dir: Optional[Path], manifest: Dict
                    ) -> List[Dict]:
    """
    For each (model, policy, bench) cell, return a dict:
      model, policy, bench,
      chunks_done, chunks_started, chunks_expected,
      status (complete / partial / pending),
      mean_ppl, ci_low, ci_high,        (ppl cells only; NaN otherwise)
      niah_correct, niah_total, niah_acc (niah cells only)
    """
    bench_dirs = walk_cells(run_dir) if run_dir else {}
    models = manifest.get("models") or sorted({m for (m, _, _) in bench_dirs.keys()})
    policies = manifest.get("policies") or sorted({p for (_, p, _) in bench_dirs.keys()})
    benches = manifest.get("benchmarks") or ["ppl", "niah"]
    chunks_per_cell = manifest.get("n_ppl_chunks_per_cell_group",
                                   DEFAULT_CHUNKS_PER_CELL)

    rng = random.Random(0)
    rows: List[Dict] = []

    # We have to be tolerant about model-name conventions: the manifest says
    # "phi3" / "llama1b" / "gemma2b", but the on-device dir names are the
    # gguf-stems ("Llama-3.2-1B", "Phi-3-mini-...", "gemma-2-2b-it"). Match by
    # the tag the on-device runner actually emits.
    on_disk_models = sorted({m for (m, _, _) in bench_dirs.keys()})
    if not models:
        models = on_disk_models
    # If manifest models don't match on-disk, prefer on-disk (we report what
    # exists, not what was planned).
    elif set(models).isdisjoint(set(on_disk_models)) and on_disk_models:
        models = on_disk_models

    on_disk_policies = sorted({p for (_, p, _) in bench_dirs.keys()})
    if not policies:
        policies = on_disk_policies

    for model in models:
        for policy in policies:
            for bench in benches:
                key = (model, policy, bench)
                bdir = bench_dirs.get(key)
                row: Dict = {
                    "model": model,
                    "policy": policy,
                    "bench": bench,
                    "chunks_expected": chunks_per_cell,
                    "chunks_started": 0,
                    "chunks_done": 0,
                    "mean_ppl": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "niah_correct": 0,
                    "niah_total": 0,
                    "niah_acc": float("nan"),
                    "status": "pending",
                }
                if bdir is None:
                    rows.append(row)
                    continue
                if bench == "ppl":
                    ppls, n_iter = aggregate_ppl_cell(bdir)
                    row["chunks_started"] = n_iter
                    row["chunks_done"] = len(ppls)
                    if ppls:
                        row["mean_ppl"] = mean(ppls)
                        lo, hi = bootstrap_ci(ppls, rng=rng)
                        row["ci_low"] = lo
                        row["ci_high"] = hi
                else:  # niah
                    n_correct, n_total = aggregate_niah_cell(bdir)
                    row["chunks_started"] = n_total
                    row["chunks_done"] = n_total
                    row["niah_correct"] = n_correct
                    row["niah_total"] = n_total
                    if n_total:
                        row["niah_acc"] = n_correct / n_total
                if row["chunks_done"] >= chunks_per_cell:
                    row["status"] = "complete"
                elif row["chunks_done"] > 0 or row["chunks_started"] > 0:
                    row["status"] = "partial"
                rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _label(model: str, policy: str) -> str:
    return f"{model}\n{policy}"


def render_ppl_plot(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ppl_rows = [r for r in rows
                if r["bench"] == "ppl" and r["chunks_done"] > 0
                and math.isfinite(r["mean_ppl"])]
    if not ppl_rows:
        _placeholder(out_path, "No Wave-11 PPL chunks finished yet")
        return

    # Group by model so policies cluster together visually.
    models = sorted({r["model"] for r in ppl_rows})
    policies = sorted({r["policy"] for r in ppl_rows})
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
            match = [r for r in ppl_rows
                     if r["model"] == model and r["policy"] == policy]
            if not match:
                continue
            r = match[0]
            xs.append(x_base[mi] - group_width / 2 + (pi + 0.5) * bar_width)
            heights.append(r["mean_ppl"])
            err_lo.append(max(0.0, r["mean_ppl"] - r["ci_low"]))
            err_hi.append(max(0.0, r["ci_high"] - r["mean_ppl"]))
            ns.append(r["chunks_done"])
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
    ax.set_title("Wave-11 INTERIM perplexity by policy (95% bootstrap CI)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="Policy", fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_niah_plot(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    niah_rows = [r for r in rows
                 if r["bench"] == "niah" and r["niah_total"] > 0]
    if not niah_rows:
        _placeholder(out_path, "No Wave-11 NIAH generations yet")
        return

    items = sorted(niah_rows, key=lambda r: (r["model"], r["policy"]))
    labels = [_label(r["model"], r["policy"]) for r in items]
    accs = [r["niah_acc"] * 100.0 for r in items]
    counts = [(r["niah_correct"], r["niah_total"]) for r in items]

    fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(items) + 2), 4.8))
    bars = ax.bar(range(len(items)), accs, color="#3a78b4",
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("NIAH accuracy (%)  -  rule judge")
    ax.set_title(f'Wave-11 INTERIM NIAH accuracy '
                 f'(substring "{NEEDLE_KEY}")')
    ax.grid(axis="y", alpha=0.3)
    for b, acc, (c, t) in zip(bars, accs, counts):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 1.5,
                f"{acc:.0f}%\n{c}/{t}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _placeholder(out_path: Path, message: str) -> None:
    """Skip placeholder PNG generation.

    Earlier versions of this script wrote a faux figure showing the message
    string when no data was available. Those placeholder PNGs were
    indistinguishable from real figures at thumbnail size and led reviewers
    to mistake them for results. We now refuse to write them and emit a
    diagnostic to stderr instead — `figures/eval_plots/` will be empty
    until real data lands on disk.
    """
    import sys as _sys
    _sys.stderr.write(
        f"[skip] {message}; not writing placeholder PNG to {out_path}\n"
    )


def render_markdown(rows: List[Dict], out_path: Path,
                    run_dir: Optional[Path],
                    progress: Dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Wave-11 INTERIM status")
    lines.append("")
    ts = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines.append(f"Snapshot at: **{ts}**  ")
    lines.append(f"Run dir: `{run_dir if run_dir else '(none)'}`  ")
    lines.append(f"Progress: **{progress['cells_complete']} / "
                 f"{progress['cells_total']} cells complete**, "
                 f"{progress['cells_partial']} partial, "
                 f"{progress['cells_pending']} pending.  ")
    eta_h = progress["eta_hours"]
    eta_str = f"{eta_h:.1f}h" if math.isfinite(eta_h) else "n/a"
    lines.append(f"ETA to finish (flat per-cell estimate): **{eta_str}**")
    lines.append("")
    lines.append("Rule-based NIAH judge: case-insensitive substring "
                 f"`\"{NEEDLE_KEY}\"`.")
    lines.append("")
    lines.append("| model | policy | bench | cells_complete | cells_pending "
                 "| chunks_done/expected | mean_ppl | niah_acc | status |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")

    # Roll up cells_complete / cells_pending per (model, policy):
    # there are two benches; we report both rows but include the cross-bench
    # rollup at the model/policy level too.
    rows_sorted = sorted(rows, key=lambda r: (r["model"], r["policy"], r["bench"]))
    for r in rows_sorted:
        complete = 1 if r["status"] == "complete" else 0
        pending = 0 if r["status"] == "complete" else 1
        chunks = f"{r['chunks_done']}/{r['chunks_expected']}"
        mean_ppl_s = (f"{r['mean_ppl']:.3f}"
                      if math.isfinite(r['mean_ppl']) else "-")
        niah_s = (f"{r['niah_acc']*100:.0f}% ({r['niah_correct']}/{r['niah_total']})"
                  if r["bench"] == "niah" and r["niah_total"] > 0 else "-")
        lines.append(
            f"| {r['model']} | {r['policy']} | {r['bench']} | "
            f"{complete} | {pending} | {chunks} | {mean_ppl_s} | "
            f"{niah_s} | {r['status']} |"
        )

    lines.append("")
    lines.append("Outputs:")
    lines.append(f"- `{PPL_PNG}`")
    lines.append(f"- `{NIAH_PNG}`")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Progress / ETA
# --------------------------------------------------------------------------- #

def compute_progress(rows: List[Dict], manifest: Dict,
                     total_cells: int) -> Dict:
    n_complete = sum(1 for r in rows if r["status"] == "complete")
    n_partial = sum(1 for r in rows if r["status"] == "partial")
    n_pending = sum(1 for r in rows if r["status"] == "pending")
    n_total = len(rows) if rows else total_cells

    total_minutes = float(manifest.get("total_expected_minutes")
                          or (total_cells *
                              DEFAULT_CHUNKS_PER_CELL * 5.0))
    # Manifest expresses total over 240 chunk-runs but we report at the
    # cell-group level (30 cells = 30 (model, policy, bench)). The per-cell
    # estimate is total_minutes / total_cells_in_manifest.
    cells_in_manifest = max(
        1, len(manifest.get("cells", [])) // max(1, manifest.get(
            "n_ppl_chunks_per_cell_group", DEFAULT_CHUNKS_PER_CELL)),
    ) if manifest.get("cells") else n_total
    # Defensive: fall back to spec-level total_cells if cells_in_manifest is
    # implausible.
    if cells_in_manifest < 1 or cells_in_manifest > 10 * n_total:
        cells_in_manifest = n_total

    minutes_per_cell = total_minutes / max(cells_in_manifest, 1)
    remaining_cells = n_partial + n_pending
    eta_minutes = remaining_cells * minutes_per_cell
    eta_hours = eta_minutes / 60.0

    return {
        "cells_complete": n_complete,
        "cells_partial": n_partial,
        "cells_pending": n_pending,
        "cells_total": n_total,
        "eta_hours": eta_hours,
        "minutes_per_cell": minutes_per_cell,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endurkv-root", default=str(default_endurkv_root()),
                    help="EndurKV repo root (default: %(default)s)")
    ap.add_argument("--phone-logs-root", default=None,
                    help="Local mirror of phone logs "
                         "(default: <workspace>/phone-logs)")
    ap.add_argument("--no-sync", action="store_true",
                    help="Skip the adb pull step (use local mirror as-is).")
    ap.add_argument("--total-cells", type=int, default=DEFAULT_TOTAL_CELLS,
                    help="Total cell-groups for progress denominator "
                         "(default: %(default)s)")
    args = ap.parse_args(argv)

    endurkv_root = Path(args.endurkv_root).resolve()
    workspace_root = endurkv_root.parent
    phone_logs_root = Path(
        args.phone_logs_root or (workspace_root / "phone-logs")
    ).resolve()
    phone_logs_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] endurkv_root    = {endurkv_root}")
    print(f"[info] phone_logs_root = {phone_logs_root}")

    if not args.no_sync:
        maybe_sync_from_phone(endurkv_root, phone_logs_root)
    else:
        print("[sync] --no-sync set; skipping phone pull")

    manifest = load_manifest(endurkv_root)
    run_dir = latest_wave11_run(phone_logs_root)
    if run_dir is None:
        print("[warn] no wave11 run dir found locally; emitting placeholders")
    else:
        print(f"[info] using run    = {run_dir}")

    rows = summarise_cells(run_dir, manifest)
    if not rows:
        # Synthesize empty rows so the table & plots have something to render.
        models = manifest.get("models") or ["phi3", "llama1b", "gemma2b"]
        policies = manifest.get("policies") or [
            "vanilla", "v1", "tova", "h2o", "v1_fa2_stack"]
        for m in models:
            for p in policies:
                for b in ("ppl", "niah"):
                    rows.append({
                        "model": m, "policy": p, "bench": b,
                        "chunks_expected": manifest.get(
                            "n_ppl_chunks_per_cell_group",
                            DEFAULT_CHUNKS_PER_CELL),
                        "chunks_started": 0, "chunks_done": 0,
                        "mean_ppl": float("nan"),
                        "ci_low": float("nan"), "ci_high": float("nan"),
                        "niah_correct": 0, "niah_total": 0,
                        "niah_acc": float("nan"),
                        "status": "pending",
                    })

    progress = compute_progress(rows, manifest, total_cells=args.total_cells)

    # Render artefacts.
    ppl_out = endurkv_root / PPL_PNG
    niah_out = endurkv_root / NIAH_PNG
    md_out = endurkv_root / MD_TABLE
    render_ppl_plot(rows, ppl_out)
    render_niah_plot(rows, niah_out)
    render_markdown(rows, md_out, run_dir, progress)

    print(f"[ok] wrote {ppl_out}")
    print(f"[ok] wrote {niah_out}")
    print(f"[ok] wrote {md_out}")

    # Progress summary line (the contract).
    eta = progress["eta_hours"]
    eta_str = f"{eta:.1f}h" if math.isfinite(eta) else "n/a"
    print(f"{progress['cells_complete']} of {progress['cells_total']} "
          f"cells complete, ETA {eta_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
