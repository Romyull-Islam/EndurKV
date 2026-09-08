#!/usr/bin/env python3
"""
score_ppl.py — Host-side perplexity aggregator for Wave-11 evals.

Reads all phone-logs/wave11_*/{model}/{policy}/iter*/meta.json files,
extracts the `perplexity` value from each, computes per-cell statistics
(mean, bootstrap 95% CI, std dev, n_chunks) over the 8 chunks per
(model, policy) cell, and emits:

  1. A Markdown comparison table sorted by mean PPL within each model
     -> figures/master_tables/TABLE_WAVE11_PPL.md
  2. A bar chart with 95% CI error bars per (model, policy) cell
     -> figures/eval_plots/ppl_per_policy.png

stdlib only + matplotlib. No pandas, no numpy beyond what matplotlib pulls.

Usage:
  python eval_pipeline/score_ppl.py \
      [--phone-logs-root /path/to/phone-logs] \
      [--workspace-root /path/to/EndurKV_workspace] \
      [--bootstrap-samples 1000] \
      [--seed 0]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #

def default_workspace_root() -> str:
    """
    Resolve workspace root assuming this file lives at
    <workspace>/EndurKV/eval_pipeline/score_ppl.py.
    """
    here = os.path.abspath(__file__)
    eval_pipeline_dir = os.path.dirname(here)
    endurkv_dir = os.path.dirname(eval_pipeline_dir)
    workspace_dir = os.path.dirname(endurkv_dir)
    return workspace_dir


# --------------------------------------------------------------------------- #
# Discovery & parsing
# --------------------------------------------------------------------------- #

def discover_meta_files(phone_logs_root: str) -> List[str]:
    """
    Find every meta.json under the Wave-11 PPL tree. Supports BOTH the new
    bench-aware launcher layout (Wave-11 phone_wave11_eval.sh):

        phone-logs/wave11_*/<model>/<policy>/ppl/iter*/meta.json

    AND the legacy flat layout (older waves / smoke tests):

        phone-logs/wave11_*/<model>/<policy>/iter*/meta.json

    The new layout is preferred; if any new-layout hits exist for a given
    (wave, model, policy) cell, the legacy hits for that cell are ignored
    (avoids double-counting when both happen to coexist).
    """
    new_pat = os.path.join(
        phone_logs_root, "wave11_*", "*", "*", "ppl", "iter*", "meta.json"
    )
    legacy_pat = os.path.join(
        phone_logs_root, "wave11_*", "*", "*", "iter*", "meta.json"
    )
    new_hits = sorted(glob.glob(new_pat))
    legacy_hits = sorted(glob.glob(legacy_pat))

    # If a legacy hit's iter* dir is the 4th part (i.e. parts[3] == 'iterNNNN'),
    # it's a true legacy entry. If parts[3] in {'ppl','niah'} it's actually a
    # new-layout entry already captured above — exclude it from legacy_hits.
    def _is_true_legacy(meta_path: str) -> bool:
        rel = os.path.relpath(meta_path, phone_logs_root)
        parts = rel.split(os.sep)
        # legacy layout has exactly 5 parts: wave/<model>/<policy>/iter*/meta.json
        return len(parts) == 5 and parts[3].startswith("iter")

    legacy_filtered = [m for m in legacy_hits if _is_true_legacy(m)]

    # If a (wave, model, policy) cell has new-layout hits, drop its legacy hits
    # to prevent counting the same chunk twice.
    new_cells = set()
    for m in new_hits:
        rel = os.path.relpath(m, phone_logs_root)
        parts = rel.split(os.sep)
        # parts: wave/model/policy/ppl/iterNNNN/meta.json
        if len(parts) >= 6 and parts[3] == "ppl":
            new_cells.add((parts[0], parts[1], parts[2]))

    def _cell_of_legacy(meta_path: str) -> Tuple[str, str, str]:
        rel = os.path.relpath(meta_path, phone_logs_root)
        parts = rel.split(os.sep)
        return (parts[0], parts[1], parts[2])

    legacy_kept = [m for m in legacy_filtered if _cell_of_legacy(m) not in new_cells]
    return sorted(set(new_hits) | set(legacy_kept))


def parse_cell_from_path(meta_path: str, phone_logs_root: str) -> Tuple[str, str, str, str]:
    """
    Given a meta.json path under either layout:
      new:    .../wave11_XYZ/<model>/<policy>/ppl/<iter>/meta.json   (6 parts)
      legacy: .../wave11_XYZ/<model>/<policy>/<iter>/meta.json       (5 parts)
    return (wave_dir, model, policy, iter_dir).
    """
    rel = os.path.relpath(meta_path, phone_logs_root)
    parts = rel.split(os.sep)
    if len(parts) >= 6 and parts[3] in ("ppl",):
        return parts[0], parts[1], parts[2], parts[4]
    if len(parts) == 5:
        return parts[0], parts[1], parts[2], parts[3]
    raise ValueError(f"Unexpected meta.json path layout: {meta_path}")


def _load_meta_tolerant(meta_path: str) -> dict | None:
    """
    Read meta.json with one critical concession: eviction_bench currently
    emits bareword `inf` / `nan` for non-finite numeric fields (e.g.
    decode_tps when decode_ms == 0), which is NOT valid JSON per RFC 8259
    and causes the stdlib parser to abort the whole file. We work around
    this by substituting `inf`/`-inf`/`nan` with their JSON-permissive
    equivalents (`Infinity`/`-Infinity`/`NaN`) which Python's json module
    accepts when allow_nan=True (the default). The substitution is done on
    a copy of the bytes so we never mutate the on-disk file.
    """
    try:
        with open(meta_path, "r") as f:
            raw = f.read()
    except OSError as exc:
        print(f"[warn] could not open {meta_path}: {exc}", file=sys.stderr)
        return None
    # First, try a strict parse — fast path for well-formed files.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Substitute bareword non-finite numerics with the json-permissive form.
    # Use word-boundary regex so we don't clobber e.g. "infinitive" or
    # quoted strings that legitimately contain "inf".
    import re as _re
    patched = _re.sub(
        r'(?<![A-Za-z0-9_."])(-?inf|nan)(?![A-Za-z0-9_])',
        lambda m: {"inf": "Infinity", "-inf": "-Infinity",
                   "nan": "NaN"}[m.group(0).lower()],
        raw,
        flags=_re.IGNORECASE,
    )
    try:
        return json.loads(patched)
    except json.JSONDecodeError as exc:
        print(f"[warn] could not read {meta_path} even after inf/nan patch: {exc}",
              file=sys.stderr)
        return None


def load_perplexity(meta_path: str) -> float | None:
    """
    Read perplexity from meta.json. Returns None if missing / non-finite.
    """
    meta = _load_meta_tolerant(meta_path)
    if meta is None:
        return None

    ppl = meta.get("perplexity")
    if ppl is None:
        return None
    try:
        ppl_f = float(ppl)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ppl_f) or ppl_f <= 0:
        return None
    return ppl_f


def load_nll_and_tokens(meta_path: str) -> Tuple[float, int] | None:
    """
    Read (mean_nll, n_scored_tokens) from meta.json so aggregation can be done
    in log-domain (token-weighted geometric mean of PPL == exp(sum_nll/sum_tok)).
    Falls back to mean_nll = log(perplexity) and n_tokens = 1 when only the
    scalar PPL is recorded.
    """
    meta = _load_meta_tolerant(meta_path)
    if meta is None:
        return None

    nll = meta.get("mean_nll")
    # Wave-11 eviction_bench meta.json stores the per-chunk avg NLL as
    # `mean_nll` but does NOT (yet) emit an explicit `n_scored_tokens` field.
    # For teacher-forced PPL the number of scored tokens equals the number of
    # teacher-forced decode steps, so `n_decode_steps` is the correct weight.
    # Older / alternative meta variants may emit n_scored_tokens / n_ref_tokens
    # / n_tokens directly — honour those first.
    n_tok = (
        meta.get("n_scored_tokens")
        or meta.get("n_ref_tokens")
        or meta.get("n_tokens")
        or meta.get("n_decode_steps")
    )
    if nll is not None:
        try:
            nll_f = float(nll)
            n_tok_i = int(n_tok) if n_tok is not None else 1
        except (TypeError, ValueError):
            return None
        if not math.isfinite(nll_f) or n_tok_i <= 0:
            return None
        return (nll_f, n_tok_i)

    # Fallback: derive NLL from scalar PPL (unweighted equivalent).
    ppl = load_perplexity(meta_path)
    if ppl is None:
        return None
    return (math.log(ppl), 1)


# --------------------------------------------------------------------------- #
# Statistics (stdlib only)
# --------------------------------------------------------------------------- #

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def std_dev(xs: List[float]) -> float:
    """
    Sample standard deviation (ddof=1). Returns 0.0 if fewer than 2 samples.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def bootstrap_ci(
    xs: List[float],
    n_samples: int = 1000,
    alpha: float = 0.05,
    rng: random.Random | None = None,
    weights: List[int] | None = None,
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for the (optionally weighted) mean. Returns
    (lo, hi) at the (alpha/2, 1 - alpha/2) levels. If `weights` is provided,
    the resampled mean is the token-weighted mean sum(w*x)/sum(w) — this is
    used when xs holds per-chunk NLLs and weights are per-chunk token counts.
    """
    if rng is None:
        rng = random.Random(0)
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (xs[0], xs[0])

    means = []
    if weights is None:
        for _ in range(n_samples):
            resample = [xs[rng.randrange(n)] for _ in range(n)]
            means.append(sum(resample) / n)
    else:
        ws = list(weights)
        for _ in range(n_samples):
            sx = 0.0
            sw = 0
            for _i in range(n):
                k = rng.randrange(n)
                sx += xs[k] * ws[k]
                sw += ws[k]
            means.append(sx / sw if sw > 0 else float("nan"))
    means.sort()

    # Use round() so that for n=1000, alpha=0.05 we get the standard
    # 2.5th/97.5th percentile indices instead of an off-by-one upper bound.
    lo_idx = int(round((alpha / 2.0) * n_samples))
    hi_idx = int(round((1.0 - alpha / 2.0) * n_samples)) - 1
    lo_idx = max(0, min(n_samples - 1, lo_idx))
    hi_idx = max(0, min(n_samples - 1, hi_idx))
    return (means[lo_idx], means[hi_idx])


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate(
    phone_logs_root: str,
    n_bootstrap: int,
    seed: int,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Returns {(model, policy): {mean_ppl, ci_low, ci_high, std_dev, n_chunks}}.

    Aggregation method: TOKEN-WEIGHTED GEOMETRIC MEAN of per-chunk PPL, i.e.
       mean_ppl = exp( sum_i n_tok_i * mean_nll_i / sum_i n_tok_i )
    which is the conventional WikiText-2 PPL reporting used by H2O / KIVI /
    StreamingLLM / TOVA. The previous arithmetic-mean-of-PPL aggregation
    systematically over-weighted high-PPL outlier chunks and produced numbers
    not comparable to any published baseline.

    Bootstrap CI is computed in log domain on the per-chunk NLLs with token-count
    weights (paired across chunks → preserved by uniform per-cell resampling),
    then exponentiated for display. std_dev is also reported in log domain.

    Per cell, the RNG is reseeded with a stable (seed XOR hash(model, policy))
    so that the CI for a given cell is invariant to which other cells are
    present in the same run.
    """
    meta_files = discover_meta_files(phone_logs_root)
    if not meta_files:
        print(
            f"[warn] no meta.json files matched under "
            f"{phone_logs_root}/wave11_*/<model>/<policy>/iter*/",
            file=sys.stderr,
        )

    # Collect per cell across all wave11_* roots. If a (model,policy) cell
    # appears in multiple wave11_* dirs, we union all iter* observations.
    # Each entry: (mean_nll_i, n_scored_tokens_i).
    cells: Dict[Tuple[str, str], List[Tuple[float, int]]] = defaultdict(list)
    for meta_path in meta_files:
        try:
            _wave, model, policy, _iter = parse_cell_from_path(
                meta_path, phone_logs_root
            )
        except ValueError as exc:
            print(f"[warn] {exc}", file=sys.stderr)
            continue
        nt = load_nll_and_tokens(meta_path)
        if nt is None:
            print(
                f"[warn] no usable PPL/NLL in {meta_path}", file=sys.stderr
            )
            continue
        cells[(model, policy)].append(nt)

    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for cell, entries in cells.items():
        nlls = [e[0] for e in entries]
        toks = [e[1] for e in entries]
        if not nlls:
            continue
        sum_tok = sum(toks) if sum(toks) > 0 else 1
        weighted_mean_nll = sum(n * t for n, t in zip(nlls, toks)) / sum_tok
        # Per-cell deterministic RNG so CI for a cell does not depend on
        # other cells in the run (previously the single rng was drained in
        # filesystem-glob order).
        cell_seed = (seed ^ (hash(cell) & 0xFFFFFFFF)) & 0xFFFFFFFF
        rng = random.Random(cell_seed)
        lo_nll, hi_nll = bootstrap_ci(
            nlls, n_samples=n_bootstrap, rng=rng, weights=toks
        )
        # Map log-domain stats back to PPL space (exp).
        out[cell] = {
            "mean_ppl": math.exp(weighted_mean_nll),
            "ci_low": math.exp(lo_nll) if math.isfinite(lo_nll) else float("nan"),
            "ci_high": math.exp(hi_nll) if math.isfinite(hi_nll) else float("nan"),
            "std_dev": std_dev(nlls),     # std of log(PPL) — note units
            "n_chunks": len(nlls),
            "sum_tokens": sum_tok,
            "weighted_mean_nll": weighted_mean_nll,
        }
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_markdown_table(
    stats: Dict[Tuple[str, str], Dict[str, float]],
) -> str:
    """
    Build a Markdown table sorted by mean PPL within each model.
    Models appear in alphabetical order; within each model, policies are
    sorted by ascending mean PPL (best first).
    """
    lines: List[str] = []
    lines.append("# Wave-11 Perplexity Comparison")
    lines.append("")
    lines.append(
        "Per-cell stats over chunks discovered under "
        "`phone-logs/wave11_*/<model>/<policy>/ppl/iter*/meta.json` "
        "(or legacy `wave11_*/<model>/<policy>/iter*/meta.json`). "
        "PPL is the TOKEN-WEIGHTED GEOMETRIC MEAN per-cell "
        "(exp(sum_i n_tok_i * mean_nll_i / sum_i n_tok_i)) — the WikiText-2 "
        "convention used by H2O/KIVI/StreamingLLM/TOVA. "
        "CI is a 1000-sample percentile bootstrap (95%) over per-chunk NLLs, "
        "exponentiated for display. std_dev column is in log(PPL) units."
    )
    lines.append("")
    lines.append(
        "| Model | Policy | mean_PPL | CI_low | CI_high | std_dev | n_chunks |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|"
    )

    if not stats:
        lines.append("| _(no data)_ |  |  |  |  |  |  |")
        lines.append("")
        return "\n".join(lines)

    by_model: Dict[str, List[Tuple[str, Dict[str, float]]]] = defaultdict(list)
    for (model, policy), s in stats.items():
        by_model[model].append((policy, s))

    for model in sorted(by_model.keys()):
        rows = sorted(by_model[model], key=lambda kv: kv[1]["mean_ppl"])
        for policy, s in rows:
            lines.append(
                "| {model} | {policy} | {m:.4f} | {lo:.4f} | {hi:.4f} | "
                "{sd:.4f} | {n} |".format(
                    model=model,
                    policy=policy,
                    m=s["mean_ppl"],
                    lo=s["ci_low"],
                    hi=s["ci_high"],
                    sd=s["std_dev"],
                    n=s["n_chunks"],
                )
            )
    lines.append("")
    return "\n".join(lines)


def render_bar_plot(
    stats: Dict[Tuple[str, str], Dict[str, float]],
    out_path: str,
) -> None:
    """
    Grouped bar chart: one group per model, bars per policy, error bars
    from the bootstrap CI.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not stats:
        # Refuse to write a placeholder image — those have been mistaken for
        # results in the past. Print a clear message and leave the figures/
        # directory empty so reviewers see "no figure" rather than a fake one.
        print(
            f"[skip] no PPL data available; not writing {out_path}",
            file=sys.stderr,
        )
        return

    models = sorted({m for (m, _p) in stats.keys()})
    policies = sorted({p for (_m, p) in stats.keys()})

    n_models = len(models)
    n_policies = len(policies)
    group_width = 0.8
    bar_width = group_width / max(n_policies, 1)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * n_models * n_policies), 4.5))

    x_base = list(range(n_models))
    cmap = plt.get_cmap("tab10")

    for pi, policy in enumerate(policies):
        heights: List[float] = []
        err_lo: List[float] = []
        err_hi: List[float] = []
        xs: List[float] = []
        for mi, model in enumerate(models):
            s = stats.get((model, policy))
            if s is None:
                continue
            xs.append(x_base[mi] - group_width / 2 + (pi + 0.5) * bar_width)
            heights.append(s["mean_ppl"])
            err_lo.append(max(0.0, s["mean_ppl"] - s["ci_low"]))
            err_hi.append(max(0.0, s["ci_high"] - s["mean_ppl"]))
        if not xs:
            continue
        ax.bar(
            xs,
            heights,
            width=bar_width,
            label=policy,
            color=cmap(pi % 10),
            yerr=[err_lo, err_hi],
            capsize=3,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Wave-11 Perplexity by Policy (95% bootstrap CI)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="Policy", fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workspace-root",
        default=default_workspace_root(),
        help="EndurKV_workspace root (contains phone-logs/ and EndurKV/)",
    )
    ap.add_argument(
        "--phone-logs-root",
        default=None,
        help="Override phone-logs directory (default: <workspace>/phone-logs)",
    )
    ap.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resample count for 95%% CI (default: 1000)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for bootstrap reproducibility (default: 0)",
    )
    args = ap.parse_args()

    workspace_root = os.path.abspath(args.workspace_root)
    phone_logs_root = os.path.abspath(
        args.phone_logs_root
        if args.phone_logs_root is not None
        else os.path.join(workspace_root, "phone-logs")
    )

    endurkv_root = os.path.join(workspace_root, "EndurKV")
    table_path = os.path.join(
        endurkv_root, "figures", "master_tables", "TABLE_WAVE11_PPL.md"
    )
    plot_path = os.path.join(
        endurkv_root, "figures", "eval_plots", "ppl_per_policy.png"
    )

    print(f"[info] workspace_root  = {workspace_root}")
    print(f"[info] phone_logs_root = {phone_logs_root}")
    print(f"[info] table  -> {table_path}")
    print(f"[info] plot   -> {plot_path}")

    stats = aggregate(
        phone_logs_root=phone_logs_root,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    with open(table_path, "w") as f:
        f.write(render_markdown_table(stats))
    print(f"[ok] wrote {table_path}  ({len(stats)} cells)")

    render_bar_plot(stats, plot_path)
    if stats:
        print(f"[ok] wrote {plot_path}")
    else:
        print(f"[skip] no data; did not write {plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
