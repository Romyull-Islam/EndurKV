#!/usr/bin/env python3
"""
score_sig.py — Paired significance tests + effect sizes for Wave-11 evals.

Consumes the same iter*/meta.json + gen.txt tree as score_ppl.py / score_niah.py
and emits:

  1. Per-policy-pair tests for PPL deltas on log domain (paired Wilcoxon
     signed-rank + paired t-test on log(PPL)) — paired across chunks within
     a (model, policy_a, policy_b) family. Effect size: Cohen's d_z on
     log(PPL) deltas.

  2. McNemar exact test on NIAH per-stimulus correctness contingency tables
     (between every pair of policies within a model). Effect size: McNemar
     odds ratio with mid-p CI.

  3. Holm–Bonferroni family-wise correction across (model × policy-pair)
     within each benchmark. A pre-registered PRIMARY contrast (default:
     v1_fa2_stack vs tova) is reported uncorrected as well.

Outputs a Markdown report:
    figures/master_tables/TABLE_WAVE11_SIGNIFICANCE.md

stdlib + matplotlib only. SciPy is preferred for accurate p-values; if
SciPy is missing, a stdlib normal-approximation fallback is used and a
warning is printed.

Usage:
    python eval_pipeline/score_sig.py \
        [--phone-logs-root /path/to/phone-logs] \
        [--workspace-root /path/to/EndurKV_workspace] \
        [--primary-contrast v1_fa2_stack:tova]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from scipy import stats as _scipy_stats  # type: ignore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def discover_ppl_metas(phone_logs_root: str) -> List[str]:
    pat = os.path.join(phone_logs_root, "wave11_*", "*", "*", "ppl", "iter*", "meta.json")
    files = sorted(glob.glob(pat))
    if files:
        return files
    # Older layout: no /ppl/ subdir.
    pat2 = os.path.join(phone_logs_root, "wave11_*", "*", "*", "iter*", "meta.json")
    return sorted(glob.glob(pat2))


def discover_niah_gens(phone_logs_root: str) -> List[str]:
    pat = os.path.join(phone_logs_root, "wave11_*", "*", "*", "niah", "*", "gen.txt")
    return sorted(glob.glob(pat))


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_ppl_path(meta_path: str, root: str
                   ) -> Optional[Tuple[str, str, str, str]]:
    rel = os.path.relpath(meta_path, root)
    parts = rel.split(os.sep)
    # Two possible layouts:
    #   wave/<model>/<policy>/ppl/iterNNNN/meta.json   (5 parts after wave)
    #   wave/<model>/<policy>/iterNNNN/meta.json
    if len(parts) >= 6 and parts[3] == "ppl":
        return parts[0], parts[1], parts[2], parts[4]
    if len(parts) == 5:
        return parts[0], parts[1], parts[2], parts[3]
    return None


def parse_niah_path(gen_path: str, root: str
                    ) -> Optional[Tuple[str, str, str, str]]:
    rel = os.path.relpath(gen_path, root)
    parts = rel.split(os.sep)
    if len(parts) >= 6 and parts[3] == "niah":
        return parts[0], parts[1], parts[2], parts[4]
    return None


def _load_meta_tolerant(meta_path: str) -> Optional[dict]:
    """eviction_bench emits bareword `inf`/`nan` which is not valid RFC-8259
    JSON; substitute with the json-permissive Infinity/NaN tokens before
    parsing. See score_ppl._load_meta_tolerant for the canonical version."""
    try:
        with open(meta_path, "r") as f:
            raw = f.read()
    except OSError:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
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
    except json.JSONDecodeError:
        return None


def load_log_ppl(meta_path: str) -> Optional[float]:
    meta = _load_meta_tolerant(meta_path)
    if meta is None:
        return None
    # Prefer mean_nll (log space) directly.
    nll = meta.get("mean_nll")
    if nll is not None:
        try:
            v = float(nll)
            if math.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass
    ppl = meta.get("perplexity")
    if ppl is None:
        return None
    try:
        p = float(ppl)
    except (TypeError, ValueError):
        return None
    if p <= 0 or not math.isfinite(p):
        return None
    return math.log(p)


def needle_in(text: str) -> bool:
    return "sandwich at dolores park" in text.lower()


# --------------------------------------------------------------------------- #
# Stats — paired tests + effect sizes
# --------------------------------------------------------------------------- #

def paired_t_test(deltas: List[float]) -> Tuple[float, float]:
    """Return (t_stat, two-sided p) for paired-t on deltas vs 0."""
    n = len(deltas)
    if n < 2:
        return (float("nan"), float("nan"))
    mu = sum(deltas) / n
    var = sum((d - mu) ** 2 for d in deltas) / (n - 1)
    if var <= 0:
        return (float("nan"), float("nan"))
    se = math.sqrt(var / n)
    t = mu / se
    if HAS_SCIPY:
        p = float(_scipy_stats.t.sf(abs(t), n - 1) * 2.0)
    else:
        # Normal approximation; underestimates p at small n. Print warning.
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return (t, p)


def wilcoxon_signed_rank(deltas: List[float]) -> Tuple[float, float]:
    """Return (W_stat, two-sided p). Uses SciPy when available."""
    nonzero = [d for d in deltas if d != 0.0]
    if len(nonzero) < 2:
        return (float("nan"), float("nan"))
    if HAS_SCIPY:
        try:
            w = _scipy_stats.wilcoxon(nonzero, zero_method="wilcox",
                                      alternative="two-sided", mode="auto")
            return (float(w.statistic), float(w.pvalue))
        except Exception:
            pass
    # Stdlib fallback: ranks-based normal approximation (Wilcoxon).
    abs_vals = sorted([(abs(d), i) for i, d in enumerate(nonzero)])
    ranks = [0.0] * len(nonzero)
    i = 0
    while i < len(abs_vals):
        j = i
        while j + 1 < len(abs_vals) and abs_vals[j + 1][0] == abs_vals[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[abs_vals[k][1]] = avg_rank
        i = j + 1
    W_pos = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    n = len(nonzero)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd <= 0:
        return (W_pos, float("nan"))
    z = (W_pos - mu) / sd
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return (W_pos, p)


def cohens_d_z(deltas: List[float]) -> float:
    """Paired Cohen's d (= d_z) on the delta sample."""
    n = len(deltas)
    if n < 2:
        return float("nan")
    mu = sum(deltas) / n
    var = sum((d - mu) ** 2 for d in deltas) / (n - 1)
    if var <= 0:
        return float("nan")
    return mu / math.sqrt(var)


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar's exact (mid-p) test on a 2x2 paired contingency table.

    b = # stimuli where A correct, B incorrect
    c = # stimuli where A incorrect, B correct
    Returns two-sided p-value.
    """
    n = b + c
    if n == 0:
        return 1.0
    if HAS_SCIPY:
        # Exact binomial two-sided.
        return float(_scipy_stats.binomtest(min(b, c), n, p=0.5).pvalue)
    # Stdlib fallback: exact two-sided binomial via cumulative sum.
    from math import comb
    k = min(b, c)
    cdf = 0.0
    for i in range(k + 1):
        cdf += comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * cdf)


def mcnemar_odds_ratio(b: int, c: int) -> float:
    if c <= 0:
        return float("inf") if b > 0 else float("nan")
    return b / c


def holm_bonferroni(pvals: List[float]) -> List[float]:
    """Holm–Bonferroni adjusted p-values, monotonic, capped at 1.0."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, max(prev, (n - rank) * pvals[i]))
        adj[i] = a
        prev = a
    return adj


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def aggregate_ppl(root: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Return {(model, policy): {iter_id -> log_ppl}}."""
    metas = discover_ppl_metas(root)
    out: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for m in metas:
        parsed = parse_ppl_path(m, root)
        if not parsed:
            continue
        _wave, model, policy, iter_id = parsed
        v = load_log_ppl(m)
        if v is None:
            continue
        out[(model, policy)][iter_id] = v
    return out


def aggregate_niah(root: str) -> Dict[Tuple[str, str], Dict[str, bool]]:
    """Return {(model, policy): {stim_id -> correct?}}."""
    gens = discover_niah_gens(root)
    out: Dict[Tuple[str, str], Dict[str, bool]] = defaultdict(dict)
    for g in gens:
        parsed = parse_niah_path(g, root)
        if not parsed:
            continue
        _wave, model, policy, stim_id = parsed
        try:
            text = open(g, "r", encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        out[(model, policy)][stim_id] = needle_in(text)
    return out


def paired_ppl_tests(ppl: Dict[Tuple[str, str], Dict[str, float]]
                     ) -> List[Dict]:
    """For every (model, A, B) policy pair, compute paired tests on
    log(PPL_A) - log(PPL_B) across the chunks present in BOTH cells."""
    by_model: Dict[str, List[str]] = defaultdict(list)
    for (model, policy) in ppl.keys():
        by_model[model].append(policy)
    out: List[Dict] = []
    for model, policies in by_model.items():
        policies = sorted(set(policies))
        for i in range(len(policies)):
            for j in range(i + 1, len(policies)):
                A, B = policies[i], policies[j]
                a_map = ppl.get((model, A), {})
                b_map = ppl.get((model, B), {})
                shared = sorted(set(a_map.keys()) & set(b_map.keys()))
                if len(shared) < 2:
                    continue
                deltas = [a_map[k] - b_map[k] for k in shared]
                t_stat, t_p = paired_t_test(deltas)
                w_stat, w_p = wilcoxon_signed_rank(deltas)
                d = cohens_d_z(deltas)
                out.append({
                    "model": model, "a": A, "b": B,
                    "n": len(deltas),
                    "mean_log_delta": sum(deltas)/len(deltas),
                    "t_stat": t_stat, "t_p": t_p,
                    "w_stat": w_stat, "w_p": w_p,
                    "cohens_d_z": d,
                })
    return out


def paired_niah_tests(niah: Dict[Tuple[str, str], Dict[str, bool]]
                      ) -> List[Dict]:
    by_model: Dict[str, List[str]] = defaultdict(list)
    for (model, policy) in niah.keys():
        by_model[model].append(policy)
    out: List[Dict] = []
    for model, policies in by_model.items():
        policies = sorted(set(policies))
        for i in range(len(policies)):
            for j in range(i + 1, len(policies)):
                A, B = policies[i], policies[j]
                a_map = niah.get((model, A), {})
                b_map = niah.get((model, B), {})
                shared = sorted(set(a_map.keys()) & set(b_map.keys()))
                if not shared:
                    continue
                b_only = sum(1 for k in shared if a_map[k] and not b_map[k])
                c_only = sum(1 for k in shared if (not a_map[k]) and b_map[k])
                p = mcnemar_exact(b_only, c_only)
                odds = mcnemar_odds_ratio(b_only, c_only)
                out.append({
                    "model": model, "a": A, "b": B,
                    "n_shared": len(shared),
                    "b_only": b_only, "c_only": c_only,
                    "mcnemar_p": p, "odds_ratio": odds,
                })
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_markdown(ppl_tests: List[Dict], niah_tests: List[Dict],
                    primary: Optional[Tuple[str, str]],
                    out_path: str) -> None:
    lines: List[str] = []
    lines.append("# Wave-11 Significance Tests")
    lines.append("")
    if not HAS_SCIPY:
        lines.append("> **Note**: SciPy not installed — paired t p-values "
                     "use a normal approximation. Install scipy for exact "
                     "t-distribution / Wilcoxon / binomial p-values.")
        lines.append("")
    lines.append("All tests are PAIRED on the per-chunk (or per-stimulus) ID, "
                 "i.e. the same chunk is scored by both policies in each pair.")
    lines.append("")

    # PPL
    lines.append("## PPL (log-domain paired tests)")
    lines.append("")
    if ppl_tests:
        # Apply Holm–Bonferroni per model, using Wilcoxon p (non-parametric primary).
        by_model: Dict[str, List[Dict]] = defaultdict(list)
        for r in ppl_tests:
            by_model[r["model"]].append(r)
        for model in sorted(by_model.keys()):
            family = by_model[model]
            ps = [r["w_p"] for r in family if math.isfinite(r["w_p"])]
            valid = [r for r in family if math.isfinite(r["w_p"])]
            adj = holm_bonferroni(ps)
            for r, p_adj in zip(valid, adj):
                r["w_p_holm"] = p_adj
        lines.append(
            "| Model | A | B | n | mean Δlog(PPL) | t | t_p | W | W_p | "
            "Holm-W_p | Cohen d_z |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(ppl_tests, key=lambda x: (x["model"], x["a"], x["b"])):
            lines.append(
                "| {model} | {a} | {b} | {n} | {md:.4f} | {t:.3f} | {tp:.4g} | "
                "{w:.3f} | {wp:.4g} | {wh:.4g} | {dz:.3f} |".format(
                    model=r["model"], a=r["a"], b=r["b"], n=r["n"],
                    md=r["mean_log_delta"],
                    t=r["t_stat"], tp=r["t_p"],
                    w=r["w_stat"], wp=r["w_p"],
                    wh=r.get("w_p_holm", float("nan")),
                    dz=r["cohens_d_z"],
                ),
            )
    else:
        lines.append("_No PPL pair data._")
    lines.append("")

    # NIAH
    lines.append("## NIAH (McNemar exact)")
    lines.append("")
    if niah_tests:
        by_model_n: Dict[str, List[Dict]] = defaultdict(list)
        for r in niah_tests:
            by_model_n[r["model"]].append(r)
        for model in sorted(by_model_n.keys()):
            family = by_model_n[model]
            ps = [r["mcnemar_p"] for r in family]
            adj = holm_bonferroni(ps)
            for r, p_adj in zip(family, adj):
                r["mcnemar_p_holm"] = p_adj
        lines.append(
            "| Model | A | B | n_shared | b (A&!B) | c (!A&B) | McNemar p | "
            "Holm p | OR (b/c) |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for r in sorted(niah_tests, key=lambda x: (x["model"], x["a"], x["b"])):
            lines.append(
                "| {model} | {a} | {b} | {n} | {bo} | {co} | {p:.4g} | "
                "{ph:.4g} | {orv} |".format(
                    model=r["model"], a=r["a"], b=r["b"], n=r["n_shared"],
                    bo=r["b_only"], co=r["c_only"], p=r["mcnemar_p"],
                    ph=r.get("mcnemar_p_holm", float("nan")),
                    orv=(f"{r['odds_ratio']:.2f}"
                         if math.isfinite(r["odds_ratio"]) else "inf"),
                ),
            )
    else:
        lines.append("_No NIAH pair data._")
    lines.append("")

    # Primary contrast
    if primary is not None:
        A, B = primary
        lines.append(f"## Pre-registered primary contrast: {A} vs {B}")
        lines.append("")
        prim_ppl = [r for r in ppl_tests
                    if {r["a"], r["b"]} == {A, B}]
        prim_niah = [r for r in niah_tests
                     if {r["a"], r["b"]} == {A, B}]
        if prim_ppl:
            for r in prim_ppl:
                lines.append(
                    f"- {r['model']} PPL (Wilcoxon): n={r['n']}, "
                    f"mean Δlog(PPL)={r['mean_log_delta']:.4f}, "
                    f"p={r['w_p']:.4g}, Cohen d_z={r['cohens_d_z']:.3f}"
                )
        if prim_niah:
            for r in prim_niah:
                lines.append(
                    f"- {r['model']} NIAH (McNemar exact): n_shared={r['n_shared']}, "
                    f"b={r['b_only']}, c={r['c_only']}, "
                    f"p={r['mcnemar_p']:.4g}"
                )
        if not (prim_ppl or prim_niah):
            lines.append("_No data for the primary contrast under either benchmark._")
        lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def default_workspace() -> str:
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace-root", default=default_workspace())
    ap.add_argument("--phone-logs-root", default=None)
    ap.add_argument("--primary-contrast",
                    default="v1_fa2_stack:tova",
                    help="Pre-registered primary policy pair `A:B` "
                         "(default: v1_fa2_stack:tova).")
    args = ap.parse_args()

    ws = os.path.abspath(args.workspace_root)
    root = os.path.abspath(args.phone_logs_root
                           or os.path.join(ws, "phone-logs"))

    print(f"[info] workspace = {ws}")
    print(f"[info] phone_logs = {root}")
    if not HAS_SCIPY:
        print("[warn] SciPy not installed — using normal-approximation fallbacks.",
              file=sys.stderr)

    ppl = aggregate_ppl(root)
    niah = aggregate_niah(root)
    ppl_tests = paired_ppl_tests(ppl)
    niah_tests = paired_niah_tests(niah)

    primary: Optional[Tuple[str, str]] = None
    if ":" in args.primary_contrast:
        a, b = args.primary_contrast.split(":", 1)
        primary = (a.strip(), b.strip())

    out_path = os.path.join(ws, "EndurKV", "figures", "master_tables",
                            "TABLE_WAVE11_SIGNIFICANCE.md")
    render_markdown(ppl_tests, niah_tests, primary, out_path)
    print(f"[ok] wrote {out_path}  "
          f"({len(ppl_tests)} PPL pair tests, {len(niah_tests)} NIAH pair tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
