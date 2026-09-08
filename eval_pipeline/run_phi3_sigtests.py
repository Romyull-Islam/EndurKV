#!/usr/bin/env python3
"""Paired stat tests for the 4 Phi-3 PPL cells in wave11_eval_1780862534.

Pairing key: prompt_id (== chunk identifier shared across policies).
Tests: paired t on log(PPL) deltas, Wilcoxon signed-rank, Cohen's d_z.
Multiple-comparison correction: Holm–Bonferroni over the 6 pairs.

stdlib-only. Uses exact incomplete-beta for Student-t two-sided p, and
exact (enumeration) null distribution for Wilcoxon signed-rank when n <= 25.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from typing import Dict, List, Tuple

CELL_ROOT = ("/home/mislam22/EndurKV_workspace/phone-logs/"
             "wave11_eval_1780862534/Phi-3-mini-128k")
POLICIES = ["vanilla", "h2o", "tova", "v1_fa2_stack"]

PAIRS: List[Tuple[str, str]] = [
    ("h2o", "vanilla"),
    ("tova", "vanilla"),
    ("v1_fa2_stack", "vanilla"),
    ("h2o", "tova"),
    ("h2o", "v1_fa2_stack"),
    ("tova", "v1_fa2_stack"),
]

OUT_PATH = ("/home/mislam22/EndurKV_workspace/EndurKV/figures/"
            "master_tables/WAVE11_PHI3_SIGTESTS.md")


# --------------------------------------------------------------------------- #
# JSON loading (eviction_bench emits bareword inf/nan).
# --------------------------------------------------------------------------- #

_INFNAN = re.compile(
    r'(?<![A-Za-z0-9_."])(-?inf|nan)(?![A-Za-z0-9_])', re.IGNORECASE,
)


def _patch(raw: str) -> str:
    return _INFNAN.sub(
        lambda m: {"inf": "Infinity",
                   "-inf": "-Infinity",
                   "nan": "NaN"}[m.group(0).lower()],
        raw,
    )


def load_meta(path: str) -> dict:
    with open(path, "r") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_patch(raw))


# --------------------------------------------------------------------------- #
# Stats utilities (stdlib).
# --------------------------------------------------------------------------- #

def _betacf(a: float, b: float, x: float) -> float:
    """Lentz's algorithm for continued fraction of incomplete beta."""
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 1001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            return h
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf_twosided(t: float, df: float) -> float:
    """Two-sided p for Student-t."""
    if not math.isfinite(t) or df <= 0:
        return float("nan")
    x = df / (df + t * t)
    # P(|T| > |t|) = I_x(df/2, 1/2)
    return betai(df / 2.0, 0.5, x)


def paired_t(deltas: List[float]) -> Tuple[float, float, float]:
    """Return (t, two-sided p, df)."""
    n = len(deltas)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"))
    mu = sum(deltas) / n
    var = sum((d - mu) ** 2 for d in deltas) / (n - 1)
    if var <= 0:
        return (float("nan"), float("nan"), float(n - 1))
    se = math.sqrt(var / n)
    t = mu / se
    p = t_sf_twosided(t, n - 1)
    return (t, p, float(n - 1))


def wilcoxon_signed_rank(deltas: List[float]) -> Tuple[float, float, int]:
    """Two-sided Wilcoxon signed-rank.

    Returns (W_pos, p, n_used). Zero deltas are dropped (Wilcoxon convention).
    For n_used <= 25, the exact null distribution is enumerated; otherwise a
    normal approximation with continuity correction (and tie correction) is
    used.
    """
    nz = [d for d in deltas if d != 0.0]
    n = len(nz)
    if n < 1:
        return (float("nan"), float("nan"), 0)
    # Rank |d| with average ranks for ties.
    indexed = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[indexed[j + 1]]) == abs(nz[indexed[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    W_pos = sum(r for r, d in zip(ranks, nz) if d > 0)
    W_neg = sum(r for r, d in zip(ranks, nz) if d < 0)
    W_min = min(W_pos, W_neg)

    if n <= 25:
        # Exact enumeration over 2^n sign assignments of the ranks.
        # Probability under H0 of W+ <= W_min (two-sided = 2 * one-sided).
        # Use DP over the integer doubled ranks (handle ties: ranks may be x.5).
        scaled = [int(round(2 * r)) for r in ranks]
        total = sum(scaled)
        target = int(round(2 * W_min))
        # dp[s] = number of subsets with sum s.
        dp = [0] * (total + 1)
        dp[0] = 1
        for s in scaled:
            for v in range(total, s - 1, -1):
                dp[v] += dp[v - s]
        total_count = 2 ** n
        leq = sum(dp[:target + 1])
        p_one = leq / total_count
        p = min(1.0, 2.0 * p_one)
    else:
        # Normal approximation with tie correction + continuity.
        # Tie correction: sum over tied groups of (t^3 - t).
        from collections import Counter
        cnt = Counter(abs(d) for d in nz)
        tie_corr = sum(t * t * t - t for t in cnt.values() if t > 1)
        mu = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr / 48.0
        if var <= 0:
            return (W_pos, float("nan"), n)
        z = (abs(W_pos - mu) - 0.5) / math.sqrt(var)
        p = math.erfc(z / math.sqrt(2.0))
    return (W_pos, p, n)


def cohens_d_z(deltas: List[float]) -> float:
    n = len(deltas)
    if n < 2:
        return float("nan")
    mu = sum(deltas) / n
    var = sum((d - mu) ** 2 for d in deltas) / (n - 1)
    if var <= 0:
        return float("nan")
    return mu / math.sqrt(var)


def holm_bonferroni(ps: List[float]) -> List[float]:
    n = len(ps)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: ps[i])
    adj = [0.0] * n
    prev = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, max(prev, (n - rank) * ps[i]))
        adj[i] = a
        prev = a
    return adj


# --------------------------------------------------------------------------- #
# Load Phi-3 cells, indexed by prompt_id.
# --------------------------------------------------------------------------- #

def load_policy(policy: str) -> Dict[str, float]:
    """Return {prompt_id -> log(perplexity)} for a Phi-3 policy cell."""
    out: Dict[str, float] = {}
    pat = os.path.join(CELL_ROOT, policy, "ppl", "iter*", "meta.json")
    for m in sorted(glob.glob(pat)):
        meta = load_meta(m)
        pid = meta.get("prompt_id")
        nll = meta.get("mean_nll")
        ppl = meta.get("perplexity")
        if pid is None:
            continue
        v: float
        if isinstance(nll, (int, float)) and math.isfinite(float(nll)):
            v = float(nll)
        elif isinstance(ppl, (int, float)) and float(ppl) > 0 and math.isfinite(float(ppl)):
            v = math.log(float(ppl))
        else:
            continue
        out[pid] = v
    return out


def main() -> int:
    cells = {p: load_policy(p) for p in POLICIES}
    print("[info] cell sizes:",
          {p: len(c) for p, c in cells.items()}, file=sys.stderr)

    rows: List[dict] = []
    for A, B in PAIRS:
        a_map = cells[A]
        b_map = cells[B]
        shared = sorted(set(a_map) & set(b_map))
        deltas = [a_map[k] - b_map[k] for k in shared]
        n = len(deltas)
        if n < 2:
            rows.append({
                "a": A, "b": B, "n": n, "shared": shared,
                "mean_log_delta": float("nan"),
                "t": float("nan"), "t_p": float("nan"), "df": float("nan"),
                "w": float("nan"), "w_p": float("nan"), "w_n": n,
                "dz": float("nan"),
                "deltas": deltas,
            })
            continue
        t, t_p, df = paired_t(deltas)
        w, w_p, w_n = wilcoxon_signed_rank(deltas)
        dz = cohens_d_z(deltas)
        rows.append({
            "a": A, "b": B, "n": n, "shared": shared,
            "mean_log_delta": sum(deltas) / n,
            "t": t, "t_p": t_p, "df": df,
            "w": w, "w_p": w_p, "w_n": w_n,
            "dz": dz,
            "deltas": deltas,
        })

    # Holm–Bonferroni on the 6-test family, separately for t and Wilcoxon.
    t_ps = [r["t_p"] if math.isfinite(r["t_p"]) else 1.0 for r in rows]
    w_ps = [r["w_p"] if math.isfinite(r["w_p"]) else 1.0 for r in rows]
    t_adj = holm_bonferroni(t_ps)
    w_adj = holm_bonferroni(w_ps)
    for r, ta, wa in zip(rows, t_adj, w_adj):
        r["t_p_holm"] = ta
        r["w_p_holm"] = wa

    # ------------------------------------------------------------------ #
    # Render Markdown.
    # ------------------------------------------------------------------ #
    L: List[str] = []
    L.append("# Wave-11 Phi-3-mini-128k — Paired PPL Significance Tests")
    L.append("")
    L.append("**Source cells:** "
             "`phone-logs/wave11_eval_1780862534/Phi-3-mini-128k/"
             "{vanilla,h2o,tova,v1_fa2_stack}/ppl/iter*/meta.json`")
    L.append("")
    L.append("**Pairing key:** `prompt_id` (per-chunk identifier shared "
             "across policies). Deltas are computed only on chunks where "
             "BOTH policies have a finite `mean_nll`.")
    L.append("")
    L.append("**Tests per pair (A vs B), Δ = log(PPL_A) − log(PPL_B):**")
    L.append("- Paired Student t on the n deltas (df = n−1), two-sided, "
             "exact via regularized incomplete-beta.")
    L.append("- Wilcoxon signed-rank, two-sided. Exact enumeration of the "
             "null permutation distribution over the doubled (tie-averaged) "
             "ranks for n ≤ 25; otherwise normal approximation with tie + "
             "continuity correction.")
    L.append("- Cohen's d_z = mean(Δ) / sd(Δ) on the paired deltas.")
    L.append("")
    L.append("**Multiplicity:** Holm–Bonferroni over the 6 pairs (FWER ≤ "
             "0.05). Reported separately for t and Wilcoxon p-values.")
    L.append("")

    L.append("## Cell sizes (chunks with finite mean_nll)")
    L.append("")
    L.append("| Policy | n_chunks |")
    L.append("|---|---:|")
    for p in POLICIES:
        L.append(f"| {p} | {len(cells[p])} |")
    L.append("")

    L.append("## Per-pair test statistics")
    L.append("")
    L.append("| A | B | n | mean Δlog(PPL) | t | df | t_p | Holm t_p | "
             "W+ | W_p | Holm W_p | Cohen d_z |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        def f(x, fmt: str) -> str:
            if isinstance(x, (int, float)) and math.isfinite(float(x)):
                return fmt.format(float(x))
            return "n/a"
        L.append(
            "| {a} | {b} | {n} | {md} | {t} | {df} | {tp} | {tph} | "
            "{w} | {wp} | {wph} | {dz} |".format(
                a=r["a"], b=r["b"], n=r["n"],
                md=f(r["mean_log_delta"], "{:+.4f}"),
                t=f(r["t"], "{:+.3f}"),
                df=("{:.0f}".format(r["df"]) if math.isfinite(r["df"]) else "n/a"),
                tp=f(r["t_p"], "{:.4g}"),
                tph=f(r["t_p_holm"], "{:.4g}"),
                w=f(r["w"], "{:.1f}"),
                wp=f(r["w_p"], "{:.4g}"),
                wph=f(r["w_p_holm"], "{:.4g}"),
                dz=f(r["dz"], "{:+.3f}"),
            )
        )
    L.append("")

    # Significance verdicts after Holm.
    L.append("## Significance verdict (α = 0.05, after Holm correction)")
    L.append("")
    L.append("**Power note on Wilcoxon at small n.** With n ≤ 8 chunks and a "
             "family of 6 tests, the smallest two-sided Wilcoxon p the data "
             "can produce is 2/2^n: 0.0156 at n=7, 0.0078 at n=8. After "
             "Holm × 6 the floor is ~0.094 at n=7 and ~0.047 at n=8. Wilcoxon "
             "is therefore underpowered to *reject* Holm-corrected at n=7 no "
             "matter how strong the signal — the test is reported for "
             "completeness but the **paired-t is the primary inferential "
             "test** because log(PPL) deltas are approximately normal "
             "(n=7-8 cells, central limit support; t is robust to mild "
             "deviation).")
    L.append("")
    L.append("**Decision rule.** Primary: paired-t Holm-adjusted "
             "p-value < 0.05 ⇒ **significant**; otherwise "
             "**indistinguishable** at α = 0.05. The Wilcoxon Holm column is "
             "shown as a non-parametric sanity check; values flagged "
             "`floor` are at or below the n-induced detection floor and "
             "should not be interpreted as evidence of no effect.")
    L.append("")
    L.append("| A | B | Holm t_p | Holm W_p | Verdict (paired-t) | "
             "Wilcoxon sanity |")
    L.append("|---|---|---:|---:|---|---|")
    verdicts: Dict[Tuple[str, str], str] = {}
    for r in rows:
        tp = r["t_p_holm"]
        wp = r["w_p_holm"]
        tp_ok = math.isfinite(tp) and tp < 0.05
        wp_ok = math.isfinite(wp) and wp < 0.05
        v = "SIGNIFICANT" if tp_ok else "indistinguishable"
        verdicts[(r["a"], r["b"])] = v
        # Wilcoxon floor heuristic: if the raw two-sided p is at the n=2/2^n
        # minimum it's at the detection floor for that pair.
        n_used = r["w_n"]
        if isinstance(n_used, int) and n_used >= 1:
            min_raw = 2.0 / (2 ** n_used)
            at_floor = (math.isfinite(r["w_p"])
                        and r["w_p"] <= min_raw + 1e-9)
        else:
            at_floor = False
        w_flag = ("significant" if wp_ok
                  else ("floor (n={n}, raw p={p:.4f})".format(n=n_used, p=r["w_p"])
                        if at_floor else "ns"))
        L.append(f"| {r['a']} | {r['b']} | "
                 f"{tp:.4g} | {wp:.4g} | **{v}** | {w_flag} |")
    L.append("")

    # Per-pair delta tables (audit trail).
    L.append("## Paired deltas (audit trail)")
    L.append("")
    for r in rows:
        L.append(f"### {r['a']} vs {r['b']}  (n = {r['n']})")
        L.append("")
        L.append("| prompt_id | log(PPL_A) | log(PPL_B) | Δ |")
        L.append("|---|---:|---:|---:|")
        a_map = cells[r["a"]]
        b_map = cells[r["b"]]
        for pid in r["shared"]:
            L.append(f"| {pid} | {a_map[pid]:.4f} | {b_map[pid]:.4f} | "
                     f"{a_map[pid] - b_map[pid]:+.4f} |")
        L.append("")

    L.append("## Headline")
    L.append("")
    indist_vs_vanilla = [r for r in rows
                        if r["b"] == "vanilla"
                        and verdicts[(r["a"], r["b"])] == "indistinguishable"]
    sig_vs_vanilla = [r for r in rows
                      if r["b"] == "vanilla"
                      and verdicts[(r["a"], r["b"])] == "SIGNIFICANT"]
    if indist_vs_vanilla:
        names = ", ".join(r["a"] for r in indist_vs_vanilla)
        L.append(f"- **Statistically indistinguishable from vanilla "
                 f"(paired-t Holm, α=0.05):** {names}")
    if sig_vs_vanilla:
        details = ", ".join(
            f"{r['a']} (Δlog PPL = {r['mean_log_delta']:+.4f}, "
            f"d_z = {r['dz']:+.2f}, Holm t_p = {r['t_p_holm']:.3g})"
            for r in sig_vs_vanilla
        )
        L.append(f"- **Significantly different from vanilla "
                 f"(paired-t Holm, α=0.05):** {details}")
    L.append("")
    L.append("**Inter-policy verdicts (eviction policies against each other):**")
    for r in rows:
        if r["b"] == "vanilla":
            continue
        L.append(f"- {r['a']} vs {r['b']}: "
                 f"Δlog PPL = {r['mean_log_delta']:+.4f}, "
                 f"d_z = {r['dz']:+.2f}, "
                 f"Holm t_p = {r['t_p_holm']:.3g} ⇒ "
                 f"**{verdicts[(r['a'], r['b'])]}**.")
    L.append("")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(L))
    print(f"[ok] wrote {OUT_PATH}")
    # Print the headline rows to stdout for the orchestrator.
    print("VERDICTS:")
    for r in rows:
        print(f"  {r['a']} vs {r['b']}: "
              f"t_p_holm={r['t_p_holm']:.4g} "
              f"w_p_holm={r['w_p_holm']:.4g} "
              f"dz={r['dz']:+.3f} n={r['n']} -> "
              f"{verdicts[(r['a'], r['b'])]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
