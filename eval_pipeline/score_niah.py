#!/usr/bin/env python3
"""Score NIAH generations in phone-logs/wave11_*/<model>/<policy>/niah/<stim>/gen.txt.

Judge: "sandwich at Dolores Park" substring, plus GPT-4 when OPENAI_API_KEY is set.
Usage: score_niah.py [--phone-logs DIR] [--stim-dir DIR] [--repo-root DIR] [--no-gpt4]"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# matplotlib is needed for plots. Import it guarded so --help still works.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception as _mpl_err:  # pragma: no cover - environment-dependent
    HAS_MPL = False
    _MPL_ERR = _mpl_err

# Optional GPT-4 judge.
try:
    import openai  # type: ignore
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False


NEEDLE_FULL = ("The best thing to do in San Francisco is eat a sandwich at "
               "Dolores Park on a sunny day.")
NEEDLE_KEY = "sandwich at dolores park"  # case-insensitive substring match

CTX_LENGTHS = [2048, 4096, 6144, 8192]
DEPTHS = [0, 12, 25, 37, 50, 62, 75, 87]

STIM_NAME_RE = re.compile(r"^niah_c(?P<ctx>\d+)_d(?P<depth>\d+)$")
# The launcher names NIAH trial dirs iter<NN>, mapped to stimuli by ordinal.
ITER_NAME_RE = re.compile(r"^iter(?P<idx>\d+)$")


# Discovery

def find_gen_files(phone_logs_root: Path) -> List[Path]:
    """Find every wave11_*/<model>/<policy>/niah/<stim>/gen.txt."""
    pattern = str(phone_logs_root / "wave11_*" / "*" / "*" / "niah" / "*" / "gen.txt")
    return [Path(p) for p in sorted(glob.glob(pattern))]


def parse_gen_path(gen_path: Path, phone_logs_root: Path
                   ) -> Optional[Tuple[str, str, str, str, int, int]]:
    """Return (wave_dir, model, policy, stim_id, ctx, depth) for a gen.txt path, or None.
    For iter<NN> trial dirs, ctx holds the ordinal NN and depth is -1."""
    try:
        rel = gen_path.relative_to(phone_logs_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) != 6 or parts[3] != "niah" or parts[5] != "gen.txt":
        return None
    wave_dir, model, policy, _, stim_id, _ = parts

    m = STIM_NAME_RE.match(stim_id)
    if m:
        return (wave_dir, model, policy, stim_id,
                int(m.group("ctx")), int(m.group("depth")))

    it = ITER_NAME_RE.match(stim_id)
    if it:
        # Ordinal goes in the ctx slot with depth -1 so renderers bucket by ordinal.
        idx = int(it.group("idx"))
        return (wave_dir, model, policy, stim_id, idx, -1)
    return None


# Judges

_NEGATORS = (
    " not ", " no ", "n't ", " never ", " wrong", " incorrect",
    "cannot", "can not", "don't", "doesn't", "isn't", "wasn't", "won't",
    "there is no ", "there's no ", "no such", "no mention",
)


def rule_based_judge(gen_text: str) -> bool:
    """Needle substring match that fails if a negator appears in the same sentence within
    ~120 chars before it (e.g. "There is no sandwich at Dolores Park")."""
    t = gen_text.lower()
    idx = t.find(NEEDLE_KEY)
    if idx < 0:
        return False
    # Both content words must appear (already implied by idx >= 0).
    if "sandwich" not in t or "dolores park" not in t:
        return False

    # Sentence-bounded window ~120 chars before the match.
    window_start = max(0, idx - 120)
    window = t[window_start: idx + len(NEEDLE_KEY) + 40]

    # Trim to the local sentence (split on . ! ? \n) so a negator from a
    # previous sentence does not invalidate a positive answer.
    for sep in (". ", "! ", "? ", "\n"):
        last_sep = window.rfind(sep, 0, idx - window_start)
        if last_sep >= 0:
            window = window[last_sep + len(sep):]
            break

    for neg in _NEGATORS:
        if neg in window:
            return False
    return True


def gpt4_judge(gen_text: str, question: str = "What is the best thing to do in San Francisco?",
               model: str = "gpt-4o") -> Optional[bool]:
    """Ask GPT-4 whether the answer identifies the needle. Returns True/False, or None on
    error or any reply other than an exact YES or NO."""
    if not HAS_OPENAI:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    prompt = (
        "You are grading a question-answering system.\n"
        f"Question: {question}\n"
        f"Ground-truth answer (needle): \"{NEEDLE_FULL}\"\n"
        f"Assistant's answer: \"{gen_text.strip()[:2000]}\"\n\n"
        "Did the assistant correctly identify the needle (eating a sandwich at "
        "Dolores Park on a sunny day, in San Francisco)? "
        "Respond with exactly one token: YES or NO."
    )
    try:
        # New-style (>=1.0) openai client
        try:
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
            out = resp.choices[0].message.content.strip().upper()
        except AttributeError:
            # Legacy (<1.0) openai client
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
            out = resp["choices"][0]["message"]["content"].strip().upper()
        out = out.rstrip(".,!?;:").strip()
        if out == "YES":
            return True
        if out == "NO":
            return False
        sys.stderr.write(f"[gpt4_judge] ambiguous response: {out!r}\n")
        return None
    except Exception as e:
        sys.stderr.write(f"[gpt4_judge] error: {e}\n")
        return None


# Scoring

def score_all(gen_files: List[Path],
              phone_logs_root: Path,
              stim_dir: Path,
              use_gpt4: bool
              ) -> List[Dict]:
    """Returns a list of result dicts, one per gen.txt."""
    results: List[Dict] = []
    for gen_path in gen_files:
        parsed = parse_gen_path(gen_path, phone_logs_root)
        if parsed is None:
            sys.stderr.write(f"[skip] unparseable path: {gen_path}\n")
            continue
        wave_dir, model, policy, stim_id, ctx, depth = parsed

        # The stimulus body is not needed (the needle is fixed), but checking it exists
        # catches orphan generations.
        stim_path = stim_dir / f"{stim_id}.txt"
        stim_present = stim_path.is_file()

        try:
            gen_text = gen_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            sys.stderr.write(f"[skip] cannot read {gen_path}: {e}\n")
            continue

        rule_correct = rule_based_judge(gen_text)
        gpt4_correct: Optional[bool] = None
        if use_gpt4:
            gpt4_correct = gpt4_judge(gen_text)

        results.append({
            "wave_dir": wave_dir,
            "model": model,
            "policy": policy,
            "stim_id": stim_id,
            "ctx": ctx,
            "depth": depth,
            "stim_present": stim_present,
            "rule_correct": bool(rule_correct),
            "gpt4_correct": gpt4_correct,
            "gen_path": str(gen_path),
        })
    return results


# Aggregation

def aggregate(results: List[Dict]):
    """Return (per_cell, per_ctx, per_depth, grid). The first three map to {correct, total, acc}.
    grid[(model, policy)][(ctx, depth)] is the pass/fail bool."""
    per_cell: Dict[Tuple[str, str], Dict] = defaultdict(
        lambda: {"correct": 0, "total": 0})
    per_ctx: Dict[Tuple[str, str, int], Dict] = defaultdict(
        lambda: {"correct": 0, "total": 0})
    per_depth: Dict[Tuple[str, str, int], Dict] = defaultdict(
        lambda: {"correct": 0, "total": 0})
    grid: Dict[Tuple[str, str], Dict[Tuple[int, int], Optional[bool]]] = defaultdict(dict)

    for r in results:
        key = (r["model"], r["policy"])
        ok = r["rule_correct"]
        per_cell[key]["correct"] += int(ok)
        per_cell[key]["total"] += 1
        # Per-ctx, per-depth and grid tables only for trials with real (ctx, depth).
        # Ordinal iter<NN> trials (depth -1) count toward per-cell accuracy only.
        if r["depth"] >= 0:
            per_ctx[(*key, r["ctx"])]["correct"] += int(ok)
            per_ctx[(*key, r["ctx"])]["total"] += 1
            per_depth[(*key, r["depth"])]["correct"] += int(ok)
            per_depth[(*key, r["depth"])]["total"] += 1
            grid[key][(r["ctx"], r["depth"])] = ok

    def finalize(d):
        for k, v in d.items():
            v["acc"] = (v["correct"] / v["total"]) if v["total"] else 0.0
        return d

    return finalize(per_cell), finalize(per_ctx), finalize(per_depth), grid


# Reporting

def write_markdown_table(out_path: Path,
                         results: List[Dict],
                         per_cell: Dict,
                         per_ctx: Dict,
                         per_depth: Dict,
                         grid: Dict,
                         used_gpt4: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Wave-11 NIAH Judge Report")
    lines.append("")
    lines.append("Needle: \"" + NEEDLE_FULL + "\"")
    lines.append("")
    lines.append(f"Rule judge: case-insensitive substring \"{NEEDLE_KEY}\".  "
                 f"GPT-4 judge: {'enabled' if used_gpt4 else 'disabled'}.")
    lines.append("")
    lines.append(f"Total scored generations: **{len(results)}**  "
                 f"(grid = {len(CTX_LENGTHS)} ctx x {len(DEPTHS)} depth = "
                 f"{len(CTX_LENGTHS)*len(DEPTHS)} per (model, policy)).")
    lines.append("")

    # per-(model, policy) summary
    lines.append("## Per-(model, policy) accuracy")
    lines.append("")
    header = "| model | policy | correct | total | accuracy |"
    sep    = "|---|---|---:|---:|---:|"
    if used_gpt4:
        header = "| model | policy | correct (rule) | total | accuracy (rule) | accuracy (GPT-4) |"
        sep    = "|---|---|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for (model, policy), v in sorted(per_cell.items()):
        row = f"| {model} | {policy} | {v['correct']} | {v['total']} | {v['acc']*100:.1f}% |"
        if used_gpt4:
            gpt_correct = sum(
                1 for r in results
                if r["model"] == model and r["policy"] == policy
                and r["gpt4_correct"] is True)
            gpt_total = sum(
                1 for r in results
                if r["model"] == model and r["policy"] == policy
                and r["gpt4_correct"] is not None)
            gpt_acc = (gpt_correct / gpt_total * 100.0) if gpt_total else float("nan")
            row = (f"| {model} | {policy} | {v['correct']} | {v['total']} | "
                   f"{v['acc']*100:.1f}% | {gpt_acc:.1f}% ({gpt_correct}/{gpt_total}) |")
        lines.append(row)
    lines.append("")

    # per-context accuracy
    lines.append("## Accuracy by context length")
    lines.append("")
    ctx_header = "| model | policy | " + " | ".join(f"{c}" for c in CTX_LENGTHS) + " |"
    ctx_sep    = "|---|---|" + "|".join(["---:"] * len(CTX_LENGTHS)) + "|"
    lines.append(ctx_header)
    lines.append(ctx_sep)
    for (model, policy) in sorted(per_cell.keys()):
        cells = []
        for ctx in CTX_LENGTHS:
            v = per_ctx.get((model, policy, ctx))
            cells.append(f"{v['acc']*100:.0f}% ({v['correct']}/{v['total']})"
                         if v else "-")
        lines.append(f"| {model} | {policy} | " + " | ".join(cells) + " |")
    lines.append("")

    # per-depth accuracy
    lines.append("## Accuracy by depth percentile")
    lines.append("")
    depth_header = "| model | policy | " + " | ".join(f"{d}%" for d in DEPTHS) + " |"
    depth_sep    = "|---|---|" + "|".join(["---:"] * len(DEPTHS)) + "|"
    lines.append(depth_header)
    lines.append(depth_sep)
    for (model, policy) in sorted(per_cell.keys()):
        cells = []
        for d in DEPTHS:
            v = per_depth.get((model, policy, d))
            cells.append(f"{v['acc']*100:.0f}% ({v['correct']}/{v['total']})"
                         if v else "-")
        lines.append(f"| {model} | {policy} | " + " | ".join(cells) + " |")
    lines.append("")

    # per-(model, policy) text heatmap
    lines.append("## Pass/fail heatmaps (rows = ctx, cols = depth %)")
    lines.append("")
    lines.append("Legend: `O` = correct, `.` = wrong, `?` = missing.")
    lines.append("")
    for (model, policy) in sorted(grid.keys()):
        lines.append(f"### {model} / {policy}")
        lines.append("")
        lines.append("```")
        header_cols = "ctx \\ d% " + " ".join(f"{d:>3d}" for d in DEPTHS)
        lines.append(header_cols)
        for ctx in CTX_LENGTHS:
            row = [f"{ctx:>8d} "]
            for d in DEPTHS:
                v = grid[(model, policy)].get((ctx, d))
                if v is None:
                    row.append("  ?")
                else:
                    row.append("  O" if v else "  .")
            lines.append(" ".join(row))
        lines.append("```")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_heatmap(grid_cell: Dict[Tuple[int, int], Optional[bool]],
                 model: str, policy: str, out_path: Path) -> None:
    if not HAS_MPL:
        return
    n_rows = len(CTX_LENGTHS)
    n_cols = len(DEPTHS)
    mat = [[0.5] * n_cols for _ in range(n_rows)]  # 0.5 = missing
    for i, ctx in enumerate(CTX_LENGTHS):
        for j, d in enumerate(DEPTHS):
            v = grid_cell.get((ctx, d))
            if v is True:
                mat[i][j] = 1.0
            elif v is False:
                mat[i][j] = 0.0
            else:
                mat[i][j] = 0.5

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{d}%" for d in DEPTHS])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(c) for c in CTX_LENGTHS])
    ax.set_xlabel("Needle depth")
    ax.set_ylabel("Context length (tokens)")
    n_correct = sum(1 for v in grid_cell.values() if v is True)
    n_total = sum(1 for v in grid_cell.values() if v is not None)
    acc = (n_correct / n_total * 100.0) if n_total else 0.0
    ax.set_title(f"NIAH: {model} / {policy}  -  {n_correct}/{n_total} ({acc:.1f}%)")
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i][j]
            mark = "O" if v == 1.0 else ("." if v == 0.0 else "?")
            ax.text(j, i, mark, ha="center", va="center",
                    color="black", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, ticks=[0, 0.5, 1.0],
                 label="0=wrong  0.5=missing  1=correct")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_summary_bar(per_cell: Dict[Tuple[str, str], Dict], out_path: Path) -> None:
    if not HAS_MPL or not per_cell:
        return
    items = sorted(per_cell.items())
    labels = [f"{m}\n{p}" for (m, p), _ in items]
    accs = [v["acc"] * 100.0 for _, v in items]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(items) + 2), 4.5))
    bars = ax.bar(range(len(items)), accs, color="#3a78b4")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("NIAH accuracy (%)")
    ax.set_title("Wave-11 NIAH accuracy (rule-based judge) per (model, policy)")
    ax.grid(axis="y", alpha=0.3)
    for b, acc in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 1.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# Main

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phone-logs",
                    default="/home/mislam22/EndurKV_workspace/phone-logs",
                    help="Root of on-device log captures (default: %(default)s)")
    ap.add_argument("--repo-root",
                    default="/home/mislam22/EndurKV_workspace/EndurKV",
                    help="EndurKV repo root for figures/ output (default: %(default)s)")
    ap.add_argument("--stim-dir",
                    default=None,
                    help="NIAH stimulus dir (default: <repo-root>/eval_pipeline/data/niah)")
    ap.add_argument("--no-gpt4", action="store_true",
                    help="Skip the GPT-4 judge even if OPENAI_API_KEY is set.")
    ap.add_argument("--results-json", default=None,
                    help="Optional path to dump the raw per-trial results JSON.")
    args = ap.parse_args(argv)

    phone_logs_root = Path(args.phone_logs)
    repo_root = Path(args.repo_root)
    stim_dir = Path(args.stim_dir) if args.stim_dir else (
        repo_root / "eval_pipeline" / "data" / "niah")

    if not HAS_MPL:
        sys.stderr.write(
            f"[warn] matplotlib unavailable ({_MPL_ERR}); plots will be skipped.\n")

    use_gpt4 = (not args.no_gpt4) and HAS_OPENAI and bool(os.environ.get("OPENAI_API_KEY"))
    if not use_gpt4:
        reason = []
        if args.no_gpt4:
            reason.append("--no-gpt4")
        if not HAS_OPENAI:
            reason.append("openai package not importable")
        if not os.environ.get("OPENAI_API_KEY"):
            reason.append("OPENAI_API_KEY not set")
        sys.stderr.write(f"[info] GPT-4 judge skipped ({', '.join(reason) or 'unknown'}).\n")

    gen_files = find_gen_files(phone_logs_root)
    if not gen_files:
        sys.stderr.write(
            f"[error] no gen.txt files found under {phone_logs_root}/wave11_*.\n")
        # still write empty outputs so downstream consumers don't crash
    sys.stderr.write(f"[info] found {len(gen_files)} gen.txt files.\n")

    results = score_all(gen_files, phone_logs_root, stim_dir, use_gpt4=use_gpt4)
    per_cell, per_ctx, per_depth, grid = aggregate(results)

    md_path = repo_root / "figures" / "master_tables" / "TABLE_WAVE11_NIAH.md"
    write_markdown_table(md_path, results, per_cell, per_ctx, per_depth, grid,
                         used_gpt4=use_gpt4)
    sys.stderr.write(f"[ok] wrote {md_path}\n")

    plot_root = repo_root / "figures" / "eval_plots"
    for (model, policy), gcell in sorted(grid.items()):
        safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        safe_policy = re.sub(r"[^A-Za-z0-9._-]", "_", policy)
        png = plot_root / f"niah_heatmap_{safe_model}_{safe_policy}.png"
        plot_heatmap(gcell, model, policy, png)
        sys.stderr.write(f"[ok] wrote {png}\n")

    summary_png = plot_root / "niah_accuracy_summary.png"
    plot_summary_bar(per_cell, summary_png)
    sys.stderr.write(f"[ok] wrote {summary_png}\n")

    if args.results_json:
        out = Path(args.results_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        sys.stderr.write(f"[ok] wrote {out}\n")

    # Console summary.
    print(f"NIAH judge: {len(results)} generations across "
          f"{len(per_cell)} (model, policy) cells.")
    for (model, policy), v in sorted(per_cell.items()):
        print(f"  {model:<28s} {policy:<24s}  "
              f"rule acc = {v['correct']:>2d}/{v['total']:<2d} "
              f"({v['acc']*100:5.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
