#!/usr/bin/env python3
"""
host_plot_perchunk_delta.py — per-chunk PPL delta plot for Wave-11 (live).

Walks iter0000..iter000N/meta.json under
  <run_dir>/<model>/<policy>/ppl/
for an arbitrary list of policies, extracts "perplexity" per chunk, and emits:

  Top:    grouped bar chart of per-chunk PPL — one bar per policy per chunk
          (vanilla blue, h2o green, v1_fa2_stack orange, additional policies
          cycle through the matplotlib default cycle).
  Middle: per-chunk Δ_PPL (absolute, policy − vanilla) for each non-vanilla
          policy. The vanilla policy is always the baseline; the script
          requires "vanilla" to be in the policy list.
  Bottom: per-chunk Δ_log_PPL = log(ppl_policy) − log(ppl_vanilla)
          (the metric H2O paper / Zhang et al. Table 1 reports).

For the top grouped chart we include EVERY chunk that appears in ANY policy;
missing values are drawn as gaps. For the Δ panels we only show chunks where
the comparison policy AND vanilla both have a parseable value.

Also writes a markdown table summarising per-policy paired chunks vs vanilla.

Inputs may include bare-token `inf`/`-inf`/`nan` in meta.json (llama.cpp
default printf), so we sanitise before json.loads.

Usage:
  python host_plot_perchunk_delta.py \
      --run-dir /home/mislam22/EndurKV_workspace/phone-logs/wave11_eval_1780862534 \
      --model Phi-3-mini-128k \
      --policies vanilla,h2o,v1_fa2_stack,tova \
      --out-png /home/mislam22/EndurKV_workspace/EndurKV/figures/eval_plots/wave11_perchunk_ppl_delta.png \
      --out-md  /home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/WAVE11_PHI3_PPL_LIVE.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Colour map (vanilla blue, h2o green, v1_fa2_stack orange; extend as needed)
# ---------------------------------------------------------------------------
POLICY_COLORS: dict[str, str] = {
    "vanilla":      "#1f77b4",  # blue
    "h2o":          "#2ca02c",  # green
    "v1_fa2_stack": "#ff7f0e",  # orange
    "tova":         "#9467bd",  # purple
    "streamingllm": "#8c564b",  # brown
    "v1":           "#e377c2",  # pink
    "v1_fa2":       "#17becf",  # cyan
}
_CYCLE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
          "#8c564b", "#e377c2", "#17becf", "#bcbd22"]


def color_for(policy: str, idx: int) -> str:
    return POLICY_COLORS.get(policy, _CYCLE[idx % len(_CYCLE)])


# ---------------------------------------------------------------------------
# meta.json loader (sanitises bare inf/-inf/nan tokens emitted by llama.cpp)
# ---------------------------------------------------------------------------
_BARE_NUM = re.compile(r":\s*(-?inf|nan)\b", re.IGNORECASE)


def load_meta(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        raw = path.read_text()
    except OSError as e:
        print(f"[warn] could not read {path}: {e}", file=sys.stderr)
        return None

    def repl(m: re.Match) -> str:
        tok = m.group(1).lower()
        return f': "__{tok}__"'

    sanitised = _BARE_NUM.sub(repl, raw)
    try:
        d = json.loads(sanitised)
    except json.JSONDecodeError as e:
        print(f"[warn] could not parse {path}: {e}", file=sys.stderr)
        return None

    def fix(v):
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fix(x) for x in v]
        if v == "__inf__":
            return math.inf
        if v == "__-inf__":
            return -math.inf
        if v == "__nan__":
            return math.nan
        return v

    return fix(d)


# ---------------------------------------------------------------------------
# walk iter0000..iter000N for a single policy
# ---------------------------------------------------------------------------
def collect_policy(policy_dir: Path, max_iters: int = 8) -> dict[int, float]:
    """Return {chunk_idx: perplexity} for iters that have a parseable meta.json."""
    out: dict[int, float] = {}
    if not policy_dir.is_dir():
        return out
    for i in range(max_iters):
        meta = policy_dir / f"iter{i:04d}" / "meta.json"
        d = load_meta(meta)
        if d is None:
            continue
        ppl = d.get("perplexity")
        if ppl is None:
            continue
        try:
            ppl_f = float(ppl)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ppl_f):
            print(f"[warn] iter{i:04d} {policy_dir.parent.parent.name}/"
                  f"{policy_dir.parent.name}: non-finite ppl={ppl_f}",
                  file=sys.stderr)
            continue
        out[i] = ppl_f
    return out


# ---------------------------------------------------------------------------
# plotting (3-panel, N-policy)
# ---------------------------------------------------------------------------
def make_plot(policy_data: dict[str, dict[int, float]],
              policies: list[str],
              out_png: Path,
              model: str,
              run_id: str,
              baseline: str = "vanilla") -> None:
    # Universe of chunks = union across all policies
    all_chunks = sorted({c for d in policy_data.values() for c in d.keys()})
    if not all_chunks:
        return
    x = np.arange(len(all_chunks))
    n_pol = len(policies)
    # Top panel: grouped bars with one bar per policy per chunk
    total_group_width = 0.82
    w = total_group_width / n_pol

    fig, axes = plt.subplots(
        3, 1,
        figsize=(max(8.0, 1.4 * len(all_chunks) + 4), 11),
        sharex=True, constrained_layout=True,
    )

    # ----------------------- TOP: grouped per-chunk PPL --------------------
    ax0 = axes[0]
    for pi, pol in enumerate(policies):
        col = color_for(pol, pi)
        vals = [policy_data[pol].get(c, np.nan) for c in all_chunks]
        # offset bars symmetrically around the chunk centre
        offset = (pi - (n_pol - 1) / 2.0) * w
        xs = x + offset
        # NaNs would draw zero-height bars; filter for the bar call and
        # annotations
        bar_xs = []
        bar_vs = []
        for xi, v in zip(xs, vals):
            if np.isfinite(v):
                bar_xs.append(xi)
                bar_vs.append(v)
        if bar_xs:
            ax0.bar(bar_xs, bar_vs, width=w * 0.95, color=col, label=pol)
            for xi, v in zip(bar_xs, bar_vs):
                ax0.text(xi, v, f"{v:.2f}", ha="center", va="bottom",
                         fontsize=7, color=col)
    ax0.set_ylabel("Perplexity")
    ax0.set_title(f"Per-chunk PPL — {model}  ({run_id}, live partial)")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, axis="y", alpha=0.3)

    # ----------------------- MIDDLE: absolute ΔPPL --------------------------
    ax1 = axes[1]
    deltas = [p for p in policies if p != baseline]
    n_d = max(1, len(deltas))
    w_d = total_group_width / n_d
    base_vals = policy_data.get(baseline, {})
    legend_handles = []
    for di, pol in enumerate(deltas):
        col = color_for(pol, policies.index(pol))
        xs_d, vs_d = [], []
        for ci, c in enumerate(all_chunks):
            if c not in base_vals or c not in policy_data[pol]:
                continue
            d_abs = policy_data[pol][c] - base_vals[c]
            offset = (di - (n_d - 1) / 2.0) * w_d
            xs_d.append(x[ci] + offset)
            vs_d.append(d_abs)
        if xs_d:
            bars = ax1.bar(xs_d, vs_d, width=w_d * 0.95, color=col,
                           edgecolor="black", linewidth=0.4,
                           label=f"{pol} − {baseline}")
            legend_handles.append(bars)
            for xi, v in zip(xs_d, vs_d):
                ax1.text(xi, v, f"{v:+.3f}", ha="center",
                         va="bottom" if v >= 0 else "top", fontsize=7,
                         color=col)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel(r"$\Delta$PPL  (policy − vanilla)")
    ax1.set_title("Per-chunk absolute PPL delta  (positive = policy worse "
                  "than vanilla)")
    ax1.grid(True, axis="y", alpha=0.3)
    if legend_handles:
        ax1.legend(loc="upper right", fontsize=9)

    # ----------------------- BOTTOM: Δlog PPL ------------------------------
    ax2 = axes[2]
    legend_handles2 = []
    for di, pol in enumerate(deltas):
        col = color_for(pol, policies.index(pol))
        xs_d, vs_d = [], []
        for ci, c in enumerate(all_chunks):
            if c not in base_vals or c not in policy_data[pol]:
                continue
            v_pol = policy_data[pol][c]
            v_base = base_vals[c]
            if v_pol <= 0 or v_base <= 0:
                continue
            d_log = math.log(v_pol) - math.log(v_base)
            offset = (di - (n_d - 1) / 2.0) * w_d
            xs_d.append(x[ci] + offset)
            vs_d.append(d_log)
        if xs_d:
            bars = ax2.bar(xs_d, vs_d, width=w_d * 0.95, color=col,
                           edgecolor="black", linewidth=0.4,
                           label=f"{pol} − {baseline}")
            legend_handles2.append(bars)
            for xi, v in zip(xs_d, vs_d):
                ax2.text(xi, v, f"{v:+.4f}", ha="center",
                         va="bottom" if v >= 0 else "top", fontsize=7,
                         color=col)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel(r"$\Delta\log$PPL  (policy − vanilla)")
    ax2.set_title(r"Per-chunk $\Delta\log$PPL  (Zhang et al. Table 1 metric)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"chunk {c}" for c in all_chunks])
    ax2.set_xlabel("WikiText-2 PPL chunk")
    ax2.grid(True, axis="y", alpha=0.3)
    if legend_handles2:
        ax2.legend(loc="upper right", fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# markdown table (one block per non-baseline policy + raw long table)
# ---------------------------------------------------------------------------
def write_table(policy_data: dict[str, dict[int, float]],
                policies: list[str],
                out_md: Path,
                model: str,
                run_id: str,
                run_dir: Path,
                baseline: str = "vanilla") -> None:
    lines: list[str] = []
    lines.append(f"# Wave-11 live per-chunk PPL ({model})")
    lines.append("")
    lines.append(f"- Run id: `{run_id}`")
    lines.append(f"- Source: `{run_dir}/{model}/<policy>/ppl/iter*/meta.json`")
    counts = ", ".join(
        f"{p}={len(policy_data.get(p, {}))}" for p in policies
    )
    lines.append(f"- Policies & chunk counts: {counts}")
    lines.append("")

    base_vals = policy_data.get(baseline, {})

    # Wide per-chunk table: chunk | ppl_<policy>... | (no deltas, just raw)
    all_chunks = sorted({c for d in policy_data.values() for c in d.keys()})
    if all_chunks:
        header = "| chunk |"
        sep    = "|------:|"
        for p in policies:
            header += f" ppl_{p} |"
            sep    += "---------:|"
        lines.append(header)
        lines.append(sep)
        for c in all_chunks:
            row = f"| {c} |"
            for p in policies:
                v = policy_data.get(p, {}).get(c)
                row += f" {v:.4f} |" if v is not None else " — |"
            lines.append(row)

        # mean row (per policy, over the policy's own chunks)
        mean_row = "| **mean** |"
        for p in policies:
            vs = list(policy_data.get(p, {}).values())
            mean_row += (f" **{(sum(vs)/len(vs)):.4f}** |"
                         if vs else " **—** |")
        lines.append(mean_row)
        lines.append("")

    # Per non-baseline policy: deltas vs vanilla
    for pol in policies:
        if pol == baseline:
            continue
        pol_vals = policy_data.get(pol, {})
        common = sorted(set(base_vals.keys()) & set(pol_vals.keys()))
        lines.append(f"## {pol} vs {baseline}")
        lines.append("")
        if not common:
            lines.append(f"_No overlapping chunks yet for `{pol}` vs "
                         f"`{baseline}`._")
            lines.append("")
            continue
        lines.append(f"- Paired chunks: **{len(common)}**")
        lines.append("")
        lines.append("| chunk | ppl_vanilla | ppl_{pol} | "
                     "Δ_PPL (abs) | Δ_PPL (%) | Δ_log_PPL |"
                     .replace("{pol}", pol))
        lines.append("|------:|------------:|----------:|"
                     "------------:|----------:|---------:|")
        d_abs_list, d_pct_list, d_log_list = [], [], []
        for c in common:
            v = base_vals[c]
            h = pol_vals[c]
            d_abs = h - v
            d_pct = 100.0 * (h - v) / v if v != 0 else float("nan")
            d_log = math.log(h) - math.log(v) if (h > 0 and v > 0) else float("nan")
            d_abs_list.append(d_abs)
            d_pct_list.append(d_pct)
            d_log_list.append(d_log)
            lines.append(f"| {c} | {v:.4f} | {h:.4f} | "
                         f"{d_abs:+.4f} | {d_pct:+.2f}% | {d_log:+.4f} |")
        # summary row
        mean_van = sum(base_vals[c] for c in common) / len(common)
        mean_pol = sum(pol_vals[c]  for c in common) / len(common)
        mean_abs = sum(d_abs_list) / len(d_abs_list)
        mean_pct = sum(d_pct_list) / len(d_pct_list)
        mean_log = sum(d_log_list) / len(d_log_list)
        lines.append(f"| **mean** | **{mean_van:.4f}** | "
                     f"**{mean_pol:.4f}** | **{mean_abs:+.4f}** | "
                     f"**{mean_pct:+.2f}%** | **{mean_log:+.4f}** |")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `Δ_log_PPL = log(ppl_policy) − log(ppl_vanilla)` is the "
                 "form Zhang et al. (H2O, NeurIPS 2023) report in Table 1.")
    lines.append("- Positive Δ = the policy is **worse** (higher PPL) than "
                 "vanilla.")
    lines.append("- This table is **live partial** — regenerate as more iters "
                 "land.")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=Path("/home/mislam22/EndurKV_workspace/phone-logs/"
                                 "wave11_eval_1780862534"))
    ap.add_argument("--model", default="Phi-3-mini-128k")
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument("--policies", default="vanilla,h2o,v1_fa2_stack,tova",
                    help="comma-separated list of policy dirs to plot "
                         "(first must be the baseline, default 'vanilla')")
    ap.add_argument("--baseline", default="vanilla",
                    help="policy to use as the baseline for delta panels")
    ap.add_argument("--extra-policy-dir", action="append", default=[],
                    help="name=path override for a policy's ppl directory. "
                         "Useful when v1_fa2_stack lives outside the main "
                         "wave11 run-dir (repeatable).")
    ap.add_argument("--out-png", type=Path,
                    default=Path("/home/mislam22/EndurKV_workspace/EndurKV/"
                                 "figures/eval_plots/"
                                 "wave11_perchunk_ppl_delta.png"))
    ap.add_argument("--out-md", type=Path,
                    default=Path("/home/mislam22/EndurKV_workspace/EndurKV/"
                                 "figures/master_tables/"
                                 "WAVE11_PHI3_PPL_LIVE.md"))
    args = ap.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if args.baseline not in policies:
        print(f"[err] baseline {args.baseline!r} not in --policies "
              f"{policies!r}", file=sys.stderr)
        return 2

    base = args.run_dir / args.model

    # Parse override map
    overrides: dict[str, Path] = {}
    for spec in args.extra_policy_dir:
        if "=" not in spec:
            print(f"[err] --extra-policy-dir expects name=path, got {spec!r}",
                  file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        overrides[name.strip()] = Path(path.strip())

    policy_data: dict[str, dict[int, float]] = {}
    for p in policies:
        pdir = overrides.get(p, base / p / "ppl")
        d = collect_policy(pdir, max_iters=args.max_iters)
        policy_data[p] = d
        print(f"[info] {p:>14s} iters parsed: "
              f"{sorted(d.keys())}  ({len(d)})  ← {pdir}")

    run_id = args.run_dir.name
    write_table(policy_data, policies, args.out_md, args.model, run_id,
                args.run_dir, baseline=args.baseline)
    print(f"[ok] wrote {args.out_md}")

    any_data = any(policy_data[p] for p in policies)
    if any_data:
        make_plot(policy_data, policies, args.out_png, args.model, run_id,
                  baseline=args.baseline)
        print(f"[ok] wrote {args.out_png}")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                "No PPL chunks yet for any requested policy.\n"
                f"policies: {policies}",
                ha="center", va="center", fontsize=12, family="monospace")
        ax.set_axis_off()
        ax.set_title(f"Wave-11 per-chunk PPL delta — {args.model} ({run_id}) "
                     f"[no data]")
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=150)
        plt.close(fig)
        print(f"[ok] wrote placeholder {args.out_png}")

    # Per-policy summary lines for logs
    base_vals = policy_data.get(args.baseline, {})
    for p in policies:
        if p == args.baseline:
            arr = np.array(list(policy_data[p].values()))
            if arr.size:
                print(f"[summary] {p:>14s} n={arr.size} "
                      f"mean_ppl={arr.mean():.4f}")
            continue
        common = sorted(set(base_vals.keys()) & set(policy_data[p].keys()))
        if not common:
            print(f"[summary] {p:>14s} n_paired=0")
            continue
        van_arr = np.array([base_vals[c]        for c in common])
        pol_arr = np.array([policy_data[p][c]   for c in common])
        print(f"[summary] {p:>14s} n_paired={len(common)} "
              f"mean_{args.baseline}={van_arr.mean():.4f} "
              f"mean_{p}={pol_arr.mean():.4f} "
              f"mean_dlogppl="
              f"{float(np.mean(np.log(pol_arr) - np.log(van_arr))):+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
