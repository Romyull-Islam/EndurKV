#!/usr/bin/env python3
"""Multi-wave Pareto comparator.

Combines per-cell results from waves 4, 9, 10, 11 (and gracefully handles
absent / partial wave-11 data) and renders three relationship plots:

  07_pareto_multiwave_thermal_throughput.png
        x = peak DDR temperature (C)              [lower is better]
        y = mean decode throughput (tok/s)        [higher is better]
        Every cell from waves 4/9/10/11 is plotted and labelled.

  08_pareto_multiwave_ppl_thermal.png
        x = peak DDR temperature (C)              [lower is better]
        y = perplexity                            [lower is better]
        Wave-11 cells only (held-out PPL).  Wave-3..10 PPL values are
        derived from the *generated* tokens (sampling-NLL on the
        long-decode prompt) and are NOT comparable to held-out
        wikitext PPL.  We deliberately do NOT mix the two metrics on
        this figure; the wave-3..10 numbers are surfaced as a faded
        "sampling-NLL" reference panel with a prominent annotation.
        When wave-11 has zero cells, the figure is still produced with
        only the reference panel + a 'wave-11 pending' watermark.

  09_efficiency_frontier.png
        2 side-by-side panels representing the (PPL, peak-DDR, mean-tps)
        triple per cell.  Left panel: PCA projection of standardized
        (ppl, ddr, tps_inv) onto its two leading components, with the
        Pareto frontier drawn through the dominant cells.  Right panel:
        the standardized triple plotted as DDR vs tps with marker
        AREA encoding sampling-NLL (or held-out PPL when wave-11
        cells are present) so the third dimension is preserved.

Outputs (also printed to stdout in JSON as ART_SCHEMA):
  * three PNGs under EndurKV/figures/relationship_plots/
  * an `art_schema.json` next to the PNGs describing every artifact
    and the per-cell input rows.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path("/home/mislam22/EndurKV_workspace")
PHONE_LOGS = WORKSPACE / "phone-logs"
FIG_DIR = WORKSPACE / "EndurKV" / "figures" / "relationship_plots"

WAVE4_DIR = PHONE_LOGS / "wave4_longdecode_1780750084"
WAVE9_DIR = PHONE_LOGS / "wave9_v1fa2_stack_1780796320"
WAVE10_DIR = PHONE_LOGS / "wave10_ksweep_1780815847"

# Wave-11 dir is *globbed* because the run-id suffix is timestamp-driven and
# may be created by a later eval.  We pick the most recent dir each call.
WAVE11_GLOB = "wave11_eval_*"

OUT_07 = FIG_DIR / "07_pareto_multiwave_thermal_throughput.png"
OUT_08 = FIG_DIR / "08_pareto_multiwave_ppl_thermal.png"
OUT_09 = FIG_DIR / "09_efficiency_frontier.png"
ART_SCHEMA_PATH = FIG_DIR / "art_schema.json"


# ---------------------------------------------------------------------------
# Per-cell record
# ---------------------------------------------------------------------------
@dataclass
class Cell:
    wave: str                     # 'wave4' | 'wave9' | 'wave10' | 'wave11'
    label: str                    # short tag used as annotation
    policy: str
    k_nominal: Optional[int]
    model: str
    n_iter: int
    mean_decode_tps: float
    std_decode_tps: float
    peak_ddr_c: float             # peak across sensors.csv ddr_temp_mc
    mean_ddr_c: float
    sampling_nll_ppl: Optional[float]   # waves 3..10 (sampling NLL on long-decode)
    heldout_ppl: Optional[float]        # wave 11 (held-out wikitext chunks)
    peak_kv_cells: int
    peak_kv_mb: float
    src_dir: str

    def short(self) -> str:
        # Compact label used on scatter plots / Pareto annotations.
        k = f"K{self.k_nominal}" if self.k_nominal else ""
        return f"{self.wave}/{self.policy}{('/' + k) if k else ''}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_sensors_ddr(sensors_csv: Path) -> tuple[float, float]:
    """Return (peak_ddr_c, mean_ddr_c) from a sensors.csv.

    Falls back to (NaN, NaN) if file is missing or malformed.
    """
    if not sensors_csv.is_file():
        return float("nan"), float("nan")
    try:
        df = pd.read_csv(
            sensors_csv,
            usecols=["wall_clock_s", "ddr_temp_mc"],
            low_memory=False,
        )
    except Exception as exc:
        print(f"[warn] could not read {sensors_csv}: {exc}", file=sys.stderr)
        return float("nan"), float("nan")
    ddr = pd.to_numeric(df["ddr_temp_mc"], errors="coerce") / 1000.0
    ddr = ddr.dropna()
    if ddr.empty:
        return float("nan"), float("nan")
    return float(ddr.max()), float(ddr.mean())


_BARE_NONFINITE = re.compile(
    r"(?<![\"A-Za-z0-9_])(-?Inf(?:inity)?|NaN|inf|-inf|nan)(?![\"A-Za-z0-9_])"
)


def load_meta(iter_dir: Path) -> Optional[dict]:
    mp = iter_dir / "meta.json"
    if not mp.is_file():
        return None
    raw = mp.read_text()
    # The on-device emitter occasionally writes bare `inf` / `nan` tokens
    # (e.g. when decode_tps overflows on a zero-decode chunk).  Standard
    # json.loads rejects those, so we quote them as "Infinity"/"NaN"
    # before parsing and then coerce them back to floats.
    if _BARE_NONFINITE.search(raw):
        def _q(m):
            t = m.group(0)
            return {
                "inf": "Infinity", "Inf": "Infinity", "Infinity": "Infinity",
                "-inf": "-Infinity", "-Inf": "-Infinity", "-Infinity": "-Infinity",
                "nan": "NaN", "NaN": "NaN",
            }.get(t, t)
        raw = _BARE_NONFINITE.sub(lambda m: _q(m), raw)
    try:
        # parse_constant lets json convert Infinity/NaN to Python float.
        return json.loads(raw, parse_constant=lambda c: float(c))
    except Exception as exc:
        print(f"[warn] meta.json parse failed at {mp}: {exc}", file=sys.stderr)
        return None


def aggregate_cell(
    cell_dir: Path,
    wave: str,
    model: str = "Phi-3-mini-128k",
) -> Optional[Cell]:
    """Aggregate one policy / K cell into a Cell record.

    Cell layout we accept:
        cell_dir/
            sensors.csv
            stress.csv
            iter*/meta.json
        cell_dir/<bench>/   (wave-11 nests bench under the policy)
    """
    sensors = cell_dir / "sensors.csv"
    stress = cell_dir / "stress.csv"
    if not stress.is_file():
        return None

    sdf = pd.read_csv(stress)
    if sdf.empty:
        return None

    # decode_tps is occasionally inf when the wave-11 PPL script set
    # tps=0 on a chunk for which only prefill ran -- drop those for the
    # mean.  Also drop the (rare) absurd 1.992e9 / 2.105e9 entries that
    # came from a logger glitch on the device.
    tps = pd.to_numeric(sdf.get("decode_tps", pd.Series(dtype=float)),
                        errors="coerce")
    tps = tps.replace([np.inf, -np.inf], np.nan)
    tps = tps[(tps > 0) & (tps < 1e3)]
    if tps.empty:
        # Fall back to per-iter meta.json decode_tps.
        tps_vals = []
        for sub in sorted(cell_dir.glob("iter*")):
            meta = load_meta(sub)
            if meta and isinstance(meta.get("decode_tps"), (int, float)):
                v = float(meta["decode_tps"])
                if np.isfinite(v) and 0 < v < 1e3:
                    tps_vals.append(v)
        if tps_vals:
            tps = pd.Series(tps_vals)
        else:
            tps = pd.Series(dtype=float)

    if tps.empty:
        mean_tps = float("nan")
        std_tps = float("nan")
    else:
        mean_tps = float(tps.mean())
        std_tps = float(tps.std(ddof=0))

    # Per-iter PPL: prefer meta.json (authoritative) and average across
    # iterations.  Distinguish 'sampling NLL on generated tokens' (waves
    # 4 / 9 / 10) from 'held-out PPL on wikitext-2 chunks' (wave 11).
    ppl_vals = []
    for sub in sorted(cell_dir.glob("iter*")):
        meta = load_meta(sub)
        if meta and isinstance(meta.get("perplexity"), (int, float)):
            v = float(meta["perplexity"])
            if np.isfinite(v) and 0 < v < 1e4:
                ppl_vals.append(v)
    ppl_mean: Optional[float] = float(np.mean(ppl_vals)) if ppl_vals else None

    # Discover K, policy from first meta.json (constant across iters).
    k_nominal = None
    policy = "unknown"
    n_kv_cells = 0
    kv_mb = float("nan")
    iters = sorted(cell_dir.glob("iter*"))
    if iters:
        meta0 = load_meta(iters[0]) or {}
        k_nominal = meta0.get("k_nominal")
        policy = meta0.get("policy") or policy
    # Peak KV across stress.csv (one number per iter, take max).
    if "peak_kv_cells" in sdf.columns:
        try:
            n_kv_cells = int(pd.to_numeric(sdf["peak_kv_cells"],
                                           errors="coerce").max())
        except Exception:
            n_kv_cells = 0
    if iters:
        meta_last = load_meta(iters[-1]) or {}
        if isinstance(meta_last.get("peak_kv_mb"), (int, float)):
            kv_mb = float(meta_last["peak_kv_mb"])

    peak_ddr, mean_ddr = read_sensors_ddr(sensors)
    if np.isnan(peak_ddr) and "ddr_start_c" in sdf.columns:
        # Use the per-iter starting DDR as a (worse) proxy.
        ddrcol = pd.to_numeric(sdf["ddr_start_c"], errors="coerce").dropna()
        if not ddrcol.empty:
            peak_ddr = float(ddrcol.max())
            mean_ddr = float(ddrcol.mean())

    return Cell(
        wave=wave,
        label="",  # filled in by caller (uses canonical naming)
        policy=policy,
        k_nominal=int(k_nominal) if isinstance(k_nominal, (int, float)) else None,
        model=model,
        n_iter=int(len(sdf)),
        mean_decode_tps=mean_tps,
        std_decode_tps=std_tps,
        peak_ddr_c=peak_ddr,
        mean_ddr_c=mean_ddr,
        sampling_nll_ppl=ppl_mean if wave != "wave11" else None,
        heldout_ppl=ppl_mean if wave == "wave11" else None,
        peak_kv_cells=n_kv_cells,
        peak_kv_mb=kv_mb,
        src_dir=str(cell_dir),
    )


# ---------------------------------------------------------------------------
# Wave collectors
# ---------------------------------------------------------------------------
def collect_wave4(d: Path) -> list[Cell]:
    """Wave-4 long-decode smoking gun: vanilla, v1_K512, v1_fa_K512."""
    cells: list[Cell] = []
    if not d.is_dir():
        return cells
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        c = aggregate_cell(sub, wave="wave4")
        if c is None:
            continue
        # canonical label
        c.label = f"W4 {sub.name}"
        cells.append(c)
    return cells


def collect_wave9(d: Path) -> list[Cell]:
    cells: list[Cell] = []
    if not d.is_dir():
        return cells
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        c = aggregate_cell(sub, wave="wave9")
        if c is None:
            continue
        c.label = f"W9 {sub.name}"
        cells.append(c)
    return cells


def collect_wave10(d: Path) -> list[Cell]:
    cells: list[Cell] = []
    if not d.is_dir():
        return cells
    for sub in sorted(d.iterdir()):
        if not sub.is_dir() or not re.match(r"^K\d+$", sub.name):
            continue
        c = aggregate_cell(sub, wave="wave10")
        if c is None:
            continue
        c.label = f"W10 {sub.name}"
        cells.append(c)
    return cells


def find_wave11_dir() -> Optional[Path]:
    candidates = sorted(PHONE_LOGS.glob(WAVE11_GLOB))
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        return None
    # Newest by name suffix (timestamp), then mtime.
    return candidates[-1]


def collect_wave11(d: Optional[Path]) -> list[Cell]:
    """Wave-11 has a Model/Policy/Bench/iter* nesting."""
    cells: list[Cell] = []
    if d is None or not d.is_dir():
        return cells
    for model_dir in sorted(d.iterdir()):
        if not model_dir.is_dir():
            continue
        for policy_dir in sorted(model_dir.iterdir()):
            if not policy_dir.is_dir():
                continue
            for bench_dir in sorted(policy_dir.iterdir()):
                if not bench_dir.is_dir():
                    continue
                if bench_dir.name != "ppl":
                    # 08 figure focuses on held-out PPL.  Other benches
                    # (e.g. niah) are still aggregated for the 07 figure.
                    pass
                c = aggregate_cell(bench_dir, wave="wave11",
                                   model=model_dir.name)
                if c is None:
                    continue
                c.label = f"W11 {model_dir.name}/{policy_dir.name}/{bench_dir.name}"
                cells.append(c)
    return cells


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
WAVE_STYLE = {
    "wave4":  {"color": "#7570b3", "marker": "o", "label": "Wave-4 (long-decode)"},
    "wave9":  {"color": "#d95f02", "marker": "s", "label": "Wave-9 (v1_fa2_stack)"},
    "wave10": {"color": "#1b9e77", "marker": "^", "label": "Wave-10 (K-sweep)"},
    "wave11": {"color": "#e7298a", "marker": "D", "label": "Wave-11 (held-out)"},
}


def pareto_front(xs: np.ndarray, ys: np.ndarray,
                 x_better: str = "lower",
                 y_better: str = "higher") -> list[int]:
    """Return indices on the Pareto frontier."""
    n = len(xs)
    if n == 0:
        return []
    idx = list(range(n))
    idx.sort(key=lambda i: (xs[i] if x_better == "lower" else -xs[i],
                            -(ys[i] if y_better == "higher" else -ys[i])))
    front: list[int] = []
    best_y = None
    for i in idx:
        yi = ys[i]
        if np.isnan(yi):
            continue
        if best_y is None:
            front.append(i)
            best_y = yi
            continue
        if (y_better == "higher" and yi > best_y) or (
                y_better == "lower" and yi < best_y):
            front.append(i)
            best_y = yi
    return front


def annotate_point(ax, x, y, text, *, color, dx=0.35, dy=0.05):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        fontsize=7.5,
        color=color,
        arrowprops=dict(arrowstyle="-", lw=0.4, color=color, alpha=0.6),
    )


# ---------------------------------------------------------------------------
# Figure 07: peak DDR vs mean tps
# ---------------------------------------------------------------------------
def fig_07(cells: list[Cell], out: Path) -> dict:
    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=140)

    drawn: list[tuple[float, float, str]] = []
    by_wave: dict[str, list[Cell]] = {}
    for c in cells:
        by_wave.setdefault(c.wave, []).append(c)

    for wave, lst in by_wave.items():
        style = WAVE_STYLE.get(wave, {"color": "#666", "marker": "o", "label": wave})
        xs = np.array([c.peak_ddr_c for c in lst], dtype=float)
        ys = np.array([c.mean_decode_tps for c in lst], dtype=float)
        ax.scatter(xs, ys,
                   c=style["color"], marker=style["marker"], s=70,
                   edgecolor="black", linewidth=0.4,
                   label=style["label"], zorder=3)
        for c in lst:
            if np.isnan(c.peak_ddr_c) or np.isnan(c.mean_decode_tps):
                continue
            drawn.append((c.peak_ddr_c, c.mean_decode_tps, c.short()))
            annotate_point(ax, c.peak_ddr_c, c.mean_decode_tps, c.short(),
                           color=style["color"])

    # Pareto frontier across ALL cells (lower DDR + higher tps dominates).
    xs = np.array([c.peak_ddr_c for c in cells], dtype=float)
    ys = np.array([c.mean_decode_tps for c in cells], dtype=float)
    keep = ~(np.isnan(xs) | np.isnan(ys))
    if keep.any():
        kept = np.where(keep)[0]
        kept_xs = xs[kept]
        kept_ys = ys[kept]
        fidx = pareto_front(kept_xs, kept_ys,
                            x_better="lower", y_better="higher")
        fx = kept_xs[fidx]
        fy = kept_ys[fidx]
        order = np.argsort(fx)
        ax.plot(fx[order], fy[order],
                color="black", linewidth=1.3, alpha=0.5,
                linestyle="--", label="Pareto frontier", zorder=2)

    ax.set_xlabel("Peak DDR temperature during cell  (degC, lower is better)")
    ax.set_ylabel("Mean decode throughput  (tok/s, higher is better)")
    ax.set_title(
        "Multi-wave Pareto: thermal vs throughput\n"
        "(every cell from waves 4 / 9 / 10 / 11 labelled)"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return {
        "path": str(out),
        "n_cells_drawn": len(drawn),
        "x_axis": "peak_ddr_c",
        "y_axis": "mean_decode_tps",
    }


# ---------------------------------------------------------------------------
# Figure 08: PPL vs peak DDR  (wave-11 only;
#           wave-4..10 sampling-NLL shown as reference panel)
# ---------------------------------------------------------------------------
def fig_08(cells: list[Cell], out: Path) -> dict:
    wave11 = [c for c in cells if c.wave == "wave11" and
              c.heldout_ppl is not None and not np.isnan(c.peak_ddr_c)]
    ref = [c for c in cells if c.wave != "wave11" and
           c.sampling_nll_ppl is not None and not np.isnan(c.peak_ddr_c)]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), dpi=140,
                             gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax_main, ax_ref = axes

    # ----- left: wave-11 held-out PPL (real metric) ------------------
    ax_main.set_xlabel("Peak DDR temperature  (degC, lower is better)")
    ax_main.set_ylabel("Held-out wikitext PPL  (lower is better)")
    ax_main.set_title("Wave-11: held-out PPL vs peak DDR")
    ax_main.grid(alpha=0.25)

    if not wave11:
        ax_main.text(
            0.5, 0.5,
            "Wave-11 cells not yet available.\n"
            "Re-run host_plot_pareto_multiwave.py\n"
            "once phone-logs/wave11_eval_* is populated.",
            transform=ax_main.transAxes,
            ha="center", va="center",
            fontsize=12, color="#aa1133",
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#fff7d6",
                      edgecolor="#aa1133", linewidth=1.0),
        )
    else:
        xs = np.array([c.peak_ddr_c for c in wave11])
        ys = np.array([c.heldout_ppl for c in wave11])
        style = WAVE_STYLE["wave11"]
        ax_main.scatter(xs, ys,
                        c=style["color"], marker=style["marker"], s=85,
                        edgecolor="black", linewidth=0.5, zorder=3)
        for c, x, y in zip(wave11, xs, ys):
            annotate_point(ax_main, x, y, c.short(),
                           color=style["color"], dx=0.25, dy=0.02)
        fidx = pareto_front(xs, ys, x_better="lower", y_better="lower")
        if fidx:
            order = np.argsort(xs[fidx])
            ax_main.plot(xs[fidx][order], ys[fidx][order],
                         color="black", lw=1.2, alpha=0.5, ls="--",
                         label="Pareto frontier")
            ax_main.legend(loc="best", fontsize=9)

    # ----- right: wave-4..10 sampling-NLL reference panel ------------
    ax_ref.set_xlabel("Peak DDR temperature  (degC)")
    ax_ref.set_ylabel("Sampling-NLL exp(NLL)  on generated tokens")
    ax_ref.set_title("Reference: Wave-4/9/10 generated-token NLL\n"
                     "(NOT comparable to held-out PPL; do not mix axes)")
    ax_ref.grid(alpha=0.25)
    for wave, lst in {w: [c for c in ref if c.wave == w]
                      for w in ("wave4", "wave9", "wave10")}.items():
        if not lst:
            continue
        st = WAVE_STYLE[wave]
        xs = np.array([c.peak_ddr_c for c in lst])
        ys = np.array([c.sampling_nll_ppl for c in lst])
        ax_ref.scatter(xs, ys,
                       c=st["color"], marker=st["marker"], s=60,
                       edgecolor="black", linewidth=0.4,
                       alpha=0.55, label=st["label"], zorder=3)
        for c, x, y in zip(lst, xs, ys):
            ax_ref.annotate(c.short(),
                            xy=(x, y),
                            xytext=(x + 0.3, y + 0.02),
                            fontsize=6.5, color=st["color"], alpha=0.85)
    ax_ref.legend(loc="best", fontsize=8)
    # Big warning watermark on the reference panel
    ax_ref.text(0.5, 0.97,
                "sampling-NLL  -  do not compare to wave-11 PPL",
                transform=ax_ref.transAxes,
                ha="center", va="top",
                fontsize=9, color="#aa1133", fontweight="bold")

    fig.suptitle("Multi-wave Pareto: perplexity vs peak DDR")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return {
        "path": str(out),
        "n_wave11_cells": len(wave11),
        "n_reference_cells": len(ref),
        "metric_left":  "heldout_ppl",
        "metric_right": "sampling_nll_ppl",
        "metrics_mixed": False,
    }


# ---------------------------------------------------------------------------
# Figure 09: efficiency frontier (PPL x DDR x tps)
# ---------------------------------------------------------------------------
def fig_09(cells: list[Cell], out: Path) -> dict:
    # Build the (ppl, ddr, tps) triple for every cell.  Use held-out PPL
    # when present (wave 11); otherwise use sampling NLL (wave 4 / 9 / 10)
    # but mark those points as 'reference' so the legend is honest.
    rows = []
    for c in cells:
        ppl = c.heldout_ppl if c.heldout_ppl is not None else c.sampling_nll_ppl
        if ppl is None or not np.isfinite(ppl):
            continue
        if not np.isfinite(c.peak_ddr_c) or not np.isfinite(c.mean_decode_tps):
            continue
        rows.append(
            {"label": c.short(),
             "wave": c.wave,
             "ppl": float(ppl),
             "ddr": float(c.peak_ddr_c),
             "tps": float(c.mean_decode_tps),
             "ppl_kind": "heldout" if c.heldout_ppl is not None
                         else "sampling_nll"}
        )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), dpi=140)
    ax_pca, ax_sc = axes

    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No cells with all of (ppl, ddr, tps)",
                    transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#aa1133")
        fig.suptitle("Efficiency frontier: (PPL, DDR, tps)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return {"path": str(out), "n_cells_drawn": 0}

    # ---- PCA panel -----------------------------------------------------
    # Direction-of-better: lower PPL, lower DDR, higher tps.  We invert
    # tps so that 'lower = better' on all three axes; then standardize
    # and PCA-project to 2 components.  The Pareto frontier in the
    # *original* 3-space is overlaid as black-edged markers.
    from sklearn.decomposition import PCA  # local import keeps the
                                          # module callable from tests

    X = df[["ppl", "ddr", "tps"]].to_numpy(dtype=float).copy()
    X[:, 2] = -X[:, 2]  # invert tps so 'lower is better'
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    if len(df) >= 2:
        pca = PCA(n_components=min(2, Xs.shape[1]))
        Y = pca.fit_transform(Xs)
        if Y.shape[1] == 1:
            Y = np.column_stack([Y[:, 0], np.zeros(len(Y))])
    else:
        Y = np.column_stack([Xs[:, 0], Xs[:, 1] if Xs.shape[1] > 1 else np.zeros(len(Xs))])

    # 3-D Pareto frontier (over original objectives, lower-is-better
    # on all three after the tps inversion).
    pareto_mask = np.ones(len(X), dtype=bool)
    for i in range(len(X)):
        if not pareto_mask[i]:
            continue
        for j in range(len(X)):
            if i == j:
                continue
            # j dominates i if it is no worse on all axes and strictly
            # better on at least one.
            if (X[j] <= X[i]).all() and (X[j] < X[i]).any():
                pareto_mask[i] = False
                break

    for wave in df["wave"].unique():
        mask = (df["wave"] == wave).to_numpy()
        st = WAVE_STYLE.get(wave, {"color": "#666", "marker": "o",
                                    "label": wave})
        ax_pca.scatter(Y[mask, 0], Y[mask, 1],
                       c=st["color"], marker=st["marker"],
                       s=70, edgecolor="black", linewidth=0.4,
                       alpha=0.85, label=st["label"], zorder=3)
    # Highlight 3-D Pareto points
    ax_pca.scatter(Y[pareto_mask, 0], Y[pareto_mask, 1],
                   facecolor="none", edgecolor="black", s=170,
                   linewidth=1.6, zorder=4, label="3-D Pareto")
    for i, lab in enumerate(df["label"]):
        ax_pca.annotate(lab, xy=(Y[i, 0], Y[i, 1]),
                        xytext=(Y[i, 0] + 0.05, Y[i, 1] + 0.05),
                        fontsize=6.5, color="#333")

    ax_pca.set_xlabel("PC1  (standardized (ppl, ddr, -tps))")
    ax_pca.set_ylabel("PC2")
    ax_pca.grid(alpha=0.25)
    ax_pca.set_title("PCA(ppl, peak_ddr, -mean_tps)")
    ax_pca.legend(loc="best", fontsize=8)

    # ---- Right panel: DDR vs tps with marker size = PPL ---------------
    # Larger marker = worse PPL.  Heldout cells outlined in black, sampling
    # cells outlined in grey to preserve metric provenance.
    p_min, p_max = df["ppl"].min(), df["ppl"].max()
    rng = max(p_max - p_min, 1e-9)
    sizes = 60 + 360 * (df["ppl"] - p_min) / rng
    for wave in df["wave"].unique():
        mask = (df["wave"] == wave).to_numpy()
        st = WAVE_STYLE.get(wave, {"color": "#666", "marker": "o",
                                    "label": wave})
        edges = ["black" if k == "heldout" else "#666"
                 for k in df.loc[mask, "ppl_kind"]]
        ax_sc.scatter(df.loc[mask, "ddr"], df.loc[mask, "tps"],
                      c=st["color"], marker=st["marker"],
                      s=sizes[mask], edgecolor=edges, linewidth=0.8,
                      alpha=0.75, label=st["label"], zorder=3)
    for _, row in df.iterrows():
        ax_sc.annotate(row["label"], xy=(row["ddr"], row["tps"]),
                       xytext=(row["ddr"] + 0.35, row["tps"] + 0.05),
                       fontsize=6.5, color="#333")

    ax_sc.set_xlabel("Peak DDR  (degC)")
    ax_sc.set_ylabel("Mean decode tps  (tok/s)")
    ax_sc.set_title("Side-by-side: DDR vs tps  (marker size = PPL)")
    ax_sc.grid(alpha=0.25)

    # Custom PPL-size legend  (uses min/median/max ppl as reference dots).
    qs = np.quantile(df["ppl"], [0.0, 0.5, 1.0])
    size_legend = []
    for q in qs:
        s = 60 + 360 * (q - p_min) / rng
        size_legend.append(
            ax_sc.scatter([], [], s=s, c="#888", edgecolor="black",
                          linewidth=0.5, label=f"PPL~{q:.2f}")
        )
    leg1 = ax_sc.legend(loc="upper right", fontsize=8, title="marker size")
    ax_sc.add_artist(leg1)
    ax_sc.legend([Patch(facecolor=WAVE_STYLE[w]["color"],
                         label=WAVE_STYLE[w]["label"])
                  for w in df["wave"].unique() if w in WAVE_STYLE],
                 [WAVE_STYLE[w]["label"]
                  for w in df["wave"].unique() if w in WAVE_STYLE],
                 loc="lower left", fontsize=8)

    fig.suptitle("Efficiency frontier across PPL x DDR x throughput")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return {
        "path": str(out),
        "n_cells_drawn": int(len(df)),
        "n_pareto_cells": int(pareto_mask.sum()),
        "metrics_mixed": bool((df["ppl_kind"].nunique() > 1)),
        "ppl_kinds": sorted(df["ppl_kind"].unique().tolist()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    cells: list[Cell] = []
    cells += collect_wave4(WAVE4_DIR)
    cells += collect_wave9(WAVE9_DIR)
    cells += collect_wave10(WAVE10_DIR)
    w11_dir = find_wave11_dir()
    wave11_cells = collect_wave11(w11_dir)
    cells += wave11_cells

    # ---- audit table -------------------------------------------------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    audit = pd.DataFrame([asdict(c) for c in cells])
    print("[multi-wave aggregate]")
    if not audit.empty:
        cols = ["wave", "label", "policy", "k_nominal", "n_iter",
                "mean_decode_tps", "peak_ddr_c",
                "sampling_nll_ppl", "heldout_ppl",
                "peak_kv_mb"]
        print(audit[cols].to_string(index=False))
    else:
        print("(no cells collected)")
    print()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    art = {
        "kind": "ART_SCHEMA",
        "version": 1,
        "generator": "host_plot_pareto_multiwave.py",
        "wave_sources": {
            "wave4":  str(WAVE4_DIR),
            "wave9":  str(WAVE9_DIR),
            "wave10": str(WAVE10_DIR),
            "wave11": str(w11_dir) if w11_dir else None,
        },
        "n_cells_per_wave": {
            w: int(sum(1 for c in cells if c.wave == w))
            for w in ("wave4", "wave9", "wave10", "wave11")
        },
        "wave11_available": bool(wave11_cells),
        "ppl_metric_provenance": {
            "wave4":  "sampling_nll_on_generated_tokens",
            "wave9":  "sampling_nll_on_generated_tokens",
            "wave10": "sampling_nll_on_generated_tokens",
            "wave11": "heldout_wikitext_chunk_ppl",
        },
        "figures": {
            "07_pareto_multiwave_thermal_throughput": fig_07(cells, OUT_07),
            "08_pareto_multiwave_ppl_thermal":        fig_08(cells, OUT_08),
            "09_efficiency_frontier":                  fig_09(cells, OUT_09),
        },
        "cells": [asdict(c) for c in cells],
    }

    ART_SCHEMA_PATH.write_text(json.dumps(art, indent=2, default=str))

    # Print ART_SCHEMA to stdout (the contract demanded by the caller).
    print("ART_SCHEMA")
    print(json.dumps({k: v for k, v in art.items() if k != "cells"},
                     indent=2, default=str))
    print(f"[wrote] {ART_SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
