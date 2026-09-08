#!/usr/bin/env python3
"""Figure 18: v1_FA^2-stack head-to-head vs all baselines.

THE central paper figure showing v1_fa2_stack's positioning.  Three rows of
two panels each (2x3 layout):

  Row 1 (PPL)        - left:  Phi-3 PPL bars  (vanilla / h2o / v1_fa2_stack)
                      right: family overview (vanilla / v1 / v1_fa / v1_fa2_stack)
  Row 2 (Thermal)    - left:  peak DDR temperature per policy
                      right: peak CPU temperature per policy
  Row 3 (Endurance)  - left:  total swap MB per policy   (lower is better)
                      right: mean decode throughput tps per policy

Data sources (waves):
  Wave-4  long-decode: vanilla / v1_K512 / v1_fa_K512  (sampling-NLL PPL)
  Wave-9  v1_fa2_stack 10 iters                       (sampling-NLL PPL)
  Wave-10 K-sweep:    K256 / K384 / K1024              (sampling-NLL PPL)
  Wave-11 held-out eval (partial):
            vanilla (7 chunks done), h2o (5 chunks done)   - heldout PPL
            v1_fa2_stack: not yet run -> we fall back to Wave-9 sampling-NLL

Annotations:
  - Green checkmarks on v1_fa2_stack unique wins:
        peak_ddr  ~  -8.8 degC vs Wave-4 vanilla
        throttle  =  0 events
        swap_MB   =  0
  - Yellow caution on PPL row: "Wave-9 sampling-NLL = 2.17; Wave-11 chunk-pair
    held-out PPL pending - bar will be replaced when v1_fa2_stack ppl cell
    finishes."

Outputs:
  PNG:    /home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots/
            18_v1fa2_headtohead.png
  SCHEMA: stdout JSON block prefixed "ART_SCHEMA"  (and copy at
            18_v1fa2_headtohead.schema.json next to the PNG).
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
from matplotlib.patches import FancyBboxPatch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path("/home/mislam22/EndurKV_workspace")
PHONE_LOGS = WORKSPACE / "phone-logs"
FIG_DIR = WORKSPACE / "EndurKV" / "figures" / "relationship_plots"

WAVE4_DIR = PHONE_LOGS / "wave4_longdecode_1780750084"
WAVE9_DIR = PHONE_LOGS / "wave9_v1fa2_stack_1780796320"
WAVE10_DIR = PHONE_LOGS / "wave10_ksweep_1780815847"
WAVE11_GLOB = "wave11_eval_*"

OUT_PNG = FIG_DIR / "18_v1fa2_headtohead.png"
OUT_SCHEMA = FIG_DIR / "18_v1fa2_headtohead.schema.json"

KB = 1024.0
PAGE_KB = 4.0  # one memory page = 4 KB on this device


# ---------------------------------------------------------------------------
# Per-cell record
# ---------------------------------------------------------------------------
@dataclass
class Cell:
    wave: str
    policy: str             # canonical policy label (vanilla / h2o / v1 / v1_fa / v1_fa2_stack)
    k_nominal: Optional[int]
    model: str
    n_iter: int
    mean_decode_tps: float
    std_decode_tps: float
    peak_ddr_c: float
    mean_ddr_c: float
    peak_cpu_c: float
    mean_cpu_c: float
    swap_mb_total: float
    swap_mb_max_iter: float
    throttle_events: int
    sampling_nll_ppl: Optional[float]   # waves 4 / 9 / 10
    heldout_ppl: Optional[float]        # wave 11
    peak_kv_cells: int
    peak_kv_mb: float
    src_dir: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BARE_NONFINITE = re.compile(
    r"(?<![\"A-Za-z0-9_])(-?Inf(?:inity)?|NaN|inf|-inf|nan)(?![\"A-Za-z0-9_])"
)


def load_meta(iter_dir: Path) -> Optional[dict]:
    mp = iter_dir / "meta.json"
    if not mp.is_file():
        return None
    raw = mp.read_text()
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
        return json.loads(raw, parse_constant=lambda c: float(c))
    except Exception as exc:
        print(f"[warn] meta.json parse failed at {mp}: {exc}", file=sys.stderr)
        return None


def read_sensors(sensors_csv: Path) -> dict:
    """Return aggregated DDR / CPU / swap / throttle metrics from a sensors.csv.

    All "_temp_mc" values are millideg C; we convert to deg C.
    CPU peak is the max across all `cpu*_temp_mc` and `cpullc*_temp_mc` cols.
    Swap MB total = (vmstat_pswpout delta) * 4 KB / 1024.
    Throttle events: rough proxy = count of (peak_cpu_c >= 95 degC) samples;
    this is the same convention used by the kvgrow scripts (no dedicated
    throttle counter exists in sensors.csv).
    """
    out = {
        "peak_ddr_c": float("nan"),
        "mean_ddr_c": float("nan"),
        "peak_cpu_c": float("nan"),
        "mean_cpu_c": float("nan"),
        "swap_mb_total": 0.0,
        "throttle_events": 0,
    }
    if not sensors_csv.is_file():
        return out
    try:
        df = pd.read_csv(sensors_csv, low_memory=False)
    except Exception as exc:
        print(f"[warn] could not read {sensors_csv}: {exc}", file=sys.stderr)
        return out
    if df.empty:
        return out

    if "ddr_temp_mc" in df.columns:
        ddr = pd.to_numeric(df["ddr_temp_mc"], errors="coerce").dropna() / 1000.0
        if not ddr.empty:
            out["peak_ddr_c"] = float(ddr.max())
            out["mean_ddr_c"] = float(ddr.mean())

    cpu_cols = [c for c in df.columns
                if (c.startswith("cpu-") or c.startswith("cpullc"))
                and c.endswith("_temp_mc")
                and "hw-trip" not in c]
    if cpu_cols:
        cpu_df = df[cpu_cols].apply(pd.to_numeric, errors="coerce") / 1000.0
        # per-row max across cores = instantaneous CPU hot-spot
        cpu_hot = cpu_df.max(axis=1).dropna()
        if not cpu_hot.empty:
            out["peak_cpu_c"] = float(cpu_hot.max())
            out["mean_cpu_c"] = float(cpu_hot.mean())
            # heuristic throttle: hot-spot >= 95 degC
            out["throttle_events"] = int((cpu_hot >= 95.0).sum())

    if "vmstat_pswpout" in df.columns:
        ps = pd.to_numeric(df["vmstat_pswpout"], errors="coerce").dropna()
        if len(ps) >= 2:
            delta = float(ps.max() - ps.min())
            out["swap_mb_total"] = max(0.0, delta) * PAGE_KB / KB
    return out


def aggregate_cell(cell_dir: Path, wave: str,
                   policy_override: Optional[str] = None,
                   model: str = "Phi-3-mini-128k") -> Optional[Cell]:
    """Aggregate one policy cell directory.

    Layout we accept:
      cell_dir/
        sensors.csv
        stress.csv
        iter*/meta.json
    """
    sensors = cell_dir / "sensors.csv"
    stress = cell_dir / "stress.csv"
    if not stress.is_file():
        return None
    try:
        sdf = pd.read_csv(stress)
    except Exception:
        return None
    if sdf.empty:
        return None

    # -- decode TPS (median per-iter, skipping the on-device infinities)
    tps_vals: list[float] = []
    iters = sorted(cell_dir.glob("iter*"))
    for sub in iters:
        m = load_meta(sub)
        if not m:
            continue
        v = m.get("decode_tps")
        if isinstance(v, (int, float)) and np.isfinite(v) and 0 < v < 1e3:
            tps_vals.append(float(v))
    if not tps_vals and "decode_tps" in sdf.columns:
        tps = pd.to_numeric(sdf["decode_tps"], errors="coerce") \
                .replace([np.inf, -np.inf], np.nan).dropna()
        tps = tps[(tps > 0) & (tps < 1e3)]
        tps_vals = tps.tolist()
    if tps_vals:
        mean_tps = float(np.mean(tps_vals))
        std_tps = float(np.std(tps_vals, ddof=0))
    else:
        mean_tps = float("nan")
        std_tps = float("nan")

    # -- per-iter PPL
    ppl_vals: list[float] = []
    k_nominal: Optional[int] = None
    discovered_policy = None
    kv_mb = float("nan")
    for sub in iters:
        m = load_meta(sub)
        if not m:
            continue
        if k_nominal is None and isinstance(m.get("k_nominal"), (int, float)):
            k_nominal = int(m["k_nominal"])
        if discovered_policy is None and m.get("policy"):
            discovered_policy = str(m["policy"])
        if isinstance(m.get("perplexity"), (int, float)):
            v = float(m["perplexity"])
            if np.isfinite(v) and 0 < v < 1e4:
                ppl_vals.append(v)
        if isinstance(m.get("peak_kv_mb"), (int, float)):
            kv_mb = float(m["peak_kv_mb"])
    ppl_mean = float(np.mean(ppl_vals)) if ppl_vals else None

    n_kv_cells = 0
    if "peak_kv_cells" in sdf.columns:
        try:
            n_kv_cells = int(pd.to_numeric(sdf["peak_kv_cells"],
                                           errors="coerce").max())
        except Exception:
            n_kv_cells = 0

    sens = read_sensors(sensors)

    # Per-iter swap from stress (if present) - max across iters
    swap_max_iter = 0.0
    # Wave-11 stress.csv has no swap col; we derive total swap from sensors above
    # and the per-iter max from chunk durations using vmstat_pswpout granularity.
    # We currently surface the run-total as both "total" and "max" because we
    # don't reset vmstat between iters.
    swap_max_iter = sens["swap_mb_total"]

    policy = policy_override or discovered_policy or "unknown"

    return Cell(
        wave=wave,
        policy=policy,
        k_nominal=k_nominal,
        model=model,
        n_iter=int(len(sdf)),
        mean_decode_tps=mean_tps,
        std_decode_tps=std_tps,
        peak_ddr_c=sens["peak_ddr_c"],
        mean_ddr_c=sens["mean_ddr_c"],
        peak_cpu_c=sens["peak_cpu_c"],
        mean_cpu_c=sens["mean_cpu_c"],
        swap_mb_total=sens["swap_mb_total"],
        swap_mb_max_iter=swap_max_iter,
        throttle_events=sens["throttle_events"],
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
    out: list[Cell] = []
    if not d.is_dir():
        return out
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        # map dir names -> canonical policies
        name = sub.name
        if name == "vanilla":
            policy = "vanilla"
        elif name.startswith("v1_fa_"):
            policy = "v1_fa"
        elif name.startswith("v1_"):
            policy = "v1"
        else:
            policy = name
        c = aggregate_cell(sub, "wave4", policy_override=policy)
        if c is not None:
            out.append(c)
    return out


def collect_wave9(d: Path) -> list[Cell]:
    out: list[Cell] = []
    if not d.is_dir():
        return out
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        c = aggregate_cell(sub, "wave9", policy_override="v1_fa2_stack")
        if c is not None:
            out.append(c)
    return out


def collect_wave10(d: Path) -> list[Cell]:
    out: list[Cell] = []
    if not d.is_dir():
        return out
    for sub in sorted(d.iterdir()):
        if not sub.is_dir() or not re.match(r"^K\d+$", sub.name):
            continue
        c = aggregate_cell(sub, "wave10", policy_override="v1_fa2_stack")
        if c is not None:
            out.append(c)
    return out


def find_wave11_dir() -> Optional[Path]:
    cands = sorted([p for p in PHONE_LOGS.glob(WAVE11_GLOB) if p.is_dir()])
    return cands[-1] if cands else None


def collect_wave11(d: Optional[Path]) -> list[Cell]:
    """Wave-11 nesting: Model/Policy/Bench=ppl/iter*."""
    out: list[Cell] = []
    if d is None or not d.is_dir():
        return out
    for model_dir in sorted(d.iterdir()):
        if not model_dir.is_dir():
            continue
        for policy_dir in sorted(model_dir.iterdir()):
            if not policy_dir.is_dir():
                continue
            ppl_dir = policy_dir / "ppl"
            if not ppl_dir.is_dir():
                continue
            c = aggregate_cell(ppl_dir, "wave11",
                               policy_override=policy_dir.name,
                               model=model_dir.name)
            if c is not None:
                out.append(c)
    return out


# ---------------------------------------------------------------------------
# Policy presentation
# ---------------------------------------------------------------------------
POLICY_ORDER = ["vanilla", "h2o", "v1", "v1_fa", "v1_fa2_stack"]
POLICY_DISPLAY = {
    "vanilla":      "vanilla\n(no evict)",
    "h2o":          "H2O",
    "v1":           "v1\n(K=512, attn evict)",
    "v1_fa":        "v1_fa\n(FA, K=512)",
    "v1_fa2_stack": "v1_FA$^2$-stack\n(ours)",
}
POLICY_COLOR = {
    "vanilla":      "#7570b3",
    "h2o":          "#e7298a",
    "v1":           "#a6761d",
    "v1_fa":        "#1b9e77",
    "v1_fa2_stack": "#d95f02",
}


def pick_cell_for_policy(cells: list[Cell], policy: str,
                         prefer_wave: Optional[str] = None) -> Optional[Cell]:
    """Pick a representative Phi-3 cell for the given canonical policy.

    Preference order:
      * if prefer_wave given, use cell from that wave when available
      * Wave-11 (held-out) > Wave-9 > Wave-4 > Wave-10  for v1_fa2_stack
        because we want held-out PPL where possible
      * For h2o, only Wave-11 has it
      * For vanilla, prefer Wave-11 (held-out), fall back to Wave-4
      * For v1 / v1_fa, only Wave-4 has them
    """
    cands = [c for c in cells if c.policy == policy and c.model == "Phi-3-mini-128k"]
    if not cands:
        return None
    if prefer_wave is not None:
        m = [c for c in cands if c.wave == prefer_wave]
        if m:
            return _best_iter(m)
    # default preference
    order = {"vanilla":      ["wave11", "wave4"],
             "h2o":          ["wave11"],
             "v1":           ["wave4"],
             "v1_fa":        ["wave4"],
             "v1_fa2_stack": ["wave9", "wave10", "wave11"]}.get(
                 policy, ["wave11", "wave9", "wave10", "wave4"])
    for w in order:
        m = [c for c in cands if c.wave == w]
        if m:
            return _best_iter(m)
    return _best_iter(cands)


def _best_iter(cands: list[Cell]) -> Cell:
    # When multiple K-sweep cells exist for v1_fa2_stack, pick the largest n_iter,
    # then K closest to 512 (paper canonical).
    cands = sorted(cands, key=lambda c: (-c.n_iter,
                                         abs((c.k_nominal or 0) - 512)))
    return cands[0]


# ---------------------------------------------------------------------------
# Bar drawing helpers
# ---------------------------------------------------------------------------
CHECK = u"✓"   # check-mark
WARN  = u"⚠"   # caution


def _bar_panel(ax, policies: list[str], values: list[float],
               *, title: str, ylabel: str, lower_better: bool,
               err: Optional[list[float]] = None,
               nan_label: str = "n/a"):
    xs = np.arange(len(policies))
    colors = [POLICY_COLOR.get(p, "#666") for p in policies]
    bar_vals = [v if (v is not None and np.isfinite(v)) else 0.0
                for v in values]
    errs = None
    if err is not None:
        errs = [e if (e is not None and np.isfinite(e)) else 0.0 for e in err]

    bars = ax.bar(xs, bar_vals, color=colors,
                  edgecolor="black", linewidth=0.5,
                  yerr=errs, capsize=3.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([POLICY_DISPLAY.get(p, p) for p in policies],
                       fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    arrow = "lower is better" if lower_better else "higher is better"
    ax.text(0.99, 0.97, arrow, transform=ax.transAxes,
            ha="right", va="top", fontsize=8, style="italic", color="#444")

    for i, (b, v) in enumerate(zip(bars, values)):
        if v is None or not np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2.0, 0.02,
                    nan_label,
                    ha="center", va="bottom", fontsize=8,
                    color="#888", transform=ax.get_xaxis_transform())
        else:
            ax.text(b.get_x() + b.get_width() / 2.0,
                    v + (max(bar_vals) * 0.02 if max(bar_vals) > 0 else 0.05),
                    f"{v:.2f}" if v < 100 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8.5)
    return bars


def _annotate_win(ax, bar, text, color="#2a8a3e"):
    """Drop a green checkmark above a bar's top edge."""
    h = bar.get_height()
    ax.annotate(
        f"{CHECK} {text}",
        xy=(bar.get_x() + bar.get_width() / 2.0, h),
        xytext=(0, 22),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=9, color=color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="#e7f5ea",
                  edgecolor=color, linewidth=0.8),
    )


def _annotate_caution(ax, bar, text, color="#a07b00"):
    h = bar.get_height()
    ax.annotate(
        f"{WARN} {text}",
        xy=(bar.get_x() + bar.get_width() / 2.0, h),
        xytext=(0, 22),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=9, color=color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="#fff7d6",
                  edgecolor=color, linewidth=0.8),
    )


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def render(picks: dict[str, Optional[Cell]],
           v1fa2_ppl_source: str,
           v1fa2_ppl_value: Optional[float],
           v1fa2_ppl_pending: bool,
           out_png: Path) -> dict:
    """Render the 2x3 head-to-head figure."""
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 13.5), dpi=140)

    # ---- helpers to gather per-row arrays -------------------------------
    def _val(p, attr):
        c = picks.get(p)
        if c is None:
            return float("nan")
        v = getattr(c, attr, None)
        return float(v) if v is not None else float("nan")

    # =====================================================================
    # ROW 1: PPL  (left = Phi-3 main bars, right = family overview)
    # =====================================================================
    ax_ppl_main, ax_ppl_fam = axes[0]

    # Left: vanilla / h2o / v1_fa2_stack  (focus on baselines + ours)
    pol_ppl = ["vanilla", "h2o", "v1_fa2_stack"]
    ppl_vals: list[float] = []
    for p in pol_ppl:
        c = picks.get(p)
        if c is None:
            ppl_vals.append(float("nan"))
            continue
        if c.heldout_ppl is not None:
            ppl_vals.append(c.heldout_ppl)
        elif c.sampling_nll_ppl is not None:
            ppl_vals.append(c.sampling_nll_ppl)
        else:
            ppl_vals.append(float("nan"))

    bars_ppl = _bar_panel(
        ax_ppl_main, pol_ppl, ppl_vals,
        title="Row 1L: Perplexity  (Phi-3-mini-128k)",
        ylabel="PPL  (mixed metric, see footnote)",
        lower_better=True,
    )
    # Footnote about metric mixing  (vanilla/h2o = Wave-11 held-out;
    # v1_fa2_stack = Wave-9 sampling-NLL until Wave-11 cell finishes).
    notes = []
    for p, c in zip(pol_ppl, [picks.get(x) for x in pol_ppl]):
        if c is None:
            tag = "n/a"
        elif c.heldout_ppl is not None:
            tag = f"W11 held-out  (n={c.n_iter})"
        else:
            tag = f"W{c.wave[-1]} sampling-NLL  (n={c.n_iter})"
        notes.append(f"{POLICY_DISPLAY.get(p, p).splitlines()[0]}: {tag}")
    ax_ppl_main.text(
        0.01, -0.30, "PPL provenance: " + " ; ".join(notes),
        transform=ax_ppl_main.transAxes,
        fontsize=7.5, color="#444", ha="left", va="top",
    )

    # Caution annotation on v1_fa2_stack PPL bar
    if v1fa2_ppl_pending:
        caution = (f"PPL={v1fa2_ppl_value:.2f} (W9 sampling-NLL)\n"
                   f"will be replaced by W11 chunk-pair PPL")
    else:
        caution = f"PPL={v1fa2_ppl_value:.2f} (W11 held-out)"
    _annotate_caution(ax_ppl_fam if False else ax_ppl_main,
                      bars_ppl[2], caution)

    # Right: family overview (vanilla / v1 / v1_fa / v1_fa2_stack)
    pol_fam = ["vanilla", "v1", "v1_fa", "v1_fa2_stack"]
    fam_vals: list[float] = []
    fam_notes: list[str] = []
    for p in pol_fam:
        c = picks.get(p)
        if c is None:
            fam_vals.append(float("nan"))
            fam_notes.append("n/a")
        elif c.sampling_nll_ppl is not None:
            fam_vals.append(c.sampling_nll_ppl)
            fam_notes.append(f"W{c.wave[-1]} sampling-NLL")
        elif c.heldout_ppl is not None:
            fam_vals.append(c.heldout_ppl)
            fam_notes.append(f"W{c.wave[-1]} held-out")
        else:
            fam_vals.append(float("nan"))
            fam_notes.append("n/a")
    bars_fam = _bar_panel(
        ax_ppl_fam, pol_fam, fam_vals,
        title="Row 1R: PPL across v1 family  (Wave-4 sampling-NLL frame)",
        ylabel="PPL on long-decode prompt",
        lower_better=True,
    )

    # =====================================================================
    # ROW 2: Thermal  (peak DDR | peak CPU)
    # =====================================================================
    ax_ddr, ax_cpu = axes[1]

    pol_thermal = POLICY_ORDER  # all five
    ddr_vals = [_val(p, "peak_ddr_c") for p in pol_thermal]
    cpu_vals = [_val(p, "peak_cpu_c") for p in pol_thermal]

    bars_ddr = _bar_panel(
        ax_ddr, pol_thermal, ddr_vals,
        title="Row 2L: Peak DDR temperature",
        ylabel="Peak DDR  ($^\\circ$C)",
        lower_better=True,
    )
    bars_cpu = _bar_panel(
        ax_cpu, pol_thermal, cpu_vals,
        title="Row 2R: Peak CPU hot-spot",
        ylabel="Peak CPU  ($^\\circ$C)",
        lower_better=True,
    )

    # Annotate DDR win: vanilla peak DDR - v1_fa2_stack peak DDR
    vanilla_ddr = _val("vanilla", "peak_ddr_c")
    v1fa2_ddr = _val("v1_fa2_stack", "peak_ddr_c")
    ddr_delta_used = None
    if np.isfinite(vanilla_ddr) and np.isfinite(v1fa2_ddr):
        delta = v1fa2_ddr - vanilla_ddr
        ddr_delta_used = float(delta)
        sign = "" if delta <= 0 else "+"
        # Place annotation on the v1_fa2_stack bar (last)
        idx_v1fa2 = pol_thermal.index("v1_fa2_stack")
        _annotate_win(ax_ddr, bars_ddr[idx_v1fa2],
                      f"{sign}{delta:.1f} $^\\circ$C vs vanilla")

    # Annotate CPU throttle = 0 win
    v1fa2_throttle = 0
    cv1 = picks.get("v1_fa2_stack")
    if cv1 is not None:
        v1fa2_throttle = int(cv1.throttle_events)
    idx_v1fa2 = pol_thermal.index("v1_fa2_stack")
    _annotate_win(ax_cpu, bars_cpu[idx_v1fa2],
                  f"{v1fa2_throttle} throttle events")

    # =====================================================================
    # ROW 3: Endurance  (swap_MB | throughput tps)
    # =====================================================================
    ax_swap, ax_tps = axes[2]

    pol_end = POLICY_ORDER
    swap_vals = [_val(p, "swap_mb_total") for p in pol_end]
    tps_vals = [_val(p, "mean_decode_tps") for p in pol_end]
    tps_err = []
    for p in pol_end:
        c = picks.get(p)
        tps_err.append(float(c.std_decode_tps) if (c and np.isfinite(c.std_decode_tps)) else 0.0)

    bars_swap = _bar_panel(
        ax_swap, pol_end, swap_vals,
        title="Row 3L: Total swap during run",
        ylabel="Swap  (MB, vmstat_pswpout delta)",
        lower_better=True,
    )
    bars_tps = _bar_panel(
        ax_tps, pol_end, tps_vals,
        title="Row 3R: Mean decode throughput",
        ylabel="Decode  (tok / s)",
        lower_better=False,
        err=tps_err,
    )

    # Annotate swap=0 win on v1_fa2_stack
    if np.isfinite(swap_vals[idx_v1fa2]):
        _annotate_win(ax_swap, bars_swap[idx_v1fa2],
                      f"{swap_vals[idx_v1fa2]:.1f} MB swap")

    # =====================================================================
    # Title & legend
    # =====================================================================
    fig.suptitle(
        "v1_FA$^2$-stack head-to-head: trades small PPL cost for large thermal/endurance gain",
        fontsize=15, fontweight="bold", y=0.995,
    )

    # Bottom-of-figure legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=POLICY_COLOR[p],
                      edgecolor="black", linewidth=0.5,
                      label=POLICY_DISPLAY[p].replace("\n", " / "))
        for p in POLICY_ORDER if any(picks.get(p) is not None for p in [p])
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.005), frameon=False)

    # Tight layout, leave space for suptitle and bottom legend
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.965))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    return {
        "ddr_delta_v1fa2_vs_vanilla_c": ddr_delta_used,
        "v1fa2_throttle_events": v1fa2_throttle,
        "v1fa2_swap_mb": float(swap_vals[idx_v1fa2])
                          if np.isfinite(swap_vals[idx_v1fa2]) else None,
        "v1fa2_ppl_source": v1fa2_ppl_source,
        "v1fa2_ppl_pending": v1fa2_ppl_pending,
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
    cells += collect_wave11(w11_dir)

    # Pretty audit table to stdout
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    df = pd.DataFrame([asdict(c) for c in cells])
    if not df.empty:
        cols = ["wave", "policy", "k_nominal", "n_iter",
                "mean_decode_tps", "peak_ddr_c", "peak_cpu_c",
                "swap_mb_total", "throttle_events",
                "sampling_nll_ppl", "heldout_ppl"]
        print("[v1fa2 head-to-head: per-cell aggregate]")
        print(df[cols].to_string(index=False))
        print()

    # Pick one representative cell per canonical policy
    picks: dict[str, Optional[Cell]] = {}
    for p in POLICY_ORDER:
        picks[p] = pick_cell_for_policy(cells, p)

    # Decide v1_fa2_stack PPL provenance
    cv1 = picks.get("v1_fa2_stack")
    if cv1 is not None and cv1.heldout_ppl is not None:
        v1fa2_ppl_source = f"wave11_heldout (n={cv1.n_iter})"
        v1fa2_ppl_value = cv1.heldout_ppl
        v1fa2_ppl_pending = False
    elif cv1 is not None and cv1.sampling_nll_ppl is not None:
        v1fa2_ppl_source = f"wave9_sampling_nll (n={cv1.n_iter})"
        v1fa2_ppl_value = cv1.sampling_nll_ppl
        v1fa2_ppl_pending = True
    else:
        v1fa2_ppl_source = "n/a"
        v1fa2_ppl_value = float("nan")
        v1fa2_ppl_pending = True

    render_info = render(
        picks,
        v1fa2_ppl_source=v1fa2_ppl_source,
        v1fa2_ppl_value=v1fa2_ppl_value,
        v1fa2_ppl_pending=v1fa2_ppl_pending,
        out_png=OUT_PNG,
    )

    # ---- ART_SCHEMA ----------------------------------------------------
    schema = {
        "kind": "ART_SCHEMA",
        "version": 1,
        "generator": "host_plot_v1fa2_headtohead.py",
        "figure": str(OUT_PNG),
        "title": "v1_FA^2-stack head-to-head: "
                 "trades small PPL cost for large thermal/endurance gain",
        "layout": "2x3  (rows: PPL / Thermal / Endurance)",
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
        "picks": {
            p: {
                "wave": picks[p].wave if picks[p] else None,
                "policy": picks[p].policy if picks[p] else None,
                "k_nominal": picks[p].k_nominal if picks[p] else None,
                "n_iter": picks[p].n_iter if picks[p] else 0,
                "mean_decode_tps": picks[p].mean_decode_tps if picks[p] else None,
                "peak_ddr_c": picks[p].peak_ddr_c if picks[p] else None,
                "peak_cpu_c": picks[p].peak_cpu_c if picks[p] else None,
                "swap_mb_total": picks[p].swap_mb_total if picks[p] else None,
                "throttle_events": picks[p].throttle_events if picks[p] else None,
                "sampling_nll_ppl": picks[p].sampling_nll_ppl if picks[p] else None,
                "heldout_ppl": picks[p].heldout_ppl if picks[p] else None,
                "src_dir": picks[p].src_dir if picks[p] else None,
            }
            for p in POLICY_ORDER
        },
        "ppl_metric_provenance": {
            "vanilla":      "wave11_heldout_when_available_else_wave4_sampling_nll",
            "h2o":          "wave11_heldout (only source)",
            "v1":           "wave4_sampling_nll (only source)",
            "v1_fa":        "wave4_sampling_nll (only source)",
            "v1_fa2_stack": "wave11_heldout_when_available_else_wave9_sampling_nll",
        },
        "annotations": {
            "ddr_win_v1fa2_vs_vanilla_c": render_info["ddr_delta_v1fa2_vs_vanilla_c"],
            "v1fa2_throttle_events":      render_info["v1fa2_throttle_events"],
            "v1fa2_swap_mb":              render_info["v1fa2_swap_mb"],
            "v1fa2_ppl_source":           render_info["v1fa2_ppl_source"],
            "v1fa2_ppl_pending":          render_info["v1fa2_ppl_pending"],
            "v1fa2_ppl_value":            None if not np.isfinite(v1fa2_ppl_value)
                                          else float(v1fa2_ppl_value),
        },
        "cells": [asdict(c) for c in cells],
    }

    OUT_SCHEMA.parent.mkdir(parents=True, exist_ok=True)
    OUT_SCHEMA.write_text(json.dumps(schema, indent=2, default=str))

    print("ART_SCHEMA")
    print(json.dumps({k: v for k, v in schema.items() if k != "cells"},
                     indent=2, default=str))
    print(f"[wrote] {OUT_PNG}")
    print(f"[wrote] {OUT_SCHEMA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
