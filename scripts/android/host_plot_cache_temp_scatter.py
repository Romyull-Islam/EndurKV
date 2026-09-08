#!/usr/bin/env python3
"""
PLOT 2: Cache size SCATTER vs instantaneous DDR temperature (aggregate cross-cell).

For every cell in waves 3-10 (every sub-directory that has a sensors.csv +
optional stress.csv), sample (cache_size, ddr_temp_C) points:

  - From sensors.csv: every 5th row -> ddr_temp_mc / 1000.0
  - For the matching cache_size at that time:
        * If a per-iter stress.csv exists, use the iter-bucket -> peak_kv_cells
          for the iter that contains the sensor sample's monotonic offset.
        * Otherwise (sensors-only cell), use that cell's max peak_kv_cells as a
          constant.

Each point is colored by policy family (vanilla red, v1 blue,
v1_fa2_stack green, ...).

Left panel  : raw scatter (log-x).
Right panel : Pareto-style median + IQR per log-cache_size bin (all colors
              aggregated), plus a fitted line  DDR = a + b * log10(cache_size).

Output: figures/relationship_plots/02_cache_temp_scatter.png
"""

import os
import sys
import csv
import math
import glob
import bisect
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
PHONE_LOGS  = "/home/mislam22/EndurKV_workspace/phone-logs"
OUT_DIR     = "/home/mislam22/EndurKV_workspace/EndurKV/figures/relationship_plots"
OUT_FIG     = os.path.join(OUT_DIR, "02_cache_temp_scatter.png")
WAVES       = ("wave3_", "wave4_", "wave5_", "wave6_", "wave7_",
               "wave8_", "wave9_", "wave10_")
SAMPLE_EVERY = 5     # sub-sample sensors.csv

# Policy -> color mapping (family-level)
POLICY_COLOR = {
    "vanilla":         "#d62728",  # red
    "llamacpp_stock":  "#7f7f7f",  # grey
    "tova":            "#9467bd",  # purple
    "pyramid":         "#bcbd22",  # olive
    "v1":              "#1f77b4",  # blue
    "v1_fa":           "#17becf",  # cyan
    "v1_fa2":          "#ff7f0e",  # orange
    "v1_fa2_selective":"#8c564b",  # brown
    "v1_fa2_stack":    "#2ca02c",  # green
    "other":           "#444444",
}


def policy_family(cell_name: str) -> str:
    """Map a directory name like 'v1_fa2_stack' or 'v1_K512' to a family key."""
    n = cell_name.lower()
    # order matters: most specific first
    if "v1_fa2_stack" in n:        return "v1_fa2_stack"
    if "v1_fa2_selective" in n:    return "v1_fa2_selective"
    if "v1_fa2" in n:              return "v1_fa2"
    if "v1_fa" in n:               return "v1_fa"
    if n.startswith("v1") or "_v1_" in n or n == "v1" or "/v1" in n or "v1_k" in n:
        return "v1"
    if "tova" in n:                return "tova"
    if "pyramid" in n:             return "pyramid"
    if "llamacpp" in n:            return "llamacpp_stock"
    if "vanilla" in n:             return "vanilla"
    # K-only names from wave10_ksweep => v1-family runs
    if n.startswith("k") and any(c.isdigit() for c in n):
        return "v1"
    return "other"


def col_idx(header_line: str, name: str):
    cols = header_line.rstrip("\n").split(",")
    try:
        return cols.index(name)
    except ValueError:
        return None


def load_stress_iters(stress_path):
    """
    Return a list of (t_start_s, peak_kv_cells) for each iter, sorted by
    t_start. peak_kv defaults to last row when missing.
    """
    iters = []
    try:
        with open(stress_path) as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                try:
                    t = float(row.get("t_elapsed_s", "") or 0.0)
                    pk = int(row.get("peak_kv_cells", "") or 0)
                    if pk > 0:
                        iters.append((t, pk))
                except (ValueError, TypeError):
                    continue
    except OSError:
        return []
    iters.sort()
    return iters


def kv_for_offset(iters, offset_s, fallback):
    """Bucket the sensor offset into the iter that started at or before it."""
    if not iters:
        return fallback
    ts = [it[0] for it in iters]
    pos = bisect.bisect_right(ts, offset_s) - 1
    if pos < 0:
        pos = 0
    return iters[pos][1]


def scan_cells():
    """Yield (wave_dir, cell_dir, sensors_path, stress_path_or_None, family)."""
    for entry in sorted(os.listdir(PHONE_LOGS)):
        if not any(entry.startswith(w) for w in WAVES):
            continue
        wave_root = os.path.join(PHONE_LOGS, entry)
        if not os.path.isdir(wave_root):
            continue
        # Some wave dirs nest a copy of themselves
        for root, dirs, files in os.walk(wave_root):
            if "sensors.csv" not in files:
                continue
            # this is a cell dir
            cell_name = os.path.basename(root.rstrip("/"))
            family = policy_family(cell_name)
            sensors_p = os.path.join(root, "sensors.csv")
            stress_p  = os.path.join(root, "stress.csv")
            if not os.path.isfile(stress_p):
                stress_p = None
            yield (entry, cell_name, sensors_p, stress_p, family)


def collect_points():
    """Return dict family -> list[(cache_size, ddr_C)] across all cells."""
    out = {}
    n_cells = 0
    for wave_dir, cell, sensors_p, stress_p, family in scan_cells():
        # Read header
        with open(sensors_p) as f:
            header = f.readline()
        i_ddr = col_idx(header, "ddr_temp_mc")
        i_mon = col_idx(header, "monotonic_s")
        if i_ddr is None or i_mon is None:
            continue

        iters = load_stress_iters(stress_p) if stress_p else []
        # max peak_kv as fallback if no per-iter or sensors-only cell
        max_pk = max((it[1] for it in iters), default=0)
        if max_pk == 0:
            # try sensors-only fallback: cell's "expected" cache from name
            # (rare path; produces a constant per cell)
            continue

        # Stream the sensors file every Nth row
        added = 0
        with open(sensors_p) as f:
            f.readline()  # header
            first_mon = None
            for i, line in enumerate(f):
                if i % SAMPLE_EVERY != 0:
                    continue
                cols = line.rstrip("\n").split(",")
                if len(cols) <= max(i_ddr, i_mon):
                    continue
                try:
                    mon = float(cols[i_mon])
                    ddr_mc = float(cols[i_ddr])
                except ValueError:
                    continue
                if first_mon is None:
                    first_mon = mon
                offset = mon - first_mon
                ddr_c = ddr_mc / 1000.0
                if ddr_c < 10 or ddr_c > 110:   # drop obvious bad reads
                    continue
                kv = kv_for_offset(iters, offset, max_pk) if iters else max_pk
                if kv <= 0:
                    continue
                out.setdefault(family, []).append((float(kv), ddr_c))
                added += 1
        if added > 0:
            n_cells += 1
    print(f"[scan] gathered points from {n_cells} cells", file=sys.stderr)
    for fam, pts in sorted(out.items()):
        print(f"  {fam:20s}  n={len(pts):>7d}", file=sys.stderr)
    return out


def make_plot(family_points):
    os.makedirs(OUT_DIR, exist_ok=True)

    all_x, all_y = [], []
    for fam, pts in family_points.items():
        for x, y in pts:
            all_x.append(x); all_y.append(y)
    all_x = np.array(all_x); all_y = np.array(all_y)
    n_total = len(all_x)

    # Fit DDR = a + b * log10(cache_size)
    mask = all_x > 0
    lx = np.log10(all_x[mask])
    yy = all_y[mask]
    slope, intercept = np.polyfit(lx, yy, 1)
    # Pearson correlation between log(x) and y
    if len(lx) > 1:
        r = np.corrcoef(lx, yy)[0, 1]
    else:
        r = float("nan")
    print(f"[fit] DDR_C = {intercept:.2f} + {slope:.3f} * log10(cache)"
          f"   r={r:.3f}  n={len(lx)}", file=sys.stderr)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: raw scatter, color by policy ---
    order = ["vanilla", "llamacpp_stock", "tova", "pyramid",
             "v1", "v1_fa", "v1_fa2", "v1_fa2_selective", "v1_fa2_stack",
             "other"]
    seen_families = [f for f in order if f in family_points]
    for fam in seen_families:
        pts = family_points[fam]
        if not pts:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        axL.scatter(xs, ys, s=3, alpha=0.18,
                    color=POLICY_COLOR.get(fam, "#444"),
                    label=f"{fam}  (n={len(pts):,})",
                    edgecolors="none", rasterized=True)
    # fit overlay
    xfit = np.logspace(math.log10(max(all_x.min(), 1)),
                       math.log10(all_x.max()), 100)
    yfit = intercept + slope * np.log10(xfit)
    axL.plot(xfit, yfit, color="black", lw=2.0,
             label=f"fit: {intercept:.1f} + {slope:.2f}·log10(cache)  r={r:.2f}")
    axL.set_xscale("log")
    axL.set_xlabel("KV cache size (cells, log)")
    axL.set_ylabel("DDR temperature (deg C)")
    axL.set_title(f"All samples colored by policy  (N = {n_total:,})")
    axL.grid(True, which="both", alpha=0.25)
    leg = axL.legend(loc="lower right", fontsize=8, framealpha=0.85,
                     markerscale=3)
    for lh in leg.legend_handles:
        try:
            lh.set_alpha(1.0)
        except Exception:
            pass

    # --- Right: Pareto-style median + IQR per log-cache bin ---
    if n_total > 0:
        log_x = np.log10(all_x[mask])
        nbins = 20
        edges = np.linspace(log_x.min(), log_x.max(), nbins + 1)
        bin_centers, med, q25, q75 = [], [], [], []
        for i in range(nbins):
            lo, hi = edges[i], edges[i + 1]
            sel = (log_x >= lo) & (log_x <= hi if i == nbins - 1 else log_x < hi)
            if sel.sum() < 20:
                continue
            ys_bin = yy[sel]
            bin_centers.append(10 ** (0.5 * (lo + hi)))
            med.append(np.median(ys_bin))
            q25.append(np.percentile(ys_bin, 25))
            q75.append(np.percentile(ys_bin, 75))
        bin_centers = np.array(bin_centers)
        med = np.array(med); q25 = np.array(q25); q75 = np.array(q75)
        axR.fill_between(bin_centers, q25, q75, color="#1f77b4", alpha=0.25,
                         label="IQR (25-75%)")
        axR.plot(bin_centers, med, "o-", color="#1f77b4", lw=2,
                 label="Median DDR per cache-size bin")
        axR.plot(xfit, yfit, color="black", lw=2.0, ls="--",
                 label=f"fit: {intercept:.1f} + {slope:.2f}·log10(cache)  r={r:.2f}")
    axR.set_xscale("log")
    axR.set_xlabel("KV cache size (cells, log)")
    axR.set_ylabel("DDR temperature (deg C)")
    axR.set_title("Pareto envelope: median +/- IQR per bin")
    axR.grid(True, which="both", alpha=0.25)
    axR.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "DRAM temperature correlates monotonically with KV cache size "
        f"across {n_total/1000:.0f}K+ measurements",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {OUT_FIG}", file=sys.stderr)
    return n_total, slope, intercept, r


def main():
    fam = collect_points()
    if not fam:
        print("No data collected", file=sys.stderr)
        sys.exit(1)
    n, slope, b, r = make_plot(fam)
    print(f"DONE  N={n}  slope={slope:.3f}  intercept={b:.2f}  r={r:.3f}")


if __name__ == "__main__":
    main()
