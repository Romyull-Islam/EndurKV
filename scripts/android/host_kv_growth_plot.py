#!/usr/bin/env python3
"""
host_kv_growth_plot.py — generate the foundational KV-growth thermal figure.

Reads each kvgrow condition (cond_{A,B,C,D}_*) directory and aligns:
  - sensors.csv: time-series of DDR temp, CPU temp, skin temp, battery temp
  - steps.csv:   per-decode-step KV cell count and tok/s

Produces:
  - kv_growth_temperature.png — DDR temp vs time, one line per condition,
    with KV cache size as a secondary annotation.

Usage:
  python host_kv_growth_plot.py <kvgrow_dir>

Where <kvgrow_dir> is the host-side directory holding cond_*/sensors.csv etc.
"""
import sys, os, json, csv, glob
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# sensors.csv schema (from sample_sensors.sh):
#   t_wall_us, then a bunch of thermal_zone* / battery / freq / pid columns
# We robustly find: t (time), ddr (NSP HMX or "ddr"-named zone), cpu, skin, batt
# ---------------------------------------------------------------------------
def load_sensors(path):
    """Return list of dicts with t_s, ddr_c, cpu_c, skin_c, batt_c (best effort)."""
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        # heuristic column matching
        def find(*needles):
            for c in cols:
                lc = c.lower()
                if all(n in lc for n in needles):
                    return c
            return None
        # sample_sensors.sh v3 columns: wall_clock_s, monotonic_s, then many *_temp_mc
        t_col   = find("monotonic_s") or find("wall_clock_s") or find("t_wall_us") or cols[0]
        # DDR temp: `ddr_temp_mc` (millicelsius)
        ddr_col = find("ddr_temp_mc") or find("ddr") or find("nsphmx-2")
        # CPU big cluster: cpullc-0-0_temp_mc is fine
        cpu_col = find("cpullc-0-0") or find("cpullc") or find("cpu0") or find("cpu-0-0-0")
        # "Skin" proxy: shell_front_temp_mc (phone skin), fallback to bat_phone_temp_dc
        skin_col = find("shell_front") or find("shell_frame") or find("skin")
        batt_col = find("bat_phone_temp") or find("battery_temp_mc") or find("batt")
        t0 = None
        for r in rd:
            try:
                # monotonic_s is already seconds (float)
                tv = float(r[t_col])
            except Exception:
                continue
            t = tv if "wall_us" not in t_col else tv / 1e6
            if t0 is None: t0 = t
            def g(col, scale=1.0):
                if not col: return None
                try: return float(r[col]) * scale
                except Exception: return None
            rows.append({
                "t_s":   t - t0,
                # most temp columns are millicelsius → /1000
                "ddr_c": g(ddr_col,  1/1000.0) if ddr_col else None,
                "cpu_c": g(cpu_col,  1/1000.0) if cpu_col else None,
                # shell_front_temp_mc is millicelsius; bat_phone_temp_dc is deci-celsius
                "skin_c":(g(skin_col, 1/1000.0) if skin_col and "_mc" in skin_col else
                          g(skin_col, 0.1)      if skin_col else None),
            })
    return rows, cols

def load_steps(path):
    """Return list of (step, t_s_from_start, n_kv_cells, tok_text)."""
    out = []
    if not Path(path).exists(): return out
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                out.append({
                    "step":     int(r.get("step", -1)),
                    "wall_us":  int(r.get("wall_us", 0) or 0),
                    "n_kv":     int(r.get("n_kv_cells", 0) or 0),
                    "tok_text": r.get("token_text", ""),
                })
            except Exception:
                pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kvgrow_dir", help="path containing cond_*_*_K* dirs")
    ap.add_argument("--out", default=None, help="output PNG path (default: <kvgrow_dir>/kv_growth_temperature.png)")
    args = ap.parse_args()

    root = Path(args.kvgrow_dir)
    if not root.is_dir():
        print(f"ERR: {root} not a dir"); sys.exit(1)

    cond_dirs = sorted(root.glob("cond_*"))
    if not cond_dirs:
        print(f"ERR: no cond_* dirs in {root}"); sys.exit(1)

    fig, ax_t = plt.subplots(figsize=(11, 6))
    ax_kv = ax_t.twinx()

    colors  = {"A": "tab:red", "B": "tab:orange", "C": "tab:olive", "D": "tab:green"}
    labels  = {"A": "vanilla (KV grows)", "B": "v1 K=2048", "C": "v1 K=1024", "D": "v1 K=256"}

    for d in cond_dirs:
        cond = d.name.split("_")[1]  # cond_A_vanilla_K0 → A
        s_path = d / "sensors.csv"
        st_path = d / "steps.csv"
        if not s_path.exists():
            print(f"  skip {d}: no sensors.csv")
            continue
        rows, cols = load_sensors(s_path)
        if not rows:
            print(f"  skip {d}: empty sensors")
            continue
        steps = load_steps(st_path)

        ts = [r["t_s"] for r in rows]
        ddr = [r["ddr_c"] for r in rows if r["ddr_c"] is not None]
        ts_ddr = [r["t_s"] for r in rows if r["ddr_c"] is not None]
        if ddr:
            color = colors.get(cond, "tab:gray")
            ax_t.plot(ts_ddr, ddr, label=f"{labels.get(cond, cond)} - DDR",
                      color=color, linewidth=2, alpha=0.85)

        # Overlay KV cells if steps available
        if steps:
            # steps wall_us is from t_start (eviction_bench), need to align with sensors.
            # Use the first step's wall_us as origin offset.
            t0_step = steps[0]["wall_us"]
            xs = [(s["wall_us"] - t0_step)/1e6 for s in steps]
            ys = [s["n_kv"] for s in steps]
            color = colors.get(cond, "tab:gray")
            ax_kv.plot(xs, ys, "--", color=color, linewidth=1, alpha=0.5)

    ax_t.set_xlabel("Time since cell start (s)")
    ax_t.set_ylabel("DDR temperature (°C)", color="tab:red")
    ax_kv.set_ylabel("KV cells (dashed)", color="tab:gray")
    ax_t.legend(loc="upper left", fontsize=9)
    ax_t.set_title("KV cache size vs DDR temperature — foundational thermal study (Llama-3.2-1B)")
    ax_t.grid(True, alpha=0.3)

    out = args.out or (root / "kv_growth_temperature.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
