#!/usr/bin/env python3
"""
host_kv_growth_summary.py — quantitative summary of the KV-growth thermal study.

For each condition (A=vanilla, B=v1 K=2048, C=v1 K=1024, D=v1 K=256), computes:
  - peak DDR temperature
  - peak CPU temperature (max big-core)
  - max KV cells held in cache
  - mean decode tok/s
  - peak — start DDR (the heating amount)
  - Pearson r between DDR temp and KV cells (the coupling)

Outputs a markdown table to <kvgrow_dir>/THERMAL_COUPLING.md plus prints it.

Usage:
  python host_kv_growth_summary.py <kvgrow_dir>
"""
import sys, csv, statistics
from pathlib import Path

def load_sensors(path):
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        def find(*needles):
            for c in cols:
                lc = c.lower()
                if all(n in lc for n in needles): return c
            return None
        t_col = find("monotonic_s") or cols[1]
        ddr_col = find("ddr_temp_mc") or find("ddr")
        cpu_col = find("cpullc-0-0") or find("cpullc")
        skin_col = find("shell_front")
        t0 = None
        for r in rd:
            try: t = float(r[t_col])
            except Exception: continue
            if t0 is None: t0 = t
            def g(col, scale=1.0):
                if not col: return None
                try: return float(r[col]) * scale
                except Exception: return None
            rows.append((t - t0,
                         g(ddr_col, 1/1000.0),
                         g(cpu_col, 1/1000.0),
                         g(skin_col, 1/1000.0)))
    return rows

def load_steps(path):
    if not Path(path).exists(): return []
    out = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                out.append((int(r.get("step", 0) or 0),
                            int(r.get("wall_us", 0) or 0),
                            int(r.get("n_kv_cells", 0) or 0)))
            except Exception:
                pass
    return out

def load_meta(path):
    if not Path(path).exists(): return {}
    import json
    try:
        return json.load(open(path))
    except Exception:
        return {}

def pearson(xs, ys):
    if len(xs) < 2: return float("nan")
    mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = (sum((x-mx)**2 for x in xs))**0.5
    dy = (sum((y-my)**2 for y in ys))**0.5
    return num/(dx*dy) if dx*dy > 1e-12 else float("nan")

def main():
    if len(sys.argv) < 2:
        print("usage: host_kv_growth_summary.py <kvgrow_dir>"); sys.exit(1)
    root = Path(sys.argv[1])
    cond_dirs = sorted(root.glob("cond_*"))
    if not cond_dirs:
        print(f"no cond_* in {root}"); sys.exit(1)

    rows = []
    header = ["Cond","Policy","K","Peak DDR (°C)","Peak CPU (°C)","Peak Skin (°C)",
              "DDR rise (°C)","Max KV cells","Mean tok/s","Pearson r (T vs KV)"]
    for d in cond_dirs:
        parts = d.name.split("_")
        cond = parts[1] if len(parts) > 1 else "?"
        policy = parts[2] if len(parts) > 2 else "?"
        k = parts[3].replace("K","") if len(parts) > 3 else "0"
        s = load_sensors(d / "sensors.csv")
        st = load_steps(d / "steps.csv")
        meta = load_meta(d / "meta.json")
        if not s:
            rows.append([cond, policy, k, "no data", "", "", "", "", "", ""])
            continue
        ddrs = [r[1] for r in s if r[1] is not None]
        cpus = [r[2] for r in s if r[2] is not None]
        skins = [r[3] for r in s if r[3] is not None]
        peak_ddr = max(ddrs) if ddrs else float("nan")
        peak_cpu = max(cpus) if cpus else float("nan")
        peak_skin = max(skins) if skins else float("nan")
        ddr_rise = peak_ddr - ddrs[0] if len(ddrs) >= 1 else float("nan")
        max_kv = max((x[2] for x in st), default=meta.get("peak_kv_cells", 0))
        mean_tps = meta.get("decode_tps", 0.0)
        # Pearson r between DDR temp and KV cells across the decode window
        # we need to align by time — sensors at 5 Hz, steps at ~3 Hz; pick samples
        if st and ddrs:
            # Step wall_us is from t_start; sensor t_s is from sampler start (close enough).
            # Build (t_s, n_kv) interpolation via step CSV
            t_kv = [(s_row[1]/1e6, s_row[2]) for s_row in st]   # (t in sec, n_kv)
            # paired (ddr, kv) at sensor times
            xs, ys = [], []
            j = 0
            for (t, ddr, cpu, skin) in s:
                if ddr is None: continue
                while j+1 < len(t_kv) and t_kv[j+1][0] < t: j += 1
                xs.append(ddr); ys.append(t_kv[j][1])
            r_val = pearson(xs, ys)
        else:
            r_val = float("nan")
        rows.append([cond, policy, k,
                     f"{peak_ddr:.1f}", f"{peak_cpu:.1f}", f"{peak_skin:.1f}",
                     f"{ddr_rise:+.1f}",
                     f"{int(max_kv)}", f"{mean_tps:.2f}", f"{r_val:+.2f}"])

    # Render markdown
    md = ["# KV-growth thermal coupling — summary\n",
          f"source: `{root}`\n",
          ""]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---"]*len(header)) + "|")
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")

    # Interpretation
    md.append("")
    md.append("## Interpretation")
    md.append("- **Pearson r near +1**: DDR temperature rises in lockstep with KV cells → KV-bandwidth → heat chain is REAL on this device.")
    md.append("- **Pearson r near 0**: KV size and DDR temp uncorrelated → KV bandwidth is not the dominant heat source. Track-2's KV lever has limited authority.")
    md.append("- **DDR rise**: total heating during the cell. Compare vanilla vs v1 K=256 — if vanilla rises by ~3-5× more, eviction is buying real thermal headroom.")
    md.append("- **Peak DDR**: closer to the throttle threshold (~70 °C on Snapdragon 8 Elite Gen 5) is worse. Aggressive eviction (K=256) should stay well below.")
    out_md = root / "THERMAL_COUPLING.md"
    out_md.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {out_md}")

if __name__ == "__main__":
    main()
