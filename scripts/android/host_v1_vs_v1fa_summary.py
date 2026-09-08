#!/usr/bin/env python3
"""
host_v1_vs_v1fa_summary.py — quantitative summary of v1 vs v1_fa thermal A/B.

For each policy cell (v1, v1_fa) the experiment runs N iterations of
prefill+decode. We need to extract:
  - Peak DDR during each iteration's decode (where FA-on benefit shows)
  - Mean decode tok/s
  - Quality (perplexity / mean_nll)

Outputs a markdown table and prints it.

Usage:
  python host_v1_vs_v1fa_summary.py <v1_vs_v1fa_dir>
"""
import sys, csv, json
from pathlib import Path

def load_sensors(path):
    rows = []
    if not Path(path).exists(): return rows
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        def find(*needles):
            for c in cols:
                lc = c.lower()
                if all(n in lc for n in needles): return c
            return None
        t_col = find("monotonic_s") or cols[1]
        ddr_col = find("ddr_temp_mc")
        cpu_col = find("cpullc-0-0")
        skin_col = find("shell_front")
        t0 = None
        for r in rd:
            try: t = float(r[t_col])
            except Exception: continue
            if t0 is None: t0 = t
            def g(c, s=1.0):
                if not c: return None
                try: return float(r[c]) * s
                except Exception: return None
            rows.append((t - t0,
                         g(ddr_col, 1/1000.0),
                         g(cpu_col, 1/1000.0),
                         g(skin_col, 1/1000.0)))
    return rows

def main():
    if len(sys.argv) < 2:
        print("usage: host_v1_vs_v1fa_summary.py <v1_vs_v1fa_dir>"); sys.exit(1)
    root = Path(sys.argv[1])
    cells = sorted([d for d in root.glob("*_K*") if d.is_dir()])
    if not cells:
        print(f"no */*_K* cells in {root}"); sys.exit(1)

    rows = []
    header = ["Policy", "K", "N iters", "Mean prefill (s)", "Mean decode tok/s",
              "Mean PPL", "Mean peak DDR (°C)", "Mean peak CPU (°C)", "Mean DDR rise (°C)"]
    for cell in cells:
        name = cell.name
        policy = name.split("_K")[0]
        k = name.split("_K")[1]
        sensors = load_sensors(cell / "sensors.csv")
        ddrs = [r[1] for r in sensors if r[1] is not None]
        cpus = [r[2] for r in sensors if r[2] is not None]

        iters = sorted(cell.glob("iter*"))
        prefills, tps, ppls = [], [], []
        for it in iters:
            meta_path = it / "meta.json"
            if not meta_path.exists(): continue
            try:
                m = json.loads(meta_path.read_text())
                prefills.append(m.get("prefill_ms", 0) / 1000.0)
                tps.append(m.get("decode_tps", 0))
                ppls.append(m.get("perplexity", 0))
            except Exception:
                pass
        n_iter = len(prefills)
        mp = sum(prefills)/n_iter if n_iter else 0
        mt = sum(tps)/n_iter if n_iter else 0
        mq = sum(ppls)/n_iter if n_iter else 0
        peak_ddr = max(ddrs) if ddrs else float("nan")
        peak_cpu = max(cpus) if cpus else float("nan")
        ddr_rise = peak_ddr - ddrs[0] if ddrs else float("nan")
        rows.append([policy, k, n_iter,
                     f"{mp:.1f}", f"{mt:.2f}", f"{mq:.3f}",
                     f"{peak_ddr:.1f}", f"{peak_cpu:.1f}", f"{ddr_rise:+.1f}"])

    md = ["# v1 vs v1_fa thermal A/B — summary\n",
          f"source: `{root}`\n", ""]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---"]*len(header)) + "|")
    for r in rows: md.append("| " + " | ".join(str(x) for x in r) + " |")
    md.append("")
    md.append("## Interpretation")
    md.append("- **Peak DDR delta**: v1_fa runs FA-on during decode → less DRAM bandwidth → expected to be cooler than v1. If v1_fa peak DDR is 2-5 °C below v1, that's the thermal benefit of FA-on.")
    md.append("- **tok/s delta**: v1_fa decode should be FASTER (FA-on is more efficient) — typical 1.5-2× vs FA-off.")
    md.append("- **Quality (PPL)**: should be approximately equal (both use same v1 spread-gate; only the decode kernel differs).")
    out = root / "V1_VS_V1FA_SUMMARY.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
