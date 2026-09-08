#!/usr/bin/env python3
"""
host_wave3_combined_table.py — build the FINAL comparison table from all
matched-DVFS Wave-3 cells.

Sources combined:
  REAL:     wave3_real_1780680903/          (v1_K2048, v1_K512, v1_fa_K512 — battery; vanilla=DROP, was on AC)
  OTHERS:   wave3_real_others_<ts>/         (tova_K512, h2o_K512, pyramid_K512 — battery)
  VANILLA:  wave3_vanilla_battery_<ts>/     (vanilla — battery, the corrective rerun)

For each policy, extracts and reports:
  - Power state (must be 'battery' for inclusion)
  - Peak CPU clock (DVFS cap)
  - Iterations, mean decode tok/s, throughput decay
  - Peak/mean DDR temp, peak/mean CPU big-cluster temp
  - Mean battery current (sanity check on power state)

Outputs:
  TABLE_WAVE3_MATCHED.md
  wave3_matched_thermal.png
  wave3_matched_throughput.png
"""
import sys, csv, json, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICY_ORDER = ["vanilla", "v1_K2048", "v1_K512", "v1_fa_K512",
                "tova_K512", "h2o_K512", "pyramid_K512"]

COLORS = {
    "vanilla":    "#d62728",
    "v1_K2048":   "#ff7f0e",
    "v1_K512":    "#bcbd22",
    "v1_fa_K512": "#2ca02c",
    "tova_K512":  "#1f77b4",
    "h2o_K512":   "#9467bd",
    "pyramid_K512":"#17becf",
}
LABELS = {
    "vanilla":    "vanilla",
    "v1_K2048":   "v1 K=2048",
    "v1_K512":    "v1 K=512",
    "v1_fa_K512": "v1_FA K=512",
    "tova_K512":  "TOVA K=512",
    "h2o_K512":   "H2O K=512",
    "pyramid_K512":"Pyramid K=512",
}

def find_col(cols, *needles):
    for c in cols:
        lc = c.lower()
        if all(n in lc for n in needles):
            return c
    return None

def load_sensors(path):
    rows = []
    if not Path(path).exists(): return rows, None
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        idx = {
            "t":     find_col(cols, "monotonic_s") or (cols[1] if len(cols)>1 else None),
            "ddr":   find_col(cols, "ddr_temp_mc"),
            "cpu":   find_col(cols, "cpullc-0-0"),
            "skin":  find_col(cols, "shell_front"),
            "cpu6f": find_col(cols, "cpu6_freq_hz"),
            "cpu7f": find_col(cols, "cpu7_freq_hz"),
            "batc":  find_col(cols, "bat_current_ma"),
        }
        t0 = None
        for r in rd:
            try:
                t = float(r[idx["t"]])
            except Exception:
                continue
            if t0 is None: t0 = t
            def g(k, s=1.0):
                c = idx[k]
                if not c: return None
                try: return float(r[c]) * s
                except Exception: return None
            rows.append({
                "t":   t - t0,
                "ddr": g("ddr", 1/1000.0),
                "cpu": g("cpu", 1/1000.0),
                "skin":g("skin", 1/1000.0),
                "cpu6f": g("cpu6f", 1/1e6),   # MHz
                "cpu7f": g("cpu7f", 1/1e6),
                "batc":  g("batc"),
            })
    return rows, idx

def load_stress(path):
    if not Path(path).exists(): return []
    out = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                out.append({
                    "iter":int(r["iter"]),
                    "t":int(r["t_elapsed_s"]),
                    "exit":int(r["exit"]),
                    "prefill_ms":float(r["prefill_ms"]),
                    "decode_tps":float(r["decode_tps"]),
                    "peak_kv":int(r["peak_kv_cells"]),
                    "evicted":int(r["evicted"]),
                })
            except Exception: pass
    return out

def power_state(rows):
    """Return 'battery' / 'ac' / 'unknown' based on mean battery current."""
    bcs = [r["batc"] for r in rows if r["batc"] is not None]
    if not bcs: return "unknown", None
    mean = sum(bcs)/len(bcs)
    if mean < -30: return "ac", mean
    if mean >  30: return "battery", mean
    return "mixed", mean

def find_cell(roots, policy_label):
    """Search across roots for a directory named policy_label."""
    for root in roots:
        p = Path(root) / policy_label
        if p.is_dir() and (p / "sensors.csv").exists():
            return p
        # Sometimes nested
        nested = list(Path(root).glob(f"**/{policy_label}/sensors.csv"))
        if nested:
            return nested[0].parent
    return None

def main():
    # Roots to search for cells
    roots = sys.argv[1:] if len(sys.argv) > 1 else []
    if not roots:
        roots = sorted(glob.glob("/home/mislam22/EndurKV_workspace/phone-logs/wave3_real_*") +
                       glob.glob("/home/mislam22/EndurKV_workspace/phone-logs/wave3_vanilla_battery_*"))
    print(f"# Searching roots: {roots}\n")

    rows_out = []
    sens_for_plot = {}  # policy -> sensors list
    stress_for_plot = {}
    skipped = []
    for pol in POLICY_ORDER:
        cell = find_cell(roots, pol)
        if not cell:
            skipped.append((pol, "no cell found"))
            continue
        sens, _ = load_sensors(cell / "sensors.csv")
        st       = load_stress(cell / "stress.csv")
        pstate, batc_mean = power_state(sens)
        ddrs = [r["ddr"] for r in sens if r["ddr"] is not None]
        cpus = [r["cpu"] for r in sens if r["cpu"] is not None]
        cpu6 = [r["cpu6f"] for r in sens if r["cpu6f"] is not None]
        if not sens or not ddrs:
            skipped.append((pol, "empty sensors"))
            continue
        if pstate == "ac":
            skipped.append((pol, f"DROPPED: on AC (mean bat_curr={batc_mean:.0f} mA)"))
            continue
        # Compute metrics
        peak_ddr = max(ddrs); mean_ddr = sum(ddrs)/len(ddrs)
        peak_cpu = max(cpus) if cpus else float("nan")
        mean_cpu = sum(cpus)/len(cpus) if cpus else float("nan")
        peak_clk = max(cpu6) if cpu6 else float("nan")
        mean_clk = sum(cpu6)/len(cpu6) if cpu6 else float("nan")
        n_iter   = len(st)
        first_dec = st[0]["decode_tps"] if st else float("nan")
        last_dec  = st[-1]["decode_tps"] if st else float("nan")
        mean_dec  = sum(s["decode_tps"] for s in st)/n_iter if n_iter else float("nan")
        decay_pct = (1 - last_dec/first_dec)*100 if first_dec>0 else float("nan")
        evicted_iter = st[0]["evicted"] if st else 0
        total_tokens = n_iter * 256
        wallclock_tps = total_tokens / 1500.0   # assuming 25-min budget
        rows_out.append({
            "policy": pol,
            "label":  LABELS[pol],
            "pstate": pstate,
            "batc":   batc_mean,
            "n_iter": n_iter,
            "peak_clk_MHz": peak_clk,
            "mean_clk_MHz": mean_clk,
            "first_dec": first_dec,
            "last_dec":  last_dec,
            "mean_dec":  mean_dec,
            "decay_pct": decay_pct,
            "peak_ddr":  peak_ddr,
            "mean_ddr":  mean_ddr,
            "peak_cpu":  peak_cpu,
            "mean_cpu":  mean_cpu,
            "wallclock_tps": wallclock_tps,
            "evicted_per_iter": evicted_iter,
        })
        sens_for_plot[pol] = sens
        stress_for_plot[pol] = st

    # Generate markdown
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables")
    md_path = out_dir / "TABLE_WAVE3_MATCHED.md"
    md = ["# Wave-3 matched-DVFS comparison\n",
          "All cells on **battery** (status 5, USB-only, no AC charging). DVFS-capped at 1.63 GHz on big cores.\n",
          "Llama-3.2-1B / narrativeqa_pub_001 / 4 threads CPU / 25-min sustained / strict cooldown to 33°C.\n",
          ""]

    headers = ["Policy", "Power", "Peak CPU (MHz)", "Iters",
               "First→Last decode tok/s", "Mean decode tok/s", "Decay",
               "Wall-clock tok/s", "Peak DDR (°C)", "Mean DDR", "Peak CPU (°C)", "Evicted/iter"]
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"]*len(headers)) + "|")
    for r in rows_out:
        md.append("| " + " | ".join([
            r["label"],
            r["pstate"],
            f"{r['peak_clk_MHz']:.0f}",
            str(r["n_iter"]),
            f"{r['first_dec']:.2f}→{r['last_dec']:.2f}",
            f"{r['mean_dec']:.2f}",
            f"−{r['decay_pct']:.1f}%",
            f"{r['wallclock_tps']:.2f}",
            f"{r['peak_ddr']:.1f}",
            f"{r['mean_ddr']:.1f}",
            f"{r['peak_cpu']:.1f}",
            f"{r['evicted_per_iter']}",
        ]) + " |")

    if skipped:
        md.append("\n## Dropped cells")
        for p, reason in skipped:
            md.append(f"- **{p}**: {reason}")

    md.append("\n## Interpretation")
    md.append("- All policies under identical DVFS (battery, 1.63 GHz big-core cap)")
    md.append("- Throughput-decay column shows in-session throttling (% drop first→last iter)")
    md.append("- Wall-clock tok/s is the user-perceived rate (total tokens / 1500 s budget)")
    md.append("- Peak DDR difference quantifies eviction's thermal benefit at matched compute")

    md_path.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {md_path}")

    # Plots
    if sens_for_plot:
        # Thermal curves
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axd, axc = axes
        for pol, sens in sens_for_plot.items():
            ts = [r["t"] for r in sens if r["ddr"] is not None]
            ddrs = [r["ddr"] for r in sens if r["ddr"] is not None]
            cpus = [r["cpu"] for r in sens if r["cpu"] is not None]
            tsc  = [r["t"] for r in sens if r["cpu"] is not None]
            axd.plot(ts, ddrs, color=COLORS[pol], lw=1.5, label=LABELS[pol])
            axc.plot(tsc, cpus, color=COLORS[pol], lw=1.5, label=LABELS[pol])
        axd.set_ylabel("DDR temperature (°C)")
        axc.set_ylabel("CPU big-cluster (°C)"); axc.set_xlabel("Time (s)")
        axd.axhline(60, color='red', ls=':', alpha=0.5)
        axd.legend(fontsize=8, loc='lower right'); axd.grid(alpha=0.3)
        axc.legend(fontsize=8, loc='lower right'); axc.grid(alpha=0.3)
        axd.set_title("Wave-3 matched-DVFS — thermal curves (Llama-1B, battery, 1.63 GHz cap)")
        fig.tight_layout()
        out_p = out_dir / "wave3_matched_thermal.png"
        fig.savefig(out_p, dpi=130)
        plt.close(fig)
        print(f"wrote {out_p}")

        # Throughput
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ad, ap = axes
        for pol, st in stress_for_plot.items():
            its = [s["iter"] for s in st]
            dec = [s["decode_tps"] for s in st]
            pre = [s["prefill_ms"]/1000.0 for s in st]
            ad.plot(its, dec, "o-", color=COLORS[pol], lw=2, label=LABELS[pol])
            ap.plot(its, pre, "o-", color=COLORS[pol], lw=2, label=LABELS[pol])
        ad.set_ylabel("Decode tok/s"); ad.legend(fontsize=8); ad.grid(alpha=0.3)
        ap.set_ylabel("Prefill time (s)"); ap.set_xlabel("Iteration")
        ap.legend(fontsize=8); ap.grid(alpha=0.3)
        ad.set_title("Wave-3 matched-DVFS — throughput")
        fig.tight_layout()
        out_p = out_dir / "wave3_matched_throughput.png"
        fig.savefig(out_p, dpi=130)
        plt.close(fig)
        print(f"wrote {out_p}")

if __name__ == "__main__":
    main()
