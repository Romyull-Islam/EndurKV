#!/usr/bin/env python3
"""
host_wave3_real_analyze.py — analyze Wave-3 REAL sustained-stress data.

For each policy cell, produces:
  - Time-series plot: DDR + CPU temperature over the full 25-min cell
  - Per-iteration tok/s + prefill_ms trace (shows throttling)
  - Aggregate: peak/mean/p95 DDR + CPU, throttle onset, time at >55°C, cumulative tokens

Outputs:
  <wave3_dir>/WAVE3_REAL_SUMMARY.md
  <wave3_dir>/wave3_thermal_curves.png
  <wave3_dir>/wave3_throughput_decay.png
"""
import sys, csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICIES = ["vanilla", "v1_K2048", "v1_K512", "v1_fa_K512"]
COLORS   = {"vanilla":"#d62728", "v1_K2048":"#ff7f0e", "v1_K512":"#bcbd22", "v1_fa_K512":"#2ca02c"}
LABELS   = {"vanilla":"vanilla", "v1_K2048":"v1 K=2048", "v1_K512":"v1 K=512", "v1_fa_K512":"v1_fa K=512"}

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
            except Exception:
                pass
    return out

def pct(xs, p):
    if not xs: return float("nan")
    s = sorted(xs)
    k = int(p * (len(s)-1))
    return s[k]

def main():
    root = Path(sys.argv[1])

    # ---------- Figure 1: thermal curves ----------
    fig1, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_ddr, ax_cpu = axes
    for pol in POLICIES:
        s = load_sensors(root / pol / "sensors.csv")
        if not s: continue
        ts   = [r[0] for r in s]
        ddrs = [r[1] for r in s if r[1] is not None]
        ts_d = [r[0] for r in s if r[1] is not None]
        cpus = [r[2] for r in s if r[2] is not None]
        ts_c = [r[0] for r in s if r[2] is not None]
        c = COLORS[pol]
        ax_ddr.plot(ts_d, ddrs, color=c, linewidth=1.5, label=LABELS[pol])
        ax_cpu.plot(ts_c, cpus, color=c, linewidth=1.5, label=LABELS[pol])
    ax_ddr.set_ylabel("DDR temperature (°C)")
    ax_cpu.set_ylabel("CPU big-cluster (°C)")
    ax_cpu.set_xlabel("Time (s)")
    ax_ddr.axhline(60, color='red', linestyle=':', alpha=0.5, label='throttle ~60°C')
    ax_cpu.axhline(70, color='red', linestyle=':', alpha=0.5, label='throttle ~70°C')
    ax_ddr.legend(loc='lower right', fontsize=9); ax_ddr.grid(alpha=0.3)
    ax_cpu.legend(loc='lower right', fontsize=9); ax_cpu.grid(alpha=0.3)
    ax_ddr.set_title("Wave-3 REAL — sustained-stress thermal curves (Llama-3.2-1B)")
    fig1.tight_layout()
    out1 = root / "wave3_thermal_curves.png"
    fig1.savefig(out1, dpi=130)
    plt.close(fig1)

    # ---------- Figure 2: throughput decay ----------
    fig2, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_dec, ax_pre = axes
    for pol in POLICIES:
        st = load_stress(root / pol / "stress.csv")
        if not st: continue
        iters = [s["iter"] for s in st]
        dec = [s["decode_tps"] for s in st]
        pre = [s["prefill_ms"]/1000.0 for s in st]
        c = COLORS[pol]
        ax_dec.plot(iters, dec, "o-", color=c, label=LABELS[pol], linewidth=2)
        ax_pre.plot(iters, pre, "o-", color=c, label=LABELS[pol], linewidth=2)
    ax_dec.set_ylabel("Decode tok/s"); ax_dec.grid(alpha=0.3); ax_dec.legend(fontsize=9)
    ax_pre.set_ylabel("Prefill time (s)"); ax_pre.set_xlabel("Iteration")
    ax_pre.grid(alpha=0.3); ax_pre.legend(fontsize=9)
    ax_dec.set_title("Wave-3 REAL — throughput decay across iterations")
    fig2.tight_layout()
    out2 = root / "wave3_throughput_decay.png"
    fig2.savefig(out2, dpi=130)
    plt.close(fig2)

    # ---------- Summary table ----------
    rows = []
    for pol in POLICIES:
        s = load_sensors(root / pol / "sensors.csv")
        st = load_stress(root / pol / "stress.csv")
        if not s or not st:
            rows.append([LABELS[pol], "(no data)", "", "", "", "", "", "", ""])
            continue
        ddrs = [r[1] for r in s if r[1] is not None]
        cpus = [r[2] for r in s if r[2] is not None]
        n_iter = len(st)
        peak_ddr = max(ddrs) if ddrs else float("nan")
        mean_ddr = sum(ddrs)/len(ddrs) if ddrs else float("nan")
        p95_ddr  = pct(ddrs, 0.95) if ddrs else float("nan")
        peak_cpu = max(cpus) if cpus else float("nan")
        mean_cpu = sum(cpus)/len(cpus) if cpus else float("nan")
        time_above_55 = sum(1 for d in ddrs if d > 55) / len(ddrs) * 100 if ddrs else 0
        # Throughput evolution
        decs = [it["decode_tps"] for it in st]
        first_decode = decs[0]
        last_decode  = decs[-1]
        throttle_pct = (1 - last_decode/first_decode) * 100 if first_decode > 0 else 0
        total_tokens = sum(256 for _ in st)  # approx: each iter generates 256 tokens
        rows.append([
            LABELS[pol], n_iter,
            f"{peak_ddr:.1f}", f"{mean_ddr:.1f}", f"{p95_ddr:.1f}",
            f"{peak_cpu:.1f}", f"{mean_cpu:.1f}",
            f"{time_above_55:.0f}%",
            f"{first_decode:.2f}→{last_decode:.2f} (−{throttle_pct:.0f}%)"
        ])

    md = ["# Wave-3 REAL — sustained-stress summary\n",
          f"source: `{root}`  \nLlama-3.2-1B / narrativeqa_pub_001 / 4 threads CPU\n",
          "Each cell: 25 min continuous prefill+decode iterations after cold start (skin ≤ 33°C target)\n", ""]
    headers = ["Policy", "Iters", "Peak DDR", "Mean DDR", "P95 DDR",
               "Peak CPU", "Mean CPU", "Time DDR>55°C",
               "Decode tok/s (first→last)"]
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"]*len(headers)) + "|")
    for r in rows: md.append("| " + " | ".join(str(x) for x in r) + " |")

    md.append("")
    md.append("## Interpretation")
    md.append("- **Peak/mean DDR**: closer to throttle threshold (~60°C DDR / ~70°C CPU on Snapdragon 8 Elite Gen 5) → more throttling.")
    md.append("- **Time DDR>55°C**: fraction of session spent in the throttle-risk zone. Lower is better.")
    md.append("- **Decode tok/s first→last**: shows throughput decay due to throttling within the session.")
    md.append("  - Vanilla starts slow (full KV) but loses % over time.")
    md.append("  - v1/v1_fa start faster (smaller KV) but also throttle.")
    md.append("  - The policy that LOSES LESS THROUGHPUT under sustained load wins for HotMobile.")

    out = root / "WAVE3_REAL_SUMMARY.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {out}")
    print(f"wrote {out1}")
    print(f"wrote {out2}")

if __name__ == "__main__":
    main()
