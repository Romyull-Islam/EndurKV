#!/usr/bin/env python3
"""
host_make_io_thermal_plots.py — visualise the I/O <-> heat coupling.

For a given log directory (short-ctx or long-ctx study), produce:

  01_ddr_temp_vs_memory.png        — RAM controller temp over time + MemAvailable
                                     overlay. Tests: does the LPDDR5X PHY heat
                                     visibly with memory pressure?
  02_flash_temp_vs_writes.png      — UFS controller temp over time + cumulative
                                     pswpout (4 KB pages) overlay. Tests: does
                                     the flash chip warm with swap activity?
  03_temp_decomposition.png        — Per-prompt peak temps split into CPU/DDR/
                                     flash families. Tests: which subsystem
                                     dominates the thermal envelope?
  04_io_to_heat_scatter.png        — Cross-prompt scatter: bytes-to-UFS (from
                                     pswpout + storaged delta when available)
                                     vs Δflash_temp_C. Quantifies "MB per
                                     degree of flash heating".
  05_ram_pressure_to_heat.png      — Cross-prompt scatter: working-set drop
                                     (max MemTotal − MemAvail) vs ΔDDR_temp_C.

Run:
  python scripts/android/host_make_io_thermal_plots.py --log-dir logs/study_phone_8b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", r"D:/Research/EndurKV_workspace"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir.parent / f"{log_dir.name}_iothermal"
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, pearsonr

    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9})

    # ----------------------------------------------------------------
    # Load every prompt's sensors.csv + run.json
    # ----------------------------------------------------------------
    prompts = []
    for sen_p in sorted(log_dir.glob("*.sensors.csv")):
        pid = sen_p.name[: -len(".sensors.csv")]
        run_p = log_dir / f"{pid}.run.json"
        if not run_p.exists():
            continue
        try:
            sen = pd.read_csv(sen_p)
            run = json.loads(run_p.read_text())
        except Exception:
            continue
        if sen.empty:
            continue
        sen = sen.sort_values("wall_clock_s").reset_index(drop=True)
        t0 = float(sen["wall_clock_s"].iloc[0])
        sen["t_rel"] = sen["wall_clock_s"] - t0
        task = pid.rsplit("_", 1)[0] if "_" in pid else "?"
        # Long-ctx prompt_id format is "task_lc_NN"; collapse to task name.
        if task.endswith("_lc"):
            task = task[:-3]
        prompts.append({"id": pid, "task": task, "sen": sen, "run": run})

    if not prompts:
        print(f"ERROR: no prompts in {log_dir}", file=sys.stderr)
        return 1
    print(f"[load] {len(prompts)} prompts from {log_dir}")

    def col(p, c):
        if c not in p["sen"].columns:
            return None
        return pd.to_numeric(p["sen"][c], errors="coerce")

    # ===============================================================
    # 01 — DDR (RAM controller) temp + MemAvailable, 4 sample prompts
    # ===============================================================
    pick_long = sorted(prompts, key=lambda p: -len(p["sen"]))[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, p in zip(axes.flat, pick_long):
        sen = p["sen"]
        ddr = col(p, "ddr_temp_mc")
        ma = col(p, "mem_avail_kb")
        if ddr is None or ma is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        ax2 = ax.twinx()
        ax.plot(sen["t_rel"], ddr / 1000.0, color="tab:red", lw=1.4, label="ddr_temp (°C)")
        ax2.plot(sen["t_rel"], ma / 1024.0, color="tab:blue", lw=1.0, alpha=0.7,
                 label="MemAvailable (MiB)")
        ax.set_xlabel("time since prompt start (s)")
        ax.set_ylabel("DDR temperature (°C)", color="tab:red")
        ax2.set_ylabel("MemAvailable (MiB)", color="tab:blue")
        ax.set_title(f"{p['id']}  ({len(sen)} samples)")
        ax.grid(alpha=0.3)
    fig.suptitle("01 — RAM controller temperature vs MemAvailable (4 longest prompts)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "01_ddr_temp_vs_memory.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 01_ddr_temp_vs_memory.png")

    # ===============================================================
    # 02 — Flash temp + cumulative pswpout, 4 prompts with most I/O
    # ===============================================================
    def pswpout_delta(p):
        c = col(p, "vmstat_pswpout")
        if c is None: return 0
        c = c.dropna()
        if len(c) < 2: return 0
        return int(c.iloc[-1] - c.iloc[0])
    pick_io = sorted(prompts, key=lambda p: -pswpout_delta(p))[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, p in zip(axes.flat, pick_io):
        sen = p["sen"]
        flash = col(p, "flash_temp_temp_mc")
        ps = col(p, "vmstat_pswpout")
        if flash is None or ps is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        ax2 = ax.twinx()
        ax.plot(sen["t_rel"], flash / 1000.0, color="tab:red", lw=1.4, label="flash_temp (°C)")
        # cumulative bytes from the run-start (delta), MiB
        ps_start = float(ps.iloc[0])
        ax2.plot(sen["t_rel"], (ps - ps_start) * 4 / 1024.0, color="tab:purple",
                 lw=1.1, alpha=0.8, label="Δpswpout (MiB to UFS via swap)")
        delta_total = int(ps.iloc[-1] - ps_start)
        ax.set_xlabel("time since prompt start (s)")
        ax.set_ylabel("flash_temp (°C)", color="tab:red")
        ax2.set_ylabel("cumulative UFS write (MiB) via swap", color="tab:purple")
        ax.set_title(f"{p['id']}  Δpswpout={delta_total*4/1024:.1f} MiB ({len(sen)} samples)")
        ax.grid(alpha=0.3)
    fig.suptitle("02 — Storage controller temperature vs cumulative UFS writes (top-I/O prompts)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "02_flash_temp_vs_writes.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 02_flash_temp_vs_writes.png")

    # ===============================================================
    # 03 — Per-prompt peak temperatures: CPU / DDR / flash side-by-side
    # ===============================================================
    rows = []
    TRIP_PATTERNS = ("trip", "bcl-lvl", "ibat-lvl", "vbat", "pmh", "pmr", "pmih",
                     "wireless", "usb", "sdr")
    def is_trip(c): return any(p in c.lower() for p in TRIP_PATTERNS)

    for p in prompts:
        sen = p["sen"]
        cpu_cols = [c for c in sen.columns if "cpu" in c.lower()
                    and c.endswith("_temp_mc") and not is_trip(c)]
        max_cpu = 0.0
        if cpu_cols:
            cpu_vals = sen[cpu_cols].apply(pd.to_numeric, errors="coerce") / 1000.0
            try: max_cpu = float(cpu_vals.max().max())
            except Exception: pass
        ddr = col(p, "ddr_temp_mc")
        flash = col(p, "flash_temp_temp_mc")
        max_ddr = float(ddr.max())/1000.0 if ddr is not None and not ddr.dropna().empty else 0.0
        max_fl  = float(flash.max())/1000.0 if flash is not None and not flash.dropna().empty else 0.0
        rows.append({"task": p["task"], "max_cpu": max_cpu, "max_ddr": max_ddr, "max_fl": max_fl})
    df = pd.DataFrame(rows)
    tasks = sorted(df.task.unique())
    fig, ax = plt.subplots(figsize=(14, 5.5))
    positions = np.arange(len(tasks))
    width = 0.27
    cpu_d  = [df[df.task == t]["max_cpu"].values for t in tasks]
    ddr_d  = [df[df.task == t]["max_ddr"].values for t in tasks]
    fl_d   = [df[df.task == t]["max_fl"].values for t in tasks]
    bp_cpu = ax.boxplot(cpu_d, positions=positions - width, widths=width*0.9,
                        patch_artist=True, boxprops=dict(facecolor="#f4a8a8"))
    bp_ddr = ax.boxplot(ddr_d, positions=positions,         widths=width*0.9,
                        patch_artist=True, boxprops=dict(facecolor="#cfe1f5"))
    bp_fl  = ax.boxplot(fl_d,  positions=positions + width, widths=width*0.9,
                        patch_artist=True, boxprops=dict(facecolor="#fbd9bf"))
    ax.set_xticks(positions); ax.set_xticklabels(tasks, rotation=35, ha="right")
    ax.axhline(70.0, color="red", ls="--", lw=1.0, alpha=0.6, label="proposal CPU trip 70°C")
    ax.set_ylabel("peak temperature per prompt (°C)")
    ax.set_title("03 — Per-prompt peak temp: CPU (red), DDR/RAM ctrl (blue), UFS flash (orange)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(handles=[bp_cpu["boxes"][0], bp_ddr["boxes"][0], bp_fl["boxes"][0],
                       plt.Line2D([0], [0], color="red", ls="--")],
              labels=["max CPU active zone", "ddr_temp (RAM PHY)",
                      "flash_temp (UFS ctrl)", "70 °C CPU trip"],
              loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "03_temp_decomposition.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 03_temp_decomposition.png")

    # ===============================================================
    # 04 — Cross-prompt scatter: bytes-to-UFS vs ΔFlash_temp
    # ===============================================================
    rows = []
    for p in prompts:
        sen = p["sen"]
        flash = col(p, "flash_temp_temp_mc")
        ps = col(p, "vmstat_pswpout")
        if flash is None or ps is None or flash.dropna().empty or ps.dropna().empty:
            continue
        d_flash = float(flash.iloc[-1] - flash.iloc[0]) / 1000.0
        d_ps_bytes = max(int(ps.iloc[-1] - ps.iloc[0]), 0) * 4 * 1024  # bytes
        # ground-truth from run.json storaged delta if present
        st_start = p["run"].get("storaged_bytes_written_at_start")
        st_end   = p["run"].get("storaged_bytes_written_at_end")
        d_storaged = None
        if st_start is not None and st_end is not None:
            try:
                d_storaged = max(int(st_end) - int(st_start), 0)
            except Exception:
                d_storaged = None
        rows.append({"task": p["task"], "d_pswpout_bytes": d_ps_bytes,
                     "d_storaged_bytes": d_storaged, "d_flash_C": d_flash})
    df_io = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    # plot pswpout delta (always available)
    sub = df_io[df_io["d_pswpout_bytes"] > 0]
    if len(sub) >= 5:
        x = sub["d_pswpout_bytes"] / (1024*1024)
        y = sub["d_flash_C"]
        ax.scatter(x, y, s=18, alpha=0.6, c="tab:purple")
        try:
            rho, p_val = spearmanr(x, y)
            ax.set_title(f"04a — Δpswpout (swap-out) vs Δflash_temp  (ρ={rho:+.2f}, n={len(sub)})")
        except Exception:
            ax.set_title("04a — Δpswpout vs Δflash_temp")
    else:
        ax.text(0.5, 0.5, f"n={len(sub)} prompts with pswpout>0 — insufficient for correlation",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("04a — Δpswpout vs Δflash_temp")
    ax.set_xlabel("Δpswpout per prompt (MiB to UFS via swap)")
    ax.set_ylabel("Δflash_temp per prompt (°C)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)

    # storaged scatter (only if any prompts have it)
    ax = axes[1]
    sub2 = df_io[df_io["d_storaged_bytes"].notna() & (df_io["d_storaged_bytes"] > 0)]
    if len(sub2) >= 5:
        x = sub2["d_storaged_bytes"] / (1024*1024)
        y = sub2["d_flash_C"]
        ax.scatter(x, y, s=18, alpha=0.6, c="tab:orange")
        try:
            rho, _ = spearmanr(x, y)
            ax.set_title(f"04b — Δstoraged_bytes_written vs Δflash_temp  (ρ={rho:+.2f}, n={len(sub2)})")
        except Exception:
            ax.set_title("04b — Δstoraged vs Δflash_temp")
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5,
                "storaged delta not in run.json yet\n(captured starting this run)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("04b — Δstoraged (system-wide UFS writes) vs Δflash_temp")
    ax.set_xlabel("Δstoraged_bytes_written per prompt (MiB)")
    ax.set_ylabel("Δflash_temp per prompt (°C)")
    ax.grid(alpha=0.3)

    fig.suptitle("04 — I/O activity drives storage thermal change (cross-prompt)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "04_io_to_heat_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 04_io_to_heat_scatter.png")

    # ===============================================================
    # 05 — Memory pressure vs DDR thermal change
    # ===============================================================
    rows = []
    for p in prompts:
        sen = p["sen"]
        ma = col(p, "mem_avail_kb")
        ddr = col(p, "ddr_temp_mc")
        mt = col(p, "mem_total_kb")
        if ma is None or ddr is None or mt is None:
            continue
        if ma.dropna().empty or ddr.dropna().empty or mt.dropna().empty:
            continue
        # working-set drop = (start MemAvail) - (min MemAvail) in MiB
        ws_drop_MiB = float(ma.iloc[0] - ma.min()) / 1024.0
        d_ddr = float(ddr.max() - ddr.iloc[0]) / 1000.0
        rows.append({"task": p["task"], "ws_drop_MiB": ws_drop_MiB, "d_ddr_C": d_ddr})
    df_mem = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(df_mem) >= 5:
        ax.scatter(df_mem["ws_drop_MiB"], df_mem["d_ddr_C"], s=20, alpha=0.6, c="tab:blue")
        try:
            rho, _ = spearmanr(df_mem["ws_drop_MiB"], df_mem["d_ddr_C"])
            ax.set_title(f"05 — working-set drop vs DDR thermal change (ρ={rho:+.2f}, n={len(df_mem)})")
        except Exception:
            ax.set_title("05 — working-set drop vs DDR thermal change")
    ax.set_xlabel("working-set drop per prompt (MiB)  = MemAvail(start) - min MemAvail")
    ax.set_ylabel("DDR temp rise (°C)  = max(ddr_temp) - ddr_temp(start)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "05_ram_pressure_to_heat.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 05_ram_pressure_to_heat.png")

    print(f"\n[plot] all figures in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
