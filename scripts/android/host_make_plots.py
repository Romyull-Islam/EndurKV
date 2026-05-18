#!/usr/bin/env python3
"""
host_make_plots.py — generate PNG plots from a completed phone study.

Inputs (per prompt_id in --log-dir):
  <id>.entropy.csv   (per-step probe metrics)
  <id>.attn.bin      (per-step per-layer attention, optional)
  <id>.sensors.csv   (per-sample thermal + memory + endurance)
  <id>.run.json      (wall-clock join keys)

Outputs (in --out-dir, default <log_dir>_figures/):
  01_thermal_trace.png         — multi-zone temp(t) for 4 representative prompts
  02_thermal_aggregate.png     — per-task max-active-zone warming (boxplot)
  03_kv_cache_growth.png       — n_kv vs step_index across prompts
  04_decode_latency.png        — ms/decode-step over the run
  05_entropy_vs_attention.png  — slide-22-style scatter + quantile binning
  06_rho_per_task.png          — slide-23-style per-task Spearman bar chart
  07_memory_trajectory.png     — MemAvailable + pswpout delta over time

Usage:
  python scripts/android/host_make_plots.py [--log-dir logs/study_phone_1b]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
WORKSPACE = Path(os.environ.get("WORKSPACE", HERE.parents[3]))


# -------------- attn.bin parser (also in host_join_and_rho.py) --------------
def parse_attn_bin_top1(path: Path) -> tuple[list[float], list[int]]:
    """Return (per_step_layer_avg_top1, per_step_n_kv)."""
    if not path.exists():
        return [], []
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"ATTN":
                return [], []
            n_steps, n_layers, _ = struct.unpack("<III", f.read(12))
            tops, nkvs = [], []
            for _s in range(n_steps):
                top1_per_layer = []
                last_nkv = 0
                for _l in range(n_layers):
                    (n_kv,) = struct.unpack("<I", f.read(4))
                    if n_kv == 0:
                        continue
                    last_nkv = n_kv
                    vals = struct.unpack(f"<{n_kv}f", f.read(4 * n_kv))
                    if vals:
                        top1_per_layer.append(max(vals))
                if top1_per_layer:
                    tops.append(sum(top1_per_layer) / len(top1_per_layer))
                else:
                    tops.append(float("nan"))
                nkvs.append(last_nkv)
            return tops, nkvs
    except Exception as e:
        print(f"  WARN: parse_attn_bin({path}) failed: {e}")
        return [], []


TRIP_PATTERNS = ("cpu-hw-trip", "_trip_", "bcl-lvl", "ibat-lvl", "vbat",
                 "pmh", "pmr", "pmih010", "wireless", "usb", "sdr0")


def is_trip_zone(col: str) -> bool:
    low = col.lower()
    return any(p in low for p in TRIP_PATTERNS)


# -------------- main --------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=str(WORKSPACE / "logs" / "study_phone_1b"))
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir.parent / f"{log_dir.name}_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not log_dir.exists():
        print(f"ERROR: {log_dir} does not exist", file=sys.stderr)
        return 1

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9})

    # --------- load every prompt --------------------------------------------
    prompts = []  # list of dicts: id, task, ent (df), sen (df), run (dict), attn_top1 (list), nkv (list)
    for ent_path in sorted(log_dir.glob("*.entropy.csv")):
        pid = ent_path.name[: -len(".entropy.csv")]
        sen_path  = log_dir / f"{pid}.sensors.csv"
        run_path  = log_dir / f"{pid}.run.json"
        attn_path = log_dir / f"{pid}.attn.bin"
        if not sen_path.exists() or not run_path.exists():
            continue
        try:
            ent = pd.read_csv(ent_path)
            sen = pd.read_csv(sen_path)
            run = json.loads(run_path.read_text())
        except Exception:
            continue
        if ent.empty:
            continue
        ent["step_wall_s"] = run["start_wall_s"] + ent["wall_clock_us"] / 1e6
        tops, nkvs = parse_attn_bin_top1(attn_path)
        if tops:
            n = min(len(tops), len(ent))
            ent = ent.iloc[:n].copy()
            ent["attn_top1_layer_avg"] = tops[:n]
            ent["n_kv"] = nkvs[:n]
        # task is parsed from prompt_id prefix (e.g. "gov_report_001" -> "gov_report")
        task = "_".join(pid.split("_")[:-1]) if "_" in pid else "smoke"
        prompts.append({"id": pid, "task": task, "ent": ent, "sen": sen, "run": run})

    if not prompts:
        print("ERROR: nothing to plot", file=sys.stderr)
        return 1
    print(f"[plot] loaded {len(prompts)} prompts from {log_dir}")
    print(f"[plot] tasks: {sorted({p['task'] for p in prompts})}")
    print(f"[plot] out:    {out_dir}")

    # task ordering for paper-comparable plots
    PAPER_TASK_ORDER = [
        "gov_report", "multi_news", "cnn_dailymail", "xsum",
        "qasper", "multifieldqa_en", "hotpotqa", "narrativeqa",
        "qmsum", "samsum", "triviaqa", "trec", "lcc",
        "piqa", "openbookqa", "smoke",
    ]
    def task_key(t): return (PAPER_TASK_ORDER.index(t) if t in PAPER_TASK_ORDER else 99, t)

    # ========================================================================
    # 01_thermal_trace.png — multi-zone temp(t) for 4 representative prompts
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=False)
    # pick one from each end of the task spectrum, by id, robust to missing
    pick_ids = []
    for task in ("gov_report", "qasper", "samsum", "piqa"):
        match = [p for p in prompts if p["task"] == task]
        if match: pick_ids.append(match[0])
    while len(pick_ids) < 4 and len(prompts) > len(pick_ids):
        pick_ids.append(prompts[len(pick_ids)])

    for ax, p in zip(axes.flat, pick_ids):
        sen = p["sen"].copy()
        zone_cols = [c for c in sen.columns if c.endswith("_temp_mc") and not is_trip_zone(c)]
        if not zone_cols:
            ax.set_title(f"{p['id']}  (no zones)")
            continue
        sen = sen.sort_values("wall_clock_s")
        t0 = float(sen["wall_clock_s"].iloc[0])
        sen["t_rel"] = sen["wall_clock_s"] - t0
        # group zones by family for colouring
        families = {"cpu": [], "gpuss": [], "skin": [], "nsphvx": [], "nsphmx": [], "qmx": [], "ddr": [], "other": []}
        for c in zone_cols:
            placed = False
            for fam in families:
                if fam in c.lower():
                    families[fam].append(c); placed = True; break
            if not placed:
                if "shell" in c or "board" in c or "skin" in c:
                    families["skin"].append(c)
                else:
                    families["other"].append(c)
        colour = {"cpu": "tab:red", "gpuss": "tab:blue", "skin": "tab:green",
                  "nsphvx": "tab:orange", "nsphmx": "tab:purple", "qmx": "tab:olive",
                  "ddr": "tab:cyan", "other": "tab:gray"}
        for fam, cols in families.items():
            if not cols: continue
            sub = sen[cols].apply(pd.to_numeric, errors="coerce") / 1000.0
            mean_t = sub.mean(axis=1)
            max_t  = sub.max(axis=1)
            ax.plot(sen["t_rel"], max_t, color=colour[fam], lw=1.2, label=f"{fam} max", alpha=0.9)
            ax.fill_between(sen["t_rel"], mean_t, max_t, color=colour[fam], alpha=0.10)
        ax.axhline(70.0, color="red", ls="--", lw=0.8, alpha=0.6, label="proposal trip 70°C")
        ax.set_title(f"{p['id']}  (n_steps={len(p['ent'])}, dur={sen['t_rel'].iloc[-1]:.1f}s)")
        ax.set_xlabel("time since prompt start (s)")
        ax.set_ylabel("temperature (°C)")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-zone thermal trajectory — selected prompts", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "01_thermal_trace.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 01_thermal_trace.png")

    # ========================================================================
    # 02_thermal_aggregate.png — warming per task (boxplot of dT)
    # ========================================================================
    warmings = []  # list of (task, dT_C)
    max_active = []  # list of (task, max active zone temp)
    for p in prompts:
        sen = p["sen"]
        zone_cols = [c for c in sen.columns if c.endswith("_temp_mc") and not is_trip_zone(c)]
        if not zone_cols or len(sen) < 2: continue
        active = sen[zone_cols].apply(pd.to_numeric, errors="coerce") / 1000.0
        try:
            dT = float(active.iloc[-1].max() - active.iloc[0].max())
            mx = float(active.max().max())
            warmings.append((p["task"], dT))
            max_active.append((p["task"], mx))
        except Exception:
            continue
    if warmings:
        df = pd.DataFrame(warmings, columns=["task", "dT_C"])
        df2 = pd.DataFrame(max_active, columns=["task", "maxT_C"])
        order = sorted(df["task"].unique(), key=task_key)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        ax = axes[0]
        data = [df[df.task == t]["dT_C"].values for t in order]
        ax.boxplot(data, labels=order, showmeans=True)
        ax.set_ylabel("warming over prompt (°C)")
        ax.set_title("Per-prompt warming (max active zone end - start)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.3, axis="y")
        ax = axes[1]
        data2 = [df2[df2.task == t]["maxT_C"].values for t in order]
        ax.boxplot(data2, labels=order, showmeans=True)
        ax.axhline(70.0, color="red", ls="--", lw=0.8, alpha=0.6, label="proposal trip 70°C")
        ax.set_ylabel("max active zone temp during prompt (°C)")
        ax.set_title("Peak active temperature per prompt")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "02_thermal_aggregate.png", bbox_inches="tight")
        plt.close(fig)
        print("  wrote 02_thermal_aggregate.png")

    # ========================================================================
    # 03_kv_cache_growth.png — n_kv vs step across prompts (and a few samples)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for p in prompts:
        if "n_kv" not in p["ent"].columns: continue
        ax.plot(p["ent"]["step_index"], p["ent"]["n_kv"], alpha=0.18, lw=0.7, color="tab:blue")
    ax.set_xlabel("decode step index")
    ax.set_ylabel("KV cache size (n_kv, source positions)")
    ax.set_title(f"KV cache growth across {len(prompts)} prompts")
    ax.grid(alpha=0.3)
    # right panel: sample 3 prompts of very different prompt lengths
    ax = axes[1]
    samples = []
    if "n_kv" in prompts[0]["ent"].columns:
        sorted_by_len = sorted(
            [p for p in prompts if "n_kv" in p["ent"].columns and len(p["ent"]) > 5],
            key=lambda p: -int(p["ent"]["n_kv"].iloc[0])  # initial n_kv = prompt length
        )
        n = len(sorted_by_len)
        if n >= 3:
            samples = [sorted_by_len[0], sorted_by_len[n // 2], sorted_by_len[-1]]
        else:
            samples = sorted_by_len
    for p in samples:
        ax.plot(p["ent"]["step_index"], p["ent"]["n_kv"], lw=1.5,
                label=f"{p['id']} (start n_kv={int(p['ent']['n_kv'].iloc[0])})")
    ax.set_xlabel("decode step index")
    ax.set_ylabel("n_kv")
    ax.set_title("Three prompts: longest / median / shortest")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "03_kv_cache_growth.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 03_kv_cache_growth.png")

    # ========================================================================
    # 04_decode_latency.png — ms per decode step, distribution + over-time
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    all_dt_ms = []
    for p in prompts:
        e = p["ent"].copy()
        if len(e) < 2: continue
        e = e.sort_values("step_index")
        dt = e["wall_clock_us"].diff().iloc[1:].values / 1000.0
        all_dt_ms.append((p["task"], dt))
    order = sorted({t for t, _ in all_dt_ms}, key=task_key)
    data = [np.concatenate([dt for t, dt in all_dt_ms if t == task]) if any(t == task for t, _ in all_dt_ms) else np.array([])
            for task in order]
    ax.boxplot([d for d in data if len(d) > 0],
               labels=[t for t, d in zip(order, data) if len(d) > 0],
               showfliers=False, showmeans=True)
    ax.set_ylabel("per-step decode latency (ms)")
    ax.set_title("Per-task decode-step latency (boxplot, outliers off)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    # time series: concatenate all prompts in arrival order, color by task
    cumulative_step = 0
    tasks_seen = []
    for p in prompts:
        e = p["ent"].sort_values("step_index")
        if len(e) < 2: continue
        dt_ms = e["wall_clock_us"].diff().iloc[1:].values / 1000.0
        xs = np.arange(cumulative_step, cumulative_step + len(dt_ms))
        col = "C{}".format(PAPER_TASK_ORDER.index(p["task"]) % 10) if p["task"] in PAPER_TASK_ORDER else "gray"
        ax.scatter(xs, dt_ms, s=2, color=col, alpha=0.5)
        if p["task"] not in tasks_seen:
            tasks_seen.append(p["task"])
        cumulative_step += len(dt_ms)
    ax.set_xlabel("global decode step (concatenated across prompts)")
    ax.set_ylabel("per-step latency (ms)")
    ax.set_title(f"Decode latency over full run — {len(prompts)} prompts")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "04_decode_latency.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 04_decode_latency.png")

    # ========================================================================
    # 05_entropy_vs_attention.png — slide-22 LEFT panel style
    # ========================================================================
    have_attn = any("attn_top1_layer_avg" in p["ent"].columns for p in prompts)
    if have_attn:
        all_e = pd.concat([p["ent"][p["ent"].columns.intersection(
            ["H_nats", "top1_prob", "attn_top1_layer_avg"])] for p in prompts
            if "attn_top1_layer_avg" in p["ent"].columns], ignore_index=True).dropna()
        if len(all_e) >= 20:
            rho, p_val = spearmanr(all_e["H_nats"], all_e["attn_top1_layer_avg"])
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(all_e["H_nats"], all_e["attn_top1_layer_avg"],
                       s=8, color="gray", alpha=0.35, label=f"all decode steps (n={len(all_e)})")
            # quantile binning
            bins = pd.qcut(all_e["H_nats"], q=10, duplicates="drop")
            grouped = all_e.groupby(bins, observed=True)["attn_top1_layer_avg"].agg(["mean", "quantile"])
            centres = [iv.mid for iv in grouped.index]
            means   = grouped["mean"].values
            ax.plot(centres, means, "o-", color="tab:blue", lw=2, label="per-bin mean (10 H-quantiles)")
            # safe/danger shading
            try:
                h_lo = float(all_e["H_nats"].quantile(0.20))
                h_hi = float(all_e["H_nats"].quantile(0.80))
                ax.axvspan(all_e["H_nats"].min(), h_lo, color="green", alpha=0.10, label="low-H (safe to prune)")
                ax.axvspan(h_hi, all_e["H_nats"].max(), color="red",   alpha=0.10, label="high-H (do not prune)")
            except Exception:
                pass
            ax.set_xlabel("output entropy H (nats)")
            ax.set_ylabel("max attention probability (layer-averaged)")
            ax.set_title(f"Entropy vs attention concentration on OnePlus 15\n"
                         f"Spearman ρ = {rho:+.3f}, n = {len(all_e)}, p = {p_val:.2e}")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "05_entropy_vs_attention.png", bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote 05_entropy_vs_attention.png  (pooled ρ={rho:+.3f})")

    # ========================================================================
    # 06_rho_per_task.png — slide-23 RIGHT panel style
    # ========================================================================
    if have_attn:
        rows = []
        for task, grp_prompts in pd.DataFrame(
                [{"task": p["task"], "p": p["id"]} for p in prompts]
            ).groupby("task"):
            ents = [p["ent"] for p in prompts if p["task"] == task and "attn_top1_layer_avg" in p["ent"].columns]
            if not ents: continue
            df_t = pd.concat(ents, ignore_index=True).dropna(subset=["H_nats", "attn_top1_layer_avg"])
            if len(df_t) < 10: continue
            r, pp = spearmanr(df_t["H_nats"], df_t["attn_top1_layer_avg"])
            rows.append((task, r, pp, len(df_t)))
        if rows:
            rows.sort(key=lambda r: r[1])  # most negative first (bottom)
            fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(rows) + 2)))
            tasks = [r[0] for r in rows]
            rhos  = [r[1] for r in rows]
            ns    = [r[3] for r in rows]
            colors = []
            for r in rhos:
                if r <= -0.40: colors.append("#1b6e1b")
                elif r <= -0.20: colors.append("#5fa55f")
                elif r <  0:     colors.append("#cccccc")
                else:            colors.append("#c33")
            ax.barh(tasks, rhos, color=colors)
            ax.axvspan(-1.0, -0.20, color="green", alpha=0.07)
            ax.axvline(-0.20, color="green", ls="--", lw=1.2, label="fallback gate ρ ≤ −0.20")
            ax.axvline(-0.37, color="black", ls=":",  lw=1.0, label="paper headline ρ = −0.37")
            ax.set_xlabel("Spearman ρ (output entropy ↔ max attention)")
            ax.set_title(f"Per-task ρ on OnePlus 15 (1B Q4_K_M short-ctx)")
            for i, (t, r, _, n) in enumerate(rows):
                ax.text(r - 0.01 if r < 0 else r + 0.01, i, f"ρ={r:+.2f} n={n}",
                        va="center", ha="right" if r < 0 else "left", fontsize=8)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3, axis="x")
            ax.set_xlim(min(-0.6, min(rhos) - 0.05), max(0.1, max(rhos) + 0.05))
            fig.tight_layout()
            fig.savefig(out_dir / "06_rho_per_task.png", bbox_inches="tight")
            plt.close(fig)
            print("  wrote 06_rho_per_task.png")

    # ========================================================================
    # 07_memory_trajectory.png — MemAvailable + Δpswpout over time per prompt
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for p in prompts:
        sen = p["sen"].copy()
        if "mem_avail_kb" not in sen.columns or sen.empty: continue
        sen = sen.sort_values("wall_clock_s")
        t0 = float(sen["wall_clock_s"].iloc[0])
        sen["t_rel"] = sen["wall_clock_s"] - t0
        ma = pd.to_numeric(sen["mem_avail_kb"], errors="coerce") / 1024.0  # MB
        ax.plot(sen["t_rel"], ma, alpha=0.2, color="tab:blue", lw=0.8)
    ax.set_xlabel("time since prompt start (s)")
    ax.set_ylabel("MemAvailable (MiB)")
    ax.set_title(f"Memory headroom across {len(prompts)} prompts")
    ax.grid(alpha=0.3)
    ax = axes[1]
    deltas = []
    for p in prompts:
        sen = p["sen"]
        if "vmstat_pswpout" not in sen.columns: continue
        ps = pd.to_numeric(sen["vmstat_pswpout"], errors="coerce").dropna()
        if len(ps) >= 2:
            delta_pages = int(ps.iloc[-1] - ps.iloc[0])
            deltas.append((p["task"], delta_pages))
    if deltas:
        df = pd.DataFrame(deltas, columns=["task", "pswpout_delta"])
        order = sorted(df["task"].unique(), key=task_key)
        data = [df[df.task == t]["pswpout_delta"].values * 4 for t in order]  # ×4 KB → KiB
        ax.boxplot(data, labels=order, showmeans=True, showfliers=True)
        ax.set_ylabel("Δpswpout per prompt (KiB swap-out)")
        ax.set_title("Per-prompt UFS swap activity (4 KB pages)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "07_memory_trajectory.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 07_memory_trajectory.png")

    print(f"\n[plot] all figures in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
