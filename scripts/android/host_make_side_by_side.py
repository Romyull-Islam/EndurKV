#!/usr/bin/env python3
"""
host_make_side_by_side.py — render every per-model plot in a 1B-left | 8B-right
layout for direct visual comparison.

Outputs to logs/sidebyside_figures/:
    01_thermal_trace_1B_vs_8B.png
    02_thermal_aggregate_1B_vs_8B.png
    03_kv_cache_growth_1B_vs_8B.png
    04_decode_latency_1B_vs_8B.png
    05_entropy_vs_attention_1B_vs_8B.png
    06_rho_per_task_1B_vs_8B.png
    07_memory_trajectory_1B_vs_8B.png
"""
import os
import sys
import json
import struct
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", r"D:/Research/EndurKV_workspace"))
OUT = WORKSPACE / "logs" / "sidebyside_figures"
OUT.mkdir(parents=True, exist_ok=True)

TRIP_PATTERNS = ("cpu-hw-trip", "_trip_", "bcl-lvl", "ibat-lvl", "vbat",
                 "pmh", "pmr", "pmih010", "wireless", "usb", "sdr0")
def is_trip(c): return any(p in c.lower() for p in TRIP_PATTERNS)


def parse_attn_top1(path: Path):
    if not path.exists():
        return [], []
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"ATTN": return [], []
            n_steps, n_layers, _ = struct.unpack("<III", f.read(12))
            tops, nkvs = [], []
            for _ in range(n_steps):
                tp = []; last_nkv = 0
                for _ in range(n_layers):
                    (nk,) = struct.unpack("<I", f.read(4))
                    if nk == 0: continue
                    last_nkv = nk
                    v = struct.unpack(f"<{nk}f", f.read(4 * nk))
                    if v: tp.append(max(v))
                tops.append(sum(tp)/len(tp) if tp else float("nan"))
                nkvs.append(last_nkv)
            return tops, nkvs
    except Exception:
        return [], []


def load_run(log_dir: Path):
    """Return list of {id, task, ent (df), sen (df), run (dict)} for every prompt."""
    import pandas as pd
    prompts = []
    for ent_p in sorted(log_dir.glob("*.entropy.csv")):
        pid = ent_p.name[:-len(".entropy.csv")]
        sen_p = log_dir / f"{pid}.sensors.csv"
        run_p = log_dir / f"{pid}.run.json"
        if not sen_p.exists() or not run_p.exists(): continue
        try:
            ent = pd.read_csv(ent_p); sen = pd.read_csv(sen_p)
            run = json.loads(run_p.read_text())
        except Exception: continue
        if ent.empty: continue
        ent["step_wall_s"] = run["start_wall_s"] + ent["wall_clock_us"]/1e6
        tops, nkvs = parse_attn_top1(log_dir / f"{pid}.attn.bin")
        if tops:
            n = min(len(tops), len(ent))
            ent = ent.iloc[:n].copy()
            ent["attn_top1_layer_avg"] = tops[:n]
            ent["n_kv"] = nkvs[:n]
        task = pid.rsplit("_",1)[0] if "_" in pid else "?"
        prompts.append({"id": pid, "task": task, "ent": ent, "sen": sen, "run": run})
    return prompts


def family_of(col: str):
    low = col.lower()
    for fam in ("cpu", "gpuss", "nsphvx", "nsphmx", "qmx", "ddr"):
        if fam in low: return fam
    if any(k in low for k in ("shell","board","skin")): return "skin"
    return "other"

FAM_COLORS = {"cpu":"tab:red","gpuss":"tab:blue","skin":"tab:green",
              "nsphvx":"tab:orange","nsphmx":"tab:purple","qmx":"tab:olive",
              "ddr":"tab:cyan","other":"tab:gray"}


def main():
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9})

    log_1b = WORKSPACE / "logs" / "study_phone_1b"
    log_8b = WORKSPACE / "logs" / "study_phone_8b"
    print(f"[load] 1B from {log_1b}")
    p1 = load_run(log_1b)
    print(f"[load] 8B from {log_8b}")
    p8 = load_run(log_8b)
    print(f"[load] 1B prompts: {len(p1)}, 8B prompts: {len(p8)}")

    # ------------------------------------------------------------------
    # 01 — thermal trace, same 4 representative prompts for each model
    # ------------------------------------------------------------------
    pick = ("gov_report_001", "qasper_001", "samsum_001", "piqa_001")
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=False)
    for col_idx, (model_prompts, model_name) in enumerate(
            [(p1, "1B Q4_K_M"), (p8, "8B Q4_K_M")]):
        by_id = {p["id"]: p for p in model_prompts}
        for row, pid in enumerate(pick):
            ax = axes[row, col_idx]
            if pid not in by_id:
                ax.text(0.5, 0.5, f"{pid}\n(not in {model_name})",
                        ha="center", va="center", transform=ax.transAxes)
                continue
            p = by_id[pid]
            sen = p["sen"].sort_values("wall_clock_s").copy()
            zone_cols = [c for c in sen.columns if c.endswith("_temp_mc") and not is_trip(c)]
            if not zone_cols: continue
            t0 = float(sen["wall_clock_s"].iloc[0])
            sen["t_rel"] = sen["wall_clock_s"] - t0
            fams = {}
            for c in zone_cols:
                fams.setdefault(family_of(c), []).append(c)
            for fam, cols in fams.items():
                sub = sen[cols].apply(pd.to_numeric, errors="coerce") / 1000.0
                if sub.empty: continue
                mx = sub.max(axis=1)
                mn = sub.mean(axis=1)
                ax.plot(sen["t_rel"], mx, lw=1.2, color=FAM_COLORS.get(fam, "gray"),
                        label=f"{fam} max", alpha=0.9)
                ax.fill_between(sen["t_rel"], mn, mx,
                                color=FAM_COLORS.get(fam, "gray"), alpha=0.10)
            ax.axhline(70.0, color="red", ls="--", lw=0.8, alpha=0.6)
            ax.set_title(f"{model_name}  ·  {pid}  ·  n_steps={len(p['ent'])}", fontsize=10)
            if col_idx == 0: ax.set_ylabel("temp (°C)")
            ax.set_xlabel("time since start (s)")
            ax.grid(alpha=0.3)
            ax.set_ylim(30, 80)
            if row == 0 and col_idx == 1:
                ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.suptitle("01 — Thermal trace per zone family · 1B (left) vs 8B (right)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "01_thermal_trace_1B_vs_8B.png", bbox_inches="tight")
    plt.close(fig); print("  wrote 01_thermal_trace_1B_vs_8B.png")

    # ------------------------------------------------------------------
    # 02 — peak active-zone temp per task — paired boxplot
    # ------------------------------------------------------------------
    def per_prompt_peak(prompts):
        out = []
        for p in prompts:
            zcols = [c for c in p["sen"].columns if c.endswith("_temp_mc") and not is_trip(c)]
            if not zcols: continue
            v = p["sen"][zcols].apply(pd.to_numeric, errors="coerce")
            out.append({"task": p["task"], "max_C": float(v.max().max())/1000.0,
                        "dT_C": (float(v.iloc[-1].max()) - float(v.iloc[0].max()))/1000.0})
        return pd.DataFrame(out)
    pp1, pp8 = per_prompt_peak(p1), per_prompt_peak(p8)
    common = sorted(set(pp1.task) & set(pp8.task))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, df, name, color in [
            (axes[0], pp1, "1B Q4_K_M", "#cfe1f5"),
            (axes[1], pp8, "8B Q4_K_M", "#fbd9bf")]:
        data = [df[df.task == t]["max_C"].values for t in common]
        ax.boxplot(data, tick_labels=common, showmeans=True, patch_artist=True,
                   boxprops=dict(facecolor=color))
        ax.axhline(70.0, color="red", ls="--", lw=1.0, label="proposal trip 70°C")
        ax.set_title(f"{name} — peak active-zone temp / prompt")
        ax.set_ylabel("°C")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(40, 80)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("02 — Peak temperature per task · 1B (left) vs 8B (right)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "02_thermal_aggregate_1B_vs_8B.png", bbox_inches="tight")
    plt.close(fig); print("  wrote 02_thermal_aggregate_1B_vs_8B.png")

    # ------------------------------------------------------------------
    # 03 — KV cache size (n_kv at decode step 0) across prompts
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, prompts, name, color in [
            (axes[0], p1, "1B Q4_K_M", "tab:blue"),
            (axes[1], p8, "8B Q4_K_M", "tab:orange")]:
        for p in prompts:
            if "n_kv" not in p["ent"].columns: continue
            ax.plot(p["ent"]["step_index"], p["ent"]["n_kv"],
                    alpha=0.18, lw=0.7, color=color)
        ax.set_xlabel("decode step")
        ax.set_ylabel("n_kv  (source positions in KV cache)")
        ax.set_title(f"{name} — KV cache size across {len(prompts)} prompts")
        ax.grid(alpha=0.3)
    fig.suptitle("03 — KV cache size · 1B (left) vs 8B (right)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "03_kv_cache_growth_1B_vs_8B.png", bbox_inches="tight")
    plt.close(fig); print("  wrote 03_kv_cache_growth_1B_vs_8B.png")

    # ------------------------------------------------------------------
    # 04 — decode latency per-task boxplot
    # ------------------------------------------------------------------
    def latency_table(prompts):
        rows = []
        for p in prompts:
            e = p["ent"].sort_values("step_index")
            dt = e["wall_clock_us"].diff().iloc[1:].values / 1000.0
            for v in dt:
                if 1.0 <= v <= 600.0:
                    rows.append({"task": p["task"], "dt_ms": v})
        return pd.DataFrame(rows)
    L1, L8 = latency_table(p1), latency_table(p8)
    common = sorted(set(L1.task) & set(L8.task))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, df, name, color in [
            (axes[0], L1, "1B Q4_K_M", "#cfe1f5"),
            (axes[1], L8, "8B Q4_K_M", "#fbd9bf")]:
        data = [df[df.task == t]["dt_ms"].values for t in common]
        ax.boxplot(data, tick_labels=common, showmeans=True, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor=color))
        ax.set_ylabel("per-step decode latency (ms)")
        ax.set_title(f"{name} — decode latency per task")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("04 — Decode latency per task · 1B (left) vs 8B (right)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "04_decode_latency_1B_vs_8B.png", bbox_inches="tight")
    plt.close(fig); print("  wrote 04_decode_latency_1B_vs_8B.png")

    # ------------------------------------------------------------------
    # 05 — entropy vs attention scatter, both models in matched scale
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for ax, prompts, name, color in [
            (axes[0], p1, "1B Q4_K_M", "#4a90e2"),
            (axes[1], p8, "8B Q4_K_M", "#e07c3a")]:
        rows = []
        for p in prompts:
            if "attn_top1_layer_avg" not in p["ent"].columns: continue
            rows.append(p["ent"][["H_nats","attn_top1_layer_avg"]].dropna())
        if not rows: continue
        df = pd.concat(rows, ignore_index=True)
        if len(df) < 20: continue
        r, pv = spearmanr(df["H_nats"], df["attn_top1_layer_avg"])
        ax.scatter(df["H_nats"], df["attn_top1_layer_avg"],
                   s=4, color=color, alpha=0.25, label=f"all decode steps (n={len(df)})")
        try:
            bins = pd.qcut(df["H_nats"], q=10, duplicates="drop")
            mu = df.groupby(bins, observed=True)["attn_top1_layer_avg"].mean()
            centres = [iv.mid for iv in mu.index]
            ax.plot(centres, mu.values, "o-", color="black", lw=1.4,
                    label="per-bin mean (10 H-quantiles)")
        except Exception:
            pass
        ax.set_xlabel("output entropy H (nats)")
        ax.set_ylabel("max attention probability (layer-averaged)")
        ax.set_title(f"{name} — ρ = {r:+.3f}  (n = {len(df)})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("05 — Entropy ↔ attention scatter · 1B (left) vs 8B (right)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "05_entropy_vs_attention_1B_vs_8B.png", bbox_inches="tight")
    plt.close(fig); print("  wrote 05_entropy_vs_attention_1B_vs_8B.png")

    # ------------------------------------------------------------------
    # 06 — per-task rho bars
    # ------------------------------------------------------------------
    def per_task_rho(prompts):
        rows = []
        groups = {}
        for p in prompts:
            if "attn_top1_layer_avg" not in p["ent"].columns: continue
            groups.setdefault(p["task"], []).append(
                p["ent"][["H_nats","attn_top1_layer_avg"]].dropna())
        for t, gs in groups.items():
            df = pd.concat(gs, ignore_index=True)
            if len(df) >= 10:
                r, pv = spearmanr(df["H_nats"], df["attn_top1_layer_avg"])
                rows.append({"task": t, "rho": float(r), "n": len(df)})
        return pd.DataFrame(rows).sort_values("rho")
    r1, r8 = per_task_rho(p1), per_task_rho(p8)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, df, name in [(axes[0], r1, "1B Q4_K_M"),
                         (axes[1], r8, "8B Q4_K_M")]:
        ys = list(range(len(df)))
        colors = ["#1b6e1b" if r<=-0.40 else "#5fa55f" if r<=-0.20
                  else "#cccccc" if r<0 else "#c33" for r in df["rho"]]
        ax.barh(ys, df["rho"], color=colors)
        ax.set_yticks(ys); ax.set_yticklabels(df["task"])
        ax.axvline(-0.20, color="green", ls="--", lw=1.2, label="fallback gate (−0.20)")
        ax.axvline(-0.37, color="black", ls=":",  lw=1.0, label="paper headline (−0.37)")
        ax.axvspan(-0.7, -0.20, color="green", alpha=0.06)
        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(row["rho"]-0.01 if row["rho"]<0 else row["rho"]+0.01, i,
                    f"{row['rho']:+.2f}",
                    va="center", ha="right" if row["rho"]<0 else "left", fontsize=8)
        ax.set_xlim(-0.65, 0.4)
        ax.set_xlabel("Spearman ρ (H ↔ max attention)")
        ax.set_title(f"{name} — per-task ρ")
        ax.grid(alpha=0.3, axis="x")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("06 — Per-task ρ · 1B (left) vs 8B (right)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "06_rho_per_task_1B_vs_8B.png", bbox_inches="tight")
    plt.close(fig); print("  wrote 06_rho_per_task_1B_vs_8B.png")

    # ------------------------------------------------------------------
    # 07 — MemAvailable timeline + Δpswpout boxplot per task
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for col_idx, (prompts, name, color) in enumerate([
            (p1, "1B Q4_K_M", "tab:blue"),
            (p8, "8B Q4_K_M", "tab:orange")]):
        ax = axes[0, col_idx]
        for p in prompts:
            sen = p["sen"]
            if "mem_avail_kb" not in sen.columns or sen.empty: continue
            sen = sen.sort_values("wall_clock_s")
            t0 = float(sen["wall_clock_s"].iloc[0])
            sen = sen.copy(); sen["t_rel"] = sen["wall_clock_s"] - t0
            ma = pd.to_numeric(sen["mem_avail_kb"], errors="coerce") / 1024.0
            ax.plot(sen["t_rel"], ma, color=color, alpha=0.2, lw=0.7)
        ax.set_xlabel("time since start (s)")
        ax.set_ylabel("MemAvailable (MiB)")
        ax.set_title(f"{name} — memory headroom across prompts")
        ax.grid(alpha=0.3)

        ax = axes[1, col_idx]
        rows = []
        for p in prompts:
            sen = p["sen"]
            if "vmstat_pswpout" not in sen.columns: continue
            s = pd.to_numeric(sen["vmstat_pswpout"], errors="coerce").dropna()
            if len(s) >= 2:
                d = max(int(s.iloc[-1] - s.iloc[0]), 0) * 4  # × 4 KB
                rows.append({"task": p["task"], "kib": d})
        if not rows:
            ax.text(0.5, 0.5, "no pswpout data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        df = pd.DataFrame(rows)
        common = sorted(df.task.unique())
        data = [np.clip(df[df.task == t]["kib"].values, 0.5, None) for t in common]
        ax.boxplot(data, tick_labels=common, showmeans=True, patch_artist=True,
                   boxprops=dict(facecolor="#cfe1f5" if col_idx==0 else "#fbd9bf"))
        ax.set_yscale("log")
        ax.set_ylabel("Δpswpout per prompt (KiB, log)")
        ax.set_title(f"{name} — UFS swap activity per task")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.3, axis="y", which="both")
    fig.suptitle("07 — Memory + Endurance · 1B (left) vs 8B (right)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "07_memory_trajectory_1B_vs_8B.png", bbox_inches="tight")
    plt.close(fig); print("  wrote 07_memory_trajectory_1B_vs_8B.png")

    print(f"\nall side-by-side figures in {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)
