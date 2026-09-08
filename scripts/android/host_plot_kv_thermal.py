#!/usr/bin/env python3
"""
Plot the relationship between KV cache size, memory utilization, DRAM heat,
and SoC (CPU) heat across all measured cells.

Pulls per-cell aggregates from sensors.csv + stress.csv across:
  - Wave-3 Phi-3 narrativeqa (5 policies)
  - Wave-3 Llama-1B (vanilla)
  - Wave-4 Phi-3 long-decode (3 policies)
  - Wave-5 Phi-3 narrativeqa v1_FA file-backed
  - Wave-6 Phi-3 long-decode v1_FA bounded (partial)

Output: figures/kv_thermal_relations.png
"""

import os
import sys
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PHONE_LOGS = "/home/mislam22/EndurKV_workspace/phone-logs"
OUT_FIG    = "/home/mislam22/EndurKV_workspace/EndurKV/figures/kv_thermal_relations.png"

CELLS = [
    # (label, host_dir, cell_subdir, marker, color, workload)
    ("L1B vanilla (narr)",        "wave3_real_1780680903",          "vanilla",                "o", "#888888", "narrativeqa"),
    ("Phi3 vanilla (narr)",       "wave3_phi3_1780719530",          "vanilla",                "o", "#1f77b4", "narrativeqa"),
    ("Phi3 v1 K=512 (narr)",      "wave3_phi3_1780719530",          "v1_K512",                "s", "#d62728", "narrativeqa"),
    ("Phi3 v1_FA K=512 (narr)",   "wave3_phi3_1780719530",          "v1_fa_K512",             "^", "#2ca02c", "narrativeqa"),
    ("Phi3 TOVA K=512 (narr)",    "wave3_phi3_1780719530",          "tova_K512",              "D", "#9467bd", "narrativeqa"),
    ("Phi3 vanilla (longdec)",    "wave4_longdecode_1780750084",    "vanilla",                "o", "#1f77b4", "long-decode"),
    ("Phi3 v1 K=512 (longdec)",   "wave4_longdecode_1780750084",    "v1_K512",                "s", "#d62728", "long-decode"),
    ("Phi3 v1_FA frozen (ld)",    "wave4_longdecode_1780750084",    "v1_fa_K512",             "^", "#2ca02c", "long-decode"),
    ("Phi3 v1_FA bounded (ld)",   None,                             None,                     "*", "#ff7f0e", "long-decode"),  # Wave-6 partial, filled below
]

def find_wave6():
    cand = sorted([d for d in os.listdir(PHONE_LOGS) if d.startswith("wave6_v1fa_bounded_")])
    return cand[-1] if cand else None

def col_idx(csv_path, name):
    with open(csv_path) as f:
        header = f.readline().rstrip().split(',')
    try:
        return header.index(name)
    except ValueError:
        return None

def aggregate_cell(host_dir, cell_subdir):
    """Return dict with peak_kv, peak_ddr, peak_cpu, peak_rss_gb, swap_mb, min_free_gb."""
    cell_root = os.path.join(PHONE_LOGS, host_dir, cell_subdir)
    # Some wave dirs are nested
    if not os.path.isdir(cell_root):
        nest = os.path.join(PHONE_LOGS, host_dir, host_dir, cell_subdir)
        if os.path.isdir(nest):
            cell_root = nest
        else:
            return None

    sensors = os.path.join(cell_root, "sensors.csv")
    stress  = os.path.join(cell_root, "stress.csv")
    if not (os.path.isfile(sensors) and os.path.isfile(stress)):
        return None

    # Stress: peak KV from any iter
    peak_kv = 0
    peak_rss_kb = 0
    with open(stress) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                pk = int(row.get('peak_kv_cells', 0) or 0)
                if pk > peak_kv: peak_kv = pk
                pr = int(row.get('peak_rss_kb', 0) or 0)
                if pr > peak_rss_kb: peak_rss_kb = pr
            except (ValueError, KeyError):
                pass

    # Sensors: peak DDR/CPU, swap-out delta, min free
    i_ddr = col_idx(sensors, 'ddr_temp_mc')
    i_cpu = col_idx(sensors, 'cpullc-0-0_temp_mc')
    i_psw = col_idx(sensors, 'vmstat_pswpout')
    i_mfr = col_idx(sensors, 'mem_avail_kb')

    if any(i is None for i in [i_ddr, i_cpu, i_psw, i_mfr]):
        return None

    peak_ddr_mc = 0
    peak_cpu_mc = 0
    min_free_kb = None
    psw_first = None
    psw_last = None
    with open(sensors) as f:
        f.readline()  # header
        for line in f:
            cols = line.split(',')
            try:
                ddr = int(cols[i_ddr]); cpu = int(cols[i_cpu])
                psw = int(cols[i_psw]); mfr = int(cols[i_mfr])
            except (ValueError, IndexError):
                continue
            if ddr > peak_ddr_mc: peak_ddr_mc = ddr
            if cpu > peak_cpu_mc: peak_cpu_mc = cpu
            if min_free_kb is None or mfr < min_free_kb: min_free_kb = mfr
            if psw_first is None: psw_first = psw
            psw_last = psw

    return dict(
        peak_kv = peak_kv,
        peak_ddr_c = peak_ddr_mc / 1000.0,
        peak_cpu_c = peak_cpu_mc / 1000.0,
        peak_rss_gb = peak_rss_kb / 1024.0 / 1024.0,
        swap_mb = (psw_last - psw_first) * 4 / 1024.0 if (psw_first is not None and psw_last is not None) else 0,
        min_free_gb = min_free_kb / 1024.0 / 1024.0 if min_free_kb else 0,
    )

w6 = find_wave6()
points = []
for lbl, hdir, csub, mk, clr, wl in CELLS:
    if lbl == "Phi3 v1_FA bounded (ld)":
        if not w6: continue
        agg = aggregate_cell(w6, "v1_fa_K512_bounded")
    else:
        agg = aggregate_cell(hdir, csub)
    if not agg: continue
    agg['label'] = lbl
    agg['marker'] = mk
    agg['color'] = clr
    agg['workload'] = wl
    points.append(agg)
    print(f"  {lbl}: kv={agg['peak_kv']:5d}  ddr={agg['peak_ddr_c']:.1f}°C  cpu={agg['peak_cpu_c']:.1f}°C  rss={agg['peak_rss_gb']:.2f}GB  swap={agg['swap_mb']:.0f}MB")

if not points:
    print("No data collected. Wave directories not found.")
    sys.exit(1)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("KV cache size vs memory utilization vs DRAM heat vs SoC heat", fontsize=14, fontweight='bold')

ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]

def scatter_with_labels(ax, x, y, points, x_lbl, y_lbl, log_x=True):
    seen = set()
    for p in points:
        legend = p['label'] if p['label'] not in seen else None
        seen.add(p['label'])
        ax.scatter(p[x], p[y], marker=p['marker'], c=p['color'], s=140,
                   edgecolors='black', linewidth=0.8, label=legend, zorder=3)
    if log_x:
        ax.set_xscale('log')
    ax.set_xlabel(x_lbl, fontsize=11)
    ax.set_ylabel(y_lbl, fontsize=11)
    ax.grid(True, which='both', alpha=0.3, zorder=1)

scatter_with_labels(ax_a, 'peak_kv', 'peak_ddr_c', points,
                    "Peak KV cache size (cells, log scale)", "Peak DDR temperature (°C)")
ax_a.set_title("(a) DRAM heat vs KV cache size", fontsize=12)
ax_a.axhline(105, color='red', linestyle='--', alpha=0.5, label='DDR trip point 105°C')
ax_a.set_ylim(45, 70)

scatter_with_labels(ax_b, 'peak_kv', 'peak_cpu_c', points,
                    "Peak KV cache size (cells, log scale)", "Peak CPU (cpullc-0-0) temperature (°C)")
ax_b.set_title("(b) SoC (CPU) heat vs KV cache size", fontsize=12)
ax_b.axhline(95, color='red', linestyle='--', alpha=0.5, label='CPU trip point 95°C')
ax_b.set_ylim(45, 75)

scatter_with_labels(ax_c, 'peak_kv', 'peak_rss_gb', points,
                    "Peak KV cache size (cells, log scale)", "Peak RSS (GB)")
ax_c.set_title("(c) Memory utilization vs KV cache size", fontsize=12)
ax_c.set_ylim(0, 8.5)

scatter_with_labels(ax_d, 'peak_kv', 'swap_mb', points,
                    "Peak KV cache size (cells, log scale)", "UFS swap-out per cell (MB)")
ax_d.set_title("(d) Memory spillover (UFS pressure) vs KV cache size", fontsize=12)
ax_d.set_ylim(-30, 800)

# Shared legend below
handles, labels = ax_a.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
plt.savefig(OUT_FIG, dpi=140, bbox_inches='tight')
print(f"\nSaved: {OUT_FIG}")
