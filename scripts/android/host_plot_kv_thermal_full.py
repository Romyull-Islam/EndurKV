#!/usr/bin/env python3
"""
Generate comprehensive cache-thermal-memory relationship plots across all waves.

Produces:
  1. kv_cache_vs_ddr_temp.png      — the smoking gun
  2. kv_cache_vs_cpu_temp.png
  3. memory_rss_vs_ddr_temp.png    — memory pressure
  4. swap_vs_kv_cache.png          — endurance arm
  5. throughput_vs_kv_cache.png    — trade-off
  6. ppl_vs_kv_cache.png           — accuracy trade-off
  7. wave9_watchdog_trajectory.png — closed-loop control timeline
  8. wave_thermal_timelines.png    — DDR/CPU/freq vs time, per wave
  9. ksweep_pareto.png             — Wave-10 K-sweep (when available)
"""

import os
import sys
import csv
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PHONE_LOGS = "/home/mislam22/EndurKV_workspace/phone-logs"
OUT_DIR    = "/home/mislam22/EndurKV_workspace/EndurKV/figures/thermal_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# (label, dir-glob, cell-subdir, marker, color, workload_class)
CELLS = [
    ("L1B vanilla",        "wave3_real_1780680903",        "vanilla",            "o", "#888888", "narrativeqa-L1B"),
    ("L1B v1 K=512",        "wave3_real_1780680903",        "v1_K512",            "s", "#a0a0a0", "narrativeqa-L1B"),
    ("L1B v1 K=2048",       "wave3_real_1780680903",        "v1_K2048",           "D", "#b0b0b0", "narrativeqa-L1B"),
    ("L1B v1_FA",           "wave3_real_1780680903",        "v1_fa_K512",         "^", "#c0c0c0", "narrativeqa-L1B"),
    ("Phi3 vanilla narrqa", "wave3_phi3_1780719530",        "vanilla",            "o", "#1f77b4", "narrativeqa-Phi3"),
    ("Phi3 v1 narrqa",      "wave3_phi3_1780719530",        "v1_K512",            "s", "#aec7e8", "narrativeqa-Phi3"),
    ("Phi3 v1_FA narrqa",   "wave3_phi3_1780719530",        "v1_fa_K512",         "^", "#7fb3d5", "narrativeqa-Phi3"),
    ("Phi3 TOVA narrqa",    "wave3_phi3_1780719530",        "tova_K512",          "D", "#5dade2", "narrativeqa-Phi3"),
    ("Phi3 llamacpp",       "wave3_phi3_1780719530",        "llamacpp_stock",     "x", "#85c1e9", "narrativeqa-Phi3"),
    ("Phi3 vanilla longd",  "wave4_longdecode_1780750084",  "vanilla",            "o", "#d62728", "long-decode-Phi3"),
    ("Phi3 v1 longd",       "wave4_longdecode_1780750084",  "v1_K512",            "s", "#e74c3c", "long-decode-Phi3"),
    ("Phi3 v1_FA frozen",   "wave4_longdecode_1780750084",  "v1_fa_K512",         "^", "#ff7f0e", "long-decode-Phi3"),
    ("Phi3 v1_FA bounded",  "wave6_v1fa_bounded_*",         "v1_fa_K512_bounded", "P", "#2ca02c", "long-decode-Phi3"),
    ("Phi3 v1_FA² over",    "wave7_v1fa2_*",                "v1_fa2",             "v", "#9467bd", "long-decode-Phi3"),
    ("Phi3 v1_FA² sel",     "wave8_v1fa2_sel_*",            "v1_fa2_selective",   "*", "#bcbd22", "long-decode-Phi3"),
    ("Phi3 v1_FA²-stack",   "wave9_v1fa2_stack_1780796320", "v1_fa2_stack",       "h", "#17becf", "long-decode-Phi3"),
]

def find_cell_dir(dir_glob, cell_subdir):
    candidates = sorted(glob.glob(os.path.join(PHONE_LOGS, dir_glob)))
    for c in candidates:
        full = os.path.join(c, cell_subdir)
        if os.path.isdir(full):
            return full
        # nested form
        nested = os.path.join(c, os.path.basename(c), cell_subdir)
        if os.path.isdir(nested):
            return nested
    return None

def col_idx(csv_path, name):
    with open(csv_path) as f:
        header = f.readline().rstrip().split(',')
    try:
        return header.index(name)
    except ValueError:
        return None

def aggregate(cell_root):
    sensors = os.path.join(cell_root, "sensors.csv")
    stress  = os.path.join(cell_root, "stress.csv")
    if not (os.path.isfile(sensors) and os.path.isfile(stress)):
        return None
    peak_kv = 0; peak_rss_kb = 0; mean_tps = 0; sum_tps = 0; n_iters = 0
    ppl = None
    with open(stress) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                pk = int(row.get('peak_kv_cells',0) or 0); peak_kv = max(peak_kv, pk)
                pr = int(row.get('peak_rss_kb',0) or 0);   peak_rss_kb = max(peak_rss_kb, pr)
                t = float(row.get('decode_tps',0) or 0)
                if t > 0:
                    sum_tps += t; n_iters += 1
                if ppl is None:
                    p = row.get('ppl') or row.get('perplexity')
                    if p:
                        try: ppl = float(p)
                        except: pass
            except (ValueError,KeyError):
                pass
    mean_tps = sum_tps / max(n_iters,1)

    # if PPL not in stress.csv, try iter*/meta.json
    if not ppl:
        for mj in sorted(glob.glob(os.path.join(cell_root,"iter*/meta.json")))[:1]:
            try:
                with open(mj) as f:
                    m = json.load(f)
                ppl = m.get('perplexity')
            except Exception: pass

    i_ddr = col_idx(sensors,'ddr_temp_mc')
    i_cpu = col_idx(sensors,'cpullc-0-0_temp_mc')
    i_psw = col_idx(sensors,'vmstat_pswpout')
    i_mfr = col_idx(sensors,'mem_avail_kb')
    i_f   = col_idx(sensors,'cpu6_freq_hz')

    if i_ddr is None or i_cpu is None: return None
    peak_ddr=0; peak_cpu=0; min_free_kb=None; psw_first=None; psw_last=None
    freq_min=None; freq_max=0
    with open(sensors) as f:
        f.readline()
        for line in f:
            cols = line.split(',')
            try:
                d = int(cols[i_ddr]); c = int(cols[i_cpu])
                p = int(cols[i_psw]) if i_psw is not None else 0
                m = int(cols[i_mfr]) if i_mfr is not None else 0
                fr = int(cols[i_f]) if i_f is not None else 0
            except (ValueError, IndexError):
                continue
            if d>peak_ddr: peak_ddr=d
            if c>peak_cpu: peak_cpu=c
            if i_psw is not None and psw_first is None: psw_first=p
            if i_psw is not None: psw_last=p
            if i_mfr is not None and (min_free_kb is None or m<min_free_kb): min_free_kb=m
            if i_f is not None and fr>0:
                if freq_min is None or fr<freq_min: freq_min=fr
                if fr>freq_max: freq_max=fr
    swap_mb = (psw_last-psw_first)*4/1024 if (psw_first is not None and psw_last is not None) else 0
    return dict(
        peak_kv=peak_kv, peak_rss_gb=peak_rss_kb/1024/1024,
        mean_tps=mean_tps, n_iters=n_iters, ppl=ppl,
        peak_ddr_c=peak_ddr/1000, peak_cpu_c=peak_cpu/1000,
        swap_mb=swap_mb, min_free_gb=(min_free_kb/1024/1024) if min_free_kb else 0,
        freq_min_mhz=(freq_min/1000) if freq_min else 0,
        freq_max_mhz=(freq_max/1000) if freq_max else 0,
    )

# Collect points
points = []
for lbl, dg, cs, mk, clr, wc in CELLS:
    cd = find_cell_dir(dg, cs)
    if not cd: continue
    a = aggregate(cd)
    if not a: continue
    a.update(label=lbl, marker=mk, color=clr, workload=wc, cell_dir=cd)
    points.append(a)
    print(f"  {lbl:25s} kv={a['peak_kv']:5d} ddr={a['peak_ddr_c']:.1f}C cpu={a['peak_cpu_c']:.1f}C rss={a['peak_rss_gb']:.2f}GB tps={a['mean_tps']:.2f} ppl={a['ppl']}")

print(f"\nCollected {len(points)} cells\n")

# Also include Wave-10 partial if available
W10_GLOB = sorted(glob.glob(os.path.join(PHONE_LOGS,"wave10_ksweep_*")))
if W10_GLOB:
    w10_root = W10_GLOB[-1]
    for k in [256,384,512,768,1024,2048]:
        kdir = os.path.join(w10_root, f"K{k}")
        if not os.path.isdir(kdir):
            kdir = os.path.join(w10_root, os.path.basename(w10_root), f"K{k}")
        if os.path.isdir(kdir):
            a = aggregate(kdir)
            if a:
                a.update(label=f"W10 K={k}", marker="X", color="#000000", workload="K-sweep", cell_dir=kdir)
                points.append(a)
                print(f"  Wave-10 K={k:4d}: kv={a['peak_kv']:5d} ddr={a['peak_ddr_c']:.1f}C cpu={a['peak_cpu_c']:.1f}C tps={a['mean_tps']:.2f} ppl={a['ppl']}")

# ============================================================================
# Plot helpers
# ============================================================================
def scatter_plot(xkey, ykey, xlabel, ylabel, title, out_name, log_x=True, ylim=None, hline=None):
    fig, ax = plt.subplots(figsize=(10,7))
    seen = set()
    for p in points:
        if p.get(xkey) is None or p.get(ykey) is None: continue
        leg = p['label'] if p['label'] not in seen else None
        seen.add(p['label'])
        size = 200 if 'v1_FA²' in p['label'] or 'W10' in p['label'] else 130
        ax.scatter(p[xkey], p[ykey], marker=p['marker'], c=p['color'], s=size,
                   edgecolors='black', linewidth=0.8, label=leg, zorder=3, alpha=0.85)
    if log_x: ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    if ylim: ax.set_ylim(ylim)
    if hline:
        for h, lbl in hline:
            ax.axhline(h, color='red', linestyle='--', alpha=0.4, label=lbl)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=140)
    plt.close()
    print(f"  saved {out_name}")

# 1. KV cache vs DDR temperature
scatter_plot('peak_kv','peak_ddr_c',
             'Peak KV cache (cells, log scale)','Peak DDR temperature (°C)',
             '(1) KV cache size vs DDR temperature  —  the cache→bandwidth→heat coupling',
             '1_kv_vs_ddr.png', hline=[(105,'DDR trip 105°C'),(65,'Kernel throttle ~65°C')],
             ylim=(45,110))

# 2. KV cache vs CPU temperature
scatter_plot('peak_kv','peak_cpu_c',
             'Peak KV cache (cells, log scale)','Peak CPU big-core temperature (°C)',
             '(2) KV cache size vs CPU (cpullc-0-0) temperature',
             '2_kv_vs_cpu.png', hline=[(95,'CPU trip 95°C')], ylim=(50,100))

# 3. Memory RSS vs DDR temperature
scatter_plot('peak_rss_gb','peak_ddr_c',
             'Peak RSS (GB)','Peak DDR temperature (°C)',
             '(3) Memory pressure (RSS) vs DDR temperature',
             '3_memrss_vs_ddr.png', log_x=False, ylim=(45,80))

# 4. Swap-out vs KV cache
scatter_plot('peak_kv','swap_mb',
             'Peak KV cache (cells, log scale)','UFS swap-out per cell (MB)',
             '(4) UFS endurance pressure vs KV cache size',
             '4_swap_vs_kv.png', log_x=True, ylim=(-50, 1700))

# 5. Throughput vs KV cache
scatter_plot('peak_kv','mean_tps',
             'Peak KV cache (cells, log scale)','Mean decode throughput (tok/s)',
             '(5) Throughput vs KV cache size',
             '5_throughput_vs_kv.png', log_x=True)

# 6. PPL vs KV cache  (filter cells with PPL data)
ppl_points = [p for p in points if p.get('ppl')]
fig, ax = plt.subplots(figsize=(10,7))
for p in ppl_points:
    leg = p['label']
    size = 200 if 'v1_FA²' in p['label'] or 'W10' in p['label'] else 130
    ax.scatter(p['peak_kv'], p['ppl'], marker=p['marker'], c=p['color'], s=size,
               edgecolors='black', linewidth=0.8, label=leg, zorder=3, alpha=0.85)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Peak KV cache (cells, log scale)', fontsize=12)
ax.set_ylabel('Perplexity (log scale)', fontsize=12)
ax.set_title('(6) PPL vs KV cache size  —  the accuracy trade-off', fontsize=13, fontweight='bold')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=8, loc='best', ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'6_ppl_vs_kv.png'), dpi=140); plt.close()
print('  saved 6_ppl_vs_kv.png')

# 7. Wave-9 watchdog trajectory (DDR + freq tier over time)
W9_DIR = sorted(glob.glob(os.path.join(PHONE_LOGS,"wave9_v1fa2_stack_1780796320*")))
if W9_DIR:
    w9 = W9_DIR[-1]
    sensors_csv = os.path.join(w9,"v1_fa2_stack/sensors.csv")
    if not os.path.isfile(sensors_csv):
        sensors_csv = os.path.join(w9, os.path.basename(w9), "v1_fa2_stack/sensors.csv")
    if os.path.isfile(sensors_csv):
        i_t   = col_idx(sensors_csv,'wall_sec')
        i_ddr = col_idx(sensors_csv,'ddr_temp_mc')
        i_f   = col_idx(sensors_csv,'cpu6_freq_hz')
        ts=[]; ddrs=[]; freqs=[]
        with open(sensors_csv) as f:
            f.readline()
            for line in f:
                cols=line.split(',')
                try:
                    t = float(cols[i_t]) if i_t is not None else len(ts)*0.2
                    d = int(cols[i_ddr])/1000
                    fr = int(cols[i_f])/1000 if i_f is not None and cols[i_f] else 0
                except (ValueError,IndexError): continue
                ts.append(t); ddrs.append(d); freqs.append(fr)
        if ts:
            t0 = ts[0]
            tarr = [(x-t0)/60 for x in ts]  # minutes
            fig, ax1 = plt.subplots(figsize=(12,6))
            ax1.plot(tarr, ddrs, color='#d62728', lw=1.5, label='DDR temp (°C)')
            ax1.axhline(58, color='red', linestyle=':', alpha=0.4, label='watchdog tier 1 (58°C)')
            ax1.axhline(62, color='red', linestyle='--', alpha=0.4, label='watchdog tier 2 (62°C)')
            ax1.axhline(65, color='red', linestyle='-.', alpha=0.4, label='watchdog tier 3 (65°C)')
            ax1.set_xlabel('Wall time (min)', fontsize=12)
            ax1.set_ylabel('DDR temperature (°C)', color='#d62728', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='#d62728')
            ax1.set_ylim(30, 80)
            ax2 = ax1.twinx()
            ax2.plot(tarr, freqs, color='#1f77b4', lw=1, alpha=0.5, label='CPU6 freq (MHz)')
            ax2.set_ylabel('CPU6 freq (MHz)', color='#1f77b4', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='#1f77b4')
            ax1.set_title('(7) Wave-9 closed-loop control trajectory  —  DDR drives CPU freq via watchdog',
                          fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1+lines2, labels1+labels2, loc='upper right', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR,'7_wave9_watchdog_trajectory.png'), dpi=140); plt.close()
            print('  saved 7_wave9_watchdog_trajectory.png')

# 8. Matched-conditions bar: Wave-4 cache-size -> DDR (the smoking gun)
# Previously hard-coded W4_DDR=[62.9,62.5,54.4] / W4_THROT=['YES','YES','NO'].
# Now derived live from phone-logs/wave4_longdecode_1780750084/{vanilla,
# v1_fa_K512,v1_K512}/sensors.csv via the existing aggregate() helper so the
# figure is reproducible from raw data.
W4_KERNEL_THROTTLE_C = 65.0  # documented threshold; see CAPTIONS.md
W4_CELLS = [
    ('vanilla',     'vanilla',      '#d62728'),
    ('v1_FA frozen','v1_fa_K512',   '#ff7f0e'),
    ('v1 K=512',    'v1_K512',      '#2ca02c'),
]
W4_DG = 'wave4_longdecode_1780750084'
w4_labels: list = []
w4_ddr:    list = []
w4_throt:  list = []
w4_color:  list = []
for nice, cell_name, color in W4_CELLS:
    cd = find_cell_dir(W4_DG, cell_name)
    if not cd:
        print(f"  [warn] missing wave4 cell dir for {cell_name}; skipping")
        continue
    a = aggregate(cd)
    if not a:
        print(f"  [warn] aggregate failed for {cd}; skipping")
        continue
    ddr_c = float(a['peak_ddr_c'])
    cache = int(a.get('peak_kv') or 0)
    w4_labels.append(f"{nice}\n(cache {cache})")
    w4_ddr.append(ddr_c)
    w4_throt.append('YES' if ddr_c >= W4_KERNEL_THROTTLE_C else 'NO')
    w4_color.append(color)

if w4_ddr:
    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.bar(w4_labels, w4_ddr, color=w4_color, edgecolor='black')
    for b, v, t in zip(bars, w4_ddr, w4_throt):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}°C\n(throttle: {t})',
                ha='center', fontsize=10, fontweight='bold')
    ax.axhline(W4_KERNEL_THROTTLE_C, color='red', linestyle='--',
               label=f'Kernel throttle ~{W4_KERNEL_THROTTLE_C:.0f}°C')
    ax.set_ylabel('Peak DDR temperature (°C)', fontsize=12)
    ax.set_ylim(40, 75)
    ax.set_title('(8) Wave-4 matched-conditions: cache size IS the binding thermal lever',
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'8_wave4_smokinggun.png'), dpi=140); plt.close()
    print('  saved 8_wave4_smokinggun.png (live from sensors.csv)')
else:
    print('  [skip] 8_wave4_smokinggun.png — no wave4 sensor data found')

# 9. K-sweep partial — only if we have Wave-10 data
KSWEEP = sorted([p for p in points if p['label'].startswith('W10') or 'K=512' in p['label']],
                key=lambda x: x['peak_kv'])
if len(KSWEEP) >= 2:
    Ks   = [p['peak_kv'] for p in KSWEEP]
    DDRs = [p['peak_ddr_c'] for p in KSWEEP]
    PPLs = [p['ppl'] for p in KSWEEP if p.get('ppl')]
    TPSs = [p['mean_tps'] for p in KSWEEP]
    fig, axes = plt.subplots(1,3, figsize=(15,5))
    axes[0].plot(Ks, DDRs, 'o-', color='#d62728', lw=2, markersize=10)
    axes[0].set_xlabel('Peak KV cells'); axes[0].set_ylabel('Peak DDR (°C)')
    axes[0].set_title('K vs peak DDR'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(Ks[:len(PPLs)], PPLs, 'o-', color='#1f77b4', lw=2, markersize=10)
    axes[1].set_xlabel('Peak KV cells'); axes[1].set_ylabel('Perplexity')
    axes[1].set_title('K vs PPL'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(Ks, TPSs, 'o-', color='#2ca02c', lw=2, markersize=10)
    axes[2].set_xlabel('Peak KV cells'); axes[2].set_ylabel('Mean tok/s')
    axes[2].set_title('K vs throughput'); axes[2].grid(True, alpha=0.3)
    plt.suptitle('(9) Wave-10 K-sweep Pareto curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'9_ksweep_pareto.png'), dpi=140); plt.close()
    print('  saved 9_ksweep_pareto.png')

print(f"\nAll plots saved to {OUT_DIR}/")
