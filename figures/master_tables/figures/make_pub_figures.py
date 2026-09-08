#!/usr/bin/env python3
"""Publication figures for the HotMobile draft, following MobiSys/MobiCom conventions:
small multiples, one fixed color+hatch per policy everywhere, Okabe-Ito palette,
direct value labels, dashed threshold lines, stacked panels (no twin axes).
Data: /tmp/def_cpu (cold-protocol CPU campaign). Regenerate when new cells land."""
import json, re, os, csv, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = '/tmp/nat_cpu'
OUT  = os.path.dirname(os.path.abspath(__file__))

# Okabe-Ito; one fixed color per policy across ALL figures. muKV = vermilion accent.
C = dict(vanilla='#7a7a7a', snapkv='#0072B2', adakv='#56B4E9',
         mukv='#D55E00', mukv_swap='#E69F00',
         streamingllm='#009E73', h2o='#CC79A7', tova='#F0E442', tova_canon='#999933')
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.8, 'xtick.direction': 'out', 'ytick.direction': 'out'})

def load(p):
    try: return json.loads(re.sub(r'\binf\b|\bnan\b', 'null', open(p, encoding='utf-8', errors='replace').read()))
    except Exception: return None

def cell(name):
    d = os.path.join(ROOT, name)
    g = load(os.path.join(d, 'gen.json'))
    if not g: return None
    pp = load(os.path.join(d, 'ppl.json'))
    # energy: total system power (usb + battery discharge), phase split
    E = dict(pf=0.0, dc=0.0); pk = dict(ddr=0, cpu=0)
    f = os.path.join(d, 'sensors.csv')
    if os.path.exists(f):
        rows = list(csv.DictReader(open(f, encoding='utf-8', errors='replace')))
        t = [float(r['monotonic_s']) for r in rows]
        tN = t[-1]; bs = tN - (g['prefill_ms'] + g['decode_ms'])/1e3; pe = bs + g['prefill_ms']/1e3
        cpu_cols = ['cpu-1-0-0_temp_mc','cpu-1-0-1_temp_mc','cpu-1-1-0_temp_mc','cpu-1-1-1_temp_mc']
        def fnum(r, c):
            try: return float(r[c])
            except Exception: return None
        for i in range(1, len(rows)):
            dt = t[i]-t[i-1]
            if dt <= 0 or dt > 5: continue
            iu, vu = fnum(rows[i],'usb_current_ua'), fnum(rows[i],'usb_voltage_uv')
            p = (iu/1e6)*(vu/1e6) if (iu and vu) else 0.0
            ib, vb = fnum(rows[i],'bat_current_ma'), fnum(rows[i],'bat_voltage_mv')
            p += (ib/1e3)*(vb/1e3) if (ib and vb and ib > 0) else 0.0
            mid = (t[i]+t[i-1])/2
            if bs <= mid < pe: E['pf'] += p*dt
            elif mid >= pe:    E['dc'] += p*dt
            if mid >= bs:
                dd = fnum(rows[i],'ddr_temp_mc'); pk['ddr'] = max(pk['ddr'], dd/1000 if dd else 0)
                cc = max([fnum(rows[i],c) or 0 for c in cpu_cols]); pk['cpu'] = max(pk['cpu'], cc/1000)
    return dict(g=g, ppl=(pp or {}).get('perplexity'), E=E, pk=pk)

ORDER = [('vanilla','vanilla'), ('snapkv','SnapKV'), ('adakv','AdaKV'),
         ('streamingllm','StrLLM'), ('h2o','H2O'), ('tova','TOVA'), ('tova_canon','TOVA-c'),
         ('mukv_swap','μKV-swap'), ('mukv_faon','μKV')]
cells = {k: cell(k) for k, _ in ORDER}
have  = [(k, l) for k, l in ORDER if cells.get(k)]
def col(k): return C.get('mukv' if k == 'mukv_faon' else k, '#888888')

# ---------------- Fig B: main results, 1x4 small multiples ----------------
fig, axs2d = plt.subplots(2, 2, figsize=(3.35, 3.0), dpi=200); axs = axs2d.flatten()
panels = [('decode tps',        lambda c: c['g']['decode_tps'],                 'Decode (tok/s)', None),
          ('energy',            lambda c: (c['E']['pf']+c['E']['dc'])/3.6,      'Energy (mWh)',   None),
          ('cache',             lambda c: c['g'].get('retained_kv_bytes', 0)/1048576, 'Retained KV (MiB)', None),
          ('peak ddr',          lambda c: c['pk']['ddr'],                       'Peak DDR (°C)', 65)]
for ax, (name, fn, ylab, thr) in zip(axs, panels):
    xs = np.arange(len(have))
    vals = [fn(cells[k]) for k, _ in have]
    if name == 'peak ddr':
        # dot plot: bars on a truncated axis misrepresent length; points do not
        for x, v, (k, _) in zip(xs, vals, have):
            ax.plot([x, x], [50, v], color='#dddddd', lw=1.0, zorder=1)
            ax.plot(x, v, 'o', ms=5.5, color=col(k), zorder=3,
                    markeredgecolor='white', markeredgewidth=0.6)
        ax.set_ylim(50, 70)
        ax.axhline(thr, color='#c00000', ls='--', lw=0.9)
        ax.text(len(have)-0.4, thr+0.5, 'throttle', color='#c00000', fontsize=5.8, ha='right')
    else:
        ax.bar(xs, vals, width=0.62,
               color=[col(k) for k, _ in have],
               hatch=['//' if k.startswith('mukv') else '' for k, _ in have],
               edgecolor='white', linewidth=0.4)
        if name == 'cache':
            ax.set_yscale('log')
            ax.set_yticks([10, 100, 1000]); ax.set_yticklabels(['10', '100', '1000'], fontsize=6)
        for x, v, (k, _) in zip(xs, vals, have):   # value labels: vanilla + muKV only
            if k in ('mukv_faon', 'vanilla'):
                ax.text(x, v*1.06 if name == 'cache' else v + max(vals)*0.02,
                        f'{v:.0f}' if v >= 10 else f'{v:.1f}',
                        ha='center', va='bottom', fontsize=6.2, fontweight='bold', color=col(k))
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in have], rotation=45, ha='right', fontsize=6.2)
    ax.set_ylabel(ylab, fontsize=7)
fig.tight_layout(w_pad=0.8, h_pad=1.2)
fig.savefig(f'{OUT}/fig_main_results.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig_main_results.png', bbox_inches='tight', dpi=300)
plt.close(fig)

# ---------------- Fig C: cache trajectory (prefill + decode) ----------------
fig, ax = plt.subplots(figsize=(3.5, 2.1), dpi=200)
per_cell_b = None
v = cells.get('vanilla')
if v and v['g'].get('retained_kv_bytes'): per_cell_b = v['g']['retained_kv_bytes']/v['g']['n_prompt_tokens']
SHOW = [k for k in ('vanilla','snapkv','adakv','mukv_faon') if cells.get(k)]
LBL  = dict(vanilla='vanilla', snapkv='SnapKV', adakv='AdaKV', mukv_faon='μKV')
for k in SHOW:
    d = os.path.join(ROOT, k); g = cells[k]['g']
    xs, ys = [], []
    pf = os.path.join(d, 'gen_prefill.csv')
    tsum = 0.0
    if os.path.exists(pf):
        for r in csv.DictReader(open(pf, encoding='utf-8', errors='replace')):
            try:
                xs.append(len(xs))  # placeholder; replaced by time below
            except Exception: pass
    # simpler: token-index axis. prefill: tokens processed; decode: prompt + step
    xs, ys = [], []
    if os.path.exists(pf):
        for r in csv.DictReader(open(pf, encoding='utf-8', errors='replace')):
            try: xs.append(int(r['n_tokens_processed'])); ys.append(int(r['n_kv_cells']))
            except Exception: pass
    ret = round(g['retained_kv_bytes']/per_cell_b) if per_cell_b and g.get('retained_kv_bytes') else None
    st = os.path.join(d, 'gen_steps.csv'); live = ret if ret else (ys[-1] if ys else 0)
    if ys and ret is not None:
        xs.append(xs[-1]); ys.append(ret)   # the eviction drop at end of prefill
    if os.path.exists(st):
        n0 = g['n_prompt_tokens']
        for i, r in enumerate(csv.DictReader(open(st, encoding='utf-8', errors='replace'))):
            try: ev = int(float(r.get('evicted_this_step', 0) or 0))
            except Exception: ev = 0
            live += 1 - ev
            if i % 16 == 0: xs.append(n0 + i); ys.append(live)
    ax.plot(xs, ys, color=col(k), lw=1.4 if k == 'mukv_faon' else 1.0,
            ls='-' if k == 'mukv_faon' else '-', zorder=3 if k == 'mukv_faon' else 2)
    if xs: ax.annotate(LBL[k], (xs[-1], ys[-1]), textcoords='offset points', xytext=(3, -2),
                       fontsize=6.4, color=col(k), fontweight='bold' if k == 'mukv_faon' else 'normal')
ax.axhline(1024, color='#555', ls=':', lw=0.8)
ax.text(150, 1250, 'budget K=1024', fontsize=6, color='#555')
ax.annotate('all policies build the full\nprompt cache during prefill', xy=(5200, 5600),
            xytext=(1300, 8200), fontsize=5.8, color='#666',
            arrowprops=dict(arrowstyle='->', color='#999', lw=0.7))
npt = cells['vanilla']['g']['n_prompt_tokens'] if cells.get('vanilla') else 9737
ax.axvline(npt, color='#bbb', lw=0.7)
ax.text(npt-300, ax.get_ylim()[1]*0.75, 'prefill', fontsize=6, color='#888', ha='right')
ax.text(npt+300, ax.get_ylim()[1]*0.75, 'decode', fontsize=6, color='#888')
ax.set_xlabel('Tokens processed', fontsize=7); ax.set_ylabel('Live KV cells', fontsize=7)
ax.set_xlim(0, 15200)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_cache_trajectory.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig_cache_trajectory.png', bbox_inches='tight', dpi=300)
plt.close(fig)

# ---------------- Fig D: thermal + clock trace, 3 stacked panels ----------------
# DDR (kernel cliff 65C), shell (watchdog trigger 42C), CPU clock. GPU clock is idle
# on the CPU workload; the GPU campaign produces its own trace figure.
fig, (a1, a1b, a2) = plt.subplots(3, 1, figsize=(3.5, 3.4), dpi=200, sharex=True,
                                  gridspec_kw={'hspace': 0.14})
for k in ('vanilla', 'mukv_faon'):
    d = os.path.join(ROOT, k)
    f = os.path.join(d, 'sensors.csv')
    if not os.path.exists(f): continue
    g = cells[k]['g']
    rows = list(csv.DictReader(open(f, encoding='utf-8', errors='replace')))
    t0 = float(rows[0]['monotonic_s'])
    bench_min = (g['prefill_ms'] + g['decode_ms'])/60000.0
    T, dd, sh, ck = [], [], [], []
    for r in rows:
        tm = (float(r['monotonic_s'])-t0)/60
        if tm > bench_min: break            # trim trailing idle
        T.append(tm)
        dd.append(float(r['ddr_temp_mc'])/1000 if r.get('ddr_temp_mc') else np.nan)
        sh.append(float(r['shell_front_temp_mc'])/1000 if r.get('shell_front_temp_mc') else np.nan)
        ck.append(float(r['cpu6_freq_hz'])/1e3 if r.get('cpu6_freq_hz') else np.nan)
    ck = np.convolve(np.nan_to_num(ck, nan=883), np.ones(30)/30, mode='same')
    a1.plot(T, dd, color=col(k), lw=1.0, label=LBL.get(k, k))
    a1b.plot(T, sh, color=col(k), lw=1.0)
    a2.plot(T, ck[:len(T)], color=col(k), lw=1.0)
    if k == 'mukv_faon':
        a1.annotate(f'μKV done {T[-1]:.1f} min', xy=(T[-1], dd[-1]), xytext=(T[-1]+1.2, 44),
                    fontsize=6, color=col(k), fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=col(k), lw=0.7))
a1.axhline(65, color='#c00000', ls='--', lw=0.9)
a1.text(0.3, 65.6, 'kernel throttle 65°C', color='#c00000', fontsize=5.8)
a1.set_ylabel('DDR (°C)', fontsize=7); a1.set_ylim(35, 70)
a1.legend(fontsize=6.2, frameon=False, loc='lower right', ncols=2)
a1b.axhline(42, color='#E69F00', ls='--', lw=0.9)
a1b.text(0.3, 42.4, 'watchdog threshold 42°C', color='#B07800', fontsize=5.8)
a1b.set_ylabel('Shell (°C)', fontsize=7); a1b.set_ylim(30, 48)
a2.set_ylabel('CPU clock (MHz)', fontsize=7); a2.set_xlabel('Time (min)', fontsize=7)
a2.set_ylim(800, 1750); a2.axhline(1632, color='#555', ls=':', lw=0.7)
a2.text(0.3, 1660, 'kernel sustained ceiling ~1632 (both self-limit)', fontsize=5.6, color='#555')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_thermal_trace.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig_thermal_trace.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print('saved: fig_main_results, fig_cache_trajectory, fig_thermal_trace ->', OUT)

# ---------------- Fig E: GPU trace (built when GPU campaign data exists) ----------------
# 3 stacked panels for the GPU run: GPU temp (max gpuss zone), shell temp, GPU clock.
# Sources, in preference order: /tmp/wd_demo (watchdog demo: vanilla_nowd/mukv_nowd/mukv_wd)
# else /tmp/deploy_gpu (vanilla/mukv). Skips silently if neither exists yet.
def gpu_trace():
    src = None
    for root, names in [('/tmp/wd_demo', [('vanilla_nowd','vanilla'), ('mukv_nowd','μKV no wd'), ('mukv_wd','μKV + wd')]),
                        ('/tmp/deploy_gpu', [('vanilla','vanilla'), ('mukv','μKV + wd')])]:
        if all(os.path.exists(os.path.join(root, n, 'sensors.csv')) for n, _ in names):
            src = (root, names); break
    if not src:
        print('gpu trace: no GPU campaign data yet, skipped'); return
    root, names = src
    gcol = dict(vanilla_nowd='#7a7a7a', vanilla='#7a7a7a', mukv_nowd='#0072B2', mukv_wd='#D55E00', mukv='#D55E00')
    fig, (g1, g2, g3) = plt.subplots(3, 1, figsize=(3.5, 3.4), dpi=200, sharex=True,
                                     gridspec_kw={'hspace': 0.14})
    for n, lbl in names:
        rows = list(csv.DictReader(open(os.path.join(root, n, 'sensors.csv'), encoding='utf-8', errors='replace')))
        if not rows: continue
        gz = [c for c in rows[0] if c.startswith('gpuss-')]
        t0 = float(rows[0]['monotonic_s'])
        T, gt, sh, ck = [], [], [], []
        for r in rows:
            T.append((float(r['monotonic_s'])-t0)/60)
            try: gt.append(max(float(r[c]) for c in gz if r.get(c))/1000)
            except Exception: gt.append(np.nan)
            sh.append(float(r['shell_front_temp_mc'])/1000 if r.get('shell_front_temp_mc') else np.nan)
            try: ck.append(float(r['gpu_clk_hz'])/1e6)   # Hz -> MHz
            except Exception: ck.append(np.nan)
        ck = np.convolve(np.nan_to_num(ck, nan=0), np.ones(15)/15, mode='same')
        c = gcol.get(n, '#333')
        g1.plot(T, gt, color=c, lw=1.0, label=lbl)
        g2.plot(T, sh, color=c, lw=1.0)
        g3.plot(T, ck[:len(T)], color=c, lw=1.0)
    g1.set_ylabel('GPU (°C)', fontsize=7)
    g1.axhline(90, color='#c00000', ls='--', lw=0.9); g1.text(0.2, 90.6, 'GPU cap 90°C', color='#c00000', fontsize=5.8)
    g1.legend(fontsize=6, frameon=False, loc='lower right', ncols=len(names))
    g2.set_ylabel('Shell (°C)', fontsize=7)
    g2.axhline(39.5, color='#E69F00', ls='--', lw=0.9); g2.text(0.2, 39.9, 'watchdog 39.5°C', color='#B07800', fontsize=5.8)
    g3.set_ylabel('GPU clock (MHz)', fontsize=7); g3.set_xlabel('Time (min)', fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_gpu_trace.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig_gpu_trace.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('saved: fig_gpu_trace (from', root + ')')
gpu_trace()
