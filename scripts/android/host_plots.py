#!/usr/bin/env python3
"""Generate publication-grade plots from a sweep directory.

Reads:
  <sweep_dir>/<model>/<policy>/K*/<prompt>/repN/repN/
    meta.json, sensors.csv, steps.csv, gen.txt

Outputs PNG figures into <sweep_dir>/figures/ with informative filenames.

Tries to use matplotlib if available; falls back to pure-text "ASCII plots"
when matplotlib is missing.

Usage:
    python3 host_plots.py --sweep-dir <path>  [--truth-jsonl <path>]
"""
import argparse, csv, json, re, statistics, sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    print("WARN: matplotlib not available; only writing CSV summaries", file=sys.stderr)

SHORT = {
    "Llama-3.2-1B-Instruct-Q4_K_M": "Llama-1B",
    "gemma-2-2b-it-Q4_K_M":           "Gemma-2-2B",
    "Phi-3-mini-128k-instruct-Q4_K_M":"Phi-3-128k",
}
POLICY_COLOR = {"vanilla":"black","tova":"tab:orange","v1":"tab:blue","pyramid":"tab:green","h2o":"tab:red"}
POLICY_ORDER = ["vanilla","tova","v1","pyramid","h2o"]


def load_meta(p):
    txt = p.read_text(errors='replace')
    txt = re.sub(r'\bnan\b','null', re.sub(r'\b-?inf\b','null', txt))
    return json.loads(txt)


def load_sensors(p):
    """Return (times_s, dict_field_to_values)."""
    times, fields = [], defaultdict(list)
    try:
        with open(p, errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row.get('monotonic_s') or 0)
                times.append(t)
                for k, v in row.items():
                    if k in ('wall_clock_s', 'monotonic_s'): continue
                    try: fields[k].append(float(v) if v else None)
                    except ValueError: fields[k].append(None)
    except FileNotFoundError:
        return [], {}
    t0 = times[0] if times else 0
    return [t - t0 for t in times], dict(fields)


def load_steps(p):
    """Be tolerant of malformed rows — some logs have CSV rows where a
    multi-line token leaks across rows, leaving fields None."""
    def _i(s, default=0):
        try: return int(s) if s not in (None, '', 'nan') else default
        except (TypeError, ValueError): return default
    def _f(s, default=float('nan')):
        try: return float(s) if s not in (None, '', 'nan', '-nan') else default
        except (TypeError, ValueError): return default
    rows = []
    try:
        with open(p, errors='replace') as f:
            for r in csv.DictReader(f):
                try: rows.append({
                    'step':     _i(r.get('step')),
                    'wall_us':  _i(r.get('wall_us')),
                    'n_kv':     _i(r.get('n_kv_cells')),
                    'rss':      _i(r.get('rss_kb')),
                    'log_prob': _f(r.get('log_prob')),
                    'evicted':  _i(r.get('evicted_this_step')),
                })
                except Exception: pass
    except FileNotFoundError: pass
    return rows


def gather_cells(sweep_dir):
    """Return list of cell dicts."""
    cells = []
    for mf in Path(sweep_dir).rglob("meta.json"):
        try: meta = load_meta(mf)
        except Exception: continue
        d = {
            'model':  SHORT.get(meta.get("model","?").split("/")[-1].replace(".gguf",""), "?"),
            'policy': meta.get("policy","?"),
            'prompt_id': meta.get("prompt_id","?"),
            'meta':   meta,
            'dir':    mf.parent,
            'sensors': mf.parent / 'sensors.csv',
            'steps':   mf.parent / 'steps.csv',
            'gen':     mf.parent / 'gen.txt',
        }
        cells.append(d)
    return cells


# ---------------------------------------------------------------------------
# F1 / EM / ROUGE scoring (pandas-free)
# ---------------------------------------------------------------------------
import string
from collections import Counter

def normalize_text(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())

def _f1(pred, gold):
    p = normalize_text(pred).split(); g = normalize_text(gold).split()
    if not p or not g: return 0.0
    common = Counter(p) & Counter(g); n = sum(common.values())
    if n == 0: return 0.0
    pr = n/len(p); rc = n/len(g); return 2*pr*rc/(pr+rc)

def _em(pred, gold): return int(normalize_text(pred) == normalize_text(gold))

def _rouge_l(pred, gold):
    p = normalize_text(pred).split(); g = normalize_text(gold).split()
    if not p or not g: return 0.0
    m, n = len(p), len(g)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = dp[i-1][j-1]+1 if p[i-1]==g[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0: return 0.0
    return 2*(lcs/m)*(lcs/n)/((lcs/m)+(lcs/n))


def attach_scores(cells, truth_jsonl):
    truth = {}
    if truth_jsonl and Path(truth_jsonl).exists():
        for line in open(truth_jsonl):
            r = json.loads(line)
            truth[r['prompt_id']] = {'task': r['task'], 'gold': r.get('ground_truth', [])}
    for c in cells:
        info = truth.get(c['prompt_id'])
        c['task'] = info['task'] if info else None
        gold_list = info['gold'] if info else []
        gen = c['gen'].read_text(errors='replace').strip() if c['gen'].exists() else ""
        c['gen_text'] = gen
        c['f1']  = max((_f1(gen, g) for g in gold_list), default=0.0)
        c['em']  = max((_em(gen, g) for g in gold_list), default=0)
        c['rouge'] = max((_rouge_l(gen, g) for g in gold_list), default=0.0)
    return cells


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def figdir(sweep_dir):
    d = Path(sweep_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_thermal_per_cell(cells, out_dir):
    """One plot per (model, prompt): skin temp vs time, one line per policy."""
    if not HAVE_MPL: return
    grouped = defaultdict(list)  # (model, prompt) -> [(policy, times, temps)]
    for c in cells:
        ts, fields = load_sensors(c['sensors'])
        if not ts or 'shell_front_temp_mc' not in fields: continue
        temps = [v/1000.0 if v else None for v in fields['shell_front_temp_mc']]
        grouped[(c['model'], c['prompt_id'])].append((c['policy'], ts, temps))

    for (model, prompt), entries in grouped.items():
        plt.figure(figsize=(8,4))
        for pol, ts, temps in sorted(entries, key=lambda e: POLICY_ORDER.index(e[0]) if e[0] in POLICY_ORDER else 99):
            plt.plot(ts, temps, label=pol, color=POLICY_COLOR.get(pol,'gray'), lw=1.5)
        plt.xlabel("time (s)"); plt.ylabel("skin temp (°C)")
        plt.title(f"Thermal buildup — {model} / {prompt}")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir/f"thermal_{model}_{prompt}.png", dpi=120)
        plt.close()


def plot_power_per_cell(cells, out_dir):
    if not HAVE_MPL: return
    grouped = defaultdict(list)
    for c in cells:
        ts, fields = load_sensors(c['sensors'])
        if not ts or 'bat_current_ma' not in fields: continue
        amps = [v if v else None for v in fields['bat_current_ma']]
        grouped[(c['model'], c['prompt_id'])].append((c['policy'], ts, amps))
    for (model, prompt), entries in grouped.items():
        plt.figure(figsize=(8,4))
        for pol, ts, amps in sorted(entries, key=lambda e: POLICY_ORDER.index(e[0]) if e[0] in POLICY_ORDER else 99):
            plt.plot(ts, amps, label=pol, color=POLICY_COLOR.get(pol,'gray'), lw=1.0)
        plt.xlabel("time (s)"); plt.ylabel("battery current (mA, negative = drawing)")
        plt.title(f"Power draw — {model} / {prompt}")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir/f"power_{model}_{prompt}.png", dpi=120)
        plt.close()


def plot_kv_trajectory(cells, out_dir):
    if not HAVE_MPL: return
    grouped = defaultdict(list)
    for c in cells:
        rows = load_steps(c['steps'])
        if not rows: continue
        steps = [r['step'] for r in rows]
        retention_proxy = c['meta'].get('mean_retention_ratio', 1.0) or 1.0
        # n_kv_cells is max_pos+1, so "live KV" ≈ n_kv × retention_ratio
        live_kv = [r['n_kv'] * retention_proxy for r in rows]
        grouped[(c['model'], c['prompt_id'])].append((c['policy'], steps, live_kv))
    for (model, prompt), entries in grouped.items():
        plt.figure(figsize=(8,4))
        for pol, steps, live in sorted(entries, key=lambda e: POLICY_ORDER.index(e[0]) if e[0] in POLICY_ORDER else 99):
            plt.plot(steps, live, label=pol, color=POLICY_COLOR.get(pol,'gray'), lw=1.5)
        plt.xlabel("decode step"); plt.ylabel("live KV positions (n_kv × retention)")
        plt.title(f"KV trajectory — {model} / {prompt}")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir/f"kv_{model}_{prompt}.png", dpi=120)
        plt.close()


def plot_f1_bars(cells, out_dir):
    if not HAVE_MPL: return
    # group by (model, task, policy) — mean over reps
    g = defaultdict(list)
    for c in cells:
        if c['task'] is None: continue
        g[(c['model'], c['task'], c['policy'])].append(c['f1'])
    # Per-model figure: x = task, grouped bars per policy
    models = sorted({k[0] for k in g})
    for model in models:
        tasks = sorted({k[1] for k in g if k[0] == model})
        pols  = [p for p in POLICY_ORDER if any((model, t, p) in g for t in tasks)]
        if not tasks or not pols: continue
        plt.figure(figsize=(max(6, len(tasks)*1.5), 4))
        width = 0.8 / len(pols)
        for i, pol in enumerate(pols):
            xs = [t_i + i*width for t_i in range(len(tasks))]
            ys = [statistics.mean(g.get((model, t, pol), [0])) for t in tasks]
            plt.bar(xs, ys, width=width*0.9, label=pol, color=POLICY_COLOR.get(pol,'gray'))
        plt.xticks([i + 0.4 for i in range(len(tasks))], tasks, rotation=20, ha='right')
        plt.ylabel("F1")
        plt.title(f"F1 per task — {model}")
        plt.legend(); plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(out_dir/f"f1_bars_{model}.png", dpi=120)
        plt.close()


def plot_avg_thermal_per_model(cells, out_dir):
    """For each model, average thermal trajectories across all cells of each
    policy, plot one figure per (model, zone). Covers:
      - skin (user-facing)
      - battery
      - CPU max (across all per-core sensors)
      - GPU max (across all gpuss shader clusters)
      - DDR (memory)
    Time axis = seconds since cell start (cells are time-aligned at t=0)."""
    if not HAVE_MPL: return
    import numpy as np

    # Define which fields to track per zone, with how to derive a single value per timestep
    def _max_temp(fields_present, name_predicate, scale=1000.0):
        """Return per-step max across temperature fields matching `name_predicate`.
        Only fields ending in `_temp_mc` are considered — avoids catching
        `cpu0_freq_hz` (frequency) or `cpu0_cool_state` (governor state).
        Also excludes hw-trip-* (constant thresholds, not real-time)."""
        keys = [k for k in fields_present
                if k.endswith('_temp_mc') and name_predicate(k)
                and 'hw-trip' not in k]
        if not keys: return None
        n = len(fields_present[keys[0]])
        out = []
        for i in range(n):
            vals = [fields_present[k][i] / scale for k in keys
                    if fields_present[k][i] is not None and fields_present[k][i] > 0]
            out.append(max(vals) if vals else None)
        return out

    # zone_extractor: zone_name -> function(fields) -> list of values aligned with times
    zone_extractors = {
        'skin':    lambda f: [v/1000.0 if v else None for v in f.get('shell_front_temp_mc', [])],
        'battery': lambda f: [v/10.0   if v else None for v in f.get('bat_phone_temp_dc',  [])],
        # CPU = max across all CPU cores + the LLC caches (cpu-*, cpullc-*)
        'cpu':     lambda f: _max_temp(f, lambda k: k.startswith('cpu')),
        # GPU = max across the 11 GPU shader-subsystem clusters
        'gpu':     lambda f: _max_temp(f, lambda k: k.startswith('gpuss-')),
        # NPU/DSP = max across the Hexagon vector/matrix extensions
        'npu':     lambda f: _max_temp(f, lambda k: k.startswith('nsph')),
        'ddr':     lambda f: [v/1000.0 if v else None for v in f.get('ddr_temp_mc', [])],
    }
    zone_labels = {
        'skin':    'skin temperature (°C)',
        'battery': 'battery temperature (°C)',
        'cpu':     'CPU max core temperature (°C)',
        'gpu':     'GPU max shader cluster temperature (°C)',
        'npu':     'Hexagon DSP max temperature (°C)',
        'ddr':     'memory (DDR) temperature (°C)',
    }

    # Group: (model, zone) -> policy -> list of (times, values)
    grouped = defaultdict(lambda: defaultdict(list))
    for c in cells:
        ts, fields = load_sensors(c['sensors'])
        if not ts: continue
        for zone, extract in zone_extractors.items():
            vals = extract(fields)
            if vals is None or len(vals) != len(ts): continue
            if all(v is None for v in vals): continue
            grouped[(c['model'], zone)][c['policy']].append((ts, vals))

    def _bucket_avg(traces, bucket_s=5.0):
        if not traces: return None, None
        max_t = max(t[-1] for t, _ in traces if t)
        bins = np.arange(0, max_t + bucket_s, bucket_s)
        sums = np.zeros(len(bins))
        cnts = np.zeros(len(bins))
        for ts, vals in traces:
            for t, v in zip(ts, vals):
                if v is None: continue
                bidx = int(t // bucket_s)
                if 0 <= bidx < len(bins):
                    sums[bidx] += v
                    cnts[bidx] += 1
        avg = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        return bins, avg

    for (model, zone), polmap in grouped.items():
        plt.figure(figsize=(10, 4))
        any_data = False
        for pol in POLICY_ORDER:
            traces = polmap.get(pol, [])
            if not traces: continue
            bins, avg = _bucket_avg(traces)
            if bins is None: continue
            plt.plot(bins, avg, label=f"{pol} (n={len(traces)})",
                     color=POLICY_COLOR.get(pol, 'gray'), lw=2)
            any_data = True
        if any_data:
            plt.xlabel('time since cell start (s)')
            plt.ylabel(zone_labels[zone])
            plt.title(f'Average {zone} temperature during inference — {model}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"avg_{zone}_temp_{model}.png", dpi=120)
        plt.close()


def plot_footprint_vs_f1(cells, out_dir):
    """Pareto-style: KV footprint (vs vanilla) on x, F1 on y, one point per (model, policy, prompt)."""
    if not HAVE_MPL: return
    # group by (model, prompt) so vanilla is per-prompt reference
    by_mp = defaultdict(dict)
    for c in cells:
        m = c['meta']
        peak_kv = m.get('peak_kv_cells', 0)
        retention = m.get('mean_retention_ratio', 1.0) or 1.0
        n_steps = m.get('n_decode_steps', 0)
        if peak_kv == 0 or n_steps == 0: continue
        kv_per_token = 2 * m.get('n_layers',1) * m.get('n_kv_heads',1) * m.get('head_dim',1) * 2  # bytes
        footprint = peak_kv * retention * kv_per_token * n_steps
        by_mp[(c['model'], c['prompt_id'])][c['policy']] = (footprint, c['f1'])
    # one figure per model
    by_model_data = defaultdict(list)
    for (model, prompt), polmap in by_mp.items():
        van = polmap.get('vanilla', (1,0))
        for pol, (fp, f1) in polmap.items():
            by_model_data[model].append((pol, prompt, fp/van[0] if van[0] else 1, f1))
    for model, pts in by_model_data.items():
        plt.figure(figsize=(7,5))
        for pol in POLICY_ORDER:
            xs = [p[2] for p in pts if p[0] == pol]
            ys = [p[3] for p in pts if p[0] == pol]
            if xs:
                plt.scatter(xs, ys, label=pol, color=POLICY_COLOR.get(pol,'gray'), s=80, alpha=0.7)
        plt.xlabel("KV footprint ratio (vs vanilla, lower = better)")
        plt.ylabel("F1 score")
        plt.title(f"Quality vs. KV footprint — {model}")
        plt.xscale('log')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir/f"pareto_{model}.png", dpi=120)
        plt.close()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--truth-jsonl", default="/home/mislam22/EndurKV_workspace/prompts/prompts_pub_longbench.jsonl")
    args = ap.parse_args()

    cells = gather_cells(args.sweep_dir)
    print(f"loaded {len(cells)} cells")
    cells = attach_scores(cells, args.truth_jsonl)

    out_dir = figdir(args.sweep_dir)
    print(f"writing figures to {out_dir}")

    if HAVE_MPL:
        plot_thermal_per_cell(cells, out_dir)
        plot_power_per_cell(cells, out_dir)
        plot_kv_trajectory(cells, out_dir)
        plot_f1_bars(cells, out_dir)
        plot_footprint_vs_f1(cells, out_dir)
        plot_avg_thermal_per_model(cells, out_dir)
        print("done")
    else:
        print("matplotlib missing; only CSV summary written")

    # Also write a flat summary CSV
    csvp = out_dir / "summary.csv"
    fields = ['model','policy','prompt_id','task','f1','em','rouge','peak_kv_mb','peak_rss_kb',
              'prefill_ms','decode_tps','mean_mass_retained','mean_retention_ratio',
              'mean_eviction_efficiency','evicted_prefill','evicted_total_decode']
    with open(csvp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cells:
            row = {k: c['meta'].get(k) for k in fields if k not in ('model','policy','prompt_id','task','f1','em','rouge')}
            row.update({'model':c['model'],'policy':c['policy'],'prompt_id':c['prompt_id'],
                        'task':c['task'],'f1':c['f1'],'em':c['em'],'rouge':c['rouge']})
            w.writerow(row)
    print(f"wrote {csvp}")


if __name__ == "__main__":
    main()
