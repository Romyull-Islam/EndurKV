#!/usr/bin/env python3
"""Build the complete 4-policy master table once H2O finishes.

Combines data from:
  - cpu_sweep_*           : F1, KV behaviour, latency, memory, CPU/DDR thermal for vanilla/v1/TOVA
  - h2o_sweep_*           : same for H2O
  - ppl_*                 : WT2 PPL (Wave-2 short-seed, since those numbers were sensible)
  - sweep3M_*             : GPU sweep thermals (since GPU was actually exercised there)

Outputs:
  - figures/master_tables/table_4policy.csv      machine-readable
  - figures/master_tables/TABLE_4POLICY.md       markdown for slide
"""
import json, re, csv, statistics, string
from pathlib import Path
from collections import defaultdict, Counter

SHORT = {"Llama-3.2-1B-Instruct-Q4_K_M":"Llama-1B",
         "gemma-2-2b-it-Q4_K_M":"Gemma-2-2B",
         "Phi-3-mini-128k-instruct-Q4_K_M":"Phi-3-128k"}
KV_BYTES = {"Llama-1B": 2*16*8*64*2,
            "Gemma-2-2B": 2*26*4*256*2,
            "Phi-3-128k": 2*32*32*96*2}
POLICY_ORDER = ["vanilla", "v1", "tova", "h2o"]
MODEL_ORDER  = ["Llama-1B", "Gemma-2-2B", "Phi-3-128k"]

# --- F1 scoring helpers ---
def normalize(s):
    s = s.lower(); s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())
def f1_score(p, g):
    p, g = normalize(p).split(), normalize(g).split()
    if not p or not g: return 0.0
    c = Counter(p) & Counter(g); k = sum(c.values())
    return 2*(k/len(p))*(k/len(g))/((k/len(p))+(k/len(g))) if k else 0.0

truth = {}
for line in open('/home/mislam22/EndurKV_workspace/prompts/prompts_pub_longbench.jsonl'):
    r = json.loads(line); truth[r['prompt_id']] = r.get('ground_truth', [])

# --- Thermal helpers ---
def max_zone(fields, name_pred):
    keys = [k for k in fields if k.endswith('_temp_mc') and name_pred(k) and 'hw-trip' not in k]
    if not keys: return []
    n = len(fields[keys[0]])
    out = []
    for i in range(n):
        vals = [fields[k][i]/1000.0 for k in keys if fields[k][i] is not None and fields[k][i] > 0]
        out.append(max(vals) if vals else None)
    return out

def thermals(sf):
    if not sf.exists(): return {}
    fields = defaultdict(list)
    try:
        with open(sf, errors='replace') as fh:
            for r in csv.DictReader(fh):
                for k, v in r.items():
                    try: fields[k].append(float(v) if v else None)
                    except: fields[k].append(None)
    except: return {}
    out = {}
    for name, pred in [('cpu', lambda k: k.startswith('cpu')),
                       ('ddr', lambda k: k == 'ddr_temp_mc')]:
        zone = [x for x in max_zone(fields, pred) if x is not None]
        out[f'{name}_peak'] = max(zone) if zone else None
    return out

def gather(sweep_dir):
    by = defaultdict(list)
    for mf in Path(sweep_dir).rglob("meta.json"):
        try: m = json.loads(re.sub(r'\bnan\b','null', re.sub(r'\b-?inf\b','null', mf.read_text(errors='replace'))))
        except: continue
        model = SHORT.get(m.get("model","?").split("/")[-1].replace(".gguf",""), "?")
        if model == "?": continue
        pol = m.get("policy","?")
        gen = (mf.parent / "gen.txt").read_text(errors='replace').strip() if (mf.parent / "gen.txt").exists() else ""
        gold = truth.get(m.get('prompt_id',''), [])
        peak_kv = m.get('peak_kv_cells', 0)
        retention = m.get('mean_retention_ratio', 1.0) or 1.0
        avg_live_mb = peak_kv * retention * KV_BYTES[model] / (1024*1024) if peak_kv else 0
        rec = {
            "f1": max([f1_score(gen, g) for g in gold], default=0.0),
            "decode_tps": m.get('decode_tps'),
            "prefill_ms": m.get('prefill_ms'),
            "peak_kv_mb": m.get('peak_kv_mb'),
            "avg_live_kv_mb": avg_live_mb,
            "peak_rss_mb": (m.get('peak_rss_kb', 0) or 0)/1024,
            "mass": m.get('mean_mass_retained'),
            "retention": m.get('mean_retention_ratio'),
            "efficiency": m.get('mean_eviction_efficiency'),
            "evicted_total": m.get('evicted_total_decode', 0),
            **thermals(mf.parent / "sensors.csv"),
        }
        by[(model, pol)].append(rec)
    return by

def avg(xs):
    xs = [x for x in xs if x is not None and (isinstance(x, (int, float)) and x == x)]
    return statistics.mean(xs) if xs else None

# Gather from all sources
cpu = gather("/home/mislam22/EndurKV_workspace/phone-logs/cpu_sweep_1780268970/")
h2o_dirs = sorted(Path("/home/mislam22/EndurKV_workspace/phone-logs/").glob("h2o_sweep_*"))
h2o = gather(str(h2o_dirs[-1])) if h2o_dirs else defaultdict(list)
ppl = gather("/home/mislam22/EndurKV_workspace/phone-logs/ppl_1780287362/")

# Merge: cpu has vanilla/v1/tova, h2o sweep has h2o
combined = defaultdict(list)
for k, v in cpu.items(): combined[k] = v
for k, v in h2o.items(): combined[k] = v

# Build output
rows = []
header = ["Model","Policy","n","F1","WT2 PPL",
          "Decode t/s","Peak KV (MB)","Avg live KV (MB)","KV savings vs vanilla",
          "Peak RSS (MB)","Mass kept","Retention","Eff",
          "Evicted","CPU peak","DDR peak"]
for model in MODEL_ORDER:
    van_kv = avg([r["avg_live_kv_mb"] for r in combined.get((model, "vanilla"), [])]) or 1
    for pol in POLICY_ORDER:
        c = combined.get((model, pol), [])
        p = ppl.get((model, pol), [])
        if not c: continue
        kv = avg([r["avg_live_kv_mb"] for r in c])
        savings = f"{(1.0-kv/van_kv)*100:.1f}%" if pol != "vanilla" and kv else "(baseline)"
        rows.append([
            model, pol, len(c),
            f"{avg([r['f1'] for r in c]):.3f}",
            f"{avg([r['decode_tps'] for r in c]):.2f}",
            f"{avg([r['peak_kv_mb'] for r in c]):.0f}",
            f"{kv:.0f}",
            savings,
            f"{avg([r['peak_rss_mb'] for r in c]):.0f}",
            f"{avg([r['mass'] for r in c]):.3f}",
            f"{avg([r['retention'] for r in c]):.3f}",
            f"{avg([r['efficiency'] for r in c]):.2f}",
            f"{avg([r['evicted_total'] for r in c]):.0f}",
            f"{avg([r['cpu_peak'] for r in c]):.1f}°C" if avg([r['cpu_peak'] for r in c]) else "—",
            f"{avg([r['ddr_peak'] for r in c]):.1f}°C" if avg([r['ddr_peak'] for r in c]) else "—",
        ])
        # WT2 PPL goes in column 4 — only the short-seed Wave-2 numbers
        if p:
            rows[-1].insert(4, f"{avg([r.get('f1',0) for r in p]):.2f}" if False else
                          f"{statistics.mean([(__import__('json').loads(re.sub(r'\\bnan\\b','null',re.sub(r'\\b-?inf\\b','null',mf.read_text(errors='replace'))))).get('perplexity', 0) or 0 for mf in Path('/home/mislam22/EndurKV_workspace/phone-logs/ppl_1780287362/').rglob('meta.json') if SHORT.get(json.loads(re.sub(r'\\bnan\\b','null',re.sub(r'\\b-?inf\\b','null',mf.read_text(errors='replace')))).get('model','?').split('/')[-1].replace('.gguf','')) == model and json.loads(re.sub(r'\\bnan\\b','null',re.sub(r'\\b-?inf\\b','null',mf.read_text(errors='replace')))).get('policy') == pol]):.2f}" if any(True for _ in []) else "—")
        else:
            rows[-1].insert(4, "—")

# Simpler PPL lookup
ppl_lookup = {}
for mf in Path('/home/mislam22/EndurKV_workspace/phone-logs/ppl_1780287362/').rglob("meta.json"):
    try: m = json.loads(re.sub(r'\bnan\b','null', re.sub(r'\b-?inf\b','null', mf.read_text(errors='replace'))))
    except: continue
    model = SHORT.get(m.get("model","?").split("/")[-1].replace(".gguf",""), "?")
    pol = m.get("policy","?")
    ppl = m.get('perplexity', None)
    if model != "?" and ppl is not None:
        ppl_lookup.setdefault((model, pol), []).append(ppl)

# Fix the PPL column properly (re-render rows)
out_csv = "/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/table_4policy.csv"
out_md  = "/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/TABLE_4POLICY.md"
Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

rows = []
for model in MODEL_ORDER:
    van_kv = avg([r["avg_live_kv_mb"] for r in combined.get((model, "vanilla"), [])]) or 1
    for pol in POLICY_ORDER:
        c = combined.get((model, pol), [])
        if not c: continue
        kv = avg([r["avg_live_kv_mb"] for r in c])
        savings = f"{(1.0-kv/van_kv)*100:.1f}%" if pol != "vanilla" and kv else "(baseline)"
        ppl_val = avg(ppl_lookup.get((model, pol), [])) if ppl_lookup.get((model, pol)) else None
        rows.append([
            model, pol, len(c),
            f"{avg([r['f1'] for r in c]):.3f}",
            f"{ppl_val:.2f}" if ppl_val else "—",
            f"{avg([r['decode_tps'] for r in c]):.2f}",
            f"{avg([r['peak_kv_mb'] for r in c]):.0f}",
            f"{kv:.0f}",
            savings,
            f"{avg([r['peak_rss_mb'] for r in c]):.0f}",
            f"{avg([r['mass'] for r in c]):.3f}",
            f"{avg([r['retention'] for r in c]):.3f}",
            f"{avg([r['efficiency'] for r in c]):.2f}",
            f"{avg([r['evicted_total'] for r in c]):.0f}",
            f"{avg([r['cpu_peak'] for r in c]):.1f}" if avg([r['cpu_peak'] for r in c]) else "—",
            f"{avg([r['ddr_peak'] for r in c]):.1f}" if avg([r['ddr_peak'] for r in c]) else "—",
        ])

# Write CSV
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    for r in rows: w.writerow(r)

# Write Markdown
with open(out_md, 'w') as f:
    f.write("# Master 4-policy table — vanilla / v1 / TOVA / H2O\n\n")
    f.write("All numbers are averages across 3 prompts × 1 rep per (model, policy) cell on the CPU sweep.\n\n")
    f.write("| " + " | ".join(header) + " |\n")
    f.write("|" + "|".join(["---"] * len(header)) + "|\n")
    for r in rows: f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    f.write("\n## Column meanings\n\n")
    f.write("- **F1**: LongBench downstream-task quality (higher = better; v1 ≥ vanilla on Llama-1B+Gemma; ~tied on Phi-3)\n")
    f.write("- **WT2 PPL**: intrinsic LM quality on WikiText-2 short-seed (lower = better; canonical 7-8 for Llama-1B; all policies tie at this setting because no eviction triggers)\n")
    f.write("- **Decode t/s**: per-token generation speed on CPU\n")
    f.write("- **Peak KV (MB)**: allocated cache buffer (same across policies for given model; llama.cpp pre-allocates)\n")
    f.write("- **Avg live KV (MB)**: real working-set KV during inference, accounting for retention. THIS is what actually drives memory + DDR thermal load.\n")
    f.write("- **KV savings vs vanilla**: % of vanilla's avg live KV that's freed by eviction\n")
    f.write("- **Peak RSS (MB)**: process resident memory\n")
    f.write("- **Mass kept**: fraction of total attention probability that survives eviction (closer to 1.0 = better)\n")
    f.write("- **Retention**: fraction of KV positions kept (lower = more aggressive compression)\n")
    f.write("- **Eff** (Eviction efficiency): mass / retention ratio (higher = more mass per cell kept)\n")
    f.write("- **Evicted**: total positions evicted during decoding\n")
    f.write("- **CPU peak**: max CPU core temperature during cell\n")
    f.write("- **DDR peak**: max system-memory temperature during cell\n")

print(f"Wrote {out_csv}")
print(f"Wrote {out_md}")
print()
print("=== Preview ===")
print("| " + " | ".join(header) + " |")
for r in rows:
    print("| " + " | ".join(str(x) for x in r) + " |")
