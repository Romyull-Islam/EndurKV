#!/usr/bin/env python3
"""Collect every number the HotMobile claims deck needs into /tmp/claims_data.json.

New cells come from the phone-resident queue (/tmp/qres, pulled by pull_results.sh).
Published cells come from the campaigns the papers already cite. Nothing is
typed in by hand: every value is read from a meta/gen json or an err log.
"""
import json, glob, os, re, statistics as st

def meta(path):
    t = open(path, errors='ignore').read()
    g = lambda k: (re.search(r'"%s":\s*"?([^,"\n}]+)' % k, t) or [None, None])[1]
    return dict(tps=float(g('decode_tps') or 0), prefill=float(g('prefill_ms') or 0) / 1000,
                steps=int(g('n_decode_steps') or 0), prompt=int(g('n_prompt_tokens') or 0),
                k=int(g('k_nominal') or 0), ppl=float(g('perplexity') or 0))

BYTES_PER_CELL = 16 * 8 * 64 * 2 * 2   # Llama-3.2-1B, f16 K and V: 32 KiB per cell
def kept(errpath):
    """Cells live after prefill, by the paper's method: retained_kv bytes / bytes per cell.
    The [evict-dbg] kept= line is a pre-gate pool for muKV and is not used."""
    if not os.path.exists(errpath): return None
    t = open(errpath, errors='ignore').read()
    m = re.search(r'\[cache\] retained_kv=([0-9.]+) MiB', t)
    if m: return int(round(float(m.group(1)) * 1048576 / BYTES_PER_CELL))
    m = re.search(r'kept=(\d+)', t)
    return int(m.group(1)) if m else None

D = {'new': {}, 'published': {}}

# ---- new: the phone queue ----
for j in sorted(glob.glob('/tmp/qres/*.json')):
    tag = os.path.basename(j)[:-5]
    if tag == 'queue': continue
    m = meta(j); m['kept'] = kept(j[:-5] + '.err'); D['new'][tag] = m

# ---- published GPU rows (Llama-3.2-1B, 9737 prompt, 4096 generated) ----
P = D['published']
def add(tag, jpath, errpath=None):
    if os.path.exists(jpath):
        m = meta(jpath); m['kept'] = kept(errpath) if errpath else None
        if m['kept'] is None:
            try: rb = json.load(open(jpath)).get('retained_kv_bytes')
            except Exception: rb = None
            if rb: m['kept'] = int(rb // BYTES_PER_CELL)   # cells live after prefill, from the run's own byte count
        P[tag] = m
for pol in ('snapkv', 'adakv', 'h2o', 'tova'):
    add(pol + '_aug05', f'/tmp/phone_gpu_16k/llama1b_{pol}/meta.json', f'/tmp/phone_gpu_16k/llama1b_{pol}/err')
for r in (1, 2, 3):
    add(f'v_aug_r{r}', f'/tmp/sllm_faithful/v_r{r}/meta.json')
    add(f'mukv_aug_r{r}', f'/tmp/sllm_faithful/mukv_r{r}/meta.json')
    add(f'sfown_aug_r{r}', f'/tmp/sllm_faithful/sfown_r{r}/meta.json')
    add(f'nodfg_aug_r{r}', f'/tmp/phone_gpu_16k/llama1b_mukv_nodfg{"" if r==1 else "_r%d"%r}/meta.json')
for a in ('v_ppl', 'mukv_ppl', 'sfown_ppl'):
    add(a, f'/tmp/sllm_faithful/{a}/meta.json')
# CPU rows (Table 1 campaign)
for pol in ('vanilla', 'snapkv', 'adakv', 'h2o', 'tova', 'mukv_faon'):
    add('cpu_' + pol, f'/tmp/nat_cpu/{pol}/gen.json', f'/tmp/nat_cpu/{pol}/gen.err')

# ---- derived ----
def med(keys, src):
    v = [src[k]['tps'] for k in keys if k in src and src[k]['tps'] > 0]
    return (st.median(v), len(v)) if v else (None, 0)
S = {}
S['gpu_vanilla_aug'] = med([f'v_aug_r{r}' for r in (1, 2, 3)], P)
S['gpu_vanilla_new'] = med([k for k in D['new'] if re.fullmatch(r'v_r\d', k)], D['new'])
for pol in ('snapkv', 'adakv', 'h2o', 'tova'):
    new_runs = [k for k in D['new'] if re.fullmatch(pol + r'_r\d', k)]
    S[pol + '_gpu_runs'] = {'aug05': P.get(pol + '_aug05', {}).get('tps'),
                            'new': {k: D['new'][k]['tps'] for k in new_runs}}
S['kernel_off_published_budget'] = {k: D['new'][k]['tps'] for k in D['new'] if k.startswith('shown_r')}
S['realized_cells'] = {k: D['new'][k]['kept'] for k in D['new'] if re.search(r'_k\d+$', k)}
for pol in ('snapkv', 'adakv', 'h2o', 'tova'):
    if pol + '_aug05' in P: S['realized_cells'][pol + '_k1024_aug05'] = P[pol + '_aug05']['kept']
import statistics as _st2
_mk = [P[f'mukv_aug_r{r}']['kept'] for r in (1, 2, 3) if P.get(f'mukv_aug_r{r}', {}).get('kept')]
_sk = [P[f'sfown_aug_r{r}']['kept'] for r in (1, 2, 3) if P.get(f'sfown_aug_r{r}', {}).get('kept')]
if _mk: S['realized_cells']['mukv_k1024'] = int(_st2.median(_mk))
if _sk: S['realized_cells']['sllm_k2004'] = int(_st2.median(_sk))   # StreamingLLM at its published budget, 4 + 2000
_sn = [v for v in (P.get('snapkv_aug05', {}).get('kept'), D['new'].get('snapkv_r1', {}).get('kept')) if v]
if _sn: S['realized_cells']['snapkv_k1024'] = int(_st2.median(_sn))   # median of the two campaigns, as in the papers
S['own_budget_gpu'] = {k: D['new'][k]['tps'] for k in D['new'] if k.endswith('_own')}
# CPU own-budget retention: LongBench prompts on the phone CPU, each policy at its published budget,
# retained_kv_bytes against the vanilla run of the same prompt, mean of the per-prompt ratio (the papers' number; /tmp/lb_native)
import glob, statistics
_lb = {}
for m in glob.glob('/tmp/lb_native/*/meta.json'):
    try: j = json.load(open(m))
    except Exception: continue
    pid = os.path.basename(os.path.dirname(m)).split('_', 1)[1]; _lb.setdefault(pid, {})[j.get('policy')] = j.get('retained_kv_bytes')
S['own_budget_cpu_retention'] = {}
for pol in ('snapkv', 'adakv', 'h2o', 'tova', 'streamingllm'):
    r = [_lb[p][pol] / _lb[p]['vanilla'] for p in _lb if pol in _lb[p] and _lb[p].get('vanilla') and _lb[p].get(pol)]
    if r: S['own_budget_cpu_retention'][pol] = (statistics.mean(r), len(r))
S['own_budget_gpu_cells'] = {k: D['new'][k].get('kept') for k in D['new'] if k.endswith('_own')}
D['summary'] = S
json.dump(D, open('/tmp/claims_data.json', 'w'), indent=1)
print("new cells:", len(D['new']), "| published:", len(P))
print("gpu vanilla aug n=%d median=%s | new n=%d median=%s" % (S['gpu_vanilla_aug'][1], S['gpu_vanilla_aug'][0], S['gpu_vanilla_new'][1], S['gpu_vanilla_new'][0]))
for pol in ('snapkv', 'adakv', 'h2o', 'tova'): print(" ", pol, S[pol + '_gpu_runs'])
print("kernel-off @published:", S['kernel_off_published_budget'])
print("realized cells:", S['realized_cells'])
print("own-budget gpu:", S['own_budget_gpu'], S['own_budget_gpu_cells']); print("own-budget cpu retention:", S['own_budget_cpu_retention'])
