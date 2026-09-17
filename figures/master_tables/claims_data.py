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

def kept(errpath):
    if not os.path.exists(errpath): return None
    m = re.search(r'kept=(\d+)', open(errpath, errors='ignore').read())
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
        m = meta(jpath); m['kept'] = kept(errpath) if errpath else None; P[tag] = m
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
S['own_budget_gpu'] = {k: D['new'][k]['tps'] for k in D['new'] if k.endswith('_own')}
D['summary'] = S
json.dump(D, open('/tmp/claims_data.json', 'w'), indent=1)
print("new cells:", len(D['new']), "| published:", len(P))
print("gpu vanilla aug n=%d median=%s | new n=%d median=%s" % (S['gpu_vanilla_aug'][1], S['gpu_vanilla_aug'][0], S['gpu_vanilla_new'][1], S['gpu_vanilla_new'][0]))
for pol in ('snapkv', 'adakv', 'h2o', 'tova'): print(" ", pol, S[pol + '_gpu_runs'])
print("kernel-off @published:", S['kernel_off_published_budget'])
print("realized cells:", S['realized_cells'])
print("own-budget gpu:", S['own_budget_gpu'])
