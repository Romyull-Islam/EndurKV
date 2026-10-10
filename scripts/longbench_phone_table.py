#!/usr/bin/env python3
"""Phone LongBench table (Table 3): token F1 per policy and a paired bootstrap against the full cache.
Scores the prompts on which every policy gives a non-empty answer; F1 is the mean of the task means.
Usage: longbench_phone_table.py --runs <dir of <policy>_<task>_<id>/gen.txt> --gold benchmarks/longbench/gold.json
       [--extra NAME=<dir of streamingllm_<task>_<id>.gen>]  # score another run of one policy on the same prompts
"""
import argparse, json, os, random, statistics as st
import longbench_score as L

POLICIES = ['vanilla', 'mukv', 'snapkv', 'adakv', 'tova', 'h2o', 'keydiff', 'streamingllm']


def read(path):
    return open(path, errors='ignore').read() if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--extra', action='append', default=[], help='NAME=DIR with streamingllm_<task>_<id>.gen files')
    ap.add_argument('--boot', type=int, default=10000)
    a = ap.parse_args()
    gold = json.load(open(a.gold))
    keys = sorted({(d.rsplit('_', 2)[1], d.rsplit('_', 1)[1])
                   for d in os.listdir(a.runs) if d.startswith('streamingllm_')})
    gen = lambda p, t, i: read(os.path.join(a.runs, f'{p}_{t}_{i}', 'gen.txt'))
    sel = [(t, i) for t, i in keys
           if all((g := gen(p, t, i)) is not None and L.clean(g, False).strip() for p in POLICIES)]
    f1 = lambda o, t, i: L.qa_f1_score(L.clean(o or '', False), gold[t][i]['answers'])
    scores = {p: {(t, i): f1(gen(p, t, i), t, i) for t, i in sel} for p in POLICIES}
    for spec in a.extra:
        name, d = spec.split('=', 1)
        scores[name] = {(t, i): f1(read(os.path.join(d, f'streamingllm_{t}_{i}.gen')), t, i) for t, i in sel}
    tasks = sorted({t for t, _ in sel})
    by = {t: [k for k in sel if k[0] == t] for t in tasks}
    agg = lambda sc, ks: 100 * st.mean(st.mean(sc[k] for k in ks[t]) for t in tasks)
    print(f'{len(sel)} prompts scored by every policy, tasks: {", ".join(tasks)}')
    random.seed(0)
    for p, sc in scores.items():
        line = f'{p:16s} F1 {agg(sc, by):5.1f}'
        if p != 'vanilla':
            d0 = agg(sc, by) - agg(scores['vanilla'], by)
            ds = sorted(agg(sc, rs) - agg(scores['vanilla'], rs)
                        for rs in ({t: [random.choice(by[t]) for _ in by[t]] for t in tasks}
                                   for _ in range(a.boot)))
            lo, hi = ds[int(.025 * a.boot)], ds[int(.975 * a.boot)]
            line += f'  gap {d0:+5.1f}  95% CI [{lo:+5.1f}, {hi:+5.1f}]'
        print(line)


if __name__ == '__main__':
    main()
