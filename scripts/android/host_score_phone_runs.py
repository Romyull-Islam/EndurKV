#!/usr/bin/env python3
"""Direct accuracy scoring for phone-pulled `gen.txt` outputs of eviction_bench.

For each run directory (containing meta.json + gen.txt), computes:
  - F1 (token-overlap, LongBench-standard)
  - Exact-Match (case-insensitive, post-normalization)
  - ROUGE-L (longest common subsequence)
  - Needle-recall (NIAH-specific)

Ground truth comes from the LongBench/NIAH jsonl on host.

Usage:
    host_score_phone_runs.py --runs-dir <pulled-phone-logs> [--out-csv path]
"""
import argparse, json, re, string
from pathlib import Path
from collections import Counter
import pandas as pd


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p = normalize_text(pred).split(); g = normalize_text(gold).split()
    if not p or not g: return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0: return 0.0
    pr = n / len(p); rc = n / len(g)
    return 2 * pr * rc / (pr + rc)


def em_score(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def rouge_l(pred: str, gold: str) -> float:
    p = normalize_text(pred).split(); g = normalize_text(gold).split()
    if not p or not g: return 0.0
    m, n = len(p), len(g)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if p[i-1] == g[j-1]: dp[i][j] = dp[i-1][j-1] + 1
            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0: return 0.0
    pr = lcs / m; rc = lcs / n
    return 2 * pr * rc / (pr + rc)


PRIMARY = {
    'narrativeqa':'f1','qasper':'f1','hotpotqa':'f1','multifieldqa_en':'f1',
    '2wikimqa':'f1','musique':'f1','triviaqa':'f1',
    'gov_report':'rouge_l','qmsum':'rouge_l','multi_news':'rouge_l','samsum':'rouge_l',
    'trec':'em','passage_retrieval_en':'em','passage_count':'em',
    'lcc':'em','repobench-p':'em','gsm8k':'em',
    'niah':'needle',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--truth", default="/home/mislam22/EndurKV_workspace/prompts/prompts_pub_longbench.jsonl")
    ap.add_argument("--niah-truth", default="/home/mislam22/EndurKV_workspace/prompts/prompts_pub_niah.jsonl")
    ap.add_argument("--out-csv", default="phone_accuracy.csv")
    args = ap.parse_args()

    # truth lookup
    truth = {}
    for p in [args.truth, args.niah_truth]:
        if Path(p).exists():
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    truth[r['prompt_id']] = {
                        'task': r['task'],
                        'gold': r.get('ground_truth', []),
                        'needle': (r.get('_meta', {}) or {}).get('needle', ''),
                    }

    rows = []
    for meta_p in sorted(Path(args.runs_dir).rglob("meta.json")):
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            continue
        gen_p = meta_p.parent / "gen.txt"
        gen = gen_p.read_text(errors='replace').strip() if gen_p.exists() else ""
        pid = meta.get('prompt_id', '')
        info = truth.get(pid)
        if not info:
            rows.append({**meta, 'gen_first_200': gen[:200], 'task':'?',
                         'has_truth': False, 'f1':0.0, 'em':0, 'rouge_l':0.0,
                         'needle_recall':0, 'primary_metric':'?', 'primary_score':0.0})
            continue
        gold_list = info['gold']
        if isinstance(gold_list, str): gold_list = [gold_list]
        f1 = max((f1_score(gen, g) for g in gold_list), default=0.0)
        em = max((em_score(gen, g) for g in gold_list), default=0)
        rg = max((rouge_l(gen, g) for g in gold_list), default=0.0)
        nh = int(info['needle'].lower() in gen.lower()) if info['needle'] else 0
        primary = PRIMARY.get(info['task'], 'f1')
        score = {'f1':f1, 'em':em, 'rouge_l':rg, 'needle':nh}[primary]
        rows.append({**meta, 'gen_first_200': gen[:200],
                     'task': info['task'], 'has_truth': True,
                     'f1':f1, 'em':em, 'rouge_l':rg, 'needle_recall':nh,
                     'primary_metric': primary, 'primary_score': score})

    if not rows:
        print(f"no runs found in {args.runs_dir}"); return 1
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(df)} rows, {df.policy.nunique()} policies)")

    # Per-policy headline
    print("\n=== ACCURACY by policy (mean across prompts) ===")
    agg = df.groupby('policy').agg(
        n=('prompt_id','count'),
        mean_f1=('f1','mean'),
        mean_em=('em','mean'),
        mean_rouge_l=('rouge_l','mean'),
        mean_needle=('needle_recall','mean'),
        mean_primary=('primary_score','mean'),
    ).reset_index().sort_values('mean_primary', ascending=False)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 220)
    print(agg.to_string(index=False))

    # Per-(prompt, K) breakdown
    print("\n=== Per-(prompt, K) side-by-side ===")
    for key, grp in df.groupby(['prompt_id','k_nominal']):
        pid, k = key
        print(f"\n  {pid} K={k}:")
        for _, r in grp.sort_values('primary_score', ascending=False).iterrows():
            print(f"    {r['policy']:8s} primary={r['primary_score']:.3f} "
                  f"f1={r['f1']:.3f} em={r['em']} rouge={r['rouge_l']:.3f}  "
                  f"gen={r['gen_first_200'][:80]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
