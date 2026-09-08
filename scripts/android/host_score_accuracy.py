"""Accuracy harness — score model outputs (from per-step entropy.csv) against
ground_truth in the prompt jsonl files.

For each prompt:
  1. Reconstruct the model's generated text from <prompt_id>.entropy.csv
     (each row has token_id + token_text, concatenated in order)
  2. Apply task-specific answer extraction
  3. Score vs prompts/<file>.jsonl's ground_truth using the appropriate metric

Per-task metrics (matches LongBench / GSM8K convention):
  gsm8k, aime24                    : exact_match on extracted boxed/last-number
  niah                             : substring match for the needle phrase
  qasper, narrativeqa, hotpotqa,   : token-level F1 (set-overlap on lowercase
   multifieldqa_en, 2wikimqa,        tokenized words, modulo stopwords/punct)
   musique, triviaqa
  gov_report, qmsum, multi_news,   : ROUGE-L (simple LCS-based, no external dep)
   samsum, xsum, cnn_dailymail
  trec                             : exact_match on first category token
  lcc, repobench-p                 : edit_similarity (1 - lev/maxlen, like LongBench)

Outputs:
  logs/_accuracy/<study_dir>_scores.csv   per-prompt scores
  logs/_accuracy/<study_dir>_summary.csv  per-task aggregates (mean + 95% CI)
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import string
import sys
from pathlib import Path
from collections import defaultdict

WORKSPACE = Path(os.environ.get("WORKSPACE", str(Path(__file__).resolve().parents[3])))
LOGS = WORKSPACE / "logs"


# ----- answer extractors -------------------------------------------------------

_BOX_RE  = re.compile(r"\\boxed\{([^}]+)\}")
_NUM_RE  = re.compile(r"-?\d+(?:\.\d+)?")

def extract_boxed_or_last_number(text: str) -> str:
    """For GSM8K / AIME — extract \\boxed{} if present, else last number."""
    m = _BOX_RE.search(text)
    if m:
        return m.group(1).strip()
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1]
    return text.strip().split()[0] if text.strip() else ""


def extract_first_line(text: str) -> str:
    """For QA / classification — output until first newline."""
    return text.strip().split("\n")[0].strip()


def normalize_answer(s: str) -> str:
    """Lower, strip punctuation, collapse whitespace (LongBench convention)."""
    s = s.lower()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----- metrics -----------------------------------------------------------------

def exact_match(pred: str, gold: str | list) -> float:
    """1.0 if normalized pred equals any normalized gold (gold can be list)."""
    p = normalize_answer(pred)
    if isinstance(gold, str):
        gold = [gold]
    for g in gold:
        if p == normalize_answer(str(g)):
            return 1.0
    return 0.0


def token_f1(pred: str, gold: str | list) -> float:
    if isinstance(gold, str):
        gold = [gold]
    pt = normalize_answer(pred).split()
    best = 0.0
    for g in gold:
        gt = normalize_answer(str(g)).split()
        if not pt and not gt: best = max(best, 1.0); continue
        if not pt or not gt: continue
        common = {}
        for w in pt: common[w] = common.get(w, 0) + 1
        overlap = 0
        for w in gt:
            if common.get(w, 0) > 0:
                overlap += 1
                common[w] -= 1
        if overlap == 0: continue
        prec = overlap / len(pt)
        rec  = overlap / len(gt)
        f1 = 2 * prec * rec / (prec + rec)
        best = max(best, f1)
    return best


def rouge_l(pred: str, gold: str | list) -> float:
    """Simple ROUGE-L via LCS on whitespace tokens, F1 form."""
    if isinstance(gold, str):
        gold = [gold]
    pt = normalize_answer(pred).split()
    best = 0.0
    for g in gold:
        gt = normalize_answer(str(g)).split()
        if not pt or not gt: continue
        # LCS length via O(mn) DP
        m, n = len(pt), len(gt)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pt[i-1] == gt[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        if lcs == 0: continue
        prec = lcs / m
        rec  = lcs / n
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def edit_similarity(pred: str, gold: str | list) -> float:
    """1 - lev / maxlen on raw strings (LongBench code-completion metric)."""
    if isinstance(gold, str):
        gold = [gold]
    pred = pred.strip()
    best = 0.0
    for g in gold:
        g = str(g).strip()
        if not pred and not g: best = max(best, 1.0); continue
        if not pred or not g: continue
        # Levenshtein, char-level
        m, n = len(pred), len(g)
        if m * n > 200_000:  # cap to avoid O(N^2) blow-up
            pred = pred[:300]; g = g[:300]
            m, n = len(pred), len(g)
        prev = list(range(n + 1))
        cur  = [0] * (n + 1)
        for i in range(1, m + 1):
            cur[0] = i
            for j in range(1, n + 1):
                cost = 0 if pred[i-1] == g[j-1] else 1
                cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
            prev, cur = cur, prev
        lev = prev[n]
        best = max(best, 1.0 - lev / max(m, n))
    return best


def niah_match(pred: str, gold: str | list) -> float:
    """Needle in a Haystack — 1.0 if gold substring appears (case-insensitive) in pred."""
    if isinstance(gold, str):
        gold = [gold]
    p = pred.lower()
    for g in gold:
        if str(g).lower() in p:
            return 1.0
    return 0.0


TASK_METRIC = {
    "gsm8k":           ("exact_match_boxed", extract_boxed_or_last_number, exact_match),
    "aime24":          ("exact_match_boxed", extract_boxed_or_last_number, exact_match),
    "niah":            ("substring_match",   lambda x: x,                  niah_match),
    "qasper":          ("token_f1",          extract_first_line,           token_f1),
    "narrativeqa":     ("token_f1",          extract_first_line,           token_f1),
    "hotpotqa":        ("token_f1",          extract_first_line,           token_f1),
    "multifieldqa_en": ("token_f1",          extract_first_line,           token_f1),
    "2wikimqa":        ("token_f1",          extract_first_line,           token_f1),
    "musique":         ("token_f1",          extract_first_line,           token_f1),
    "triviaqa":        ("token_f1",          extract_first_line,           token_f1),
    "gov_report":      ("rouge_l",           lambda x: x,                  rouge_l),
    "qmsum":           ("rouge_l",           lambda x: x,                  rouge_l),
    "multi_news":      ("rouge_l",           lambda x: x,                  rouge_l),
    "samsum":          ("rouge_l",           lambda x: x,                  rouge_l),
    "xsum":            ("rouge_l",           lambda x: x,                  rouge_l),
    "cnn_dailymail":   ("rouge_l",           lambda x: x,                  rouge_l),
    "trec":            ("exact_match",       extract_first_line,           exact_match),
    "lcc":             ("edit_similarity",   lambda x: x,                  edit_similarity),
    "repobench-p":     ("edit_similarity",   lambda x: x,                  edit_similarity),
}


# ----- data loading -----------------------------------------------------------

def load_ground_truth(prompt_jsonl: Path) -> dict[str, dict]:
    """Returns {prompt_id: {"task": ..., "ground_truth": ...}}"""
    gt = {}
    with open(prompt_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            pid = r.get("prompt_id")
            task = r.get("task")
            ground = r.get("ground_truth")
            if pid is None: continue
            gt[pid] = {"task": task, "ground_truth": ground}
    return gt


def reconstruct_output(entropy_csv: Path) -> str:
    """Concatenate per-step token_text from the entropy probe CSV."""
    text_parts = []
    with open(entropy_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("token_text", "")
            # CSV escape used probe-side wraps strings in quotes; csv.DictReader already strips them.
            text_parts.append(t)
    return "".join(text_parts)


# ----- scoring -----------------------------------------------------------------

def score_study_dir(study_dir: Path, gt: dict, out_dir: Path) -> tuple[Path, Path]:
    """Score all prompts in a study dir; write per-prompt + per-task CSVs."""
    rows = []
    for ent_csv in sorted(study_dir.glob("*.entropy.csv")):
        pid = ent_csv.name[:-len(".entropy.csv")]
        if pid not in gt:
            continue
        task = gt[pid]["task"]
        ground = gt[pid]["ground_truth"]
        if task not in TASK_METRIC:
            continue
        metric_name, extractor, scorer = TASK_METRIC[task]
        try:
            output_text = reconstruct_output(ent_csv)
        except Exception as e:
            rows.append({"prompt_id": pid, "task": task, "metric": metric_name,
                         "score": None, "extracted": "", "error": str(e)})
            continue
        extracted = extractor(output_text)
        if ground is None:
            rows.append({"prompt_id": pid, "task": task, "metric": metric_name,
                         "score": None, "extracted": extracted, "error": "no ground_truth"})
            continue
        score = scorer(extracted, ground)
        rows.append({"prompt_id": pid, "task": task, "metric": metric_name,
                     "score": score, "extracted": extracted[:200], "error": ""})

    out_dir.mkdir(parents=True, exist_ok=True)
    per_prompt = out_dir / f"{study_dir.name}_scores.csv"
    with open(per_prompt, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "task", "metric", "score",
                                          "extracted", "error"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Per-task aggregate w/ Wilson 95% CI for binary metrics
    by_task = defaultdict(list)
    for r in rows:
        if r["score"] is None: continue
        by_task[r["task"]].append(r["score"])
    agg = []
    for task, scores in sorted(by_task.items()):
        n = len(scores)
        mean = sum(scores) / n
        # 95% normal CI via t-approx (good for n >= 20; OK to report on n>=10 with caveat)
        if n > 1:
            var = sum((s - mean) ** 2 for s in scores) / (n - 1)
            se = (var / n) ** 0.5
            ci_lo = max(0.0, mean - 1.96 * se)
            ci_hi = min(1.0, mean + 1.96 * se)
        else:
            ci_lo = ci_hi = mean
        agg.append({"task": task, "n": n, "metric": TASK_METRIC[task][0],
                    "mean": round(mean, 4), "ci_lo": round(ci_lo, 4),
                    "ci_hi": round(ci_hi, 4)})
    summary = out_dir / f"{study_dir.name}_summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "n", "metric", "mean", "ci_lo", "ci_hi"])
        w.writeheader()
        for r in agg:
            w.writerow(r)
    return per_prompt, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", required=True,
                    help="Path to logs/study_phone_<...>/ containing *.entropy.csv files")
    ap.add_argument("--prompts", required=True, nargs="+",
                    help="One or more prompt jsonl files providing ground_truth")
    ap.add_argument("--out-dir", default=str(LOGS / "_accuracy"))
    args = ap.parse_args()

    study_dir = Path(args.study_dir)
    if not study_dir.is_dir():
        print(f"ERROR: {study_dir} not a dir", file=sys.stderr); return 1

    gt = {}
    for p in args.prompts:
        gt.update(load_ground_truth(Path(p)))
    print(f"[score] loaded {len(gt)} ground_truth entries from {len(args.prompts)} files")

    per_prompt, summary = score_study_dir(study_dir, gt, Path(args.out_dir))
    print(f"[score] per-prompt: {per_prompt}")
    print(f"[score] summary:    {summary}")
    with open(summary) as f:
        for line in f: print("  " + line.rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
