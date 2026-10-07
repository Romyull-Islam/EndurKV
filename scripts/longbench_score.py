#!/usr/bin/env python3
# LongBench token-F1 scorer, a re-implementation of qa_f1_score in LongBench's metrics.py
# (normalize, token F1, max over gold answers). --strip-prefix removes a leading
# "Answer:" and is off by default because LongBench does not do it.
# Usage: longbench_score.py --runs /tmp/lb_out --gold .../gold.json
import os, re, json, string, argparse
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truths):
    """LongBench takes the max over all reference answers."""
    return max((f1_score(prediction, gt) for gt in ground_truths), default=0.0)


# eviction_bench writes raw text, so chat/EOS control tokens reach gen.txt. LongBench's
# pred.py decodes with skip_special_tokens=True, so remove them to stay on its scale.
SPECIAL = re.compile(r'<\|eot_id\|>|<\|end\|>|<\|endoftext\|>|<\|im_end\|>|'
                     r'<\|assistant\|>|<\|user\|>|</s>|<end_of_turn>|<bos>|<eos>')


def clean(pred, strip_prefix):
    # Cut at the first special token instead of replacing it, so the newline that ends
    # the answer line survives for the truncation below.
    m = SPECIAL.search(pred)
    if m:
        pred = pred[:m.start()]
    pred = pred.strip()
    if strip_prefix:
        pred = re.sub(r'^\s*(answer\s*:)\s*', '', pred, flags=re.I)
    # LongBench truncates the prediction at the first newline for QA tasks
    return pred.split("\n")[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="dir of <model>__<policy>__<task>__<id>/gen.txt")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--strip-prefix", action="store_true")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    gold = json.load(open(a.gold))
    per = {}   # (model, policy, task) -> [f1, ...]
    skipped = []
    detail = []
    for d in sorted(os.listdir(a.runs)):
        g = os.path.join(a.runs, d, "gen.txt")
        if not os.path.exists(g):
            continue
        # A crashed cell leaves an empty gen.txt but no meta.json. Skip it instead of
        # scoring F1=0, since meta.json is only written on success.
        if not os.path.exists(os.path.join(a.runs, d, "meta.json")):
            skipped.append(d)
            continue
        parts = d.split("__")
        if len(parts) != 4:
            continue
        model, policy, task, pid = parts
        if task not in gold or pid not in gold[task]:
            continue
        pred = clean(open(g, errors="replace").read(), a.strip_prefix)
        f1 = qa_f1_score(pred, gold[task][pid]["answers"])
        per.setdefault((model, policy, task), []).append(f1)
        detail.append({"model": model, "policy": policy, "task": task, "id": pid,
                       "f1": round(f1, 4), "pred": pred[:120],
                       "gold": gold[task][pid]["answers"]})

    print("%-10s %-10s %-16s %5s  %6s" % ("model", "policy", "task", "n", "F1"))
    rows = []
    for k in sorted(per):
        v = per[k]
        rows.append({"model": k[0], "policy": k[1], "task": k[2],
                     "n": len(v), "f1": round(100.0 * sum(v) / len(v), 2)})
        print("%-10s %-10s %-16s %5d  %6.2f" % (k[0], k[1], k[2], len(v), 100.0 * sum(v) / len(v)))
    if skipped:
        print("\n%d cell(s) SKIPPED (no meta.json = crashed/incomplete, not scored):" % len(skipped))
        for d in skipped[:8]:
            print("   ", d)
        if len(skipped) > 8:
            print("    ... and %d more" % (len(skipped) - 8))
    if a.json_out:
        json.dump({"rows": rows, "detail": detail}, open(a.json_out, "w"), indent=1)
        print("->", a.json_out)
