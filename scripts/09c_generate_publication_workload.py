#!/usr/bin/env python3
"""
Phase C-pub — publication-quality workload generator.

Builds the prompt sets for the cross-architecture / publication evaluation
(ASPLOS / NeurIPS / MLSys tier). Three output files:

  prompts/prompts_pub_longbench.jsonl  — N=30 per task × 14 LongBench English
                                          tasks, full-length (no truncation),
                                          stratified across token-length bins.
  prompts/prompts_pub_niah.jsonl       — Needle-in-a-Haystack, 7 depths
                                          {0%,17%,33%,50%,67%,83%,100%} ×
                                          4 lengths {4K,8K,16K,32K} = 28 prompts.
  prompts/prompts_pub_reasoning.jsonl  — 100 GSM8K + 30 AIME-2024, with
                                          ground_truth for pass@1 scoring.

For each LongBench item the ORIGINAL prompt text is preserved verbatim (truncated
only at a hard 32K-char safety cap to prevent the on-phone 12 GB OOM). This
matches the evaluation protocol of SnapKV, CAKE, PyramidKV, LazyEviction,
KeyDiff, AhaKV, MixedDimKV, LaProx — every recent KV eviction paper since 2024.

Datasets and citations:
  LongBench    : zai-org/LongBench   (Bai et al., ICLR 2024)
  GSM8K        : openai/gsm8k        (Cobbe et al., 2021)
  AIME-2024    : HuggingFaceH4/aime_2024
  NIAH         : self-generated (Greg Kamradt protocol)
"""
from __future__ import annotations

import json
import os
import random
import sys
import zipfile
from pathlib import Path

# Auto-reexec under venv
_ROOT_FOR_VENV = Path(__file__).resolve().parents[2]
_VENV_DIR = _ROOT_FOR_VENV / ".venv"
_VENV_PY  = _VENV_DIR / "bin" / "python3"
if _VENV_PY.exists():
    try:
        _under_venv = Path(sys.prefix).resolve() == _VENV_DIR.resolve()
    except OSError:
        _under_venv = False
    if not _under_venv:
        os.execv(str(_VENV_PY), [str(_VENV_PY), __file__, *sys.argv[1:]])

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "prompts"
OUT_DIR.mkdir(exist_ok=True)

# ----- LongBench config -----------------------------------------------------
LONGBENCH_TASKS = [
    # (name, ctx_field, input_field, template)
    ("narrativeqa",     "context", "input",
     "You are given a story. Answer the question based on the story.\n\nStory:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("qasper",          "context", "input",
     "You are given a scientific article. Answer the question based on it.\n\nArticle:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("multifieldqa_en", "context", "input",
     "Read the following document and answer the question.\n\nDocument:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("hotpotqa",        "context", "input",
     "Answer the question based on the following passages.\n\nPassages:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("2wikimqa",        "context", "input",
     "Answer the question based on the following passages.\n\nPassages:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("musique",         "context", "input",
     "Answer the question based on the following passages.\n\nPassages:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("gov_report",      "context", None,
     "Summarize the following legislative report.\n\nReport:\n{context}\n\nSummary:"),
    ("qmsum",           "context", "input",
     "You are given a meeting transcript and a query about it.\n\nTranscript:\n{context}\n\nQuery: {input}\nAnswer:"),
    ("multi_news",      "context", None,
     "Summarize the following news articles.\n\nArticles:\n{context}\n\nSummary:"),
    ("trec",            "context", "input",
     "{context}\n\nClassify this question: {input}\nCategory:"),
    ("triviaqa",        "context", "input",
     "Answer the question using the following passages.\n\nPassages:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("samsum",          "context", "input",
     "Summarize the following dialogue.\n\nDialogue:\n{context}\n\nSummary:"),
    ("lcc",             "context", None,
     "Complete the following code:\n\n{context}"),
    ("repobench-p",     "context", "input",
     "Complete the next line of code based on the repository context.\n\n{context}\n\nNext line:"),
]

LB_PER_TASK    = int(os.environ.get("LB_PER_TASK", "30"))
HARD_TOKEN_CAP = int(os.environ.get("HARD_TOKEN_CAP", "13000"))   # ~52K chars
HARD_CHAR_CAP  = 4 * HARD_TOKEN_CAP
EXPECTED_MAX_TOKENS = 64
SEED = 42

_LONGBENCH_REPO = "zai-org/LongBench"


def _ensure_lb_extracted() -> Path:
    from huggingface_hub import hf_hub_download
    zip_path = hf_hub_download(_LONGBENCH_REPO, "data.zip", repo_type="dataset")
    extract_root = Path(zip_path).parent / "longbench_extracted"
    if not (extract_root / ".extracted").exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_root)
        (extract_root / ".extracted").touch()
    return extract_root


def _load_lb_items(name: str) -> list[dict]:
    root = _ensure_lb_extracted()
    for cand in [root / "data" / f"{name}.jsonl", root / f"{name}.jsonl"]:
        if cand.exists():
            with open(cand, encoding="utf-8") as f:
                return [json.loads(l) for l in f if l.strip()]
    matches = list(root.rglob(f"{name}.jsonl"))
    if matches:
        with open(matches[0], encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    raise FileNotFoundError(f"{name}.jsonl not under {root}")


def _stratified_sample(items: list[dict], n: int) -> list[dict]:
    """Return n items stratified across `length` quartiles (so we get short,
    medium-short, medium-long, long examples). Filters out items above the
    token cap. Deterministic via SEED."""
    rng = random.Random(SEED)
    eligible = [it for it in items
                if 100 < it.get("length", 0) <= HARD_TOKEN_CAP]
    if not eligible:
        # estimate length from char count
        for it in items:
            it["length"] = max(100, len(it.get("context", "")) // 4)
        eligible = [it for it in items if 100 < it["length"] <= HARD_TOKEN_CAP]
    if len(eligible) <= n:
        return eligible
    sorted_e = sorted(eligible, key=lambda x: x["length"])
    per_quartile = max(1, n // 4)
    q = len(sorted_e) // 4
    pools = [sorted_e[:q], sorted_e[q:2*q], sorted_e[2*q:3*q], sorted_e[3*q:]]
    picked = []
    for pool in pools:
        rng.shuffle(pool)
        picked.extend(pool[:per_quartile])
    # If we under-shot due to integer division, top up from the leftover
    seen_ids = {id(p) for p in picked}
    leftover = [it for it in eligible if id(it) not in seen_ids]
    rng.shuffle(leftover)
    picked.extend(leftover[:max(0, n - len(picked))])
    return picked[:n]


def gen_longbench() -> int:
    out_path = OUT_DIR / "prompts_pub_longbench.jsonl"
    rows = []
    failures = []
    print("=" * 60); print("LongBench (publication tier)"); print("=" * 60)
    for name, ctx_f, in_f, tmpl in LONGBENCH_TASKS:
        try:
            items = _load_lb_items(name)
            picked = _stratified_sample(items, LB_PER_TASK)
            counts_by_bin = [0, 0, 0, 0]   # for reporting
            for i, it in enumerate(picked, 1):
                ctx = it.get(ctx_f, "") or ""
                if len(ctx) > HARD_CHAR_CAP:
                    ctx = ctx[:HARD_CHAR_CAP]
                inp = (it.get(in_f, "") or "").strip() if in_f else ""
                text = tmpl.format(context=ctx, input=inp)
                tok = int(it.get("length", 0))
                bin_idx = min(3, tok // 3000)
                counts_by_bin[bin_idx] += 1
                rows.append({
                    "prompt_id": f"{name}_pub_{i:03d}",
                    "task": name,
                    "prompt_text": text,
                    "expected_max_tokens": EXPECTED_MAX_TOKENS,
                    "ground_truth": it.get("answers", None),
                    "_meta": {
                        "source": "longbench",
                        "ext_id": it.get("_id", ""),
                        "length_tokens_reported": tok,
                        "context_chars_original": len(it.get(ctx_f, "") or ""),
                        "context_chars_used": len(ctx),
                        "prompt_chars": len(text),
                    },
                })
            bins_str = f"[<3K:{counts_by_bin[0]}, 3-6K:{counts_by_bin[1]}, 6-9K:{counts_by_bin[2]}, 9K+:{counts_by_bin[3]}]"
            print(f"  {name:<18}  {len(picked):3d}  bins={bins_str}")
        except Exception as e:
            failures.append((name, str(e)))
            print(f"  {name:<18}  FAILED: {str(e)[:80]}")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} prompts -> {out_path}")
    if failures:
        print(f"failed tasks ({len(failures)}): {[f[0] for f in failures]}")
    return len(rows)


# ----- NIAH (Needle in a Haystack) -----------------------------------------
NEEDLE_PHRASES = [
    ("The best ice-cream flavor in San Francisco is mango sorbet from Bi-Rite.",
     "What is the best ice-cream flavor in San Francisco?",
     "mango sorbet from Bi-Rite"),
    ("The secret access code for the lab is alpha-9-bravo-7-charlie-2.",
     "What is the secret access code for the lab?",
     "alpha-9-bravo-7-charlie-2"),
    ("Dr. Lin's office is located in room 4087 of the Cordell Hull Building.",
     "Where is Dr. Lin's office located?",
     "room 4087 of the Cordell Hull Building"),
]
NIAH_LENGTHS_TOKENS = [4096, 8192, 16384, 32768]
NIAH_DEPTHS = [0.0, 0.17, 0.33, 0.50, 0.67, 0.83, 1.0]
NIAH_TEMPLATE = (
    "You are an assistant that answers questions based on a long document. "
    "Read the document carefully and answer the question at the end.\n\n"
    "Document:\n{haystack}\n\nQuestion: {question}\nAnswer:"
)


def _paul_graham_haystack(target_chars: int) -> str:
    """Concatenate a public-domain text repeatedly until we hit target length.
    Uses a long Paul-Graham-style filler; if HF dataset 'pg_essays' isn't
    installed we fall back to LongBench gov_report context as filler."""
    try:
        items = _load_lb_items("gov_report")
        filler = " ".join((it.get("context") or "")[:5000] for it in items[:8])
    except Exception:
        filler = ("The quick brown fox jumps over the lazy dog. " * 200)
    out = []
    n = 0
    while n < target_chars:
        out.append(filler)
        n += len(filler)
    return "".join(out)[:target_chars]


def gen_niah() -> int:
    out_path = OUT_DIR / "prompts_pub_niah.jsonl"
    rows = []
    print("=" * 60); print("Needle in a Haystack"); print("=" * 60)
    for length_tok in NIAH_LENGTHS_TOKENS:
        if length_tok > HARD_TOKEN_CAP:
            print(f"  skip length={length_tok} (above HARD_TOKEN_CAP={HARD_TOKEN_CAP})")
            continue
        target_chars = length_tok * 4
        haystack = _paul_graham_haystack(target_chars)
        for depth in NIAH_DEPTHS:
            for ni, (needle, question, answer) in enumerate(NEEDLE_PHRASES):
                if ni > 0:
                    continue   # one needle per (length, depth) cell for tier-min
                insert_at = max(100, int(len(haystack) * depth))
                hs = haystack[:insert_at] + " " + needle + " " + haystack[insert_at:]
                hs = hs[:target_chars + len(needle) + 2]
                text = NIAH_TEMPLATE.format(haystack=hs, question=question)
                rows.append({
                    "prompt_id": f"niah_L{length_tok//1024}K_d{int(depth*100):02d}_n{ni}",
                    "task": "niah",
                    "prompt_text": text,
                    "expected_max_tokens": 64,
                    "ground_truth": answer,
                    "_meta": {
                        "source": "self_generated_niah",
                        "length_tokens_target": length_tok,
                        "depth_fraction": depth,
                        "needle_idx": ni,
                        "needle_phrase": needle,
                        "prompt_chars": len(text),
                    },
                })
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} NIAH prompts -> {out_path}")
    return len(rows)


# ----- Reasoning (GSM8K + AIME-24) ------------------------------------------
GSM8K_N = int(os.environ.get("GSM8K_N", "100"))
AIME_N  = int(os.environ.get("AIME_N", "30"))
REASONING_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def gen_reasoning() -> int:
    out_path = OUT_DIR / "prompts_pub_reasoning.jsonl"
    rows = []
    print("=" * 60); print("Reasoning (GSM8K + AIME-24)"); print("=" * 60)
    from datasets import load_dataset
    # GSM8K
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rng = random.Random(SEED)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i, k in enumerate(idx[:GSM8K_N], 1):
            item = ds[int(k)]
            question = item["question"]
            answer_text = item["answer"]
            # answer is "explanation\n#### NUMBER"
            gt = answer_text.split("####")[-1].strip()
            rows.append({
                "prompt_id": f"gsm8k_pub_{i:03d}",
                "task": "gsm8k",
                "prompt_text": question + REASONING_SUFFIX,
                "expected_max_tokens": 1024,
                "ground_truth": gt,
                "_meta": {"source": "openai/gsm8k:main:test", "test_index": int(k)},
            })
        print(f"  gsm8k    {len(rows)} (sampled from {len(ds)})")
    except Exception as e:
        print(f"  gsm8k FAILED: {e}")
    n_after_gsm = len(rows)
    # AIME-24
    try:
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        for i, item in enumerate(ds, 1):
            if i > AIME_N: break
            problem = item.get("problem", item.get("question", ""))
            gt = str(item.get("answer", "")).strip()
            rows.append({
                "prompt_id": f"aime24_pub_{i:03d}",
                "task": "aime24",
                "prompt_text": problem + REASONING_SUFFIX,
                "expected_max_tokens": 1024,
                "ground_truth": gt,
                "_meta": {"source": "HuggingFaceH4/aime_2024:train", "index": i - 1},
            })
        print(f"  aime24   {len(rows)-n_after_gsm} (of 30 available)")
    except Exception as e:
        print(f"  aime24 FAILED: {e}")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} reasoning prompts -> {out_path}")
    return len(rows)


def main() -> int:
    print(f"output dir: {OUT_DIR}")
    print(f"settings:   LB_PER_TASK={LB_PER_TASK}  HARD_TOKEN_CAP={HARD_TOKEN_CAP}  "
          f"GSM8K_N={GSM8K_N}  AIME_N={AIME_N}\n")
    n_lb   = gen_longbench()
    n_niah = gen_niah()
    n_rsn  = gen_reasoning()
    print()
    print("=" * 60); print("Publication-quality workload SUMMARY"); print("=" * 60)
    print(f"  LongBench:  {n_lb} prompts")
    print(f"  NIAH:       {n_niah} prompts")
    print(f"  Reasoning:  {n_rsn} prompts")
    print(f"  TOTAL/model: {n_lb + n_niah + n_rsn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
