#!/usr/bin/env python3
"""
Phase C — long-context workload generator (extended).

Companion to 09_generate_workload.py. The 09 file truncates contexts to ~2000
chars (~500 tokens) so a 1B model can fit them. This file does the opposite:
it pulls FULL-LENGTH (or close-to-full-length) prompts so we can verify the
entropy ↔ attention-concentration relationship at the long-context regime.

Tasks covered (13):
    LongBench tier 1 (already covered in earlier runs):
        narrativeqa, gov_report, qmsum, multi_news, hotpotqa, qasper
    LongBench tier 2 (new — adds long-ctx coverage to short-ctx-only tasks):
        triviaqa, samsum, multifieldqa_en, lcc, trec
    HELM tier (full-length article summarization):
        xsum, cnn_dailymail

Excluded (and noted in any comparison table as "n/a — inherently short"):
    piqa, openbookqa  — multi-choice tasks with ≤200-token prompts by design

Per task: 4 prompts. For LongBench tasks, picked across [4K, 6K, 8K, 12K]
length bins where available. For HELM, picked at natural length (typically
1K-3K). Hard cap of ~12K tokens per prompt to avoid OOM with FA disabled.

Output: data/prompts_longctx.jsonl (overwrites — same path as before).
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

# Auto-reexec under venv.
_ROOT_FOR_VENV = Path(__file__).resolve().parents[1]
_VENV_DIR = _ROOT_FOR_VENV / ".venv"
_VENV_PY  = _VENV_DIR / "bin" / "python3"
if _VENV_PY.exists():
    try:
        _under_venv = Path(sys.prefix).resolve() == _VENV_DIR.resolve()
    except OSError:
        _under_venv = False
    if not _under_venv:
        os.execv(str(_VENV_PY), [str(_VENV_PY), __file__, *sys.argv[1:]])

ROOT     = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "prompts_longctx.jsonl"

# ----- per-task prompt templates ---------------------------------------------
LONGBENCH_TASKS = [
    # already covered tier
    ("narrativeqa",     "context", "input",
     "You are given a story. Answer the question based on the story.\n\n"
     "Story:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("gov_report",      "context", None,
     "Summarize the following legislative report.\n\nReport:\n{context}\n\nSummary:"),
    ("qmsum",           "context", "input",
     "You are given a meeting transcript and a query about it.\n\n"
     "Transcript:\n{context}\n\nQuery: {input}\nAnswer:"),
    ("multi_news",      "context", None,
     "Summarize the following news articles.\n\nArticles:\n{context}\n\nSummary:"),
    ("hotpotqa",        "context", "input",
     "Answer the question based on the following passages.\n\n"
     "Passages:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("qasper",          "context", "input",
     "You are given a scientific article. Answer the question based on it.\n\n"
     "Article:\n{context}\n\nQuestion: {input}\nAnswer:"),
    # new — long-context variants of tasks that were truncated in the standard run
    ("triviaqa",        "context", "input",
     "Answer the question using the following passages.\n\n"
     "Passages:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("samsum",          "context", None,
     "Summarize the following dialogue.\n\nDialogue:\n{context}\n\nSummary:"),
    ("multifieldqa_en", "context", "input",
     "Read the following document and answer the question.\n\n"
     "Document:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("lcc",             "context", None,
     "Complete the following code:\n\n{context}"),
    ("trec",            "context", "input",
     "{context}\n\nClassify this question: {input}\nCategory:"),
]

# HELM tier — full-length article summarisation. (We don't truncate; natural
# length is typically 1K-3K tokens which already exceeds the 09 short workload.)
HELM_TASKS = [
    # (task_name, repo, config, split, ctx_field, template, n_samples)
    ("xsum",          "EdinburghNLP/xsum",    None,    "validation", "document",
     "Summarize the following article in one sentence.\n\nArticle:\n{context}\n\nSummary:", 4),
    ("cnn_dailymail", "abisee/cnn_dailymail", "3.0.0", "validation", "article",
     "Summarize the following article.\n\nArticle:\n{context}\n\nSummary:", 4),
]

TARGET_BINS_LB = [4096, 6144, 8192, 12288]   # 4 LongBench picks per task
HARD_TOKEN_CAP = 13000                        # safety cap to avoid OOM with FA disabled
HARD_CHAR_CAP  = 4 * HARD_TOKEN_CAP           # ~52000 chars (~13K tokens)
EXPECTED_MAX_TOKENS = 64


# ----- LongBench loader (same data.zip cached by 09) ------------------------
_LONGBENCH_REPO = "zai-org/LongBench"
_LB_EXTRACTED: Path | None = None


def _ensure_lb_extracted() -> Path:
    global _LB_EXTRACTED
    if _LB_EXTRACTED is not None: return _LB_EXTRACTED
    from huggingface_hub import hf_hub_download
    zip_path = hf_hub_download(_LONGBENCH_REPO, "data.zip", repo_type="dataset")
    extract_root = Path(zip_path).parent / "longbench_extracted"
    if not (extract_root / ".extracted").exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_root)
        (extract_root / ".extracted").touch()
    _LB_EXTRACTED = extract_root
    return extract_root


def _find_lb_jsonl(root: Path, name: str) -> Path:
    for cand in [root / "data" / f"{name}.jsonl", root / f"{name}.jsonl"]:
        if cand.exists(): return cand
    matches = list(root.rglob(f"{name}.jsonl"))
    if matches: return matches[0]
    raise FileNotFoundError(f"{name}.jsonl not under {root}")


def _load_lb_items(name: str) -> list[dict]:
    path = _find_lb_jsonl(_ensure_lb_extracted(), name)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: items.append(json.loads(line))
    return items


def _pick_lb_at_bins(items: list[dict], bins: list[int]) -> list[dict]:
    """For each target token-length bin, pick the prompt whose `length` is
    closest. Items below HARD_TOKEN_CAP only."""
    eligible = [it for it in items
                if isinstance(it.get("length"), (int, float))
                and 0 < it.get("length", 0) <= HARD_TOKEN_CAP]
    if not eligible:
        # fall back to char-based estimate
        for it in items:
            it["length"] = max(1, len(it.get("context", "")) // 4)
        eligible = [it for it in items if it["length"] <= HARD_TOKEN_CAP]
    if not eligible:
        return []
    chosen = []
    used = set()
    for tgt in bins:
        best, best_dist = None, None
        for it in eligible:
            ext = id(it)
            if ext in used: continue
            dist = abs(it["length"] - tgt)
            if best_dist is None or dist < best_dist:
                best, best_dist = it, dist
        if best is not None:
            chosen.append(best)
            used.add(id(best))
    return chosen


# ----- HELM loader (HF parquet) ---------------------------------------------
def _load_hf_simple(repo: str, config: str | None, split: str, ctx_field: str,
                    n: int) -> list[dict]:
    from datasets import load_dataset
    if config:
        ds = load_dataset(repo, config, split=split)
    else:
        ds = load_dataset(repo, split=split)
    out = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        ctx = item.get(ctx_field, "") or ""
        # HELM articles are usually < 8K tokens; cap to be safe.
        if len(ctx) > HARD_CHAR_CAP:
            ctx = ctx[:HARD_CHAR_CAP]
        out.append({
            "context": ctx, "input": "",
            "_id": str(item.get("id", i)),
            "_orig_chars": len(item.get(ctx_field, "") or ""),
            "_used_chars": len(ctx),
            "_length_tokens": max(1, len(ctx) // 4),
        })
    return out


# ----- main ----------------------------------------------------------------
def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []

    print("=" * 60); print("LONG-CONTEXT WORKLOAD"); print("=" * 60)

    # LongBench
    for name, ctx_f, in_f, tmpl in LONGBENCH_TASKS:
        try:
            items = _load_lb_items(name)
            picked = _pick_lb_at_bins(items, TARGET_BINS_LB)
            for it in picked:
                ctx = it.get(ctx_f, "") or ""
                inp = (it.get(in_f, "") or "").strip() if in_f else ""
                # HARD_CHAR_CAP safety
                if len(ctx) > HARD_CHAR_CAP:
                    ctx = ctx[:HARD_CHAR_CAP]
                text = tmpl.format(context=ctx, input=inp)
                rows.append({
                    "task": name, "prompt_text": text,
                    "_orig_chars": len(it.get(ctx_f, "") or ""),
                    "_used_chars": len(ctx),
                    "_length_tokens": int(it.get("length", 0)),
                    "_id": it.get("_id", ""),
                    "_source": "longbench-longctx",
                })
            print(f"[{name:<20}] picked {len(picked)} at lengths "
                  f"{[it.get('length', 0) for it in picked]} tokens")
        except Exception as e:
            failures.append((name, str(e)))
            print(f"[{name:<20}] FAILED: {e}")

    # HELM
    for (name, repo, cfg, split, ctx_f, tmpl, n) in HELM_TASKS:
        try:
            items = _load_hf_simple(repo, cfg, split, ctx_f, n)
            for it in items:
                text = tmpl.format(context=it["context"])
                rows.append({
                    "task": name, "prompt_text": text,
                    "_orig_chars": it["_orig_chars"],
                    "_used_chars": it["_used_chars"],
                    "_length_tokens": it["_length_tokens"],
                    "_id": it.get("_id", ""),
                    "_source": f"hf:{repo}",
                })
            print(f"[{name:<20}] kept {len(items)} prompts (HELM, full-length)")
        except Exception as e:
            failures.append((name, str(e)))
            print(f"[{name:<20}] FAILED: {e}")

    # number prompts
    by_task: dict[str, int] = {}
    out_rows = []
    for r in rows:
        t = r["task"]
        by_task[t] = by_task.get(t, 0) + 1
        prompt_id = f"{t}_lc_{by_task[t]:02d}"
        out_rows.append({
            "prompt_id": prompt_id,
            "task": t,
            "prompt_text": r["prompt_text"],
            "expected_max_tokens": EXPECTED_MAX_TOKENS,
            "_meta": {
                "source": r["_source"],
                "ext_id": r.get("_id", ""),
                "length_tokens_reported": r["_length_tokens"],
                "context_chars_used": r["_used_chars"],
                "prompt_chars": len(r["prompt_text"]),
            },
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"wrote {len(out_rows)} long-context prompts to {OUT_PATH}")
    print("by task / length:")
    for r in out_rows:
        print(f"  {r['prompt_id']:<28}  task={r['task']:<18}  "
              f"length≈{r['_meta']['length_tokens_reported']:>5} tokens  "
              f"chars={r['_meta']['prompt_chars']:>6}")
    if failures:
        print()
        print("FAILED tasks (skipped):")
        for n, e in failures:
            print(f"  {n}: {e[:160]}")
    print()
    print("Tasks NOT in this workload (intentionally — they're inherently short):")
    print("  piqa, openbookqa  (multi-choice, ≤200 tokens by design)")
    print()
    print("To run the full long-context study against the 8B model:")
    print(f"  PROMPTS_PATH=data/prompts_longctx.jsonl \\")
    print(f"  MODEL_PATH=models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \\")
    print(f"  LOG_SUBDIR=study_8b_longctx \\")
    print(f"  python3 scripts/10_run_study.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
