#!/usr/bin/env python3
"""
Phase C — build the measurement workload from real benchmarks.

Three tiers, all common across recent KV-cache management papers:

  Tier 1 — LongBench (Bai et al., ACL 2024, the gold-standard long-context
           bench used by KVSwap, KIVI, CAKE, SnapKV).
           8 tasks × 6 prompts = 48 prompts:
             qasper, multifieldqa_en, triviaqa, samsum,
             hotpotqa, gov_report, trec, lcc

  Tier 2 — HELM-style summarization (used by H2O):
           xsum, cnn_dailymail × 8 = 16 prompts

  Tier 3 — lm-eval-harness style multiple-choice (used by H2O):
           piqa, openbookqa × 8 = 16 prompts

Total target: 80 prompts. If any HF dataset 401s / breaks under
`datasets` 4.x (which now rejects script-based datasets), that
tier is logged as 'skipped' and the rest still write.

Output: data/prompts.jsonl  one row per prompt with
    {prompt_id, task, prompt_text, expected_max_tokens, _meta}
"""
from __future__ import annotations

import json
import os
import sys
import traceback
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
# Allow the on-phone port to redirect output to a workspace-relative location.
_OUT_OVERRIDE = os.environ.get("PROMPTS_OUT")
OUT_PATH = Path(_OUT_OVERRIDE) if _OUT_OVERRIDE else (ROOT / "data" / "prompts.jsonl")

CONTEXT_CHAR_LIMIT  = 2000
EXPECTED_MAX_TOKENS = 64


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def truncate_context(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for marker in ("\n\n", ". ", "\n"):
        idx = cut.rfind(marker)
        if idx > limit * 0.7:
            return cut[: idx + len(marker)].rstrip() + " ..."
    return cut.rstrip() + " ..."


# --------------------------------------------------------------------------- #
# tier 1 — LongBench (one zip with all task jsonl files)                      #
# --------------------------------------------------------------------------- #
_LONGBENCH_REPO = "zai-org/LongBench"
_LONGBENCH_EXTRACTED: Path | None = None


def _ensure_longbench_extracted() -> Path:
    global _LONGBENCH_EXTRACTED
    if _LONGBENCH_EXTRACTED is not None:
        return _LONGBENCH_EXTRACTED
    import zipfile
    from huggingface_hub import hf_hub_download
    zip_path = hf_hub_download(_LONGBENCH_REPO, "data.zip", repo_type="dataset")
    extract_root = Path(zip_path).parent / "longbench_extracted"
    if not (extract_root / ".extracted").exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"  extracting {zip_path}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_root)
        (extract_root / ".extracted").touch()
    _LONGBENCH_EXTRACTED = extract_root
    return extract_root


def _find_task_jsonl(extract_root: Path, task_name: str) -> Path:
    for cand in [extract_root / "data" / f"{task_name}.jsonl",
                 extract_root / f"{task_name}.jsonl"]:
        if cand.exists(): return cand
    matches = list(extract_root.rglob(f"{task_name}.jsonl"))
    if matches: return matches[0]
    raise FileNotFoundError(f"no {task_name}.jsonl under {extract_root}")


def load_longbench(name: str, n: int, template: str, ctx_field: str = "context",
                   in_field: str | None = "input") -> list[dict]:
    extract_root = _ensure_longbench_extracted()
    path = _find_task_jsonl(extract_root, name)
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if len(rows) >= n: break
            line = line.strip()
            if not line: continue
            item = json.loads(line)
            ctx_raw = item.get(ctx_field, "") or ""
            inp     = (item.get(in_field, "") or "").strip() if in_field else ""
            ctx_t   = truncate_context(ctx_raw, CONTEXT_CHAR_LIMIT)
            text    = template.format(context=ctx_t, input=inp)
            rows.append({
                "task": name, "prompt_text": text,
                "_orig_chars": len(ctx_raw), "_used_chars": len(ctx_t),
                "_id": item.get("_id", str(line_no)),
                "_source": "longbench",
            })
    return rows


# --------------------------------------------------------------------------- #
# tier 2 — HF parquet-backed datasets                                         #
# --------------------------------------------------------------------------- #
def _try_load_dataset(repo: str, config: str | None, split: str):
    """Return list[dict] or raise."""
    from datasets import load_dataset
    if config:
        ds = load_dataset(repo, config, split=split)
    else:
        ds = load_dataset(repo, split=split)
    return ds


def load_hf_simple(name: str, repo: str, config: str | None, split: str,
                   ctx_field: str, template: str, n: int) -> list[dict]:
    ds = _try_load_dataset(repo, config, split)
    rows: list[dict] = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        ctx_raw = item.get(ctx_field, "") or ""
        ctx_t   = truncate_context(ctx_raw, CONTEXT_CHAR_LIMIT)
        text    = template.format(context=ctx_t)
        rows.append({
            "task": name, "prompt_text": text,
            "_orig_chars": len(ctx_raw), "_used_chars": len(ctx_t),
            "_id": str(item.get("id", i)),
            "_source": f"hf:{repo}",
        })
    return rows


def load_piqa(name: str, n: int) -> list[dict]:
    """PIQA — physical reasoning, multi-choice. Low-entropy answer."""
    # Try in order: lighteval mirror (parquet, modern), then ybisk official.
    last_err = None
    for repo, config in [("lighteval/piqa", None), ("ybisk/piqa", "plain_text"), ("piqa", None)]:
        try:
            ds = _try_load_dataset(repo, config, "validation")
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise last_err  # type: ignore

    rows: list[dict] = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        goal = (item.get("goal") or item.get("question") or "").strip()
        sol1 = (item.get("sol1") or item.get("solution1") or "").strip()
        sol2 = (item.get("sol2") or item.get("solution2") or "").strip()
        if not (goal and sol1 and sol2):
            continue
        prompt = (
            f"Question: {goal}\n\n"
            f"Option A: {sol1}\n"
            f"Option B: {sol2}\n\n"
            "The better solution is option"
        )
        rows.append({
            "task": name, "prompt_text": prompt,
            "_orig_chars": len(prompt), "_used_chars": len(prompt),
            "_id": str(i), "_source": f"hf:{repo}",
        })
    return rows


def load_openbookqa(name: str, n: int) -> list[dict]:
    """OpenBookQA — knowledge-based 4-way multiple choice."""
    last_err = None
    for repo, config in [("lighteval/openbookqa", None),
                         ("allenai/openbookqa", "main"),
                         ("openbookqa", "main")]:
        try:
            ds = _try_load_dataset(repo, config, "validation")
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise last_err  # type: ignore

    rows: list[dict] = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        stem = (item.get("question_stem") or item.get("question") or "").strip()
        if isinstance(stem, dict):
            stem = stem.get("stem", "")
        choices = item.get("choices", {})
        if isinstance(choices, dict):
            texts = choices.get("text", []) or []
            labels = choices.get("label", []) or []
        else:
            texts = []; labels = []
        if not stem or len(texts) < 2:
            continue
        choice_str = "\n".join(f"{lab}. {ch}" for lab, ch in zip(labels, texts))
        prompt = f"Question: {stem}\n\n{choice_str}\n\nAnswer:"
        rows.append({
            "task": name, "prompt_text": prompt,
            "_orig_chars": len(prompt), "_used_chars": len(prompt),
            "_id": str(item.get("id", i)),
            "_source": f"hf:{repo}",
        })
    return rows


# --------------------------------------------------------------------------- #
# task spec                                                                   #
# --------------------------------------------------------------------------- #
LONGBENCH_TASKS = [
    ("qasper",          6, "context", "input",
     "You are given a scientific article. Answer the question based on it.\n\nArticle:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("multifieldqa_en", 6, "context", "input",
     "Read the following document and answer the question.\n\nDocument:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("triviaqa",        6, "context", "input",
     "Answer the question using the following passages.\n\nPassages:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("samsum",          6, "context", None,
     "Summarize the following dialogue.\n\nDialogue:\n{context}\n\nSummary:"),
    ("hotpotqa",        6, "context", "input",
     "Answer the question based on the following passages.\n\nPassages:\n{context}\n\nQuestion: {input}\nAnswer:"),
    ("gov_report",      6, "context", None,
     "Summarize the following report.\n\nReport:\n{context}\n\nSummary:"),
    ("trec",            6, "context", "input",
     "{context}\n\nClassify this question: {input}\nCategory:"),
    ("lcc",             6, "context", None,
     "Complete the following code:\n\n{context}"),
]

HELM_TASKS = [
    # (name, repo, config, split, ctx_field, template, n_samples)
    ("xsum",          "EdinburghNLP/xsum",   None,    "validation", "document",
     "Summarize the following article in one sentence.\n\nArticle:\n{context}\n\nSummary:", 8),
    ("cnn_dailymail", "abisee/cnn_dailymail","3.0.0", "validation", "article",
     "Summarize the following article.\n\nArticle:\n{context}\n\nSummary:", 8),
]


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    # Tier 1
    print("=" * 60); print("TIER 1 — LongBench"); print("=" * 60)
    for name, n, ctx_f, in_f, tmpl in LONGBENCH_TASKS:
        try:
            print(f"[{name}] loading...", flush=True)
            r = load_longbench(name, n, tmpl, ctx_f, in_f)
            print(f"[{name}] kept {len(r)} prompts")
            rows.extend(r)
        except Exception as e:
            failures.append((name, repr(e)))
            print(f"[{name}] FAILED: {e}")
            traceback.print_exc(limit=2)

    # Tier 2
    print(); print("=" * 60); print("TIER 2 — HELM (summarization)"); print("=" * 60)
    for (name, repo, cfg, split, ctx_f, tmpl, n) in HELM_TASKS:
        try:
            print(f"[{name}] loading from {repo}...", flush=True)
            r = load_hf_simple(name, repo, cfg, split, ctx_f, tmpl, n)
            print(f"[{name}] kept {len(r)} prompts")
            rows.extend(r)
        except Exception as e:
            failures.append((name, repr(e)))
            print(f"[{name}] FAILED: {e}")

    # Tier 3
    print(); print("=" * 60); print("TIER 3 — lm-eval-harness style (multi-choice)"); print("=" * 60)
    for fn, name, n in [(load_piqa, "piqa", 8), (load_openbookqa, "openbookqa", 8)]:
        try:
            print(f"[{name}] loading...", flush=True)
            r = fn(name, n)
            print(f"[{name}] kept {len(r)} prompts")
            rows.extend(r)
        except Exception as e:
            failures.append((name, repr(e)))
            print(f"[{name}] FAILED: {e}")

    # Number prompts and write
    by_task_count: dict[str, int] = {}
    out_rows = []
    for r in rows:
        t = r["task"]
        by_task_count[t] = by_task_count.get(t, 0) + 1
        prompt_id = f"{t}_{by_task_count[t]:03d}"
        out_rows.append({
            "prompt_id": prompt_id,
            "task": t,
            "prompt_text": r["prompt_text"],
            "expected_max_tokens": EXPECTED_MAX_TOKENS,
            "_meta": {
                "source": r["_source"],
                "ext_id": r.get("_id", ""),
                "context_chars_original": r["_orig_chars"],
                "context_chars_used": r["_used_chars"],
                "prompt_chars": len(r["prompt_text"]),
            },
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print("=" * 60)
    print(f"wrote {len(out_rows)} prompts to {OUT_PATH}")
    print("=" * 60)
    print("by task:")
    for t in sorted(by_task_count):
        print(f"  {t:<22}  {by_task_count[t]:>3} prompts")
    if failures:
        print()
        print("FAILED to load (skipped):")
        for name, err in failures:
            print(f"  {name}: {err[:200]}")
    print()
    print("preview — 1 sample per task (first 240 chars):")
    seen = set()
    for r in out_rows:
        if r["task"] in seen: continue
        seen.add(r["task"])
        snip = r["prompt_text"][:240].replace("\n", "\\n")
        print(f"  [{r['prompt_id']}] {r['task']:<20} {snip}{' ...' if len(r['prompt_text']) > 240 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
