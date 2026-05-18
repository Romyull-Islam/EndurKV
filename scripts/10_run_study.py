#!/usr/bin/env python3
"""
Phase D — run attention_probe over every prompt in data/prompts.jsonl.

Outputs:
    logs/study/<prompt_id>.csv         entropy CSV
    logs/study/<prompt_id>.attn.bin    attention sidecar
    logs/study/<prompt_id>.stderr      probe log
    logs/study_full.csv                concatenation + task column
    logs/study_meta.json               run metadata

Usage:
    python3 scripts/10_run_study.py
    N_TOKENS=128 SEED=7 python3 scripts/10_run_study.py
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Auto-reexec under the project venv (same trick as 09_generate_workload.py).
_ROOT_FOR_VENV = Path(__file__).resolve().parents[1]
_VENV_DIR = _ROOT_FOR_VENV / ".venv"
_VENV_PY = _VENV_DIR / "bin" / "python3"
if _VENV_PY.exists():
    try:
        _under_venv = Path(sys.prefix).resolve() == _VENV_DIR.resolve()
    except OSError:
        _under_venv = False
    if not _under_venv:
        os.execv(str(_VENV_PY), [str(_VENV_PY), __file__, *sys.argv[1:]])

ROOT     = Path(__file__).resolve().parents[1]
PROBE    = ROOT / "entropy_probe" / "build" / "attention_probe"

# Model / log-dir / prompt-list overridable via env vars so we can run
# multiple model sizes and workloads without losing each other's results.
MODEL    = Path(os.environ.get("MODEL_PATH",
                ROOT / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
LOG_SUB  = os.environ.get("LOG_SUBDIR", "study")
PROMPTS  = Path(os.environ.get("PROMPTS_PATH",
                ROOT / "data" / "prompts.jsonl"))
LOG_DIR  = ROOT / "logs" / LOG_SUB

N_TOKENS     = int(os.environ.get("N_TOKENS", "64"))
SEED         = int(os.environ.get("SEED", "42"))
# Limit how many prompts run, useful for quick smoke tests on the 8B model:
#   MAX_PER_TASK=2  -> take only the first 2 prompts of each task (preserves
#                     per-task balance, recommended for the quick 8B run)
#   MAX_PROMPTS=N   -> hard cap on the total number of prompts (overrides MAX_PER_TASK)
MAX_PER_TASK = int(os.environ.get("MAX_PER_TASK", "0"))   # 0 = unlimited
MAX_PROMPTS  = int(os.environ.get("MAX_PROMPTS",  "0"))   # 0 = unlimited


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, timeout=10).strip()
    except Exception:
        return ""


def main() -> int:
    if not PROBE.exists():
        print(f"ERROR: {PROBE} not built. Run scripts/04_build_probe.sh first.", file=sys.stderr)
        return 1
    if not MODEL.exists():
        print(f"ERROR: {MODEL} missing.", file=sys.stderr)
        return 1
    if not PROMPTS.exists():
        print(f"ERROR: {PROMPTS} missing. Run scripts/09_generate_workload.py first.", file=sys.stderr)
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    prompts: list[dict] = []
    with open(PROMPTS) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    # Optional sub-sampling.
    if MAX_PER_TASK > 0:
        per_task: dict[str, int] = {}
        kept = []
        for p in prompts:
            t = p["task"]
            if per_task.get(t, 0) < MAX_PER_TASK:
                kept.append(p)
                per_task[t] = per_task.get(t, 0) + 1
        prompts = kept
    if MAX_PROMPTS > 0 and len(prompts) > MAX_PROMPTS:
        prompts = prompts[:MAX_PROMPTS]
    n_total = len(prompts)

    print(f"[study] {n_total} prompts (after sampling: per_task={MAX_PER_TASK or 'all'}, total={MAX_PROMPTS or 'all'})")
    print(f"[study] max_tokens={N_TOKENS}, seed={SEED}, model={MODEL.name}")
    print(f"[study] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print()

    t_start = time.time()
    n_ok = 0
    n_fail = 0

    for i, item in enumerate(prompts, 1):
        prompt_id   = item["prompt_id"]
        task        = item["task"]
        prompt_text = item["prompt_text"]

        prompt_tmp = LOG_DIR / f".{prompt_id}.prompt.tmp"
        prompt_tmp.write_text(prompt_text, encoding="utf-8")

        out_csv = LOG_DIR / f"{prompt_id}.csv"
        out_bin = LOG_DIR / f"{prompt_id}.attn.bin"
        out_log = LOG_DIR / f"{prompt_id}.stderr"

        print(f"[study] [{i:>2}/{n_total}] {prompt_id:<24} task={task:<18} ... ", end="", flush=True)

        with open(out_log, "w") as ferr:
            r = subprocess.run(
                [
                    str(PROBE),
                    "--model",        str(MODEL),
                    "--prompt-file",  str(prompt_tmp),
                    "--prompt-id",    prompt_id,
                    "--max-tokens",   str(N_TOKENS),
                    "--seed",         str(SEED),
                    "--output",       str(out_csv),
                    "--output-attn",  str(out_bin),
                ],
                stdout=subprocess.DEVNULL,
                stderr=ferr,
                env=env,
            )

        prompt_tmp.unlink(missing_ok=True)

        if r.returncode == 0:
            summary = ""
            try:
                for ln in out_log.read_text().splitlines():
                    if "steps=" in ln and "total_ms=" in ln:
                        toks = ln.split()
                        bits = [t for t in toks if t.startswith(("steps=", "eos_step=", "total_ms="))]
                        summary = " ".join(bits)
                        break
            except Exception:
                pass
            print(f"OK    {summary}")
            n_ok += 1
        else:
            print(f"FAIL  see {out_log}")
            n_fail += 1

    elapsed = int(time.time() - t_start)
    print()
    print(f"[study] runs ok={n_ok}  failed={n_fail}  elapsed={elapsed}s")

    # ---- Concatenate per-prompt CSVs into logs/<sub>_full.csv ----
    print("[study] concatenating per-prompt CSVs...")
    full_csv_name = "study_full.csv" if LOG_SUB == "study" else f"{LOG_SUB}_full.csv"
    full_csv = ROOT / "logs" / full_csv_name
    task_by_id = {p["prompt_id"]: p["task"] for p in prompts}
    n_files = 0
    n_rows = 0
    n_skipped = 0
    writer = None
    with open(full_csv, "w", newline="") as out_f:
        for csv_path in sorted(LOG_DIR.glob("*.csv")):
            with open(csv_path, newline="") as in_f:
                # restkey="_extras" so that a row with extra fields (e.g. from
                # a token_text whose CSV escaping produced more comma-separated
                # tokens than expected) doesn't crash the writer; we just drop
                # the extras.
                reader = csv.DictReader(in_f, restkey="_extras")
                for row in reader:
                    row.pop("_extras", None)
                    if not row.get("prompt_id"):
                        n_skipped += 1
                        continue
                    row["task"] = task_by_id.get(row["prompt_id"], "unknown")
                    if writer is None:
                        fields = [k for k in row.keys() if k is not None]
                        writer = csv.DictWriter(out_f, fieldnames=fields, extrasaction="ignore")
                        writer.writeheader()
                    writer.writerow(row)
                    n_rows += 1
            n_files += 1
    if n_skipped:
        print(f"  warn: skipped {n_skipped} malformed rows during concat")
    print(f"  concatenated {n_files} files, {n_rows} rows -> {full_csv}")

    # ---- Metadata ----
    print("[study] writing logs/study_meta.json...")
    llama_dir = ROOT / "llama.cpp"
    meta = {
        "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "n_prompts_ok": n_ok,
        "n_prompts_failed": n_fail,
        "elapsed_seconds": elapsed,
        "settings": {
            "max_tokens_per_prompt": N_TOKENS,
            "seed": SEED,
            "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        },
        "model": {
            "path": str(MODEL),
            "sha256": sha256_of(MODEL) if MODEL.exists() else "",
            "size_bytes": MODEL.stat().st_size if MODEL.exists() else 0,
        },
        "llama_cpp": {
            "path": str(llama_dir),
            "commit": safe_run(["git", "-C", str(llama_dir), "rev-parse", "HEAD"]),
            "branch": safe_run(["git", "-C", str(llama_dir), "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty_files": safe_run(["git", "-C", str(llama_dir), "status", "--short"]).splitlines(),
        },
        "workload": {
            "prompts_jsonl": str(PROMPTS),
            "n_prompts_total": len(prompts),
        },
    }
    meta_name = "study_meta.json" if LOG_SUB == "study" else f"{LOG_SUB}_meta.json"
    meta_path = ROOT / "logs" / meta_name
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in ("hostname", "elapsed_seconds", "n_prompts_ok", "n_prompts_failed", "settings")}, indent=2))

    print()
    print("[study] === summary ===")
    print(f"  full csv:     {full_csv}  ({n_rows + 1 if writer else 0} lines incl. header)")
    print(f"  per-prompt:   {LOG_DIR}/  ({len(list(LOG_DIR.glob('*.csv')))} csv files)")
    print(f"  attn binary:  {LOG_DIR}/  ({len(list(LOG_DIR.glob('*.attn.bin')))} bin files)")
    print(f"  metadata:     {meta_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
