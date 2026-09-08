#!/usr/bin/env python3
"""
Phase E.2 — run prune_probe over a workload and concatenate results.

For each prompt in PROMPTS_PATH (default data/prompts.jsonl), runs prune_probe
to record entropy + KL(P_full || P_pruned) at K ∈ {16, 64, 256} for each
decode step. Output: logs/prune/<prompt_id>.csv.

Env vars (same conventions as 10_run_study.py):
    MODEL_PATH      default Llama-3.2-1B-Instruct-Q4_K_M.gguf
    PROMPTS_PATH    default data/prompts.jsonl
    LOG_SUBDIR      default "prune"
    N_TOKENS        default 32   (per-step prune-and-redecode is heavy, so we keep this short)
    SEED            default 42
    KS              default "16,64,256"
    MAX_PER_TASK    default 0 (unlimited)
    MAX_PROMPTS     default 0 (unlimited)
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
PROBE    = ROOT / "entropy_probe" / "build" / "prune_probe"
MODEL    = Path(os.environ.get("MODEL_PATH",
                ROOT / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
PROMPTS  = Path(os.environ.get("PROMPTS_PATH",
                ROOT / "data" / "prompts.jsonl"))
LOG_SUB  = os.environ.get("LOG_SUBDIR", "prune")
LOG_DIR  = ROOT / "logs" / LOG_SUB

N_TOKENS     = int(os.environ.get("N_TOKENS", "32"))
SEED         = int(os.environ.get("SEED", "42"))
KS_CSV       = os.environ.get("KS", "16,64,256")
MAX_PER_TASK = int(os.environ.get("MAX_PER_TASK", "0"))
MAX_PROMPTS  = int(os.environ.get("MAX_PROMPTS",  "0"))


def main() -> int:
    if not PROBE.exists():
        print(f"ERROR: {PROBE} not built. Run scripts/04_build_probe.sh first.", file=sys.stderr)
        return 1
    if not MODEL.exists():
        print(f"ERROR: {MODEL} missing.", file=sys.stderr)
        return 1
    if not PROMPTS.exists():
        print(f"ERROR: {PROMPTS} missing.", file=sys.stderr)
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

    print(f"[prune] {n_total} prompts (after sampling: per_task={MAX_PER_TASK or 'all'}, total={MAX_PROMPTS or 'all'})")
    print(f"[prune] max_tokens={N_TOKENS}, seed={SEED}, Ks={KS_CSV}, model={MODEL.name}")
    print(f"[prune] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
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
        out_log = LOG_DIR / f"{prompt_id}.stderr"

        print(f"[prune] [{i:>2}/{n_total}] {prompt_id:<24} task={task:<18} ... ", end="", flush=True)

        with open(out_log, "w") as ferr:
            r = subprocess.run(
                [
                    str(PROBE),
                    "--model",       str(MODEL),
                    "--prompt-file", str(prompt_tmp),
                    "--prompt-id",   prompt_id,
                    "--max-tokens",  str(N_TOKENS),
                    "--seed",        str(SEED),
                    "--output",      str(out_csv),
                    "--ks",          KS_CSV,
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
                        bits = [t for t in ln.split() if t.startswith(("steps=", "eos_step=", "total_ms="))]
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
    print(f"[prune] runs ok={n_ok}  failed={n_fail}  elapsed={elapsed}s")

    # Concatenate per-prompt CSVs into a global file.
    full_csv_name = "prune_full.csv" if LOG_SUB == "prune" else f"{LOG_SUB}_full.csv"
    full_csv = ROOT / "logs" / full_csv_name
    task_by_id = {p["prompt_id"]: p["task"] for p in prompts}
    n_files = 0
    n_rows  = 0
    writer  = None
    with open(full_csv, "w", newline="") as out_f:
        for csv_path in sorted(LOG_DIR.glob("*.csv")):
            with open(csv_path, newline="") as in_f:
                reader = csv.DictReader(in_f, restkey="_extras")
                for row in reader:
                    row.pop("_extras", None)
                    if not row.get("prompt_id"):
                        continue
                    row["task"] = task_by_id.get(row["prompt_id"], "unknown")
                    if writer is None:
                        fields = [k for k in row.keys() if k is not None]
                        writer = csv.DictWriter(out_f, fieldnames=fields, extrasaction="ignore")
                        writer.writeheader()
                    writer.writerow(row)
                    n_rows += 1
            n_files += 1
    print(f"[prune] concatenated {n_files} files, {n_rows} rows -> {full_csv}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
