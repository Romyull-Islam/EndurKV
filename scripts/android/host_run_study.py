#!/usr/bin/env python3
"""
host_run_study.py — host-side master orchestrator for the OnePlus 15 port.

For each prompt in PROMPTS_PATH (a JSONL with prompt_id / task / prompt_text):
  1. adb push the prompt text to /data/local/tmp/endurkv/prompts/<prompt_id>.txt
  2. adb shell run_one_prompt.sh ... which itself
       - starts the sensor sampler in background
       - runs entropy_probe (or attention_probe) on the prompt
       - stops the sampler
       - writes the join metadata
  3. adb pull the four outputs back to the host's logs/<sub>/ dir:
       <prompt_id>.entropy.csv
       <prompt_id>.sensors.csv
       <prompt_id>.probe.stderr
       <prompt_id>.run.json

After all prompts: concatenate the entropy CSVs into logs/<sub>_full.csv
(matching the host-side study_full.csv format), and join with sensors via
host_join_sensors.py for the controller-training analyses.

Env vars (all optional):
  WORKSPACE              defaults to "../../.." from this file
  MODEL_PATH             path on HOST to the GGUF; pushed to phone once
  PROMPTS_PATH           path on HOST to prompts.jsonl
  LOG_SUBDIR             default "study_phone_1b"
  N_TOKENS               max new tokens per prompt (default 64)
  SEED                   default 42
  PROBE                  default "entropy_probe" (or "attention_probe")
  MAX_PROMPTS            cap on number of prompts (0 = all)
  MAX_PER_TASK           per-task cap (0 = all)
  SENSORS_HZ             default 10
  ADB                    path to adb (default: just "adb")
  PHONE_ROOT             dir on phone (default /data/local/tmp/endurkv)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE      = Path(__file__).resolve()
WORKSPACE = Path(os.environ.get("WORKSPACE", HERE.parents[3]))
DEFAULT_PROMPTS  = WORKSPACE / "prompts" / "prompts.jsonl"
DEFAULT_MODEL    = WORKSPACE / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

ADB         = os.environ.get("ADB", "adb")
PHONE_ROOT  = os.environ.get("PHONE_ROOT", "/data/local/tmp/endurkv")
LOG_SUBDIR  = os.environ.get("LOG_SUBDIR", "study_phone_1b")
N_TOKENS    = int(os.environ.get("N_TOKENS", "64"))
SEED        = int(os.environ.get("SEED", "42"))
PROBE_NAME  = os.environ.get("PROBE", "entropy_probe")
MAX_PROMPTS = int(os.environ.get("MAX_PROMPTS", "0"))
MAX_PER_TASK = int(os.environ.get("MAX_PER_TASK", "0"))
SENSORS_HZ  = int(os.environ.get("SENSORS_HZ", "10"))


def adb(*args, check=True, capture=False) -> subprocess.CompletedProcess:
    cmd = [ADB, *args]
    return subprocess.run(cmd, check=check,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None,
                          text=True)


def adb_shell(cmd: str, check=True, capture=True) -> subprocess.CompletedProcess:
    return adb("shell", cmd, check=check, capture=capture)


def ensure_phone_layout() -> None:
    """Create the phone-side directory tree once at session start."""
    for sub in ("bin", "models", "prompts", f"logs/{LOG_SUBDIR}"):
        adb_shell(f"mkdir -p {PHONE_ROOT}/{sub}")


def push_static_artifacts(deploy_dir: Path, model_path: Path) -> str:
    """Push the binaries, .so files, scripts, and model only if missing.

    Returns the on-phone path of the model.
    """
    # Probe + libs + llama-bench (idempotent, but skip if already-same-size).
    for f in sorted((deploy_dir / "bin").iterdir()):
        remote = f"{PHONE_ROOT}/bin/{f.name}"
        local_size = f.stat().st_size
        r = adb_shell(f"stat -c %s {remote} 2>/dev/null || echo -1", capture=True)
        try:
            remote_size = int(r.stdout.strip())
        except ValueError:
            remote_size = -1
        if remote_size == local_size:
            continue
        print(f"[push] {f.name} ({local_size/1024/1024:.1f} MB)")
        adb("push", str(f), remote)

    # Scripts (small, always push to keep them in sync).
    for f in sorted((deploy_dir / "scripts").iterdir()):
        adb("push", str(f), f"{PHONE_ROOT}/scripts/{f.name}")
        adb_shell(f"chmod 755 {PHONE_ROOT}/scripts/{f.name}")

    # Chmod the binaries so they execute.
    adb_shell(f"chmod 755 {PHONE_ROOT}/bin/*")

    # Model: only push if size differs.
    remote_model = f"{PHONE_ROOT}/models/{model_path.name}"
    local_size = model_path.stat().st_size
    r = adb_shell(f"stat -c %s {remote_model} 2>/dev/null || echo -1", capture=True)
    try:
        remote_size = int(r.stdout.strip())
    except ValueError:
        remote_size = -1
    if remote_size != local_size:
        print(f"[push] {model_path.name} ({local_size/1024/1024:.0f} MB) — may take a minute")
        adb("push", str(model_path), remote_model)
    else:
        print(f"[push] {model_path.name} — already on phone, skipping")

    return remote_model


def load_prompts(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    if MAX_PER_TASK > 0:
        per: dict[str, int] = {}
        kept = []
        for p in out:
            t = p.get("task", "?")
            if per.get(t, 0) < MAX_PER_TASK:
                kept.append(p)
                per[t] = per.get(t, 0) + 1
        out = kept
    if MAX_PROMPTS > 0:
        out = out[:MAX_PROMPTS]
    return out


def run_one(prompt: dict, remote_model: str, host_log_dir: Path) -> int:
    pid  = prompt["prompt_id"]
    text = prompt["prompt_text"]

    # 1) Write prompt text to a temp file on host, push.
    host_tmp = host_log_dir / f".{pid}.txt"
    host_tmp.write_text(text, encoding="utf-8")
    remote_prompt = f"{PHONE_ROOT}/prompts/{pid}.txt"
    adb("push", str(host_tmp), remote_prompt)
    host_tmp.unlink(missing_ok=True)

    # 2) Run the orchestrator on phone.
    remote_log = f"{PHONE_ROOT}/logs/{LOG_SUBDIR}"
    cmd = (
        f"sh {PHONE_ROOT}/scripts/run_one_prompt.sh "
        f"--probe {PHONE_ROOT}/bin/{PROBE_NAME} "
        f"--sampler {PHONE_ROOT}/scripts/sample_sensors.sh "
        f"--model {remote_model} "
        f"--prompt {remote_prompt} "
        f"--prompt-id {pid} "
        f"--max-tokens {N_TOKENS} "
        f"--seed {SEED} "
        f"--sensors-hz {SENSORS_HZ} "
        f"--out-dir {remote_log}"
    )
    t0 = time.time()
    r = adb_shell(cmd, check=False, capture=True)
    elapsed = time.time() - t0

    # 3) Pull the outputs. attention_probe also produces .attn.bin (ATNH per-head)
    # and (since 2026-05-24 dual-format probe) .v1.attn.bin (ATTN head-averaged).
    pull_exts = [".entropy.csv", ".sensors.csv", ".probe.stderr", ".run.json"]
    if PROBE_NAME == "attention_probe":
        pull_exts.append(".attn.bin")
        pull_exts.append(".v1.attn.bin")
    for ext in pull_exts:
        remote = f"{remote_log}/{pid}{ext}"
        local  = host_log_dir / f"{pid}{ext}"
        # ignore pull errors (e.g. sensors.csv missing if sampler died)
        adb("pull", remote, str(local), check=False, capture=True)

    rc = r.returncode
    status = "OK  " if rc == 0 else f"FAIL({rc})"
    print(f"[run] {pid:<28} {status}  {elapsed:>5.1f}s   stderr-tail:")
    if r.stdout:
        for line in r.stdout.strip().splitlines()[-3:]:
            print(f"      {line}")
    if r.stderr and rc != 0:
        for line in r.stderr.strip().splitlines()[-5:]:
            print(f"      [stderr] {line}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--model",   default=str(DEFAULT_MODEL))
    ap.add_argument("--deploy",  default=str(WORKSPACE / "phone-deploy"))
    args = ap.parse_args()

    prompts_path = Path(args.prompts)
    model_path   = Path(args.model)
    deploy_dir   = Path(args.deploy)

    if not prompts_path.exists():
        print(f"ERROR: {prompts_path} does not exist", file=sys.stderr)
        print("       Generate it with EndurKV/scripts/09_generate_workload.py first,",
              file=sys.stderr)
        print(f"       or pass --prompts <path-to-smoke-jsonl>.", file=sys.stderr)
        return 1
    if not model_path.exists():
        print(f"ERROR: {model_path} does not exist", file=sys.stderr)
        return 1
    if not (deploy_dir / "bin" / PROBE_NAME).exists():
        print(f"ERROR: {deploy_dir}/bin/{PROBE_NAME} not found", file=sys.stderr)
        return 1

    # Verify adb sees a device.
    r = adb("devices", capture=True)
    if "device\n" not in r.stdout and "device\r\n" not in r.stdout:
        print("ERROR: no device returned by `adb devices`. Plug in the phone "
              "and enable USB debugging.", file=sys.stderr)
        print(r.stdout, file=sys.stderr)
        return 1

    host_log_dir = WORKSPACE / "logs" / LOG_SUBDIR
    host_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[host] workspace:  {WORKSPACE}")
    print(f"[host] deploy dir: {deploy_dir}")
    print(f"[host] phone root: {PHONE_ROOT}")
    print(f"[host] log subdir: logs/{LOG_SUBDIR}")
    print(f"[host] probe:      {PROBE_NAME}")
    print(f"[host] model:      {model_path.name}")
    print(f"[host] prompts:    {prompts_path}")
    print()

    ensure_phone_layout()
    remote_model = push_static_artifacts(deploy_dir, model_path)

    prompts = load_prompts(prompts_path)
    print(f"[host] running {len(prompts)} prompts")
    print()

    n_ok = 0
    n_fail = 0
    t_start = time.time()
    for i, p in enumerate(prompts, 1):
        print(f"[host] [{i}/{len(prompts)}]", end=" ")
        rc = run_one(p, remote_model, host_log_dir)
        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1
    elapsed = time.time() - t_start

    print()
    print(f"[host] done: ok={n_ok} fail={n_fail} elapsed={elapsed:.1f}s")
    print(f"[host] logs in {host_log_dir}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
