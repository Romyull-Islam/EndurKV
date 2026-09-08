#!/bin/bash
# host_wave11_progress.sh — Wave-11 interim watchdog.
#
# Purpose
# -------
# While `phone_wave11_eval.sh` is grinding through the 30-cell Wave-11 Tier-1
# manifest on the OnePlus 15, we want a *cheap* host-side check we can run
# from a cron / a tmux babysitter (or just by hand) that:
#
#   1. Reads /tmp/wave11_phone.txt for the absolute phone-side run directory
#      (written by phone_wave11_eval.sh at launch time).
#   2. Pulls NEW data only — rsync-style by comparing on-device mtime to the
#      host mirror mtime so a wedged USB cable doesn't cost us 10 GB of
#      redundant pulls. Uses scripts/android/adb_resilient.sh helpers so
#      transient adb hiccups self-heal.
#   3. Computes interim aggregates from the mirror:
#        - cells_complete / cells_total           (cell == model x policy x bench)
#        - per-policy mean PPL (across chunks done so far, all models pooled)
#        - ETA based on observed per-cell wall time when available, otherwise
#          the manifest's expected_cell_minutes.
#   4. Prints ONE clean status line per invocation (cron-friendly):
#        [wave11 watchdog HH:MM:SS] cells X/30 (Y partial) | per-policy ppl: ...
#                                 | ETA Zh | last cell done HH:MM:SS
#   5. Optionally renders figures/eval_plots/wave11_interim_pareto.png — a
#      tiny per-policy mean-PPL vs cells-complete pareto — IF matplotlib is
#      importable in the workspace venv. The full ppl/niah/markdown render
#      is delegated to eval_pipeline/wave11_interim_plot.py.
#
# Idempotent: every step is safe to re-run. Locks itself with flock so two
# cron ticks don't fight over the same adb session.
#
# Exit codes:
#   0 — produced a status line (regardless of whether evals are done)
#   2 — /tmp/wave11_phone.txt missing or empty (cannot determine run dir)
#
# Usage:
#   bash /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/host_wave11_progress.sh
#
# Cron example (every 5 minutes):
#   */5 * * * * /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/host_wave11_progress.sh \
#       >> /tmp/wave11_watchdog.log 2>&1

set -u
set -o pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENDURKV_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$ENDURKV_ROOT/.." && pwd)"
PHONE_TXT="/tmp/wave11_phone.txt"
LOCK_FILE="/tmp/wave11_watchdog.lock"

PHONE_LOGS_LOCAL="$WORKSPACE_ROOT/phone-logs"
MTIME_DB="$PHONE_LOGS_LOCAL/.wave11_watchdog_mtimes"   # path<TAB>mtime per line
FIG_OUT_DIR="$ENDURKV_ROOT/figures/eval_plots"
PARETO_PNG="$FIG_OUT_DIR/wave11_interim_pareto.png"
INTERIM_PY="$ENDURKV_ROOT/eval_pipeline/wave11_interim_plot.py"
MANIFEST_JSON="$ENDURKV_ROOT/eval_pipeline/wave11_cells.json"

PY=""
for cand in "$WORKSPACE_ROOT/.venv/bin/python" "$WORKSPACE_ROOT/.venv/bin/python3" \
            "$(command -v python3 2>/dev/null || true)"; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then PY="$cand"; break; fi
done
[ -z "$PY" ] && PY="python3"

# ---------------------------------------------------------------------------
# Resilient ADB helpers
# ---------------------------------------------------------------------------
# shellcheck source=adb_resilient.sh
source "$SCRIPT_DIR/adb_resilient.sh"

ts() { date '+%H:%M:%S'; }
say() { printf '[wave11 watchdog %s] %s\n' "$(ts)" "$*"; }

# ---------------------------------------------------------------------------
# Single-instance lock (cron safety)
# ---------------------------------------------------------------------------
exec 9>"$LOCK_FILE" || { say "could not open lock $LOCK_FILE"; exit 0; }
if ! flock -n 9; then
    # Another invocation already running — nothing to do.
    say "another instance is running (lock $LOCK_FILE); exiting"
    exit 0
fi

# ---------------------------------------------------------------------------
# 1. Resolve the phone-side run dir
# ---------------------------------------------------------------------------
if [ ! -s "$PHONE_TXT" ]; then
    say "missing or empty $PHONE_TXT — no run to watch"
    exit 2
fi
PHONE_RUN_DIR="$(tr -d '[:space:]' < "$PHONE_TXT")"
if [ -z "$PHONE_RUN_DIR" ]; then
    say "$PHONE_TXT contains no path"
    exit 2
fi
RUN_NAME="$(basename "$PHONE_RUN_DIR")"
LOCAL_RUN_DIR="$PHONE_LOGS_LOCAL/$RUN_NAME"
mkdir -p "$LOCAL_RUN_DIR"
mkdir -p "$FIG_OUT_DIR"
mkdir -p "$(dirname "$MTIME_DB")"
touch "$MTIME_DB"

# ---------------------------------------------------------------------------
# 2. rsync-style pull: enumerate remote files with mtimes, compare to local DB
# ---------------------------------------------------------------------------
ADB_OK=0
if adb get-state 2>/dev/null | grep -q '^device$'; then
    ADB_OK=1
fi

PULLED=0
SKIPPED=0
PULL_ERRORS=0

if [ "$ADB_OK" -eq 1 ]; then
    # `find -printf` isn't available on Android toybox find, so use stat per
    # file via a single shell command. Listing files newer than the recorded
    # global watermark would also work, but a per-file mtime sweep is more
    # robust to partially-pulled trees (a previous tick that died mid-pull).
    REMOTE_LIST_RAW="$(adb_safe_shell "
        if [ -d '$PHONE_RUN_DIR' ]; then
            find '$PHONE_RUN_DIR' -type f 2>/dev/null \
                | while read -r f; do
                    m=\$(stat -c %Y \"\$f\" 2>/dev/null)
                    [ -n \"\$m\" ] && printf '%s\t%s\n' \"\$m\" \"\$f\"
                done
        fi
    " 2>/dev/null || true)"

    # Build an associative map from the local mtime DB.
    declare -A LOCAL_MTIME
    while IFS=$'\t' read -r p m; do
        [ -n "$p" ] && LOCAL_MTIME["$p"]="$m"
    done < "$MTIME_DB"

    # New DB content is accumulated in a tmp file and atomically swapped at end.
    NEW_DB="$MTIME_DB.tmp.$$"
    : > "$NEW_DB"

    while IFS=$'\t' read -r m_remote f_remote; do
        [ -z "$f_remote" ] && continue
        rel="${f_remote#$PHONE_RUN_DIR/}"
        f_local="$LOCAL_RUN_DIR/$rel"
        prev="${LOCAL_MTIME[$f_remote]:-0}"
        if [ "$m_remote" != "$prev" ] || [ ! -f "$f_local" ]; then
            mkdir -p "$(dirname "$f_local")"
            if adb_safe_pull "$f_remote" "$f_local" >/dev/null 2>&1; then
                PULLED=$((PULLED + 1))
                printf '%s\t%s\n' "$f_remote" "$m_remote" >> "$NEW_DB"
            else
                PULL_ERRORS=$((PULL_ERRORS + 1))
                # Preserve the prior mtime so we retry next tick.
                [ "$prev" != "0" ] && printf '%s\t%s\n' "$f_remote" "$prev" >> "$NEW_DB"
            fi
        else
            SKIPPED=$((SKIPPED + 1))
            printf '%s\t%s\n' "$f_remote" "$m_remote" >> "$NEW_DB"
        fi
    done <<< "$REMOTE_LIST_RAW"

    mv "$NEW_DB" "$MTIME_DB"
else
    say "adb device offline; using local mirror as-is"
fi

# ---------------------------------------------------------------------------
# 3. Compute interim aggregates (pure Python, stdlib only).
#    Emits a single JSON line on stdout that the shell parses below.
# ---------------------------------------------------------------------------
AGG_JSON="$("$PY" - "$LOCAL_RUN_DIR" "$MANIFEST_JSON" <<'PYEOF' 2>/dev/null || echo '{}'
import json, math, os, sys, time
from pathlib import Path

run_dir = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])

try:
    manifest = json.loads(manifest_path.read_text())
except Exception:
    manifest = {}

# Defaults that match wave11_interim_plot.py's contract.
chunks_per_cell = manifest.get("n_ppl_chunks_per_cell_group", 8)
total_cells = 30                              # 3 models x 5 policies x 2 benches
total_expected_minutes = float(manifest.get("total_expected_minutes") or 1395.6)
# wave11_cells.json lists 240 cells (== 30 cell-groups x 8 chunks). Convert
# expected minutes/chunk into minutes/cell-group:
minutes_per_cell = total_expected_minutes / max(total_cells, 1)

def read_ppl(meta):
    try:
        v = float(json.loads(Path(meta).read_text()).get("perplexity"))
        return v if math.isfinite(v) and v > 0 else None
    except Exception:
        return None

cells = []
if run_dir.is_dir():
    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for policy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for bench_dir in sorted(p for p in policy_dir.iterdir() if p.is_dir()):
                bench = bench_dir.name.lower()
                if bench not in ("ppl", "niah"):
                    continue
                iter_dirs = sorted(p for p in bench_dir.iterdir()
                                   if p.is_dir() and p.name.startswith("iter"))
                done = 0
                started = len(iter_dirs)
                ppls = []
                cell_mtimes = []
                for it in iter_dirs:
                    meta = it / "meta.json"
                    if bench == "ppl":
                        v = read_ppl(meta)
                        if v is not None:
                            done += 1
                            ppls.append(v)
                            cell_mtimes.append(meta.stat().st_mtime)
                    else:
                        gen = it / "gen.txt"
                        if gen.exists() and gen.stat().st_size > 0:
                            done += 1
                            cell_mtimes.append(gen.stat().st_mtime)
                status = ("complete" if done >= chunks_per_cell
                          else ("partial" if started > 0 else "pending"))
                cells.append({
                    "model": model_dir.name,
                    "policy": policy_dir.name,
                    "bench": bench,
                    "done": done,
                    "started": started,
                    "ppls": ppls,
                    "last_mtime": max(cell_mtimes) if cell_mtimes else 0.0,
                    "status": status,
                })

n_complete = sum(1 for c in cells if c["status"] == "complete")
n_partial = sum(1 for c in cells if c["status"] == "partial")
n_pending_seen = sum(1 for c in cells if c["status"] == "pending")
# The denominator is the wave manifest size even if some cells haven't been
# created on disk yet.
remaining = max(0, total_cells - n_complete)

# Per-policy mean PPL (pool across models & chunks).
per_policy = {}
for c in cells:
    if c["bench"] != "ppl" or not c["ppls"]:
        continue
    bucket = per_policy.setdefault(c["policy"], {"n": 0, "sum": 0.0})
    for v in c["ppls"]:
        bucket["n"] += 1
        bucket["sum"] += v
per_policy_mean = {p: (b["sum"] / b["n"]) for p, b in per_policy.items() if b["n"] > 0}

# Per-policy progress (cells complete out of (models * benches) per policy).
per_policy_progress = {}
for c in cells:
    pp = per_policy_progress.setdefault(c["policy"], {"complete": 0, "total": 0})
    pp["total"] += 1
    if c["status"] == "complete":
        pp["complete"] += 1

# ETA: prefer observed mean-time-per-completed-cell, fall back to manifest.
done_mtimes = sorted(c["last_mtime"] for c in cells
                     if c["status"] == "complete" and c["last_mtime"] > 0)
observed_min_per_cell = None
if len(done_mtimes) >= 2:
    span_s = done_mtimes[-1] - done_mtimes[0]
    if span_s > 0:
        observed_min_per_cell = (span_s / max(len(done_mtimes) - 1, 1)) / 60.0
min_per_cell_used = observed_min_per_cell if observed_min_per_cell else minutes_per_cell
eta_minutes = remaining * min_per_cell_used
eta_hours = eta_minutes / 60.0

last_done_ts = max((c["last_mtime"] for c in cells), default=0.0)

out = {
    "cells_complete": n_complete,
    "cells_partial": n_partial,
    "cells_pending": max(0, total_cells - n_complete - n_partial),
    "cells_total": total_cells,
    "per_policy_mean_ppl": per_policy_mean,
    "per_policy_progress": per_policy_progress,
    "eta_hours": eta_hours,
    "min_per_cell_used": min_per_cell_used,
    "min_per_cell_source": "observed" if observed_min_per_cell else "manifest",
    "last_done_ts": last_done_ts,
}
sys.stdout.write(json.dumps(out))
PYEOF
)"

# ---------------------------------------------------------------------------
# 4. Print a single clean status line
# ---------------------------------------------------------------------------
SUMMARY="$("$PY" - <<PYEOF 2>/dev/null
import json, math, time
try:
    d = json.loads('''$AGG_JSON''') or {}
except Exception:
    d = {}
cells_c = d.get("cells_complete", 0)
cells_t = d.get("cells_total", 30)
cells_p = d.get("cells_partial", 0)
eta = d.get("eta_hours", float("nan"))
eta_s = f"{eta:.1f}h" if isinstance(eta, (int, float)) and math.isfinite(eta) else "n/a"
src = d.get("min_per_cell_source", "?")
mpc = d.get("min_per_cell_used", float("nan"))
mpc_s = f"{mpc:.1f} min/cell ({src})" if isinstance(mpc, (int, float)) and math.isfinite(mpc) else "?"
ppl_d = d.get("per_policy_mean_ppl", {}) or {}
ppl_s = (" ".join(f"{k}={v:.2f}" for k, v in sorted(ppl_d.items()))
         if ppl_d else "no PPL chunks yet")
last_ts = d.get("last_done_ts", 0.0) or 0.0
last_s = time.strftime("%H:%M:%S", time.localtime(last_ts)) if last_ts > 0 else "never"
print(f"cells {cells_c}/{cells_t} ({cells_p} partial) | "
      f"ppl: {ppl_s} | ETA {eta_s} ({mpc_s}) | last cell done {last_s}")
PYEOF
)"
if [ -z "$SUMMARY" ]; then
    SUMMARY="aggregator failed; mirror=$LOCAL_RUN_DIR"
fi
say "pulled=$PULLED skipped=$SKIPPED errors=$PULL_ERRORS | $SUMMARY"

# ---------------------------------------------------------------------------
# 5. Optional matplotlib render of the interim pareto
# ---------------------------------------------------------------------------
HAVE_MPL=0
if "$PY" -c "import matplotlib" >/dev/null 2>&1; then
    HAVE_MPL=1
fi

if [ "$HAVE_MPL" -eq 1 ]; then
    # Delegate the heavy lifting (ppl bar, niah bar, markdown table) to the
    # existing audit-produced module. Pass --no-sync so we don't re-pull —
    # we already did that above with rsync-style mtime tracking.
    if [ -f "$INTERIM_PY" ]; then
        "$PY" "$INTERIM_PY" --no-sync \
            --endurkv-root "$ENDURKV_ROOT" \
            --phone-logs-root "$PHONE_LOGS_LOCAL" \
            >/dev/null 2>&1 || say "wave11_interim_plot.py failed (non-fatal)"
    fi

    # Render the per-policy mean-PPL pareto. Tiny, deterministic, overwrites.
    "$PY" - "$AGG_JSON" "$PARETO_PNG" <<'PYEOF' 2>/dev/null || say "pareto render failed (non-fatal)"
import json, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

raw, out_path = sys.argv[1], sys.argv[2]
try:
    d = json.loads(raw) if raw else {}
except Exception:
    d = {}

ppl_d = d.get("per_policy_mean_ppl", {}) or {}
prog_d = d.get("per_policy_progress", {}) or {}
items = []
for policy, ppl in ppl_d.items():
    if not (isinstance(ppl, (int, float)) and math.isfinite(ppl)):
        continue
    pp = prog_d.get(policy, {}) or {}
    complete = pp.get("complete", 0)
    total = pp.get("total", 0) or 1
    items.append((policy, ppl, complete, total))

if not items:
    sys.exit(0)  # nothing meaningful to draw yet — leave any prior PNG in place

items.sort(key=lambda x: x[1])  # by ppl ascending
fig, ax = plt.subplots(figsize=(6.5, 4.2))
xs = [c / t * 100.0 for (_, _, c, t) in items]
ys = [ppl for (_, ppl, _, _) in items]
ax.scatter(xs, ys, s=90, edgecolor="black", linewidth=0.6, zorder=3)
for (p, ppl, c, t), x, y in zip(items, xs, ys):
    ax.annotate(f"{p}\nn={c}/{t}", (x, y),
                xytext=(6, 6), textcoords="offset points", fontsize=9)
ax.set_xlim(-5, 105)
ax.set_xlabel("Per-policy cells complete (%)")
ax.set_ylabel("Per-policy mean PPL (lower is better)")
ax.set_title(f"Wave-11 INTERIM pareto — "
             f"{d.get('cells_complete', 0)}/{d.get('cells_total', 30)} cells")
ax.grid(linestyle=":", alpha=0.5)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.tight_layout()
fig.savefig(out_path, dpi=150, bbox_inches="tight")
PYEOF
    if [ -f "$PARETO_PNG" ]; then
        say "pareto figure: $PARETO_PNG"
    fi
else
    say "matplotlib unavailable; skipping figure render"
fi

exit 0
