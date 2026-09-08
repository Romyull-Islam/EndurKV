#!/bin/bash
# host_wave11_smoke.sh — host-side end-to-end smoke test of phone_wave11_eval.sh.
#
# Goal:
#   Validate that the full Wave-11 launcher (preflight, cool gate, per-cell run,
#   sensors, watchdog, sweep logic, deliverables) works correctly with a tiny
#   restricted matrix in under 15 wall-clock minutes.
#
# Matrix (6 cells, not 30):
#   MODELS   = Llama-3.2-1B
#   POLICIES = vanilla v1 v1_fa2_stack
#   BENCHES  = ppl niah
#   PPL_N_CHUNKS    = 1   (single 2048-token chunk)
#   NIAH_N_STIMULI  = 1   (single Tier-1 stimulus)
#   COOL_MAX_S      = 60  (short cool gate so timeouts don't dominate)
#
# Validation gates (every check must pass to report PASS):
#   PRE-FLIGHT
#     * ADB device online.
#     * /data/local/tmp/endurkv/eval_data/wiki.test.raw.chunk0 present
#       (chunks 0..7 confirmed).
#     * NIAH stimuli present at /data/local/tmp/endurkv/eval_data/niah/
#       and at least one staged as niah_stimulus_00.txt for the launcher.
#     * On-device eviction_bench sha256 matches the host build at
#       entropy_probe/build-android/eviction_bench (current build proof).
#   RUNTIME
#     * Launcher invoked with WAVE11_FOREGROUND=1 so it runs synchronously.
#     * progress.log streamed to stdout while the launcher executes.
#   POST-RUN
#     (a) 6 "CELL DONE" markers in progress.log (matches the user's
#         "cell exit" criterion — the launcher emits "CELL DONE" lines).
#     (b) Every ppl cell's stress.csv has a non-zero ppl column.
#     (c) Every niah cell's gen.txt is > 50 bytes.
#     (d) watchdog.log exists ONLY under v1_fa2_stack cells.
#     (e) No leftover preempt_throttle_watchdog or sample_sensors processes.
#
# Exit codes:
#   0  PASS — all gates green.
#   1  FAIL — at least one gate failed. The script prints the exact reason.
#   2  ABORTED — environment / preflight unrecoverable (e.g., no device).
#
# Usage:
#   bash /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/host_wave11_smoke.sh

set -u
set -o pipefail

# ---------------------------------------------------------------------------
# Resilient ADB helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=adb_resilient.sh
source "$SCRIPT_DIR/adb_resilient.sh"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHONE_WORKDIR="/data/local/tmp/endurkv"
PHONE_BIN="$PHONE_WORKDIR/bin_cpu/eviction_bench"
PHONE_LAUNCHER="$PHONE_WORKDIR/scripts/phone_wave11_eval.sh"
HOST_BIN="$SCRIPT_DIR/../../entropy_probe/build-android/eviction_bench"
HOST_LAUNCHER="$SCRIPT_DIR/phone_wave11_eval.sh"

PHONE_EVAL_DATA="$PHONE_WORKDIR/eval_data"
PHONE_NIAH_DIR="$PHONE_EVAL_DATA/niah"
PHONE_PPL_CHUNK0="$PHONE_EVAL_DATA/wiki.test.raw.chunk0"

# Smoke output goes to a dedicated, timestamped directory so it never
# collides with a real Wave-11 sweep.
SMOKE_TS="$(date +%Y%m%d_%H%M%S)"
PHONE_OUT_DIR="$PHONE_WORKDIR/logs/wave11_smoke_$SMOKE_TS"
HOST_PULL_DIR="/tmp/wave11_smoke_$SMOKE_TS"
mkdir -p "$HOST_PULL_DIR"

PROGRESS_PHONE="$PHONE_OUT_DIR/progress.log"
PROGRESS_HOST="$HOST_PULL_DIR/progress.log"

# Hard wall-clock cap for the launcher itself (just under the 15 min budget).
LAUNCHER_MAX_S=720   # 12 minutes
STREAM_POLL_S=5

# Restricted sweep
SMOKE_MODELS="Llama-3.2-1B|$PHONE_WORKDIR/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf|4096"
SMOKE_POLICIES="vanilla v1 v1_fa2_stack"
SMOKE_BENCHES="ppl niah"
SMOKE_PPL_CHUNKS=1
SMOKE_NIAH=1
SMOKE_COOL_MAX_S=60

EXPECTED_CELLS=6   # 1 model x 3 policies x 2 benches

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
say()  { printf '[smoke %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
fail() { printf '[smoke FAIL] %s\n' "$*" >&2; FAIL_REASONS+=("$*"); }
note() { printf '[smoke note] %s\n' "$*"; }

FAIL_REASONS=()

# Print a section header so the log is readable.
section() { printf '\n========== %s ==========\n' "$*"; }

# Return the value of FAIL_REASONS as a single string. POSIX-safe.
join_fails() {
    local IFS='; '
    printf '%s' "${FAIL_REASONS[*]}"
}

# ---------------------------------------------------------------------------
# Gate 0: ADB device online
# ---------------------------------------------------------------------------
section "GATE 0 — ADB device"
if ! adb get-state 2>/dev/null | grep -q '^device$'; then
    say "no device online; waiting up to 60s via adb_wait..."
    # adb_wait blocks indefinitely; bound it ourselves.
    ( adb_wait ) &
    WAIT_PID=$!
    waited=0
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        if [ "$waited" -ge 60 ]; then
            kill -9 "$WAIT_PID" 2>/dev/null
            say "ABORTED: no device after 60s"
            exit 2
        fi
        sleep 2
        waited=$((waited + 2))
    done
fi
say "device online: $(adb get-serialno 2>/dev/null)"

# ---------------------------------------------------------------------------
# Gate 1: PPL chunk0 (and chunks 0..7) present
# ---------------------------------------------------------------------------
section "GATE 1 — PPL chunk presence"
if ! adb_safe_shell "test -f $PHONE_PPL_CHUNK0 && echo OK" | grep -q OK; then
    fail "missing $PHONE_PPL_CHUNK0"
else
    say "OK: $PHONE_PPL_CHUNK0 exists"
fi

# Spot-check chunks 0..7 (user said 8 were pushed)
missing_chunks=""
for i in 0 1 2 3 4 5 6 7; do
    if ! adb_safe_shell "test -f $PHONE_EVAL_DATA/wiki.test.raw.chunk${i} && echo OK" | grep -q OK; then
        missing_chunks="$missing_chunks chunk${i}"
    fi
done
if [ -n "$missing_chunks" ]; then
    fail "missing PPL chunks:$missing_chunks"
else
    say "OK: PPL chunks 0..7 present"
fi

# ---------------------------------------------------------------------------
# Gate 2: NIAH stimuli present (and stage 1 with launcher-expected name)
# ---------------------------------------------------------------------------
section "GATE 2 — NIAH stimuli"
NIAH_COUNT=$(adb_safe_shell "ls $PHONE_NIAH_DIR/*.txt 2>/dev/null | wc -l" | tr -d '[:space:]')
if [ -z "$NIAH_COUNT" ] || [ "$NIAH_COUNT" -lt 1 ] 2>/dev/null; then
    fail "no NIAH stimuli in $PHONE_NIAH_DIR"
else
    say "OK: $NIAH_COUNT NIAH stimulus files in $PHONE_NIAH_DIR"
fi

# The launcher expects files named niah_stimulus_NN.txt.  The pushed corpus
# uses the niah_cCTX_dDD.txt naming convention.  Stage a single canonical
# stimulus (the launcher only needs NIAH_N_STIMULI=1 here).  Pick a 4096-ctx
# file because the smoke matrix uses ctx=4096.
SOURCE_STIM="$PHONE_NIAH_DIR/niah_c4096_d50.txt"
TARGET_STIM="$PHONE_NIAH_DIR/niah_stimulus_00.txt"
if ! adb_safe_shell "test -f $TARGET_STIM && echo OK" | grep -q OK; then
    if adb_safe_shell "test -f $SOURCE_STIM && echo OK" | grep -q OK; then
        adb_safe_shell "cp $SOURCE_STIM $TARGET_STIM" >/dev/null
        if adb_safe_shell "test -f $TARGET_STIM && echo OK" | grep -q OK; then
            say "staged $TARGET_STIM (copy of niah_c4096_d50.txt)"
        else
            fail "could not stage $TARGET_STIM"
        fi
    else
        fail "no source NIAH file to stage (expected $SOURCE_STIM)"
    fi
else
    say "OK: $TARGET_STIM already staged"
fi

# ---------------------------------------------------------------------------
# Gate 3: eviction_bench sha256 (phone == host build)
# ---------------------------------------------------------------------------
section "GATE 3 — eviction_bench is current build"
if [ ! -f "$HOST_BIN" ]; then
    fail "host build missing: $HOST_BIN"
else
    HOST_SHA="$(sha256sum "$HOST_BIN" | awk '{print $1}')"
    PHONE_SHA="$(adb_safe_shell "sha256sum $PHONE_BIN 2>/dev/null" | awk '{print $1}')"
    if [ -z "$PHONE_SHA" ]; then
        fail "phone eviction_bench missing or sha256 unavailable"
    elif [ "$HOST_SHA" != "$PHONE_SHA" ]; then
        fail "sha256 mismatch host=$HOST_SHA phone=$PHONE_SHA — phone binary is not current build"
    else
        say "OK: sha256 match ($HOST_SHA)"
    fi
fi

# ---------------------------------------------------------------------------
# Push launcher (every smoke run gets a fresh copy)
# ---------------------------------------------------------------------------
section "STAGE — push launcher"
if [ ! -f "$HOST_LAUNCHER" ]; then
    fail "host launcher missing: $HOST_LAUNCHER"
else
    if adb_safe_push "$HOST_LAUNCHER" "$PHONE_LAUNCHER"; then
        adb_safe_shell "chmod 755 $PHONE_LAUNCHER" >/dev/null
        say "OK: launcher pushed to $PHONE_LAUNCHER"
    else
        fail "could not push launcher to phone"
    fi
fi

# Bail out early if any pre-launch gate failed: running the launcher would just
# eat the whole budget on a broken environment.
if [ "${#FAIL_REASONS[@]}" -gt 0 ]; then
    section "RESULT"
    echo "FAIL: pre-launch gates: $(join_fails)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Gate 4: launch the wave11 launcher in FOREGROUND mode
# ---------------------------------------------------------------------------
section "GATE 4 — launcher (foreground, restricted matrix)"
say "OUT_DIR on phone: $PHONE_OUT_DIR"
say "starting launcher (cap=${LAUNCHER_MAX_S}s) ..."

# We run the launcher on the phone with WAVE11_FOREGROUND=1 so it returns
# synchronously, then immediately background that adb shell invocation on the
# host so we can stream progress.log in parallel.  We capture the launcher's
# exit code via a sentinel file on the phone.
RUN_LOG="$HOST_PULL_DIR/launcher_run.log"
RC_PHONE_FILE="$PHONE_OUT_DIR/.launcher_rc"
adb_safe_shell "mkdir -p $PHONE_OUT_DIR" >/dev/null

# Build env-var prefix the launcher reads.
ENV_PREFIX="WAVE11_FOREGROUND=1 \
OUT_DIR=$PHONE_OUT_DIR \
MODELS='$SMOKE_MODELS' \
POLICIES='$SMOKE_POLICIES' \
BENCHES='$SMOKE_BENCHES' \
PPL_N_CHUNKS=$SMOKE_PPL_CHUNKS \
NIAH_N_STIMULI=$SMOKE_NIAH \
NIAH_TIER=1 \
COOL_MAX_S=$SMOKE_COOL_MAX_S"

# Background the launcher (adb shell call); foreground-mode in the script means
# it executes main() inline rather than re-execing under setsid.
( adb shell "$ENV_PREFIX sh $PHONE_LAUNCHER; echo \$? > $RC_PHONE_FILE" \
    > "$RUN_LOG" 2>&1 ) &
LAUNCH_PID=$!

# ---------------------------------------------------------------------------
# Gate 5: stream progress.log to stdout while launcher runs
# ---------------------------------------------------------------------------
section "GATE 5 — streaming progress.log"
LAST_LINES=0
T0="$(date +%s)"
while kill -0 "$LAUNCH_PID" 2>/dev/null; do
    NOW="$(date +%s)"
    if [ "$((NOW - T0))" -ge "$LAUNCHER_MAX_S" ]; then
        say "WARN: launcher exceeded ${LAUNCHER_MAX_S}s — killing"
        # Kill local adb shell process.
        kill -9 "$LAUNCH_PID" 2>/dev/null
        # And kill any phone-side processes the launcher may have spawned.
        adb_safe_shell "pkill -f phone_wave11_eval || true" >/dev/null 2>&1
        adb_safe_shell "pkill -f eviction_bench    || true" >/dev/null 2>&1
        adb_safe_shell "pkill -f sample_sensors    || true" >/dev/null 2>&1
        break
    fi

    # Pull the current progress.log (may not exist yet during the very first
    # second; ignore errors).
    if adb pull -q "$PROGRESS_PHONE" "$PROGRESS_HOST" 2>/dev/null; then
        TOTAL_LINES="$(wc -l < "$PROGRESS_HOST" 2>/dev/null || echo 0)"
        if [ "$TOTAL_LINES" -gt "$LAST_LINES" ] 2>/dev/null; then
            NEW=$((TOTAL_LINES - LAST_LINES))
            tail -n "$NEW" "$PROGRESS_HOST" | sed 's/^/[phone] /'
            LAST_LINES="$TOTAL_LINES"
        fi
    fi
    sleep "$STREAM_POLL_S"
done

wait "$LAUNCH_PID" 2>/dev/null || true

# Final pull of progress.log so we have the last lines.
adb pull -q "$PROGRESS_PHONE" "$PROGRESS_HOST" 2>/dev/null || true
if [ -f "$PROGRESS_HOST" ]; then
    TOTAL_LINES="$(wc -l < "$PROGRESS_HOST")"
    if [ "$TOTAL_LINES" -gt "$LAST_LINES" ] 2>/dev/null; then
        NEW=$((TOTAL_LINES - LAST_LINES))
        tail -n "$NEW" "$PROGRESS_HOST" | sed 's/^/[phone] /'
    fi
fi

LAUNCHER_RC="$(adb_safe_shell "cat $RC_PHONE_FILE 2>/dev/null" | head -1 | tr -d '[:space:]')"
[ -z "$LAUNCHER_RC" ] && LAUNCHER_RC="UNKNOWN"
say "launcher exit code: $LAUNCHER_RC"

# ---------------------------------------------------------------------------
# Pull the smoke OUT_DIR for offline inspection (even on failure).
# ---------------------------------------------------------------------------
section "POST — pull smoke OUT_DIR"
if adb_safe_pull "$PHONE_OUT_DIR" "$HOST_PULL_DIR/"; then
    LOCAL_OUT="$HOST_PULL_DIR/$(basename "$PHONE_OUT_DIR")"
    say "OK: pulled to $LOCAL_OUT"
else
    LOCAL_OUT=""
    fail "could not pull $PHONE_OUT_DIR"
fi

# ---------------------------------------------------------------------------
# Validation (a): exactly EXPECTED_CELLS "CELL DONE" markers in progress.log
# The user described these as "cell exit" lines; the launcher emits the line
#   === CELL DONE  model=... policy=... bench=... ===
# at the end of every cell.  That marker is the canonical "cell exited" event.
# ---------------------------------------------------------------------------
section "VALIDATE (a) — 6 cell-exit markers in progress.log"
if [ ! -f "$PROGRESS_HOST" ]; then
    fail "progress.log not pulled"
else
    CELL_DONE=$(grep -c 'CELL DONE' "$PROGRESS_HOST" || true)
    say "found $CELL_DONE 'CELL DONE' markers (expected $EXPECTED_CELLS)"
    if [ "$CELL_DONE" -ne "$EXPECTED_CELLS" ]; then
        fail "(a) expected $EXPECTED_CELLS CELL DONE lines, got $CELL_DONE"
    fi
fi

# ---------------------------------------------------------------------------
# Validation (b): every ppl stress.csv has at least one non-zero ppl value
# stress.csv schema (PPL row):
#   chunk_idx,t_elapsed_s,exit,prefill_ms,decode_tps,n_decode_steps,
#   peak_kv_cells,peak_rss_kb,evicted,ppl,niah_correct,k_used,
#   ddr_start_c,mem_free_gb_start
#   columns:    1         2          3    4          5         6
#               7              8           9       10  11           12
#               13           14
# We check column 10 (ppl) for at least one numeric value != 0 (and != "0"
# with optional decimal) across the data rows.
# ---------------------------------------------------------------------------
section "VALIDATE (b) — ppl != 0 in every ppl/stress.csv"
if [ -n "$LOCAL_OUT" ]; then
    PPL_CSVS=$(find "$LOCAL_OUT" -type f -path '*/ppl/stress.csv' | sort)
    if [ -z "$PPL_CSVS" ]; then
        fail "(b) no ppl/stress.csv files found under $LOCAL_OUT"
    else
        while IFS= read -r CSV; do
            # Skip header, look at column 10 (ppl).
            NZ=$(awk -F',' 'NR>1 {
                v=$10; gsub(/[[:space:]]/,"",v);
                if (v != "" && v != "0" && v+0 != 0) found=1;
            } END {print (found?"yes":"no")}' "$CSV")
            if [ "$NZ" != "yes" ]; then
                fail "(b) ppl all zero in $CSV"
            else
                say "OK ppl: $(echo "$CSV" | sed "s|$LOCAL_OUT/||")"
            fi
        done <<< "$PPL_CSVS"
    fi
else
    fail "(b) skipped — no local OUT_DIR"
fi

# ---------------------------------------------------------------------------
# Validation (c): every niah gen.txt > 50 bytes
# ---------------------------------------------------------------------------
section "VALIDATE (c) — niah gen.txt > 50 bytes"
if [ -n "$LOCAL_OUT" ]; then
    GEN_FILES=$(find "$LOCAL_OUT" -type f -path '*/niah/iter*/gen.txt' | sort)
    if [ -z "$GEN_FILES" ]; then
        fail "(c) no niah gen.txt files found under $LOCAL_OUT"
    else
        while IFS= read -r GEN; do
            SZ=$(wc -c < "$GEN" 2>/dev/null || echo 0)
            if [ "$SZ" -le 50 ] 2>/dev/null; then
                fail "(c) $GEN is $SZ bytes (<=50)"
            else
                say "OK gen ($SZ B): $(echo "$GEN" | sed "s|$LOCAL_OUT/||")"
            fi
        done <<< "$GEN_FILES"
    fi
else
    fail "(c) skipped — no local OUT_DIR"
fi

# ---------------------------------------------------------------------------
# Validation (d): watchdog.log exists ONLY under v1_fa2_stack cells
# ---------------------------------------------------------------------------
section "VALIDATE (d) — watchdog.log scope"
if [ -n "$LOCAL_OUT" ]; then
    WD_FILES=$(find "$LOCAL_OUT" -type f -name 'watchdog.log' | sort)
    if [ -z "$WD_FILES" ]; then
        fail "(d) no watchdog.log files found — v1_fa2_stack should have emitted some"
    else
        STRAY=""
        V1FA_COUNT=0
        while IFS= read -r WD; do
            if echo "$WD" | grep -q '/v1_fa2_stack/'; then
                V1FA_COUNT=$((V1FA_COUNT + 1))
                say "OK watchdog: $(echo "$WD" | sed "s|$LOCAL_OUT/||")"
            else
                STRAY="$STRAY $WD"
            fi
        done <<< "$WD_FILES"
        if [ -n "$STRAY" ]; then
            fail "(d) watchdog.log found outside v1_fa2_stack:$STRAY"
        fi
        # Expect exactly 2 watchdog.log files (v1_fa2_stack x 2 benches).
        if [ "$V1FA_COUNT" -ne 2 ]; then
            fail "(d) expected 2 watchdog.log under v1_fa2_stack, got $V1FA_COUNT"
        fi
    fi
else
    fail "(d) skipped — no local OUT_DIR"
fi

# ---------------------------------------------------------------------------
# Validation (e): no leftover watchdog (or sampler) processes on the phone
# ---------------------------------------------------------------------------
section "VALIDATE (e) — no leftover watchdog/sampler procs on phone"
LEFTOVER_WD=$(adb_safe_shell "pgrep -fa preempt_throttle_watchdog 2>/dev/null" | grep -v 'pgrep' || true)
LEFTOVER_SS=$(adb_safe_shell "pgrep -fa sample_sensors 2>/dev/null"         | grep -v 'pgrep' || true)
if [ -n "$LEFTOVER_WD" ]; then
    fail "(e) leftover watchdog procs: $LEFTOVER_WD"
else
    say "OK: no leftover watchdog procs"
fi
if [ -n "$LEFTOVER_SS" ]; then
    fail "(e) leftover sampler procs: $LEFTOVER_SS"
else
    say "OK: no leftover sampler procs"
fi

# Also surface the launcher rc as a soft check; non-zero is suspicious but the
# detailed validations above are the authoritative gates.
if [ "$LAUNCHER_RC" != "0" ]; then
    note "launcher exit code was $LAUNCHER_RC (non-zero) — see $RUN_LOG"
fi

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
section "RESULT"
if [ "${#FAIL_REASONS[@]}" -eq 0 ]; then
    echo "PASS — wave11 launcher end-to-end smoke green ($EXPECTED_CELLS cells)"
    echo "       smoke OUT_DIR (host): $LOCAL_OUT"
    echo "       smoke OUT_DIR (phone): $PHONE_OUT_DIR"
    exit 0
else
    echo "FAIL — gates that failed:"
    for r in "${FAIL_REASONS[@]}"; do
        echo "  - $r"
    done
    echo "  smoke OUT_DIR (host): $LOCAL_OUT"
    echo "  smoke OUT_DIR (phone): $PHONE_OUT_DIR"
    echo "  launcher stdout/stderr: $RUN_LOG"
    exit 1
fi
