#!/usr/bin/env bash
# Phase B — measure probe overhead.
# Builds two variants of entropy_probe (ENABLE_PROBE on/off), runs each 10 times
# on the same 128-token prompt, and reports mean decode times + overhead %.
# Run from the EndurKV root:  bash scripts/06_probe_overhead.sh
#
# Pins to a single GPU (CUDA_VISIBLE_DEVICES=0) so the overhead numbers aren't
# polluted by 8-way pipeline-parallel init/sync. Prints progress to stderr.

set -e
cd "$(dirname "$0")/.."

ROOT="$(pwd)"
SRC="$ROOT/entropy_probe"
BUILD_ON="$SRC/build"
BUILD_OFF="$SRC/build-noprobe"
MODEL=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
N_RUNS=10
N_TOKENS=128

[ -f "$MODEL" ] || { echo "ERROR: $MODEL missing."; exit 1; }

# --- Build the off variant if not already built ---
if [ ! -x "$BUILD_OFF/entropy_probe" ]; then
    echo "=== configure ENABLE_PROBE=OFF ==="
    cmake -S "$SRC" -B "$BUILD_OFF" \
        -DLLAMA_CPP_DIR="$ROOT/llama.cpp" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_PROBE=OFF
    echo "=== build ==="
    cmake --build "$BUILD_OFF" -j
fi

[ -x "$BUILD_ON/entropy_probe"  ] || { echo "ERROR: $BUILD_ON/entropy_probe not built. Run scripts/04_build_probe.sh first."; exit 1; }
[ -x "$BUILD_OFF/entropy_probe" ] || { echo "ERROR: $BUILD_OFF/entropy_probe build failed."; exit 1; }

mkdir -p logs/overhead
PROMPT=$(mktemp /tmp/probe_oh.XXXX.txt)
trap "rm -f $PROMPT" EXIT
printf 'The capital of France is' > "$PROMPT"

# Pin to a single GPU so model load is fast and timings are stable.
export CUDA_VISIBLE_DEVICES=0
echo "[overhead] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, n_runs=$N_RUNS, n_tokens=$N_TOKENS" >&2

# Warmup (cache + GPU init) for both variants.
echo "[overhead] warmup..." >&2
"$BUILD_ON/entropy_probe"  --model "$MODEL" --prompt-file "$PROMPT" --prompt-id warm --max-tokens 8 --seed 1 --output logs/overhead/warm_on.csv  >/dev/null 2>&1 || true
"$BUILD_OFF/entropy_probe" --model "$MODEL" --prompt-file "$PROMPT" --prompt-id warm --max-tokens 8 --seed 1 --output /dev/null              >/dev/null 2>&1 || true

# Run N_RUNS iterations of one binary, write per-step decode time to TSV-like list.
# Prints progress to stderr; final TIMES list to stdout.
run_n() {
    local BIN="$1" LABEL="$2" STASH="$3"
    mkdir -p "$STASH"
    local TIMES=""
    local STEPS=""
    for i in $(seq 1 "$N_RUNS"); do
        local LOG="$STASH/${LABEL}_run${i}.stderr"
        local CSV="$STASH/${LABEL}_run${i}.csv"
        echo "  [$LABEL] run $i/$N_RUNS ..." >&2
        "$BIN" --model "$MODEL" --prompt-file "$PROMPT" \
               --prompt-id "${LABEL}-${i}" --max-tokens "$N_TOKENS" \
               --seed 42 --output "$CSV" 2>"$LOG"
        local TOT
        TOT=$(grep -oE 'total_ms=[0-9.]+'           "$LOG" | sed 's/total_ms=//')
        local STP
        STP=$(grep -oE 'decode_ms_per_step=[0-9.]+' "$LOG" | sed 's/decode_ms_per_step=//')
        echo "    total_ms=$TOT  ms_per_step=$STP" >&2
        TIMES="$TIMES $TOT"
        STEPS="$STEPS $STP"
    done
    # Emit machine-readable lines for the caller to parse.
    echo "TIMES_$LABEL:$TIMES"
    echo "STEPS_$LABEL:$STEPS"
}

echo "[overhead] === probe-OFF ===" >&2
OFF_OUT=$(run_n "$BUILD_OFF/entropy_probe" off logs/overhead)
echo "$OFF_OUT"

echo "[overhead] === probe-ON ===" >&2
ON_OUT=$(run_n "$BUILD_ON/entropy_probe"  on  logs/overhead)
echo "$ON_OUT"

# Extract mean of ms_per_step for each variant.
mean_of() {
    awk -v vals="$1" 'BEGIN{
        n = split(vals, a, " ");
        s = 0; for (i=1;i<=n;i++) s += a[i];
        printf "%.5f\n", (n>0 ? s/n : 0);
    }'
}

OFF_STEPS=$(echo "$OFF_OUT" | awk -F: '/^STEPS_off:/ {print $2}')
ON_STEPS=$( echo "$ON_OUT"  | awk -F: '/^STEPS_on:/  {print $2}')
MEAN_OFF=$(mean_of "$OFF_STEPS")
MEAN_ON=$( mean_of "$ON_STEPS")

echo
echo "=== summary (mean over $N_RUNS runs, max_tokens=$N_TOKENS, single GPU) ==="
awk -v on="$MEAN_ON" -v off="$MEAN_OFF" 'BEGIN{
    overhead = (on - off) / off * 100.0;
    printf "mean_decode_time_no_probe_ms_per_step   = %.4f\n", off;
    printf "mean_decode_time_with_probe_ms_per_step = %.4f\n", on;
    printf "overhead_percent                        = %.3f %%\n", overhead;
}'
