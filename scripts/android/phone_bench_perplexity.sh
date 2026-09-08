#!/system/bin/sh
# phone_bench_perplexity.sh — Vanilla llama.cpp perplexity benchmark.
#
# Runs llama-perplexity on a corpus file with thermal/memory/timing capture.
# Use this for perplexity metric (separate from generation latency runs).
#
# Usage:
#   sh scripts/phone_bench_perplexity.sh \
#       --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
#       --corpus corpora/wiki.test.raw \
#       --tag wiki2_llama1b_ctx4k \
#       --ctx-size 4096 \
#       --out-dir logs/ppl_$(date +%s)

set -e

MODEL=""
CORPUS=""
TAG=""
CTX_SIZE=4096
OUT_DIR=""
N_THREADS=4
SAMPLER="./scripts/sample_sensors.sh"

while [ $# -gt 0 ]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --corpus)     CORPUS="$2"; shift 2 ;;
        --tag)        TAG="$2"; shift 2 ;;
        --ctx-size)   CTX_SIZE="$2"; shift 2 ;;
        --out-dir)    OUT_DIR="$2"; shift 2 ;;
        --threads)    N_THREADS="$2"; shift 2 ;;
        --sampler)    SAMPLER="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -z "$MODEL" ] && { echo "--model required" >&2; exit 1; }
[ -z "$CORPUS" ] && { echo "--corpus required" >&2; exit 1; }
[ -z "$TAG" ] && { echo "--tag required" >&2; exit 1; }
[ -z "$OUT_DIR" ] && { echo "--out-dir required" >&2; exit 1; }

mkdir -p "$OUT_DIR"

ROOT=/data/local/tmp/endurkv
BIN="$ROOT/bin"
LLAMA_PPL="$BIN/llama-perplexity"
export LD_LIBRARY_PATH="$BIN"

PPL_TXT="$OUT_DIR/${TAG}.ppl.txt"
SENSORS_CSV="$OUT_DIR/${TAG}.sensors.csv"
MEM_CSV="$OUT_DIR/${TAG}.mem.csv"
META_JSON="$OUT_DIR/${TAG}.meta.json"

# Start sensor sampling
sh "$SAMPLER" --out "$SENSORS_CSV" --hz 10 &
SAMPLER_PID=$!

START_WALL=$(date +%s.%N)

# llama-perplexity flags:
#   -m model -f corpus_file -c ctx_size --perplexity is default
"$LLAMA_PPL" \
    -m "$MODEL" \
    -f "$CORPUS" \
    -c "$CTX_SIZE" \
    -t "$N_THREADS" \
    --perplexity \
    > "$PPL_TXT" 2>&1 &
LLAMA_PID=$!

echo "wall_clock_s,rss_kb,vmpeak_kb,vmsize_kb,vmhwm_kb" > "$MEM_CSV"
while kill -0 "$LLAMA_PID" 2>/dev/null; do
    if [ -f /proc/$LLAMA_PID/status ]; then
        rss=$(awk '/^VmRSS:/{print $2}' /proc/$LLAMA_PID/status)
        vmpk=$(awk '/^VmPeak:/{print $2}' /proc/$LLAMA_PID/status)
        vmsz=$(awk '/^VmSize:/{print $2}' /proc/$LLAMA_PID/status)
        vmhw=$(awk '/^VmHWM:/{print $2}' /proc/$LLAMA_PID/status)
        echo "$(date +%s.%N),$rss,$vmpk,$vmsz,$vmhw" >> "$MEM_CSV"
    fi
    sleep 0.2
done
wait "$LLAMA_PID"
EXIT_CODE=$?
END_WALL=$(date +%s.%N)
kill "$SAMPLER_PID" 2>/dev/null || true

cat > "$META_JSON" <<EOF
{
  "tag":           "$TAG",
  "model":         "$MODEL",
  "corpus":        "$CORPUS",
  "ctx_size":      $CTX_SIZE,
  "n_threads":     $N_THREADS,
  "policy":        "vanilla",
  "metric":        "perplexity",
  "start_wall_s":  $START_WALL,
  "end_wall_s":    $END_WALL,
  "exit_code":     $EXIT_CODE
}
EOF

echo "[bench_ppl] DONE pid=$LLAMA_PID exit=$EXIT_CODE"
echo "===== perplexity summary ====="
grep -E "^Final estimate|estimate" "$PPL_TXT" | tail -5
echo "===== peak memory ====="
sort -t, -k2 -n "$MEM_CSV" | tail -3
exit $EXIT_CODE
