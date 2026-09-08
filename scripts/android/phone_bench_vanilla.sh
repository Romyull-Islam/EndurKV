#!/system/bin/sh
# phone_bench_vanilla.sh — Vanilla llama.cpp baseline benchmark, on-phone.
#
# Captures all four metric streams per prompt:
#   1. Latency: prefill/decode tokens/sec (from llama-completion timing output)
#   2. Perplexity: per-chunk PPL via llama-perplexity (separate run on corpus)
#   3. Memory: RSS peak via /proc/$PID/status sampled at 5 Hz
#   4. Thermal/battery: existing sample_sensors.sh at 10 Hz
#
# Layout on phone (after adb push):
#   /data/local/tmp/endurkv/
#     bin/                  llama-completion, llama-perplexity, llama-bench, *.so
#     models/               *.gguf
#     prompts/              *.txt (each prompt as its own file)
#     scripts/              this script + sample_sensors.sh
#     logs/                 per-run output
#
# Usage on phone:
#   cd /data/local/tmp/endurkv
#   sh scripts/phone_bench_vanilla.sh \
#       --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
#       --prompt prompts/narrativeqa_lc_01.txt \
#       --prompt-id narrativeqa_lc_01 \
#       --ctx-size 4096 \
#       --max-tokens 64 \
#       --out-dir logs/run_$(date +%s)
#
# Output (in --out-dir):
#   prompt_id.timing.txt    raw stderr from llama-completion
#   prompt_id.gen.txt       generated text
#   prompt_id.mem.csv       per-sample RSS (5 Hz) during the run
#   prompt_id.sensors.csv   per-sample thermal+battery+UFS (10 Hz)
#   prompt_id.meta.json     start/end wall clock, exit code, model+ctx params

set -e

# ------ defaults / args ------
MODEL=""
PROMPT=""
PROMPT_ID=""
CTX_SIZE=4096
MAX_TOKENS=64
SEED=42
OUT_DIR=""
SAMPLER="./scripts/sample_sensors.sh"
SENSORS_HZ=10
MEM_HZ=5
N_THREADS=4   # default conservative; tune per phone

while [ $# -gt 0 ]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --prompt)       PROMPT="$2"; shift 2 ;;
        --prompt-id)    PROMPT_ID="$2"; shift 2 ;;
        --ctx-size)     CTX_SIZE="$2"; shift 2 ;;
        --max-tokens)   MAX_TOKENS="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --out-dir)      OUT_DIR="$2"; shift 2 ;;
        --sampler)      SAMPLER="$2"; shift 2 ;;
        --sensors-hz)   SENSORS_HZ="$2"; shift 2 ;;
        --mem-hz)       MEM_HZ="$2"; shift 2 ;;
        --threads)      N_THREADS="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -z "$MODEL" ] && { echo "--model required" >&2; exit 1; }
[ -z "$PROMPT" ] && { echo "--prompt required" >&2; exit 1; }
[ -z "$PROMPT_ID" ] && { echo "--prompt-id required" >&2; exit 1; }
[ -z "$OUT_DIR" ] && { echo "--out-dir required" >&2; exit 1; }

mkdir -p "$OUT_DIR"

# ------ paths ------
ROOT=/data/local/tmp/endurkv
BIN="$ROOT/bin"
LLAMA_BIN="$BIN/llama-completion"
export LD_LIBRARY_PATH="$BIN"

# ------ start sensor sampler (background) ------
SENSORS_CSV="$OUT_DIR/${PROMPT_ID}.sensors.csv"
sh "$SAMPLER" --out "$SENSORS_CSV" --hz "$SENSORS_HZ" &
SAMPLER_PID=$!

# ------ record start ------
START_WALL=$(date +%s.%N)
START_MONO=$(awk '{print $1}' /proc/uptime)

# ------ launch llama-completion in background, then sample memory ------
TIMING_TXT="$OUT_DIR/${PROMPT_ID}.timing.txt"
GEN_TXT="$OUT_DIR/${PROMPT_ID}.gen.txt"
MEM_CSV="$OUT_DIR/${PROMPT_ID}.mem.csv"

# Read the prompt file (llama-completion needs prompt via stdin or -f)
PROMPT_FILE="$PROMPT"

# llama-completion CLI:
#   -m model
#   -f prompt-file  (prompt text from file)
#   -c ctx-size
#   -n n-predict (max new tokens)
#   -s seed
#   -t threads
#   --no-warmup (skip warmup pass)
#   --no-conversation (single-shot mode, not chat)
"$LLAMA_BIN" \
    -m "$MODEL" \
    -f "$PROMPT_FILE" \
    -c "$CTX_SIZE" \
    -n "$MAX_TOKENS" \
    -s "$SEED" \
    -t "$N_THREADS" \
    --no-warmup \
    --no-conversation \
    > "$GEN_TXT" 2> "$TIMING_TXT" &
LLAMA_PID=$!

# Sample memory at MEM_HZ Hz while llama runs
echo "wall_clock_s,rss_kb,vmpeak_kb,vmsize_kb,vmhwm_kb" > "$MEM_CSV"
PERIOD=$(awk -v hz="$MEM_HZ" 'BEGIN{printf "%.3f", 1.0/hz}')
while kill -0 "$LLAMA_PID" 2>/dev/null; do
    if [ -f /proc/$LLAMA_PID/status ]; then
        rss=$(awk '/^VmRSS:/{print $2}' /proc/$LLAMA_PID/status)
        vmpk=$(awk '/^VmPeak:/{print $2}' /proc/$LLAMA_PID/status)
        vmsz=$(awk '/^VmSize:/{print $2}' /proc/$LLAMA_PID/status)
        vmhw=$(awk '/^VmHWM:/{print $2}' /proc/$LLAMA_PID/status)
        now=$(date +%s.%N)
        echo "$now,$rss,$vmpk,$vmsz,$vmhw" >> "$MEM_CSV"
    fi
    sleep $PERIOD
done
wait "$LLAMA_PID"
EXIT_CODE=$?

# ------ stop sensor sampler ------
kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true

# ------ record end ------
END_WALL=$(date +%s.%N)
END_MONO=$(awk '{print $1}' /proc/uptime)

# ------ write metadata ------
META_JSON="$OUT_DIR/${PROMPT_ID}.meta.json"
cat > "$META_JSON" <<EOF
{
  "prompt_id":     "$PROMPT_ID",
  "model":         "$MODEL",
  "prompt_file":   "$PROMPT_FILE",
  "ctx_size":      $CTX_SIZE,
  "max_tokens":    $MAX_TOKENS,
  "seed":          $SEED,
  "n_threads":     $N_THREADS,
  "policy":        "vanilla",
  "start_wall_s":  $START_WALL,
  "end_wall_s":    $END_WALL,
  "start_mono_s":  $START_MONO,
  "end_mono_s":    $END_MONO,
  "exit_code":     $EXIT_CODE
}
EOF

# ------ extract timing summary ------
# llama-completion writes timing summary to stderr (TIMING_TXT)
# Lines like:  load time = X ms,  prefill = Y tokens, Z ms, etc.
echo "[bench_vanilla] DONE  $(date)  pid=$LLAMA_PID  exit=$EXIT_CODE"
echo "                 out: $OUT_DIR/"
echo "                 mem:    $MEM_CSV       ($(wc -l < $MEM_CSV) samples)"
echo "                 sensors:$SENSORS_CSV  ($(wc -l < $SENSORS_CSV) samples)"
echo "                 timing: $TIMING_TXT"
echo ""
echo "===== timing ====="
grep -E "(llama_perf|eval time|prompt eval|n_prompt|n_predict|tokens per second|prefill|decode|throughput)" "$TIMING_TXT" || tail -20 "$TIMING_TXT"
echo "===== final memory ====="
tail -3 "$MEM_CSV"
exit $EXIT_CODE
