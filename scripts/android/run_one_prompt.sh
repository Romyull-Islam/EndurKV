#!/system/bin/sh
# run_one_prompt.sh — single-prompt orchestrator for the OnePlus 15.
#
# What it does, in order:
#   1. Starts sample_sensors.sh in the background (thermal + UFS + meminfo, 10 Hz).
#   2. Records the wall-clock epoch when the probe starts (joins probe's
#      relative wall_clock_us back to the sampler's wall-clock seconds).
#   3. Runs entropy_probe on a single prompt.
#   4. Stops the sampler.
#   5. Writes a join metadata file with start/end timestamps and exit code.
#
# Designed so the host driver (PC) can adb-shell this for each prompt and
# pull the four output files per prompt:
#   <prompt_id>.entropy.csv    per-step entropy / top-k
#   <prompt_id>.sensors.csv    per-sample thermal / UFS / mem
#   <prompt_id>.probe.stderr   probe log (steps, total_ms, eos_step)
#   <prompt_id>.run.json       join metadata
#
# Usage on phone (already cd'd to /data/local/tmp/endurkv):
#   run_one_prompt.sh \
#       --probe ./bin/entropy_probe \
#       --model ./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
#       --prompt ./prompts/qasper_0.txt \
#       --prompt-id qasper_0 \
#       --max-tokens 64 \
#       --out-dir ./logs/study \
#       [--sensors-hz 10] [--sampler ./bin/sample_sensors.sh]

PROBE=""
MODEL=""
PROMPT=""
PROMPT_ID=""
MAX_TOKENS=64
OUT_DIR=""
SAMPLER="./sample_sensors.sh"
SENSORS_HZ=10
SEED=42

while [ $# -gt 0 ]; do
    case "$1" in
        --probe)        PROBE="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --prompt)       PROMPT="$2"; shift 2 ;;
        --prompt-id)    PROMPT_ID="$2"; shift 2 ;;
        --max-tokens)   MAX_TOKENS="$2"; shift 2 ;;
        --out-dir)      OUT_DIR="$2"; shift 2 ;;
        --sampler)      SAMPLER="$2"; shift 2 ;;
        --sensors-hz)   SENSORS_HZ="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

for var in PROBE MODEL PROMPT PROMPT_ID OUT_DIR; do
    eval val=\$$var
    if [ -z "$val" ]; then
        echo "missing required --$(echo $var | tr A-Z- a-z_-)" >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR"
ENT_CSV="$OUT_DIR/${PROMPT_ID}.entropy.csv"
ATTN_BIN="$OUT_DIR/${PROMPT_ID}.attn.bin"
SEN_CSV="$OUT_DIR/${PROMPT_ID}.sensors.csv"
STD_ERR="$OUT_DIR/${PROMPT_ID}.probe.stderr"
RUN_META="$OUT_DIR/${PROMPT_ID}.run.json"

# attention_probe takes an extra --output-attn flag and disables flash-attn
# under the hood to expose kq_soft_max tensors. We detect it by the binary
# basename so the host orchestrator can switch probes via --probe alone.
PROBE_BASENAME=$(basename "$PROBE")
EXTRA_PROBE_ARGS=""
if [ "$PROBE_BASENAME" = "attention_probe" ]; then
    EXTRA_PROBE_ARGS="--output-attn $ATTN_BIN"
fi

# Make sure LD_LIBRARY_PATH points at the bundled .so files (they sit next to
# the probe binary thanks to the $ORIGIN rpath we set in CMakeLists, but
# Android's loader still consults LD_LIBRARY_PATH first and we want to be
# robust against the rpath being stripped by some build flag).
PROBE_DIR=$(dirname "$PROBE")
export LD_LIBRARY_PATH="$PROBE_DIR:$PROBE_DIR/../lib:$LD_LIBRARY_PATH"

# ---- 1. start sampler ----
SAMP_LOG="$OUT_DIR/${PROMPT_ID}.sampler.stderr"
"$SAMPLER" --out "$SEN_CSV" --hz "$SENSORS_HZ" 2>"$SAMP_LOG" &
SAMP_PID=$!
# Brief settle so the sampler writes at least one row before the probe starts;
# helps the join script confirm sampler is alive.
sleep 0.2

# ---- 2. snapshot the start instant ----
START_WALL=$(date +%s.%N)
START_MONO=$(awk '{print $1; exit}' /proc/uptime)
THERMAL_AT_START=$(for z in /sys/class/thermal/thermal_zone*; do
    [ -r "$z/temp" ] || continue
    name=$(cat "$z/type" 2>/dev/null | tr ' ' '_')
    t=$(cat "$z/temp" 2>/dev/null)
    printf '"%s":%s,' "$name" "$t"
done | sed 's/,$//')

# ---- 3. run the probe ----
"$PROBE" \
    --model       "$MODEL" \
    --prompt-file "$PROMPT" \
    --prompt-id   "$PROMPT_ID" \
    --max-tokens  "$MAX_TOKENS" \
    --seed        "$SEED" \
    --output      "$ENT_CSV" \
    $EXTRA_PROBE_ARGS \
    >/dev/null 2>"$STD_ERR"
PROBE_RC=$?

END_WALL=$(date +%s.%N)
END_MONO=$(awk '{print $1; exit}' /proc/uptime)
THERMAL_AT_END=$(for z in /sys/class/thermal/thermal_zone*; do
    [ -r "$z/temp" ] || continue
    name=$(cat "$z/type" 2>/dev/null | tr ' ' '_')
    t=$(cat "$z/temp" 2>/dev/null)
    printf '"%s":%s,' "$name" "$t"
done | sed 's/,$//')

# ---- 4. stop the sampler cleanly ----
kill -TERM "$SAMP_PID" 2>/dev/null
# Give it ~200ms to flush.
i=0
while kill -0 "$SAMP_PID" 2>/dev/null && [ "$i" -lt 10 ]; do
    sleep 0.05
    i=$((i+1))
done
kill -KILL "$SAMP_PID" 2>/dev/null

# ---- 5. write join metadata ----
{
    printf '{\n'
    printf '  "prompt_id": "%s",\n' "$PROMPT_ID"
    printf '  "probe_path": "%s",\n' "$PROBE"
    printf '  "model_path": "%s",\n' "$MODEL"
    printf '  "prompt_path": "%s",\n' "$PROMPT"
    printf '  "max_tokens": %s,\n' "$MAX_TOKENS"
    printf '  "seed": %s,\n' "$SEED"
    printf '  "sensors_hz": %s,\n' "$SENSORS_HZ"
    printf '  "start_wall_s": %s,\n' "$START_WALL"
    printf '  "end_wall_s": %s,\n' "$END_WALL"
    printf '  "start_monotonic_s": %s,\n' "$START_MONO"
    printf '  "end_monotonic_s": %s,\n' "$END_MONO"
    printf '  "thermal_at_start_mc": {%s},\n' "$THERMAL_AT_START"
    printf '  "thermal_at_end_mc": {%s},\n' "$THERMAL_AT_END"
    printf '  "probe_exit_code": %s\n' "$PROBE_RC"
    printf '}\n'
} > "$RUN_META"

echo "[run_one_prompt] prompt_id=$PROMPT_ID rc=$PROBE_RC outputs in $OUT_DIR/" >&2
exit $PROBE_RC
