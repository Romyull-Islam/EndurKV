#!/system/bin/sh
# run_one_prompt.sh: run the probe on one prompt on the phone with sample_sensors.sh
# in the background, then write <prompt_id>.run.json with start/end timestamps,
# thermal snapshots and exit code so the host can join probe and sensor logs.
# Outputs per prompt: .entropy.csv, .sensors.csv, .probe.stderr, .run.json
#
# Usage on phone (from /data/local/tmp/endurkv):
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

# attention_probe needs --output-attn. Detect it by basename so --probe alone
# selects the probe.
PROBE_BASENAME=$(basename "$PROBE")
EXTRA_PROBE_ARGS=""
if [ "$PROBE_BASENAME" = "attention_probe" ]; then
    EXTRA_PROBE_ARGS="--output-attn $ATTN_BIN"
fi

# Point LD_LIBRARY_PATH at the bundled .so files in case the $ORIGIN rpath
# was stripped by the build.
PROBE_DIR=$(dirname "$PROBE")
export LD_LIBRARY_PATH="$PROBE_DIR:$PROBE_DIR/../lib:$LD_LIBRARY_PATH"

# 0. Optional pre-cooldown: if COOLDOWN_TEMP_C is set, wait (up to 5 min) until the
# hottest thermal zone is at or below it, so every run starts from a similar temperature.
if [ -n "$COOLDOWN_TEMP_C" ]; then
    threshold_mc=$((COOLDOWN_TEMP_C * 1000))
    wait_start=$(date +%s)
    max_wait=300   # cap at 5 minutes; never block indefinitely
    while :; do
        hottest=$(for z in /sys/class/thermal/thermal_zone*; do
            [ -r "$z/temp" ] || continue
            name=$(cat "$z/type" 2>/dev/null)
            case "$name" in *trip*|*pmh*|*pmr*|*vbat*|*bcl*) continue;; esac
            t=$(cat "$z/temp" 2>/dev/null)
            [ -n "$t" ] && [ "$t" -gt 30000 ] && [ "$t" -lt 100000 ] && echo "$t"
        done | sort -n -r | head -1)
        [ -z "$hottest" ] && break
        if [ "$hottest" -le "$threshold_mc" ]; then
            echo "[cooldown] hottest=$((hottest/1000))C  ok (<=${COOLDOWN_TEMP_C}C)" >&2
            break
        fi
        elapsed=$(( $(date +%s) - wait_start ))
        if [ "$elapsed" -ge "$max_wait" ]; then
            echo "[cooldown] timed out at $((hottest/1000))C after ${max_wait}s — proceeding" >&2
            break
        fi
        sleep 5
    done
fi

# 1. start sampler
SAMP_LOG="$OUT_DIR/${PROMPT_ID}.sampler.stderr"
"$SAMPLER" --out "$SEN_CSV" --hz "$SENSORS_HZ" 2>"$SAMP_LOG" &
SAMP_PID=$!
# Let the sampler write at least one row before the probe starts.
sleep 0.2

# 2. snapshot the start instant
START_WALL=$(date +%s.%N)
START_MONO=$(awk '{print $1; exit}' /proc/uptime)
THERMAL_AT_START=$(for z in /sys/class/thermal/thermal_zone*; do
    [ -r "$z/temp" ] || continue
    name=$(cat "$z/type" 2>/dev/null | tr ' ' '_')
    t=$(cat "$z/temp" 2>/dev/null)
    printf '"%s":%s,' "$name" "$t"
done | sed 's/,$//')

# 2b. System-wide UFS bytes written, from dumpsys storaged (summed over UIDs).
# It costs about 50-100 ms, so it is read only at the start and end of the prompt.
storaged_total_bytes_written() {
    dumpsys storaged 2>/dev/null | awk '
        /^[0-9]+ / { fg_w += $7; bg_w += $9 }
        END { print fg_w + bg_w }
    '
}
STORAGED_W_AT_START=$(storaged_total_bytes_written)

# 3. run the probe
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
STORAGED_W_AT_END=$(storaged_total_bytes_written)

# 4. stop the sampler cleanly
kill -TERM "$SAMP_PID" 2>/dev/null
# Give it ~200ms to flush.
i=0
while kill -0 "$SAMP_PID" 2>/dev/null && [ "$i" -lt 10 ]; do
    sleep 0.05
    i=$((i+1))
done
kill -KILL "$SAMP_PID" 2>/dev/null

# 5. write join metadata
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
    printf '  "storaged_bytes_written_at_start": %s,\n' "${STORAGED_W_AT_START:-0}"
    printf '  "storaged_bytes_written_at_end": %s,\n' "${STORAGED_W_AT_END:-0}"
    printf '  "probe_exit_code": %s\n' "$PROBE_RC"
    printf '}\n'
} > "$RUN_META"

echo "[run_one_prompt] prompt_id=$PROMPT_ID rc=$PROBE_RC outputs in $OUT_DIR/" >&2
exit $PROBE_RC
