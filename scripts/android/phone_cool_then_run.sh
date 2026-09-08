#!/system/bin/sh
# phone_cool_then_run.sh — Cool-and-measure wrapper for fair benchmarking.
#
# Standard practice: wait until phone shell_front skin temp ≤ THRESH_C
# (default 38°C) OR a max timeout (default 5 min), THEN run the command.
# Records starting temps in <out-dir>/cooldown.json.
#
# Usage:
#   sh phone_cool_then_run.sh --out-dir DIR --thresh-c 38 --max-wait 300 -- \
#       <command to run>
#
# Behaviour:
#   - Polls shell_front_temp every 5 s
#   - Returns when temp ≤ thresh OR max-wait elapses
#   - Writes cooldown.json: {start_skin_c, start_battery_c, cooldown_s, timed_out, peak_cpu_c}

set -e

OUT_DIR=""
THRESH_C=38
MAX_WAIT=300
POLL_S=5

while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir)   OUT_DIR="$2"; shift 2 ;;
        --thresh-c)  THRESH_C="$2"; shift 2 ;;
        --max-wait)  MAX_WAIT="$2"; shift 2 ;;
        --poll-s)    POLL_S="$2"; shift 2 ;;
        --)          shift; break ;;
        *) break ;;
    esac
done

[ -z "$OUT_DIR" ] && { echo "--out-dir required" >&2; exit 1; }
mkdir -p "$OUT_DIR"

# Helper: read shell_front temp in °C (returns 0 if not found)
read_skin_c() {
    for z in /sys/class/thermal/thermal_zone*; do
        if [ "$(cat "$z/type" 2>/dev/null)" = "shell_front" ]; then
            t=$(cat "$z/temp" 2>/dev/null)
            echo $((t / 1000))
            return
        fi
    done
    echo 0
}

read_batt_c() {
    for z in /sys/class/thermal/thermal_zone*; do
        if [ "$(cat "$z/type" 2>/dev/null)" = "battery" ]; then
            t=$(cat "$z/temp" 2>/dev/null)
            echo $((t / 1000))
            return
        fi
    done
    echo 0
}

read_max_cpu_c() {
    max=0
    for z in /sys/class/thermal/thermal_zone*; do
        name=$(cat "$z/type" 2>/dev/null)
        case "$name" in
            cpu-*|cpullc-*)
                t=$(cat "$z/temp" 2>/dev/null)
                tc=$((t / 1000))
                if [ "$tc" -gt "$max" ]; then max="$tc"; fi
                ;;
        esac
    done
    echo "$max"
}

T0=$(date +%s)
TIMED_OUT=0
while :; do
    SKIN=$(read_skin_c)
    NOW=$(date +%s)
    ELAPSED=$((NOW - T0))
    if [ "$SKIN" -le "$THRESH_C" ]; then
        echo "  [cool] skin=${SKIN}C ≤ ${THRESH_C}C (elapsed ${ELAPSED}s) — proceeding"
        break
    fi
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "  [cool] skin=${SKIN}C still > ${THRESH_C}C after ${MAX_WAIT}s — timing out"
        TIMED_OUT=1
        break
    fi
    echo "  [cool] skin=${SKIN}C > ${THRESH_C}C, waiting ${POLL_S}s (elapsed ${ELAPSED}s)..."
    sleep "$POLL_S"
done

START_SKIN=$(read_skin_c)
START_BATT=$(read_batt_c)
START_CPU=$(read_max_cpu_c)

cat > "$OUT_DIR/cooldown.json" <<EOF
{
  "start_skin_c": $START_SKIN,
  "start_battery_c": $START_BATT,
  "start_max_cpu_c": $START_CPU,
  "cooldown_s": $ELAPSED,
  "thresh_c": $THRESH_C,
  "timed_out": $TIMED_OUT
}
EOF

# Now execute the actual command (the user-supplied "$@" after --)
exec "$@"
