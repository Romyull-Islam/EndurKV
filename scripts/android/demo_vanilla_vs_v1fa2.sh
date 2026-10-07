#!/bin/bash
# Side-by-side terminal demo on the connected phone: vanilla vs v1_fa2_stack.
# Runs eviction_bench once per policy and prints generated text, thermal, energy and perf.
#
# Usage: bash demo_vanilla_vs_v1fa2.sh [OPTIONS] [PROMPT] [MODEL_TAG]
#   --long-decode       MAX_TOKENS=2048 with the long-story prompt
#   --k=N               nominal KV size for all policies (default 512, env K_NOMINAL)
#   --anchor=N          ANCHOR_TOP_K for v1_fa2_stack (default 32), ANCHOR + N_SINK < K
#   --repeat-penalty=F  repeat penalty for all policies, >= 1.0 (binary default 1.10)
#   --ignore-eos        always emit MAX_TOKENS (env IGNORE_EOS=1)
#   --policy=NAME       vanilla, v1_fa2_stack, adakv, both (default) or three-way (env POLICY)
#   --three-way         same as --policy=three-way, adakv runs eviction_bench_v8
#   --energy-mode       preset K=512, anchor 128, repeat 1.15 (env ENERGY_MODE=1), flags override it

set -u

# Args
LONG_DECODE=0
# Set when K_NOMINAL came from the env, so --energy-mode does not override it.
K_NOMINAL_ENV_SET=0
if [ -n "${K_NOMINAL+x}" ]; then
    K_NOMINAL_ENV_SET=1
fi
# Env vars seed the defaults, CLI flags override them.
K_NOMINAL="${K_NOMINAL:-512}"
K_CLI=""
IGNORE_EOS="${IGNORE_EOS:-0}"
# Energy mode does not add --ignore-eos itself but honors an explicit request.
IGNORE_EOS_EXPLICIT=0
if [ "${IGNORE_EOS:-0}" = "1" ]; then
    IGNORE_EOS_EXPLICIT=1
fi
ANCHOR_CLI=""
REPEAT_PENALTY_CLI=""
ENERGY_MODE="${ENERGY_MODE:-0}"
# vanilla, v1_fa2_stack, adakv, both or three-way
POLICY="${POLICY:-both}"

while [ $# -gt 0 ]; do
    case "$1" in
        --long-decode)
            LONG_DECODE=1
            shift
            ;;
        --ignore-eos)
            IGNORE_EOS=1
            IGNORE_EOS_EXPLICIT=1
            shift
            ;;
        --energy-mode)
            ENERGY_MODE=1
            shift
            ;;
        --three-way|--3-way|--threeway)
            POLICY="three-way"
            shift
            ;;
        --k=*)
            K_CLI="${1#--k=}"
            shift
            ;;
        --k)
            if [ $# -lt 2 ]; then
                echo "Error: --k requires a value" >&2
                exit 1
            fi
            K_CLI="$2"
            shift 2
            ;;
        --anchor=*)
            ANCHOR_CLI="${1#--anchor=}"
            shift
            ;;
        --anchor)
            if [ $# -lt 2 ]; then
                echo "Error: --anchor requires a value" >&2
                exit 1
            fi
            ANCHOR_CLI="$2"
            shift 2
            ;;
        --repeat-penalty=*)
            REPEAT_PENALTY_CLI="${1#--repeat-penalty=}"
            shift
            ;;
        --repeat-penalty)
            if [ $# -lt 2 ]; then
                echo "Error: --repeat-penalty requires a value" >&2
                exit 1
            fi
            REPEAT_PENALTY_CLI="$2"
            shift 2
            ;;
        --policy=*)
            POLICY="${1#--policy=}"
            shift
            ;;
        --policy)
            if [ $# -lt 2 ]; then
                echo "Error: --policy requires a value (vanilla | v1_fa2_stack | both)" >&2
                exit 1
            fi
            POLICY="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        --*)
            echo "Error: unknown flag '$1'" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# --energy-mode preset. Each value is filled in only when the user did not set
# it, and the model is still allowed to stop on EOS.
if [ "$ENERGY_MODE" = "1" ]; then
    if [ "$K_NOMINAL_ENV_SET" -eq 0 ] && [ -z "$K_CLI" ]; then
        K_NOMINAL=512
    fi
    if [ -z "$ANCHOR_CLI" ]; then
        ANCHOR_CLI=128
    fi
    if [ -z "$REPEAT_PENALTY_CLI" ]; then
        REPEAT_PENALTY_CLI=1.15
    fi
fi

if [ -n "$K_CLI" ]; then
    K_NOMINAL="$K_CLI"
fi

# Validate K_NOMINAL is a positive integer
if ! [[ "$K_NOMINAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: K_NOMINAL must be a positive integer (got '$K_NOMINAL')" >&2
    exit 1
fi

# Validate POLICY value
case "$POLICY" in
    vanilla|v1_fa2_stack|adakv|both|three-way) ;;
    *)
        echo "Error: --policy must be one of: vanilla | v1_fa2_stack | adakv | both | three-way (got '$POLICY')" >&2
        exit 1
        ;;
esac

# Convenience flags for downstream logic
RUN_VANILLA=0
RUN_V1FA2=0
RUN_ADAKV=0
case "$POLICY" in
    vanilla)       RUN_VANILLA=1 ;;
    v1_fa2_stack)  RUN_V1FA2=1 ;;
    adakv)         RUN_ADAKV=1 ;;
    both)          RUN_VANILLA=1; RUN_V1FA2=1 ;;
    three-way)    RUN_VANILLA=1; RUN_ADAKV=1; RUN_V1FA2=1 ;;
esac

DEFAULT_PROMPT_SHORT_REGIME="The early autumn morning was crisp and clear as the small fishing village awoke to the sound of seagulls circling the harbor. Wooden boats creaked against the dock while fishermen prepared their nets, exchanging quiet greetings over steaming cups of coffee in the chill air before sailing out."
DEFAULT_PROMPT_LONG_REGIME="Write a long story about a wizard who discovers an ancient artifact deep in a forgotten library."

if [ "$LONG_DECODE" -eq 1 ]; then
    DEFAULT_PROMPT="$DEFAULT_PROMPT_LONG_REGIME"
else
    DEFAULT_PROMPT="$DEFAULT_PROMPT_SHORT_REGIME"
fi

PROMPT="${1:-$DEFAULT_PROMPT}"
MODEL_TAG="${2:-Llama-3.2-1B}"

# K_NOMINAL-derived knobs (shared by vanilla & v1_fa2_stack for fair comparison)
ANCHOR_TOP_K=32
N_SINK=4

# CLI --anchor overrides ANCHOR_TOP_K (affects only v1_fa2_stack).
if [ -n "$ANCHOR_CLI" ]; then
    if ! [[ "$ANCHOR_CLI" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: --anchor must be a positive integer (got '$ANCHOR_CLI')" >&2
        exit 1
    fi
    ANCHOR_TOP_K="$ANCHOR_CLI"
fi

# Sanity check: ANCHOR + N_SINK must leave room for recent_budget >= 1.
if [ $(( ANCHOR_TOP_K + N_SINK )) -ge "$K_NOMINAL" ]; then
    echo "Error: ANCHOR_TOP_K=$ANCHOR_TOP_K + N_SINK=$N_SINK must be < K_NOMINAL=$K_NOMINAL (recent_budget would be invalid)" >&2
    exit 1
fi

RECENT_BUDGET=$(( K_NOMINAL - ANCHOR_TOP_K - N_SINK ))
if [ "$RECENT_BUDGET" -lt 1 ]; then
    echo "Error: K_NOMINAL=$K_NOMINAL too small (must be > anchor_top_k($ANCHOR_TOP_K) + n_sink($N_SINK) = $((ANCHOR_TOP_K + N_SINK)))" >&2
    exit 1
fi

# Validate --repeat-penalty (default in binary is 1.10).
REPEAT_PENALTY=""
if [ -n "$REPEAT_PENALTY_CLI" ]; then
    # Must be a float (or integer) >= 1.0
    if ! [[ "$REPEAT_PENALTY_CLI" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: --repeat-penalty must be a float (got '$REPEAT_PENALTY_CLI')" >&2
        exit 1
    fi
    if ! awk -v x="$REPEAT_PENALTY_CLI" 'BEGIN{ exit !(x+0 >= 1.0) }'; then
        echo "Error: --repeat-penalty must be >= 1.0 (got '$REPEAT_PENALTY_CLI')" >&2
        exit 1
    fi
    REPEAT_PENALTY="$REPEAT_PENALTY_CLI"
fi

# K-regime annotation for banner
if [ "$K_NOMINAL" -lt 256 ]; then
    K_REGIME_NOTE=" [aggressive eviction]"
elif [ "$K_NOMINAL" -ge 1024 ]; then
    K_REGIME_NOTE=" [loose eviction]"
else
    K_REGIME_NOTE=""
fi

case "$MODEL_TAG" in
    Llama-3.2-1B|llama|llama-3.2-1b)
        MODEL_FILE="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        MODEL_PRETTY="Llama-3.2-1B Q4_K_M"
        ;;
    Phi-3-mini-128k|phi|phi-3)
        MODEL_FILE="Phi-3-mini-128k-instruct-Q4_K_M.gguf"
        MODEL_PRETTY="Phi-3-mini-128k Q4_K_M"
        ;;
    gemma-2-2b-it|gemma|gemma-2-2b)
        MODEL_FILE="gemma-2-2b-it-Q4_K_M.gguf"
        MODEL_PRETTY="gemma-2-2b-it Q4_K_M"
        ;;
    *)
        echo "Unknown model tag: $MODEL_TAG (expected: Llama-3.2-1B | Phi-3-mini-128k | gemma-2-2b-it)" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/adb_resilient.sh"

# Settings
PHONE_WORK="${PHONE_WORK:-/data/local/tmp/endurkv}"
PHONE_BIN="$PHONE_WORK/bin_cpu/eviction_bench"
# AdaKV baseline scoring lives in the v8 binary.
PHONE_BIN_ADAKV="$PHONE_WORK/bin_cpu/eviction_bench_v8"
PHONE_MODEL="$PHONE_WORK/models/$MODEL_FILE"
PHONE_SAMPLER="$PHONE_WORK/scripts/sample_sensors.sh"
# DDR thermal zone for the watchdog (zone47 on the OnePlus 15). Override with env DDR_ZONE.
DDR_ZONE="${DDR_ZONE:-/sys/class/thermal/thermal_zone47}"
RUN_ID="demo_$(date +%Y%m%d_%H%M%S)_$$"
PHONE_OUT_ROOT="$PHONE_WORK/logs/$RUN_ID"
HOST_OUT_ROOT="/tmp/$RUN_ID"
mkdir -p "$HOST_OUT_ROOT"

if [ "$LONG_DECODE" -eq 1 ]; then
    MAX_TOKENS=2048
else
    MAX_TOKENS=256
fi
CTX_SIZE=4096
THREADS=4
N_BATCH=512
UBATCH=64
SEED=42
COOL_TARGET_C=45        # DDR target before each run
COOL_TIMEOUT_S=60
SAMPLE_HZ=5             # sensor sample rate (matches wave-9 default)

# Long-decode banner
if [ "$IGNORE_EOS" -eq 1 ]; then
    IGNORE_EOS_NOTE=" [ignore-eos]"
else
    IGNORE_EOS_NOTE=""
fi

if [ "$LONG_DECODE" -eq 1 ]; then
    echo "" >&2
    echo "[demo] === LONG-DECODE MODE ===${IGNORE_EOS_NOTE}" >&2
    echo "[demo]   MAX_TOKENS=$MAX_TOKENS (vs 256 in short-decode mode)" >&2
    echo "[demo]   Decode-heavy regime — exercises eviction policy / KV growth." >&2
    echo "[demo]   Expect v1_fa2_stack to show wall-time savings, non-zero eviction" >&2
    echo "[demo]   count, and a substantially smaller peak KV cache vs vanilla." >&2
    if [ "$IGNORE_EOS" -eq 1 ]; then
        echo "[demo]   --ignore-eos: model will emit full MAX_TOKENS regardless of EOS." >&2
    fi
    echo "" >&2
elif [ "$IGNORE_EOS" -eq 1 ]; then
    echo "[demo] [ignore-eos] enabled — full MAX_TOKENS regardless of EOS." >&2
fi

if [ "$ENERGY_MODE" = "1" ]; then
    echo "[demo] ENERGY MODE: small cache (K=$K_NOMINAL) + strong anchoring ($ANCHOR_TOP_K) + cliff-only watchdog → optimized for energy per token" >&2
fi

REPEAT_PENALTY_DISPLAY="${REPEAT_PENALTY:-1.10 (default)}"
echo "[demo] K_NOMINAL=$K_NOMINAL  (anchor_top_k=$ANCHOR_TOP_K, n_sink=$N_SINK, recent_budget=$RECENT_BUDGET, repeat_penalty=$REPEAT_PENALTY_DISPLAY)${K_REGIME_NOTE}${IGNORE_EOS_NOTE}" >&2

if [ "$POLICY" = "three-way" ]; then
    echo "[demo] THREE-WAY MODE: running vanilla + adakv + v1_fa2_stack (eviction_bench_v8 for adakv)" >&2
elif [ "$POLICY" != "both" ]; then
    echo "[demo] SINGLE-POLICY MODE: running only '$POLICY' (comparison table disabled)" >&2
fi

# Pre-flight: check that the phone has every required file
echo "[demo] checking phone connectivity..." >&2
adb_wait
PREFLIGHT_FILES=("$PHONE_BIN" "$PHONE_MODEL" "$PHONE_SAMPLER")
if [ "$RUN_ADAKV" -eq 1 ]; then
    PREFLIGHT_FILES+=("$PHONE_BIN_ADAKV")
fi
for f in "${PREFLIGHT_FILES[@]}"; do
    if ! adb_safe_shell "[ -e '$f' ] && echo OK" | grep -q OK; then
        echo "Missing on phone: $f" >&2
        exit 2
    fi
done
adb_safe_shell "mkdir -p '$PHONE_OUT_ROOT'" >/dev/null

# Push the prompt as a file to avoid shell quoting of arbitrary text.
HOST_PROMPT_FILE="$HOST_OUT_ROOT/prompt.txt"
printf '%s\n' "$PROMPT" > "$HOST_PROMPT_FILE"
adb_safe_push "$HOST_PROMPT_FILE" "$PHONE_OUT_ROOT/prompt.txt" >/dev/null

# Helpers
# DDR temp in degC (zone "ddr" reports milliC)
ddr_temp_c() {
    adb_safe_shell "for z in /sys/class/thermal/thermal_zone*; do
        t=\$(cat \$z/type 2>/dev/null)
        if [ \"\$t\" = ddr ]; then awk '{printf \"%d\\n\", \$1/1000}' \$z/temp; break; fi
    done" 2>/dev/null | tr -d ' \r\n'
}

cool_phone() {
    local label="$1"
    local t0=$(date +%s)
    while true; do
        local d
        d=$(ddr_temp_c)
        [ -z "$d" ] && d=0
        if [ "$d" -le "$COOL_TARGET_C" ] 2>/dev/null; then
            echo "[demo] cooled before $label  DDR=${d}C" >&2
            return 0
        fi
        local elapsed=$(( $(date +%s) - t0 ))
        if [ "$elapsed" -ge "$COOL_TIMEOUT_S" ]; then
            echo "[demo] cool timeout before $label at DDR=${d}C (proceeding)" >&2
            return 0
        fi
        sleep 3
    done
}

# jget FIELD PATH: pull one field out of a meta.json.
jget() {
    local field="$1" path="$2"
    grep -oE "\"$field\"[^,}]*" "$path" 2>/dev/null | sed -E "s/^\"$field\"[[:space:]]*:[[:space:]]*//; s/^\"//; s/\"$//" | head -1
}

# Signed % delta: (new - base) / base * 100
pct_delta() {
    awk -v a="$1" -v b="$2" 'BEGIN{
        if (a+0==0) { print "n/a"; exit }
        d=(b-a)/a*100.0;
        printf "%+.1f%%", d
    }'
}
abs_delta() {
    awk -v a="$1" -v b="$2" 'BEGIN{ printf "%+.1f", (b-a) }'
}

# Watchdog helpers. start_watchdog launches preempt_throttle_watchdog_v2.sh as
# root, stop_watchdog stops it through a sentinel file and pulls watchdog.log.
start_watchdog() {
    local cell_name="$1"
    local host_cell="$HOST_OUT_ROOT/$cell_name"
    local phone_cell="$PHONE_OUT_ROOT/$cell_name"
    local log="$phone_cell/watchdog.log"
    local stop="$phone_cell/watchdog.stop"
    local pidf="$phone_cell/watchdog.pid"
    adb_safe_shell "rm -f '$log' '$stop' '$pidf'" >/dev/null
    # The watchdog runs detached under su, so its PID is not reliable from
    # here. It is stopped through the sentinel file, and a host marker records it.
    adb_safe_shell "su -c 'sh $PHONE_WORK/scripts/preempt_throttle_watchdog_v2.sh $log $stop $DDR_ZONE' </dev/null >/dev/null 2>&1 &" >/dev/null
    echo started > "$host_cell/watchdog.marker"
}

stop_watchdog() {
    local cell_name="$1"
    local host_cell="$HOST_OUT_ROOT/$cell_name"
    local phone_cell="$PHONE_OUT_ROOT/$cell_name"
    if [ -f "$host_cell/watchdog.marker" ]; then
        adb_safe_shell "touch '$phone_cell/watchdog.stop'" >/dev/null
        # Wait up to 10 s for a clean exit, then kill any leftover process.
        local waited=0
        while [ "$waited" -lt 10 ]; do
            if ! adb_safe_shell "su -c 'pgrep -f preempt_throttle_watchdog_v2.sh' 2>/dev/null" | grep -q '[0-9]'; then
                break
            fi
            sleep 1
            waited=$(( waited + 1 ))
        done
        adb_safe_shell "su -c 'pkill -f preempt_throttle_watchdog_v2.sh' 2>/dev/null" >/dev/null
        adb_safe_pull "$phone_cell/watchdog.log" "$host_cell/" >/dev/null 2>&1 || true
        rm -f "$host_cell/watchdog.marker"
    fi
}

# Prints "tier1=N tier2=N tier3=N total=N" for a host watchdog.log.
# v2 transition lines look like "[ts] tier OLD -> NEW (...) engaged: ..." and
# are counted by destination tier. v1 "TIER1" or "tier=1" tokens are a fallback.
parse_watchdog_log() {
    local log_path="$1"
    if [ ! -f "$log_path" ]; then
        # Watchdog never started or the pull failed.
        echo "tier1=0 tier2=0 tier3=0 total=0 missing=1"
        return
    fi
    if [ ! -s "$log_path" ]; then
        echo "tier1=0 tier2=0 tier3=0 total=0 empty=1"
        return
    fi
    awk '
        # Skip v2 metadata lines. The exit line contains "last tier=N", which
        # the v1 /tier=N/ fallback would otherwise count.
        /watchdog_v2 start|watchdog_v2 exit|\(initial\)|^\[[0-9]+\] zones:|^\[[0-9]+\] DDR  |^\[[0-9]+\] CPU  |^\[[0-9]+\] SOC  |^\[[0-9]+\] SKIN |^\[[0-9]+\] BAT  |^\[[0-9]+\] BAT_I |^\[[0-9]+\] tier thresholds|^\[[0-9]+\] LOG=/ { next }

        # match() and substr() keep this portable to non-gawk awk.
        {
            if (match($0, /tier[ ]+[0-9]+[ ]*->[ ]*[0-9]+/)) {
                seg = substr($0, RSTART, RLENGTH)
                # seg is "tier 0 -> 1", the last field is the destination
                n = split(seg, parts, /[ \t]+/)
                new_tier = parts[n] + 0
                if      (new_tier == 1) { t1++; next }
                else if (new_tier == 2) { t2++; next }
                else if (new_tier == 3) { t3++; next }
                next
            }
        }
        # v1 log format
        /engaged/ {
            if      (/TIER1|tier=1/) { t1++; next }
            else if (/TIER2|tier=2/) { t2++; next }
            else if (/TIER3|tier=3/) { t3++; next }
        }
        END {
            t1 = t1+0; t2 = t2+0; t3 = t3+0;
            printf "tier1=%d tier2=%d tier3=%d total=%d", t1, t2, t3, (t1+t2+t3)
        }
    ' "$log_path"
}

# Battery charge counter in uAh from sysfs
bat_charge_uah() {
    adb_safe_shell "cat /sys/class/power_supply/battery/charge_counter 2>/dev/null" \
        | tr -d ' \r\n'
}
# Fallback: voltage (mV) and current (mA) from dumpsys
bat_volt_mv() {
    adb_safe_shell "dumpsys battery 2>/dev/null | sed -n 's/.*Charger voltage *: *//p' | head -1" \
        | tr -d ' \r\n'
}
bat_curr_ma() {
    adb_safe_shell "dumpsys battery 2>/dev/null | sed -n 's/.*Battery current *: *//p' | head -1" \
        | tr -d ' \r\n'
}

# run_one_policy CELL POLICY CACHE_K CACHE_V [/BIN] [EXTRA_FLAGS...]
run_one_policy() {
    local cell_name="$1"
    local policy="$2"
    local cache_k="$3"
    local cache_v="$4"
    # An optional 5th arg starting with / overrides the binary path.
    local bin_path="$PHONE_BIN"
    if [ $# -ge 5 ] && [ "${5#/}" != "$5" ]; then
        bin_path="$5"
        shift 5
    else
        shift 4
    fi
    local extra_flags=("$@")

    local phone_cell="$PHONE_OUT_ROOT/$cell_name"
    local host_cell="$HOST_OUT_ROOT/$cell_name"
    mkdir -p "$host_cell"
    adb_safe_shell "mkdir -p '$phone_cell'" >/dev/null

    echo "" >&2
    echo "[demo] === $cell_name ===" >&2
    cool_phone "$cell_name"

    local bat_q0; bat_q0=$(bat_charge_uah)
    local bat_v0; bat_v0=$(bat_volt_mv)
    local bat_i0; bat_i0=$(bat_curr_ma)
    local t_wall_0; t_wall_0=$(date +%s.%N)

    # nohup so an adb disconnect does not kill the sampler
    local pid_file="$phone_cell/sampler.pid"
    adb_safe_shell "nohup sh '$PHONE_SAMPLER' --out '$phone_cell/sensors.csv' --hz $SAMPLE_HZ \
        </dev/null >'$phone_cell/sampler.log' 2>&1 & echo \$! > '$pid_file'" >/dev/null
    sleep 1   # let sampler warm up

    # Watchdog for v1_fa2_stack only. The vanilla baseline runs without thermal control.
    if [ "$policy" = "v1_fa2" ]; then
        start_watchdog "$cell_name"
        echo "[demo] watchdog v2 started (multi-sensor: DDR/CPU/skin/battery)" >&2
    fi

    # Run inline so the timing line and generated text reach the host.
    local args=(
        "$bin_path"
        --model "$PHONE_MODEL"
        --prompt "$PHONE_OUT_ROOT/prompt.txt"
        --prompt-id "demo_$cell_name"
        --policy "$policy"
        --cache-type-k "$cache_k"
        --cache-type-v "$cache_v"
        --max-tokens $MAX_TOKENS
        --ctx-size $CTX_SIZE
        --threads $THREADS
        --n-gpu-layers 0
        --n-batch $N_BATCH
        --ubatch-size $UBATCH
        --greedy
        --seed $SEED
        --eval-mode gen
        --out-csv  "$phone_cell/steps.csv"
        --out-meta "$phone_cell/meta.json"
        --out-gen  "$phone_cell/gen.txt"
    )
    args+=("${extra_flags[@]}")

    local cmd
    cmd="LD_LIBRARY_PATH='$PHONE_WORK/bin_cpu'"
    local a
    for a in "${args[@]}"; do
        cmd+=" $(printf '%q' "$a")"
    done
    # Merge stderr into stdout, tee on the host shows and saves the stream.
    cmd+=" 2>&1; echo \$? > '$phone_cell/exit.code'"
    # DEBUG=1 prints the args and the final phone command to check flag propagation.
    if [ "${DEBUG:-0}" = "1" ]; then
        {
            echo "[debug] === cell=$cell_name policy=$policy ==="
            echo "[debug] args count=${#args[@]}"
            local _i=0
            for a in "${args[@]}"; do
                echo "[debug]   args[$_i]=$(printf '%q' "$a")"
                _i=$(( _i + 1 ))
            done
            echo "[debug] IGNORE_EOS=$IGNORE_EOS  COMMON_EXTRA=(${COMMON_EXTRA[*]+${COMMON_EXTRA[*]}})"
            echo "[debug] final phone cmd: $cmd"
        } >&2
        printf '%s\n' "$cmd" > "$host_cell/cmd.debug"
    fi
    echo "[demo] ── streaming inference output ($policy) ──"
    local host_log="$host_cell/inference.log"
    mkdir -p "$host_cell"
    adb_safe_shell "$cmd" 2>&1 | tee "$host_log"
    echo "[demo] ── end of $policy stream ──"
    # Later parsing reads stderr.log
    cp -f "$host_log" "$host_cell/stderr.log" 2>/dev/null || true
    local t_wall_1; t_wall_1=$(date +%s.%N)

    adb_safe_shell "pid=\$(cat '$pid_file' 2>/dev/null); [ -n \"\$pid\" ] && kill -TERM \$pid 2>/dev/null; sleep 1; [ -n \"\$pid\" ] && kill -KILL \$pid 2>/dev/null; true" >/dev/null

    if [ "$policy" = "v1_fa2" ]; then
        stop_watchdog "$cell_name"
        echo "[demo] watchdog stopped — log pulled to host" >&2
    fi

    local bat_q1; bat_q1=$(bat_charge_uah)
    local bat_v1; bat_v1=$(bat_volt_mv)
    local bat_i1; bat_i1=$(bat_curr_ma)

    adb_safe_pull "$phone_cell/meta.json"   "$host_cell/" >/dev/null 2>&1 || true
    adb_safe_pull "$phone_cell/gen.txt"     "$host_cell/" >/dev/null 2>&1 || true
    adb_safe_pull "$phone_cell/sensors.csv" "$host_cell/" >/dev/null 2>&1 || true
    adb_safe_pull "$phone_cell/stderr.log"  "$host_cell/" >/dev/null 2>&1 || true
    adb_safe_pull "$phone_cell/stdout.log"  "$host_cell/" >/dev/null 2>&1 || true

    local meta="$host_cell/meta.json"
    local prefill_ms decode_ms decode_tps total_ms peak_kv evicted_dec peak_rss_kb
    prefill_ms=$(jget prefill_ms "$meta");      : "${prefill_ms:=0}"
    decode_ms=$(jget decode_ms "$meta");        : "${decode_ms:=0}"
    decode_tps=$(jget decode_tps "$meta");      : "${decode_tps:=0}"
    total_ms=$(jget total_ms "$meta");          : "${total_ms:=0}"
    peak_kv=$(jget peak_kv_cells "$meta");      : "${peak_kv:=0}"
    evicted_dec=$(jget evicted_total_decode "$meta"); : "${evicted_dec:=0}"
    peak_rss_kb=$(jget peak_rss_kb "$meta");    : "${peak_rss_kb:=0}"

    local wall_s
    wall_s=$(awk -v a="$t_wall_0" -v b="$t_wall_1" 'BEGIN{printf "%.3f", b-a}')

    # Peak DDR, CPU and skin temperatures from sensors.csv
    local peak_ddr peak_cpu peak_skin
    peak_ddr=$(awk -F, '
        NR==1 {
            for (i=1;i<=NF;i++) if ($i=="ddr_temp_mc") col=i;
            next
        }
        col && $col != "" { v=$col+0; if (v>p) p=v }
        END { if (p>0) printf "%.1f", p/1000.0; else print "n/a" }
    ' "$host_cell/sensors.csv" 2>/dev/null)
    peak_cpu=$(awk -F, '
        NR==1 {
            # Only the big-core sensor cpu-1-0-0, the zone the watchdog protects.
            # A max over all 20+ CPU zones would mix in idle clusters.
            for (i=1;i<=NF;i++) {
                if ($i == "cpu-1-0-0_temp_mc") cols[++n]=i;
            }
            next
        }
        {
            for (k=1;k<=n;k++) {
                c=cols[k]; if ($c=="") continue; v=$c+0; if (v>p) p=v
            }
        }
        END { if (p>0) printf "%.1f", p/1000.0; else print "n/a" }
    ' "$host_cell/sensors.csv" 2>/dev/null)
    peak_skin=$(awk -F, '
        NR==1 {
            for (i=1;i<=NF;i++) if ($i ~ /^shell_(front|frame|back)_temp_mc$/ || $i ~ /^skin.*_temp_mc$/) cols[++n]=i;
            next
        }
        {
            for (k=1;k<=n;k++) {
                c=cols[k]; if ($c=="") continue; v=$c+0; if (v>p) p=v
            }
        }
        END { if (p>0) printf "%.1f", p/1000.0; else print "n/a" }
    ' "$host_cell/sensors.csv" 2>/dev/null)

    # Energy in mAh from the first method that works: (1) power integrated over
    # sensors.csv (power_now, else current_now x voltage_now, else dumpsys mA x mV,
    # with mAh = mWh / 4.0 at a nominal 4 V), (2) charge_counter delta, (3) mean |I| x wall_s.
    # A "Charging" status during the run is flagged, since the number is then unreliable.
    local energy_mah="n/a"
    local energy_method="n/a"
    local energy_fallback_chain=""
    local charging_detected=0
    local sensors_csv="$host_cell/sensors.csv"

    if [ -s "$sensors_csv" ]; then
        local _charging_hits
        _charging_hits=$(awk -F, '
            NR==1 { for (i=1;i<=NF;i++) if ($i=="bat_status") c=i; next }
            c && $c=="Charging" { n++ }
            END { print n+0 }
        ' "$sensors_csv" 2>/dev/null)
        if [ -n "$_charging_hits" ] && [ "$_charging_hits" -gt 0 ] 2>/dev/null; then
            charging_detected=1
        fi
    fi

    # Method 1. A source must be populated in more than half the rows, since a
    # column can exist but stay empty when the PMIC read fails.
    local m1_result=""
    local m1_method=""
    if [ -s "$sensors_csv" ]; then
        local m1_pair
        m1_pair=$(awk -F, -v hz="$SAMPLE_HZ" '
            NR==1 {
                for (i=1;i<=NF;i++) {
                    if      ($i=="monotonic_s")        c_t=i;
                    else if ($i=="bat_power_now_uw")   c_p_uw=i;
                    else if ($i=="bat_current_now_ua") c_i_ua=i;
                    else if ($i=="bat_voltage_now_uv") c_v_uv=i;
                    else if ($i=="bat_current_ma")     c_i_ma=i;
                    else if ($i=="bat_voltage_mv")     c_v_mv=i;
                }
                next
            }
            {
                if (c_p_uw != "" && $c_p_uw != "")                                   n_p++;
                if (c_i_ua != "" && $c_i_ua != "" && c_v_uv != "" && $c_v_uv != "") n_iv_now++;
                if (c_i_ma != "" && $c_i_ma != "" && c_v_mv != "" && $c_v_mv != "") n_iv_ma++;

                dt = 0
                if (c_t != "" && $c_t != "") {
                    tnow = $c_t + 0
                    if (have_prev) {
                        d = tnow - tprev
                        if (d > 0 && d < 5.0) { dt = d; have_dt = 1 }
                    }
                    tprev = tnow
                    have_prev = 1
                }

                if (dt > 0) {
                    if (c_p_uw != "" && $c_p_uw != "") {
                        pp = $c_p_uw + 0; if (pp < 0) pp = -pp
                        # mWh = uW x dt_s / 3.6e9
                        mwh_p += pp * dt / 3.6e9
                    }
                    if (c_i_ua != "" && $c_i_ua != "" && c_v_uv != "" && $c_v_uv != "") {
                        ii = $c_i_ua + 0; if (ii < 0) ii = -ii
                        vv = $c_v_uv + 0
                        # uW = uA x uV / 1e6
                        if (vv > 0) mwh_iv_now += (ii * vv / 1.0e6) * dt / 3.6e9
                    }
                    if (c_i_ma != "" && $c_i_ma != "" && c_v_mv != "" && $c_v_mv != "") {
                        ii = $c_i_ma + 0; if (ii < 0) ii = -ii
                        vv = $c_v_mv + 0
                        # mW = mA x mV / 1000, mWh = mW x dt / 3600
                        if (vv > 0) mwh_iv_ma += (ii * vv / 1000.0) * dt / 3600.0
                    }
                }
                n_samples++
            }
            END {
                if (n_samples < 2 || !have_dt) { print "FAIL"; exit }
                # Priority power_now, then vi_now, then vi_ma
                thresh = n_samples * 0.5
                if (n_p > thresh && mwh_p > 0) {
                    printf "power_now_integrated %.2f", mwh_p / 4.0
                } else if (n_iv_now > thresh && mwh_iv_now > 0) {
                    printf "vi_now_integrated %.2f", mwh_iv_now / 4.0
                } else if (n_iv_ma > thresh && mwh_iv_ma > 0) {
                    printf "vi_ma_integrated %.2f", mwh_iv_ma / 4.0
                } else {
                    print "FAIL"
                }
            }
        ' "$sensors_csv" 2>/dev/null)
        if [ -n "$m1_pair" ] && [ "$m1_pair" != "FAIL" ]; then
            m1_method=$(echo "$m1_pair" | awk '{print $1}')
            m1_result=$(echo "$m1_pair" | awk '{print $2}')
        fi
    fi
    if [ -n "$m1_result" ]; then
        local m1_ok
        m1_ok=$(awk -v x="$m1_result" 'BEGIN{ print (x+0 > 0.0) ? "1" : "0" }')
        if [ "$m1_ok" = "1" ]; then
            energy_mah="$m1_result"
            energy_method="$m1_method"
            energy_fallback_chain="m1:$m1_method"
        else
            energy_fallback_chain="m1_zero"
        fi
    else
        energy_fallback_chain="m1_missing"
    fi

    # Method 2: charge_counter delta
    if [ "$energy_method" = "n/a" ]; then
        if [ -n "${bat_q0:-}" ] && [ -n "${bat_q1:-}" ] \
           && [ "$bat_q0" -gt 0 ] 2>/dev/null && [ "$bat_q1" -gt 0 ] 2>/dev/null; then
            # Positive delta means discharge.
            local m2_result m2_ok
            m2_result=$(awk -v a="$bat_q0" -v b="$bat_q1" 'BEGIN{ printf "%.2f", (a-b)/1000.0 }')
            # Reject deltas under 1 mAh (quantization) or negative (charging).
            m2_ok=$(awk -v d="$m2_result" 'BEGIN{
                ad = (d<0)?-d:d
                if (d >= 1.0 && ad >= 1.0) print "1"; else print "0"
            }')
            if [ "$m2_ok" = "1" ]; then
                energy_mah="$m2_result"
                energy_method="charge_counter"
                energy_fallback_chain="${energy_fallback_chain}>m2"
            else
                energy_fallback_chain="${energy_fallback_chain}>m2_reject"
            fi
        else
            energy_fallback_chain="${energy_fallback_chain}>m2_missing"
        fi
    fi

    # Method 3: mean(|I|) * wall_s
    if [ "$energy_method" = "n/a" ]; then
        local m3_result=""
        if [ -s "$sensors_csv" ]; then
            m3_result=$(awk -F, -v t="$wall_s" '
                NR==1 {
                    for (i=1;i<=NF;i++) if ($i=="bat_current_ma") c_i=i;
                    next
                }
                {
                    if (c_i=="") next
                    ii = $c_i + 0
                    if (ii < 0) ii = -ii
                    i_sum += ii; n++
                }
                END {
                    if (n < 2 || t+0 <= 0) { print "FAIL"; exit }
                    iavg = i_sum / n
                    printf "%.2f", iavg * (t+0) / 3600.0
                }
            ' "$sensors_csv" 2>/dev/null)
        fi
        if [ -n "$m3_result" ] && [ "$m3_result" != "FAIL" ]; then
            local m3_ok
            m3_ok=$(awk -v x="$m3_result" 'BEGIN{ print (x+0 > 0.0) ? "1" : "0" }')
            if [ "$m3_ok" = "1" ]; then
                energy_mah="$m3_result"
                energy_method="approx"
                energy_fallback_chain="${energy_fallback_chain}>m3"
            else
                energy_fallback_chain="${energy_fallback_chain}>m3_zero"
            fi
        else
            energy_fallback_chain="${energy_fallback_chain}>m3_missing"
        fi
    fi

    # Keep energy_mah but flag it, since charge current can flow into the cell.
    if [ "$charging_detected" = "1" ]; then
        energy_fallback_chain="${energy_fallback_chain}|charging_detected"
    fi

    # Strip NUL bytes from the generated text
    local gen_text
    if [ -s "$host_cell/gen.txt" ]; then
        gen_text=$(tr -d '\000' < "$host_cell/gen.txt")
    else
        gen_text="<empty>"
    fi

    # Per-cell results, read back by the main flow
    cat > "$host_cell/summary.env" <<EOF
prefill_ms='$prefill_ms'
decode_ms='$decode_ms'
decode_tps='$decode_tps'
total_ms='$total_ms'
wall_s='$wall_s'
peak_kv='$peak_kv'
evicted_dec='$evicted_dec'
peak_rss_kb='$peak_rss_kb'
peak_ddr_c='$peak_ddr'
peak_cpu_c='$peak_cpu'
peak_skin_c='$peak_skin'
energy_mah='$energy_mah'
energy_method='$energy_method'
energy_fallback_chain='$energy_fallback_chain'
charging_detected='$charging_detected'
EOF
    # Generated text goes in its own file to avoid quoting issues
    printf '%s' "$gen_text" > "$host_cell/gen_clean.txt"
}

# Run the policies. These flags apply to every policy.
COMMON_EXTRA=()
if [ "$IGNORE_EOS" -eq 1 ]; then
    COMMON_EXTRA+=(--ignore-eos)
fi
if [ -n "$REPEAT_PENALTY" ]; then
    COMMON_EXTRA+=(--repeat-penalty "$REPEAT_PENALTY")
fi

if [ "$RUN_VANILLA" -eq 1 ]; then
    run_one_policy "vanilla" "vanilla" "f16" "f16" \
        --k-nominal "$K_NOMINAL" \
        ${COMMON_EXTRA[@]+"${COMMON_EXTRA[@]}"}
fi

if [ "$RUN_ADAKV" -eq 1 ]; then
    # Same budget split as v1_fa2_stack so the cache size matches across policies.
    run_one_policy "adakv" "adakv" "f16" "f16" "$PHONE_BIN_ADAKV" \
        --k-nominal "$K_NOMINAL" \
        --anchor-top-k "$ANCHOR_TOP_K" \
        --recent-budget "$RECENT_BUDGET" \
        --n-sink "$N_SINK" \
        ${COMMON_EXTRA[@]+"${COMMON_EXTRA[@]}"}
fi

if [ "$RUN_V1FA2" -eq 1 ]; then
    run_one_policy "v1_fa2_stack" "v1_fa2" "q8_0" "f16" \
        --k-nominal "$K_NOMINAL" \
        --anchor-top-k "$ANCHOR_TOP_K" \
        --recent-budget "$RECENT_BUDGET" \
        --n-sink "$N_SINK" \
        --snapkv-decode \
        --no-evict-decode \
        ${COMMON_EXTRA[@]+"${COMMON_EXTRA[@]}"}
fi

# Read back cell summaries (only for policies that actually ran)
if [ "$RUN_VANILLA" -eq 1 ]; then
    # shellcheck disable=SC1091
    { . "$HOST_OUT_ROOT/vanilla/summary.env";       V_PREFILL_MS=$prefill_ms; V_DECODE_TPS=$decode_tps;
      V_WALL_S=$wall_s; V_PEAK_DDR=$peak_ddr_c;     V_PEAK_CPU=$peak_cpu_c;
      V_PEAK_SKIN=$peak_skin_c;  V_PEAK_RSS_KB=$peak_rss_kb;
      V_KV=$peak_kv;  V_EV=$evicted_dec;  V_ENERGY=$energy_mah;
      V_ENERGY_METHOD=$energy_method;  V_ENERGY_CHAIN=$energy_fallback_chain;
      V_CHARGING=${charging_detected:-0}; }
    V_GEN=$(cat "$HOST_OUT_ROOT/vanilla/gen_clean.txt" 2>/dev/null || echo "<missing>")
fi
if [ "$RUN_V1FA2" -eq 1 ]; then
    # shellcheck disable=SC1091
    { . "$HOST_OUT_ROOT/v1_fa2_stack/summary.env";  S_PREFILL_MS=$prefill_ms; S_DECODE_TPS=$decode_tps;
      S_WALL_S=$wall_s; S_PEAK_DDR=$peak_ddr_c;     S_PEAK_CPU=$peak_cpu_c;
      S_PEAK_SKIN=$peak_skin_c;  S_PEAK_RSS_KB=$peak_rss_kb;
      S_KV=$peak_kv;  S_EV=$evicted_dec;  S_ENERGY=$energy_mah;
      S_ENERGY_METHOD=$energy_method;  S_ENERGY_CHAIN=$energy_fallback_chain;
      S_CHARGING=${charging_detected:-0}; }
    S_GEN=$(cat "$HOST_OUT_ROOT/v1_fa2_stack/gen_clean.txt" 2>/dev/null || echo "<missing>")
fi
if [ "$RUN_ADAKV" -eq 1 ]; then
    # shellcheck disable=SC1091
    { . "$HOST_OUT_ROOT/adakv/summary.env";         A_PREFILL_MS=$prefill_ms; A_DECODE_TPS=$decode_tps;
      A_WALL_S=$wall_s; A_PEAK_DDR=$peak_ddr_c;     A_PEAK_CPU=$peak_cpu_c;
      A_PEAK_SKIN=$peak_skin_c;  A_PEAK_RSS_KB=$peak_rss_kb;
      A_KV=$peak_kv;  A_EV=$evicted_dec;  A_ENERGY=$energy_mah;
      A_ENERGY_METHOD=$energy_method;  A_ENERGY_CHAIN=$energy_fallback_chain;
      A_CHARGING=${charging_detected:-0}; }
    A_GEN=$(cat "$HOST_OUT_ROOT/adakv/gen_clean.txt" 2>/dev/null || echo "<missing>")
fi

# Parse watchdog log (v1_fa2_stack only)
S_WD_MISSING=0
S_WD_T1=0; S_WD_T2=0; S_WD_T3=0; S_WD_TOTAL=0
S_WD_TIERS_COMPACT="n/a"
if [ "$RUN_V1FA2" -eq 1 ]; then
    S_WD_LOG="$HOST_OUT_ROOT/v1_fa2_stack/watchdog.log"
    S_WD_SUMMARY=$(parse_watchdog_log "$S_WD_LOG")
    if [ ! -f "$S_WD_LOG" ]; then
        echo "[demo] Watchdog: log not captured (may not have started — check stderr)" >&2
        echo "[demo]   expected at: $S_WD_LOG" >&2
        echo "[demo]   phone path:  $PHONE_OUT_ROOT/v1_fa2_stack/watchdog.log" >&2
        S_WD_MISSING=1
    elif [ ! -s "$S_WD_LOG" ]; then
        echo "[demo] Watchdog: log present but empty at $S_WD_LOG" >&2
    fi
    S_WD_T1=$(printf '%s' "$S_WD_SUMMARY" | sed -nE 's/.*tier1=([0-9]+).*/\1/p'); : "${S_WD_T1:=0}"
    S_WD_T2=$(printf '%s' "$S_WD_SUMMARY" | sed -nE 's/.*tier2=([0-9]+).*/\1/p'); : "${S_WD_T2:=0}"
    S_WD_T3=$(printf '%s' "$S_WD_SUMMARY" | sed -nE 's/.*tier3=([0-9]+).*/\1/p'); : "${S_WD_T3:=0}"
    S_WD_TOTAL=$(printf '%s' "$S_WD_SUMMARY" | sed -nE 's/.*total=([0-9]+).*/\1/p'); : "${S_WD_TOTAL:=0}"
    # Compact T1/T2/T3 for the table cell
    if [ "$S_WD_MISSING" -eq 1 ]; then
        S_WD_TIERS_COMPACT="no-log"
    else
        S_WD_TIERS_COMPACT="${S_WD_T1}/${S_WD_T2}/${S_WD_T3}"
    fi
fi

# Prefill ms to s, RSS KB to GB
if [ "$RUN_VANILLA" -eq 1 ]; then
    V_PREFILL_S=$(awk -v x="$V_PREFILL_MS" 'BEGIN{printf "%.2f", x/1000.0}')
    V_RSS_GB=$(awk -v x="$V_PEAK_RSS_KB" 'BEGIN{printf "%.2f", x/1024.0/1024.0}')
fi
if [ "$RUN_V1FA2" -eq 1 ]; then
    S_PREFILL_S=$(awk -v x="$S_PREFILL_MS" 'BEGIN{printf "%.2f", x/1000.0}')
    S_RSS_GB=$(awk -v x="$S_PEAK_RSS_KB" 'BEGIN{printf "%.2f", x/1024.0/1024.0}')
fi
if [ "$RUN_ADAKV" -eq 1 ]; then
    A_PREFILL_S=$(awk -v x="$A_PREFILL_MS" 'BEGIN{printf "%.2f", x/1000.0}')
    A_RSS_GB=$(awk -v x="$A_PEAK_RSS_KB" 'BEGIN{printf "%.2f", x/1024.0/1024.0}')
fi

# Pretty-print the comparison
# Truncate prompt for header
PROMPT_DISPLAY=$(printf '%s' "$PROMPT" | tr '\n' ' ' | cut -c1-60)
[ "${#PROMPT}" -gt 60 ] && PROMPT_DISPLAY="${PROMPT_DISPLAY}..."

BAR_DOUBLE='════════════════════════════════════════════════════════════════════════'
BAR_SINGLE='────────────────────────────────────────────────────────────────────────'

# Banner summary, e.g. K=512 anchor=64 repeat=1.2 [ignore-eos]
REPEAT_BANNER="${REPEAT_PENALTY:-1.10}"
TUNING_SUMMARY="K=$K_NOMINAL anchor=$ANCHOR_TOP_K repeat=$REPEAT_BANNER${IGNORE_EOS_NOTE}"

MODE_SUFFIX=""
if [ "$ENERGY_MODE" = "1" ]; then
    MODE_SUFFIX="${MODE_SUFFIX}, energy-mode"
fi
if [ "$LONG_DECODE" -eq 1 ]; then
    MODE_SUFFIX="${MODE_SUFFIX}, long-decode"
fi

case "$POLICY" in
    both)         HEADER_POLICIES="vanilla vs v1_fa2_stack" ;;
    three-way)   HEADER_POLICIES="vanilla vs adakv vs v1_fa2_stack" ;;
    vanilla)      HEADER_POLICIES="vanilla ONLY" ;;
    v1_fa2_stack) HEADER_POLICIES="v1_fa2_stack ONLY" ;;
    adakv)        HEADER_POLICIES="adakv ONLY" ;;
esac

if [ "$LONG_DECODE" -eq 1 ]; then
    HEADER_LINE="EndurKV Demo ($TUNING_SUMMARY$MODE_SUFFIX): $HEADER_POLICIES on $MODEL_PRETTY (max_tokens=$MAX_TOKENS)${K_REGIME_NOTE}"
else
    HEADER_LINE="EndurKV Demo ($TUNING_SUMMARY$MODE_SUFFIX): $HEADER_POLICIES on $MODEL_PRETTY${K_REGIME_NOTE}"
fi

printf '\n'
printf '╔%s╗\n' "$BAR_DOUBLE"
printf '║ %-70s ║\n' "$HEADER_LINE"
printf '║ %-70s ║\n' "Prompt: \"$PROMPT_DISPLAY\""
printf '╚%s╝\n' "$BAR_DOUBLE"

if [ "$RUN_VANILLA" -eq 1 ]; then
    printf '\n── VANILLA OUTPUT %s\n' "${BAR_SINGLE:18}"
    printf '%s\n' "$V_GEN"
fi

if [ "$RUN_ADAKV" -eq 1 ]; then
    printf '\n── ADAKV OUTPUT %s\n' "${BAR_SINGLE:16}"
    printf '%s\n' "$A_GEN"
fi

if [ "$RUN_V1FA2" -eq 1 ]; then
    printf '\n── V1_FA2_STACK OUTPUT %s\n' "${BAR_SINGLE:22}"
    printf '%s\n' "$S_GEN"
fi

if [ "$POLICY" = "three-way" ]; then
    # 3-way table, deltas relative to vanilla
    printf '\n── COMPARISON (3-way) %s\n' "${BAR_SINGLE:22}"
    printf '%-16s│ %-11s │ %-11s │ %-12s │ %-10s │ %-10s\n' \
        "Metric" "Vanilla" "AdaKV" "v1_fa2_stack" "Δ AdaKV" "Δ v1fa2"
    printf '%-16s┼%s┼%s┼%s┼%s┼%s\n' \
        "────────────────" "─────────────" "─────────────" "──────────────" "────────────" "────────────"

    # One row with two %-deltas vs vanilla
    emit_row3_pct() {
        local label="$1" vv="$2" av="$3" sv="$4"
        printf '%-16s│ %-11s │ %-11s │ %-12s │ %-10s │ %-10s\n' \
            "$label" "$vv" "$av" "$sv" \
            "$(pct_delta "$vv" "$av")" "$(pct_delta "$vv" "$sv")"
    }
    emit_row3_abs() {
        local label="$1" vv="$2" av="$3" sv="$4" suffix="$5"
        printf '%-16s│ %-11s │ %-11s │ %-12s │ %-10s │ %-10s\n' \
            "$label" "$vv" "$av" "$sv" \
            "$(abs_delta "$vv" "$av")$suffix" "$(abs_delta "$vv" "$sv")$suffix"
    }

    emit_row3_pct "Prefill (s)"    "$V_PREFILL_S"   "$A_PREFILL_S"   "$S_PREFILL_S"
    emit_row3_pct "Decode tps"     "$V_DECODE_TPS"  "$A_DECODE_TPS"  "$S_DECODE_TPS"
    emit_row3_pct "Total wall (s)" "$V_WALL_S"      "$A_WALL_S"      "$S_WALL_S"
    emit_row3_abs "Peak DDR (°C)"  "$V_PEAK_DDR"    "$A_PEAK_DDR"    "$S_PEAK_DDR"    "°C"
    emit_row3_abs "Peak CPU (°C)"  "$V_PEAK_CPU"    "$A_PEAK_CPU"    "$S_PEAK_CPU"    "°C"
    emit_row3_abs "Peak Skin (°C)" "$V_PEAK_SKIN"   "$A_PEAK_SKIN"   "$S_PEAK_SKIN"   "°C"
    emit_row3_pct "Peak RSS (GB)"  "$V_RSS_GB"      "$A_RSS_GB"      "$S_RSS_GB"

    # No energy delta when either side is n/a
    if [ "${V_ENERGY:-n/a}" = "n/a" ] || [ "${A_ENERGY:-n/a}" = "n/a" ]; then
        A_ENERGY_DELTA="—"
    else
        A_ENERGY_DELTA=$(pct_delta "$V_ENERGY" "$A_ENERGY")
    fi
    if [ "${V_ENERGY:-n/a}" = "n/a" ] || [ "${S_ENERGY:-n/a}" = "n/a" ]; then
        S_ENERGY_DELTA="—"
    else
        S_ENERGY_DELTA=$(pct_delta "$V_ENERGY" "$S_ENERGY")
    fi
    printf '%-16s│ %-11s │ %-11s │ %-12s │ %-10s │ %-10s\n' \
        "Energy (mAh)"   "${V_ENERGY:-n/a}"  "${A_ENERGY:-n/a}"  "${S_ENERGY:-n/a}" \
        "$A_ENERGY_DELTA"  "$S_ENERGY_DELTA"
    emit_row3_pct "KV cells final" "$V_KV"          "$A_KV"          "$S_KV"
    printf '%-16s│ %-11s │ %-11s │ %-12s │ %-10s │ %-10s\n' \
        "Evicted total"  "0"             "$A_EV"            "$S_EV"            "—" "—"
    printf '%-16s│ %-11s │ %-11s │ %-12s │ %-10s │ %-10s\n' \
        "Watchdog tiers" "n/a"           "n/a"              "$S_WD_TIERS_COMPACT" "—" "T1/T2/T3"

    printf '  energy method: vanilla=%s, adakv=%s, v1_fa2_stack=%s\n' \
        "${V_ENERGY_METHOD:-n/a}" "${A_ENERGY_METHOD:-n/a}" "${S_ENERGY_METHOD:-n/a}"
    if [ "${V_CHARGING:-0}" = "1" ] || [ "${A_CHARGING:-0}" = "1" ] || [ "${S_CHARGING:-0}" = "1" ]; then
        printf '  energy WARNING: charging_detected during run (vanilla=%s, adakv=%s, v1_fa2_stack=%s) — energy values UNRELIABLE\n' \
            "${V_CHARGING:-0}" "${A_CHARGING:-0}" "${S_CHARGING:-0}"
    fi
elif [ "$POLICY" = "both" ]; then
    printf '\n── COMPARISON %s\n' "${BAR_SINGLE:14}"
    printf '%-16s│ %-11s │ %-12s │ %-10s\n' "Metric" "Vanilla" "v1_fa2_stack" "Δ"
    printf '%-16s┼%s┼%s┼%s\n' "────────────────" "─────────────" "──────────────" "──────────"

    printf '%-16s│ %-11s │ %-12s │ %-10s\n' \
        "Prefill (s)"    "$V_PREFILL_S"   "$S_PREFILL_S"   "$(pct_delta "$V_PREFILL_S" "$S_PREFILL_S")"
    printf '%-16s│ %-11s │ %-12s │ %-10s\n' \
        "Decode tps"     "$V_DECODE_TPS"  "$S_DECODE_TPS"  "$(pct_delta "$V_DECODE_TPS" "$S_DECODE_TPS")"
    printf '%-16s│ %-11s │ %-12s │ %-10s\n' \
        "Total wall (s)" "$V_WALL_S"      "$S_WALL_S"      "$(pct_delta "$V_WALL_S" "$S_WALL_S")"
    printf '%-16s│ %-11s │ %-12s │ %-10s\n' \
        "Peak DDR (°C)"  "$V_PEAK_DDR"    "$S_PEAK_DDR"    "$(abs_delta "$V_PEAK_DDR" "$S_PEAK_DDR")°C"
    printf '%-16s│ %-11s │ %-12s │ %-10s\n' \
        "Peak CPU (°C)"  "$V_PEAK_CPU"    "$S_PEAK_CPU"    "$(abs_delta "$V_PEAK_CPU" "$S_PEAK_CPU")°C"
    printf '%-16s│ %-11s │ %-12s │ %-10s\n' \
        "Peak Skin (°C)" "$V_PEAK_SKIN"   "$S_PEAK_SKIN"   "$(abs_delta "$V_PEAK_SKIN" "$S_PEAK_SKIN")°C"
    printf '%-16s│ %-11s │ %-12s │ %-10s\n' \
        "Peak RSS (GB)"  "$V_RSS_GB"      "$S_RSS_GB"      "$(pct_delta "$V_RSS_GB" "$S_RSS_GB")"
    # No energy delta when either side is n/a. The method is printed below the table.
    if [ "${V_ENERGY:-n/a}" = "n/a" ] || [ "${S_ENERGY:-n/a}" = "n/a" ]; then
        V_ENERGY_DELTA="—"
    else
        V_ENERGY_DELTA=$(pct_delta "$V_ENERGY" "$S_ENERGY")
    fi
    printf '%-16s│ %-11s │ %-12s │ %-10s\n' \
        "Energy (mAh)"   "${V_ENERGY:-n/a}"  "${S_ENERGY:-n/a}"  "$V_ENERGY_DELTA"
    printf '%-16s│ %-11s │ %-12s │ %-10s\n' \
        "KV cells final" "$V_KV"          "$S_KV"          "$(pct_delta "$V_KV" "$S_KV")"
    printf '%-16s│ %-11s │ %-12s │ %-10s\n' \
        "Evicted total"  "0"              "$S_EV"          "—"
    printf '%-16s│ %-11s │ %-12s │ %-10s\n' \
        "Watchdog tiers" "n/a"            "$S_WD_TIERS_COMPACT" "T1/T2/T3"

    # Energy method names are defined in run_one_policy.
    printf '  energy method: vanilla=%s, v1_fa2_stack=%s\n' \
        "${V_ENERGY_METHOD:-n/a}" "${S_ENERGY_METHOD:-n/a}"
    if [ "${V_CHARGING:-0}" = "1" ] || [ "${S_CHARGING:-0}" = "1" ]; then
        printf '  energy WARNING: charging_detected during run (vanilla=%s, v1_fa2_stack=%s) — energy values UNRELIABLE\n' \
            "${V_CHARGING:-0}" "${S_CHARGING:-0}"
    fi
else
    # Single policy: one-column table
    if [ "$RUN_VANILLA" -eq 1 ]; then
        SINGLE_LABEL="vanilla"
        S1_PREFILL_S="$V_PREFILL_S"; S1_DECODE_TPS="$V_DECODE_TPS"; S1_WALL_S="$V_WALL_S"
        S1_PEAK_DDR="$V_PEAK_DDR"; S1_PEAK_CPU="$V_PEAK_CPU"; S1_PEAK_SKIN="$V_PEAK_SKIN"
        S1_RSS_GB="$V_RSS_GB"; S1_ENERGY="${V_ENERGY:-n/a}"; S1_ENERGY_METHOD="${V_ENERGY_METHOD:-n/a}"
        S1_KV="$V_KV"; S1_EV="0"; S1_WD_TIERS="n/a"; S1_CHARGING="${V_CHARGING:-0}"
    elif [ "$RUN_ADAKV" -eq 1 ]; then
        SINGLE_LABEL="adakv"
        S1_PREFILL_S="$A_PREFILL_S"; S1_DECODE_TPS="$A_DECODE_TPS"; S1_WALL_S="$A_WALL_S"
        S1_PEAK_DDR="$A_PEAK_DDR"; S1_PEAK_CPU="$A_PEAK_CPU"; S1_PEAK_SKIN="$A_PEAK_SKIN"
        S1_RSS_GB="$A_RSS_GB"; S1_ENERGY="${A_ENERGY:-n/a}"; S1_ENERGY_METHOD="${A_ENERGY_METHOD:-n/a}"
        S1_KV="$A_KV"; S1_EV="$A_EV"; S1_WD_TIERS="n/a"; S1_CHARGING="${A_CHARGING:-0}"
    else
        SINGLE_LABEL="v1_fa2_stack"
        S1_PREFILL_S="$S_PREFILL_S"; S1_DECODE_TPS="$S_DECODE_TPS"; S1_WALL_S="$S_WALL_S"
        S1_PEAK_DDR="$S_PEAK_DDR"; S1_PEAK_CPU="$S_PEAK_CPU"; S1_PEAK_SKIN="$S_PEAK_SKIN"
        S1_RSS_GB="$S_RSS_GB"; S1_ENERGY="${S_ENERGY:-n/a}"; S1_ENERGY_METHOD="${S_ENERGY_METHOD:-n/a}"
        S1_KV="$S_KV"; S1_EV="$S_EV"; S1_WD_TIERS="$S_WD_TIERS_COMPACT"; S1_CHARGING="${S_CHARGING:-0}"
    fi

    printf '\n── SINGLE-POLICY SUMMARY (%s) %s\n' "$SINGLE_LABEL" "${BAR_SINGLE:34}"
    printf '%-16s│ %-12s\n' "Metric" "$SINGLE_LABEL"
    printf '%-16s┼%s\n' "────────────────" "──────────────"
    printf '%-16s│ %-12s\n' "Prefill (s)"    "$S1_PREFILL_S"
    printf '%-16s│ %-12s\n' "Decode tps"     "$S1_DECODE_TPS"
    printf '%-16s│ %-12s\n' "Total wall (s)" "$S1_WALL_S"
    printf '%-16s│ %-12s\n' "Peak DDR (°C)"  "$S1_PEAK_DDR"
    printf '%-16s│ %-12s\n' "Peak CPU (°C)"  "$S1_PEAK_CPU"
    printf '%-16s│ %-12s\n' "Peak Skin (°C)" "$S1_PEAK_SKIN"
    printf '%-16s│ %-12s\n' "Peak RSS (GB)"  "$S1_RSS_GB"
    printf '%-16s│ %-12s\n' "Energy (mAh)"   "$S1_ENERGY"
    printf '%-16s│ %-12s\n' "KV cells final" "$S1_KV"
    printf '%-16s│ %-12s\n' "Evicted total"  "$S1_EV"
    printf '%-16s│ %-12s\n' "Watchdog tiers" "$S1_WD_TIERS"
    printf '  energy method: %s=%s\n' "$SINGLE_LABEL" "$S1_ENERGY_METHOD"
    if [ "$S1_CHARGING" = "1" ]; then
        printf '  energy WARNING: charging_detected during run (%s) — energy values UNRELIABLE\n' "$SINGLE_LABEL"
    fi
fi

# Watchdog activity summary (v1_fa2_stack only).
if [ "$RUN_V1FA2" -eq 1 ]; then
    if [ "$S_WD_MISSING" -eq 1 ]; then
        printf '\nWatchdog activity (v1_fa2_stack only): log not captured (may not have started — check stderr)\n'
    elif [ "$S_WD_TOTAL" -gt 0 ] 2>/dev/null; then
        printf '\nWatchdog activity (v1_fa2_stack only): %d tier transitions (tier1=%d, tier2=%d, tier3=%d)\n' \
            "$S_WD_TOTAL" "$S_WD_T1" "$S_WD_T2" "$S_WD_T3"
    else
        printf '\nWatchdog activity (v1_fa2_stack only): no transitions (stayed at MAX freq throughout)\n'
    fi
fi

# Verdict, only when at least two policies ran
if [ "$POLICY" = "three-way" ]; then
    printf '\n── VERDICT (3-way) %s\n' "${BAR_SINGLE:19}"
    VERDICT3=$(awk \
        -v vtps="$V_DECODE_TPS"  -v atps="$A_DECODE_TPS"  -v stps="$S_DECODE_TPS" \
        -v vwall="$V_WALL_S"     -v awall="$A_WALL_S"     -v swall="$S_WALL_S" \
        -v vkv="$V_KV"           -v akv="$A_KV"           -v skv="$S_KV" 'BEGIN{
        function pct(a,b){ return (a>0) ? (b-a)/a*100.0 : 0 }
        da_tps  = pct(vtps,  atps);  ds_tps  = pct(vtps,  stps)
        da_wall = pct(vwall, awall); ds_wall = pct(vwall, swall)
        da_kv   = pct(vkv,   akv);   ds_kv   = pct(vkv,   skv)
        function s(x){ return (x>=0)?"+":"" }
        printf "vs vanilla: AdaKV=%s%.1f%% tps, %s%.1f%% wall, %s%.1f%% KV  |  v1_fa2_stack=%s%.1f%% tps, %s%.1f%% wall, %s%.1f%% KV",
            s(da_tps),  da_tps,  s(da_wall), da_wall, s(da_kv), da_kv,
            s(ds_tps),  ds_tps,  s(ds_wall), ds_wall, s(ds_kv), ds_kv
    }')
    printf '%s\n' "$VERDICT3"
    if [ "$S_WD_MISSING" -eq 1 ]; then
        printf 'Watchdog: log not captured (may not have started — check stderr)\n\n'
    elif [ "$S_WD_TOTAL" -gt 0 ] 2>/dev/null; then
        printf 'Watchdog engaged (v1_fa2_stack only): %d tier transitions (tier1=%d, tier2=%d, tier3=%d)\n\n' \
            "$S_WD_TOTAL" "$S_WD_T1" "$S_WD_T2" "$S_WD_T3"
    else
        printf 'Watchdog (v1_fa2_stack only): no transitions (stayed at MAX freq throughout)\n\n'
    fi
    echo "Artifacts (host):  $HOST_OUT_ROOT"
    echo "Artifacts (phone): $PHONE_OUT_ROOT"
    exit 0
fi

if [ "$POLICY" != "both" ]; then
    echo ""
    echo "Artifacts (host):  $HOST_OUT_ROOT"
    echo "Artifacts (phone): $PHONE_OUT_ROOT"
    exit 0
fi

printf '\n── VERDICT %s\n' "${BAR_SINGLE:11}"
if [ "$LONG_DECODE" -eq 1 ]; then
    VERDICT=$(awk -v vwall="$V_WALL_S"   -v swall="$S_WALL_S" \
                  -v vkv="$V_KV"         -v skv="$S_KV" \
                  -v sev="$S_EV"         -v vrss="$V_RSS_GB" -v srss="$S_RSS_GB" 'BEGIN{
        dwall = (vwall>0)? (swall-vwall)/vwall*100.0 : 0;
        dkvp  = (vkv>0)?   (skv-vkv)/vkv*100.0       : 0;
        drss  = (vrss>0)?  (srss-vrss)/vrss*100.0    : 0;
        sign_wall = (dwall>=0)?"+":"";
        sign_kv   = (dkvp>=0)?"+":"";
        sign_rss  = (drss>=0)?"+":"";
        printf "v1_fa2_stack (long-decode): %s%.1f%% wall time, %d decode-evictions, %s%.1f%% peak KV cells (cache shrink), %s%.1f%% peak RSS vs vanilla",
            sign_wall, dwall, sev, sign_kv, dkvp, sign_rss, drss
    }')
else
    VERDICT=$(awk -v vtps="$V_DECODE_TPS" -v stps="$S_DECODE_TPS" \
                  -v vwall="$V_WALL_S"    -v swall="$S_WALL_S" \
                  -v vpref="$V_PREFILL_S" -v spref="$S_PREFILL_S" 'BEGIN{
        dtps  = (vtps>0)?  (stps-vtps)/vtps*100.0    : 0;
        dwall = (vwall>0)? (swall-vwall)/vwall*100.0 : 0;
        dpref = (vpref>0)? (spref-vpref)/vpref*100.0 : 0;
        sign_tps  = (dtps>=0)?"+":"";
        sign_wall = (dwall>=0)?"+":"";
        sign_pref = (dpref>=0)?"+":"";
        printf "v1_fa2_stack (short-decode): %s%.1f%% decode tps, %s%.1f%% prefill latency, %s%.1f%% total wall vs vanilla",
            sign_tps, dtps, sign_pref, dpref, sign_wall, dwall
    }')
fi
printf '%s\n' "$VERDICT"

if [ "$S_WD_MISSING" -eq 1 ]; then
    printf 'Watchdog: log not captured (may not have started — check stderr)\n\n'
elif [ "$S_WD_TOTAL" -gt 0 ] 2>/dev/null; then
    printf 'Watchdog engaged: %d tier transitions (tier1=%d, tier2=%d, tier3=%d)\n\n' \
        "$S_WD_TOTAL" "$S_WD_T1" "$S_WD_T2" "$S_WD_T3"
else
    printf 'Watchdog: no transitions (stayed at MAX freq throughout)\n\n'
fi

echo "Artifacts (host):  $HOST_OUT_ROOT"
echo "Artifacts (phone): $PHONE_OUT_ROOT"
