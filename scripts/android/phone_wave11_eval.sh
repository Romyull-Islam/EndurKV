#!/system/bin/sh
# Phone-side eval sweep: MODELS x POLICIES x BENCHES (ppl on WikiText-2, NIAH).
# Each policy keeps its own FA mode. Before each cell: cool gate and a 4 GB
# memory gate. DVFS is pinned for the whole sweep. The script detaches itself
# so ADB drops do not kill the run.
#
# Per cell, under $OUT_DIR/<MODEL>/<POLICY>/<BENCH>/: stress.csv, sensors.csv
# and iter*/ (steps.csv, meta.json, logs, gen.txt).
# Progress: $OUT_DIR/progress.log. $OUT_DIR/DONE is touched on success.

set -u

# Knobs (overridable via env)
WORKDIR=${WORKDIR:-/data/local/tmp/endurkv}
BIN=${BIN:-$WORKDIR/bin_cpu/eviction_bench}
SCRIPTS=${SCRIPTS:-$WORKDIR/scripts}

COOL_TARGET_DC=${COOL_TARGET_DC:-330}        # battery temperature is decideg-C (33.0 °C = 330)
COOL_DDR_C=${COOL_DDR_C:-40}                 # °C
COOL_MAX_S=${COOL_MAX_S:-1800}
SAMPLE_HZ=${SAMPLE_HZ:-5}
MIN_FREE_GB=${MIN_FREE_GB:-4}
DDR_ZONE=${DDR_ZONE:-/sys/class/thermal/thermal_zone47}

# Bench params
K_NOMINAL=${K_NOMINAL:-512}
ANCHOR_TOP_K=${ANCHOR_TOP_K:-32}
N_SINK=${N_SINK:-4}
RECENT_BUDGET=$(( K_NOMINAL - ANCHOR_TOP_K - N_SINK ))
[ "$RECENT_BUDGET" -lt 32 ] && RECENT_BUDGET=32

THREADS=${THREADS:-4}
NGL=${NGL:-0}                                 # CPU stack
N_BATCH=${N_BATCH:-512}
UBATCH=${UBATCH:-64}
CTX_SIZE=${CTX_SIZE:-4096}
SEED=${SEED:-42}

# PPL: 8 chunks x 2048 tokens, pre-split on the host. Prefill chunk i and
# teacher-force chunk i+1. Eval text must be disjoint from the prefill text,
# or PPL collapses to about 1.0.
PPL_N_CHUNKS=${PPL_N_CHUNKS:-8}
PPL_CHUNK_TOKENS=${PPL_CHUNK_TOKENS:-2048}
PPL_TEXT_DIR=${PPL_TEXT_DIR:-$WORKDIR/eval_data}     # wiki.test.raw.chunk0..chunk7

# NIAH: tier 1 uses 8 of the 32 stimuli, tier 2 uses all 32.
NIAH_TIER=${NIAH_TIER:-1}
if [ "$NIAH_TIER" = "1" ]; then
    NIAH_N_STIMULI=${NIAH_N_STIMULI:-8}
else
    NIAH_N_STIMULI=${NIAH_N_STIMULI:-32}
fi
NIAH_PROMPT_DIR=${NIAH_PROMPT_DIR:-$WORKDIR/eval_data/niah}   # niah_stimulus_00..31.txt
NIAH_MAX_TOKENS=${NIAH_MAX_TOKENS:-64}

# Output root + progress
OUT_DIR=${OUT_DIR:-$WORKDIR/logs/wave11_eval_$(date +%s)}
mkdir -p "$OUT_DIR"
PROG="$OUT_DIR/progress.log"

# Watchdog state. start_watchdog sets these per cell so cells do not clobber
# each other's logs or race on start/stop.
WATCHDOG_STOP=""
WATCHDOG_LOG=""
WATCHDOG_PID=""

# Models on phone (tag | gguf | ctx) - env-overridable
MODELS=${MODELS:-"
Phi-3-mini-128k|$WORKDIR/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf|$CTX_SIZE
Llama-3.2-1B|$WORKDIR/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf|$CTX_SIZE
Gemma-2-2B|$WORKDIR/models/gemma-2-2b-it-Q4_K_M.gguf|$CTX_SIZE
"}

POLICIES=${POLICIES:-"vanilla h2o v1_fa2_stack tova streamingllm"}
BENCHES=${BENCHES:-"ppl niah"}

# Helpers
log()        { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$PROG"; }
ddr_temp_c() { awk '{printf "%d", $1/1000}' "$DDR_ZONE/temp" 2>/dev/null || echo 0; }
mem_free_gb(){ awk '/MemAvailable/{printf "%.2f", $2/1024/1024}' /proc/meminfo; }
skin_dc()    { dumpsys battery 2>/dev/null | awk '/temperature/{print $2; exit}'; }

cool_phone() {
    local cell="$1"; local t0; t0=$(date +%s)
    while true; do
        local sk=$(skin_dc) dd=$(ddr_temp_c)
        if [ -n "$sk" ] && [ "$sk" -le "$COOL_TARGET_DC" ] 2>/dev/null && [ "$dd" -le "$COOL_DDR_C" ]; then
            log "  $cell: cool gate OK skin=${sk}dC DDR=${dd}C"
            return 0
        fi
        local el=$(( $(date +%s) - t0 ))
        if [ "$el" -ge "$COOL_MAX_S" ]; then
            log "  $cell: WARN cool timeout skin=${sk}dC DDR=${dd}C after ${el}s — proceeding"
            return 0
        fi
        sleep 15
    done
}

wait_for_memory() {
    local cell="$1"; local t0; t0=$(date +%s)
    local target_kb=$(( MIN_FREE_GB * 1024 * 1024 ))
    while true; do
        local mfr_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
        if [ "$mfr_kb" -ge "$target_kb" ] 2>/dev/null; then
            log "  $cell: mem-gate OK free=$(mem_free_gb) GB"
            return 0
        fi
        local el=$(( $(date +%s) - t0 ))
        if [ "$el" -ge 300 ]; then
            log "  $cell: WARN mem-gate timeout free=$(mem_free_gb) GB — proceeding"
            return 0
        fi
        sleep 20
    done
}

start_sampler() {
    local out_csv="$1"
    sh "$SCRIPTS/sample_sensors.sh" --out "$out_csv" --hz "$SAMPLE_HZ" </dev/null >/dev/null 2>&1 &
    echo $!
}

stop_sampler() {
    local pid="$1"
    kill "$pid" 2>/dev/null
    pgrep -f sample_sensors | xargs -r kill 2>/dev/null
}

start_watchdog() {
    # Log, stop sentinel and pid file live in cell_dir so the next cell cannot
    # truncate this cell's log. Default is the multi-sensor v2 watchdog,
    # WATCHDOG_VERSION=v1 selects the DDR-only script.
    local cell_dir="$1"
    local policy="${2:-}"
    WATCHDOG_LOG="$cell_dir/watchdog.log"
    WATCHDOG_STOP="$cell_dir/watchdog.stop"
    WATCHDOG_PID_FILE="$cell_dir/watchdog.pid"
    rm -f "$WATCHDOG_STOP" "$WATCHDOG_PID_FILE"

    local WD_SCRIPT="preempt_throttle_watchdog_v2.sh"
    if [ "${WATCHDOG_VERSION:-v2}" = "v1" ]; then
        WD_SCRIPT="preempt_throttle_watchdog.sh"
    fi

    log "  starting preempt-throttle watchdog v${WATCHDOG_VERSION:-2} (root) log=$WATCHDOG_LOG"
    su -c "sh $SCRIPTS/$WD_SCRIPT $WATCHDOG_LOG $WATCHDOG_STOP $DDR_ZONE" \
        </dev/null >/dev/null 2>&1 &
    WATCHDOG_PID=$!
    echo "$WATCHDOG_PID" > "$WATCHDOG_PID_FILE"
}

stop_watchdog() {
    # Wait for the watchdog to exit so two watchdogs never fight over
    # scaling_max_freq.
    [ -n "$WATCHDOG_STOP" ] && touch "$WATCHDOG_STOP"
    local pid="$WATCHDOG_PID"
    if [ -n "$pid" ]; then
        local waited=0
        while kill -0 "$pid" 2>/dev/null; do
            if [ "$waited" -ge 10 ]; then
                log "  watchdog pid=$pid did not exit in 10s — SIGKILL"
                kill -9 "$pid" 2>/dev/null
                break
            fi
            sleep 1
            waited=$(( waited + 1 ))
        done
    fi
    log "  watchdog stopped"
    WATCHDOG_PID=""
    WATCHDOG_STOP=""
    WATCHDOG_LOG=""
}

# Map a policy name to one line of eviction_bench flags. vanilla runs FA-on,
# the binary runs FA-off for scoring policies, v1_fa2_stack runs FA-off
# prefill and FA-on decode.
policy_flags() {
    case "$1" in
        vanilla)
            # Vanilla path, FA-on by default.
            printf -- "--policy vanilla --cache-type-k f16 --cache-type-v f16"
            ;;
        v1)
            printf -- "--policy v1 --k-nominal %s --anchor-top-k %s --recent-budget %s --n-sink %s --cache-type-k f16 --cache-type-v f16" \
                "$K_NOMINAL" "$ANCHOR_TOP_K" "$RECENT_BUDGET" "$N_SINK"
            ;;
        tova)
            printf -- "--policy tova --k-nominal %s --n-sink %s --cache-type-k f16 --cache-type-v f16" \
                "$K_NOMINAL" "$N_SINK"
            ;;
        h2o)
            printf -- "--policy h2o --k-nominal %s --n-sink %s --cache-type-k f16 --cache-type-v f16" \
                "$K_NOMINAL" "$N_SINK"
            ;;
        streamingllm)
            # StreamingLLM (Xiao et al., ICLR 2024): sinks plus a sliding window.
            # K_nominal is the total budget (n_sink + window). FA-on by default.
            printf -- "--policy streamingllm --k-nominal %s --n-sink %s --cache-type-k f16 --cache-type-v f16" \
                "$K_NOMINAL" "$N_SINK"
            ;;
        v1_fa2_stack)
            # Q8 K. V stays f16 because of the state-swap layout.
            # recent + anchor + sink = K_nominal.
            printf -- "--policy v1_fa2 --k-nominal %s --anchor-top-k %s --recent-budget %s --n-sink %s --cache-type-k q8_0 --cache-type-v f16" \
                "$K_NOMINAL" "$ANCHOR_TOP_K" "$RECENT_BUDGET" "$N_SINK"
            ;;
        v1_entropy_stack)
            # Confidence-weighted attention scoring with the v1_fa2 thermal stack
            # (FA-off prefill, state-swap, FA-on decode, Q8 K, tiered eviction).
            printf -- "--policy v1_entropy_stack --k-nominal %s --anchor-top-k %s --recent-budget %s --n-sink %s --cache-type-k q8_0 --cache-type-v f16" \
                "$K_NOMINAL" "$ANCHOR_TOP_K" "$RECENT_BUDGET" "$N_SINK"
            ;;
        v1_predictive_stack)
            # Slope scoring over time. FA-off for all of decode because slope()
            # reads kq_soft_max every step. Q8 K to cut DRAM bandwidth.
            printf -- "--policy v1_predictive_stack --k-nominal %s --n-sink %s --cache-type-k q8_0 --cache-type-v f16" \
                "$K_NOMINAL" "$N_SINK"
            ;;
        v1_fa2_hybrid)
            # Ablation: H2O eviction (recent K/2 + heavy K/2) plus the watchdog
            # and memory gate, f16 K and V, no state-swap.
            printf -- "--policy v1_fa2_hybrid --k-nominal %s --n-sink %s --cache-type-k f16 --cache-type-v f16" \
                "$K_NOMINAL" "$N_SINK"
            ;;
        v1_fa2_f16)
            # Ablation: v1_fa2_stack with f16 K, to test whether the Q8 K
            # seq_add skip costs PPL.
            printf -- "--policy v1_fa2_f16 --k-nominal %s --anchor-top-k %s --recent-budget %s --n-sink %s --cache-type-k f16 --cache-type-v f16" \
                "$K_NOMINAL" "$ANCHOR_TOP_K" "$RECENT_BUDGET" "$N_SINK"
            ;;
        endurkv_optimal)
            # H2O eviction (50/50 recent and heavy) with the v1_fa2 thermal stack:
            # state-swap to FA-on decode, Q8 K, f16 V. H2O has no anchors, so
            # only --k-nominal and --n-sink are passed.
            printf -- "--policy endurkv_optimal --k-nominal %s --n-sink %s --cache-type-k q8_0 --cache-type-v f16" \
                "$K_NOMINAL" "$N_SINK"
            ;;
        endurkv_adaptive)
            # v1 per-head spread gate plus a runtime K controller driven by DDR,
            # CPU, skin and battery temperatures (mu_thermal in [0.7, 1.3]).
            # f16 K and V and no state-swap, since both Q8 K and state-swap cost PPL.
            printf -- "--policy endurkv_adaptive --k-nominal %s --anchor-top-k %s --n-sink %s --cache-type-k f16 --cache-type-v f16" \
                "$K_NOMINAL" "$ANCHOR_TOP_K" "$N_SINK"
            ;;
        adakv)
            # Ada-KV (Feng et al.), Algorithm 1 adaptive per-head budgets with the
            # Ada-SnapKV recipe: FA-off prefill, mask frozen during decode, no
            # state-swap. f16 K and V to isolate the budget allocation.
            printf -- "--policy adakv --k-nominal %s --cache-type-k f16 --cache-type-v f16" "$K_NOMINAL"
            ;;
        *) printf -- "" ;;
    esac
}

# Policies that run with the preempt-throttle watchdog.
needs_watchdog() {
    case "$1" in
        v1_fa2_stack|v1_fa2_hybrid|v1_fa2_f16|v1_entropy_stack|v1_predictive_stack|endurkv_optimal|endurkv_adaptive) return 0 ;;
        *) return 1 ;;
    esac
}

# Read a numeric field from meta.json.
meta_num() {
    local f="$1" key="$2"
    grep -oE "\"${key}\": *[0-9.]+" "$f" 2>/dev/null | grep -oE '[0-9.]+$' | head -1
}

# Bench drivers

# PPL on WikiText-2: iterate the 8 pre-split chunks, log a row per chunk.
run_ppl_cell() {
    local model_tag="$1" model_path="$2" model_ctx="$3" policy="$4" cell_dir="$5"
    local pflags="$(policy_flags "$policy")"

    echo "chunk_idx,t_elapsed_s,exit,prefill_ms,decode_tps,n_decode_steps,peak_kv_cells,peak_rss_kb,evicted,ppl,niah_correct,k_used,ddr_start_c,mem_free_gb_start" \
        > "$cell_dir/stress.csv"

    local T0=$(date +%s)
    local i=0
    # Prefill chunk i and score chunk i+1, so there are PPL_N_CHUNKS-1 iterations.
    # Scoring the prefilled chunk itself would collapse PPL to about 1.0.
    local last_iter=$(( PPL_N_CHUNKS - 1 ))
    while [ "$i" -lt "$last_iter" ]; do
        local idir="$cell_dir/iter$(printf %04d "$i")"
        mkdir -p "$idir"

        local prefill_chunk="$PPL_TEXT_DIR/wiki.test.raw.chunk${i}"
        local eval_chunk="$PPL_TEXT_DIR/wiki.test.raw.chunk$(( i + 1 ))"
        if [ ! -f "$prefill_chunk" ] || [ ! -f "$eval_chunk" ]; then
            # Write a sentinel row (exit=-1) so chunk_idx has no gaps. A gap
            # would look like the cell never ran. preflight normally catches this.
            local ddr0=$(ddr_temp_c) mfr0=$(mem_free_gb)
            log "  PPL chunk missing: prefill=$prefill_chunk eval=$eval_chunk — emitting sentinel row exit=-1"
            echo "$i,0,-1,0,0,0,0,0,0,0,,${K_NOMINAL},${ddr0},${mfr0}" >> "$cell_dir/stress.csv"
            i=$(( i + 1 ))
            continue
        fi

        local ddr0=$(ddr_temp_c) mfr0=$(mem_free_gb)
        # shellcheck disable=SC2086
        # eviction_bench requires --prompt and --prompt-id even in ppl mode.
        LD_LIBRARY_PATH="$WORKDIR/bin_cpu" "$BIN" \
            --model "$model_path" \
            --eval-mode ppl --eval-text "$eval_chunk" \
            --prompt "$prefill_chunk" --prompt-id "ppl_chunk_${i}_eval_$((i+1))" \
            $pflags \
            --max-tokens "$PPL_CHUNK_TOKENS" --ignore-eos \
            --ctx-size "$model_ctx" --seed "$SEED" \
            --threads "$THREADS" --n-gpu-layers "$NGL" \
            --n-batch "$N_BATCH" --ubatch-size "$UBATCH" \
            --greedy \
            --out-csv "$idir/steps.csv" --out-meta "$idir/meta.json" \
            > "$idir/stdout.log" 2> "$idir/stderr.log"
        local rc=$?

        local PF DT NS KV RS EV PP
        if [ -f "$idir/meta.json" ]; then
            PF=$(meta_num "$idir/meta.json" prefill_ms)
            DT=$(meta_num "$idir/meta.json" decode_tps)
            NS=$(meta_num "$idir/meta.json" n_decode_steps)
            KV=$(meta_num "$idir/meta.json" peak_kv_cells)
            RS=$(meta_num "$idir/meta.json" peak_rss_kb)
            EV=$(meta_num "$idir/meta.json" evicted_total_decode)
            PP=$(meta_num "$idir/meta.json" perplexity)
        else
            PF=0; DT=0; NS=0; KV=0; RS=0; EV=0; PP=0
        fi
        : "${PF:=0}" "${DT:=0}" "${NS:=0}" "${KV:=0}" "${RS:=0}" "${EV:=0}" "${PP:=0}"

        local el=$(( $(date +%s) - T0 ))
        # niah_correct is empty for PPL rows.
        echo "$i,$el,$rc,$PF,$DT,$NS,$KV,$RS,$EV,$PP,,$K_NOMINAL,$ddr0,$mfr0" >> "$cell_dir/stress.csv"
        log "  PPL[$model_tag/$policy] chunk=$i exit=$rc ppl=$PP tps=$DT kv=$KV DDR=${ddr0}C free=${mfr0}GB"
        i=$(( i + 1 ))
    done
}

# NIAH: gen mode, one gen.txt per stimulus for the host-side judge.
run_niah_cell() {
    local model_tag="$1" model_path="$2" model_ctx="$3" policy="$4" cell_dir="$5"
    local pflags="$(policy_flags "$policy")"

    echo "stim_idx,t_elapsed_s,exit,prefill_ms,decode_tps,n_decode_steps,peak_kv_cells,peak_rss_kb,evicted,ppl,niah_correct,k_used,ddr_start_c,mem_free_gb_start" \
        > "$cell_dir/stress.csv"

    local T0=$(date +%s)
    local s=0
    while [ "$s" -lt "$NIAH_N_STIMULI" ]; do
        local sidx=$(printf "%02d" "$s")
        local idir="$cell_dir/iter${sidx}"
        mkdir -p "$idir"
        local stim="$NIAH_PROMPT_DIR/niah_stimulus_${sidx}.txt"
        if [ ! -f "$stim" ]; then
            log "  NIAH stimulus missing: $stim — skipping"
            s=$(( s + 1 ))
            continue
        fi

        local ddr0=$(ddr_temp_c) mfr0=$(mem_free_gb)
        # The binary writes generated text to --out-gen for the host judge.
        # shellcheck disable=SC2086
        LD_LIBRARY_PATH="$WORKDIR/bin_cpu" "$BIN" \
            --model "$model_path" \
            --prompt "$stim" --prompt-id "niah_${sidx}" \
            --eval-mode gen \
            $pflags \
            --max-tokens "$NIAH_MAX_TOKENS" --ignore-eos \
            --ctx-size "$model_ctx" --seed "$SEED" \
            --threads "$THREADS" --n-gpu-layers "$NGL" \
            --n-batch "$N_BATCH" --ubatch-size "$UBATCH" \
            --greedy \
            --out-csv "$idir/steps.csv" --out-meta "$idir/meta.json" \
            --out-gen "$idir/gen.txt" \
            > "$idir/stdout.log" 2> "$idir/stderr.log"
        local rc=$?

        local PF DT NS KV RS EV PP
        if [ -f "$idir/meta.json" ]; then
            PF=$(meta_num "$idir/meta.json" prefill_ms)
            DT=$(meta_num "$idir/meta.json" decode_tps)
            NS=$(meta_num "$idir/meta.json" n_decode_steps)
            KV=$(meta_num "$idir/meta.json" peak_kv_cells)
            RS=$(meta_num "$idir/meta.json" peak_rss_kb)
            EV=$(meta_num "$idir/meta.json" evicted_total_decode)
            PP=$(meta_num "$idir/meta.json" perplexity)
        else
            PF=0; DT=0; NS=0; KV=0; RS=0; EV=0; PP=0
        fi
        : "${PF:=0}" "${DT:=0}" "${NS:=0}" "${KV:=0}" "${RS:=0}" "${EV:=0}" "${PP:=0}"

        local el=$(( $(date +%s) - T0 ))
        # niah_correct is filled in later by the host judge over gen.txt.
        echo "$s,$el,$rc,$PF,$DT,$NS,$KV,$RS,$EV,$PP,,$K_NOMINAL,$ddr0,$mfr0" >> "$cell_dir/stress.csv"
        log "  NIAH[$model_tag/$policy] stim=$sidx exit=$rc tps=$DT kv=$KV DDR=${ddr0}C free=${mfr0}GB"
        s=$(( s + 1 ))
    done
}

# Single cell = (model, policy, bench).
run_cell() {
    local model_tag="$1" model_path="$2" model_ctx="$3" policy="$4" bench="$5"
    local cell_dir="$OUT_DIR/$model_tag/$policy/$bench"
    mkdir -p "$cell_dir"

    # Skip cells whose stress.csv already has a data row, so a relaunch with
    # the same OUT_DIR resumes where the last run stopped.
    if [ -f "$cell_dir/stress.csv" ]; then
        local _nlines
        _nlines=$(wc -l < "$cell_dir/stress.csv" 2>/dev/null || echo 0)
        if [ "$_nlines" -ge 2 ] 2>/dev/null; then
            log "SKIP cell $model_tag/$policy/$bench (already done, stress.csv lines=$_nlines)"
            return 0
        fi
    fi

    log ""
    log "=== CELL START model=$model_tag policy=$policy bench=$bench ==="

    if [ ! -f "$model_path" ]; then
        log "  model missing: $model_path — skipping cell"
        return 0
    fi

    cool_phone "$model_tag/$policy/$bench"
    wait_for_memory "$model_tag/$policy/$bench"

    # Watchdog only for the policies in needs_watchdog.
    local started_wd=0
    if needs_watchdog "$policy"; then
        start_watchdog "$cell_dir" "$policy"
        started_wd=1
    fi

    local sampler_pid
    sampler_pid=$(start_sampler "$cell_dir/sensors.csv")
    sleep 2

    case "$bench" in
        ppl)  run_ppl_cell  "$model_tag" "$model_path" "$model_ctx" "$policy" "$cell_dir" ;;
        niah) run_niah_cell "$model_tag" "$model_path" "$model_ctx" "$policy" "$cell_dir" ;;
        *) log "  unknown bench: $bench" ;;
    esac

    stop_sampler "$sampler_pid"
    [ "$started_wd" = "1" ] && stop_watchdog

    log "=== CELL DONE  model=$model_tag policy=$policy bench=$bench ==="
}

# Preflight: catch a broken environment before running the whole sweep.
# Returns non-zero and logs the reason on failure.
preflight() {
    local fail=0

    log "preflight: BEGIN"

    # (a) DDR temperature sanity: 20..110 °C is the plausible operating band.
    local ddr
    ddr=$(ddr_temp_c)
    if ! [ "$ddr" -ge 20 ] 2>/dev/null || ! [ "$ddr" -le 110 ] 2>/dev/null; then
        log "preflight: FAIL ddr_temp_c=$ddr out of range [20,110] (zone=$DDR_ZONE)"
        fail=1
    else
        log "preflight: OK ddr_temp_c=${ddr}C"
    fi

    # (b) Skin temperature must be numeric. dumpsys can be missing on
    # stripped builds, so only warn.
    local sk
    sk=$(skin_dc)
    if [ -z "$sk" ] || ! [ "$sk" -ge 0 ] 2>/dev/null; then
        log "preflight: WARN skin_dc not numeric ('$sk') — dumpsys battery temperature missing?"
    else
        log "preflight: OK skin_dc=${sk}dC"
    fi

    # (c) PPL chunk files (8 for Tier-1).
    local i=0
    while [ "$i" -lt "$PPL_N_CHUNKS" ]; do
        local chunk="$PPL_TEXT_DIR/wiki.test.raw.chunk${i}"
        if [ ! -f "$chunk" ]; then
            log "preflight: FAIL missing PPL chunk: $chunk"
            fail=1
        fi
        i=$(( i + 1 ))
    done
    [ "$fail" = "0" ] && log "preflight: OK ${PPL_N_CHUNKS} PPL chunk files present in $PPL_TEXT_DIR"

    # (d) NIAH stimuli (8 for Tier-1, 32 for Tier-2).
    local s=0
    while [ "$s" -lt "$NIAH_N_STIMULI" ]; do
        local sidx=$(printf "%02d" "$s")
        local stim="$NIAH_PROMPT_DIR/niah_stimulus_${sidx}.txt"
        if [ ! -f "$stim" ]; then
            log "preflight: FAIL missing NIAH stimulus: $stim"
            fail=1
        fi
        s=$(( s + 1 ))
    done
    log "preflight: NIAH stimulus presence checked in $NIAH_PROMPT_DIR (${NIAH_N_STIMULI} expected)"

    # (e) No live smoke run: FA-off prefill of 2k tokens takes 60 to 200 s,
    # too variable for a preflight limit. Real cells fail fast instead.
    log "preflight: SKIP live smoke (use real-cell fail-fast instead)"
    [ "$fail" = "0" ] && log "preflight: OK"
    return $fail
}

unused_preflight_smoke() {
    : # the original smoke block below is preserved but not called
    # 180 s smoke run of eviction_bench --policy v1, expects exit=0.
    # Uses the first model found on disk.
    local smoke_model=""
    echo "$MODELS" | while IFS='|' read -r _MTAG _MPATH _MCTX; do
        _MTAG=$(echo "$_MTAG" | sed 's/^ *//; s/ *$//')
        _MPATH=$(echo "$_MPATH" | sed 's/^ *//; s/ *$//')
        [ -z "$_MTAG" ] && continue
        if [ -f "$_MPATH" ]; then
            echo "$_MPATH" > "$OUT_DIR/.preflight_smoke_model"
            break
        fi
    done
    [ -f "$OUT_DIR/.preflight_smoke_model" ] && smoke_model=$(cat "$OUT_DIR/.preflight_smoke_model")
    rm -f "$OUT_DIR/.preflight_smoke_model"

    if [ -z "$smoke_model" ]; then
        log "preflight: FAIL no model file found on disk for smoke run"
        fail=1
    elif [ ! -x "$BIN" ] && [ ! -f "$BIN" ]; then
        log "preflight: FAIL eviction_bench binary not found at $BIN"
        fail=1
    else
        local smoke_dir="$OUT_DIR/_preflight_smoke"
        mkdir -p "$smoke_dir"
        local smoke_chunk="$PPL_TEXT_DIR/wiki.test.raw.chunk0"
        if [ ! -f "$smoke_chunk" ]; then
            log "preflight: SKIP smoke run (chunk0 missing — already flagged above)"
            fail=1
        else
            log "preflight: smoke eviction_bench --policy v1 (180s timeout)..."
            # 64 tokens exercises prefill and decode. Wall time is bounded with
            # a background+wait loop because busybox may lack `timeout`.
            (
                LD_LIBRARY_PATH="$WORKDIR/bin_cpu" "$BIN" \
                    --model "$smoke_model" \
                    --eval-mode ppl --eval-text "$smoke_chunk" \
                    --prompt "$smoke_chunk" --prompt-id "preflight_smoke" \
                    --policy v1 --k-nominal "$K_NOMINAL" --anchor-top-k "$ANCHOR_TOP_K" \
                    --recent-budget "$RECENT_BUDGET" --n-sink "$N_SINK" \
                    --cache-type-k f16 --cache-type-v f16 \
                    --max-tokens 64 --ignore-eos \
                    --ctx-size "$CTX_SIZE" --seed "$SEED" \
                    --threads "$THREADS" --n-gpu-layers "$NGL" \
                    --n-batch "$N_BATCH" --ubatch-size "$UBATCH" \
                    --greedy \
                    --out-csv "$smoke_dir/steps.csv" --out-meta "$smoke_dir/meta.json" \
                    > "$smoke_dir/stdout.log" 2> "$smoke_dir/stderr.log"
                echo $? > "$smoke_dir/rc"
            ) &
            local smoke_pid=$!
            local waited=0
            while kill -0 "$smoke_pid" 2>/dev/null; do
                if [ "$waited" -ge 180 ]; then
                    log "preflight: FAIL smoke run exceeded 180s — killing"
                    kill -9 "$smoke_pid" 2>/dev/null
                    fail=1
                    break
                fi
                sleep 1
                waited=$(( waited + 1 ))
            done
            wait "$smoke_pid" 2>/dev/null
            local smoke_rc=1
            [ -f "$smoke_dir/rc" ] && smoke_rc=$(cat "$smoke_dir/rc")
            if [ "$smoke_rc" != "0" ]; then
                log "preflight: FAIL smoke eviction_bench exit=$smoke_rc (see $smoke_dir/stderr.log)"
                fail=1
            else
                log "preflight: OK smoke eviction_bench exit=0"
            fi
        fi
    fi

    if [ "$fail" != "0" ]; then
        log "preflight: FAILED — aborting before sweep"
        return 1
    fi
    log "preflight: PASSED"
    return 0
}

# Main sweep (runs inside the detached subshell)
main() {
    log "wave11 eval START -> $OUT_DIR"
    log "models   : Phi-3-mini-128k, Llama-3.2-1B, Gemma-2-2B (Q4_K_M)"
    log "policies : $POLICIES"
    log "benches  : $BENCHES   (NIAH tier=$NIAH_TIER → $NIAH_N_STIMULI stimuli)"
    log "K        : K_nominal=$K_NOMINAL anchor=$ANCHOR_TOP_K sink=$N_SINK recent=$RECENT_BUDGET"
    log "ctx=$CTX_SIZE threads=$THREADS ngl=$NGL n_batch=$N_BATCH ubatch=$UBATCH"

    # Do not start the sweep if preflight fails.
    if ! preflight; then
        log "wave11 eval ABORTED at preflight"
        exit 2
    fi

    # DVFS pin once for the whole sweep.
    log "pinning DVFS"
    su -c "sh $SCRIPTS/pin_dvfs.sh pin" 2>>"$PROG"

    # Outer loop is the bench, so all PPL cells finish before any NIAH cell and
    # a partial run still gives a full PPL panel. Inner loop is model then
    # policy, which keeps each model file in page cache across its policies.
    for BEN in $BENCHES; do
        log ""
        log "================================================================="
        log "BENCH $BEN"
        log "================================================================="
        echo "$MODELS" | while IFS='|' read -r MTAG MPATH MCTX; do
            # Skip empty lines from heredoc
            MTAG=$(echo "$MTAG" | sed 's/^ *//; s/ *$//')
            [ -z "$MTAG" ] && continue
            MPATH=$(echo "$MPATH" | sed 's/^ *//; s/ *$//')
            MCTX=$(echo "$MCTX"   | sed 's/^ *//; s/ *$//')

            log ""
            log "----- MODEL $MTAG  ($MPATH  ctx=$MCTX)  bench=$BEN -----"

            for POL in $POLICIES; do
                run_cell "$MTAG" "$MPATH" "$MCTX" "$POL" "$BEN"
            done
        done
    done

    log "restoring DVFS"
    su -c "sh $SCRIPTS/pin_dvfs.sh restore" 2>>"$PROG"

    log "=== WAVE11 COMPLETE ==="
    touch "$OUT_DIR/DONE"
}

# Detached entrypoint. The first call re-execs itself with setsid or nohup,
# writes the PID and returns. WAVE11_DETACHED=1 runs main directly and
# WAVE11_FOREGROUND=1 disables detaching.
if [ "${WAVE11_DETACHED:-0}" = "1" ] || [ "${WAVE11_FOREGROUND:-0}" = "1" ]; then
    main
    exit 0
fi

PID_FILE=${PID_FILE:-/sdcard/wave11_eval.pid}
DONE_FLAG="$OUT_DIR/DONE"
echo "[wave11] detaching — out=$OUT_DIR  pid_file=$PID_FILE"
# Use setsid when available, else plain nohup.
if command -v setsid >/dev/null 2>&1; then
    WAVE11_DETACHED=1 OUT_DIR="$OUT_DIR" \
        setsid sh "$0" </dev/null >/dev/null 2>&1 &
else
    WAVE11_DETACHED=1 OUT_DIR="$OUT_DIR" \
        nohup sh "$0" </dev/null >/dev/null 2>&1 &
fi
CHILD_PID=$!
echo "$CHILD_PID" > "$PID_FILE"
echo "[wave11] launched pid=$CHILD_PID    tail -F $PROG to monitor"
echo "[wave11] when $DONE_FLAG exists, pull $OUT_DIR to host for aggregation"
exit 0
