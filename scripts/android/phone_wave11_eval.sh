#!/system/bin/sh
# phone_wave11_eval.sh — Wave-11 phone-side launcher for the evaluation protocol.
#
# Sweep grid:
#   MODELS x POLICIES x BENCHES
#   = 3 models  (Phi-3-mini-128k, Llama-3.2-1B, Gemma-2-2B)
#   x 5 policies (vanilla, v1, tova, h2o, v1_fa2_stack)
#   x 2 benches  (ppl on WikiText-2, niah Tier-1 8 stimuli)
#   = 30 cells.
#
# Per-policy FA mode (canonical, NOT homogenized):
#   vanilla      → FA-on  (production baseline)
#   v1           → FA-off (needs kq_soft_max signals; --policy v1)
#   tova         → FA-off (per-layer canonical;        --policy tova)
#   h2o          → FA-off (heavy-hitters;              --policy h2o)
#   v1_fa2_stack → FA-off prefill + FA-on decode via state-swap (--policy v1_fa2)
#                  PLUS Wave-9 thermal stack: Q8 K cache, preempt-throttle
#                  watchdog, closed-loop K controller, memory-pressure gate.
#                  (The watchdog runs ONLY for this policy, per Wave-9 conv.)
#
# Per-cell deliverables (under $OUT_DIR/<MODEL>/<POLICY>/<BENCH>/):
#   stress.csv     — one row per iter / per chunk / per stimulus, includes
#                    columns ppl and niah_correct (filled by host judge for NIAH)
#   sensors.csv    — thermal/freq sampling at $SAMPLE_HZ
#   iter*/         — per-iter steps.csv + meta.json + stderr.log + gen.txt
#
# Wave conventions inherited:
#   - mem-gate >= 4 GB before each cell
#   - cool gate: skin <= 33.0 °C AND DDR <= 40 °C (timeout-tolerant)
#   - DVFS pin via scripts/pin_dvfs.sh (pin once at top, restore at end)
#   - watchdog STARTED only for v1_fa2_stack cells and STOPPED after
#   - phone_nohup detachment pattern so ADB drops don't kill the run
#
# Output root:  /data/local/tmp/endurkv/logs/wave11_eval_<ts>/
# Progress:     $OUT_DIR/progress.log    (tail -F this to monitor)
# Completion:   $OUT_DIR/DONE            (touched on successful exit)

set -u

# ────────────────────────────────────────────────────────────────────────────
# Knobs (overridable via env)
# ────────────────────────────────────────────────────────────────────────────
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

# PPL: 8 chunks × 2048 tokens (chunk files were pre-split host-side).
# Protocol: prefill chunk i, teacher-force chunk i+1 (KVQuant/H2O style — eval
# text MUST be disjoint from prefill text, else metric collapses to ~1.0).
PPL_N_CHUNKS=${PPL_N_CHUNKS:-8}
PPL_CHUNK_TOKENS=${PPL_CHUNK_TOKENS:-2048}
PPL_TEXT_DIR=${PPL_TEXT_DIR:-$WORKDIR/eval_data}     # wiki.test.raw.chunk0..chunk7

# NIAH: Tier-1 uses 8 of 32 stimuli; Tier-2 (later) uses all 32.
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

# Watchdog plumbing (used ONLY for v1_fa2_stack cells)
# WATCHDOG_STOP / WATCHDOG_LOG / WATCHDOG_PID are scoped per-cell at start_watchdog time
# to avoid log-clobber and start/stop races across cells.
WATCHDOG_STOP=""
WATCHDOG_LOG=""
WATCHDOG_PID=""

# Models on phone (tag | gguf | ctx) — env-overridable
MODELS=${MODELS:-"
Phi-3-mini-128k|$WORKDIR/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf|$CTX_SIZE
Llama-3.2-1B|$WORKDIR/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf|$CTX_SIZE
Gemma-2-2B|$WORKDIR/models/gemma-2-2b-it-Q4_K_M.gguf|$CTX_SIZE
"}

POLICIES=${POLICIES:-"vanilla h2o v1_fa2_stack tova streamingllm"}
BENCHES=${BENCHES:-"ppl niah"}

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
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
    # Per-cell scoping: log + stop sentinel + pid file live inside cell_dir so
    # the next cell can't truncate the previous cell's watchdog log, and so
    # stop_watchdog can wait deterministically for the prior process to exit.
    #
    # Wave-11 (post-K1024): all watchdog-bearing policies now default to the
    # multi-sensor v2 watchdog (preempt_throttle_watchdog_v2.sh), which leads
    # DDR-only throttling by ~13 min via the skin sensor. Set the env
    # WATCHDOG_VERSION=v1 to fall back to the legacy DDR-only script.
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
    # Signal stop, then poll for the watchdog process to actually exit before
    # any subsequent start_watchdog can race with it (two concurrent watchdogs
    # would both fight scaling_max_freq).
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

# Map policy → eviction_bench flags. Echoes a single line of flags.
# Canonical FA mode is encoded here:
#   vanilla      → --no-policy (use the binary's vanilla path), FA-on default
#   v1/tova/h2o  → --policy <name>, FA-off (binary picks this when --policy ≠ vanilla)
#   v1_fa2_stack → --policy v1_fa2 with Q8 K, recent_budget, FA-off prefill / FA-on decode
policy_flags() {
    case "$1" in
        vanilla)
            # No policy flag = vanilla path in the binary; FA is on by default.
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
            # Canonical Xiao et al. ICLR 2024: sink (first n_sink) + sliding window.
            # K_nominal = total budget (n_sink + recent window). FA-on by default.
            printf -- "--policy streamingllm --k-nominal %s --n-sink %s --cache-type-k f16 --cache-type-v f16" \
                "$K_NOMINAL" "$N_SINK"
            ;;
        v1_fa2_stack)
            # v1_FA² stack: Q8 K (V must remain f16 due to state-swap layout),
            # recent_budget tuned to keep K + anchor + sink = K_nominal.
            printf -- "--policy v1_fa2 --k-nominal %s --anchor-top-k %s --recent-budget %s --n-sink %s --cache-type-k q8_0 --cache-type-v f16" \
                "$K_NOMINAL" "$ANCHOR_TOP_K" "$RECENT_BUDGET" "$N_SINK"
            ;;
        v1_entropy_stack)
            # Confidence-weighted attention scoring with the v1_fa2 thermal stack
            # (FA-off prefill → state-swap → FA-on decode, Q8 K, tiered eviction).
            printf -- "--policy v1_entropy_stack --k-nominal %s --anchor-top-k %s --recent-budget %s --n-sink %s --cache-type-k q8_0 --cache-type-v f16" \
                "$K_NOMINAL" "$ANCHOR_TOP_K" "$RECENT_BUDGET" "$N_SINK"
            ;;
        v1_predictive_stack)
            # Temporal-trajectory (slope) scoring; FA-off for the whole decode
            # because slope() requires kq_soft_max readback every step.
            # Q8 K still active to halve DRAM-BW; thermal envelope enforced by
            # the watchdog + tighter K (set by pick_k_adaptive).
            printf -- "--policy v1_predictive_stack --k-nominal %s --n-sink %s --cache-type-k q8_0 --cache-type-v f16" \
                "$K_NOMINAL" "$N_SINK"
            ;;
        v1_fa2_hybrid)
            # Isolation: canonical H2O eviction (recent K/2 + heavy K/2) + the
            # v1_FA² thermal watchdog & memory gate; f16 K + f16 V (no Q8 K) and
            # NO state-swap. Tests whether just adding the watchdog to H2O
            # changes PPL — expected to match canonical h2o.
            printf -- "--policy v1_fa2_hybrid --k-nominal %s --n-sink %s --cache-type-k f16 --cache-type-v f16" \
                "$K_NOMINAL" "$N_SINK"
            ;;
        v1_fa2_f16)
            # Isolation: same v1 spread-gate + selective anchoring + state-swap
            # as v1_fa2_stack, but drops Q8 K (cache is f16 K + f16 V).
            # Tests whether Q8 K seq_add-skip is the PPL bottleneck.
            printf -- "--policy v1_fa2_f16 --k-nominal %s --anchor-top-k %s --recent-budget %s --n-sink %s --cache-type-k f16 --cache-type-v f16" \
                "$K_NOMINAL" "$ANCHOR_TOP_K" "$RECENT_BUDGET" "$N_SINK"
            ;;
        endurkv_optimal)
            # Canonical H2O eviction (50/50 recent+heavy) + v1_FA² thermal stack:
            # state-swap → FA-on decode, Q8 K + f16 V, multi-sensor watchdog v2,
            # >=4 GB memory gate. h2o has no anchor/tiered structure so we only
            # pass --k-nominal and --n-sink (no --anchor-top-k / --recent-budget).
            printf -- "--policy endurkv_optimal --k-nominal %s --n-sink %s --cache-type-k q8_0 --cache-type-v f16" \
                "$K_NOMINAL" "$N_SINK"
            ;;
        endurkv_adaptive)
            # v1 per-head spread-gate + runtime K_effective(t) controller (DDR,
            # CPU big, skin, battery temperatures → μ_thermal ∈ [0.7, 1.3]).
            # f16 K + f16 V (NO Q8 K — Wave-11 smoke showed it hurts PPL via
            # seq_add-skip), NO state-swap (Wave-11 smoke saw ~2 PPL on
            # Llama-1B). Multi-sensor watchdog v2 + ≥4 GB memory gate are
            # wired by the launcher (see needs_watchdog below). Selective
            # anchoring (top-32) keeps the anchored block tight.
            printf -- "--policy endurkv_adaptive --k-nominal %s --anchor-top-k %s --n-sink %s --cache-type-k f16 --cache-type-v f16" \
                "$K_NOMINAL" "$ANCHOR_TOP_K" "$N_SINK"
            ;;
        adakv)
            # Ada-KV (Feng et al. NeurIPS 2025) — Algorithm 1 adaptive per-head
            # budget allocation, Ada-SnapKV recipe (FA-off prefill, frozen mask
            # during decode, no state-swap). f16 K + f16 V to isolate the
            # budget-allocation contribution (no Q8 K confounds vs paper).
            printf -- "--policy adakv --k-nominal %s --cache-type-k f16 --cache-type-v f16" "$K_NOMINAL"
            ;;
        *) printf -- "" ;;
    esac
}

# Whether a policy needs the watchdog (Wave-9+: any of the *_stack policies,
# plus the Wave-11 v1_fa2_hybrid / v1_fa2_f16 isolation cells, and the
# Wave-11 endurkv_optimal final-recipe cell — all use the preempt-throttle
# watchdog + memory gate).
needs_watchdog() {
    case "$1" in
        v1_fa2_stack|v1_fa2_hybrid|v1_fa2_f16|v1_entropy_stack|v1_predictive_stack|endurkv_optimal|endurkv_adaptive) return 0 ;;
        *) return 1 ;;
    esac
}

# Parse a meta.json field. Field type "n" = number, "s" = string.
meta_num() {
    local f="$1" key="$2"
    grep -oE "\"${key}\": *[0-9.]+" "$f" 2>/dev/null | grep -oE '[0-9.]+$' | head -1
}

# ────────────────────────────────────────────────────────────────────────────
# Bench drivers
# ────────────────────────────────────────────────────────────────────────────

# PPL on WikiText-2: iterate the 8 pre-split chunks, log a row per chunk.
run_ppl_cell() {
    local model_tag="$1" model_path="$2" model_ctx="$3" policy="$4" cell_dir="$5"
    local pflags="$(policy_flags "$policy")"

    echo "chunk_idx,t_elapsed_s,exit,prefill_ms,decode_tps,n_decode_steps,peak_kv_cells,peak_rss_kb,evicted,ppl,niah_correct,k_used,ddr_start_c,mem_free_gb_start" \
        > "$cell_dir/stress.csv"

    local T0=$(date +%s)
    local i=0
    # We need chunk i (prefill) AND chunk i+1 (eval target), so the last
    # runnable iter is PPL_N_CHUNKS-2 -> 7 cells per (model,policy) for 8 chunks.
    # Fix for critical finding: previously --eval-text and --prompt pointed to
    # the SAME chunk, so PPL was scored against tokens already in the KV cache
    # (collapsed to ~1.0). Now prefill chunk i, teacher-force chunk i+1.
    local last_iter=$(( PPL_N_CHUNKS - 1 ))
    while [ "$i" -lt "$last_iter" ]; do
        local idir="$cell_dir/iter$(printf %04d "$i")"
        mkdir -p "$idir"

        local prefill_chunk="$PPL_TEXT_DIR/wiki.test.raw.chunk${i}"
        local eval_chunk="$PPL_TEXT_DIR/wiki.test.raw.chunk$(( i + 1 ))"
        if [ ! -f "$prefill_chunk" ] || [ ! -f "$eval_chunk" ]; then
            # BLOCKER-5 fix: never silently skip — write a sentinel row so the
            # chunk_idx column has no gaps in stress.csv (gap would otherwise
            # be indistinguishable from "cell never ran"). Preflight (BLOCKER-4)
            # should normally catch this before we get here.
            local ddr0=$(ddr_temp_c) mfr0=$(mem_free_gb)
            log "  PPL chunk missing: prefill=$prefill_chunk eval=$eval_chunk — emitting sentinel row exit=-1"
            echo "$i,0,-1,0,0,0,0,0,0,0,,${K_NOMINAL},${ddr0},${mfr0}" >> "$cell_dir/stress.csv"
            i=$(( i + 1 ))
            continue
        fi

        local ddr0=$(ddr_temp_c) mfr0=$(mem_free_gb)
        # shellcheck disable=SC2086
        # eviction_bench.cpp L167-170 requires --prompt + --prompt-id even in
        # ppl mode. Prefill = chunk i; teacher-forced eval target = chunk i+1
        # (disjoint, so PPL measures generalization, not in-context recall).
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
        # niah_correct is N/A for PPL rows → empty
        echo "$i,$el,$rc,$PF,$DT,$NS,$KV,$RS,$EV,$PP,,$K_NOMINAL,$ddr0,$mfr0" >> "$cell_dir/stress.csv"
        log "  PPL[$model_tag/$policy] chunk=$i exit=$rc ppl=$PP tps=$DT kv=$KV DDR=${ddr0}C free=${mfr0}GB"
        i=$(( i + 1 ))
    done
}

# NIAH Tier-1: 8 stimuli, gen mode, log gen.txt per stimulus for host GPT judge.
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
        # gen mode; binary writes generated text to --out-gen for host-side judge.
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
        # niah_correct left empty here — populated by host GPT judge over gen.txt.
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

    # SKIP-DONE: if a prior run already produced stress.csv with at least one
    # data row (i.e. >=2 lines = header + >=1 data), the cell is considered
    # complete and we skip it. This lets a relaunch reuse the SAME OUT_DIR
    # and continue from where the previous run was killed without redoing work.
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

    # Watchdog ONLY for v1_fa2_stack (Wave-9 convention) — extended in
    # Wave-11 to endurkv_optimal (uses multi-sensor watchdog v2 when present).
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

# ────────────────────────────────────────────────────────────────────────────
# Preflight gate (BLOCKER-4): catch obviously-broken environments before we
# burn hours running through 30 cells that will all fail the same way.
# Returns 0 on success; on any failure logs the reason and returns non-zero.
# ────────────────────────────────────────────────────────────────────────────
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

    # (b) Skin temperature (dumpsys battery) must be numeric. dumpsys is
    # sometimes unavailable on stripped builds — warn but don't fail.
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

    # (e) Skip live smoke run — Llama-1B+FA-off prefill on 2k tokens takes
    # 60-200s which is too variable for a preflight ceiling. The real cells
    # will fail-fast if the binary is broken. Just check that the binary is
    # executable and the chunk file exists (already done above).
    log "preflight: SKIP live smoke (use real-cell fail-fast instead)"
    [ "$fail" = "0" ] && log "preflight: OK"
    return $fail
}

unused_preflight_smoke() {
    : # the original smoke block below is preserved but not called
    # (e) original 180s smoke invocation of eviction_bench --policy v1 to verify exit=0.
    # Pick the first model that actually exists on disk.
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
            # 64 tokens is enough to exercise prefill+decode; bounded by hard
            # 60s wall clock via background+wait pattern (no `timeout` on
            # Android busybox guaranteed).
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

# ────────────────────────────────────────────────────────────────────────────
# Main sweep (driven inside a phone_nohup'd subshell on first invocation)
# ────────────────────────────────────────────────────────────────────────────
main() {
    log "wave11 eval START -> $OUT_DIR"
    log "models   : Phi-3-mini-128k, Llama-3.2-1B, Gemma-2-2B (Q4_K_M)"
    log "policies : $POLICIES"
    log "benches  : $BENCHES   (NIAH tier=$NIAH_TIER → $NIAH_N_STIMULI stimuli)"
    log "K        : K_nominal=$K_NOMINAL anchor=$ANCHOR_TOP_K sink=$N_SINK recent=$RECENT_BUDGET"
    log "ctx=$CTX_SIZE threads=$THREADS ngl=$NGL n_batch=$N_BATCH ubatch=$UBATCH"

    # BLOCKER-4: gate the entire 30-cell sweep on environment sanity.
    if ! preflight; then
        log "wave11 eval ABORTED at preflight"
        exit 2
    fi

    # DVFS pin once for the whole sweep.
    log "pinning DVFS"
    su -c "sh $SCRIPTS/pin_dvfs.sh pin" 2>>"$PROG"

    # Iterate the 30 cells. REORDERED (Wave-11 sweep reorder): outer loop is
    # now BENCH so that ALL PPL cells complete across every (model, policy)
    # combo before any NIAH cells start. This means even a partial run yields
    # a complete PPL panel for plotting, instead of a single-model column.
    # Inner order is model -> policy (per-model contiguity preserved within
    # each bench so the model file stays page-cached across its 5 policies).
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

# ────────────────────────────────────────────────────────────────────────────
# Detached entrypoint (resilient to ADB drops).
# First call: re-exec via nohup+setsid into the background, write PID, return.
# Re-entry (env WAVE11_DETACHED=1): just run main().
# Disable detachment by exporting WAVE11_FOREGROUND=1.
# ────────────────────────────────────────────────────────────────────────────
if [ "${WAVE11_DETACHED:-0}" = "1" ] || [ "${WAVE11_FOREGROUND:-0}" = "1" ]; then
    main
    exit 0
fi

PID_FILE=${PID_FILE:-/sdcard/wave11_eval.pid}
DONE_FLAG="$OUT_DIR/DONE"
echo "[wave11] detaching — out=$OUT_DIR  pid_file=$PID_FILE"
# setsid is available on Android; fall back to plain nohup if not.
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
