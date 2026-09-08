#!/bin/bash
# ============================================================================
# Table C — self-contained phone NIAH table (2026-07-28)
#
# WHY THIS EXISTS. The draft's 336-cell NIAH table (tab:phone-niah-prelim) was
# swept on 2026-06-30. Two independent confounds make new muKV rows unmixable
# with it:
#   (1) ctx: that sweep allocated 16384 for EVERY cell; the 07-26 re-run used
#       4096/8192. An A/B (14 llama1b cells) showed HIT is ctx-invariant, but
#       TPS / Wall / RSS / DDR / Energy are NOT.
#   (2) build: the armv8.7-a binary (i8mm + dotprod) landed 2026-07-17, AFTER
#       that sweep. Same policy/model/stimulus measures 19.97 tps on the old
#       build vs 39.37 on the new one -- a 2.0x swing from compiler flags alone.
# Publishing new-build muKV next to old-build baselines would credit the build
# to the policy. So this script produces ONE table where every row shares the
# same build, ctx, protocol and cool gate.
#
# SEVEN POLICIES, EACH IN ITS OWN CANONICAL FORM (2026-07-28: expanded from 3 to 7
# so the table is internally consistent -- no row borrows a number from the old build):
#   adakv    -- Ada-SnapKV defaults: window 32, maxpool-7, alpha=0.2 safeguard, no sinks
#   h2o      -- 50/50 heavy-hitter + recent split, no sinks
#   tova     -- paper-preferred per-layer form (Oren et al.), scoring from the last query
#   strllm   -- 4 attention sinks + recent window
#   vanilla  -- full cache, no eviction, NO watchdog.
#   snapkv   -- canonical SnapKV (Li et al. 2024) exactly as published:
#               --policy snapkv selects per-head top-K over avgpool-5 pooled
#               observation-window attention with a uniform budget and freezes
#               the mask during decode (eviction_bench.cpp policy_snapkv()).
#               window 64, NO sinks, f16 K/V, and NO watchdog. muKV's
#               --snapkv-pool (maxpool-7) is OURS and is deliberately NOT passed.
#   mukv     -- frozen muKV config + its watchdog. The watchdog is muKV-only by
#               design; giving it to a baseline would be the thermal equivalent
#               of running the baseline with our eviction.
#
# ctx=16384 for every cell (matches the draft's protocol).
# Cool gate stays ON: this table reports TPS / Wall / Energy / DDR / CPU, all of
# which are contaminated by a hot start. Gate = DDR<=35, batt<=33, charging off.
# Fresh OUT_HOST so the 112-cell 4K/8K dataset is left untouched.
# ============================================================================
set -u
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue   # dead port + adb = squatting server that breaks ssh -R
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] using server port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

KBUD=${KBUD:-1024}
CTX=${CTX:-16384}
OUT_HOST=/tmp/niah_tableC_k${KBUD}_ctx${CTX}; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/tableC_$(date +%Y%m%d_%H%M%S)
NIAH_SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/niah
CB=/data/local/tmp/endurkv/bin_cpu_v87        # armv8.7-a build (i8mm + dotprod)
adb_safe_shell "mkdir -p $OUT" < /dev/null
for f in "$NIAH_SRC"/niah_L*_n0.txt; do adb push "$f" "$OUT/$(basename "$f")" < /dev/null >/dev/null 2>&1; done
STIMS=$(cd "$NIAH_SRC" && ls niah_L*_n0.txt)

declare -A MODELS=(
  [phi3]=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
  [llama1b]=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  [gemma2b]=/data/local/tmp/endurkv/models/gemma-2-2b-it-Q4_K_M.gguf
  [bonsai8b]=/data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf
)
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

cell(){ local MT=$1 POL=$2 STIM=$3; shift 3; local id="${MT}__${POL}__${STIM%.txt}"; local PD=$OUT/$id
  [ -f "$OUT_HOST/$id/meta.json" ] && return                     # resume
  adb_safe_shell "mkdir -p $PD" < /dev/null
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  echo "$CG" | tail -1
  case "$CG" in *"cool ddr="*) : ;;
    *) echo "[SKIP-HOT] $id -- gate failed; cell not run"; return ;; esac
  # watchdog is muKV-ONLY (thermal-aware inference is muKV's mechanism, not a baseline's)
  if [ "$POL" = mukv ]; then
    adb_safe_shell "su -c 'rm -f /data/local/tmp/tc_wd.stop; nohup sh /data/local/tmp/preempt_throttle_watchdog_v2.sh /data/local/tmp/tc_wd_$id.log /data/local/tmp/tc_wd.stop >/dev/null 2>&1 &'" < /dev/null
  fi
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$CB timeout ${TMO:-3600} $CB/eviction_bench --prompt $OUT/$STIM --prompt-id $id \
    --eval-mode gen --max-tokens 64 --ignore-eos --ctx-size $CTX --model ${MODELS[$MT]} --seed 42 \
    --threads 6 --n-gpu-layers 0 --greedy --k-nominal $KBUD $* \
    --out-meta $PD/meta.json --out-csv /dev/null --out-gen $PD/gen.txt > $PD/c.out 2> $PD/c.err" < /dev/null
  adb_safe_shell "su -c 'touch /data/local/tmp/tc_wd.stop; pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  local h=miss; grep -qiE 'mango sorbet|bi-rite' "$OUT_HOST/$id/gen.txt" 2>/dev/null && h=HIT
  echo "  [$id] $h"
}

# bonsai8b LAST so it can be cut without losing the 3 models the draft table covers
for MT in phi3 llama1b gemma2b bonsai8b; do
  i=0
  for STIM in $STIMS; do
    i=$((i+1)); echo "[$(date +%H:%M:%S)] $MT ($i/14) $STIM ctx=$CTX"
    # ALL SEVEN policies on the SAME build/ctx/protocol -- full internal consistency.
    # Each baseline in its own published configuration; none gets muKV mechanisms and
    # none gets the watchdog. Dropped vs the draft's 8: muKV-count (superseded by mass)
    # and TOVA-canonical (the draft's own protocol calls the strict no-recency form a
    # chunked-prefill measurement artifact it "does not report"). Added: SnapKV.
    cell "$MT" vanilla      "$STIM" --policy vanilla
    cell "$MT" mukv         "$STIM" $MU
    cell "$MT" snapkv       "$STIM" --policy snapkv --obs-window 64 --n-sink 0
    cell "$MT" adakv        "$STIM" --policy adakv --n-sink 0 --obs-window 32
    cell "$MT" h2o          "$STIM" --policy h2o --n-sink 0 --obs-window 64
    cell "$MT" tova         "$STIM" --policy tova
    cell "$MT" streamingllm "$STIM" --policy streamingllm --n-sink 4
  done
  echo "[$(date +%H:%M:%S)] === $MT block done ==="
done
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null
echo "[$(date +%H:%M:%S)] TABLE C DONE -> $OUT_HOST"
