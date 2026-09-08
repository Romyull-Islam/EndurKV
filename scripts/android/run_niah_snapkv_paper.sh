#!/bin/bash
# ============================================================================
# run_niah_snapkv_paper.sh -- NIAH SnapKV at its OWN published NIAH setting
#                             (2026-08-02)
#
# WHY. The 392-cell NIAH sweep ran SnapKV at window 64 / kernel 5. That was not a
# choice: run_niah_tableC.sh passed --obs-window 64, but on the v87 build the flag
# was INERT (window/kernel were hardcoded at the FasterDecoding default 64/5 and
# --obs-window was never forwarded to policy_snapkv). The rows are therefore valid
# canonical-repo-default SnapKV -- but only by coincidence, since the passed value
# happened to equal the hardcoded one.
#
# SnapKV's paper retunes per benchmark and uses window 16 / kernel 5 for NIAH.
# Our paper must not label three different configurations "SnapKV" across three
# tables, so the rule is: give the baseline its OWN published setting for the
# benchmark being run (NIAH 16/5, LongBench 32/7), and fall back to the repo
# default only where the baseline publishes none (WikiText). This re-runs the
# NIAH SnapKV block at 16/5 on the v88 build, where the flags actually take
# effect and meta.json records obs_window/snapkv_kernel so the configuration is
# auditable from the artifact rather than inferred from this script.
#
# Only the SnapKV block is re-run: 4 models x 14 stimuli = 56 cells. Every other
# policy in Table C is untouched and stays on its existing cells -- same build
# family, same ctx, same protocol, same cool gate.
# ============================================================================
set -u
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

KBUD=${KBUD:-1024}
CTX=${CTX:-16384}
OUT_HOST=/tmp/niah_snapkv_paper_k${KBUD}_ctx${CTX}; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/niah_snapkv_$(date +%Y%m%d_%H%M%S)
NIAH_SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/niah
CB=/data/local/tmp/endurkv/bin_cpu_v88     # v88: --snapkv-kernel actually works
adb_safe_shell "mkdir -p $OUT" < /dev/null
for f in "$NIAH_SRC"/niah_L*_n0.txt; do adb push "$f" "$OUT/$(basename "$f")" < /dev/null >/dev/null 2>&1; done
STIMS=$(cd "$NIAH_SRC" && ls niah_L*_n0.txt)

declare -A MODELS=(
  [phi3]=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
  [llama1b]=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  [gemma2b]=/data/local/tmp/endurkv/models/gemma-2-2b-it-Q4_K_M.gguf
  [bonsai8b]=/data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf )

cell(){ local MT=$1 STIM=$2; local id="${MT}__snapkv_paper__${STIM%.txt}"; local PD=$OUT/$id
  [ -f "$OUT_HOST/$id/meta.json" ] && return
  adb_safe_shell "mkdir -p $PD" < /dev/null
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  echo "$CG" | tail -1
  case "$CG" in *"cool ddr="*) : ;; *) echo "[SKIP-HOT] $id"; return ;; esac
  # no watchdog: it is muKV-only by design
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$CB timeout ${TMO:-3600} $CB/eviction_bench --prompt $OUT/$STIM --prompt-id $id \
    --eval-mode gen --max-tokens 64 --ignore-eos --ctx-size $CTX --model ${MODELS[$MT]} --seed 42 \
    --threads 6 --n-gpu-layers 0 --greedy --k-nominal $KBUD \
    --policy snapkv --obs-window 16 --snapkv-kernel 5 --n-sink 0 \
    --out-meta $PD/meta.json --out-csv /dev/null --out-gen $PD/gen.txt > $PD/c.out 2> $PD/c.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  local h=miss; grep -qiE 'mango sorbet|bi-rite' "$OUT_HOST/$id/gen.txt" 2>/dev/null && h=HIT
  echo "  [$id] $h"
}

for MT in phi3 llama1b gemma2b bonsai8b; do
  i=0
  for STIM in $STIMS; do
    i=$((i+1)); echo "[$(date +%H:%M:%S)] $MT ($i/14) $STIM"
    cell "$MT" "$STIM"
  done
done
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null
echo "[$(date +%H:%M:%S)] NIAH_SNAPKV_PAPER_DONE -> $OUT_HOST"
