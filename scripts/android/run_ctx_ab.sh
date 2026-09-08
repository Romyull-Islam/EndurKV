#!/bin/bash
# ============================================================================
# ctx-invariance A/B (2026-07-28)
#
# WHY: the 336-cell NIAH table in the draft was swept at ctx=16384 for EVERY
# cell. The current muKV/SnapKV re-run uses ctx=4096 (L4K) and 8192 (L8K), so
# the new rows are not protocol-matched to the retained baseline rows.
#
# Rather than re-run all 112 cells at 16384 (~6 h without cooling), test whether
# ctx changes RETRIEVAL at all. llama1b is the cheapest model (30-126 s/cell), so
# 14 cells cost ~20 min. Compare hit-for-hit against the 14 llama1b muKV cells
# already banked at 4K/8K.
#   all 14 agree -> ctx is irrelevant to retrieval; the existing dataset stands
#                   with a footnote, and no re-run is needed.
#   any differ   -> pay for the full 16384 re-run, but only then.
#
# NO COOL GATE, deliberately. This measures HIT RATE only, which the NIAH harness
# header states is thermally invariant; energy and tps from these cells are NOT
# comparable and must not be reported. The cooled 4K/8K dataset remains the source
# for anything timing- or energy-related.
# ============================================================================
set -u
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue   # never poke a dead port (spawns a squatting adb server)
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] using server port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

CTX=${CTX:-16384}
MODEL_TAG=${MODEL_TAG:-llama1b}
declare -A MODELS=(
  [llama1b]=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  [phi3]=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
  [gemma2b]=/data/local/tmp/endurkv/models/gemma-2-2b-it-Q4_K_M.gguf
)
CB=/data/local/tmp/endurkv/bin_cpu_v87
OUT_HOST=/tmp/ctx_ab_${MODEL_TAG}_${CTX}; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/ctxab_$(date +%Y%m%d_%H%M%S)
SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/niah
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

adb_safe_shell "mkdir -p $OUT" < /dev/null
for f in "$SRC"/niah_L*_n0.txt; do adb push "$f" "$OUT/$(basename "$f")" < /dev/null >/dev/null 2>&1; done

echo "=== $MODEL_TAG muKV at ctx=$CTX, no cool gate, 14 stimuli ==="
for STIM in $(cd "$SRC" && ls niah_L*_n0.txt); do
  id="${MODEL_TAG}__mukv__${STIM%.txt}"; PD=$OUT/$id
  [ -f "$OUT_HOST/$id/meta.json" ] && continue
  adb_safe_shell "mkdir -p $PD" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$CB timeout 2400 $CB/eviction_bench --prompt $OUT/$STIM --prompt-id $id \
    --eval-mode gen --max-tokens 64 --ignore-eos --ctx-size $CTX --model ${MODELS[$MODEL_TAG]} --seed 42 \
    --threads 6 --n-gpu-layers 0 --greedy --k-nominal 1024 $MU \
    --out-meta $PD/meta.json --out-csv /dev/null --out-gen $PD/gen.txt > $PD/c.out 2> $PD/c.err" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  h=$(grep -qiE 'mango sorbet|bi-rite' "$OUT_HOST/$id/gen.txt" 2>/dev/null && echo HIT || echo miss)
  echo "  ${STIM%.txt}: $h"
done
echo "=== done -> $OUT_HOST ==="
