#!/bin/bash
# Phone NIAH table: 7 policies x 4 models on one build (armv8.7-a), one ctx and one
# cool-gated protocol, so no row mixes builds or ctx sizes.
# Each baseline runs in its published form with no watchdog. SnapKV uses its own
# avgpool (muKV's --snapkv-pool is not passed). Only muKV runs with its watchdog.
# Cool gate is on because the table reports TPS, wall time, energy and temperatures.
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
  # watchdog is part of muKV, baselines run without it
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

# bonsai8b last so it can be cut without losing the other three models
for MT in phi3 llama1b gemma2b bonsai8b; do
  i=0
  for STIM in $STIMS; do
    i=$((i+1)); echo "[$(date +%H:%M:%S)] $MT ($i/14) $STIM ctx=$CTX"
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
