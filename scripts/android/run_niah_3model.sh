#!/bin/bash
# NIAH quality table on the phone CPU: models x policies x 14 needle stimuli
# (4K/8K x 7 depths), K from $1 (default 1024). Hit rate does not depend on heat
# but energy and tps do, so every cell waits on the cool gate first.
# muKV runs with its watchdog (preempt_throttle_watchdog_v2), SnapKV without.
# Resumable: cells with a meta.json are skipped.
#   hit = gen contains "mango sorbet" or "bi-rite" (case-insensitive)
#   retained_kv_bytes = compacted cache size (llama_state_seq_get_size)
set -u
# Use whichever adb server holds the device. The phone can re-enumerate onto
# another server, so a fixed port can silently lose it.
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  [ -n "$_p" ] || continue
  # Skip ports with no listener. `adb devices` would start a server there and
  # take the port from the ssh -R tunnel (see adb_resilient.sh).
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then
    export ANDROID_ADB_SERVER_PORT=$_p; break
  fi
done
echo "[adb] using server port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
KBUD=${1:-1024}
OUT_HOST=/tmp/niah_3model_k${KBUD}; mkdir -p "$OUT_HOST"
TS=$(date +%Y%m%d_%H%M%S); OUT=/data/local/tmp/endurkv/logs/niah3m_$TS
adb_safe_shell "mkdir -p $OUT" < /dev/null
NIAH_SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/niah
CB=/data/local/tmp/endurkv/bin_cpu_v87        # NEW armv8.7-a build
for f in "$NIAH_SRC"/niah_L*_n0.txt; do adb push "$f" "$OUT/$(basename "$f")" < /dev/null >/dev/null 2>&1; done
STIMS=$(cd "$NIAH_SRC" && ls niah_L*_n0.txt)

# model tag -> device path
declare -A MODELS=(
  [phi3]=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
  [llama1b]=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  [gemma2b]=/data/local/tmp/endurkv/models/gemma-2-2b-it-Q4_K_M.gguf
  [bonsai8b]=/data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf   # 1-bit qwen3 (Prism); muKV-on-Bonsai gated on new-build validation
)

# cell <model-tag> <policy-tag> <stim> <ctx> <policy-args...>
cell(){ local MT=$1 POL=$2 STIM=$3 CTX=$4; shift 4; local id="${MT}__${POL}__${STIM%.txt}"; local PD=$OUT/$id
  if [ -f "$OUT_HOST/$id/meta.json" ]; then return; fi     # resume: skip done
  adb_safe_shell "mkdir -p $PD" < /dev/null
  # No cool, no cell. Without a meta.json the cell is retried on a later pass
  # instead of keeping a hot-start measurement.
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  echo "$CG" | tail -1
  case "$CG" in
    *"cool ddr="*) : ;;
    *) echo "[SKIP-HOT] $id -- cool gate did not reach DDR<=35 / batt<=33; cell not run"; return ;;
  esac
  if [ "$POL" = mukv ]; then adb_safe_shell "su -c 'rm -f /data/local/tmp/niah_wd.stop; nohup sh /data/local/tmp/preempt_throttle_watchdog_v2.sh /data/local/tmp/niah_wd_$id.log /data/local/tmp/niah_wd.stop >/dev/null 2>&1 &'" < /dev/null; fi
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$CB timeout ${TMO:-2400} $CB/eviction_bench --prompt $OUT/$STIM --prompt-id $id --eval-mode gen \
    --max-tokens 64 --ignore-eos --ctx-size $CTX --model ${MODELS[$MT]} --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --k-nominal $KBUD $* --out-meta $PD/meta.json --out-csv /dev/null --out-gen $PD/gen.txt > $PD/c.out 2> $PD/c.err" < /dev/null
  adb_safe_shell "su -c 'touch /data/local/tmp/niah_wd.stop; pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
}
MU="--policy v1_fa2 --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --fa-on-evict"

run_policies(){ local MT=$1 STIM=$2 CTX=$3
  # vanilla is the ceiling, snapkv/adakv/h2o/tova are per-head baselines (tova is
  # the per-layer TOVA variant). Commented-out cells reuse existing data.
  # cell "$MT" vanilla      "$STIM" "$CTX" --policy vanilla
  cell "$MT" mukv         "$STIM" "$CTX" $MU
  cell "$MT" snapkv       "$STIM" "$CTX" --policy snapkv --obs-window 64 --n-sink 0
  # cell "$MT" adakv        "$STIM" "$CTX" --policy adakv --n-sink 0 --obs-window 32
  # cell "$MT" h2o          "$STIM" "$CTX" --policy h2o --n-sink 0 --obs-window 64
  # cell "$MT" tova         "$STIM" "$CTX" --policy tova
  # cell "$MT" streamingllm "$STIM" "$CTX" --policy streamingllm --n-sink 4
}
for MT in phi3 llama1b gemma2b bonsai8b; do
  i=0; n=$(echo "$STIMS" | wc -w)
  for STIM in $STIMS; do
    i=$((i+1)); case "$STIM" in *_L8K_*) CTX=8192;; *) CTX=4096;; esac
    echo "[$(date +%H:%M:%S)] $MT ($i/$n) $STIM ctx=$CTX"
    run_policies "$MT" "$STIM" "$CTX"
  done
  echo "[$(date +%H:%M:%S)] === $MT block done ==="
done
touch /tmp/niah_3model_DONE; echo "[$(date +%H:%M:%S)] NIAH 3-MODEL DONE (K=$KBUD) -> $OUT_HOST"
