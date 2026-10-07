#!/bin/bash
# run_niah_sllmw.sh: StreamingLLM needle cells as in its paper (4 sinks, rolling window of
# 2000, positions inside the cache, --sllm-window). Same stimuli, ctx, protocol and cool gate
# as run_niah_sllm2004.sh, which ran the earlier port.
set -u
for _p in ${ADB_PORTS:-5162 5152 5037 5151 5161}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue   # dead port + adb = squatting server that breaks ssh -R
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] using server port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

KBUD=2004
CTX=${CTX:-16384}
OUT_HOST=/home/mislam22/EndurKV_workspace/tmp_archive/niah_sllmw_ctx${CTX}; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/niahsw_$(date +%Y%m%d_%H%M%S)
NIAH_SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/niah
CB=/data/local/tmp/endurkv/bin_cpu_sllm
adb_safe_shell "mkdir -p $OUT" < /dev/null
for f in "$NIAH_SRC"/niah_L*_n0.txt; do adb push "$f" "$OUT/$(basename "$f")" < /dev/null >/dev/null 2>&1; done
STIMS=$(cd "$NIAH_SRC" && ls niah_L*_n0.txt)

declare -A MODELS=(
  [phi3]=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
  [llama1b]=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  [gemma2b]=/data/local/tmp/endurkv/models/gemma-2-2b-it-Q4_K_M.gguf
  [bonsai8b]=/data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf
)
SL="--policy streamingllm --n-sink 4 --sllm-window"

cell(){ local MT=$1 STIM=$2; local id="${MT}__sllmw__${STIM%.txt}"; local PD=$OUT/$id
  [ -f "$OUT_HOST/$id/meta.json" ] && { echo "  [$id] cached"; return; }
  adb_safe_shell "mkdir -p $PD" < /dev/null
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  echo "$CG" | tail -1
  case "$CG" in *"cool ddr="*) : ;;
    *) echo "[SKIP-HOT] $id -- gate failed; cell not run"; return ;; esac
  # the gate restores charging on exit; every metered cell runs with it off
  adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null >/dev/null 2>&1
  # no watchdog: it is muKV-only, as for the other baselines
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$CB timeout ${TMO:-3600} $CB/eviction_bench --prompt $OUT/$STIM --prompt-id $id \
    --eval-mode gen --max-tokens 64 --ignore-eos --ctx-size $CTX --model ${MODELS[$MT]} --seed 42 \
    --threads 6 --n-gpu-layers 0 --greedy --k-nominal $KBUD $SL \
    --out-meta $PD/meta.json --out-csv /dev/null --out-gen $PD/gen.txt > $PD/c.out 2> $PD/c.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  echo "build=bin_cpu_sllm k_nominal=$KBUD published 4+2000, --sllm-window" > "$OUT_HOST/$id/BUILD.txt" 2>/dev/null
  local h=miss; grep -qiE 'mango sorbet|bi-rite' "$OUT_HOST/$id/gen.txt" 2>/dev/null && h=HIT
  echo "  [$id] $h"
}

# llama1b first, so its block is done if the campaign is cut short.
for MT in llama1b phi3 gemma2b bonsai8b; do
  i=0
  for STIM in $STIMS; do
    i=$((i+1)); echo "[$(date +%H:%M:%S)] $MT ($i/14) $STIM ctx=$CTX"
    cell "$MT" "$STIM"
  done
  echo "[$(date +%H:%M:%S)] === $MT sllmw block done ==="
done
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null
echo NIAH_SLLMW_DONE
echo "[$(date +%H:%M:%S)] NIAH SLLM-2004 DONE -> $OUT_HOST"
