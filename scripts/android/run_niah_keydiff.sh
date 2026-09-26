#!/bin/bash
# ============================================================================
# run_niah_keydiff.sh -- the KeyDiff row of the needle table. (2026-09-20)
#
# The 392-cell table in /tmp/niah_tableC_k1024_ctx16384 covers 4 models x 7
# policies x 14 stimuli. KeyDiff is missing from it: the only KeyDiff needle
# data we have is an earlier Llama-only campaign (/tmp/niah_vs_sllm, 08-15) on
# a different build, so it cannot be put in that table's column.
#
# This adds the 8th policy on the SAME stimuli, ctx, protocol and cool gate,
# writing into the SAME OUT_HOST so the cells sit beside the other seven.
#
# BUILD. KeyDiff needs bin_cpu_kd; the other seven rows were measured on
# bin_cpu_v87. The two columns this table reports, retrieval hits and live
# cells, are both build-invariant (a build changes speed, not which cell
# survives or whether the needle is found), so the row is comparable. Speed
# and energy from these cells must NOT be mixed with the other seven rows;
# the KeyDiff speed and energy numbers in the CPU table come from its own
# same-campaign pair instead. The build actually used is recorded per cell.
#
# KeyDiff runs at its published budget of 2048, as it does everywhere else in
# the paper, not at the table's K=1024. That is the same rule the other
# published-budget rows follow.
# ============================================================================
set -u
for _p in ${ADB_PORTS:-5152 5037 5151 5161}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue   # dead port + adb = squatting server that breaks ssh -R
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] using server port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

KBUD=2048
CTX=${CTX:-16384}
OUT_HOST=/tmp/niah_tableC_k1024_ctx${CTX}; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/niahkd_$(date +%Y%m%d_%H%M%S)
NIAH_SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/niah
CB=/data/local/tmp/endurkv/bin_cpu_kd
adb_safe_shell "mkdir -p $OUT" < /dev/null
for f in "$NIAH_SRC"/niah_L*_n0.txt; do adb push "$f" "$OUT/$(basename "$f")" < /dev/null >/dev/null 2>&1; done
STIMS=$(cd "$NIAH_SRC" && ls niah_L*_n0.txt)

declare -A MODELS=(
  [phi3]=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
  [llama1b]=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  [gemma2b]=/data/local/tmp/endurkv/models/gemma-2-2b-it-Q4_K_M.gguf
  [bonsai8b]=/data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf
)
KD="--policy keydiff --n-sink 0 --compact-inplace --keydiff-decode-block 128"

cell(){ local MT=$1 STIM=$2; local id="${MT}__keydiff__${STIM%.txt}"; local PD=$OUT/$id
  [ -f "$OUT_HOST/$id/meta.json" ] && { echo "  [$id] cached"; return; }
  adb_safe_shell "mkdir -p $PD" < /dev/null
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  echo "$CG" | tail -1
  case "$CG" in *"cool ddr="*) : ;;
    *) echo "[SKIP-HOT] $id -- gate failed; cell not run"; return ;; esac
  # the gate restores charging on exit; every metered cell runs with it off
  adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null >/dev/null 2>&1
  # no watchdog: it is muKV-only by design, exactly as for the other baselines
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$CB timeout ${TMO:-3600} $CB/eviction_bench --prompt $OUT/$STIM --prompt-id $id \
    --eval-mode gen --max-tokens 64 --ignore-eos --ctx-size $CTX --model ${MODELS[$MT]} --seed 42 \
    --threads 6 --n-gpu-layers 0 --greedy --k-nominal $KBUD $KD \
    --out-meta $PD/meta.json --out-csv /dev/null --out-gen $PD/gen.txt > $PD/c.out 2> $PD/c.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  echo "build=bin_cpu_kd k_nominal=$KBUD" > "$OUT_HOST/$id/BUILD.txt" 2>/dev/null
  local h=miss; grep -qiE 'mango sorbet|bi-rite' "$OUT_HOST/$id/gen.txt" 2>/dev/null && h=HIT
  echo "  [$id] $h"
}

# llama1b first: it is the model the CPU table's KeyDiff cell reports, so if the
# campaign is cut short the most useful block is already banked.
for MT in llama1b phi3 gemma2b bonsai8b; do
  i=0
  for STIM in $STIMS; do
    i=$((i+1)); echo "[$(date +%H:%M:%S)] $MT ($i/14) $STIM ctx=$CTX"
    cell "$MT" "$STIM"
  done
  echo "[$(date +%H:%M:%S)] === $MT keydiff block done ==="
done
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null
touch /tmp/niah_keydiff_DONE
echo "[$(date +%H:%M:%S)] NIAH KEYDIFF DONE -> $OUT_HOST"
