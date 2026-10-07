#!/bin/bash
# run_prefill_v4.sh: prefill time for vanilla, StreamingLLM and muKV with matched thermal
# and governor state. Each cell cools below the target, then warms up to WARM_TARGET and
# launches inside one device-side script (pf4_cell.sh), because heat lost between separate
# adb calls made start temperature, and so prefill time, vary from cell to cell.
# Rotated order, n=3, 64 decode tokens.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }

exec 9>/tmp/.endurkv_queue.lock; flock 9
DEV=/data/local/tmp/pf3; HOST=/tmp/prefill_v4; WARM_TARGET=42
SCRATCH="${SCRATCH:-$(cd "$(dirname "$0")/../.." && pwd)/eval_corpora}"
mkdir -p $HOST
adb_safe_shell "mkdir -p $DEV/wt" < /dev/null >/dev/null 2>&1
timeout 180 adb push "$SCRATCH/wikitext_16k_p12k_d4k.txt" "$DEV/wt/prompt.txt" < /dev/null >/dev/null 2>&1
timeout 120 adb push "$SCRATCH/pf4_cell.sh" /data/local/tmp/endurkv/scripts/pf4_cell.sh < /dev/null >/dev/null 2>&1
adb_safe_shell "chmod 755 /data/local/tmp/endurkv/scripts/pf4_cell.sh" < /dev/null >/dev/null 2>&1

MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
flags(){ case "$1" in
  vanilla) echo "--policy vanilla --k-nominal 1024";;
  sllm)    echo "--policy streamingllm --n-sink 4 --k-nominal 2000";;
  mukv)    echo "$MU --k-nominal 1024";; esac; }

cell(){
  local TAG=$1; shift
  [ -s "$HOST/$TAG/meta.json" ] && { LOG "  $TAG cached"; return; }
  mkdir -p "$HOST/$TAG"
  # 1. Cool below the target so every cell warms up to it from the same side
  CG=$(adb_safe_shell "su -c 'COOL_CPU_MAX=38 . /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) : ;; *) LOG "  [SKIP-HOT] $TAG ($CG)"; return;; esac
  # 2. Warm, measure and launch on-device with no adb round trip in between
  adb_safe_shell "su -c '/data/local/tmp/endurkv/scripts/pf4_cell.sh $TAG $WARM_TARGET $*'" < /dev/null >/dev/null 2>&1
  local w=0
  while [ $w -lt 3700 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 20; w=$((w+20)); done
  adb_safe_pull "$DEV/$TAG.json" "$HOST/$TAG/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.sc"   "$HOST/$TAG/start_state.txt" >/dev/null 2>&1
  local SC PF
  SC=$(grep -o 'start_cpu_c=[0-9]*' "$HOST/$TAG/start_state.txt" 2>/dev/null | cut -d= -f2)
  PF=$(python3 -c "
import json,re
try:
  j=json.loads(re.sub(r':\s*-?nan\b',': NaN',open('$HOST/$TAG/meta.json').read())); print('%.0f'%(j['prefill_ms']/1000))
except Exception: print('--')" 2>/dev/null)
  LOG "  [$TAG] prefill=${PF}s start_cpu=${SC:-?}C"
}

LOG "=== prefill v4: warm/measure/launch fused on-device, target ${WARM_TARGET}C ==="
cell discard_warmup $(flags vanilla)      # thrown away: absorbs the campaign ramp
rm -rf $HOST/discard_warmup
for r in 1 2 3; do
  case $r in 1) O="vanilla sllm mukv";; 2) O="sllm mukv vanilla";; 3) O="mukv vanilla sllm";; esac
  LOG "repeat $r order: $O"
  for a in $O; do cell ${a}_r$r $(flags $a); done
done
LOG "PREFILL_V4_DONE"
touch /tmp/prefill_v4_DONE
