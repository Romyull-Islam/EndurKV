#!/bin/bash
# ============================================================================
# run_prefill_v3.sh -- prefill with BOTH thermal state and governor state matched.
# (2026-08-28)
#
# Why v1 and v2 both failed:
#   v1 cool-gated every cell  -> equal TEMPERATURE, but the WALT governor decayed
#                                during each gate, so prefill was set by position
#                                in the repeat (slot1 ~500 s, slot3 217 s).
#   v2 warmed once, no gates  -> equal GOVERNOR state, but temperature drifted
#                                38->49 C and the first two cells were still
#                                ramping. Those two happened to be vanilla and
#                                StreamingLLM, so muKV never ran a ramp cell and
#                                vanilla ended with n=2 against muKV's n=3.
#
# v3 pins both. Per cell: cool until DDR<=35 AND cores are BELOW the target, then
# warm_up UNTIL cores reach exactly WARM_TARGET. Every cell therefore begins at
# the same core temperature with a governor that has just seen load. A throwaway
# cell runs first and is discarded, so no arm inherits the campaign ramp.
#
# Rotated n=3. 64 decode tokens: this measures prefill.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
LOG "waiting for the compaction test ..."
while [ ! -f /tmp/compaction_allmodels_DONE ]; do sleep 60; done
exec 9>/tmp/.endurkv_queue.lock; flock 9
CB=/data/local/tmp/endurkv/bin_cpu_kd
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
DEV=/data/local/tmp/pf3; HOST=/tmp/prefill_v3; WARM_TARGET=42
mkdir -p $HOST; adb_safe_shell "mkdir -p $DEV/wt" < /dev/null >/dev/null 2>&1
timeout 180 adb push /tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad/wikitext_16k_p12k_d4k.txt "$DEV/wt/prompt.txt" < /dev/null >/dev/null 2>&1
MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
flags(){ case "$1" in
  vanilla) echo "--policy vanilla --k-nominal 1024";;
  sllm)    echo "--policy streamingllm --n-sink 4 --k-nominal 2000";;
  mukv)    echo "$MU --k-nominal 1024";; esac; }
cell(){
  local TAG=$1; shift
  [ -s "$HOST/$TAG/meta.json" ] && { LOG "$TAG cached"; return; }
  mkdir -p "$HOST/$TAG"
  # 1. cool BELOW the target
  CG=$(adb_safe_shell "su -c 'COOL_CPU_MAX=38 . /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) : ;; *) LOG "  [SKIP-HOT] $TAG ($CG)"; return;; esac
  # 2. warm UP to exactly the target, so the governor has just seen load
  WU=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; warm_up $WARM_TARGET'" < /dev/null 2>/dev/null | tail -1)
  SC=$(adb_safe_shell "su -c 'c=0; for z in /sys/class/thermal/thermal_zone*; do t=\$(cat \$z/type 2>/dev/null); case \$t in *trip*) continue;; cpu-*|cpullc-*) ;; *) continue;; esac; v=\$((\$(cat \$z/temp)/1000)); [ \$v -gt \$c ] && c=\$v; done; echo \$c'" < /dev/null 2>/dev/null | tr -d ' \r')
  echo "start_cpu_c=$SC gate=$CG warm=$WU" > "$HOST/$TAG/start_state.txt"
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 3600 env LD_LIBRARY_PATH=$CB $CB/eviction_bench \
    --prompt $DEV/wt/prompt.txt --prompt-id $TAG --eval-mode gen --max-tokens 64 --ignore-eos --ctx-size 16384 \
    --n-batch 512 --ubatch-size 64 --model $M --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > /dev/null 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  local w=0
  while [ $w -lt 3700 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 20; w=$((w+20)); done
  adb_safe_pull "$DEV/$TAG.json" "$HOST/$TAG/meta.json" >/dev/null 2>&1
  PF=$(python3 -c "
import json,re
try:
  j=json.loads(re.sub(r':\s*-?nan\b',': NaN',open('$HOST/$TAG/meta.json').read())); print('%.0f'%(j['prefill_ms']/1000))
except Exception: print('--')" 2>/dev/null)
  LOG "  [$TAG] prefill=${PF}s start_cpu=${SC}C"
}
LOG "=== prefill v3: every cell cooled below ${WARM_TARGET}C then warmed UP to it ==="
cell discard_warmup $(flags vanilla)      # thrown away: absorbs the campaign ramp
rm -rf $HOST/discard_warmup
for r in 1 2 3; do
  case $r in 1) O="vanilla sllm mukv";; 2) O="sllm mukv vanilla";; 3) O="mukv vanilla sllm";; esac
  LOG "repeat $r order: $O"
  for a in $O; do cell ${a}_r$r $(flags $a); done
done
LOG "PREFILL_V3_DONE"
touch /tmp/prefill_v3_DONE
