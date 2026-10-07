#!/bin/bash
# Splits muKV's CPU prefill overhead (Llama-1B, 6 threads) into eviction bookkeeping
# and side-node scoring:
#     vanilla       no scoring, no eviction
#     streamingllm  evicts without scoring   : vanilla to sllm = eviction bookkeeping
#     muKV          scores and evicts        : sllm to muKV    = side-node scoring
# n=3 with rotated order, 9.7K prompt, 64 decode tokens since only prefill is measured.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
exec 9>/tmp/.endurkv_queue.lock; flock 9
CB=/data/local/tmp/endurkv/bin_cpu_kd
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
DEV=/data/local/tmp/pfdec; HOST=/tmp/prefill_decompose
mkdir -p $HOST; adb_safe_shell "mkdir -p $DEV/wt" < /dev/null >/dev/null 2>&1
timeout 180 adb push "$(cd "$(dirname "$0")/../.." && pwd)/eval_corpora"/wikitext_16k_p12k_d4k.txt "$DEV/wt/prompt.txt" < /dev/null >/dev/null 2>&1
MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
flags(){ case "$1" in
  vanilla) echo "--policy vanilla --k-nominal 1024";;
  sllm)    echo "--policy streamingllm --n-sink 4 --k-nominal 2000";;
  mukv)    echo "$MU --k-nominal 1024";; esac; }
cell(){
  local TAG=$1; shift
  [ -s "$HOST/$TAG/meta.json" ] && { LOG "$TAG cached"; return; }
  mkdir -p "$HOST/$TAG"
  SC=$(adb_safe_shell "su -c 'c=0; for z in /sys/class/thermal/thermal_zone*; do t=\$(cat \$z/type 2>/dev/null); case \$t in *trip*) continue;; cpu-*|cpullc-*) ;; *) continue;; esac; v=\$((\$(cat \$z/temp)/1000)); [ \$v -gt \$c ] && c=\$v; done; echo \$c'" < /dev/null 2>/dev/null | tr -d ' \r')
  echo "start_cpu_c=$SC" > "$HOST/$TAG/start_state.txt"
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
  adb_safe_pull "$DEV/$TAG.err"  "$HOST/$TAG/err.txt"   >/dev/null 2>&1
  PF=$(python3 -c "
import json,re
try:
  j=json.loads(re.sub(r':\s*-?nan\b',': NaN',open('$HOST/$TAG/meta.json').read())); print('%.0f'%(j['prefill_ms']/1000))
except Exception: print('--')" 2>/dev/null)
  LOG "  [$TAG] prefill=${PF}s start_cpu=${SC}C"
}
# No per-cell cool gate: cooling between cells lets the WALT governor's load tracker
# decay, so prefill time tracks queue position rather than policy. The device is
# warmed once and the arms run back-to-back in a steady state.
LOG "=== prefill decomposition v2: warmed once, arms back-to-back, no inter-cell gate ==="
LOG "warming to a steady governor state ..."
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; warm_up 40'" < /dev/null 2>/dev/null | tail -1
for r in 1 2 3; do
  case $r in 1) O="vanilla sllm mukv";; 2) O="sllm mukv vanilla";; 3) O="mukv vanilla sllm";; esac
  LOG "repeat $r order: $O"
  for a in $O; do cell ${a}_r$r $(flags $a); done
done
LOG "PREFILL_DECOMPOSE_DONE"
touch /tmp/prefill_decompose_DONE
