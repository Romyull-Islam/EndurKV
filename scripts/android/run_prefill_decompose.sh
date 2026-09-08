#!/bin/bash
# ============================================================================
# run_prefill_decompose.sh -- WHY is muKV's prefill slower than vanilla? (2026-08-28)
#
# The question. On Llama-1B/GPU muKV's prefill is +2% over vanilla (134 vs 131 s):
# effectively free, and the behaviour previously observed. On Llama-1B/CPU it is
# +13.7% (207+-11 vs 182+-7, n=3) and on Phi-3 +24%. Nothing in the prefill loop
# reads the compaction flag (want_defrag is computed at line 3368; the prefill
# timer closes at 2486), so the earlier diagnostic's 452/585/647 spread across
# three muKV arms was thermal/position noise, not mechanism.
#
# The decomposition. Three arms isolate the two candidate costs, because they
# differ by exactly one property each:
#     vanilla        no scoring, no eviction        -> baseline
#     streamingllm   NO scoring, but DOES evict     -> vanilla..sllm = eviction bookkeeping
#     muKV           scoring AND evicts             -> sllm..muKV    = side-node scoring
# If sllm ~ muKV, the toll is eviction bookkeeping and the side node is free (which
# is what Phi-3/CPU hinted: vanilla 381, sllm 472, muKV 471). If sllm ~ vanilla,
# the toll is the side node and the fix belongs in the scoring path.
#
# Protocol. n=3, rotated (each arm visits each slot once), cool-gated, 6 threads,
# same 9.7K prompt as Table 1, and only 64 decode tokens because this measures
# PREFILL -- decode is a formality here and keeps each cell to ~4 min.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
exec 9>/tmp/.endurkv_queue.lock; flock 9
CB=/data/local/tmp/endurkv/bin_cpu_kd
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
DEV=/data/local/tmp/pfdec; HOST=/tmp/prefill_decompose
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
# v2 (2026-08-28): NO per-cell cool gate. v1 gated between cells, which let the WALT
# governor's load tracker decay, so prefill was set by a cell's POSITION in the repeat
# (slot1 ~500 s, slot2 333 s, slot3 217 s) and the SAME policy moved 333 -> 517 s
# between repeats: policy was invisible. Here the device is warmed once and the arms
# run back-to-back in a sustained state -- the soak protocol, which held vanilla to
# 4.97-5.02 tok/s across six generations.
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
