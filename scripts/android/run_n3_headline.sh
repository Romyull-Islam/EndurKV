#!/bin/bash
# n=3 repeats of the phone CPU headline rows (Llama-3.2-1B): vanilla, muKV, KeyDiff at
# K=2048 and StreamingLLM 4+2000. Repeats are interleaved so slow drift hits every arm
# equally. Each cell waits for the cool gate (skipped if hot), records the starting battery
# voltage, and keeps its generation.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
LOG "waiting for gap closure ..."
while [ ! -f /tmp/gap_closure_DONE ]; do sleep 60; done
exec 9>/tmp/.endurkv_queue.lock; flock 9
LOG "starting n=3 CPU headline"
CB=/data/local/tmp/endurkv/bin_cpu_kd
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
DEV=/data/local/tmp/n3head; HOST=/tmp/n3_headline
mkdir -p $HOST; adb_safe_shell "mkdir -p $DEV/wt" < /dev/null >/dev/null 2>&1
timeout 180 adb push "$(cd "$(dirname "$0")/../.." && pwd)/eval_corpora"/wikitext_16k_p12k_d4k.txt "$DEV/wt/prompt.txt" < /dev/null >/dev/null 2>&1
MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"

cell(){
  local TAG=$1; shift
  [ -s "$HOST/$TAG/meta.json" ] && { LOG "$TAG cached"; return; }
  mkdir -p "$HOST/$TAG"
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) : ;; *) LOG "  [SKIP-HOT] $TAG"; return;; esac
  BV=$(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/voltage_now'" < /dev/null 2>/dev/null | tr -d ' \r')
  echo "batt_voltage_uv=$BV" > "$HOST/$TAG/start_power.txt"
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $DEV/$TAG.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 7200 env LD_LIBRARY_PATH=$CB $CB/eviction_bench \
    --prompt $DEV/wt/prompt.txt --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 \
    --n-batch 512 --ubatch-size 64 --model $M --seed 42 --threads 4 --n-gpu-layers 0 --greedy \
    --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > $DEV/$TAG.out 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  local w=0
  while [ $w -lt 7400 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 30; w=$((w+30)); done
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null >/dev/null 2>&1
  for e in json gen csv err; do
    adb_safe_pull "$DEV/$TAG.$e" "$HOST/$TAG/$([ $e = json ] && echo meta.json || ([ $e = gen ] && echo gen.txt || ([ $e = csv ] && echo sensors.csv || echo err.txt)))" >/dev/null 2>&1
  done
  [ -s "$HOST/$TAG/gen.txt" ] && LOG "  [$TAG] ok  batt=${BV:-?}" || LOG "  [$TAG] NO OUTPUT"
}
for r in 1 2 3; do
  cell vanilla_r$r      --policy vanilla --k-nominal 1024
  cell mukv_r$r         $MU --k-nominal 1024
  cell keydiff2048_r$r  --policy keydiff --n-sink 0 --k-nominal 2048 --compact-inplace --keydiff-decode-block 128
  cell sllm_r$r         --policy streamingllm --n-sink 4 --k-nominal 2000
done
LOG "N3_HEADLINE_DONE"
touch /tmp/n3_headline_DONE
