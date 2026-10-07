#!/bin/bash
# StreamingLLM as in its paper (4 sinks, rolling window of 2000, positions inside the cache)
# via --sllm-window, against the full cache and muKV, on the phone GPU.
# Same protocol as Table 1: prompt_12k.txt (9737 Llama tokens, 11157 Phi-3 tokens), 4096
# generated tokens, ctx 16384, f16 KV, pinned, cool gate before every cell, charging off.
# Arms rotate across rounds. Llama-3.2-1B first, then Phi-3-mini.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-3C15B8003ZA00000}
export ANDROID_ADB_SERVER_PORT=${ANDROID_ADB_SERVER_PORT:-5162} ADB_CALL_TIMEOUT=${ADB_CALL_TIMEOUT:-1500}
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv_sllm
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
ML=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
MP=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
DEV=/data/local/tmp/endurkv/logs/sllmw_$(date +%Y%m%d_%H%M%S)
HOST=${HOST:-/tmp/sllm_window}; mkdir -p $HOST
PIN="taskset f0 nice -n -20"
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024"
SW="--policy streamingllm --n-sink 4 --k-nominal 2004 --compact-inplace --sllm-window"
VA="--policy vanilla --k-nominal 1024"
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null

settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    R=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $R|awk '{print int($1/1000)}'); local _sb=$(echo $R|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1; }

cell(){ # tag model flags...
  local TAG=$1 M=$2; shift 2
  local D=$HOST/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  adb_safe_shell "su -c 'rm -f /data/local/tmp/sw_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/sw_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG ..."
  adb_safe_shell "su -c 'cd $BIN && LD_LIBRARY_PATH=$BIN $PIN ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 --seed 42 \
    --threads 4 --n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 $* \
    --n-batch 512 --n-ubatch 64 --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen \
    --out-csv $DEV/$TAG.csv > /dev/null 2> $DEV/$TAG.err'" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null
  for f in json:meta.json gen:gen.txt err:err.txt csv:steps.csv; do
    adb_safe_pull "$DEV/$TAG.${f%%:*}" "$D/${f#*:}" >/dev/null 2>&1
  done
  adb_safe_pull "/data/local/tmp/sw_$TAG.csv" "$D/sensors.csv" >/dev/null 2>&1
  T=$(grep -oE '"decode_tps": *[0-9.]+' $D/meta.json 2>/dev/null | grep -oE '[0-9.]+')
  C=$(grep -oE 'retained_kv=[0-9.]+ MiB' $D/err.txt 2>/dev/null | tail -1)
  echo "  [$TAG] tok/s=$T $C"
}

# Rotated order: each arm runs first, second and third once per model.
cell llama_v_r1    $ML $VA;  cell llama_mukv_r1 $ML $MU;  cell llama_sw_r1   $ML $SW
cell llama_mukv_r2 $ML $MU;  cell llama_sw_r2   $ML $SW;  cell llama_v_r2    $ML $VA
cell llama_sw_r3   $ML $SW;  cell llama_v_r3    $ML $VA;  cell llama_mukv_r3 $ML $MU
cell phi3_v_r1     $MP $VA;  cell phi3_mukv_r1  $MP $MU;  cell phi3_sw_r1    $MP $SW
cell phi3_mukv_r2  $MP $MU;  cell phi3_sw_r2    $MP $SW;  cell phi3_v_r2     $MP $VA
cell phi3_sw_r3    $MP $SW;  cell phi3_v_r3     $MP $VA;  cell phi3_mukv_r3  $MP $MU
echo SLLM_WINDOW_DONE
