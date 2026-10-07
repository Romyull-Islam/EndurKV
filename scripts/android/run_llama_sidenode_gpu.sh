#!/bin/bash
# SnapKV and Ada-KV with the prefill side node (fused attention kept) on Llama-3.2-1B, phone
# GPU, same flags as the Phi-3 side-node rows of Table 1, against the same-session full cache.
# Same protocol as Table 1: prompt_12k.txt (9737 Llama tokens, 11157 Phi-3 tokens), 4096
# generated tokens, ctx 16384, f16 KV, pinned, cool gate before every cell, charging off.
# Arms rotate across rounds.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-3C15B8003ZA00000}
export ANDROID_ADB_SERVER_PORT=${ANDROID_ADB_SERVER_PORT:-5162} ADB_CALL_TIMEOUT=${ADB_CALL_TIMEOUT:-1500}
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv_sllm
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
ML=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
MP=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
DEV=/data/local/tmp/endurkv/logs/sidenode_$(date +%Y%m%d_%H%M%S)
HOST=${HOST:-/home/mislam22/EndurKV_workspace/tmp_archive/llama_sidenode}; mkdir -p $HOST
PIN="taskset f0 nice -n -20"
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024"
SN="--policy snapkv --obs-window 16 --snapkv-kernel 5 --n-sink 0 --k-nominal 1024 --fa-on-evict --no-evict-decode --compact-inplace"
AN="--policy adakv --n-sink 0 --k-nominal 1024 --fa-on-evict --no-evict-decode --compact-inplace"
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

# Rotated order: each arm runs first, second and third once.
cell llama_v_r1      $ML $VA;  cell llama_snapsn_r1 $ML $SN;  cell llama_adasn_r1  $ML $AN
cell llama_snapsn_r2 $ML $SN;  cell llama_adasn_r2  $ML $AN;  cell llama_v_r2      $ML $VA
cell llama_adasn_r3  $ML $AN;  cell llama_v_r3      $ML $VA;  cell llama_snapsn_r3 $ML $SN
echo LLAMA_SIDENODE_DONE
