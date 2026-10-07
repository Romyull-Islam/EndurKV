#!/bin/bash
# StreamingLLM at matched K=1024 (FA-on, compacted in place) against vanilla on the phone
# GPU. Llama-3.2-1B, ctx 16384, 9737-token prompt plus 4096 generated, 3 interleaved reps,
# then one PPL cell.
#
# StreamingLLM's own code (mit-han-lab/streaming-llm, kv_cache.py) concatenates the sinks
# and the recent window into a dense cache, so compaction is part of the policy. Its
# documented budget is start_size=4 plus recent_size=2000 (K=2004).
#
# Pinned with taskset f0 and nice -20 because unpinned GPU decode is bimodal (28 to 39 tok/s).
# In-place compaction matches the round-trip for StreamingLLM at this context but is broken
# for it at 64K.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv_n3
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
E=/data/local/tmp/endurkv/eval_data/dis.txt      # asserted 0/119 and 0/399 vs P
DEV=/data/local/tmp/endurkv/logs/sllmmatch_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/sllm_matched; mkdir -p $HOST
PIN="taskset f0 nice -n -20"
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024"
SF_OWN="--policy streamingllm --n-sink 4 --k-nominal 2004 --compact-inplace"
SH="--policy streamingllm --n-sink 4 --k-nominal 1024 --no-fa-positional --no-defrag"
SF_MATCH="--policy streamingllm --n-sink 4 --k-nominal 1024 --compact-inplace"
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

cell(){ # tag mode flags...
  local TAG=$1 MODE=$2; shift 2
  local D=$HOST/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  local EX="--eval-mode gen --max-tokens 4096 --ignore-eos"
  [ "$MODE" = ppl ] && EX="--eval-mode ppl --eval-text $E"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  adb_safe_shell "su -c 'rm -f /data/local/tmp/sl_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/sl_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG ..."
  adb_safe_shell "su -c 'cd $BIN && LD_LIBRARY_PATH=$BIN $PIN ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG $EX --ctx-size 16384 --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 $* --n-batch 512 --n-ubatch 64 \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null \
    > /dev/null 2> $DEV/$TAG.err'" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null
  adb_safe_pull "$DEV/$TAG.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.gen"  "$D/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/sl_$TAG.csv" "$D/sensors.csv" >/dev/null 2>&1
  T=$(grep -oE '"decode_tps": *[0-9.]+' $D/meta.json 2>/dev/null | grep -oE '[0-9.]+')
  echo "  [$TAG] tok/s=$T"
}

# interleaved across reps so session drift is shared, not loaded onto one arm
for r in 1 2 3; do
  cell v_r$r        gen $VA
  cell sfmatch_r$r  gen $SF_MATCH
done
cell sfmatch_ppl ppl $SF_MATCH
echo SLLM_MATCHED_DONE
