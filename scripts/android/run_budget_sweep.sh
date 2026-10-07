#!/bin/bash
# Budget sweep on the phone GPU: realized cells against nominal budget K for SnapKV,
# H2O, TOVA, Ada-KV, StreamingLLM and muKV at K = 256..2048, one run each.
# Llama-3.2-1B, ctx 16384, 64 generated tokens. Runs are pinned (taskset f0, nice -20)
# because unpinned GPU decode on this phone is bimodal. StreamingLLM uses in-place
# compaction, which has a PPL bug at 64K context, so do not reuse that setting there.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv_n3
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
E=/data/local/tmp/endurkv/eval_data/dis.txt      # asserted 0/119 and 0/399 vs P
DEV=/data/local/tmp/endurkv/logs/bsweep_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/budget_sweep; mkdir -p $HOST
PIN="taskset f0 nice -n -20"
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024"
# Per-head arm flags match run_phone_gpu_complete.sh.
SNAP="--policy snapkv --obs-window 64 --snapkv-kernel 5 --n-sink 0 --k-nominal 1024"
ADA="--policy adakv --n-sink 0 --obs-window 32 --k-nominal 1024"
H2O="--policy h2o --n-sink 0 --obs-window 64 --k-nominal 1024"
TOVA="--policy tova --k-nominal 1024"
SLLM="--policy streamingllm --n-sink 4 --compact-inplace"
MUK="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace"
SH="--policy streamingllm --n-sink 4 --k-nominal 1024 --no-fa-positional --no-defrag"
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
  local EX="--eval-mode gen --max-tokens 64 --ignore-eos"
  [ "$MODE" = ppl ] && EX="--eval-mode ppl --eval-text $E"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  adb_safe_shell "su -c 'rm -f /data/local/tmp/sl_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/sl_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG ..."
  # Launch detached and poll with short adb calls, so a dropped tunnel cannot kill the
  # run or trip the helper timeout (its adb kill-server is fatal through a tunnel).
  adb_safe_shell "su -c 'rm -f $DEV/$TAG.json; cd $BIN && LD_LIBRARY_PATH=$BIN nohup $PIN ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG $EX --ctx-size 16384 --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 $* --n-batch 512 --n-ubatch 64 \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null \
    > /dev/null 2> $DEV/$TAG.err < /dev/null &'" < /dev/null
  local T0=$(date +%s)
  while true; do
    sleep 30
    done_flag=$(adb_safe_shell "su -c 'test -s $DEV/$TAG.json && echo yes || echo no'" < /dev/null | tr -d '\r' | tail -1)
    [ "$done_flag" = "yes" ] && break
    alive=$(adb_safe_shell "su -c 'pgrep -f \"prompt-id $TAG \" | wc -l'" < /dev/null | tr -d '\r' | tail -1)
    if [ "${alive:-1}" = "0" ]; then echo "  [$TAG] process gone without a result"; break; fi
    [ $(($(date +%s)-T0)) -gt 5400 ] && { echo "  [$TAG] 90 min, giving up"; adb_safe_shell "su -c 'pkill -f \"prompt-id $TAG \"'" < /dev/null; break; }
  done
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null
  adb_safe_pull "$DEV/$TAG.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.gen"  "$D/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/sl_$TAG.csv" "$D/sensors.csv" >/dev/null 2>&1
  T=$(grep -oE '"decode_tps": *[0-9.]+' $D/meta.json 2>/dev/null | grep -oE '[0-9.]+')
  echo "  [$TAG] tok/s=$T"
}

# The per-head flags above carry --k-nominal 1024, so strip it and pass K explicitly.
nok(){ echo "$*" | sed 's/--k-nominal 1024//'; }
for K in 256 512 1024 2048; do
  cell snapkv_k$K  gen $(nok $SNAP) --k-nominal $K
  cell h2o_k$K     gen $(nok $H2O)  --k-nominal $K
  cell tova_k$K    gen $(nok $TOVA) --k-nominal $K
  cell adakv_k$K   gen $(nok $ADA)  --k-nominal $K
  cell sllm_k$K    gen $SLLM --k-nominal $K
  cell mukv_k$K    gen $MUK  --k-nominal $K
done
echo BUDGET_SWEEP_DONE
