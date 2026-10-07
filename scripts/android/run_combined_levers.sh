#!/bin/bash
# Cache and clock levers together (option B). Each battery tier pairs a cache budget
# with a clock cap, to compare against the cache-only runs at the same tier:
#   tier 0  k-pct 20, no cap     tier 1  k-pct 10, mid rung     tier 2  k-pct 5, low rung
set -u
# Pull with adb_safe_pull, not bare adb pull. adb_resilient.sh exports
# ANDROID_ADB_SERVER_PORT, which can conflict with a caller ADB_SERVER_SOCKET, and
# then a bare adb pull silently retrieves nothing.
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
DEV=/data/local/tmp/endurkv/logs/comb_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/combined_levers; mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace"
BIG="6 7"
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
restore(){ adb_safe_shell "su -c 'echo 1200 > /sys/kernel/gpu/gpu_max_clock; for c in $BIG 0 4; do f=/sys/devices/system/cpu/cpu\$c/cpufreq; [ -d \$f ] && cat \$f/cpuinfo_max_freq > \$f/scaling_max_freq 2>/dev/null; done'" < /dev/null; }
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; restore; }
trap cleanup EXIT INT TERM

settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    R=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $R|awk '{print int($1/1000)}'); local _sb=$(echo $R|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1; }

cell(){ # tag  pct  gpu_layers  gpuclk(0=none)  cpuclk(0=none)  maxtok
  local TAG=$1 PCT=$2 NGL=$3 GCLK=$4 CCLK=$5 TOK=$6
  local D=$HOST/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  restore
  [ "$GCLK" != "0" ] && adb_safe_shell "su -c 'echo $GCLK > /sys/kernel/gpu/gpu_max_clock'" < /dev/null
  [ "$CCLK" != "0" ] && adb_safe_shell "su -c 'for c in $BIG; do echo $CCLK > /sys/devices/system/cpu/cpu\$c/cpufreq/scaling_max_freq 2>/dev/null; done'" < /dev/null
  echo "    caps: gpu=${GCLK} cpu=${CCLK}  k-pct=${PCT}  ngl=${NGL}"
  adb_safe_shell "su -c 'rm -f /data/local/tmp/cb_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/cb_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG ..."
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG --eval-mode gen --max-tokens $TOK --ignore-eos --ctx-size 16384 \
    --seed 42 --threads 4 --n-gpu-layers $NGL --greedy --cache-type-k f16 --cache-type-v f16 \
    $MU --k-pct $PCT --n-batch 512 --n-ubatch 64 --out-meta $DEV/$TAG.json \
    --out-gen $DEV/$TAG.gen --out-csv /dev/null > /dev/null 2> $DEV/$TAG.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null; restore
  adb_safe_pull $DEV/$TAG.json "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull $DEV/$TAG.gen "$D/gen.txt" >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/cb_$TAG.csv" "$D/sensors.csv" >/dev/null 2>&1
  python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/clock_cell_report.py "$D" "$TAG" 2>/dev/null || echo "  [$TAG] FAILED"
}

echo "GPU, Option B: cache tier + matching GPU clock cap"
cell gpu_B_t0 20 99 1200 0 4096
cell gpu_B_t1 10 99  902 0 4096
cell gpu_B_t2  5 99  726 0 4096
echo "CPU, Option B: cache tier + matching prime-core cap"
cell cpu_B_t0 20  0 0 0       1024
cell cpu_B_t1 10  0 0 1996800 1024
cell cpu_B_t2  5  0 0 1500000 1024
echo COMBINED_DONE
