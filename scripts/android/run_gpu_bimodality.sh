#!/bin/bash
# run_gpu_bimodality.sh: 8 identical phone-GPU cells (muKV, k-pct 10) to find why the
# same run lands at about 30 or 39 tok/s. A second sampler logs kgsl gpuclk,
# thermal_pwrlevel and max_pwrlevel at 2 Hz, which sample_sensors.sh does not record.
# A higher thermal_pwrlevel or lower gpuclk in slow cells means throttling. The same clock
# with less busy time means the GPU is starved of work. Full cool gate before each cell.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv_n3
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
DEV=/data/local/tmp/endurkv/logs/bimod_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/gpu_bimod; mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-pct 10"
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; pkill -f gpuclk_sampler; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null

# GPU clock sampler (2 Hz)
adb_safe_shell "su -c 'cat > /data/local/tmp/gpuclk_sampler.sh <<SH
#!/system/bin/sh
echo t_s,gpuclk,thermal_pwrlevel,max_pwrlevel > \\\$1
while true; do
  echo \\\$(date +%s),\\\$(cat /sys/class/kgsl/kgsl-3d0/gpuclk 2>/dev/null),\\\$(cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel 2>/dev/null),\\\$(cat /sys/class/kgsl/kgsl-3d0/max_pwrlevel 2>/dev/null) >> \\\$1
  sleep 0.5
done
SH
chmod 755 /data/local/tmp/gpuclk_sampler.sh'" < /dev/null

settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    R=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $R|awk '{print int($1/1000)}'); local _sb=$(echo $R|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1; }

for i in 1 2 3 4 5 6 7 8; do
  TAG=c$i; D=$HOST/$TAG
  [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; continue; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; continue; }
  adb_safe_shell "su -c 'rm -f /data/local/tmp/bm_$TAG.csv /data/local/tmp/clk_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/bm_$TAG.csv --hz 2 >/dev/null 2>&1 & nohup sh /data/local/tmp/gpuclk_sampler.sh /data/local/tmp/clk_$TAG.csv >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG ..."
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 \
    --seed 42 --threads 4 --n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 \
    $MU --n-batch 512 --n-ubatch 64 --out-meta $DEV/$TAG.json --out-gen /dev/null \
    --out-csv /dev/null > /dev/null 2> $DEV/$TAG.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors; pkill -f gpuclk_sampler'" < /dev/null
  adb_safe_pull "$DEV/$TAG.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/bm_$TAG.csv"  "$D/sensors.csv" >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/clk_$TAG.csv" "$D/gpuclk.csv"  >/dev/null 2>&1
  T=$(grep -oE '"decode_tps": *[0-9.]+' $D/meta.json 2>/dev/null | grep -oE '[0-9.]+')
  echo "  [$TAG] tok/s=$T"
done
echo BIMOD_DONE
