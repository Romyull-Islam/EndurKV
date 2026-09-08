#!/bin/bash
# ============================================================================
# rerun_kd_speed_faithful.sh -- redo the pinned GPU speed cells for KeyDiff WITH
# its decode-time eviction. (2026-08-17)
#
# WHY. /tmp/kd_speed was produced by the binary staged at 05:33, before
# --keydiff-decode-block existed. Those cells show evicted_total_decode = 0, so
# the cache grew 2048 -> 6144 across the 4096-token decode while muKV held ~1980.
# The resulting numbers (33.7 tok/s at 337 mJ/token, ~11.4 W) measure OUR missing
# mechanism, not KeyDiff: their Sec. 2.4 specifies B=1 during generation exactly so
# the budget holds throughout. Publishing them would be the handicapped-StreamingLLM
# error a second time. The old cells are kept under _nodecodeevict for the record.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
# 2026-08-22: the KD_LADDER_DONE gate is REMOVED. This run has no real dependency on the
# LongBench ladder -- the gate existed only to stop two campaigns sharing the phone, and
# supervise_campaign.sh now enforces that with an flock. The chain cost us these cells:
# the ladder aborted after its bounded 15 h wait during the 44 h outage, so this script
# aborted too, and neither had a supervisor to restart it. Independent queues from now on.

# stage the decode-block-capable Vulkan build alongside, in its own dir
adb_safe_shell "mkdir -p /data/local/tmp/ukv_kd2" < /dev/null
for so in /home/mislam22/EndurKV_workspace/EndurKV/llama.cpp/build-android-vulkan/bin/lib*.so; do
  adb push "$so" /data/local/tmp/ukv_kd2/ < /dev/null >/dev/null 2>&1
done
adb push /home/mislam22/EndurKV_workspace/EndurKV/entropy_probe/build-android-vulkan/eviction_bench \
         /data/local/tmp/ukv_kd2/eviction_bench < /dev/null >/dev/null 2>&1
adb_safe_shell "chmod 755 /data/local/tmp/ukv_kd2/eviction_bench" < /dev/null

[ -d /tmp/kd_speed ] && mv /tmp/kd_speed /tmp/kd_speed_nodecodeevict 2>/dev/null
mkdir -p /tmp/kd_speed
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
DEV=/data/local/tmp/endurkv/logs/kdspeed2_$(date +%Y%m%d_%H%M%S)
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
trap 'adb_safe_shell "su -c \"pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable\"" < /dev/null' EXIT INT TERM

settle(){ for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    R=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    d=$(echo $R|awk '{print int($1/1000)}'); b=$(echo $R|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${d}C batt=${b}C"
    [ "${d:-99}" -le 35 ] && [ "${b:-99}" -le 33 ] && return 0
  done; return 1; }

for r in 1 2 3; do
  D=/tmp/kd_speed/kd_r$r; [ -f "$D/meta.json" ] && continue
  mkdir -p "$D"; LOG "cooling for kd_r$r"; settle || { LOG "SKIP-HOT kd_r$r"; continue; }
  adb_safe_shell "su -c 'rm -f /data/local/tmp/kds_$r.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/kds_$r.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  LOG "running kd_r$r (keydiff-decode-block 128)"
  adb_safe_shell "su -c 'cd /data/local/tmp/ukv_kd2 && LD_LIBRARY_PATH=/data/local/tmp/ukv_kd2 taskset f0 nice -n -20 ./eviction_bench \
    --model $M --prompt $P --prompt-id kd_r$r --eval-mode gen --max-tokens 4096 --ignore-eos \
    --ctx-size 16384 --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 \
    --policy keydiff --n-sink 0 --k-nominal 2048 --compact-inplace --keydiff-decode-block 128 \
    --n-batch 512 --n-ubatch 64 --out-meta $DEV/kd_r$r.json --out-gen /dev/null --out-csv /dev/null \
    > /dev/null 2> $DEV/kd_r$r.err'" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null
  adb_safe_pull "$DEV/kd_r$r.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/kds_$r.csv" "$D/sensors.csv" >/dev/null 2>&1
  LOG "kd_r$r tps=$(grep -oE '\"decode_tps\": *[0-9.]+' $D/meta.json 2>/dev/null | grep -oE '[0-9.]+')"
done
touch /tmp/kd_speed_faithful_DONE; LOG "KD_SPEED_FAITHFUL_DONE"
