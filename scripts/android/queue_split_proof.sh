#!/bin/bash
# Waits for the online bandit to finish (BANDIT_DONE in its log), pushes the engine build that
# carries --gpu-mhz-decode, rests the phone, then measures the split-clock plans.
# Log: /tmp/split_proof.log   Pid: /tmp/split_proof.pid
export ANDROID_SERIAL=${ANDROID_SERIAL:-10.0.0.127:5555}
export ADB_CALL_TIMEOUT=1500
echo $$ > /tmp/split_proof.pid
echo "[$(date '+%F %T')] queued; waiting for BANDIT_DONE in /tmp/bandit_online/log.txt"
while ! grep -q BANDIT_DONE /tmp/bandit_online/log.txt 2>/dev/null; do
  if ! ps -eo cmd | grep -q "[r]un_bandit_online"; then echo "[$(date '+%F %T')] bandit process gone without BANDIT_DONE; proceeding anyway"; break; fi
  sleep 120
done
echo "[$(date '+%F %T')] bandit finished; pushing the engine with --gpu-mhz-decode"
adb push /home/mislam22/EndurKV_workspace/EndurKV/entropy_probe/build-android-vulkan/eviction_bench /data/local/tmp/ukv/eviction_bench.new
adb shell "cd /data/local/tmp/ukv && mv eviction_bench.new eviction_bench && chmod 755 eviction_bench"
echo "[$(date '+%F %T')] resting 5 min"; sleep 300
bash /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/run_split_clock_proof.sh
echo "[$(date '+%F %T')] queue finished"
