#!/bin/bash
# boot_watch.sh: log phone reboots with the boot reason, whether eviction_bench was
# running just before, and the last DDR and battery temperatures. Polls every 30 s
# because Android keeps the boot reason only for the most recent boot.
# A reboot shows up as a change in /proc/sys/kernel/random/boot_id.
LOG=${1:-/tmp/boot_watch.log}
PORT=${ANDROID_ADB_SERVER_PORT:-5152}
prev_id=""; last_busy=0; last_ddr=""; last_bat=""
while true; do
  id=$(ANDROID_ADB_SERVER_PORT=$PORT adb shell 'cat /proc/sys/kernel/random/boot_id 2>/dev/null' 2>/dev/null | tr -d '\r')
  if [ -n "$id" ]; then
    if [ -n "$prev_id" ] && [ "$id" != "$prev_id" ]; then
      reason=$(ANDROID_ADB_SERVER_PORT=$PORT adb shell 'getprop sys.boot.reason' 2>/dev/null | tr -d '\r')
      printf '[%s] REBOOT detected  reason=%s  benchmark_was_running=%s  last_ddr=%s last_batt=%s\n' \
        "$(date '+%F %T')" "${reason:-unknown}" "$([ "$last_busy" -gt 0 ] && echo YES || echo no)" \
        "${last_ddr:-?}" "${last_bat:-?}" >> "$LOG"
    fi
    prev_id="$id"
    last_busy=$(ANDROID_ADB_SERVER_PORT=$PORT adb shell 'ps -A 2>/dev/null | grep -c eviction_bench' 2>/dev/null | tr -d '\r')
    last_busy=${last_busy:-0}
    t=$(ANDROID_ADB_SERVER_PORT=$PORT adb shell 'for z in /sys/class/thermal/thermal_zone*; do n=$(cat $z/type 2>/dev/null); case "$n" in ddr) echo "D$(( $(cat $z/temp)/1000 ))";; battery) echo "B$(( $(cat $z/temp)/1000 ))";; esac; done' 2>/dev/null | tr -d '\r')
    last_ddr=$(echo "$t" | grep -o 'D[0-9]*' | head -1 | tr -d D)
    last_bat=$(echo "$t" | grep -o 'B[0-9]*' | head -1 | tr -d B)
  fi
  sleep 30
done
