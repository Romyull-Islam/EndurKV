#!/bin/bash
# Pull whatever the phone-resident queue has finished. Safe to run any time.
export ANDROID_ADB_SERVER_PORT=${ANDROID_ADB_SERVER_PORT:-5160}
mkdir -p /tmp/qres && adb pull /data/local/tmp/endurkv/qres /tmp/ >/dev/null 2>&1
echo "--- queue log:"; tail -5 /tmp/qres/queue.log 2>/dev/null
echo "--- results:"; for j in /tmp/qres/*.json; do [ -f "$j" ] || continue; printf "  %-14s %s\n" "$(basename $j .json)" "$(grep -oE '"decode_tps": *[0-9.]+' $j | grep -oE '[0-9.]+')"; done
