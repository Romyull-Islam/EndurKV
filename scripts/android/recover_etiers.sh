#!/bin/bash
# The tier sweep's bare `adb pull` calls retrieve nothing (see the fix above), but every
# cell's meta/gen/sensors is on the device. Pull them with the resilient wrapper once the
# campaign finishes, so no phone time is wasted re-running completed cells.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
until grep -q ETIERS_DONE /tmp/etiers.log 2>/dev/null; do sleep 60; done
DEV=$(adb_safe_shell "ls -d /data/local/tmp/endurkv/logs/etiers_* 2>/dev/null | tail -1" < /dev/null | tr -d '\r')
echo "recovering from $DEV"
for MODE in gen ppl; do for TAG in vanilla pct20 pct10 pct5; do
  D=/tmp/energy_tiers/${MODE}_${TAG}; mkdir -p "$D"
  adb_safe_pull "$DEV/${MODE}_${TAG}.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/${MODE}_${TAG}.gen"  "$D/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/s_${TAG}.csv" "$D/sensors.csv" >/dev/null 2>&1
  [ -f "$D/meta.json" ] && python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/energy_cell_report.py "$D" "$MODE/$TAG" 2>/dev/null
done; done
echo RECOVER_DONE
