#!/bin/bash
# Re-enable charging on OnePlus 15 (OPlus oplus_chg driver).
# Counterpart to disable_charging.sh.
source /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
adb_safe_shell "su -c 'echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'"
sleep 4
adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/status'"
