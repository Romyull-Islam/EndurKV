#!/bin/bash
# Stop charging while keeping USB connected (OnePlus 15 / OPlus oplus_chg driver).
# Writing 0 to mmi_charging_enable causes status to flip from "Full"/"Charging"
# to "Not charging" within ~1-3 s, so energy meters reflect true discharge.
#
# Restore with: enable_charging.sh
source /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'"
sleep 2
adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/status'"
