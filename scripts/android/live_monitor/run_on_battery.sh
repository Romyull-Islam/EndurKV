#!/usr/bin/env bash
# Put the OnePlus 15 on battery while staying USB/usbip-tethered, so the live dashboard
# and energy scripts read real wattage. Disables charging (mmi_charging_enable) for a
# bounded window and re-enables it on Ctrl-C, on exit, and through an on-phone timeout
# that fires even if this host or the adb link dies.
#
#   ./run_on_battery.sh           # 600 s (10 min) on battery, then auto-restore
#   ./run_on_battery.sh 300       # custom duration in seconds
#
# While it runs: trigger inference in another terminal and watch live_dashboard.py -
# the status pill turns green (Discharging) and power (W) becomes real.
set -uo pipefail
ADB="${ADB:-/home/mislam22/tools/platform-tools/adb -s 3C15B8003ZA00000}"
MAX="${1:-600}"
NODE="/sys/class/oplus_chg/battery/mmi_charging_enable"

echo "[battery] node=$NODE  duration=${MAX}s"
echo "[battery] disabling charging (auto re-enable on Ctrl-C / exit / ${MAX}s timeout)…"

# Host-side re-enable, a backstop to the on-phone trap.
host_restore() { $ADB shell "su -c 'echo 1 > $NODE'" >/dev/null 2>&1; echo; echo "[battery] charging re-enabled."; }
trap host_restore EXIT INT TERM

# The guarded session runs on the phone, so its trap and timeout restore charging
# even if this host process is killed or adb drops.
$ADB shell "su -c '
N=$NODE
[ -e \"\$N\" ] || { echo NO_NODE; exit 1; }
ENF=\$(getenforce 2>/dev/null); setenforce 0 2>/dev/null
restore(){ echo 1 > \"\$N\" 2>/dev/null; [ \"\$ENF\" = Enforcing ] && setenforce 1 2>/dev/null; echo PHONE_RESTORED; }
trap restore EXIT INT TERM HUP
echo 0 > \"\$N\"
sleep 2
B=/sys/class/power_supply/battery
echo \"on-battery: status=\$(cat \$B/status) current_now=\$(cat \$B/current_now) voltage_now=\$(cat \$B/voltage_now)\"
i=0
while [ \$i -lt $MAX ]; do
  [ -f /data/local/tmp/op15_charge_resume ] && { echo EARLY_RESUME; break; }
  sleep 1; i=\$((i+1))
done
'" 2>&1 | sed 's/^/  /'

echo "[battery] session window ended."
