#!/system/bin/sh
# High-rate cluster+temp logger for the watchdog experiments.
# args: $1=out.csv  $2=stop_flag  $3=battery_zone_dir  $4=skin_zone_dir
# CSV columns: epoch,cpu0_cur(perf),cpu6_cur(prime),cpu0_max,cpu6_max,bat_mc,skin_mc
OUT=$1; STOP=$2; BZ=$3; SZ=$4
: > "$OUT"
while [ ! -f "$STOP" ]; do
  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$(date +%s)" \
    "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)" \
    "$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq 2>/dev/null)" \
    "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null)" \
    "$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq 2>/dev/null)" \
    "$(cat "$BZ/temp" 2>/dev/null)" \
    "$(cat "$SZ/temp" 2>/dev/null)" >> "$OUT"
  sleep 0.5
done
