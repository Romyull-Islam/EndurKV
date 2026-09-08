#!/system/bin/sh
# gpu_watchdog.sh -- a DDR-temperature ladder for the GPU user cap (2026-09-07). Runs on the phone
# as root, detached. Steps the GPU cap down one level at a time BEFORE the vendor limiter trips
# (measured: the vendor drops the Adreno 840 to 726 MHz at DDR ~64 C), and restores one level at
# a time with 1.5 C of hysteresis. Writes max_pwrlevel (the user cap the driver composes with the
# thermal cap by max()), so the vendor can still go lower; it never lifts a stricter cap written by
# the scheduler (it only raises the level index, and restores to the level it found).
#   ladder (DDR, tenths of a degree): >= 60.0 -> level 1 (1050 MHz), >= 62.0 -> level 2 (967),
#   >= 63.5 -> level 3 (902). Log: /data/local/tmp/endurkv/gpu_watchdog.log
Z=/sys/class/thermal/thermal_zone47/temp; KD=/sys/class/kgsl/kgsl-3d0; LOG=/data/local/tmp/endurkv/gpu_watchdog.log
T1=${T1:-600}; T2=${T2:-620}; T3=${T3:-635}; HYS=${HYS:-15}; PERIOD=${PERIOD:-2}
base=$(cat $KD/max_pwrlevel); lvl=$base
echo "$(date '+%F %T') start base_level=$base ladder $T1/$T2/$T3 hys $HYS" >> $LOG
trap 'echo $base > $KD/max_pwrlevel; echo "$(date +%F\ %T) stop, restored level $base" >> $LOG; exit 0' INT TERM
while :; do
  d=$(( $(cat $Z) / 100 ))
  want=$base
  [ $d -ge $T1 ] && want=1; [ $d -ge $T2 ] && want=2; [ $d -ge $T3 ] && want=3
  [ $want -lt $base ] && want=$base
  if [ $want -gt $lvl ]; then new=$want
  elif [ $want -lt $lvl ]; then
    case $lvl in 3) thr=$T3;; 2) thr=$T2;; 1) thr=$T1;; *) thr=0;; esac
    if [ $d -lt $((thr - HYS)) ]; then new=$((lvl - 1)); else new=$lvl; fi
  else new=$lvl; fi
  if [ $new -ne $lvl ]; then echo $new > $KD/max_pwrlevel; lvl=$new; echo "$(date '+%F %T') ddr=$d level -> $lvl (gpu $(cat $KD/gpuclk 2>/dev/null))" >> $LOG; fi
  sleep $PERIOD
done
