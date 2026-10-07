#!/system/bin/sh
# DDR-temperature ladder for the GPU user cap, run on the phone as root, detached.
# Steps the cap down one level before the vendor limiter trips (it drops the Adreno
# 840 to 726 MHz at DDR ~64 C) and restores one level at a time with 1.5 C hysteresis.
# Writes max_pwrlevel, which the driver combines with the thermal cap by max(), so the
# vendor can still go lower. Never lifts a stricter cap that was already set.
#   ladder (DDR): >= 60.0 C level 1 (1050 MHz), >= 62.0 C level 2 (967), >= 63.5 C level 3 (902)
# Log: /data/local/tmp/endurkv/gpu_watchdog.log
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
