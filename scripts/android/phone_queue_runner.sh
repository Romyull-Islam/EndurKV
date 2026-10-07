#!/system/bin/sh
# phone_queue_runner.sh: runs a list of eviction_bench cells on the phone itself, so a lost
# adb link does not stop the campaign. The host only pushes the list and pulls results.
# Usage (under su): setsid nohup sh phone_queue_runner.sh <cells.txt> <outdir> >/dev/null 2>&1 &
# Cell line: TAG|KIND|BINDIR|eviction_bench args (without the --out-* flags)
#   gpu:  cooling gate + DDR settle check, sensors at 2 Hz, pinned, like the host GPU campaigns
#   niah: cooling gate, sensors at 5 Hz, timeout 3600 s, like run_niah_sllm2004.sh
#   cpu:  cooling gate, sensors at 2 Hz, timeout 7200 s (CPU speed cells)
#   lb:   no cooling gate, timeout 1200 s, like run_longbench_wide.sh (accuracy only)
# A cell with OUT/TAG.done is skipped, so the runner can be restarted at any time.
# Before each cell: stop if /data has under 5 GB free, charge to 80% if the battery is under 30%.
Q=$1; OUT=$2; mkdir -p $OUT
. /data/local/tmp/endurkv/scripts/cool_gate.sh
log(){ echo "[$(date '+%F %T')] $*" >> $OUT/run.log; }
for z in /sys/class/thermal/thermal_zone*; do
  t=$(cat $z/type 2>/dev/null); [ "$t" = ddr ] && DZ=$z/temp; [ "$t" = battery ] && BZ=$z/temp
done
echo endurkv_queue > /sys/power/wake_lock 2>/dev/null
echo $$ > $OUT/runner.pid
log "runner start pid $$, list $Q"

settle(){   # cooling gate, wait until DDR stops drifting, then confirm the limits (3 tries)
  for a in 1 2 3; do
    cool_ddr36 >> $OUT/run.log 2>&1
    prev=999; same=0; i=0
    while [ $i -lt 90 ]; do
      d=$(( $(cat $DZ) / 100 )); df=$((d - prev)); [ $df -lt 0 ] && df=$((-df))
      if [ $df -le 3 ]; then same=$((same+1)); else same=0; fi
      [ $same -ge 3 ] && break
      prev=$d; sleep 10; i=$((i+1))
    done
    dd=$(( $(cat $DZ) / 1000 )); bb=$(( $(cat $BZ) / 1000 ))
    log "  post-settle ddr=${dd}C batt=${bb}C"
    [ $dd -le 35 ] && [ $bb -le 33 ] && return 0
  done
  return 1
}

guards(){   # before every cell: storage, battery charge, and heat for cells without a gate
  free_kb=$(df /data | awk 'NR==2{print $4}')
  if [ "${free_kb:-0}" -lt 5242880 ]; then
    log "STORAGE-LOW: ${free_kb} KB free on /data, stopping (restart resumes)"; return 1
  fi
  soc=$(dumpsys battery 2>/dev/null | awk '$1=="level:"{print $2}')
  if [ -n "$soc" ] && [ "$soc" -lt 30 ]; then
    log "battery ${soc}% < 30%: charging to 80%"
    echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable 2>/dev/null
    while :; do
      soc=$(dumpsys battery 2>/dev/null | awk '$1=="level:"{print $2}')
      [ -n "$soc" ] && [ "$soc" -ge 80 ] && break
      sleep 300
    done
    log "battery ${soc}%: charged, continuing"
  fi
  return 0
}
hot(){   # heat backstop for cells that have no cooling gate
  dd=$(( $(cat $DZ) / 1000 )); bb=$(( $(cat $BZ) / 1000 ))
  [ $dd -gt 45 ] || [ $bb -gt 38 ]
}

while IFS='|' read -r TAG KIND BIN ARGS; do
  case "$TAG" in ''|\#*) continue;; esac
  [ -f $OUT/$TAG.done ] && continue
  guards || { STOPPED=1; break; }
  echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable 2>/dev/null
  case $KIND in
    gpu)
      log "cooling for $TAG"
      settle || { log "SKIP-HOT $TAG"; echo hot > $OUT/$TAG.done; continue; }
      echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable 2>/dev/null
      sh /data/local/tmp/sample_sensors.sh --out $OUT/$TAG.sensors.csv --hz 2 >/dev/null 2>&1 &
      SP=$!
      log "running $TAG"
      ( cd $BIN && LD_LIBRARY_PATH=$BIN taskset f0 nice -n -20 ./eviction_bench $ARGS \
          --out-meta $OUT/$TAG.json --out-gen $OUT/$TAG.gen --out-csv $OUT/$TAG.csv \
          > /dev/null 2> $OUT/$TAG.err < /dev/null )
      kill $SP 2>/dev/null; pkill -f sample_sensors 2>/dev/null ;;
    niah)
      log "cooling for $TAG"
      CG=$(cool_ddr36); echo "$CG" | tail -1 >> $OUT/run.log
      case "$CG" in *"cool ddr="*) : ;; *) log "SKIP-HOT $TAG"; echo hot > $OUT/$TAG.done; continue ;; esac
      echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable 2>/dev/null
      sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $OUT/$TAG.sensors.csv --hz 5 >/dev/null 2>&1 &
      SP=$!
      log "running $TAG"
      LD_LIBRARY_PATH=$BIN timeout 3600 $BIN/eviction_bench $ARGS \
          --out-meta $OUT/$TAG.json --out-csv /dev/null --out-gen $OUT/$TAG.gen \
          > /dev/null 2> $OUT/$TAG.err < /dev/null
      kill $SP 2>/dev/null; pkill -f sample_sensors 2>/dev/null ;;
    cpu)
      log "cooling for $TAG"
      CG=$(cool_ddr36); echo "$CG" | tail -1 >> $OUT/run.log
      case "$CG" in *"cool ddr="*) : ;; *) log "SKIP-HOT $TAG"; echo hot > $OUT/$TAG.done; continue ;; esac
      echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable 2>/dev/null
      sh /data/local/tmp/sample_sensors.sh --out $OUT/$TAG.sensors.csv --hz 2 >/dev/null 2>&1 &
      SP=$!
      log "running $TAG"
      LD_LIBRARY_PATH=$BIN timeout 7200 $BIN/eviction_bench $ARGS \
          --out-meta $OUT/$TAG.json --out-gen $OUT/$TAG.gen --out-csv $OUT/$TAG.csv \
          > /dev/null 2> $OUT/$TAG.err < /dev/null
      kill $SP 2>/dev/null ;;
    lb)
      if hot; then log "hot before $TAG: cooling"; cool_ddr36 >> $OUT/run.log 2>&1; fi
      log "running $TAG"
      LD_LIBRARY_PATH=$BIN timeout 1200 $BIN/eviction_bench $ARGS \
          --out-meta $OUT/$TAG.json --out-gen $OUT/$TAG.gen --out-csv /dev/null \
          > /dev/null 2> $OUT/$TAG.err < /dev/null ;;
    *) log "unknown kind $KIND for $TAG"; continue ;;
  esac
  tps=$(grep -o '"decode_tps": *[0-9.]*' $OUT/$TAG.json 2>/dev/null | grep -o '[0-9.]*$')
  log "  [$TAG] tok/s=${tps:-?}"
  echo done > $OUT/$TAG.done
done < $Q

charging_restore
echo endurkv_queue > /sys/power/wake_unlock 2>/dev/null
if [ "${STOPPED:-0}" = 1 ]; then log "STOPPED"; echo STOPPED > $OUT/STOPPED
else log "ALL_DONE"; echo ALL_DONE > $OUT/ALL_DONE; fi
