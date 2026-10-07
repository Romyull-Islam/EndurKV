#!/system/bin/sh
# Discharge cycle 5 starter, runs on the phone under su. Waits with charging on
# until SoC is above START_SOC (91%), then for the cable to be out with SoC at
# START_SOC or one below, resets the tables to the seed and execs the loop.
# Holds a wake lock while waiting so checks are not skipped in deep sleep.
ROOT=/data/local/tmp/endurkv; LOGF=$ROOT/discharge5.start.log
START_SOC=${START_SOC:-91}
MKEY=Llama-3.2-1B-Instruct-Q4_K_M
log(){ echo "$(date '+%F %T') $*" >> $LOGF; }
echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable
echo endurkv_start > /sys/power/wake_lock 2>/dev/null
log "arming: waiting for SoC > ${START_SOC}% (charging on)"
while :; do
  L=$(dumpsys battery 2>/dev/null | tr -d '\r' | awk '$1=="level:"{print $2}')
  log "arm check soc=${L:-?}"
  [ -n "$L" ] && [ "$L" -gt "$START_SOC" ] && break
  sleep 120
done
log "armed at ${L}%: waiting for SoC <= ${START_SOC}% with the cable out"
while :; do
  set -- $(dumpsys battery 2>/dev/null | tr -d '\r' | awk '$1=="level:"{l=$2} $1=="status:"{s=$2} /USB powered:/{u=$3} END{print l, s, u}')
  SOC=$1; STAT=$2; USB=$3
  log "soc=${SOC:-?} status=${STAT:-?} usb=${USB:-?}"
  [ -n "$SOC" ] && [ "$USB" = "false" ] && [ "$SOC" -le "$START_SOC" ] && [ "$SOC" -ge $((START_SOC - 1)) ] && break
  sleep 60
done
cp $ROOT/ukv_sched_table.seed.txt $ROOT/ukv_sched_table.txt; chmod 666 $ROOT/ukv_sched_table.txt
rm -f $ROOT/ukv_lever_bias.txt $ROOT/tables/$MKEY.txt $ROOT/tables/$MKEY.bias.txt
echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel
echo endurkv_start > /sys/power/wake_unlock 2>/dev/null
log "start at ${SOC}% (usb $USB): tables reset to the seed, launching the loop"
cd $ROOT
OUTD=$ROOT/discharge5 STOP_SOC=${STOP_SOC:-12} exec sh $ROOT/phone_discharge_loop.sh
