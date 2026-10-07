#!/system/bin/sh
# phone_discharge_start3.sh: runs on the phone under su, detached, for a fully unplugged discharge
# run. Holds a wake lock so no check is skipped in deep sleep, waits until the cable is out and
# SoC is at or below START_SOC, resets the cost tables to the seed, then execs the loop.
ROOT=/data/local/tmp/endurkv; LOGF=$ROOT/discharge3.start.log
START_SOC=${START_SOC:-91}
MKEY=Llama-3.2-1B-Instruct-Q4_K_M
log(){ echo "$(date '+%F %T') $*" >> $LOGF; }
echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable
echo endurkv_start > /sys/power/wake_lock 2>/dev/null
log "waiting for SoC <= ${START_SOC}% with the cable out"
while :; do
  set -- $(dumpsys battery 2>/dev/null | tr -d '\r' | awk '$1=="level:"{l=$2} $1=="status:"{s=$2} /USB powered:/{u=$3} END{print l, s, u}')
  SOC=$1; STAT=$2; USB=$3
  log "soc=${SOC:-?} status=${STAT:-?} usb=${USB:-?}"
  [ -n "$SOC" ] && [ "$USB" = "false" ] && [ "$SOC" -le "$START_SOC" ] && break
  sleep 60
done
cp $ROOT/ukv_sched_table.seed.txt $ROOT/ukv_sched_table.txt; chmod 666 $ROOT/ukv_sched_table.txt
rm -f $ROOT/ukv_lever_bias.txt $ROOT/tables/$MKEY.txt $ROOT/tables/$MKEY.bias.txt
echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel
echo endurkv_start > /sys/power/wake_unlock 2>/dev/null
log "start at ${SOC}% (usb $USB): tables reset to the seed, launching the loop"
cd $ROOT
OUTD=$ROOT/discharge3 STOP_SOC=${STOP_SOC:-12} exec sh $ROOT/phone_discharge_loop.sh
