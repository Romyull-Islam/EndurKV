#!/system/bin/sh
# phone_discharge_start.sh -- runs ON THE PHONE under su, detached: waits for the battery to reach
# the start level with charging on (any charger; the laptop port left the phone at "Not charging"
# on 2026-09-25), then resets the scheduler state to the cable seed and execs the discharge loop.
# The cable can be pulled any time after the loop has started; the loop logs the USB flag itself.
# Fallback: if the cable is already out and the level is at least MIN_SOC_UNPLUGGED, start anyway.
ROOT=/data/local/tmp/endurkv; LOGF=$ROOT/discharge2.start.log
START_SOC=${START_SOC:-90}; MIN_SOC_UNPLUGGED=${MIN_SOC_UNPLUGGED:-85}
MKEY=Llama-3.2-1B-Instruct-Q4_K_M
log(){ echo "$(date '+%F %T') $*" >> $LOGF; }
echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable
log "waiting for SoC >= ${START_SOC}% (or cable out and >= ${MIN_SOC_UNPLUGGED}%)"
while :; do
  set -- $(dumpsys battery 2>/dev/null | tr -d '\r' | awk '$1=="level:"{l=$2} $1=="status:"{s=$2} /USB powered:/{u=$3} END{print l, s, u}')
  SOC=$1; STAT=$2; USB=$3
  log "soc=${SOC:-?} status=${STAT:-?} usb=${USB:-?}"
  [ -n "$SOC" ] && [ "$SOC" -ge "$START_SOC" ] && break
  [ -n "$SOC" ] && [ "$USB" = "false" ] && [ "$SOC" -ge "$MIN_SOC_UNPLUGGED" ] && { log "cable out at ${SOC}%: starting anyway"; break; }
  sleep 60
done
cp $ROOT/ukv_sched_table.seed.txt $ROOT/ukv_sched_table.txt; chmod 666 $ROOT/ukv_sched_table.txt
rm -f $ROOT/ukv_lever_bias.txt $ROOT/tables/$MKEY.txt $ROOT/tables/$MKEY.bias.txt
echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel
log "start at ${SOC}% (usb $USB): tables reset to the seed, launching the loop"
cd $ROOT
OUTD=$ROOT/discharge2 STOP_SOC=${STOP_SOC:-12} exec sh $ROOT/phone_discharge_loop.sh
