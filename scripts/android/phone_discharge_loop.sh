#!/system/bin/sh
# phone_discharge_loop.sh: runs on the phone under su, detached. Sends back-to-back requests
# through ukv_sched.sh on a real discharge with charging off and a light cool gate between
# requests (DDR <= 42 C, battery <= 36 C). Stops at STOP_SOC and turns charging back on.
# On the cable USB supplies most of the power, so the USB-powered flag is logged per request.
ROOT=/data/local/tmp/endurkv; P=$ROOT/corpora/prompt_12k.txt; OUTD=${OUTD:-$ROOT/discharge}; mkdir -p $OUTD
# MODEL (optional): a gguf name under $ROOT/models, passed to ukv_sched.sh --model (per-model cost table)
CSV=$OUTD/timeline.csv; RLOG=$OUTD/run.log; STOP_SOC=${STOP_SOC:-12}; KEEP_TABLE=${KEEP_TABLE:-0}
# the floor can be changed while running: echo N > $OUTD/STOP_SOC
stopsoc(){ cat $OUTD/STOP_SOC 2>/dev/null || echo $STOP_SOC; }
log(){ echo "$(date '+%F %T') $*" >> $RLOG; }
COOL_DDR_MAX=42; COOL_BAT_MAX=36; COOL_CPU_MAX=50; COOL_MAX_ITERS=120
. $ROOT/scripts/cool_gate.sh
USB_CUT=${USB_CUT:-0}; ICL=/sys/class/power_supply/usb/input_current_limit; ICL0=$(cat $ICL 2>/dev/null)
SAVED=""; for p in /sys/devices/system/cpu/cpufreq/policy*; do SAVED="$SAVED $(basename $p):$(cat $p/scaling_min_freq):$(cat $p/scaling_max_freq)"; done
restore(){ for s in $SAVED; do pol=${s%%:*}; r=${s#*:}; mn=${r%%:*}; mx=${r#*:}; echo $mx > /sys/devices/system/cpu/cpufreq/$pol/scaling_max_freq; echo $mn > /sys/devices/system/cpu/cpufreq/$pol/scaling_min_freq; done
  echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable
  [ "$USB_CUT" = 1 ] && [ -n "$ICL0" ] && echo $ICL0 > $ICL 2>/dev/null; echo endurkv_discharge > /sys/power/wake_unlock 2>/dev/null
  log "restored caps, charging on, wake lock released, usb input limit ${ICL0:-untouched}"; }
trap 'restore; log "interrupted"; echo INTERRUPTED > $OUTD/DONE; exit 1' INT TERM
pin(){ echo 1785600 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; echo 1785600 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; echo 1497600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq; echo 1497600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq; }
bat(){ dumpsys battery 2>/dev/null | tr -d '\r' | awk '$1=="level:"{l=$2} $1=="status:"{s=$2} /USB powered:/{u=$3} END{print l, s, u}'; }
temps(){ echo $(( $(cat /sys/class/thermal/thermal_zone47/temp)/1000 )) $(( $(cat /sys/class/thermal/thermal_zone93/temp)/1000 )); }
[ -f $CSV ] || echo "ts,n,soc,status,usb_powered,tier,lever,bias_before,bias_after,plan,K,gpu_mhz,decode_mhz,nout_cap,np,ns,pred_J,pred_s,meas_J,meas_s,pre_J,dec_J,tps,ddr_start,batt_start,ddr_end,batt_end,loop" > $CSV
rm -f $OUTD/DONE
[ "$KEEP_TABLE" = 1 ] || { [ -f $ROOT/ukv_sched_table.seed.txt ] && cp $ROOT/ukv_sched_table.seed.txt $ROOT/ukv_sched_table.txt; rm -f $ROOT/ukv_lever_bias.txt; }
echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable
# Hold a wake lock for the whole run, since deep sleep inside a request breaks its energy
# window. Released in restore().
echo endurkv_discharge > /sys/power/wake_lock 2>/dev/null
# USB_CUT=1: also cut the USB input current so the battery alone carries the load while the data link stays
[ "$USB_CUT" = 1 ] && { echo 0 > $ICL 2>/dev/null; log "usb input_current_limit $ICL0 -> $(cat $ICL 2>/dev/null)"; }
n=$(ls -d $OUTD/dis_* 2>/dev/null | wc -l); fails=0
set -- $(bat); log "start: SoC $1% status $2 usb $3; stop at $(stopsoc)%"
while :; do
  set -- $(bat); SOC=$1; STAT=$2; USB=$3
  [ -z "$SOC" ] && { fails=$((fails+1)); log "battery read failed ($fails)"; [ $fails -ge 5 ] && break; sleep 60; continue; }
  [ "$SOC" -le "$(stopsoc)" ] && { log "SoC $SOC% <= $(stopsoc)%: stop, everything off"; break; }
  cool_ddr36 >> $RLOG 2>&1
  # cool_ddr36 ends with charging_restore, so charging has to be switched off again here,
  # after the gate and before the request, or every request runs on the charger
  echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable
  pin
  n=$((n+1)); TAG=$(printf "dis_%03d" $n); mkdir -p $OUTD/$TAG
  set -- $(temps); DDR0=$1; BAT0=$2
  log "$TAG: SoC ${SOC}% status $STAT usb $USB ddr ${DDR0}C batt ${BAT0}C"
  rm -rf $ROOT/sched/$TAG
  sh $ROOT/ukv_sched.sh --prompt $P --tag $TAG --ignore-eos ${MODEL:+--model $MODEL} > $OUTD/$TAG/sched.stderr 2>&1
  echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk
  cp $ROOT/sched/$TAG/meta.json $ROOT/sched/$TAG/sensors.csv $ROOT/sched/$TAG/err $OUTD/$TAG/ 2>/dev/null
  L=$(grep "tag=$TAG " $ROOT/ukv_sched.log | tail -1)
  set -- $(temps); DDR1=$1; BAT1=$2
  if [ -f $OUTD/$TAG/meta.json ] && [ -n "$L" ]; then
    fails=0
    g(){ echo "$L" | sed -n "s/.* $1=\([^ ]*\).*/\1/p" | head -1; }
    B=$(g bias); BB=${B%%->*}; BA=${B##*->}; LOOP=${L##*; }
    echo "$(date +%F_%T),$n,$SOC,$STAT,$USB,$(g tier),$(g lever),$BB,$BA,$(g plan),$(g K),$(g gpu_mhz),$(g decode_mhz),$(g nout_cap),$(g np),$(g ns),$(g pred_J),$(g pred_s),$(g meas_J),$(g meas_s),$(g pre_J),$(g dec_J),$(g tps),$DDR0,$BAT0,$DDR1,$BAT1,\"$LOOP\"" >> $CSV
    log "  -> $(g plan) cap $(g nout_cap): $(g meas_J) J / $(g meas_s) s (pred $(g pred_J) J / $(g pred_s) s); $LOOP"
  else
    fails=$((fails+1)); log "  $TAG FAILED ($fails in a row)"; [ $fails -ge 3 ] && { log "three failures in a row: stop"; break; }
  fi
  # On USB power the battery barely moves, so run one request per 20 min. Off the cable, or with
  # FORCE_BATT=1 (cable in, charging off, pack still carries part of the load), run back to back.
  echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable
  set -- $(bat); [ "$3" = "true" ] && [ "${FORCE_BATT:-0}" != 1 ] && { log "usb powered: sparse mode, next request in 20 min"; sleep 1200; }
done
restore; echo DONE > $OUTD/DONE; log "DISCHARGE_DONE"
