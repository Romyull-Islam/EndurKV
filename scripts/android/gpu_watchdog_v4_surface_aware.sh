#!/system/bin/sh
# GPU watchdog v4, surface-aware. The vendor Thermal Engine caps the Adreno 840
# clock from the skin and battery thermistors (from about 40 C skin, 37 C battery).
# This acts on the same signals a little earlier and holds an intermediate
# freq_table rung (>= 826 MHz), so the vendor cap does not need to go deeper.
# The gpuss zones are a backstop. Reduce-only, releases when skin and battery cool.
#
# Args: $1 log  $2 stop sentinel
LOG=${1:-/data/local/tmp/gpu_wd.log}
STOP=${2:-/data/local/tmp/gpu_wd.stop}
MAXCLK=/sys/kernel/gpu/gpu_max_clock
GZONES="36 37 38 39 40 41 42 43 44 45 46"    # gpuss subsystem zones (GPU-junction backstop; gpu_tmu absent post-reboot)
SZONES="55 56 57"                            # shell_front/frame/back (post-reboot zone IDs)
BATZ=93
# Adreno 840 freq_table subset (MHz), floor 826
TIERS="1200 1050 967 902 826"; NTIER=4
# Trip points (milli-C), about 1 C before the Thermal Engine. Override from the
# environment, for example:
#   LOW:  SHELL_HI=38500 BAT_HI=35000 SHELL_LO=36500 BAT_LO=34000
#   HIGH: SHELL_HI=40500 BAT_HI=37000 SHELL_LO=38500 BAT_LO=36000
SHELL_HI=${SHELL_HI:-39500}; BAT_HI=${BAT_HI:-36000}   # enter pre-region -> step down one rung
SHELL_LO=${SHELL_LO:-37500}; BAT_LO=${BAT_LO:-35000}   # both cooled -> release up one rung
GPU_CAP=90000                                # safety net on max gpuss zone; surface predictor acts first
DWELL=4
idx=0; capped=0; hold=0
# moving avg of shell (3 samples ~1.5s) to avoid single-sample jitter
s1=0; s2=0; s3=0
rm -f $STOP
echo "$(date) gpu-watchdog v4 SURFACE-PREDICTIVE shellHI=$SHELL_HI batHI=$BAT_HI shellLO=$SHELL_LO batLO=$BAT_LO" > $LOG
tier_clk(){ i=0; for c in $TIERS; do [ $i -eq $1 ] && { echo $c; return; }; i=$((i+1)); done; }
maxz(){ m=0; for z in $1; do t=$(cat /sys/class/thermal/thermal_zone$z/temp 2>/dev/null); [ -n "$t" ] && [ "$t" -gt "$m" ] && m=$t; done; echo $m; }
while [ ! -f $STOP ]; do
  g=$(maxz "$GZONES")                           # GPU-junction proxy (max gpuss zone)
  sh=$(maxz "$SZONES")
  bat=$(cat /sys/class/thermal/thermal_zone$BATZ/temp 2>/dev/null); bat=${bat:-0}
  s3=$s2; s2=$s1; s1=$sh; shavg=$(( (s1+s2+s3)/3 ))
  if [ $hold -gt 0 ]; then hold=$((hold-1)); else
    # backstop: GPU junction too hot -> step down regardless
    if [ $g -gt $GPU_CAP ] && [ $idx -lt $NTIER ]; then
      idx=$((idx+1)); nc=$(tier_clk $idx); echo $nc > $MAXCLK; hold=$DWELL; capped=1
      echo "$(date +%s) DOWN(tmu) idx=$idx clk=$nc tmu=$((g/1000))C shell=$((shavg/1000))C bat=$((bat/1000))C" >> $LOG
    # predictive: surface pre-region entered -> step down gradually
    elif [ $shavg -gt $SHELL_HI ] && [ $bat -gt $BAT_HI ] && [ $idx -lt $NTIER ]; then
      idx=$((idx+1)); nc=$(tier_clk $idx); echo $nc > $MAXCLK; hold=$DWELL; capped=1
      echo "$(date +%s) DOWN(surf) idx=$idx clk=$nc shell=$((shavg/1000))C bat=$((bat/1000))C gpu=$((g/1000))C" >> $LOG
    # release: both surface sensors cooled -> step up toward 1200
    elif [ $capped -eq 1 ] && [ $shavg -lt $SHELL_LO ] && [ $bat -lt $BAT_LO ] && [ $idx -gt 0 ]; then
      idx=$((idx-1)); nc=$(tier_clk $idx); echo $nc > $MAXCLK; hold=$DWELL
      [ $idx -eq 0 ] && capped=0
      echo "$(date +%s) UP   idx=$idx clk=$nc shell=$((shavg/1000))C bat=$((bat/1000))C cap=$capped" >> $LOG
    fi
  fi
  sleep 0.5
done
echo 1200 > $MAXCLK
echo "$(date) gpu-watchdog v4 stop, released 1200" >> $LOG
