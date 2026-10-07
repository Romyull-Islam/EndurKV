#!/system/bin/sh
# Cooling gate for timed and energy-measured phone runs: DDR below 36 C, battery
# below 34 C, CPU cores at most 45 C. Charging stays off while cooling because the
# charge current heats the phone. The phone runs off the USB rail meanwhile.
# Prints "cool ddr=<n> batt=<n> cpu=<n>" on success and "cool-timeout ..." on failure.
# Callers must check the printed marker, since exit codes do not survive adb shell.
#
# Usage:  . cool_gate.sh ; cool_ddr36   (POSIX sh, under su)

COOL_DDR_MAX=${COOL_DDR_MAX:-35}       # DDR  <= 35  (i.e. below 36)
COOL_BAT_MAX=${COOL_BAT_MAX:-33}       # batt <= 33  (i.e. below 34)
COOL_CPU_MAX=${COOL_CPU_MAX:-45}       # CPU cores <= 45 -- a BACKSTOP, not the primary gate.
# CPU cores track DDR closely, so the 45 C core limit only binds when the DDR gate
# did not really hold. A lower core limit adds a lot of cooling time per cell.
COOL_MAX_ITERS=${COOL_MAX_ITERS:-540}  # 540 * 10 s = 90 min of patience

cool_ddr36() {
  echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable 2>/dev/null
  DZ=; BZ=
  CZS=""
  for z in /sys/class/thermal/thermal_zone*; do t=$(cat $z/type 2>/dev/null)
    [ "$t" = ddr ] && DZ=$z/temp
    [ "$t" = battery ] && BZ=$z/temp
    case "$t" in *trip*) : ;; cpu-*|cpullc-*) CZS="$CZS $z/temp";; esac
  done
  if [ -z "$DZ" ] || [ -z "$BZ" ]; then
    echo "cool-timeout no-sensor ddr_zone=$DZ bat_zone=$BZ"; return 1
  fi
  i=0
  while [ $i -lt $COOL_MAX_ITERS ]; do
    dd=$(( $(cat $DZ)/1000 )); b=$(( $(cat $BZ)/1000 ))
    c=0
    for cz in $CZS; do v=$(cat $cz 2>/dev/null); v=$((${v:-0}/1000)); [ $v -gt $c ] && c=$v; done
    if [ $dd -le $COOL_DDR_MAX ] && [ $b -le $COOL_BAT_MAX ] && { [ -z "$CZS" ] || [ $c -le $COOL_CPU_MAX ]; }; then
      echo "cool ddr=$dd batt=$b cpu=$c"; return 0
    fi
    # progress every 60 s so a long cool does not look like a hang
    [ $((i % 6)) -eq 0 ] && echo "cooling ddr=$dd batt=$b cpu=$c (need <=$COOL_DDR_MAX / <=$COOL_BAT_MAX / <=$COOL_CPU_MAX)"
    sleep 10; i=$((i+1))
  done
  echo "cool-timeout ddr=$dd batt=$b"; return 1
}

charging_restore() { echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable 2>/dev/null; }

# warm_up: run before the first cell of a campaign. The gate is only an upper bound,
# and a first cell started from an idle, colder phone prefills faster than the rest.
# About 60 s of load brings the cores to the steady state later cells start from.
warm_up() {
    _target=${1:-34}
    _i=0
    while [ $_i -lt 30 ]; do
        _c=0
        for _z in /sys/class/thermal/thermal_zone*; do
            _t=$(cat $_z/type 2>/dev/null)
            case "$_t" in *trip*) continue ;; cpu-*|cpullc-*) ;; *) continue ;; esac
            _v=$(cat $_z/temp 2>/dev/null); _v=$((${_v:-0}/1000))
            [ $_v -gt $_c ] && _c=$_v
        done
        [ $_c -ge $_target ] && { echo "warm cpu=$_c"; return 0; }
        # short busy loop on all cores
        for _k in 1 2 3 4 5 6; do (while [ $(( $(date +%s) % 3 )) -ne 0 ]; do :; done) & done
        wait
        _i=$((_i+1))
    done
    echo "warm-timeout cpu=$_c"; return 0
}
