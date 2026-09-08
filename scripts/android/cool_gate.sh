#!/system/bin/sh
# Canonical cooling gate for ALL timed / energy-measured phone runs.
#
# Rule (user-mandated 2026-07-25; tightened 2026-07-26; tightened again 2026-07-27):
#   DDR     < 36 C  -> enforced as <= 35
#   battery < 34 C  -> enforced as <= 33   (was <= 34, which is not "below 34")
# Charging stays OFF while cooling: the charge current is itself a heat source and is
# the lever that makes DDR fall. The phone runs off the USB rail (battery current_now
# reads 0 with USB online), so charging-off does not drain it during long campaigns.
#
# CONTRACT WITH THE CALLER (changed 2026-07-27):
#   success -> prints a line containing "cool ddr=<n> batt=<n>"
#   failure -> prints "cool-timeout ..." and returns 1
# The caller MUST check for the success marker in the OUTPUT and refuse to run the
# cell otherwise. Exit codes do not survive `adb shell` reliably, so the printed
# marker is the contract. Until 2026-07-27 the caller ignored the result entirely,
# so a 40-minute cooling failure silently produced a cell measured from a HOT start
# -- exactly the contamination this gate exists to prevent.
#
# Usage:  . cool_gate.sh ; cool_ddr36   (POSIX sh, under su)

COOL_DDR_MAX=${COOL_DDR_MAX:-35}       # DDR  <= 35  (i.e. below 36)
COOL_BAT_MAX=${COOL_BAT_MAX:-33}       # batt <= 33  (i.e. below 34)
COOL_CPU_MAX=${COOL_CPU_MAX:-45}       # CPU cores <= 45 -- a BACKSTOP, not the primary gate.
# Measured over 33 cells: CPU tracks DDR at r=+0.92 with a threshold. When the DDR
# gate holds (DDR<=35, reading ~36.6 at cell start) the cores land at 36.6-37.2 C
# on their own and this check never binds. Every cell that started at 48-51.5 C had
# DDR at 38-41 C, i.e. the DDR gate had NOT held. So 45 costs no extra cooling time
# in the normal case and catches exactly the pathological case. An earlier 36 C
# setting was a primary gate in disguise: it added ~37 min per cell for nothing.
COOL_MAX_ITERS=${COOL_MAX_ITERS:-540}  # 540 * 10 s = 90 min of patience
# CPU-CORE GATE ADDED 2026-08-26. The gate held DDR and battery but never the
# cores, and DDR cools FASTER than the cores do. Measured consequence on the
# Phi-3 CPU campaign: cell #1 (vanilla, run after hours of idle) started at
# 30.1 C cores and prefilled in 381 s; cells #2/#3/#8 all passed the same
# DDR/battery gate at 36.2-39.6 C cores and prefilled in 471/467/472 s -- an
# apparent "+23% prefill cost of eviction" that was entirely a 6 C colder start.
# Arithmetic confirms it: the muKV side node is 23.7 GFLOP = 0.42% of prefill's
# own attention, ~5 s at worst, not 90 s. Without this gate the first cell of
# any campaign is measured on a different machine than the rest.

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
    # progress every 60 s so a long cool is visible instead of looking like a hang
    [ $((i % 6)) -eq 0 ] && echo "cooling ddr=$dd batt=$b cpu=$c (need <=$COOL_DDR_MAX / <=$COOL_BAT_MAX / <=$COOL_CPU_MAX)"
    sleep 10; i=$((i+1))
  done
  echo "cool-timeout ddr=$dd batt=$b"; return 1
}

charging_restore() { echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable 2>/dev/null; }

# ---------------------------------------------------------------------------
# warm_up -- run before the FIRST cell of a campaign.  (added 2026-08-26)
#
# The gate is an UPPER bound, so it cannot stop a cell from starting anomalously
# COLD. That is what contaminated the Phi-3 CPU prefill column: cell #1 began at
# 30.1 C cores after hours of idle and prefilled in 381 s, while cells #2/#3/#8
# started at 36.2-39.6 C and took 471/467/472 s. The gap was read as a "+23%
# prefill cost of eviction"; it was a 6 C colder machine.
#
# Bringing the cores to the same steady state every campaign reaches after its
# first cell makes cell #1 comparable to the rest. Cheap: ~60 s of load.
# ---------------------------------------------------------------------------
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
