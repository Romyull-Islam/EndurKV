#!/system/bin/sh
# preempt_throttle_watchdog v2: preemptive CPU clock caps from battery and skin temperature.
# Each ladder caps both clusters at 1497/1382/1267/1132/1017 MHz as temperature climbs
# toward the vendor deep throttle. The tighter of the two ladders wins.
# Sampled at 2 Hz. Fixed-point math (milli-degC) so it runs under Android /system/bin/sh.
#
# Args:
#   $1  log file path
#   $2  stop sentinel file path (loop exits when this file appears)
#
# Requires root (to write cpufreq scaling_max_freq).

set -u

LOG=${1:-/sdcard/preempt_throttle_v2.log}
STOP=${2:-/sdcard/preempt_throttle_v2.stop}

# Thermal zones (OnePlus 15, Android 16) are resolved by type name because zone
# numbers change across reboots and hardcoded ones point at the wrong sensor.
resolve_zone() {  # $1 = thermal-zone type name; echoes zone dir, empty if none
    local z
    for z in /sys/class/thermal/thermal_zone*; do
        [ "$(cat "$z/type" 2>/dev/null)" = "$1" ] && { echo "$z"; return 0; }
    done
    return 1
}
DDR_ZONE=$(resolve_zone ddr)
CPU_ZONE=$(resolve_zone cpu-1-0-0)
SOC_ZONE=$(resolve_zone sys-therm-2)
SKIN_ZONE=$(resolve_zone shell_front)
BAT_ZONE=$(resolve_zone battery)

# Optional overrides for unit tests / dry runs.
DDR_ZONE=${DDR_ZONE_OVERRIDE:-$DDR_ZONE}
CPU_ZONE=${CPU_ZONE_OVERRIDE:-$CPU_ZONE}
SOC_ZONE=${SOC_ZONE_OVERRIDE:-$SOC_ZONE}
SKIN_ZONE=${SKIN_ZONE_OVERRIDE:-$SKIN_ZONE}
BAT_ZONE=${BAT_ZONE_OVERRIDE:-$BAT_ZONE}

# Threat-tier frequency caps (kHz). Tier 1 is never reached (see TH_T1).
F_MAX=1632000
F_NUDGE=1574400   # tier 1 (unreachable in minimum-energy mode)
F_MILD=1497600    # tier 2
F_MOD=1382400     # tier 3
F_STRONG=1267200  # tier 4

# Per-sensor warn/crit thresholds in milli-degC, the raw thermal_zone temp units.

# DDR: kernel cliff at 65 C, so crit is 0.5 C and warn 2 C below it.
DDR_WARN_MC=63000
DDR_CRIT_MC=64500

# CPU big cluster (cpu-1-0-0).
CPU_WARN_MC=65500
CPU_CRIT_MC=67000

# SoC sys-therm-2: advisory, not a known cliff. crit is the pre-throttle median.
SOC_WARN_MC=44400
SOC_CRIT_MC=49400

# Skin (shell_front).
SKIN_WARN_MC=42000
SKIN_CRIT_MC=42700

# Battery temperature.
BAT_WARN_MC=39000
BAT_CRIT_MC=39800

# Battery current drop from baseline (mA). A sharp drop is the power-gating
# signature seen in pre-throttle samples.
BAT_CUR_DROP_WARN_MA=128       # p10 of drops -> warn (rising threat)
BAT_CUR_DROP_CRIT_MA=392       # p50 of drops -> crit

# Sample period
SAMPLE_DT=0.5
# dumpsys battery is slow (~200-400 ms). Poll once every BAT_EVERY ticks.
BAT_EVERY=4   # 4 * 0.5 s = 2 s

# Hysteresis in milli-threat units (0.10 = 100).
HYS_MTHREAT=100

# Tier thresholds in milli-threat units. Tier 1 is disabled, so the ladder goes
# from tier 0 straight to tier 2 at 0.50.
TH_T1=1001   # unreachable (tier 1 NUDGE disabled in minimum-energy mode)
TH_T2=500    # 0.50
TH_T3=750    # 0.75
TH_T4=900    # 0.90

# Helpers

read_zone_mc() {
    # $1 = zone dir. Prints temp in milli-degC (raw), 0 if unreadable.
    local p=$1
    if [ -r "$p/temp" ]; then
        cat "$p/temp" 2>/dev/null
    else
        echo 0
    fi
}

# fmt_c <milli-degC> prints XX.X
fmt_c() {
    # awk float formatting, only called when logging.
    awk -v v="$1" 'BEGIN { printf "%.1f", v/1000.0 }'
}

# threat_mc <T_mc> <warn_mc> <crit_mc>: threat in milli-units (0..1000),
# clamp01((T - warn) / (crit - warn))
threat_mc() {
    local t=$1 w=$2 c=$3
    if [ "$t" -le "$w" ]; then
        echo 0
        return
    fi
    if [ "$t" -ge "$c" ]; then
        echo 1000
        return
    fi
    # (t - w) / (c - w) * 1000, integer math.
    local num=$(( (t - w) * 1000 ))
    local den=$(( c - w ))
    [ "$den" -le 0 ] && { echo 0; return; }
    echo $(( num / den ))
}

# threat_drop <drop_ma> <warn_ma> <crit_ma>: milli-threat (0..1000)
threat_drop() {
    local d=$1 w=$2 c=$3
    # drop may be negative (current rose). Clamp to 0.
    if [ "$d" -le "$w" ]; then echo 0; return; fi
    if [ "$d" -ge "$c" ]; then echo 1000; return; fi
    local num=$(( (d - w) * 1000 ))
    local den=$(( c - w ))
    [ "$den" -le 0 ] && { echo 0; return; }
    echo $(( num / den ))
}

# Cap both clusters. The workload runs on cpu2-5 (policy0) and cpu6-7 (policy6),
# so capping only cpu6/7 leaves 4 of 6 working cores at full clock.
# Per-cluster hardware maxima are read once at startup.
PERF_MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null)
PRIME_MAX=$(cat /sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_max_freq 2>/dev/null)
set_freq() {
    # $1 = cap in kHz, or the literal MAX to release both clusters to vendor DVFS
    local f=$1
    if [ "$f" = "MAX" ]; then
        echo "$PERF_MAX"  > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null
        echo "$PRIME_MAX" > /sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq 2>/dev/null
    else
        echo "$f" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null
        echo "$f" > /sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq 2>/dev/null
    fi
}

apply_tier() {
    case "$1" in
        0) set_freq MAX         ;;
        1) set_freq "$F_NUDGE"  ;;
        2) set_freq "$F_MILD"   ;;
        3) set_freq "$F_MOD"    ;;
        4) set_freq "$F_STRONG" ;;
    esac
}

# Per-cluster cap. Tier 0 restores the cluster max, tiers 1..4 = 1632/1497/1382/1267 MHz.
# $1=cpu0|cpu6 $2=tier $3=cluster_max
apply_cluster() {
    local f
    case "$2" in
        0) f=$3          ;;
        1) f=$F_MAX      ;;   # 1632 (gentle first step)
        2) f=$F_MILD     ;;   # 1497
        3) f=$F_MOD      ;;   # 1382
        4) f=$F_STRONG   ;;   # 1267
    esac
    echo "$f" > "/sys/devices/system/cpu/$1/cpufreq/scaling_max_freq" 2>/dev/null
}

# Battery and skin ladders. The vendor holds 1498 MHz, then deep-throttles both
# clusters to about 883 MHz when the battery reaches 50.0 C. The ladder stays idle
# until L0, then caps at 1497/1382/1267/1132/1017 MHz at L0 +0/+1.0/+1.5/+2.0/+2.5 C,
# ending above the 883 floor. Default anchors: battery 47.0 C, skin 50.0 C.
# Anchors are env-overridable, e.g. BAT_L0=35000 SKIN_L0=39500 for an early ladder.
BAT_L0=${BAT_L0:-47000}; BAT_L1=$((BAT_L0+1000)); BAT_L2=$((BAT_L0+1500)); BAT_L3=$((BAT_L0+2000)); BAT_L4=$((BAT_L0+2500))
SKIN_L0=${SKIN_L0:-50000}; SKIN_L1=$((SKIN_L0+1000)); SKIN_L2=$((SKIN_L0+1500)); SKIN_L3=$((SKIN_L0+2000)); SKIN_L4=$((SKIN_L0+2500))
perf_bat_cap() {
    if   [ "$1" -ge $BAT_L4 ]; then echo 1017600
    elif [ "$1" -ge $BAT_L3 ]; then echo 1132800
    elif [ "$1" -ge $BAT_L2 ]; then echo 1267200
    elif [ "$1" -ge $BAT_L1 ]; then echo 1382400
    elif [ "$1" -ge $BAT_L0 ]; then echo 1497600
    else echo "$PERF_MAX" ; fi
}
prime_bat_cap() {
    if   [ "$1" -ge $BAT_L4 ]; then echo 1017600
    elif [ "$1" -ge $BAT_L3 ]; then echo 1132800
    elif [ "$1" -ge $BAT_L2 ]; then echo 1267200
    elif [ "$1" -ge $BAT_L1 ]; then echo 1382400
    elif [ "$1" -ge $BAT_L0 ]; then echo 1497600
    else echo "$PRIME_MAX" ; fi
}
perf_skin_cap() {
    if   [ "$1" -ge $SKIN_L4 ]; then echo 1017600
    elif [ "$1" -ge $SKIN_L3 ]; then echo 1132800
    elif [ "$1" -ge $SKIN_L2 ]; then echo 1267200
    elif [ "$1" -ge $SKIN_L1 ]; then echo 1382400
    elif [ "$1" -ge $SKIN_L0 ]; then echo 1497600
    else echo "$PERF_MAX" ; fi
}
prime_skin_cap() {
    if   [ "$1" -ge $SKIN_L4 ]; then echo 1017600
    elif [ "$1" -ge $SKIN_L3 ]; then echo 1132800
    elif [ "$1" -ge $SKIN_L2 ]; then echo 1267200
    elif [ "$1" -ge $SKIN_L1 ]; then echo 1382400
    elif [ "$1" -ge $SKIN_L0 ]; then echo 1497600
    else echo "$PRIME_MAX" ; fi
}

freq_for_tier() {
    case "$1" in
        0) echo "$F_MAX"    ;;
        1) echo "$F_NUDGE"  ;;
        2) echo "$F_MILD"   ;;
        3) echo "$F_MOD"    ;;
        4) echo "$F_STRONG" ;;
    esac
}

label_for_tier() {
    case "$1" in
        0) echo MAX    ;;
        1) echo NUDGE  ;;
        2) echo MILD   ;;
        3) echo MOD    ;;
        4) echo STRONG ;;
    esac
}

# Battery current in mA from dumpsys, as an absolute value (negative means
# discharging on this device) so a drop in load current can be detected.
read_battery_ma() {
    # dumpsys line is "current now: <microamps>".
    local raw
    raw=$(dumpsys battery 2>/dev/null | awk -F': ' '/current now/ {print $2; exit}')
    if [ -z "$raw" ]; then
        echo 0
        return
    fi
    # microamps to mA, absolute value.
    awk -v v="$raw" 'BEGIN { x = v/1000; if (x<0) x=-x; printf "%d", x }'
}

# Header
TS=$(date +%s)
echo "[$TS] preempt_throttle_watchdog v2 (PREEMPTIVE battery+skin ladder -- gradual glide before throttle)" > "$LOG"
echo "[$TS] zones (resolved by name): ddr=$DDR_ZONE cpu=$CPU_ZONE soc=$SOC_ZONE skin=$SKIN_ZONE bat=$BAT_ZONE" >> "$LOG"
echo "[$TS] PREEMPTIVE ladders anchored at the vendor throttle (battery 50.0C = measured deep-throttle to 883):" >> "$LOG"
echo "[$TS]   BATTERY (both) full<47.0 ; >=47.0/48.0/48.5/49.0/49.5C -> 1497/1382/1267/1132/1017 MHz" >> "$LOG"
echo "[$TS]   SKIN    (both) full<50.0 ; >=50.0/51.0/51.5/52.0/52.5C -> 1497/1382/1267/1132/1017 MHz" >> "$LOG"
echo "[$TS]   DDR backstop >=71.0C -> 1267 (DDR runs 66-69C normally; not the trigger)" >> "$LOG"
echo "[$TS] caps BOTH clusters: policy0 (cpu0-5 perf, max $PERF_MAX) + policy6 (cpu6-7 prime, max $PRIME_MAX)" >> "$LOG"
echo "[$TS] DDR  warn=$(fmt_c $DDR_WARN_MC)C crit=$(fmt_c $DDR_CRIT_MC)C"     >> "$LOG"
echo "[$TS] CPU  warn=$(fmt_c $CPU_WARN_MC)C crit=$(fmt_c $CPU_CRIT_MC)C"     >> "$LOG"
echo "[$TS] SOC  warn=$(fmt_c $SOC_WARN_MC)C crit=$(fmt_c $SOC_CRIT_MC)C"     >> "$LOG"
echo "[$TS] SKIN warn=$(fmt_c $SKIN_WARN_MC)C crit=$(fmt_c $SKIN_CRIT_MC)C"   >> "$LOG"
echo "[$TS] BAT  warn=$(fmt_c $BAT_WARN_MC)C crit=$(fmt_c $BAT_CRIT_MC)C"     >> "$LOG"
echo "[$TS] BAT_I drop warn=${BAT_CUR_DROP_WARN_MA}mA crit=${BAT_CUR_DROP_CRIT_MA}mA" >> "$LOG"
echo "[$TS] tier thresholds T_eff: tier1=DISABLED / 0.50 / 0.75 / 0.90 (hysteresis 0.10)" >> "$LOG"
echo "[$TS] freq ladder: tier 1 NUDGE DISABLED / tier 2 MILD -8% / tier 3 MOD -15% / tier 4 STRONG -22%" >> "$LOG"
echo "[$TS] freq kHz: MAX=$F_MAX NUDGE=$F_NUDGE MILD=$F_MILD MOD=$F_MOD STRONG=$F_STRONG" >> "$LOG"
echo "[$TS] LOG=$LOG STOP=$STOP" >> "$LOG"
echo "[$TS] tier=0 MAX=$F_MAX kHz (initial)" >> "$LOG"
apply_tier 0

# Main loop
TIER=0
DOM_SENSOR=none
DOM_THREAT=0
DOWN_CNT=0
PRIME_TIER=0 ; PERF_TIER=0 ; LAST_PRIME=-1 ; LAST_PERF=-1 ; PDOWN=0 ; FDOWN=0

# Battery-current baseline (mA), the running max of polled values.
BAT_BASE_MA=0
BAT_TICK=0
LAST_BAT_MA=0
T_BAT_MC=0     # cached battery temp (updated on slow path)

while [ ! -f "$STOP" ]; do
    t_ddr=$(read_zone_mc "$DDR_ZONE")
    t_cpu=$(read_zone_mc "$CPU_ZONE")
    t_soc=$(read_zone_mc "$SOC_ZONE")
    t_skin=$(read_zone_mc "$SKIN_ZONE")

    # Slow path: battery temp + current (dumpsys is expensive).
    if [ "$BAT_TICK" -le 0 ]; then
        # Battery temp comes from its thermal zone, which is cheap to read.
        t_bat_zone=$(read_zone_mc "$BAT_ZONE")
        T_BAT_MC=$t_bat_zone
        ima=$(read_battery_ma)
        LAST_BAT_MA=$ima
        # Baseline is the running max.
        if [ "$BAT_BASE_MA" -lt "$ima" ]; then
            BAT_BASE_MA=$ima
        fi
        BAT_TICK=$BAT_EVERY
    fi
    BAT_TICK=$(( BAT_TICK - 1 ))

    # Compute per-sensor threats (milli-units).
    th_ddr=$(threat_mc  "$t_ddr"  "$DDR_WARN_MC"  "$DDR_CRIT_MC")
    th_cpu=$(threat_mc  "$t_cpu"  "$CPU_WARN_MC"  "$CPU_CRIT_MC")
    th_soc=$(threat_mc  "$t_soc"  "$SOC_WARN_MC"  "$SOC_CRIT_MC")
    th_skin=$(threat_mc "$t_skin" "$SKIN_WARN_MC" "$SKIN_CRIT_MC")
    th_bat=$(threat_mc  "$T_BAT_MC" "$BAT_WARN_MC" "$BAT_CRIT_MC")

    # Battery current drop threat (only after baseline established).
    th_bati=0
    if [ "$BAT_BASE_MA" -gt 0 ] && [ "$LAST_BAT_MA" -gt 0 ]; then
        drop=$(( BAT_BASE_MA - LAST_BAT_MA ))
        if [ "$drop" -gt 0 ]; then
            th_bati=$(threat_drop "$drop" "$BAT_CUR_DROP_WARN_MA" "$BAT_CUR_DROP_CRIT_MA")
        fi
    fi

    # Max of DDR/CPU/SoC/current threats, a backstop signal only. The battery
    # and skin ladders below set the caps.
    T_EFF=$th_ddr ; BS_DOM=DDR ; BS_T=$t_ddr ; BS_W=$DDR_WARN_MC ; BS_C=$DDR_CRIT_MC
    if [ "$th_cpu" -gt "$T_EFF" ];  then T_EFF=$th_cpu ;  BS_DOM=CPU_big ; BS_T=$t_cpu ;  BS_W=$CPU_WARN_MC ;  BS_C=$CPU_CRIT_MC ; fi
    if [ "$th_soc" -gt "$T_EFF" ];  then T_EFF=$th_soc ;  BS_DOM=SOC ;     BS_T=$t_soc ;  BS_W=$SOC_WARN_MC ;  BS_C=$SOC_CRIT_MC ; fi
    if [ "$th_bati" -gt "$T_EFF" ]; then T_EFF=$th_bati ; BS_DOM=Bat_I ;   BS_T=0 ;       BS_W=0 ;             BS_C=0 ; fi

    # Per-cluster cap = min(battery ladder, skin ladder).
    pf_b=$(perf_bat_cap "$T_BAT_MC") ; pf_s=$(perf_skin_cap "$t_skin")
    perf_cap=$pf_b ; [ "$pf_s" -lt "$perf_cap" ] && perf_cap=$pf_s
    pr_b=$(prime_bat_cap "$T_BAT_MC") ; pr_s=$(prime_skin_cap "$t_skin")
    prime_cap=$pr_b ; [ "$pr_s" -lt "$prime_cap" ] && prime_cap=$pr_s
    # DDR backstop. DDR runs 66-69 C on this workload and is not the throttle
    # trigger (battery 50 C is), so floor to 1267 only at 71 C.
    if [ "$t_ddr" -ge 71000 ]; then
        [ 1267200 -lt "$perf_cap" ] && perf_cap=1267200
        [ 1267200 -lt "$prime_cap" ] && prime_cap=1267200
    fi
    # apply on change + log
    if [ "$perf_cap" != "$LAST_PERF" ] || [ "$prime_cap" != "$LAST_PRIME" ]; then
        echo "$perf_cap" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null
        echo "$prime_cap" > /sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq 2>/dev/null
        ts=$(date +%s)
        echo "[$ts] PERF=$perf_cap PRIME=$prime_cap kHz | T_bat=$(fmt_c $T_BAT_MC)C T_skin=$(fmt_c $t_skin)C T_ddr=$(fmt_c $t_ddr)C T_cpu=$(fmt_c $t_cpu)C" >> "$LOG"
        LAST_PERF=$perf_cap ; LAST_PRIME=$prime_cap
    fi

    sleep "$SAMPLE_DT" 2>/dev/null || sleep 1
done

# On exit, restore both clusters to their vendor max.
set_freq MAX
ts=$(date +%s)
echo "[$ts] watchdog_v2 exit, restored MAX (last PRIME=$PRIME_TIER PERF=$PERF_TIER)" >> "$LOG"
