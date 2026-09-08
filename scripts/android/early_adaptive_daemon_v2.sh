#!/system/bin/sh
# v2 multi-sensor early-adaptive watchdog (explicit per-sensor multi-tier ladders).
# Each sensor has its OWN warn/mid/crit thresholds; a sensor's tier = highest
# threshold it has crossed; global tier = max over all sensor tiers.
#
# Empirical basis (10 kernel-BCL throttle events on ab4_vanilla):
#   skin_BACK   trigger band 41.8-43.3 C  (mean 42.5, sigma 0.48)  - PRIMARY
#   skin_FRONT  43.2-44.7 C  (mean 43.97, sigma 0.48)              - PRIMARY
#   skin_FRAME  40.6-41.8 C  (mean 41.25, sigma 0.42)              - PRIMARY
#   PMIC die    47.4-49.4 C  (mean 48.03, sigma 0.53)              - PRIMARY
#   battery     41.7-43.2 C  (mean 42.54, sigma 0.52)              - PRIMARY
#   DDR/CPU     loose (sigma 0.80-0.94), passengers near actual kernel cliff (65-67 C)
#
# Args: $1 log path  $2 stop-flag path
set -u
LOG=${1:-/sdcard/early_ad_v2.log}
STOP=${2:-/sdcard/early_ad_v2.stop}

# === Sensor sysfs zones (OnePlus 15 Snapdragon 8 Elite Gen 5) ===
DDR_ZONE=/sys/class/thermal/thermal_zone47        # ddr_temp_mc
CPU_ZONE=/sys/class/thermal/thermal_zone24        # cpu-0-3-0
SK_BACK_ZONE=/sys/class/thermal/thermal_zone70    # shell_back  (primary trigger)
SK_FRONT_ZONE=/sys/class/thermal/thermal_zone61   # shell_front (primary trigger)
PMIC_ZONE=/sys/class/thermal/thermal_zone86       # pmh0101_tz  (primary trigger)
BAT_ZONE=/sys/class/thermal/thermal_zone93        # battery     (primary trigger)

# === Per-sensor explicit tier thresholds (milli-degrees C) ===
# Each sensor has warn/mid/crit triggering STEP 1/2/3 respectively.
# Primary triggers (chassis): tight 2-3 C bands centered on the empirical fire point.
# Passengers (DDR, CPU): wider bands near the actual kernel cliffs.

# skin_back  (primary)
SK_BACK_T1=41000   # STEP 1
SK_BACK_T2=42000   # STEP 2
SK_BACK_T3=43000   # STEP 3

# skin_front (primary)
SK_FRONT_T1=43000
SK_FRONT_T2=44000
SK_FRONT_T3=45000

# PMIC (primary)
PMIC_T1=46000
PMIC_T2=47500
PMIC_T3=49000

# battery (primary)
BAT_T1=41000
BAT_T2=42000
BAT_T3=43000

# CPU mid (passenger -- near 67 C kernel CPU cliff)
CPU_T1=62000
CPU_T2=64000
CPU_T3=67000

# DDR (passenger -- near 65 C kernel DDR cliff)
DDR_T1=60000
DDR_T2=62000
DDR_T3=65000

CAP1=1497600
CAP2=1267200
CAP3=883200

current_step=0
TS=$(date +%s)
echo "[$TS] v2 multi-sensor watchdog start (explicit per-sensor 3-tier ladders)" >> "$LOG"
echo "[$TS]   PRIMARY triggers (chassis):" >> "$LOG"
echo "[$TS]     skin_back  T1=41 T2=42 T3=43 C   battery   T1=41 T2=42 T3=43 C" >> "$LOG"
echo "[$TS]     skin_front T1=43 T2=44 T3=45 C   PMIC      T1=46 T2=47.5 T3=49 C" >> "$LOG"
echo "[$TS]   PASSENGER triggers (near kernel cliff):" >> "$LOG"
echo "[$TS]     CPU mid    T1=62 T2=64 T3=67 C   DDR       T1=60 T2=62 T3=65 C" >> "$LOG"
echo "[$TS]   tier->cap: T1=1497 T2=1267 T3=883 MHz   (one-way ratchet)" >> "$LOG"

read_zone() {
    local z=$1; local out=0
    [ -r "$z/temp" ] && read out < "$z/temp" 2>/dev/null
    [ -z "$out" ] && out=0
    echo $out
}

# Compute a single sensor's tier (0-3) from its three thresholds.
# Args: T  T1  T2  T3
tier_of() {
    if   [ "$1" -ge "$4" ] 2>/dev/null; then echo 3
    elif [ "$1" -ge "$3" ] 2>/dev/null; then echo 2
    elif [ "$1" -ge "$2" ] 2>/dev/null; then echo 1
    else echo 0
    fi
}

step_down() {
    local new_step=$1; local new_cap=$2; local new_label=$3
    if [ "$current_step" -lt "$new_step" ]; then
        for c in 6 7; do
            echo "$new_cap" > /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_max_freq 2>/dev/null
        done
        local TS_NOW=$(date +%s)
        echo "[$TS_NOW] STEP $new_step ($new_label) trip_sensor=$trip_sensor trip_T=${trip_T_c}C" >> "$LOG"
        echo "[$TS_NOW]   state: DDR=${ddr_c}C CPU=${cpu_c}C skin_back=${skb_c}C skin_front=${skf_c}C PMIC=${pmic_c}C bat=${bat_c}C cap=${new_cap}kHz" >> "$LOG"
        echo "[$TS_NOW]   per-sensor tiers: DDR=$t_ddr CPU=$t_cpu skin_back=$t_skb skin_front=$t_skf PMIC=$t_pmic bat=$t_bat" >> "$LOG"
        current_step=$new_step
    fi
}

while [ ! -f "$STOP" ]; do
    ddr_mc=$(read_zone $DDR_ZONE)
    cpu_mc=$(read_zone $CPU_ZONE)
    skb_mc=$(read_zone $SK_BACK_ZONE)
    skf_mc=$(read_zone $SK_FRONT_ZONE)
    pmic_mc=$(read_zone $PMIC_ZONE)
    bat_mc=$(read_zone $BAT_ZONE)

    t_ddr=$(tier_of $ddr_mc $DDR_T1 $DDR_T2 $DDR_T3)
    t_cpu=$(tier_of $cpu_mc $CPU_T1 $CPU_T2 $CPU_T3)
    t_skb=$(tier_of $skb_mc $SK_BACK_T1 $SK_BACK_T2 $SK_BACK_T3)
    t_skf=$(tier_of $skf_mc $SK_FRONT_T1 $SK_FRONT_T2 $SK_FRONT_T3)
    t_pmic=$(tier_of $pmic_mc $PMIC_T1 $PMIC_T2 $PMIC_T3)
    t_bat=$(tier_of $bat_mc $BAT_T1 $BAT_T2 $BAT_T3)

    # Find the maximum sensor tier and which sensor produced it
    max_tier=$t_ddr; trip_sensor=DDR; trip_T_c=$((ddr_mc/1000))
    [ "$t_cpu"  -gt "$max_tier" ] && { max_tier=$t_cpu;  trip_sensor=CPU;        trip_T_c=$((cpu_mc/1000)); }
    [ "$t_skb"  -gt "$max_tier" ] && { max_tier=$t_skb;  trip_sensor=SKIN_BACK;  trip_T_c=$((skb_mc/1000)); }
    [ "$t_skf"  -gt "$max_tier" ] && { max_tier=$t_skf;  trip_sensor=SKIN_FRONT; trip_T_c=$((skf_mc/1000)); }
    [ "$t_pmic" -gt "$max_tier" ] && { max_tier=$t_pmic; trip_sensor=PMIC;       trip_T_c=$((pmic_mc/1000)); }
    [ "$t_bat"  -gt "$max_tier" ] && { max_tier=$t_bat;  trip_sensor=BATTERY;    trip_T_c=$((bat_mc/1000)); }

    ddr_c=$((ddr_mc/1000)); cpu_c=$((cpu_mc/1000)); skb_c=$((skb_mc/1000))
    skf_c=$((skf_mc/1000)); pmic_c=$((pmic_mc/1000)); bat_c=$((bat_mc/1000))

    case "$max_tier" in
        3) step_down 3 $CAP3 EMERGENCY ;;
        2) step_down 2 $CAP2 DEEP ;;
        1) step_down 1 $CAP1 PREEMPT ;;
        *) : ;;
    esac
    sleep 0.2
done

TS=$(date +%s)
echo "[$TS] v2 watchdog stop  final_step=$current_step" >> "$LOG"
