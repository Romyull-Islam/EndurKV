#!/system/bin/sh
# Root only: pin CPU DVFS so clocks do not depend on cable or charging state.
# All cores get the performance governor with max 1632000 kHz (the cap seen on
# battery). Big cores are cpu6 and cpu7. Safe to call before every cell. The
# previous state is saved to /sdcard for restore.
#
# Usage (as root, e.g. via `su -c`):
#   sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh          # pin
#   sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh restore  # restore previous state

set -u
TARGET_MHZ=${TARGET_MHZ:-1632000}
LITTLE_TARGET_MHZ=${LITTLE_TARGET_MHZ:-1632000}   # also cap little cores for safety

ACTION=${1:-pin}
STATE_FILE=/sdcard/.dvfs_state.sh

pin() {
    # Save state only if not already pinned. Otherwise the saved state would hold
    # the pinned values and restore would re-apply the cap.
    _cur_gov=$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor 2>/dev/null)
    if [ "$_cur_gov" = "performance" ] && [ -s "$STATE_FILE" ]; then
        echo "[pin_dvfs] already pinned; keeping existing $STATE_FILE" >&2
        _skip_save=1
    else
        echo "[pin_dvfs] saving current state -> $STATE_FILE" >&2
        _skip_save=0
        : > "$STATE_FILE"
    fi
    for cpu in 0 1 2 3 4 5 6 7; do
        if [ -d /sys/devices/system/cpu/cpu$cpu/cpufreq ]; then
            gov=$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor 2>/dev/null)
            maxf=$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_max_freq 2>/dev/null)
            if [ "${_skip_save:-0}" -eq 0 ]; then
              echo "echo $gov  > /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor" >> "$STATE_FILE"
              echo "echo $maxf > /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_max_freq"  >> "$STATE_FILE"
            fi
        fi
    done

    # performance governor with a capped max_freq gives stable clocks. The
    # userspace governor is not always available.
    for cpu in 0 1 2 3 4 5; do
        if [ -d /sys/devices/system/cpu/cpu$cpu/cpufreq ]; then
            echo performance     > /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor 2>/dev/null
            echo $LITTLE_TARGET_MHZ > /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_max_freq 2>/dev/null
        fi
    done
    for cpu in 6 7; do
        if [ -d /sys/devices/system/cpu/cpu$cpu/cpufreq ]; then
            echo performance    > /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor 2>/dev/null
            echo $TARGET_MHZ    > /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_max_freq 2>/dev/null
        fi
    done

    echo "[pin_dvfs] pinned big cores 6,7 to max=$TARGET_MHZ kHz (performance governor)" >&2
    echo "[pin_dvfs] verify:" >&2
    for cpu in 6 7; do
        echo "  cpu$cpu: gov=$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor 2>/dev/null) max=$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_max_freq 2>/dev/null) cur=$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_cur_freq 2>/dev/null)" >&2
    done
}

restore() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "[pin_dvfs] no saved state to restore" >&2
        return 1
    fi
    echo "[pin_dvfs] restoring from $STATE_FILE" >&2
    sh "$STATE_FILE"
    rm -f "$STATE_FILE"
}

case "$ACTION" in
    pin)     pin     ;;
    restore) restore ;;
    *)       echo "usage: $0 [pin|restore]" >&2; exit 1 ;;
esac
