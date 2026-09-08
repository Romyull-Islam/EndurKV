#!/system/bin/sh
# pin_dvfs.sh — root-only: force consistent DVFS regime across all Wave-3 cells.
#
# Eliminates the cable-state confound: regardless of whether USB is plugged in
# or the battery is charging/full, the big cores stay capped at 1.63 GHz.
#
# This matches the regime that all v1 / v1_FA cells experienced during Wave-3
# REAL when the battery was on its own (USB online but not charging).
#
# Idempotent — safe to call before every cell. Records old state to /sdcard so
# you can restore later.
#
# Usage (must run as root, e.g., via `su -c`):
#   sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh         # set to 1.63 GHz
#   sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh restore # restore previous state
#
# Big cores on Snapdragon 8 Elite Gen 5: CPU6, CPU7 (Phoenix-L prime cores).
# 1.63 GHz = 1632000 kHz — matches the battery-state observed cap.

set -u
TARGET_MHZ=${TARGET_MHZ:-1632000}
LITTLE_TARGET_MHZ=${LITTLE_TARGET_MHZ:-1632000}   # also cap little cores for safety

ACTION=${1:-pin}
STATE_FILE=/sdcard/.dvfs_state.sh

pin() {
    # Only capture the pre-pin state if we are NOT already pinned. Re-running pin()
    # while pinned used to overwrite the saved state with the PINNED values, so the
    # later restore re-applied the cap instead of undoing it -- which left the phone
    # capped at 1632 MHz (35% of hw max) after a run died mid-campaign on 2026-08-29.
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

    # Apply target — use performance governor at capped max_freq for stable clocks.
    # 'userspace' lets us pin an exact freq but isn't always available; performance
    # at the chosen max_freq is universally supported.
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
