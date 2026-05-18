#!/system/bin/sh
# discover_sensors.sh — run ONCE on the OnePlus 15 (or any target Android phone)
# to map the sysfs paths the controller will read at runtime.
#
# Captures four things:
#   1. /sys/class/thermal/thermal_zone*/type     (zone names)
#      /sys/class/thermal/thermal_zone*/temp     (current temp, millideg C)
#   2. /sys/block/sd*/stat and /sys/block/sd*/queue/*  (block-level write counts)
#   3. /sys/devices/platform/.../ufshc*/...      (vendor UFS counters if exposed)
#   4. /proc/meminfo + /proc/pressure/{cpu,memory,io}  (PSI signals)
#
# Output: /data/local/tmp/endurkv/sensor_map.txt — paste back to host so we
# wire the right paths into the sampler.
#
# Runs without root. Some paths only readable as root will be marked PERM-DENIED.
# Run as:
#   adb shell sh /data/local/tmp/endurkv/discover_sensors.sh

set +e
OUT="/data/local/tmp/endurkv/sensor_map.txt"
mkdir -p "$(dirname "$OUT")"
: > "$OUT"

log() { echo "$@" | tee -a "$OUT"; }

log "=== device ==="
log "model:   $(getprop ro.product.model)"
log "device:  $(getprop ro.product.device)"
log "soc:     $(getprop ro.soc.model 2>/dev/null)$(getprop ro.board.platform 2>/dev/null)"
log "android: $(getprop ro.build.version.release)"
log "kernel:  $(uname -r)"
log "abi:     $(getprop ro.product.cpu.abi)"
log ""

log "=== thermal zones ==="
for z in /sys/class/thermal/thermal_zone*; do
    [ -d "$z" ] || continue
    name=$(cat "$z/type" 2>/dev/null || echo "?")
    temp=$(cat "$z/temp" 2>/dev/null || echo "?")
    log "$(basename "$z")  type=$name  temp_millideg=$temp"
done
log ""

log "=== cooling devices ==="
for c in /sys/class/thermal/cooling_device*; do
    [ -d "$c" ] || continue
    typ=$(cat "$c/type" 2>/dev/null || echo "?")
    cur=$(cat "$c/cur_state" 2>/dev/null || echo "?")
    maxs=$(cat "$c/max_state" 2>/dev/null || echo "?")
    log "$(basename "$c")  type=$typ  cur=$cur/$maxs"
done
log ""

log "=== block devices ==="
# /sys/block/<dev>/stat is 17 fields (kernel >= 4.18). Field 7 = writes_completed,
# field 8 = writes_merged, field 9 = sectors_written, field 10 = ms_writing.
# These move regardless of root, which is exactly the WAF signal we need.
for b in /sys/block/sd* /sys/block/mmcblk* /sys/block/sda; do
    [ -d "$b" ] || continue
    stat=$(cat "$b/stat" 2>/dev/null || echo "PERM-DENIED")
    size=$(cat "$b/size" 2>/dev/null || echo "?")
    log "$(basename "$b")  size_sectors=$size"
    log "  stat: $stat"
done
log ""

log "=== UFS host controller (vendor paths) ==="
# Most Qualcomm-based phones expose UFS under platform/soc/<addr>.ufshc/.
# OnePlus 15 with Snapdragon 8 Elite Gen 5 will follow the same pattern.
for u in /sys/devices/platform/soc/*.ufshc/host*/scsi_host/host*/ \
         /sys/class/scsi_host/host*/ \
         /sys/bus/platform/drivers/ufshcd/*/; do
    for f in $u; do
        [ -d "$f" ] || continue
        log "ufs path: $f"
        ls -la "$f" 2>/dev/null | head -20 | sed 's/^/  /' | tee -a "$OUT" >/dev/null
        log ""
    done
done

# Look for the userdata/system partitions backing the writes we care about.
log "=== mount points (data partition is what catches KV spills) ==="
mount | grep -E "/data |/system " | tee -a "$OUT"
log ""

log "=== PSI (pressure-stall info) ==="
for p in /proc/pressure/cpu /proc/pressure/memory /proc/pressure/io; do
    if [ -r "$p" ]; then
        log "$p"
        sed 's/^/  /' < "$p" | tee -a "$OUT" >/dev/null
    else
        log "$p — not readable"
    fi
done
log ""

log "=== memory ==="
grep -E "^(MemTotal|MemFree|MemAvailable|Cached|SwapTotal|SwapFree)" /proc/meminfo | tee -a "$OUT"
log ""

log "=== summary ==="
n_zones=$(ls -d /sys/class/thermal/thermal_zone* 2>/dev/null | wc -l)
n_blocks=$(ls -d /sys/block/* 2>/dev/null | wc -l)
log "thermal_zones=$n_zones  block_devs=$n_blocks"
log "Output written to: $OUT"
