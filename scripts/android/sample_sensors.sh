#!/system/bin/sh
# sample_sensors.sh v3 — optimized for low CPU overhead at 10 Hz on aarch64 sh.
#
# Strategy:
#   * One-shot reads via `cat /sys/.../path*` glob to drain 8-98 values per fork.
#   * `read VAR <file` builtin for single values (zero forks).
#   * /proc/vmstat parsed via `while read ... done <file` (one fork total).
#   * Cooling-device + battery refreshed every 10 ticks (~1 Hz) to amortize.
#   * Probe PID cached and only refreshed every 10 ticks.
#
# Same column schema as v2; downstream joiner reads the header unchanged.
#
# Usage on phone (same as before):
#   sample_sensors.sh --out PATH [--hz N] [--duration SECONDS]

OUT=""
HZ=10
DURATION=0
ZONES_FILE=""
BLOCKS_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --out)       OUT="$2"; shift 2 ;;
        --hz)        HZ="$2"; shift 2 ;;
        --duration)  DURATION="$2"; shift 2 ;;
        --zones)     ZONES_FILE="$2"; shift 2 ;;
        --blocks)    BLOCKS_FILE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$OUT" ]; then
    echo "Usage: $0 --out PATH [--hz N] [--duration SECONDS]" >&2
    exit 1
fi

# ---- Discover what to sample (once, at startup) ----
if [ -n "$ZONES_FILE" ] && [ -r "$ZONES_FILE" ]; then
    ZONES=$(cat "$ZONES_FILE")
else
    ZONES=""
    for z in /sys/class/thermal/thermal_zone*; do
        [ -r "$z/temp" ] || continue
        ZONES="$ZONES $z"
    done
fi

if [ -n "$BLOCKS_FILE" ] && [ -r "$BLOCKS_FILE" ]; then
    BLOCKS=$(cat "$BLOCKS_FILE")
else
    BLOCKS=""
    for b in /sys/block/sd* /sys/block/mmcblk* /sys/block/sda; do
        [ -r "$b/stat" ] || continue
        BLOCKS="$BLOCKS $b"
    done
fi

# Sleep interval (busybox sleep accepts decimals on Android).
case "$HZ" in
    1)  SLEEP=1     ;;
    2)  SLEEP=0.5   ;;
    5)  SLEEP=0.2   ;;
    10) SLEEP=0.1   ;;
    20) SLEEP=0.05  ;;
    *) SLEEP=$(awk "BEGIN { printf \"%.4f\", 1.0/$HZ }") ;;
esac

# ---- Cache thermal zone names + their /temp paths once ----
# Names are munged with tr to keep the CSV clean. We avoid the per-iteration
# `cat $z/type` by reading the type values ONCE here at startup.
ZONE_NAMES=""
ZONE_TEMP_PATHS=""
for z in $ZONES; do
    read t </"$z/type" 2>/dev/null
    nm=$(echo "$t" | tr -d ',\n\r' | tr ' ' '_')
    [ -z "$nm" ] && nm=$(basename "$z")
    ZONE_NAMES="$ZONE_NAMES $nm"
    ZONE_TEMP_PATHS="$ZONE_TEMP_PATHS $z/temp"
done

# ---- Map cpufreq cooling devices ONCE ----
CPUFREQ_COOL_PATHS=""   # space-separated "i:cooling_path"
for c in /sys/class/thermal/cooling_device*; do
    [ -r "$c/type" ] || continue
    read ct </"$c/type" 2>/dev/null
    case "$ct" in
        cpufreq-cpu*)
            idx=${ct#cpufreq-cpu}
            CPUFREQ_COOL_PATHS="$CPUFREQ_COOL_PATHS ${idx}:${c}/cur_state"
            ;;
    esac
done

# ---- CPU count ----
N_CPUS=0
for f in /sys/devices/system/cpu/cpu[0-9]*; do
    [ -d "$f/cpufreq" ] && N_CPUS=$((N_CPUS + 1))
done
[ "$N_CPUS" = "0" ] && N_CPUS=8

# ---- Build CSV header ----
HEADER="wall_clock_s,monotonic_s"
for nm in $ZONE_NAMES; do
    HEADER="$HEADER,${nm}_temp_mc"
done
for b in $BLOCKS; do
    dev=$(basename "$b")
    HEADER="$HEADER,${dev}_w_completed,${dev}_w_merged,${dev}_sectors_w,${dev}_ms_w"
done
HEADER="$HEADER,mem_total_kb,mem_free_kb,mem_avail_kb"
HEADER="$HEADER,psi_cpu_a10,psi_mem_a10,psi_io_a10"
HEADER="$HEADER,vmstat_pswpout,vmstat_pgmajfault,vmstat_pgpgout,vmstat_pswpin"
HEADER="$HEADER,probe_pid,probe_write_bytes,probe_read_bytes"
i=0
while [ "$i" -lt "$N_CPUS" ]; do
    HEADER="$HEADER,cpu${i}_freq_hz"
    i=$((i + 1))
done
i=0
while [ "$i" -lt "$N_CPUS" ]; do
    HEADER="$HEADER,cpu${i}_cool_state"
    i=$((i + 1))
done
HEADER="$HEADER,gpu_busy_us,gpu_total_us"
HEADER="$HEADER,bat_current_ma,bat_voltage_mv,bat_phone_temp_dc"
# v4 additions: USB rail + PMIC direct reads (root-readable on rooted device).
# USB current/voltage gives instantaneous SoC power draw when plugged in:
#   P_soc = (usb_current_ua / 1e6) * (usb_voltage_uv / 1e6)  watts
# Battery PMIC charge_counter enables precise coulomb-counter integration:
#   dE = (charge_dt * voltage_avg) Joules between samples
HEADER="$HEADER,usb_online,usb_current_ua,usb_voltage_uv,bat_charge_uah,bat_power_now_uw"
# v5 additions: richer PMIC sources (root-required on most devices).
#   bat_power_now_uw   — direct microwatts (BEST source for energy integration)
#   bat_current_now_ua — battery current in microamps (signed: + charge / - discharge)
#   bat_voltage_now_uv — battery voltage in microvolts
#   bat_status         — Charging / Discharging / Full / Not charging
# When direct read fails (perm denied), we fall back to a single `su -c cat`
# batch read per tick (1 fork) so we don't pay 4x fork cost per sample.
HEADER="$HEADER,bat_current_now_ua,bat_voltage_now_uv,bat_status"
# v6 additions (2026-09-02): the DVFS state that decides GPU decode throughput.
#   gpu_clk_hz            current Adreno clock (kgsl gpuclk)
#   gpu_thermal_pwrlevel  thermal cap level in force (0 = none; 5 = 726 MHz on this device)
#   ddr_freq_khz          current DDR clock (bus_dcvs); bandwidth-bound decode follows this
#   llcc_freq_khz         current system-cache clock
# Appended last so every earlier column keeps its position.
HEADER="$HEADER,gpu_clk_hz,gpu_thermal_pwrlevel,ddr_freq_khz,llcc_freq_khz"

echo "$HEADER" > "$OUT"

# ---- v5: detect whether PMIC files need `su -c` to read ----
# Probe by attempting a direct read; if it returns empty, switch to `su -c`.
# Cache the read mode for the entire run: "direct" or "su" or "none".
BAT_READ_MODE="direct"
_probe=""
read _probe < /sys/class/power_supply/battery/power_now 2>/dev/null
if [ -z "$_probe" ]; then
    # Try su batch read
    _probe=$(su -c 'cat /sys/class/power_supply/battery/power_now' 2>/dev/null | head -1)
    if [ -n "$_probe" ]; then
        BAT_READ_MODE="su"
    else
        BAT_READ_MODE="none"
    fi
fi

# Cached values, refreshed every 10 ticks (~1 Hz at HZ=10)
BAT_TICK=99
PID_TICK=99
COOL_TICK=99
CACHED_PID=""
CACHED_BAT_I=""
CACHED_BAT_V=""
CACHED_BAT_T=""
CACHED_COOL_STATES=""

TRAP_FLAG=0
trap 'TRAP_FLAG=1' TERM INT

# Time bookkeeping
START_WALL=$(date +%s)
END_WALL=0
[ "$DURATION" -gt 0 ] && END_WALL=$((START_WALL + DURATION))

while [ $TRAP_FLAG -eq 0 ]; do
    NOW_WALL=$(date +%s.%N)
    read NOW_MONO _ </proc/uptime
    ROW="$NOW_WALL,$NOW_MONO"

    # ---- Thermal: one cat for all zones (1 fork instead of 98) ----
    # ZONE_TEMP_PATHS is "/sys/.../thermal_zone0/temp /sys/.../thermal_zone1/temp ..."
    # cat globs all of them in argument order; output is one temp per line.
    TEMPS=$(cat $ZONE_TEMP_PATHS 2>/dev/null)
    # Convert newlines to commas and append
    TEMP_CSV=$(echo "$TEMPS" | tr '\n' ',')
    # Strip trailing comma
    TEMP_CSV=${TEMP_CSV%,}
    ROW="$ROW,$TEMP_CSV"

    # ---- Block stats (usually empty due to perms on Android 16) ----
    for b in $BLOCKS; do
        line=""
        read line </"$b/stat" 2>/dev/null
        if [ -n "$line" ]; then
            # /sys/block/<dev>/stat columns 5..8 are write side.
            # Use positional assignment, no awk.
            set -- $line
            ROW="$ROW,$5,$6,$7,$8"
        else
            ROW="$ROW,,,,"
        fi
    done

    # ---- /proc/meminfo in one read ----
    mt=""; mf=""; ma=""
    while read k v _; do
        case "$k" in
            MemTotal:)     mt=$v ;;
            MemFree:)      mf=$v ;;
            MemAvailable:) ma=$v; break ;;  # MemAvailable is the last we need
        esac
    done </proc/meminfo
    ROW="$ROW,$mt,$mf,$ma"

    # ---- PSI (typically not readable on Android 16) ----
    pc=""; pm=""; pi=""
    if [ -r /proc/pressure/cpu ]; then
        read line </proc/pressure/cpu
        # line is like "some avg10=0.00 avg60=... ..."; field 2 = avg10=X
        set -- $line
        pc=${2#avg10=}
    fi
    if [ -r /proc/pressure/memory ]; then
        read line </proc/pressure/memory
        set -- $line
        pm=${2#avg10=}
    fi
    if [ -r /proc/pressure/io ]; then
        read line </proc/pressure/io
        set -- $line
        pi=${2#avg10=}
    fi
    ROW="$ROW,$pc,$pm,$pi"

    # ---- /proc/vmstat: parse in shell loop (1 fork to open, 0 awk) ----
    pswpout=""; pgmaj=""; pgpgout=""; pswpin=""
    while read k v _; do
        case "$k" in
            pswpout)    pswpout=$v ;;
            pgmajfault) pgmaj=$v ;;
            pgpgout)    pgpgout=$v ;;
            pswpin)     pswpin=$v ;;
        esac
    done </proc/vmstat
    ROW="$ROW,$pswpout,$pgmaj,$pgpgout,$pswpin"

    # ---- Probe pid (cached, refresh every 10 ticks via pgrep) ----
    PID_TICK=$((PID_TICK + 1))
    if [ "$PID_TICK" -ge 10 ]; then
        CACHED_PID=""
        for n in entropy_probe attention_probe prune_probe; do
            p=$(pidof "$n" 2>/dev/null)
            [ -z "$p" ] && p=$(pgrep -x "$n" 2>/dev/null | head -1)
            if [ -n "$p" ]; then
                CACHED_PID=$p
                break
            fi
        done
        PID_TICK=0
    fi
    if [ -n "$CACHED_PID" ] && [ -r "/proc/$CACHED_PID/io" ]; then
        wb=""; rb=""
        while read k v _; do
            case "$k" in
                write_bytes:) wb=$v ;;
                read_bytes:)  rb=$v ;;
            esac
        done </proc/$CACHED_PID/io
        ROW="$ROW,$CACHED_PID,$wb,$rb"
    else
        ROW="$ROW,,,"
    fi

    # ---- Per-core CPU freq: one cat for all 8 (1 fork) ----
    FREQS=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null)
    FREQ_CSV=$(echo "$FREQS" | tr '\n' ',')
    FREQ_CSV=${FREQ_CSV%,}
    ROW="$ROW,$FREQ_CSV"

    # ---- Per-core cooling cap (cached, refresh every 10 ticks) ----
    COOL_TICK=$((COOL_TICK + 1))
    if [ "$COOL_TICK" -ge 10 ]; then
        states=""
        i=0
        while [ "$i" -lt "$N_CPUS" ]; do
            cs=""
            for pair in $CPUFREQ_COOL_PATHS; do
                cidx=${pair%%:*}
                cpath=${pair#*:}
                if [ "$cidx" = "$i" ]; then
                    read cs <"$cpath" 2>/dev/null
                    break
                fi
            done
            states="$states,$cs"
            i=$((i + 1))
        done
        CACHED_COOL_STATES=${states#,}
        COOL_TICK=0
    fi
    ROW="$ROW,$CACHED_COOL_STATES"

    # ---- Adreno GPU busy (cumulative; joiner does delta/delta = %busy) ----
    gpu_b=""; gpu_t=""
    read gpu_b gpu_t _ </sys/class/kgsl/kgsl-3d0/gpubusy 2>/dev/null
    ROW="$ROW,$gpu_b,$gpu_t"

    # ---- Battery via dumpsys (cached every 10 ticks ~ 1 Hz) ----
    BAT_TICK=$((BAT_TICK + 1))
    if [ "$BAT_TICK" -ge 10 ]; then
        BAT_LINE=$(dumpsys battery 2>/dev/null)
        CACHED_BAT_I=""
        CACHED_BAT_V=""
        CACHED_BAT_T=""
        # parse the three fields without forking awk
        echo "$BAT_LINE" | while IFS= read -r ln; do
            case "$ln" in
                *"Battery current"*) CACHED_BAT_I=${ln##*: }; CACHED_BAT_I=${CACHED_BAT_I%% *} ;;
                *"Charger voltage"*) CACHED_BAT_V=${ln##*: }; CACHED_BAT_V=${CACHED_BAT_V%% *} ;;
                *"PhoneTemp"*)       CACHED_BAT_T=${ln##*: }; CACHED_BAT_T=${CACHED_BAT_T%% *} ;;
            esac
        done
        # The above while-in-pipe runs in a subshell, so the assignments are lost.
        # Do it again with a here-doc to keep parent scope.
        CACHED_BAT_I=$(echo "$BAT_LINE" | grep "Battery current" | head -1 | sed 's/.*: *//;s/ .*//')
        CACHED_BAT_V=$(echo "$BAT_LINE" | grep "Charger voltage" | head -1 | sed 's/.*: *//;s/ .*//')
        CACHED_BAT_T=$(echo "$BAT_LINE" | grep "PhoneTemp"       | head -1 | sed 's/.*: *//;s/ .*//')
        BAT_TICK=0
    fi
    ROW="$ROW,$CACHED_BAT_I,$CACHED_BAT_V,$CACHED_BAT_T"

    # ---- v4 + v5: USB rail + PMIC direct reads (root-readable, every tick) ----
    # Cheap: single read of small sysfs files, no fork.
    USB_ON=""; USB_I=""; USB_V=""; BAT_Q=""; BAT_P=""
    BAT_I_NOW=""; BAT_V_NOW=""; BAT_STATUS=""
    [ -r /sys/class/power_supply/usb/online ]      && read USB_ON < /sys/class/power_supply/usb/online
    [ -r /sys/class/power_supply/usb/current_now ] && read USB_I  < /sys/class/power_supply/usb/current_now
    [ -r /sys/class/power_supply/usb/voltage_now ] && read USB_V  < /sys/class/power_supply/usb/voltage_now
    [ -r /sys/class/power_supply/battery/charge_counter ] && read BAT_Q < /sys/class/power_supply/battery/charge_counter

    # v5: richer PMIC sources (power_now, current_now, voltage_now, status).
    # On most Android devices these need root. BAT_READ_MODE was probed at
    # startup:
    #   - direct: use `read VAR < /sys/...` (zero forks per value)
    #   - su:     one `su -c cat ...` batch reads all 4 values in 1 fork
    #   - none:   leave empty (no root, no direct perms)
    case "$BAT_READ_MODE" in
        direct)
            [ -r /sys/class/power_supply/battery/power_now ]   && read BAT_P      < /sys/class/power_supply/battery/power_now
            [ -r /sys/class/power_supply/battery/current_now ] && read BAT_I_NOW  < /sys/class/power_supply/battery/current_now
            [ -r /sys/class/power_supply/battery/voltage_now ] && read BAT_V_NOW  < /sys/class/power_supply/battery/voltage_now
            [ -r /sys/class/power_supply/battery/status ]      && read BAT_STATUS < /sys/class/power_supply/battery/status
            ;;
        su)
            # One fork → 4 values. Order matters: power_now, current_now,
            # voltage_now, status (matches the cat arg order below).
            _bat_batch=$(su -c 'cat /sys/class/power_supply/battery/power_now /sys/class/power_supply/battery/current_now /sys/class/power_supply/battery/voltage_now /sys/class/power_supply/battery/status' 2>/dev/null)
            if [ -n "$_bat_batch" ]; then
                _IFS_SAVED=$IFS
                IFS='
'
                set -- $_bat_batch
                IFS=$_IFS_SAVED
                BAT_P=$1
                BAT_I_NOW=$2
                BAT_V_NOW=$3
                BAT_STATUS=$4
            fi
            ;;
        none)
            : ;;
    esac
    # Sanitize status: it may contain trailing CR or commas (CSV-unsafe).
    BAT_STATUS=$(echo "$BAT_STATUS" | tr -d ' \r\n,')
    ROW="$ROW,$USB_ON,$USB_I,$USB_V,$BAT_Q,$BAT_P"
    ROW="$ROW,$BAT_I_NOW,$BAT_V_NOW,$BAT_STATUS"

    # ---- v6: GPU clock, thermal cap level, DDR and LLCC clocks (root-readable, no fork) ----
    GPU_CLK=""; GPU_TPL=""; DDR_F=""; LLCC_F=""
    [ -r /sys/class/kgsl/kgsl-3d0/gpuclk ]           && read GPU_CLK < /sys/class/kgsl/kgsl-3d0/gpuclk
    [ -r /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel ] && read GPU_TPL < /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel
    [ -r /sys/devices/system/cpu/bus_dcvs/DDR/cur_freq ]  && read DDR_F  < /sys/devices/system/cpu/bus_dcvs/DDR/cur_freq
    [ -r /sys/devices/system/cpu/bus_dcvs/LLCC/cur_freq ] && read LLCC_F < /sys/devices/system/cpu/bus_dcvs/LLCC/cur_freq
    ROW="$ROW,$GPU_CLK,$GPU_TPL,$DDR_F,$LLCC_F"

    echo "$ROW" >> "$OUT"

    [ "$END_WALL" -gt 0 ] && [ "$(date +%s)" -ge "$END_WALL" ] && break
    sleep $SLEEP
done

echo "[sampler] exited cleanly, samples in $OUT" >&2
