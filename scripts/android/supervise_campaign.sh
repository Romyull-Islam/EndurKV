#!/bin/bash
# supervise_campaign.sh <script-path> <log-path> <done-marker> <cells-glob> [stall_min]
# Restarts a resumable campaign when the reverse-tunnelled adb link drops or stalls.
# Relies on the campaign skipping finished cells, so a restart only loses the cell in flight.
# Stalls are detected by completed cells, not log output, since a healthy cell can
# log nothing for ~25 min.
set -u
SCRIPT=$1; LOG=$2; DONE=$3; GLOB=$4; STALL_MIN=${5:-45}
SUPLOG=${LOG%.log}_supervisor.log
say(){ echo "[$(date +%F_%H:%M:%S)] $*" >> "$SUPLOG"; }
cells(){ ls $GLOB 2>/dev/null | wc -l; }

# One supervisor per campaign. The bench runs detached on the phone, so a duplicate
# would run the same cell twice concurrently and heat the phone past the cool gate.
exec 9>/tmp/.supervise_$(basename "$SCRIPT").lock
flock -n 9 || { echo "another supervisor already owns $(basename "$SCRIPT") -- exiting"; exit 0; }

say "supervising $SCRIPT (stall threshold ${STALL_MIN}m, done=$DONE)"
last_n=$(cells); last_change=$(date +%s); restarts=0

while [ ! -f "$DONE" ]; do
    # Wait for the device before launching, a campaign started on a dead link
    # burns its retry budget and exits.
    if ! timeout 20 adb devices 2>/dev/null | grep -qw device; then
        adb kill-server >/dev/null 2>&1     # clear a wedged local server / protocol fault
        say "no device (tunnel down?) -- waiting"
        sleep 60; continue
    fi
    # Track the launched PID. pgrep on the script name would match this supervisor's
    # own argv and conclude the campaign is already running.
    if [ -z "${CPID:-}" ] || ! kill -0 "$CPID" 2>/dev/null; then
        say "campaign not running -> starting (restart #$restarts, $(cells) cells done)"
        setsid nohup bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
        CPID=$!; say "campaign pid=$CPID"
        sleep 30; last_change=$(date +%s); continue
    fi
    n=$(cells)
    if [ "$n" != "$last_n" ]; then
        say "progress: $last_n -> $n cells"
        last_n=$n; last_change=$(date +%s)
    elif [ $(( $(date +%s) - last_change )) -gt $((STALL_MIN*60)) ]; then
        restarts=$((restarts+1))
        say "STALL: no new cell in ${STALL_MIN}m -> resetting adb and restarting (#$restarts)"
        [ -n "${CPID:-}" ] && kill -9 "$CPID" 2>/dev/null; CPID=""
        adb kill-server >/dev/null 2>&1     # clears the wedged-server "protocol fault"
        # detached runs outlive the host process, kill them so the next cell
        # does not share the CPU with an orphan
        timeout 30 adb shell "su -c 'pkill -9 -f eviction_bench; pkill -9 -f sample_sensors'" >/dev/null 2>&1
        sleep 10
        last_change=$(date +%s)
    fi
    sleep 60
done
say "DONE marker present -- $(cells) cells; $restarts restarts"
