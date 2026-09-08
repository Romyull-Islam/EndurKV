#!/bin/bash
# ============================================================================
# supervise_campaign.sh <script-path> <log-path> <done-marker> <cells-glob> [stall_min]
#
# WHY. The phone is only reachable through an SSH reverse tunnel from a Windows
# host, and that path dropped four separate ways on 2026-08-18/19: a bind
# conflict, an IPv6-only listener the port pre-check could not see, a mid-call
# flap that hung adb for 4.5 h, and an adb "protocol fault" when the far server
# died. Each individual failure is now handled, but a 4-6 h campaign of ~25 min
# cells cannot finish if the link drops every ~30 min, no matter how good the
# retry logic is. Direct USB and wireless adb are both unavailable here, so the
# remaining option is to make a restart cheap and automatic.
#
# This is safe ONLY because the campaigns are resumable: run() skips any cell
# that already has a non-empty gen.json, so a restart costs at most the cell
# that was in flight. Without that, this loop would redo everything each time.
#
# Stall detection is on COMPLETED CELLS, not on log output: a hung adb call
# produces neither, but a healthy long-running cell produces no log lines either
# for ~25 min, so log mtime alone would cause false restarts mid-cell.
# ============================================================================
set -u
SCRIPT=$1; LOG=$2; DONE=$3; GLOB=$4; STALL_MIN=${5:-45}
SUPLOG=${LOG%.log}_supervisor.log
say(){ echo "[$(date +%F_%H:%M:%S)] $*" >> "$SUPLOG"; }
cells(){ ls $GLOB 2>/dev/null | wc -l; }

# SINGLE INSTANCE ONLY (2026-08-19). Duplicate supervisors each launched their own
# campaign; because the bench now runs DETACHED on the device, killing a host process
# did not stop its device-side work, so two 6-thread runs of the SAME cell executed
# concurrently. They contended for CPU, drove the phone to 57 C, made the cool gate
# unable to ever clear, and produced two mutually invalid cells. A lock is cheaper
# than detecting that after the fact.
exec 9>/tmp/.supervise_$(basename "$SCRIPT").lock
flock -n 9 || { echo "another supervisor already owns $(basename "$SCRIPT") -- exiting"; exit 0; }

say "supervising $SCRIPT (stall threshold ${STALL_MIN}m, done=$DONE)"
last_n=$(cells); last_change=$(date +%s); restarts=0

while [ ! -f "$DONE" ]; do
    # Do not launch into a dead link: a campaign started while the tunnel is down
    # burns its retry budget and dies, and the supervisor would just loop on that.
    # Wait politely for the device instead -- this is the state after the Windows-side
    # ssh -R drops, and it clears by itself when the tunnel is re-established.
    if ! timeout 20 adb devices 2>/dev/null | grep -qw device; then
        adb kill-server >/dev/null 2>&1     # clear a wedged local server / protocol fault
        say "no device (tunnel down?) -- waiting"
        sleep 60; continue
    fi
    # Track the PID we launched. Do NOT pgrep the script name: this supervisor carries
    # that path in its OWN argv, so pgrep matches the supervisor itself, concludes the
    # campaign is running, and never starts it -- exactly what happened on attempt one.
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
        # detached runs outlive the host process -- clear them or the next cell
        # competes with an orphan for the CPU (this is what produced 57 C starts).
        timeout 30 adb shell "su -c 'pkill -9 -f eviction_bench; pkill -9 -f sample_sensors'" >/dev/null 2>&1
        sleep 10
        last_change=$(date +%s)
    fi
    sleep 60
done
say "DONE marker present -- $(cells) cells; $restarts restarts"
