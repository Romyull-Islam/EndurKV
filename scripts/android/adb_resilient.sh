#!/bin/bash
# adb_resilient.sh — library of helpers that survive USB/ADB disconnects.
#
# Source this file: . scripts/android/adb_resilient.sh
#
# Provides:
#   adb_wait              — block until a device is online (auto-restarts server)
#   adb_safe_shell "..."  — run an adb shell command, retry on disconnect
#   adb_safe_pull SRC DST — pull with retry
#   adb_safe_push SRC DST — push with retry
#   phone_nohup "CMD" PID_FILE — start CMD on phone, detached (survives ADB drop)
#   phone_running PID_FILE — is the detached phone process still alive?
#   phone_waitfor PID_FILE [TIMEOUT_S] — block until detached process exits
#
# Pattern for long-running experiments:
#   1. phone_nohup "cd ... && ./eviction_bench ... > meta.json 2> err.log; echo done > /sdcard/done.flag" /sdcard/work.pid
#   2. Loop: until adb_safe_shell "test -f /sdcard/done.flag"; do sleep 10; done
#   3. adb_safe_pull /sdcard/done.flag /tmp/ (verify)
#   4. adb_safe_pull /data/local/tmp/endurkv/logs/.../ host/
#
# This way, even if USB drops or laptop sleeps, the phone-side work continues
# and we pick up results when reconnected.

export PATH=/home/mislam22/tools/platform-tools:$PATH

ADB_RETRY_LIMIT=${ADB_RETRY_LIMIT:-30}      # retries before giving up on a single op
ADB_RETRY_SLEEP=${ADB_RETRY_SLEEP:-5}        # seconds between retries
ADB_RESILIENT_LOG=${ADB_RESILIENT_LOG:-/tmp/adb_resilient.log}

_adb_log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$ADB_RESILIENT_LOG"; }

# Block until at least one device is in 'device' state. Auto-recovers the
# adb server if it's wedged.
# Candidate adb server ports, in preference order. A phone reached over an SSH
# reverse tunnel appears on whichever port the tunnel currently binds, and that
# changes when the tunnel is re-established.
ADB_PORTS=${ADB_PORTS:-"5160 5037 5177 5152 5151"}   # 2026-08-18: 5154 was briefly added here and then
       # REMOVED. That port is where the Windows host reverse-tunnels its adb server; probing it
       # from this side auto-STARTS a local adb server on 5154, which then squats the port and
       # makes the ssh -R forward fail with "remote port forwarding failed for listen port 5154".
       # The device is reachable on 5037, so never probe the tunnel port.

# Return the PID of an adb server WE own listening on $1, else empty. Used to detect
# a server this script accidentally spawned (see the race note in adb_wait). Never
# use `adb kill-server` for this -- it is forwarded over ssh -R and kills the far end.
_adb_local_server_pid() {
    ss -ltnpH "sport = :$1" 2>/dev/null \
      | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}


adb_wait() {
    # CHANGED 2026-07-27. THREE BUGS, all of which fired together and stalled the
    # NIAH campaign for 2.5 h with no error message:
    #  (1) The port was fixed at whatever was exported at launch. When the phone
    #      re-enumerated onto a different adb server, this looped forever against a
    #      server that would never have a device. Now we RE-PROBE every round and
    #      adopt whichever port actually holds a device.
    #  (2) `adb kill-server` was fatal here: when the port is an SSH reverse tunnel,
    #      that command travels DOWN the tunnel and kills the adb server on the far
    #      machine, taking the phone away from everyone. Removed. We only start a
    #      local server for a port nobody is listening on.
    #  (3) It waited silently and forever. Now it logs, so a stall is visible.
    local tries=0 p
    while true; do
        # fast path: current port still good?
        if [ -n "${ANDROID_ADB_SERVER_PORT:-}" ] && \
           adb get-state 2>/dev/null | grep -q '^device$'; then
            return 0
        fi
        # re-probe every candidate port.
        # CRITICAL (2026-07-27): pre-check with /dev/tcp and SKIP ports nobody is
        # listening on. `adb devices` against a dead port STARTS AN ADB SERVER there.
        # The probe loop therefore squatted every candidate port while waiting, and a
        # squatted port makes `ssh -R <port>` fail with "remote port forwarding failed"
        # -- i.e. the recovery code was actively preventing the reconnect it waited for.
        for p in $ADB_PORTS; do
            # 2026-08-19 IPv6 FIX. This pre-check used to probe 127.0.0.1 only. An
            # `ssh -R` forward may bind IPv6 localhost instead: the working tunnel here
            # listens on [::1]:5037, so the IPv4 probe failed, the port was SKIPPED, and
            # adb_wait looped forever against ports that never had a device -- silently
            # hanging every campaign for hours with the phone plainly reachable via
            # `adb devices`. Probe BOTH families before giving up on a port.
            if ! (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; then
                (exec 3<>/dev/tcp/::1/$p) 2>/dev/null || continue   # nothing on either family
            fi
            exec 3<&- 2>/dev/null
            # RACE (2026-08-02): the /dev/tcp pre-check above is necessary but NOT
            # sufficient. If the ssh -R forward dies in the window between that check
            # and the `adb devices` below, adb cannot connect and starts its OWN server
            # on $p -- squatting the very port the user needs to re-establish the
            # forward, so `ssh -R` fails with "remote port forwarding failed" and the
            # wait loop can never succeed. Observed twice on 5152.
            # Remember whether a server we own was already there, so we can tell an
            # adb server WE just spawned from one that legitimately pre-existed.
            _pre_pid=$(_adb_local_server_pid "$p")
            if ANDROID_ADB_SERVER_PORT=$p adb devices 2>/dev/null | grep -qw device; then
                if [ "${ANDROID_ADB_SERVER_PORT:-}" != "$p" ]; then
                    _adb_log "adb_wait: device moved to port $p (was ${ANDROID_ADB_SERVER_PORT:-unset})"
                    echo "[adb] device found on server port $p" >&2
                fi
                export ANDROID_ADB_SERVER_PORT=$p
                return 0
            fi
            # No device on $p. If an adb server appeared on $p that was NOT there
            # before this probe, WE spawned it -- reap it so it cannot block ssh -R.
            # Kill the local PID directly: `adb kill-server` would be forwarded down
            # the tunnel and kill the FAR-END server on the user's machine instead.
            _post_pid=$(_adb_local_server_pid "$p")
            if [ -n "$_post_pid" ] && [ "$_post_pid" != "$_pre_pid" ]; then
                kill "$_post_pid" 2>/dev/null
                _adb_log "adb_wait: reaped self-spawned adb server pid=$_post_pid on port $p (would block ssh -R)"
                echo "[adb] reaped stray adb server on $p so 'ssh -R $p' can bind" >&2
            fi
        done
        tries=$((tries + 1))
        # Surface the wait instead of hanging mutely. Never kill a server we may
        # not own (see bug 2 above).
        if [ $((tries % 12)) -eq 1 ]; then
            _adb_log "adb_wait: no device on any of [$ADB_PORTS]; waiting (try $tries)"
            echo "[adb] waiting for device on [$ADB_PORTS] (try $tries) — campaign paused, not aborted" >&2
        fi
        sleep "$ADB_RETRY_SLEEP"
    done
}

# Run a shell command on the device, retrying through disconnects.
# Echoes the command's stdout; returns 0 on success.
adb_safe_shell() {
    local cmd="$1"
    local tries=0
    local out rc
    while [ $tries -lt "$ADB_RETRY_LIMIT" ]; do
        adb_wait
        # 2026-08-18: HARD TIMEOUT on the adb call. The device now hangs off a
        # reverse-tunnelled adb server on a Windows host; when that tunnel flaps
        # mid-call, "adb shell" blocks FOREVER with no error. A campaign then sat
        # 4.5 h inside its cool-gate command substitution -- the enclosing loop
        # never iterated, so the loop's own 30-minute timeout could never fire.
        # Bounding the call turns an unrecoverable hang into an ordinary retry.
        out=$(timeout "${ADB_CALL_TIMEOUT:-120}" adb shell "$cmd" 2>&1)
        rc=$?
        if [ $rc -eq 124 ]; then
            _adb_log "adb_safe_shell: TIMEOUT after ${ADB_CALL_TIMEOUT:-120}s; killing server and retrying"
            adb kill-server >/dev/null 2>&1
            tries=$((tries + 1)); sleep "$ADB_RETRY_SLEEP"; continue
        fi
        # If adb itself reports no device, retry; if the command failed cleanly
        # on the device (rc!=0 but adb returned), return that to caller.
        if echo "$out" | grep -qE 'no devices|device offline|device unauthorized|closed|connection reset'; then
            _adb_log "adb_safe_shell: transient ($out); retry"
            tries=$((tries + 1))
            sleep "$ADB_RETRY_SLEEP"
            continue
        fi
        printf '%s\n' "$out"
        return $rc
    done
    _adb_log "adb_safe_shell: gave up after $tries retries: $cmd"
    return 99
}

adb_safe_pull() {
    local src="$1" dst="$2"
    local tries=0
    while [ $tries -lt "$ADB_RETRY_LIMIT" ]; do
        adb_wait
        if adb pull -q "$src" "$dst" 2>>"$ADB_RESILIENT_LOG"; then
            return 0
        fi
        tries=$((tries + 1))
        _adb_log "adb_safe_pull: retry $tries for $src"
        sleep "$ADB_RETRY_SLEEP"
    done
    return 99
}

adb_safe_push() {
    local src="$1" dst="$2"
    local tries=0
    while [ $tries -lt "$ADB_RETRY_LIMIT" ]; do
        adb_wait
        if adb push "$src" "$dst" 2>>"$ADB_RESILIENT_LOG"; then
            return 0
        fi
        tries=$((tries + 1))
        _adb_log "adb_safe_push: retry $tries for $src"
        sleep "$ADB_RETRY_SLEEP"
    done
    return 99
}

# Start a command on the phone that survives ADB session termination.
# Uses Android's `setsid` (if available) + nohup + double-fork pattern.
# Writes the child PID to PID_FILE on the phone for later checking.
#
# Usage:
#   phone_nohup "cd /data/local/tmp/endurkv && ./bin_cpu/eviction_bench ... > meta.json 2> err.log" /sdcard/work.pid
phone_nohup() {
    local cmd="$1" pid_file="$2"
    adb_wait
    # The trailing 'setsid' ensures the child process is in its own session,
    # so when adb shell disconnects (SIGHUP), the child is unaffected.
    # We write the child PID to PID_FILE so we can check on it later.
    adb shell "nohup sh -c '($cmd) </dev/null >/dev/null 2>&1 & echo \$! > $pid_file; disown' </dev/null >/dev/null 2>&1 &"
    sleep 1
    # Verify PID file was written
    local pid=$(adb_safe_shell "cat $pid_file 2>/dev/null")
    if [ -z "$pid" ]; then
        _adb_log "phone_nohup: PID file not written; command may not have launched"
        return 1
    fi
    _adb_log "phone_nohup: started PID=$pid (pid_file=$pid_file)"
    printf '%s\n' "$pid"
}

# Returns 0 if the PID in PID_FILE is alive on the phone, 1 otherwise.
phone_running() {
    local pid_file="$1"
    local pid=$(adb_safe_shell "cat $pid_file 2>/dev/null")
    [ -z "$pid" ] && return 1
    adb_safe_shell "kill -0 $pid 2>/dev/null && echo alive || echo gone" | grep -q alive
}

# Block until the phone-side process exits. Default timeout 2 hours.
phone_waitfor() {
    local pid_file="$1"
    local timeout_s="${2:-7200}"
    local start=$(date +%s)
    while phone_running "$pid_file"; do
        local now=$(date +%s)
        if [ $((now - start)) -ge $timeout_s ]; then
            _adb_log "phone_waitfor: timed out after $timeout_s s waiting on $pid_file"
            return 99
        fi
        sleep 10
    done
    _adb_log "phone_waitfor: $pid_file completed in $(($(date +%s) - start)) s"
    return 0
}
