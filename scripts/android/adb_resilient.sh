#!/bin/bash
# adb helpers that survive USB/ADB disconnects. Source: . scripts/android/adb_resilient.sh
#   adb_wait                       block until a device is online
#   adb_safe_shell "CMD"           adb shell with retry on disconnect
#   adb_safe_pull / adb_safe_push  pull/push with retry
#   phone_nohup "CMD" PID_FILE     start CMD on the phone detached from the adb session
#   phone_running PID_FILE         is the detached process alive
#   phone_waitfor PID_FILE [SECS]  block until it exits
# Long runs go through phone_nohup so phone-side work continues if USB drops.

export PATH=/home/mislam22/tools/platform-tools:$PATH

ADB_RETRY_LIMIT=${ADB_RETRY_LIMIT:-30}      # retries before giving up on a single op
ADB_RETRY_SLEEP=${ADB_RETRY_SLEEP:-5}        # seconds between retries
ADB_RESILIENT_LOG=${ADB_RESILIENT_LOG:-/tmp/adb_resilient.log}

_adb_log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$ADB_RESILIENT_LOG"; }

# Candidate adb server ports, in preference order. A phone behind an SSH reverse
# tunnel appears on whichever port the tunnel binds, which can change on reconnect.
ADB_PORTS=${ADB_PORTS:-"5160 5037 5177 5152 5151"}   # 2026-08-18: 5154 was briefly added here and then
       # removed. The Windows host tunnels its adb server to 5154, and probing it starts a local
       # adb server there that blocks the ssh -R forward. Do not probe the tunnel port.

# PID of a local adb server listening on $1, else empty. Used to find a server this
# script spawned by accident. Do not use `adb kill-server` for this, it is forwarded
# over ssh -R and kills the server on the far end.
_adb_local_server_pid() {
    ss -ltnpH "sport = :$1" 2>/dev/null \
      | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}


# Block until a device is in 'device' state. Re-probes every candidate port each
# round, since the phone can move to a different adb server, and logs while waiting.
adb_wait() {
    local tries=0 p
    while true; do
        # fast path: current port still good?
        if [ -n "${ANDROID_ADB_SERVER_PORT:-}" ] && \
           adb get-state 2>/dev/null | grep -q '^device$'; then
            return 0
        fi
        # Skip ports nobody listens on: `adb devices` against a dead port starts
        # an adb server there, which then blocks `ssh -R <port>` from binding.
        for p in $ADB_PORTS; do
            # Check both IPv4 and IPv6 localhost, an ssh -R forward may bind [::1] only.
            if ! (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; then
                (exec 3<>/dev/tcp/::1/$p) 2>/dev/null || continue   # nothing on either family
            fi
            exec 3<&- 2>/dev/null
            # The forward can still die between the check above and `adb devices`,
            # in which case adb starts its own server on $p. Record any existing
            # local server so a newly spawned one can be told apart and reaped.
            _pre_pid=$(_adb_local_server_pid "$p")
            if ANDROID_ADB_SERVER_PORT=$p adb devices 2>/dev/null | grep -qw device; then
                if [ "${ANDROID_ADB_SERVER_PORT:-}" != "$p" ]; then
                    _adb_log "adb_wait: device moved to port $p (was ${ANDROID_ADB_SERVER_PORT:-unset})"
                    echo "[adb] device found on server port $p" >&2
                fi
                export ANDROID_ADB_SERVER_PORT=$p
                return 0
            fi
            # No device on $p. Kill any server this probe spawned by local PID,
            # since `adb kill-server` would go down the tunnel to the far end.
            _post_pid=$(_adb_local_server_pid "$p")
            if [ -n "$_post_pid" ] && [ "$_post_pid" != "$_pre_pid" ]; then
                kill "$_post_pid" 2>/dev/null
                _adb_log "adb_wait: reaped self-spawned adb server pid=$_post_pid on port $p (would block ssh -R)"
                echo "[adb] reaped stray adb server on $p so 'ssh -R $p' can bind" >&2
            fi
        done
        tries=$((tries + 1))
        # Log the wait periodically so a stall is visible.
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
        # Bound each call: if the reverse tunnel flaps mid-call, `adb shell` can
        # block forever and callers' own timeouts never fire.
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

# Start a command on the phone that survives the adb session ending (nohup +
# background + disown). Writes the child PID to PID_FILE on the phone.
#   phone_nohup "cd /data/local/tmp/endurkv && ./bin_cpu/eviction_bench ... > meta.json 2> err.log" /sdcard/work.pid
phone_nohup() {
    local cmd="$1" pid_file="$2"
    adb_wait
    adb shell "nohup sh -c '($cmd) </dev/null >/dev/null 2>&1 & echo \$! > $pid_file; disown' </dev/null >/dev/null 2>&1 &"
    sleep 1
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
