#!/bin/bash
# adb_port_guard.sh -- keep the ssh -R endpoint free.  (2026-08-03)
#
# 5152 is where the user's `ssh -R 127.0.0.1:5152:127.0.0.1:5037` lands. A local
# adb server must NEVER own it: adb starts one whenever a command cannot reach a
# server, so any bare `adb push`/`adb pull`/`adb shell` issued while the tunnel is
# down claims the port and then `ssh -R` fails with "remote port forwarding failed"
# -- the recovery path blocking the reconnect it is waiting for. Observed 3x.
# adb_wait already reaps servers IT spawns, but the campaign scripts also call adb
# directly, so this guard covers those. Poll is 2s, not 20s: at 20s the campaign
# retry loop re-spawned a squatter faster than the guard cleared it, so `ssh -R`
# still landed in a window where the port was held (observed 4x). Never uses `adb kill-server`: that travels
# down the tunnel and kills the far-end server on the user's machine.
PORT=${1:-5152}
while true; do
  pid=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  if [ -n "$pid" ] && ps -p "$pid" -o args= 2>/dev/null | grep -q 'adb .*fork-server'; then
    # a real forward is owned by sshd, not adb -- so an adb owner is always stray
    kill "$pid" 2>/dev/null
    echo "[$(date '+%F %T')] reaped stray adb server pid=$pid on $PORT" >> /tmp/adb_port_guard.log
  fi
  sleep 2
done
