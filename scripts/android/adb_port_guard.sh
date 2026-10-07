#!/bin/bash
# Keep the ssh -R endpoint (default port 5152) free of local adb servers.
# adb starts a server whenever it cannot reach one, so an adb call made while the
# tunnel is down takes the port and `ssh -R` cannot reconnect. Polls every 2 s
# because a slower poll lost the race to campaign retries. Never uses
# `adb kill-server`, which would go down the tunnel and kill the remote server.
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
