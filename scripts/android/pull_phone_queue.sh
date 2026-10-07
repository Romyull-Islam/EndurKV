#!/bin/bash
# pull_phone_queue.sh: copies the phone-side queue results to the host whenever the adb tunnel
# is up. Calls adb only when port 5162 is held by the tunnel (not by a local adb server), and
# kills any local adb server that grabs the port, so a dropped tunnel can always reconnect.
PORT=5162; DEV=/data/local/tmp/endurkv/logs/phoneq
DEST=/home/mislam22/EndurKV_workspace/tmp_archive/phoneq; mkdir -p $DEST
export ADB_SERVER_SOCKET=tcp:127.0.0.1:$PORT ANDROID_SERIAL=3C15B8003ZA00000
LOG(){ echo "[$(date '+%F %T')] $*"; }
squatter(){ ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | grep -q . && \
            ss -ltnp 2>/dev/null | grep ":$PORT " | grep -q '"adb"'; }
free_port(){ for p in $(ss -ltnp 2>/dev/null | grep ":$PORT " | grep '"adb"' | grep -o 'pid=[0-9]*' | cut -d= -f2); do kill $p && LOG "killed local adb $p squatting on $PORT"; done; }
while :; do
  free_port
  if ss -ltn 2>/dev/null | grep -q "127.0.0.1:$PORT "; then
    if timeout 600 adb pull $DEV $DEST/.. >/dev/null 2>&1; then
      n=$(ls $DEST/*.done 2>/dev/null | wc -l); LOG "pulled: $n cells done; last: $(tail -1 $DEST/run.log 2>/dev/null)"
      [ -f $DEST/ALL_DONE ] && { LOG "ALL_DONE"; break; }
      [ -f $DEST/STOPPED ]  && { LOG "STOPPED on the phone, see run.log"; break; }
    else LOG "pull failed (tunnel down?)"; fi
    free_port
  else LOG "tunnel down, waiting"; fi
  sleep 300
done
