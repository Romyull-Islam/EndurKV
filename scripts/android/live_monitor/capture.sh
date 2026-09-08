#!/usr/bin/env bash
# Perfetto capture for OnePlus 15 (the Android analog of Xcode Instruments).
# Records CPU/freq/thermal(CPU,GPU,NPU)/battery-power timeline, pulls a .pftrace
# you drag into https://ui.perfetto.dev
#
# Usage:
#   capture.sh -t 30                 # fixed 30-second trace
#   capture.sh -- <cmd on phone...>  # trace exactly spans the phone command
#   capture.sh -t 60 -o myrun        # custom output name (host: myrun.pftrace)
#
# Examples:
#   ./capture.sh -- 'cd /data/local/tmp && LD_LIBRARY_PATH=. ./pi-main -m model.gguf -p "..." -n 256 -t 6'
#   ./capture.sh -t 45               # then run inference by hand on the phone within 45s
set -uo pipefail

ADB="${ADB:-/home/mislam22/tools/platform-tools/adb -s 3C15B8003ZA00000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CFG="$HERE/trace_config.textproto"
KEY="op15mon"
DEV_TRACE="/data/misc/perfetto-traces/op15.pftrace"

DUR=""; OUT="op15_$(date +%H%M%S)"; CMD=()
while [ $# -gt 0 ]; do
  case "$1" in
    -t) DUR="$2"; shift 2;;
    -o) OUT="$2"; shift 2;;
    --) shift; CMD=("$@"); break;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done
OUTFILE="$HERE/${OUT}.pftrace"

echo "[perfetto] starting trace (detached, key=$KEY)…"
$ADB shell "su -c 'rm -f $DEV_TRACE'" 2>/dev/null
# Start detached so the session survives the shell returning; runs until --stop.
$ADB shell "perfetto -c - --txt -o $DEV_TRACE --detach=$KEY" < "$CFG" 2>&1 | sed 's/^/  /'
sleep 1

if [ "${#CMD[@]}" -gt 0 ]; then
  echo "[perfetto] running workload on phone:"
  echo "           ${CMD[*]}"
  $ADB shell "${CMD[*]}"
  echo "[perfetto] workload finished."
elif [ -n "$DUR" ]; then
  echo "[perfetto] tracing for ${DUR}s — run your inference on the phone now…"
  $ADB shell "sleep $DUR"
else
  echo "[perfetto] no -t and no -- command. Tracing 20s by default."
  $ADB shell "sleep 20"
fi

echo "[perfetto] stopping trace…"
$ADB shell "perfetto --attach=$KEY --stop" 2>&1 | sed 's/^/  /'
sleep 1
$ADB pull "$DEV_TRACE" "$OUTFILE" 2>&1 | sed 's/^/  /'

if [ -s "$OUTFILE" ]; then
  SZ=$(du -h "$OUTFILE" | cut -f1)
  echo ""
  echo "[perfetto] ✅ saved $OUTFILE ($SZ)"
  echo "           Open https://ui.perfetto.dev and drag the file in."
  echo "           Tracks: CPU per-core + freq · thermal (cpu*/gpuss/nsphvx=NPU) · battery power · mem."
else
  echo "[perfetto] ❌ trace empty — see messages above."
  exit 1
fi
