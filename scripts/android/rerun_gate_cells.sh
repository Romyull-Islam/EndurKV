#!/bin/bash
# Re-run the two cells the gate BUG killed (snapkv, adakv). They died in the
# state-API round-trip (4073.4 MiB second cache -> OS-kill) because the auto-
# promote set fa_on_evict without forcing in-place compaction. Binary fixed and
# pushed 2026-08-25; the campaign script is resumable, so re-invoking it redoes
# exactly the cells with no meta.json. Runs LAST so it never races the CPU queue.
set -u
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
LOG "waiting for the CPU queue to finish ..."
while [ ! -f /tmp/remaining_queue_DONE ]; do sleep 60; done
LOG "CPU queue done -- re-running the two gate-killed cells"
rm -f /tmp/phi3_gpu_complete_DONE
exec /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/run_phi3_gpu_complete.sh
