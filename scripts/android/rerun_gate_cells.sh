#!/bin/bash
# Re-run the snapkv and adakv cells that were OS-killed in the state round-trip (the gate's
# auto-promote set fa_on_evict without in-place compaction). The campaign is resumable and
# redoes only cells without meta.json. Runs after the CPU queue so the two never overlap.
set -u
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
LOG "waiting for the CPU queue to finish ..."
while [ ! -f /tmp/remaining_queue_DONE ]; do sleep 60; done
LOG "CPU queue done -- re-running the two gate-killed cells"
rm -f /tmp/phi3_gpu_complete_DONE
exec /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/run_phi3_gpu_complete.sh
