#!/bin/bash
# Waits for the energy-aware proof to finish (EAPROOF2_DONE in its log), rests the phone,
# then runs the scheduler proof. Started with setsid so it survives the shell.
# Log: /tmp/sched_proof.log   Pid: /tmp/sched_proof.pid
export ANDROID_SERIAL=${ANDROID_SERIAL:-10.0.0.127:5555}
export ADB_CALL_TIMEOUT=1500
echo $$ > /tmp/sched_proof.pid
echo "[$(date '+%F %T')] queued; waiting for EAPROOF2_DONE in /tmp/ea_proof_v2.log"
while ! grep -q EAPROOF2_DONE /tmp/ea_proof_v2.log 2>/dev/null; do sleep 120; done
echo "[$(date '+%F %T')] proof done; resting 5 min"
sleep 300
bash /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/run_sched_proof.sh
echo "[$(date '+%F %T')] queue finished"
