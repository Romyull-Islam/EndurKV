#!/bin/bash
# Waits for the LongBench wide campaign to finish (/tmp/lb_wide_DONE), then runs the
# energy-aware proof v2 on the phone. Started with setsid so it survives the shell.
# Log: /tmp/ea_proof_v2.log   Pid: /tmp/ea_proof_v2.pid
export ANDROID_SERIAL=${ANDROID_SERIAL:-10.0.0.127:5555}
export ADB_CALL_TIMEOUT=1500
echo $$ > /tmp/ea_proof_v2.pid
echo "[$(date '+%F %T')] queued; waiting for /tmp/lb_wide_DONE"
while [ ! -f /tmp/lb_wide_DONE ]; do sleep 120; done
echo "[$(date '+%F %T')] LongBench done; letting the phone rest 10 min before the energy cells"
sleep 600
bash /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/run_energy_aware_proof_v2.sh
echo "[$(date '+%F %T')] queue finished"
