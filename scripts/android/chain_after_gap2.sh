#!/bin/bash
# Sequencer for the 2026-09-20 gap-closure work. One campaign owns the phone at a time.
#
# run_gap_closure2.sh ends by launching the 9-hour needle campaign itself. The gemma
# GPU diagnostic is GPU-only and must not overlap that CPU campaign, or both cells'
# power and timing are contaminated. So: wait for the gap-closure flag, stop the
# needle campaign it just started, run the 5-minute diagnostic, then start the needle
# campaign again. run_niah_keydiff.sh skips any cell whose meta.json exists, so
# stopping and restarting it costs nothing.
set -u
LOG(){ echo "[$(date +%F' '%H:%M:%S)] $*"; }
export ANDROID_ADB_SERVER_PORT=5161 ADB_PORTS=5161 ADB_CALL_TIMEOUT=1500
SA=/home/mislam22/EndurKV_workspace/EndurKV/scripts/android

LOG "waiting for gap closure 2 to finish"
while [ ! -f /tmp/gap_closure2_DONE ]; do sleep 60; done
sleep 30                                   # let it spawn the needle campaign first
LOG "stopping the needle campaign so the GPU diagnostic runs alone"
pkill -f run_niah_keydiff.sh 2>/dev/null
sleep 5
ANDROID_ADB_SERVER_PORT=5161 adb shell "su -c 'pkill -f eviction_bench; pkill -f sample_sensors'" < /dev/null >/dev/null 2>&1
sleep 5

LOG "running the gemma GPU diagnostic"
bash $SA/run_gemma_gpu_diag.sh 2>&1 | sed 's/^/  diag| /'

LOG "restarting the needle campaign"
nohup bash $SA/run_niah_keydiff.sh > /tmp/niah_keydiff.log 2>&1 &
LOG "chain done; needle campaign running -> /tmp/niah_keydiff.log"
