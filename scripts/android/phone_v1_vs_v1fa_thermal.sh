#!/bin/bash
# phone_v1_vs_v1fa_thermal.sh — head-to-head thermal comparison of v1 (FA-off
# throughout) vs v1_fa (FA-off prefill + FA-on decode via snapkv state-swap).
#
# Each condition: 3 iterations of (prefill 7700-token narrativeqa + decode 256
# tokens) on Llama-3.2-1B. Cool-down between conditions; sensor sampler at 5 Hz.
#
# This is the experiment that proves v1_fa's thermal benefit translates from
# theory (FA-on cuts attention-side bandwidth ~3×) to measurable on-device
# temperature reduction.
#
# Usage:
#   ./phone_v1_vs_v1fa_thermal.sh [--iters N]

set -e
export PATH=/home/mislam22/tools/platform-tools:$PATH
ITERS=${ITERS:-3}
TS=$(date +%s)
PHONE_OUT=/data/local/tmp/endurkv/logs/v1_vs_v1fa_${TS}
HOST_OUT=/home/mislam22/EndurKV_workspace/phone-logs/v1_vs_v1fa_${TS}
mkdir -p $HOST_OUT
PROG=$HOST_OUT/progress.log
echo "[$(date)] v1 vs v1_fa thermal head-to-head START -> $HOST_OUT" | tee $PROG
adb shell "mkdir -p $PHONE_OUT"

run_cond() {
    local policy=$1; local kval=$2
    echo "" | tee -a $PROG
    echo "[$(date)] === policy=$policy K=$kval | $ITERS iters ===" | tee -a $PROG
    DIR=$PHONE_OUT/${policy}_K${kval}
    adb shell "mkdir -p $DIR"

    # Cool wait
    for try in $(seq 1 90); do
        SKIN=$(adb shell "dumpsys battery | grep temperature | awk '{print \$2}'" 2>/dev/null | tr -d '\r\n ')
        if [ -n "$SKIN" ] && [ "$SKIN" -lt 360 ] 2>/dev/null; then break; fi
        sleep 5
    done
    echo "[$(date)]   skin=$SKIN/10 C" | tee -a $PROG

    # Sensor sampler in background (host-side adb shell)
    adb shell "cd /data/local/tmp/endurkv && sh scripts/sample_sensors.sh --out $DIR/sensors.csv --hz 5" > /dev/null 2>&1 &
    SAMPLER=$!
    sleep 1

    # Iterations
    for it in $(seq 1 $ITERS); do
        IDIR=$DIR/iter${it}
        adb shell "mkdir -p $IDIR"
        T0=$(date +%s)
        adb shell "
cd /data/local/tmp/endurkv
LD_LIBRARY_PATH=bin_cpu ./bin_cpu/eviction_bench \
    --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --prompt prompts_chat/Llama-3.2-1B/narrativeqa_pub_001.txt \
    --prompt-id narrativeqa_pub_001 \
    --policy $policy --k-nominal $kval \
    --max-tokens 256 --ctx-size 12288 --seed 42 \
    --threads 4 --n-gpu-layers 0 --n-batch 512 --ubatch-size 64 \
    --n-sink 4 --greedy \
    --out-csv $IDIR/steps.csv \
    --out-meta $IDIR/meta.json 2>$IDIR/stderr.log
echo iter${it}_exit=\$?" 2>&1 | tail -1 | tee -a $PROG
        T1=$(date +%s)
        echo "[$(date)]   iter${it} wall=$((T1-T0))s" | tee -a $PROG
    done

    # Stop sampler
    kill $SAMPLER 2>/dev/null || true
    adb shell "pgrep -f sample_sensors | xargs -r kill 2>/dev/null; true" >/dev/null 2>&1 || true
    sleep 2

    # Pull
    adb pull -q $DIR $HOST_OUT/ 2>&1 | tail -1 >> $PROG
}

# Cell 1: v1 (FA-off throughout)
run_cond v1 512
# Cell 2: v1_fa (FA-off prefill + FA-on decode)
run_cond v1_fa 512

echo "" | tee -a $PROG
echo "[$(date)] === v1 vs v1_fa DONE: $HOST_OUT ===" | tee -a $PROG
