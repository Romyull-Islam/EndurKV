#!/bin/bash
# Wait for WikiText-2 PPL to finish, then kill the queue, push new eviction_bench,
# and launch the full 3-policy sweep with std-protocol sampling.
set -e
export PATH=/home/mislam22/tools/platform-tools:$PATH

LOG=/home/mislam22/EndurKV_workspace/EndurKV/figures/launch_full.log
echo "[launch] starting at $(date)" | tee -a "$LOG"

# 1. Wait for llama-perplexity to finish
echo "[launch] waiting for WikiText-2 PPL (llama-perplexity) to finish ..." | tee -a "$LOG"
while adb shell "ps -A | grep -q llama-perplexity" 2>/dev/null; do
    sleep 60
done
echo "[launch] llama-perplexity done at $(date)" | tee -a "$LOG"

# Pull the WT2 PPL output before queue moves on
echo "[launch] pulling WikiText-2 PPL output ..." | tee -a "$LOG"
mkdir -p /home/mislam22/EndurKV_workspace/phone-logs/wt2_ppl
adb pull /data/local/tmp/endurkv/logs/wt2_ppl/ \
        /home/mislam22/EndurKV_workspace/phone-logs/ 2>&1 | tail -3 | tee -a "$LOG"
echo "[launch] WT2 PPL summary:" | tee -a "$LOG"
grep -E "Final estimate|estimate.*PPL" /home/mislam22/EndurKV_workspace/phone-logs/wt2_ppl/ppl_output.txt 2>/dev/null | tee -a "$LOG"

# 2. Kill the queue (PID 22691) — would otherwise start old K=2048 A/B
echo "[launch] killing old queue (PID 22691) ..." | tee -a "$LOG"
kill 22691 2>/dev/null || echo "  queue already exited"
sleep 2

# 3. Push the NEW eviction_bench (with top-p, sink-token, rep-penalty, FA-on vanilla, effective-KV)
echo "[launch] pushing new eviction_bench with top-p sampling ..." | tee -a "$LOG"
adb push /home/mislam22/EndurKV_workspace/EndurKV/entropy_probe/build-android/eviction_bench \
        /data/local/tmp/endurkv/bin/eviction_bench 2>&1 | tail -2 | tee -a "$LOG"
adb shell "chmod 755 /data/local/tmp/endurkv/bin/eviction_bench"

# Also push the cool wrapper if missing
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/phone_cool_then_run.sh \
        /data/local/tmp/endurkv/scripts/ 2>&1 | tail -1 | tee -a "$LOG"
adb shell "chmod 755 /data/local/tmp/endurkv/scripts/phone_cool_then_run.sh"

# 4. Launch the full sweep
echo "[launch] starting FULL SWEEP (3 policies × 5 prompts × 2 K × 3 reps = 90 runs) at $(date)" | tee -a "$LOG"
bash /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/phone_full_sweep_3policy.sh 2>&1 | tee -a "$LOG"
echo "[launch] FULL SWEEP DONE at $(date)" | tee -a "$LOG"
