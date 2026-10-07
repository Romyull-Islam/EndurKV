#!/bin/bash
# phone_wave3_stress.sh: sustained-stress test, 30 min per policy with no cool-down
# between iterations. Each policy starts cold (skin <= 35 C), then repeats a full prefill
# plus 256-token decode on Llama-3.2-1B (CPU) while sensors sample at 5 Hz.
# Output per policy: phone-logs/wave3_<ts>/<policy>/ (stress.csv, sensors.csv, iter*/).

set -e
export PATH=/home/mislam22/tools/platform-tools:$PATH

POLICIES="vanilla tova v1 h2o"      # all four policies
MODEL=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
MODEL_TAG=Llama-3.2-1B
NGL=0           # CPU only — matches our valid quality measurements
N_BATCH=512
UBATCH=64
CTX=10240
K_NOMINAL=1024
PROMPT_FILE=prompts_chat/Llama-3.2-1B/hotpotqa_pub_001.txt   # ~7700-token chat-templated prompt
PROMPT_ID=hotpotqa_pub_001
DURATION_S=1800        # 30 minutes per policy
COOL_START_C=35        # cool to 35°C before each policy
COOL_MAX_WAIT_S=600    # up to 10 min cool-down between policies
SAMPLE_HZ=5            # sensor sampling rate during stress

# CPU-only build, no Vulkan backend registered
BIN_DIR=bin_cpu

OUT_BASE_PHONE="/data/local/tmp/endurkv/logs/wave3_$(date +%s)"
OUT_BASE_HOST="/home/mislam22/EndurKV_workspace/phone-logs/$(basename $OUT_BASE_PHONE)"
mkdir -p "$OUT_BASE_HOST"
PROG_LOG="$OUT_BASE_HOST/progress.log"

{
echo "[$(date)] wave3_stress_start"
echo "  model: $MODEL_TAG (ngl=$NGL, CPU-only, matches Wave-1-redux backend)"
echo "  policies: $POLICIES   (each gets $DURATION_S s = $((DURATION_S/60)) min)"
echo "  prompt:   $PROMPT_ID (chat-templated)"
echo "  K_nominal: $K_NOMINAL"
echo "  sensor sample rate: $SAMPLE_HZ Hz"
echo "  cool-down before each policy: skin ≤ ${COOL_START_C}°C, max ${COOL_MAX_WAIT_S}s wait"
echo "  out (phone): $OUT_BASE_PHONE"
echo "  out (host):  $OUT_BASE_HOST"
} | tee -a "$PROG_LOG"

adb shell "mkdir -p $OUT_BASE_PHONE"

for POLICY in $POLICIES; do
    echo "" | tee -a "$PROG_LOG"
    echo "=" | tee -a "$PROG_LOG"
    echo "[$(date)] policy=$POLICY  starting $DURATION_S-second stress" | tee -a "$PROG_LOG"
    echo "=" | tee -a "$PROG_LOG"

    POL_DIR_PHONE="$OUT_BASE_PHONE/$POLICY"

    # FA-off for vanilla too, so all policies share one compute path
    EXTRA_FLAGS=""
    if [ "$POLICY" = "vanilla" ]; then
        EXTRA_FLAGS="--no-fa-vanilla"
    fi

    adb shell "
mkdir -p $POL_DIR_PHONE
cd /data/local/tmp/endurkv

# Hard cool-down before each policy (no per-iteration cool during run)
sh scripts/phone_cool_then_run.sh \
   --out-dir $POL_DIR_PHONE --thresh-c $COOL_START_C --max-wait $COOL_MAX_WAIT_S -- true

# Long-running sensor sampler (5 Hz)
sh scripts/sample_sensors.sh --out $POL_DIR_PHONE/sensors.csv --hz $SAMPLE_HZ &
SAMPLER=\$!

# Stress loop: one full prefill plus 256-token decode per call, for DURATION_S seconds.
T_START=\$(date +%s)
ITER=0
echo \"iter,t_wall_s,exit,prefill_ms,decode_tps,peak_kv_mb,peak_rss_kb,evicted\" > $POL_DIR_PHONE/stress.csv

while true; do
    T_NOW=\$(date +%s)
    ELAPSED=\$((T_NOW - T_START))
    if [ \$ELAPSED -ge $DURATION_S ]; then break; fi
    ITER=\$((ITER+1))
    RUN_DIR=\$POL_DIR_PHONE/iter\$(printf %04d \$ITER)
    mkdir -p \$RUN_DIR

    LD_LIBRARY_PATH=$BIN_DIR $BIN_DIR/eviction_bench \
      --model $MODEL \
      --prompt $PROMPT_FILE \
      --prompt-id $PROMPT_ID \
      --policy $POLICY \
      --k-nominal $K_NOMINAL \
      --max-tokens 256 \
      --ctx-size $CTX \
      --seed 42 \
      --threads 4 \
      --n-gpu-layers $NGL \
      --n-batch $N_BATCH \
      --ubatch-size $UBATCH \
      --n-sink 4 \
      --greedy \
      $EXTRA_FLAGS \
      --out-csv \$RUN_DIR/steps.csv \
      --out-meta \$RUN_DIR/meta.json 2>\$RUN_DIR/stderr.log
    EXIT=\$?

    # Parse meta.json key fields and emit one row to stress.csv
    if [ -f \$RUN_DIR/meta.json ]; then
        PF=\$(grep -oE '\"prefill_ms\": *[0-9.]+'      \$RUN_DIR/meta.json | grep -oE '[0-9.]+$')
        DT=\$(grep -oE '\"decode_tps\": *[0-9.]+'      \$RUN_DIR/meta.json | grep -oE '[0-9.]+$')
        KV=\$(grep -oE '\"peak_kv_mb\": *[0-9.]+'      \$RUN_DIR/meta.json | grep -oE '[0-9.]+$')
        RS=\$(grep -oE '\"peak_rss_kb\": *[0-9.]+'     \$RUN_DIR/meta.json | grep -oE '[0-9.]+$')
        EV=\$(grep -oE '\"evicted_total_decode\": *[0-9]+' \$RUN_DIR/meta.json | grep -oE '[0-9]+$')
    else
        PF=0; DT=0; KV=0; RS=0; EV=0
    fi
    echo \"\$ITER,\$ELAPSED,\$EXIT,\$PF,\$DT,\$KV,\$RS,\$EV\" >> $POL_DIR_PHONE/stress.csv

    if [ \$EXIT -ne 0 ]; then
        echo \"  iter \$ITER FAILED exit=\$EXIT — continuing\" >&2
    fi
done

kill \$SAMPLER 2>/dev/null
wait \$SAMPLER 2>/dev/null

# Final thermal snapshot
dumpsys battery | grep -E 'level|temperature' > $POL_DIR_PHONE/final_battery.txt
cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -10 > $POL_DIR_PHONE/final_thermals.txt
echo \"  done iters=\$ITER duration=\$ELAPSED s\"
" 2>&1 | tee -a "$PROG_LOG"

    # Pull the policy's output
    LOCAL_DIR="$OUT_BASE_HOST/$POLICY"
    mkdir -p "$LOCAL_DIR"
    adb pull -p "$POL_DIR_PHONE/" "$LOCAL_DIR/" 2>&1 | tail -1 | tee -a "$PROG_LOG"

    # Quick sustained-vs-peak comparison
    if [ -f "$LOCAL_DIR/$POLICY/stress.csv" ]; then
        echo "$POLICY tok/s by 5-min window" | tee -a "$PROG_LOG"
        awk -F, 'NR>1 && $5>0 {
            bucket = int($2/300);
            sum[bucket]+=$5; n[bucket]++;
        } END {
            for (b=0; b<6; b++) if (n[b]>0)
                printf "    %d-%d min:  mean decode_tps = %.2f (n=%d)\n", b*5, (b+1)*5, sum[b]/n[b], n[b]
        }' "$LOCAL_DIR/$POLICY/stress.csv" | tee -a "$PROG_LOG"
    fi
done

{
echo ""
echo "[$(date)] wave3_stress_done"
echo "Local logs: $OUT_BASE_HOST"
echo ""
echo "Next: run scripts/host_plot_wave3_curves.py to generate the figures."
} | tee -a "$PROG_LOG"
