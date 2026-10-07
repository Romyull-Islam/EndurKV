#!/bin/bash
# run_longbench_sllm_ownbudget.sh: LongBench (qasper, hotpotqa) with StreamingLLM at its
# own documented budget, start_size 4 plus recent_size 2000 (K=2004, mit-han-lab/streaming-llm).
# Same binary (CPU, eviction_bench_v10), prompts, ctx 16384 and max_gen as the K=1024 cells.
# No cool gate: F1 depends only on the keep-set under greedy decoding, so quote no timing
# or energy from these cells. gemma-2-2b trains at ctx 8192 but runs at 16384 here, as before.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/endurkv/bin_cpu/eviction_bench_v10
OUT=/data/local/tmp/endurkv/logs/lbsllm_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/lb_sllm_own; mkdir -p $HOST
N_SAMPLES=${N_SAMPLES:-5}
adb_safe_shell "mkdir -p $OUT" < /dev/null
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM

for task in qasper hotpotqa; do
  for i in $(seq 0 $((N_SAMPLES - 1))); do
    idx=$(printf "%03d" $i)
    src=/tmp/longbench_adaptive_3x3/phi3_vanilla_${task}/prompt_${idx}.txt
    [ -f "$src" ] && adb push "$src" "$OUT/${task}_${idx}.txt" < /dev/null >/dev/null 2>&1
  done
done
echo "prompts pushed"

get_max_gen(){ case $1 in qasper) echo 128;; hotpotqa) echo 32;; *) echo 64;; esac; }

for task in qasper hotpotqa; do
  MG=$(get_max_gen $task)
  for i in $(seq 0 $((N_SAMPLES - 1))); do
    idx=$(printf "%03d" $i)
    for MNAME in phi3 llama1b gemma2b; do
      case $MNAME in
        phi3)    M=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf ;;
        llama1b) M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf ;;
        gemma2b) M=/data/local/tmp/endurkv/models/gemma-2-2b-it-Q4_K_M.gguf ;;
      esac
      CELL="${MNAME}_sllm2004_${task}_${idx}"
      D=$HOST/$CELL; [ -f "$D/gen.txt" ] && { echo "  [$CELL] cached"; continue; }
      mkdir -p "$D"
      echo "[$(date +%H:%M:%S)] $CELL"
      adb_safe_shell "mkdir -p $OUT/$CELL; LD_LIBRARY_PATH=/data/local/tmp/endurkv/bin_cpu timeout 900 $BIN \
        --prompt $OUT/${task}_${idx}.txt --prompt-id $CELL --eval-mode gen --max-tokens $MG \
        --ignore-eos --ctx-size 16384 --model $M --seed 42 --threads 4 --n-gpu-layers 0 \
        --n-batch 512 --ubatch-size 64 --greedy \
        --policy streamingllm --k-nominal 2004 --n-sink 4 --cache-type-k f16 --cache-type-v f16 \
        --out-meta $OUT/$CELL/meta.json --out-gen $OUT/$CELL/gen.txt --out-csv /dev/null \
        > /dev/null 2> $OUT/$CELL/err" < /dev/null
      adb_safe_pull "$OUT/$CELL/gen.txt"   "$D/gen.txt"   >/dev/null 2>&1
      adb_safe_pull "$OUT/$CELL/meta.json" "$D/meta.json" >/dev/null 2>&1
      [ -f "$D/gen.txt" ] && echo "    ok" || echo "    FAILED"
    done
  done
done
echo LB_SLLM_OWN_DONE
