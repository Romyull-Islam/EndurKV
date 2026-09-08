#!/bin/bash
# ============================================================================
# run_remaining_queue.sh -- everything still required after the Phi-3 GPU table.
# (2026-08-25).  Runs STRICTLY AFTER /tmp/phi3_gpu_complete_DONE: one campaign
# owns the phone at a time. Two concurrent campaigns once cooked a cell to 57 C
# and invalidated it, so this waits rather than races.
#
# QUEUE, in the order the paper needs them:
#
#  A. COMPACTION F1 ABLATION (the question that has no data at all).
#     Compaction is worth 1.10-2.37x tok/s at IDENTICAL cell count and its PPL
#     moves 1.2% -- but no LongBench/NIAH cell has ever run with --no-defrag, so
#     "same accuracy" is unproven on a task metric. 15 hotpotqa prompts x
#     {compacted, not} = 30 cells. Quality-only: NO cool gate (nothing timed is
#     quoted), matching how every other lb_native cell was produced.
#
#  B. StreamingLLM ON THE CURRENT CPU BUILD.
#     Table 1's StreamingLLM row is the last one from bin_cpu_sol2; every other
#     row is bin_cpu_kd. Absolute energy is not comparable across builds, so this
#     one row is quoted from a different binary than the vanilla it is divided by.
#     One cell removes the asterisk.
#
#  C. CPU GRADEABILITY SPOT-CHECK.
#     Every definitive CPU cell wrote --out-gen /dev/null, so the CPU tables can
#     never be <unk>-graded. The scripts are fixed going forward; this re-runs
#     THREE representative cells (vanilla / muKV / H2O) with text kept, enough to
#     show the CPU path is clean without re-running all fourteen.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
exec 9>/tmp/.endurkv_queue.lock; flock -n 9 || { LOG "another queue holds the lock"; exit 0; }

LOG "waiting for the Phi-3 GPU campaign to finish ..."
while [ ! -f /tmp/phi3_gpu_complete_DONE ]; do sleep 60; done
LOG "GPU campaign done -- starting queue"

CB=/data/local/tmp/endurkv/bin_cpu_kd
M1=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
DEV=/data/local/tmp/rq_$(date +%Y%m%d_%H%M%S)
adb_safe_shell "mkdir -p $DEV" < /dev/null >/dev/null 2>&1

# ---------- A. compaction F1 ablation on LongBench hotpotqa ------------------
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
for ARM in compact nocompact; do
  case $ARM in compact) EX="--compact-inplace";; nocompact) EX="--no-defrag";; esac
  for i in $(seq 0 14); do
    idx=$(printf "%03d" $i)
    src=/tmp/longbench_adaptive_3x3/phi3_vanilla_hotpotqa/prompt_${idx}.txt
    [ -f "$src" ] || continue
    CELL="mukv_${ARM}_hotpotqa_${idx}"; D=/tmp/lb_compaction/$CELL
    [ -s "$D/gen.txt" ] && { LOG "$CELL cached"; continue; }
    mkdir -p "$D"
    timeout 180 adb push "$src" "$DEV/hp_${idx}.txt" < /dev/null >/dev/null 2>&1
    LOG "A: $CELL"
    adb_safe_shell "mkdir -p $DEV/$CELL; rm -f $DEV/$CELL/.done; setsid nohup sh -c \"timeout 1800 env LD_LIBRARY_PATH=$CB $CB/eviction_bench \
      --prompt $DEV/hp_${idx}.txt --prompt-id $CELL --eval-mode gen --max-tokens 32 --ctx-size 16384 \
      --model $M1 --seed 42 --threads 4 --n-gpu-layers 0 --n-batch 512 --ubatch-size 64 --greedy \
      --cache-type-k f16 --cache-type-v f16 $MU --k-nominal 1024 $EX \
      --out-meta $DEV/$CELL/meta.json --out-gen $DEV/$CELL/gen.txt --out-csv /dev/null > /dev/null 2> $DEV/$CELL/err ; \
      echo DONE > $DEV/$CELL/.done\" >/dev/null 2>&1 &" < /dev/null
    w=0; while [ $w -lt 1900 ]; do
      adb_safe_shell "[ -f $DEV/$CELL/.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
      sleep 20; w=$((w+20)); done
    adb_safe_pull "$DEV/$CELL/gen.txt"   "$D/gen.txt"   >/dev/null 2>&1
    adb_safe_pull "$DEV/$CELL/meta.json" "$D/meta.json" >/dev/null 2>&1
  done
done
LOG "A complete"
touch /tmp/lb_compaction_DONE

# ---------- B + C. timed CPU cells (cool-gated, generations KEPT) ------------
OUT=/tmp/def_cpu_v2; mkdir -p $OUT
adb_safe_shell "mkdir -p $DEV/wt" < /dev/null >/dev/null 2>&1
timeout 180 adb push /tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad/wikitext_16k_p12k_d4k.txt "$DEV/wt/prompt.txt" < /dev/null >/dev/null 2>&1

wtcell(){
  local TAG=$1; shift
  [ -s "$OUT/$TAG/meta.json" ] && { LOG "$TAG cached"; return; }
  mkdir -p "$OUT/$TAG"
  LOG "cooling for $TAG ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) LOG "  $CG";; *) LOG "  [SKIP-HOT] $TAG"; return;; esac
  BV=$(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/voltage_now'" < /dev/null 2>/dev/null | tr -d ' \r')
  BS=$(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/status'" < /dev/null 2>/dev/null | tr -d ' \r')
  echo "batt_voltage_uv=$BV status=$BS" > "$OUT/$TAG/start_power.txt"
  case "$BV" in ''|*[!0-9]*) : ;; *)
    if [ "$BV" -gt 4100000 ]; then
      LOG "  [WARN] $TAG starts at $((BV/1000)) mV -- Table 1 cells ran at ~3800 mV; timings NOT comparable"
    fi ;;
  esac
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $DEV/$TAG.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 5400 env LD_LIBRARY_PATH=$CB $CB/eviction_bench \
    --prompt $DEV/wt/prompt.txt --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 \
    --model $M1 --seed 42 --threads 4 --n-gpu-layers 0 --n-batch 512 --ubatch-size 64 --greedy \
    --cache-type-k f16 --cache-type-v f16 --k-nominal 1024 $* \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > $DEV/$TAG.out 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  local w=0; while [ $w -lt 5600 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 30; w=$((w+30)); done
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.json" "$OUT/$TAG/meta.json"   >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.gen"  "$OUT/$TAG/gen.txt"     >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.csv"  "$OUT/$TAG/sensors.csv" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.err"  "$OUT/$TAG/err.txt"     >/dev/null 2>&1
  [ -s "$OUT/$TAG/gen.txt" ] && LOG "  [$TAG] ok" || LOG "  [$TAG] NO OUTPUT"
}
# B: the mixed-build row, now on bin_cpu_kd
wtcell streamingllm --policy streamingllm --n-sink 4 --k-nominal 2000
# C: gradeability spot-check
wtcell vanilla_g --policy vanilla
wtcell mukv_g    $MU --k-nominal 1024
wtcell h2o_g     --policy h2o --n-sink 0 --k-nominal 1948
LOG "QUEUE_DONE"
touch /tmp/remaining_queue_DONE
