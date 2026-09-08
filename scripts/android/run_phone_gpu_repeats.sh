#!/bin/bash
# ============================================================================
# run_phone_gpu_repeats.sh -- error bars for the headline phone GPU cells
#                             (2026-08-02)
#
# WHY THIS IS NOT OPTIONAL.
# Every phone GPU cell so far is n=1, and we have a direct measurement of how
# noisy that is: llama1b_mukv_dfg and llama1b_mukv_nodfg differ ONLY in whether
# compaction runs, which happens strictly AFTER prefill -- so their prefill times
# must be identical. They came out 119.1s vs 130.9s, a 9.9% spread. That is the
# run-to-run variance of this device, and it is the same size as the Llama-1B
# decode speedup we would otherwise report (1.10x). A headline that inverts when
# a reviewer re-runs one cell is the most damaging thing we can ship, and we
# already had one claim ("eviction without compaction is slower than vanilla")
# that rested entirely on a single such cell and did not survive scrutiny.
#
# WHAT IT REPEATS. Three runs of each arm that carries a headline number:
#   Llama-1B {vanilla, muKV+compaction, muKV-compaction}  -- the marginal case
#   Phi-3    {vanilla, muKV+compaction, muKV-compaction}  -- the 2.39x claim
# SnapKV is excluded: at 1090s (llama1b) and 5468s (phi3) per run, three repeats
# would cost ~5.5h to tighten a number (0.14x-0.16x) whose sign is not in doubt.
# The K-sweep points are likewise single-run; they are a trend, not a headline.
#
# PROTOCOL IS IDENTICAL to run_phone_gpu_16k_wikitext.sh -- same prompt, ctx,
# generation length, q8_0 KV, cool gate before EVERY run (DDR<=35C, batt<=33C,
# charging off), watchdog muKV-only. Anything else and the repeats would not be
# repeats. Run index is appended to the tag so nothing overwrites run 1's cells,
# which are reused as the first sample.
# ============================================================================
set -u
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

VK=/data/local/tmp/endurkv/bin_vk_v2
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
OUT_HOST=/tmp/phone_gpu_16k; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/pgr_$(date +%Y%m%d_%H%M%S)
NRUN=${NRUN:-3}
adb_safe_shell "mkdir -p $OUT" < /dev/null
declare -A M=( [llama1b]=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
               [phi3]=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf )
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
flags_for(){ case "$1" in
  vanilla)    echo "--policy vanilla" ;;
  mukv_dfg)   echo "$MU --force-defrag" ;;
  mukv_nodfg) echo "$MU --no-defrag" ;;
esac; }

cell(){ local MT=$1 POL=$2 R=$3; local TAG="${MT}_${POL}_r${R}"; local PD=$OUT/$TAG
  [ -f "$OUT_HOST/$TAG/meta.json" ] && { echo "  [$TAG] cached"; return; }
  adb_safe_shell "mkdir -p $PD" < /dev/null
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  echo "$CG" | tail -1
  case "$CG" in *"cool ddr="*) : ;; *) echo "  [SKIP-HOT] $TAG"; return ;; esac
  case "$POL" in mukv_*) adb_safe_shell "su -c 'rm -f /data/local/tmp/gpu_wd.stop; nohup sh /data/local/tmp/gpu_watchdog_v5_real.sh $PD/gpu_wd_v5.log /data/local/tmp/gpu_wd.stop >/dev/null 2>&1 &'" < /dev/null ;; esac
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$VK timeout ${TMO:-9000} $VK/eviction_bench --prompt $P --prompt-id $TAG \
    --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 --n-batch 512 --n-ubatch 64 \
    --model ${M[$MT]} --seed 42 --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 \
    --cache-type-k f16 --cache-type-v f16 $(flags_for $POL) \
    --out-meta $PD/meta.json --out-gen /dev/null --out-csv /dev/null > $PD/out 2> $PD/err" < /dev/null
  adb_safe_shell "su -c 'touch /data/local/tmp/gpu_wd.stop; pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  python3 - "$OUT_HOST/$TAG" "$TAG" <<'PY'
import json,re,sys,os
f=os.path.join(sys.argv[1],'meta.json')
if not os.path.exists(f): print("  [%-24s] FAILED"%sys.argv[2]); raise SystemExit
s=open(f).read(); s=re.sub(r':\s*-?nan\b',': NaN',s); s=re.sub(r':\s*-?inf\b',': Infinity',s); j=json.loads(s)
print("  [%-24s] prefill=%7.1fs decode=%7.1fs tps=%6.2f compact=%s"%(
    sys.argv[2],j['prefill_ms']/1000,j['decode_ms']/1000,j.get('decode_tps') or 0,j.get('compaction_applied')))
PY
}
# runs 2..N; run 1 is the existing un-suffixed cell from the original sweep
for R in $(seq 2 "$NRUN"); do
  for MT in llama1b phi3; do
    for POL in vanilla mukv_dfg mukv_nodfg; do cell "$MT" "$POL" "$R"; done
  done
done
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null
echo "[$(date +%H:%M:%S)] PHONE_GPU_REPEATS_DONE -> $OUT_HOST"
