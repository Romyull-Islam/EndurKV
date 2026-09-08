#!/bin/bash
# ============================================================================
# run_phone_gpu_complete.sh -- finish the phone GPU evidence table (2026-08-02)
#
# WHAT IS ALREADY DONE (run_phone_gpu_16k_wikitext.sh + _ksweep.sh):
#   timed cells, Llama-1B and Phi-3 x {vanilla, muKV+compaction, muKV-compaction,
#   SnapKV}, plus Phi-3 muKV at K=512/256. Those give prefill/decode/wall/tps/
#   retained/energy. They do NOT give a quality number: they run --ignore-eos,
#   which produces degenerate repetition, so their generated text is unscorable.
#
# WHAT THIS ADDS:
#   PASS A -- PPL on every existing arm, both metrics the draft distinguishes:
#     PPL_dis : teacher-forced on wiki_eval_disjoint.txt (verified to share no
#               200-char window with the prompt) -> measures PREDICTION. This is
#               the slice on which the draft claims every policy matches the full
#               cache, and it is the honest quality column for the speedup table.
#     PPL_rec : teacher-forced on wiki_eval_overlap.txt, a prefix of the prompt
#               itself -> measures VERBATIM RECALL of retained text. Eviction is
#               *supposed* to lose here; reporting only this would misrepresent
#               eviction as a language-modelling regression.
#     Both slices are checked into benchmarks/ppl/ with their offsets, because the
#     previous disjoint slice lived in a scratchpad dir that no longer exists.
#   PASS B -- the four policies missing from the GPU table (StreamingLLM, Ada-KV,
#     H2O, TOVA), so the phone GPU carries the same policy set as the NIAH table.
#
# GATING. PASS A is quality-only: greedy + fixed seed + teacher forcing means the
# scored logits do not depend on clock speed, so no cool gate (nothing timed is
# reported from these cells). PASS B cells ARE timed and DO take the full gate
# (DDR<=35C, batt<=33C, charging off). The watchdog stays muKV-only throughout.
#
# MODELS. Llama-1B and Phi-3 only: Gemma-2B and Bonsai-8B abort on this driver
# under FA-on (vk::DeviceLostError) for vanilla and muKV alike. That is a driver
# fault, not a policy limit, and all four run on the phone CPU.
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
EV_DIS=/data/local/tmp/endurkv/corpora/wiki_eval_disjoint.txt
EV_REC=/data/local/tmp/endurkv/corpora/wiki_eval_overlap.txt
OUT_HOST=/tmp/phone_gpu_16k; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/pgc_$(date +%Y%m%d_%H%M%S)
# PASS=A (PPL only) | B (missing policies only) | all. Added 2026-08-02 so the
# quality column and the error-bar repeats can be sequenced ahead of the long
# FA-off policy cells, which cost ~6h and are completeness rather than defensibility.
PASS=${PASS:-all}
adb_safe_shell "mkdir -p $OUT" < /dev/null
declare -A M=( [llama1b]=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
               [phi3]=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf )
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
flags_for(){ case "$1" in
  vanilla)      echo "--policy vanilla" ;;
  mukv_dfg)     echo "$MU --force-defrag" ;;
  mukv_nodfg)   echo "$MU --no-defrag" ;;
  # WikiText has no published SnapKV setting, so the baseline gets the shipped
  # FasterDecoding default (window 64, avgpool-5). One SnapKV per table, always
  # the most appropriate published configuration for that benchmark.
  snapkv)       echo "--policy snapkv --obs-window 64 --snapkv-kernel 5 --n-sink 0" ;;
  streamingllm) echo "--policy streamingllm --n-sink 4" ;;
  adakv)        echo "--policy adakv --n-sink 0 --obs-window 32" ;;
  h2o)          echo "--policy h2o --n-sink 0 --obs-window 64" ;;
  tova)         echo "--policy tova" ;;
esac; }
is_mukv(){ case "$1" in mukv_dfg|mukv_nodfg) return 0;; *) return 1;; esac; }

# ---------- PASS A: PPL (no gate -- quality only, see header) ----------
ppl_cell(){ local MT=$1 POL=$2 KIND=$3 EV=$4; local TAG="${MT}_${POL}_ppl_${KIND}"; local PD=$OUT/$TAG
  [ -f "$OUT_HOST/$TAG/meta.json" ] && { echo "  [$TAG] cached"; return; }
  adb_safe_shell "mkdir -p $PD" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$VK timeout ${TMO:-9000} $VK/eviction_bench --prompt $P --prompt-id $TAG \
    --eval-mode ppl --eval-text $EV --max-tokens 512 --ignore-eos --ctx-size 16384 \
    --n-batch 512 --n-ubatch 64 --model ${M[$MT]} --seed 42 --threads 4 --n-gpu-layers 99 \
    --greedy --k-nominal 1024 --cache-type-k f16 --cache-type-v f16 $(flags_for $POL) \
    --out-meta $PD/meta.json --out-gen /dev/null --out-csv /dev/null > $PD/out 2> $PD/err" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  python3 - "$OUT_HOST/$TAG" "$TAG" <<'PY'
import json,re,sys,os
f=os.path.join(sys.argv[1],'meta.json')
if not os.path.exists(f):
    e=os.path.join(sys.argv[1],'err')
    print("  [%-28s] FAILED %s"%(sys.argv[2], open(e,errors='replace').read().strip().split('\n')[-1][:50] if os.path.exists(e) else '?')); raise SystemExit
s=open(f).read(); s=re.sub(r':\s*-?nan\b',': NaN',s); s=re.sub(r':\s*-?inf\b',': Infinity',s); j=json.loads(s)
print("  [%-28s] ppl=%9.4f  nll=%8.4f  retained=%8.1f MiB"%(sys.argv[2],
      j.get('perplexity') or float('nan'), j.get('mean_nll') or float('nan'),
      (j.get('retained_kv_bytes') or 0)/1048576))
PY
}
if [ "$PASS" = A ] || [ "$PASS" = all ]; then
echo "[$(date +%H:%M:%S)] ===== PASS A: PPL (disjoint = prediction, overlap = verbatim recall) ====="
for MT in llama1b phi3; do
  for POL in vanilla mukv_dfg mukv_nodfg snapkv; do
    ppl_cell "$MT" "$POL" dis "$EV_DIS"
    ppl_cell "$MT" "$POL" rec "$EV_REC"
  done
done

fi
# ---------- PASS B: timed cells for the missing policies (FULL cool gate) ----------
timed_cell(){ local MT=$1 POL=$2; local TAG="${MT}_${POL}"; local PD=$OUT/$TAG
  [ -f "$OUT_HOST/$TAG/meta.json" ] && { echo "  [$TAG] cached"; return; }
  adb_safe_shell "mkdir -p $PD" < /dev/null
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  echo "$CG" | tail -1
  case "$CG" in *"cool ddr="*) : ;; *) echo "  [SKIP-HOT] $TAG"; return ;; esac
  if is_mukv "$POL"; then
    adb_safe_shell "su -c 'rm -f /data/local/tmp/gpu_wd.stop; nohup sh /data/local/tmp/gpu_watchdog_v5_real.sh $PD/gpu_wd_v5.log /data/local/tmp/gpu_wd.stop >/dev/null 2>&1 &'" < /dev/null
  fi
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$VK timeout ${TMO:-14400} $VK/eviction_bench --prompt $P --prompt-id $TAG \
    --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 --n-batch 512 --n-ubatch 64 \
    --model ${M[$MT]} --seed 42 --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 \
    --cache-type-k f16 --cache-type-v f16 $(flags_for $POL) \
    --out-meta $PD/meta.json --out-gen $PD/gen.txt --out-csv /dev/null > $PD/out 2> $PD/err" < /dev/null
  adb_safe_shell "su -c 'touch /data/local/tmp/gpu_wd.stop; pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  python3 - "$OUT_HOST/$TAG" "$TAG" <<'PY'
import json,re,sys,os
f=os.path.join(sys.argv[1],'meta.json')
if not os.path.exists(f):
    e=os.path.join(sys.argv[1],'err')
    print("  [%-22s] FAILED %s"%(sys.argv[2], open(e,errors='replace').read().strip().split('\n')[-1][:52] if os.path.exists(e) else '?')); raise SystemExit
s=open(f).read(); s=re.sub(r':\s*-?nan\b',': NaN',s); s=re.sub(r':\s*-?inf\b',': Infinity',s); j=json.loads(s)
print("  [%-22s] prefill=%7.1fs decode=%7.1fs wall=%7.1fs tps=%6.2f ret=%8.1fMiB compact=%s"%(
    sys.argv[2],j['prefill_ms']/1000,j['decode_ms']/1000,j['total_ms']/1000,
    j.get('decode_tps') or 0,(j.get('retained_kv_bytes') or 0)/1048576,j.get('compaction_applied')))
PY
}
# llama1b first: same policy set, ~4x cheaper per cell, so the table becomes
# complete for one model before the expensive Phi-3 FA-off cells start.
if [ "$PASS" = B ] || [ "$PASS" = all ]; then
echo "[$(date +%H:%M:%S)] ===== PASS B: missing policies, timed (cool gate on every cell) ====="
for MT in llama1b phi3; do
  for POL in streamingllm adakv h2o tova; do timed_cell "$MT" "$POL"; done
done
# PPL for the new policies too, so every row of the table has a quality number
for MT in llama1b phi3; do
  for POL in streamingllm adakv h2o tova; do
    ppl_cell "$MT" "$POL" dis "$EV_DIS"; ppl_cell "$MT" "$POL" rec "$EV_REC"
  done
done
fi
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null
echo "[$(date +%H:%M:%S)] PHONE_GPU_COMPLETE_DONE -> $OUT_HOST"
