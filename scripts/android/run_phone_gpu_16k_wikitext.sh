#!/bin/bash
# ============================================================================
# Phone GPU (Adreno 840) -- FULL 16K CONTEXT, WikiText  (2026-08-02)
#
# WHY THIS EXISTS (and why it is not run_phone_gpu_matrix.sh).
# That matrix decoded only 256 tokens: 12K prompt + 256 gen. On Adreno prefill
# is ~95% of such a run (Phi-3: 610s prefill vs 35s decode), so a 2.12x DECODE
# win showed up as 0.99x wall. Eviction cannot shrink prefill -- it needs a
# long generation to pay off. This script fills the context instead:
#   12K WikiText prompt + 4096 generated tokens = 16384 = the full ctx.
# That is the regime the CPU WikiText table already reports, so the phone GPU
# becomes directly comparable to the phone CPU and to the RTX max-workload set.
#
# FOUR ARMS PER MODEL (the middle two are the compaction A/B the draft needs):
#   vanilla     full cache, NO watchdog
#   mukv_dfg    frozen muKV + --force-defrag  (compaction ON)  + GPU watchdog v5
#   mukv_nodfg  frozen muKV + --no-defrag     (compaction OFF) + GPU watchdog v5
#   snapkv      canonical SnapKV (window 16, avgpool-5), NO watchdog
# The watchdog is muKV-ONLY by design -- thermal-aware inference is muKV's own
# mechanism; handing it to a baseline would be lending it our eviction.
# GPU watchdog = gpu_watchdog_v5_real.sh (07-20). v3/v4 misanchored at idle-warm
# temps and capped the clock to 826 MHz for nothing; v5 anchors at the measured
# deep-throttle trigger and stays dormant at 1200 MHz on a cool run. This run
# therefore also produces the v5 GPU row the draft is still missing.
#
# MODELS: llama1b + phi3 only. gemma2b and bonsai8b abort under FA-on on this
# driver ("vk::DeviceLostError: vk::Queue::submit") for BOTH vanilla and muKV --
# a driver/kernel fault, not a policy or capacity effect. They are excluded here
# rather than reported as a muKV failure. Tracked separately.
#
# CORRECTED 2026-08-04: f16 KV, not q8_0. Quantized KV is NUMERICALLY BROKEN on
# this Adreno/Vulkan build: with --cache-type q8_0 the model emits random tokens
# ("Observ Observ ... Alley Alley Bundy Bundy" for a prompt whose answer is
# "Miller v. California"), and teacher-forced NLL is 12.18 against ln(vocab)=11.76
# -- i.e. worse than uniform. The SAME binary, model, prompt and seed at f16
# answers correctly, so the backend is fine and the quantized-KV path is not.
# This invalidated the first pass in the most confusing possible way: vanilla and
# muKV ran q8_0 (garbage) while SnapKV was forced to f16 by the FA-off path
# (valid), so every ratio divided a good run by a bad one. f16 everywhere is both
# correct AND quantization-matched, like the LongBench table.
#
# Cool gate before EVERY cell: DDR<=35C, battery<=33C, charging OFF while
# cooling. A failed gate SKIPS the cell -- it never runs hot.
# ============================================================================
set -u
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue   # dead port + adb = squatting server that breaks ssh -R
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

VK=/data/local/tmp/endurkv/bin_vk_v2
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt          # 12K-token WikiText slice
GEN=${GEN:-4096}                                          # 12K + 4096 = 16384 = full ctx
OUT_HOST=/tmp/phone_gpu_16k; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/pg16k_$(date +%Y%m%d_%H%M%S)
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
adb_safe_shell "mkdir -p $OUT" < /dev/null

declare -A M=( [llama1b]=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
               [phi3]=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf )

cell(){ local TAG=$1 MP=$2 WD=$3; shift 3; local PD=$OUT/$TAG
  [ -f "$OUT_HOST/$TAG/meta.json" ] && { echo "  [$TAG] cached"; return; }
  adb_safe_shell "mkdir -p $PD" < /dev/null
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  echo "$CG" | tail -1
  case "$CG" in *"cool ddr="*) : ;; *) echo "  [SKIP-HOT] $TAG -- gate failed, cell not run"; return ;; esac
  if [ "$WD" = wd ]; then   # muKV ONLY
    adb_safe_shell "su -c 'rm -f /data/local/tmp/gpu_wd.stop; nohup sh /data/local/tmp/gpu_watchdog_v5_real.sh $PD/gpu_wd_v5.log /data/local/tmp/gpu_wd.stop >/dev/null 2>&1 &'" < /dev/null
  fi
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$VK timeout ${TMO:-9000} $VK/eviction_bench --prompt $P --prompt-id $TAG \
    --eval-mode gen --max-tokens $GEN --ignore-eos --ctx-size 16384 --n-batch 512 --n-ubatch 64 \
    --model $MP --seed 42 --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 \
    --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $PD/meta.json --out-gen $PD/gen.txt --out-csv /dev/null > $PD/out 2> $PD/err" < /dev/null
  adb_safe_shell "su -c 'touch /data/local/tmp/gpu_wd.stop; pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  python3 - "$OUT_HOST/$TAG" "$TAG" <<'PY'
import json,re,sys,os
d,t=sys.argv[1],sys.argv[2]; f=os.path.join(d,'meta.json')
if not os.path.exists(f):
    e=os.path.join(d,'err'); m=open(e).read().strip().split('\n')[-1][:56] if os.path.exists(e) else '?'
    print("  [%-16s] FAILED %s"%(t,m)); raise SystemExit
s=open(f).read(); s=re.sub(r':\s*-?nan\b',': NaN',s); s=re.sub(r':\s*-?inf\b',': Infinity',s); j=json.loads(s)
wd=os.path.join(d,'gpu_wd_v5.log'); nt=0
if os.path.exists(wd): nt=sum(1 for L in open(wd) if 'tier=' in L)
print("  [%-16s] prefill=%7.1fs decode=%7.1fs wall=%7.1fs tps=%6.2f ret=%7.1fMiB compact=%s wd_tiers=%d"%(
    t,j.get('prefill_ms',0)/1000,j.get('decode_ms',0)/1000,j.get('total_ms',0)/1000,
    j.get('decode_tps') or 0,j.get('retained_kv_mib') or 0,j.get('compaction_applied'),nt))
PY
}

for k in llama1b phi3; do
  echo "[$(date +%H:%M:%S)] ================= $k (12K prompt + ${GEN} gen = full 16K ctx) ================="
  cell ${k}_vanilla    "${M[$k]}" nowd --policy vanilla
  cell ${k}_mukv_dfg   "${M[$k]}" wd   $MU --force-defrag
  cell ${k}_mukv_nodfg "${M[$k]}" wd   $MU --no-defrag
  cell ${k}_snapkv     "${M[$k]}" nowd --policy snapkv --obs-window 16 --snapkv-kernel 5 --n-sink 0
done
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null
echo "[$(date +%H:%M:%S)] PHONE_GPU_16K_DONE -> $OUT_HOST"
