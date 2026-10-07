#!/bin/bash
# Phone GPU (Adreno 840), full 16K context on WikiText: 12K prompt plus 4096 generated.
# A 256-token decode leaves prefill at ~95% of the run, which hides any decode gain.
#
# Arms per model: vanilla, muKV with compaction on (--force-defrag) and off (--no-defrag),
# and canonical SnapKV (window 16, avgpool 5). Only the muKV arms run the GPU watchdog
# (gpu_watchdog_v5_real.sh). Llama-1B and Phi-3 only: gemma2b and bonsai8b hit
# vk::DeviceLostError under FA-on on this driver for vanilla and muKV alike.
#
# KV is f16 because q8_0 KV produces garbage tokens on this Adreno/Vulkan build.
# Cool gate before every cell (DDR<=35C, battery<=33C, charging off). A failed gate skips it.
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
