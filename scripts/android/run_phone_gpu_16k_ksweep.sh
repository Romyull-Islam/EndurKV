#!/bin/bash
# run_phone_gpu_16k_ksweep.sh: Phi-3 on the phone GPU at 16K context, muKV at lower K than the
# K=1024 run of run_phone_gpu_16k_wikitext.sh. Smaller K raises the compression ratio and the
# bandwidth roofline speedup S = (W + KV) / (W + KV/r).
# Vanilla is not rerun, it does not depend on K and the K=1024 run already measured it.
# Watchdog v5 on muKV cells only, cool gate before every cell.
set -u
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
VK=/data/local/tmp/endurkv/bin_vk_v2
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
MP=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
OUT_HOST=/tmp/phone_gpu_16k; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/pg16k_ks_$(date +%Y%m%d_%H%M%S)
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
adb_safe_shell "mkdir -p $OUT" < /dev/null

for K in 512 256; do
  TAG=phi3_mukv_dfg_k$K; PD=$OUT/$TAG
  [ -f "$OUT_HOST/$TAG/meta.json" ] && { echo "  [$TAG] cached"; continue; }
  adb_safe_shell "mkdir -p $PD" < /dev/null
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  echo "$CG" | tail -1
  case "$CG" in *"cool ddr="*) : ;; *) echo "  [SKIP-HOT] $TAG"; continue ;; esac
  adb_safe_shell "su -c 'rm -f /data/local/tmp/gpu_wd.stop; nohup sh /data/local/tmp/gpu_watchdog_v5_real.sh $PD/gpu_wd_v5.log /data/local/tmp/gpu_wd.stop >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$VK timeout 9000 $VK/eviction_bench --prompt $P --prompt-id $TAG \
    --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 --n-batch 512 --n-ubatch 64 \
    --model $MP --seed 42 --threads 4 --n-gpu-layers 99 --greedy --k-nominal $K \
    --cache-type-k q8_0 --cache-type-v q8_0 $MU --force-defrag \
    --out-meta $PD/meta.json --out-gen $PD/gen.txt --out-csv /dev/null > $PD/out 2> $PD/err" < /dev/null
  adb_safe_shell "su -c 'touch /data/local/tmp/gpu_wd.stop; pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  python3 - "$OUT_HOST/$TAG" "$TAG" "$OUT_HOST/phi3_vanilla" <<'PY'
import json,re,sys,os
def m(p):
    f=os.path.join(p,'meta.json')
    if not os.path.exists(f): return None
    s=open(f).read(); s=re.sub(r':\s*-?nan\b',': NaN',s); s=re.sub(r':\s*-?inf\b',': Infinity',s); return json.loads(s)
j=m(sys.argv[1]); t=sys.argv[2]; v=m(sys.argv[3])
if not j:
    e=os.path.join(sys.argv[1],'err')
    print("  [%s] FAILED %s"%(t, open(e).read().strip().split('\n')[-1][:56] if os.path.exists(e) else '?'))
else:
    dx = v['decode_ms']/j['decode_ms'] if v else 0
    wx = v['total_ms']/j['total_ms'] if v else 0
    print("  [%-20s] prefill=%7.1fs decode=%7.1fs wall=%7.1fs tps=%6.2f dec=%.2fx wall=%.2fx ret=%7.1fMiB compact=%s"%(
        t,j['prefill_ms']/1000,j['decode_ms']/1000,j['total_ms']/1000,j.get('decode_tps') or 0,
        dx,wx,(j.get('retained_kv_bytes') or 0)/1048576.0,j.get('compaction_applied')))
PY
done
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null
echo "PHONE_GPU_KSWEEP_DONE -> $OUT_HOST"
