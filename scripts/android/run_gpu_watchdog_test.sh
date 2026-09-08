#!/bin/bash
# run_gpu_watchdog_test.sh -- does a GPU ladder that steps before the vendor limiter beat the
# limiter on the request that trips it? (2026-09-07)
#   wd_off : muKV, GPU 1200 MHz uncapped, 4096 tokens; the vendor limiter handles the heat
#   wd_on  : same, with gpu_watchdog.sh stepping 1050 / 967 / 902 at DDR 60 / 62 / 63.5 C
# n=3 per arm, interleaved, cooled (DDR <= 35 C, battery <= 33 C), charging off, CPU caps pinned,
# bench pinned to the big cores. Each request launched detached on the phone and polled.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-3C15B8003ZA00000}
export ADB_CALL_TIMEOUT=1500
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
ROOT=/data/local/tmp/endurkv; BIN=/data/local/tmp/ukv; P=$ROOT/corpora/prompt_12k.txt
MODEL=$ROOT/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024"
HOST=${HOST:-/tmp/gpu_watchdog_test}; mkdir -p $HOST
CPU_PRIME=1497600; CPU_REST=1785600
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/gpu_watchdog.sh $ROOT/gpu_watchdog.sh < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
SAVED=$(adb_safe_shell "su -c 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo \$(basename \$p) \$(cat \$p/scaling_min_freq) \$(cat \$p/scaling_max_freq); done'" < /dev/null | tr -d '\r')
restore_all(){ echo "$SAVED" | while read pol mn mx; do [ -n "${pol:-}" ] && adb_safe_shell "su -c 'echo $mx > /sys/devices/system/cpu/cpufreq/$pol/scaling_max_freq; echo $mn > /sys/devices/system/cpu/cpufreq/$pol/scaling_min_freq'" < /dev/null; done
  adb_safe_shell "su -c 'echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel; echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk'" < /dev/null; }
pin_cpu(){ adb_safe_shell "su -c 'echo $CPU_REST > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; echo $CPU_REST > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; echo $CPU_PRIME > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq; echo $CPU_PRIME > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq'" < /dev/null; }
wd_stop(){ adb_safe_shell "su -c 'pkill -TERM -f gpu_watchdo[g].sh; sleep 1; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel'" < /dev/null >/dev/null 2>&1; }
cleanup(){ wd_stop; adb_safe_shell "su -c 'pkill -f sample_sensor[s]; pkill -f eviction_benc[h]; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; restore_all; }
trap cleanup EXIT INT TERM
settle(){ for a in 1 2 3; do
    adb_safe_shell "su -c '. $ROOT/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    local _rd; _rd=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $_rd|awk '{print int($1/1000)}'); local _sb=$(echo $_rd|awk '{print int($2/1000)}'); echo "    start ddr=${_sd}C batt=${_sb}C"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0; done; return 1; }
cell(){ local TAG=$1 ARM=$2; local D=$HOST/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }; mkdir -p $D
  echo "[$(date +%H:%M:%S)] cooling for $TAG"; settle || { echo "  [SKIP-HOT] $TAG"; return; }; pin_cpu
  adb_safe_shell "su -c 'rm -rf $ROOT/wdtest/$TAG; mkdir -p $ROOT/wdtest/$TAG; echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel; echo 1200000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk; rm -f $ROOT/gpu_watchdog.log'" < /dev/null >/dev/null 2>&1
  [ "$ARM" = on ] && adb_safe_shell "su -c 'nohup sh $ROOT/gpu_watchdog.sh >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
  echo "[$(date +%H:%M:%S)] running $TAG ($ARM)"
  adb_safe_shell "su -c 'pkill -f sample_sensor[s]'" < /dev/null >/dev/null 2>&1
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/sample_sensors.sh --out $ROOT/wdtest/$TAG/sensors.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
  adb_safe_shell "su -c 'cd $BIN && LD_LIBRARY_PATH=$BIN nohup taskset f0 nice -n -20 ./eviction_bench --model $MODEL --prompt $P --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 --seed 42 --threads 4 --n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 $MU --n-batch 512 --n-ubatch 64 --out-meta $ROOT/wdtest/$TAG/meta.json --out-gen $ROOT/wdtest/$TAG/gen.txt --out-csv /dev/null > /dev/null 2> $ROOT/wdtest/$TAG/err &'" < /dev/null >/dev/null 2>&1
  for i in $(seq 1 120); do sleep 10; adb_safe_shell "[ -s $ROOT/wdtest/$TAG/meta.json ] && echo done || echo run" < /dev/null | grep -q done && break; done
  sleep 3; adb_safe_shell "su -c 'pkill -f sample_sensor[s]'" < /dev/null >/dev/null 2>&1; wd_stop
  for f in meta.json err sensors.csv; do adb_safe_pull $ROOT/wdtest/$TAG/$f $D/$f >/dev/null 2>&1; done
  [ "$ARM" = on ] && adb_safe_pull $ROOT/gpu_watchdog.log $D/gpu_watchdog.log >/dev/null 2>&1
  python3 - "$D" "$TAG" <<'PY'
import sys, json, csv, collections
d, tag = sys.argv[1], sys.argv[2]
m = json.load(open(d + "/meta.json")); rows = list(csv.DictReader(open(d + "/sensors.csv"))); T = [float(r["monotonic_s"]) for r in rows]
P = [abs(float(r["usb_current_ua"])) / 1e6 * float(r["usb_voltage_uv"]) / 1e6 for r in rows]; Q = [float(r["bat_charge_uah"]) for r in rows]; V = [float(r["bat_voltage_now_uv"]) / 1e6 for r in rows]
G = [float(r["gpu_clk_hz"]) for r in rows]; ddr = [float(r["ddr_temp_mc"] or 0) / 1000 for r in rows]; tpl = [r["gpu_thermal_pwrlevel"] for r in rows]
t0 = T[1] + 3
for i in range(1, len(T)):
    if G[i] >= 9e8 and T[i] - T[0] < 90: t0 = T[i]; break
tend = t0 + m["total_ms"] / 1000; E = sum(P[i] * min(5, T[i] - T[i-1]) for i in range(1, len(T)) if t0 < T[i] <= tend)
qa = next((Q[i] for i in range(len(T)) if T[i] >= t0), None); qc = max((Q[i] for i in range(len(T)) if T[i] <= tend + 30), default=None)
if qa and qc: E += max(0, qa - qc) / 1e6 * (sum(V) / len(V)) * 3600
clk = collections.Counter(int(g / 1e6) for i, g in enumerate(G) if t0 < T[i] <= tend).most_common(4)
print(f"  [{tag}] E={E:.0f} J  T={m['total_ms']/1000:.0f} s (prefill {m['prefill_ms']/1000:.0f}, decode {m['decode_ms']/1000:.0f}, {m['decode_tps']:.1f} tok/s)  DDR peak {max(ddr):.0f} C  vendor tpl5 samples {sum(1 for i,t in enumerate(tpl) if t=='5' and t0<T[i]<=tend)}  clocks {clk}")
PY
}
for R in 1 2 3; do cell wd_off_r$R off; cell wd_on_r$R on; done
echo GPUWD_DONE
