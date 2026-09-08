#!/bin/bash
# 2026-07-21: three alternating, deep-cooled vanilla/μKV GPU prefill trials.
# Acceptance criterion: report the paired μKV-minus-vanilla prefill delta; do
# not claim prefill parity unless its 95% range is within the stated margin.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_HOST="${1:-$ROOT/artifacts/reproductions/gpu1b_prefill_parity_$(date +%Y%m%d_%H%M%S)}"
OUT_DEV="/data/local/tmp/endurkv/logs/gpu1b_prefill_parity_$(date +%Y%m%d_%H%M%S)"
ADB=${ADB:-adb}
mkdir -p "$OUT_HOST"
bash "$ROOT/scripts/android/assert_binary_current.sh" entropy_probe/build-android-vulkan /data/local/tmp/endurkv/bin_vulkan_new/eviction_bench
$ADB push "$ROOT/scripts/android/gpu1b_prefill_parity_device.sh" /data/local/tmp/gpu1b_prefill_parity_device.sh >/dev/null

cool() {
  while true; do
    read -r gpu ddr shell bat <<<"$($ADB shell "su -c 'g=0;d=0;s=0;b=0; for z in /sys/class/thermal/thermal_zone*; do n=\$(cat \$z/type); t=\$((\$(cat \$z/temp)/1000)); case \$n in gpuss-*) [ \$t -gt \$g ] && g=\$t;; ddr) d=\$t;; shell_front) s=\$t;; battery) b=\$t;; esac; done; echo \$g \$d \$s \$b'" | tr -d '\r')"
    if [ "${gpu:-99}" -le 37 ] && [ "${ddr:-99}" -le 37 ] && [ "${shell:-99}" -le 34 ] && [ "${bat:-99}" -le 34 ]; then
      echo "cooled: GPUSS=$gpu DDR=$ddr shell=$shell battery=$bat"; return
    fi
    sleep 30
  done
}
run() {
  local cell=$1
  local mode=$2
  local dev="$OUT_DEV/$cell"
  echo "[$(date +%H:%M:%S)] $cell ($mode)"; cool
  # The device runner creates $dev itself. Keep the nohup log in its parent so
  # redirection succeeds before that runner begins (and survives its reset).
  $ADB shell "su -c 'mkdir -p $OUT_DEV; nohup sh /data/local/tmp/gpu1b_prefill_parity_device.sh $dev $mode >$OUT_DEV/${cell}.launcher.log 2>&1 &'"
  until $ADB shell "su -c 'test -f $dev/DONE'" >/dev/null 2>&1; do sleep 10; done
  mkdir -p "$OUT_HOST"; $ADB pull "$dev" "$OUT_HOST/" >/dev/null
}
for r in 1 2 3; do run vanilla_$r vanilla; run mukv_$r mukv; done
echo "Raw results: $OUT_HOST"
