#!/bin/bash
# assert_binary_current.sh — guard against running a stale eviction_bench.
# Aborts if the SOURCE is newer than the built binary, or if the on-device
# binary differs from the freshly-built host binary. Source this (or call) at
# the top of any GPU/CPU run script BEFORE launching eviction_bench.
#
# Usage: bash assert_binary_current.sh <build-dir> [device-binary-path]
#   e.g. bash assert_binary_current.sh entropy_probe/build-android-vulkan \
#            /data/local/tmp/endurkv/bin_vulkan_new/eviction_bench
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/entropy_probe/eviction_bench.cpp"
BUILDDIR="${1:?build dir required}"
BIN="$ROOT/$BUILDDIR/eviction_bench"
DEV_BIN="${2:-}"

[ -f "$SRC" ] || { echo "STALE-GUARD: source missing: $SRC" >&2; exit 2; }
[ -f "$BIN" ] || { echo "STALE-GUARD: host binary missing: $BIN — rebuild first" >&2; exit 2; }

src_t=$(stat -c %Y "$SRC"); bin_t=$(stat -c %Y "$BIN")
if [ "$src_t" -gt "$bin_t" ]; then
  echo "STALE-GUARD: FAIL — source ($(date -d @$src_t '+%m-%d %H:%M')) is NEWER than host binary ($(date -d @$bin_t '+%m-%d %H:%M'))." >&2
  echo "  Rebuild: cmake --build $BUILDDIR -j8 --target eviction_bench   (then re-push, then add a CHANGELOG.md row)" >&2
  exit 1
fi

if [ -n "$DEV_BIN" ]; then
  # Size equality can miss a different binary of the same length. Require an
  # exact content hash before a device experiment may run.
  hs=$(sha256sum "$BIN" | awk '{print $1}')
  ds=$(timeout 30 adb shell "sha256sum $DEV_BIN 2>/dev/null" 2>/dev/null | tr -d '\r' | awk '{print $1}')
  if [ -z "$ds" ]; then
    echo "STALE-GUARD: FAIL — could not read SHA-256 for device binary. Reconnect the device or re-push: adb push $BIN $DEV_BIN" >&2
    exit 1
  fi
  if [ "$ds" != "$hs" ]; then
    echo "STALE-GUARD: FAIL — device binary SHA-256 ($ds) != host build ($hs). Re-push: adb push $BIN $DEV_BIN" >&2
    exit 1
  fi
fi
echo "STALE-GUARD: OK — binary ($(date -d @$bin_t '+%m-%d %H:%M')) is at/after source; device SHA-256 matches."
