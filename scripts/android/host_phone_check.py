#!/usr/bin/env python3
"""
host_phone_check.py — first thing to run after the phone is plugged in.

Checks:
  1. adb sees the device (state == 'device', not 'unauthorized'/'offline')
  2. Reports model / device / SoC / Android version / kernel / ABI
  3. Reports whether `su` is available (root status, optional for the study)
  4. Pushes scripts/discover_sensors.sh and runs it
  5. Pulls the resulting sensor_map.txt back to logs/sensor_map_<device>.txt
  6. Prints a one-line OK / FAIL summary the user can read at a glance

No build artifacts are pushed yet; that happens in host_run_study.py.

Usage:
  python scripts/android/host_phone_check.py
  WORKSPACE=/d/Research/EndurKV_workspace python scripts/android/host_phone_check.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
WORKSPACE = Path(os.environ.get("WORKSPACE", HERE.parents[3]))
PHONE_ROOT = os.environ.get("PHONE_ROOT", "/data/local/tmp/endurkv")
ADB = os.environ.get("ADB", "adb")


def run(cmd: list[str], check=False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def shell(s: str) -> str:
    r = run([ADB, "shell", s])
    return r.stdout.strip()


def main() -> int:
    print("=" * 60)
    print("  EndurKV phone-side setup check")
    print("=" * 60)

    # 1) adb devices
    r = run([ADB, "devices"])
    print("\n[1] adb devices")
    print(r.stdout)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip() and not ln.startswith("List of devices")]
    if not lines:
        print("FAIL: no device detected.")
        print("  - Plug in OnePlus 15 via USB-C")
        print("  - Settings -> About -> tap Build number 7 times to unlock Developer Options")
        print("  - Developer Options -> USB debugging -> ON")
        print("  - Accept 'Allow USB debugging' prompt on phone")
        return 1
    state = lines[0].split()[-1] if lines[0].split() else ""
    if state != "device":
        print(f"FAIL: device is in state '{state}'. Expected 'device'.")
        print("  If 'unauthorized': accept the prompt on the phone.")
        print("  If 'offline':       unplug+replug the cable, or `adb kill-server && adb start-server`.")
        return 1
    print(f"OK: device state == device")

    # 2) device identity
    print("\n[2] device identity")
    info = {
        "model":    shell("getprop ro.product.model"),
        "device":   shell("getprop ro.product.device"),
        "manuf":    shell("getprop ro.product.manufacturer"),
        "android":  shell("getprop ro.build.version.release"),
        "api":      shell("getprop ro.build.version.sdk"),
        "abi":      shell("getprop ro.product.cpu.abi"),
        "soc":      shell("getprop ro.soc.model") or shell("getprop ro.board.platform"),
        "kernel":   shell("uname -r"),
    }
    for k, v in info.items():
        print(f"  {k:<8} {v}")

    if info["abi"] not in ("arm64-v8a", "aarch64"):
        print(f"WARN: ABI is {info['abi']}, but our binaries are arm64-v8a.")

    # 3) root status (informational only)
    print("\n[3] root status")
    r = run([ADB, "shell", "command -v su"])
    if r.returncode == 0 and r.stdout.strip():
        # Try `su -c id`
        r2 = run([ADB, "shell", "su -c 'id -u'"])
        if r2.returncode == 0 and r2.stdout.strip() == "0":
            print(f"  ROOT AVAILABLE — uid=0 via {r.stdout.strip()}")
        else:
            print(f"  su found at {r.stdout.strip()} but did not grant uid=0 "
                  f"(maybe prompt-on-phone or Magisk policy).")
    else:
        print("  no root — that's fine, study only needs world-readable sysfs.")

    # 4) push discover_sensors.sh
    print("\n[4] discover sensors")
    discover = WORKSPACE / "phone-deploy" / "scripts" / "discover_sensors.sh"
    if not discover.exists():
        print(f"FAIL: {discover} missing. Run staging step first.")
        return 1
    shell(f"mkdir -p {PHONE_ROOT}/scripts {PHONE_ROOT}/logs")
    run([ADB, "push", str(discover), f"{PHONE_ROOT}/scripts/"])
    shell(f"chmod 755 {PHONE_ROOT}/scripts/discover_sensors.sh")
    rc = run([ADB, "shell", f"sh {PHONE_ROOT}/scripts/discover_sensors.sh"])
    if rc.returncode != 0:
        print("WARN: discover_sensors.sh exited non-zero, output still pulled.")

    # 5) pull sensor map back
    safe = (info["device"] or "phone").replace(" ", "_")
    out_path = WORKSPACE / "logs" / f"sensor_map_{safe}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run([ADB, "pull", f"{PHONE_ROOT}/sensor_map.txt", str(out_path)])
    if out_path.exists():
        print(f"OK: sensor map saved to {out_path}")
        with open(out_path, encoding="utf-8", errors="replace") as f:
            head = f.read(2000)
        print("\n--- first 2 KB of sensor map ---")
        print(head)
        print("--- (truncated) ---")
    else:
        print("WARN: could not pull sensor map.")

    print("\n=== ready for host_run_study.py ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
