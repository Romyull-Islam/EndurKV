#!/system/bin/sh
# ============================================================================
# GPU throttle watchdog v5 -- REAL-TRIGGER anchored (2026-07-20).
#
# Fixes v4's misanchor: v4 stepped the GPU clock down at battery 36.3C / shell
# 39.5C -- idle-warm temperatures -- so it capped every GPU run to 826 MHz for
# nothing (the GPU holds 1200 MHz up to a 94C junction with no thermal clock
# throttle, and GPU decode keeps the battery at ~40C, far below any throttle).
# v5 anchors at the MEASURED SoC deep-throttle trigger (battery 50C, SoC-wide,
# same as the CPU watchdog) and glides the GPU clock down only through the
# 47-49.5C battery / 50-52.5C skin run-up. On a cool GPU workload it stays
# DORMANT at 1200 MHz -- no performance loss. muKV-only, reduce-only.
#
# Zones resolved BY NAME (battery, shell_front, gpuss) to avoid the hardcoded-
# zone-ID bug. GPU clock via /sys/kernel/gpu/gpu_max_clock. Authorized Adreno
# 840 rungs only. Junction backstop raised to 98C (a true safety net).
# ============================================================================
LOG="${1:-/data/local/tmp/gpu_wd_v5.log}"; STOP="${2:-/data/local/tmp/gpu_wd.stop}"
MAXCLK=/sys/kernel/gpu/gpu_max_clock
rm -f "$STOP"

# --- resolve thermal zones by name ---
zbyname(){ for z in /sys/class/thermal/thermal_zone*; do
  [ "$(cat $z/type 2>/dev/null)" = "$1" ] && { echo "${z##*thermal_zone}"; return; }; done; }
BATZ=$(zbyname battery); SKINZ=$(zbyname shell_front)
GZONES=""; for z in /sys/class/thermal/thermal_zone*; do
  case "$(cat $z/type 2>/dev/null)" in gpuss-*) GZONES="$GZONES ${z##*thermal_zone}";; esac; done

# --- authorized Adreno 840 GPU freq ladder (MHz); tier 0 = full ---
TIERS="1200 1050 967 902 826"
tier_clk(){ i=0; for c in $TIERS; do [ $i -eq $1 ] && { echo $c; return; }; i=$((i+1)); done; echo 826; }

# --- REAL-trigger ladders (mC), mirror of the CPU v2 watchdog ---
# battery: <47 full ; >=47.0/48.0/48.5/49.0/49.5 -> tier 1/2/3/4/4
# skin   : <50 full ; >=50.0/51.0/51.5/52.0/52.5 -> tier 1/2/3/4/4
bat_tier(){ b=$1
  if   [ $b -ge 49500 ]; then echo 4; elif [ $b -ge 49000 ]; then echo 4
  elif [ $b -ge 48500 ]; then echo 3; elif [ $b -ge 48000 ]; then echo 2
  elif [ $b -ge 47000 ]; then echo 1; else echo 0; fi; }
skin_tier(){ s=$1
  if   [ $s -ge 52500 ]; then echo 4; elif [ $s -ge 52000 ]; then echo 4
  elif [ $s -ge 51500 ]; then echo 3; elif [ $s -ge 51000 ]; then echo 2
  elif [ $s -ge 50000 ]; then echo 1; else echo 0; fi; }
GPU_BACKSTOP=98000   # gpuss-junction hard safety net (well above the 94C it runs at)

maxz(){ m=0; for z in $1; do t=$(cat /sys/class/thermal/thermal_zone$z/temp 2>/dev/null); [ -n "$t" ] && [ "$t" -gt "$m" ] && m=$t; done; echo $m; }
echo "$(date) gpu-wd v5 REAL-trigger: bat_z=$BATZ skin_z=$SKINZ gpuss=($GZONES) anchors bat47/skin50 backstop98" > "$LOG"

cur=0; echo 1200 > $MAXCLK
while [ ! -f "$STOP" ]; do
  bat=$(cat /sys/class/thermal/thermal_zone$BATZ/temp 2>/dev/null); bat=${bat:-0}
  skin=$(cat /sys/class/thermal/thermal_zone$SKINZ/temp 2>/dev/null); skin=${skin:-0}
  g=$(maxz "$GZONES")
  bt=$(bat_tier $bat); st=$(skin_tier $skin)
  tier=$bt; [ $st -gt $tier ] && tier=$st
  [ $g -gt $GPU_BACKSTOP ] && tier=4    # junction backstop overrides
  if [ $tier -ne $cur ]; then
    nc=$(tier_clk $tier); echo $nc > $MAXCLK; cur=$tier
    echo "$(date +%s) tier=$tier clk=$nc bat=$((bat/1000))C skin=$((skin/1000))C gpuss=$((g/1000))C" >> "$LOG"
  fi
  sleep 2
done
echo 1200 > $MAXCLK
echo "$(date) gpu-wd v5 stop, released 1200" >> "$LOG"
